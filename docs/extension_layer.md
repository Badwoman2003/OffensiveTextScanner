# 前端（浏览器扩展）层实现详解

本文档详细说明 OTS Chrome 扩展的实现 — 从原生 JS 的简陋原型，到基于 React + Vite + TS + Tailwind 的现代 MV3 扩展。

## 1. 老扩展的问题

老扩展（`legacy/OffensiveTextScanner/chrome-extension/`）只有四个 JS 文件，约 60 行代码：

```js
// 老 content.js 核心逻辑
window.onload = function() {
  const text = document.body.innerText;
  fetch("http://localhost:5000/api/v1/info", { method: "POST", body: text })
    .then(r => r.json())
    .then(d => { if (d.offensive) alert("This page contains offensive content!"); });
};
```

问题清单：

| 问题                       | 后果                                                     |
|----------------------------|----------------------------------------------------------|
| `window.onload` 只触发一次 | Twitter/微博等 SPA 完全抓不到                           |
| `document.body.innerText`  | 把 script、style、导航栏都当正文                        |
| 硬编码 `localhost:5000`    | 换后端需要改源码重打包                                  |
| `alert()` 反馈             | 打断用户，没有具体信息，没有位置标注                    |
| 没有历史、没有设置         | 每次扫一次就消失                                        |
| 不处理图片                 | 和老后端的 OCR 能力都浪费了                             |
| Manifest V2                | Chrome 2024 已全面弃用                                  |

## 2. 新架构总览

```
┌───────────────────────────────────────────────────────────────┐
│                       Chrome 浏览器                          │
├──────────────┬────────────────────┬───────────────────────────┤
│  Popup       │   Options Page     │   Content Script          │
│  (React)     │   (React)          │   (TS + MutationObserver) │
│              │                    │         │                 │
│              │                    │         │ dom.extract()   │
│              │                    │         ▼                 │
│              │                    │   ┌─────────────┐         │
│              │                    │   │ Floating    │         │
│              │                    │   │ Result Card │         │
│              │                    │   └─────────────┘         │
└────┬─────────┴─────────┬──────────┴─────────┬─────────────────┘
     │                   │                    │
     │  chrome.runtime.sendMessage                      
     ▼                   ▼                    ▼
┌───────────────────────────────────────────────────────────────┐
│            Service Worker (background.ts)                     │
│            - API client (api.ts)                              │
│            - 历史/设置 (storage.ts)                           │
│            - 扫描编排 (sendScan / pollJob)                    │
└──────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  FastAPI 后端 (HTTPS)   │
              └─────────────────────────┘
```

技术栈（`extension/package.json`）：

| 层级         | 选型                                         |
|--------------|----------------------------------------------|
| 构建         | Vite 5 + `@crxjs/vite-plugin`                |
| 语言         | TypeScript 5                                 |
| UI 框架      | React 18 + Tailwind CSS 3                    |
| 扩展标准     | Manifest V3                                  |
| 存储         | `chrome.storage.local` (5MB 配额)            |
| 通信         | `chrome.runtime.sendMessage` + `postMessage` |
| 类型共享     | 和后端 Pydantic schema 对齐的 `lib/types.ts` |

## 3. 目录结构

```
extension/
├── manifest.json            # MV3 清单
├── package.json
├── vite.config.ts           # @crxjs 插件 + alias
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
├── src/
│   ├── background/
│   │   └── background.ts    # Service Worker（MV3 后台）
│   ├── content/
│   │   ├── content.ts       # 注入到每个网页的主脚本
│   │   └── content.css      # 悬浮卡片 + 高亮样式
│   ├── popup/
│   │   ├── index.html
│   │   ├── main.tsx
│   │   └── Popup.tsx
│   ├── options/
│   │   ├── index.html
│   │   ├── main.tsx
│   │   └── Options.tsx
│   ├── components/
│   │   ├── Badge.tsx        # 风险徽标
│   │   └── ResultCard.tsx
│   └── lib/
│       ├── api.ts           # fetch 封装，支持轮询/SSE
│       ├── storage.ts       # 设置 + 历史 CRUD
│       ├── types.ts         # 与后端对齐的类型
│       └── constants.ts
└── README.md
```

## 4. Content Script（核心：SPA 适配）

`extension/src/content/content.ts`。

### 4.1 文本采集（`TreeWalker`）

不用 `innerText`，而是 `TreeWalker` 精确过滤：

```typescript
function extractVisibleText(root: Node): TextBlock[] {
  const walker = document.createTreeWalker(
    root, NodeFilter.SHOW_TEXT,
    {
      acceptNode(node: Text) {
        const p = node.parentElement;
        if (!p) return NodeFilter.FILTER_REJECT;
        const tag = p.tagName;
        if (SKIP_TAGS.has(tag)) return NodeFilter.FILTER_REJECT;     // SCRIPT/STYLE/NOSCRIPT/SVG
        if (p.getAttribute("aria-hidden") === "true") return NodeFilter.FILTER_REJECT;
        const style = getComputedStyle(p);
        if (style.display === "none" || style.visibility === "hidden") return NodeFilter.FILTER_REJECT;
        const text = (node.nodeValue ?? "").trim();
        if (text.length < 2) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );
  const blocks: TextBlock[] = [];
  while (walker.nextNode()) blocks.push({ node: walker.currentNode as Text, text: walker.currentNode.nodeValue! });
  return blocks;
}
```

每块文本保留指向原 `Text` 节点的引用，后续可以精确高亮定位。

### 4.2 图片采集（`IntersectionObserver`）

懒加载图片只有滚动到视口才会有真实 `src`，用 IntersectionObserver 统一收集：

```typescript
const imageSet = new Set<string>();
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const img = e.target as HTMLImageElement;
    const src = img.currentSrc || img.src || img.dataset.src;
    if (src && src.startsWith("http")) imageSet.add(src);
  }
}, { rootMargin: "200px" });

document.querySelectorAll("img").forEach(img => io.observe(img));
```

### 4.3 SPA 适配（`MutationObserver`）

```typescript
const mo = new MutationObserver((mutations) => {
  for (const m of mutations) {
    m.addedNodes.forEach(node => {
      if (node instanceof HTMLElement) {
        node.querySelectorAll("img").forEach(img => io.observe(img));
      }
    });
  }
  scheduleRescan();   // debounced 800ms
});
mo.observe(document.body, { childList: true, subtree: true });
```

`scheduleRescan` 做 debounce，避免 Twitter 滚动时每秒打几十次请求。

### 4.4 悬浮结果卡片

```
┌─────────────────────────────────────────┐
│ OTS Scan Result      [Close ✕]         │
├─────────────────────────────────────────┤
│ Risk: ●●●○  HIGH  (3 / 42 items)       │
├─────────────────────────────────────────┤
│ #1  "xxx xxx..."    [92%]  🚩 False?  │
│ #2  [IMG] meme.jpg  [88%]  🚩 False?  │
│ #3  "xxx..."        [71%]  🚩 False?  │
└─────────────────────────────────────────┘
```

实现：用 `attachShadow({mode: "open"})` 建 Shadow DOM 隔离页面 CSS 污染；动画用 `@keyframes` + `transform: translateY()`。

### 4.5 页内高亮

```typescript
function highlight(block: TextBlock, risk: "high" | "medium") {
  const span = document.createElement("span");
  span.className = `ots-hl ots-hl-${risk}`;
  block.node.parentNode!.replaceChild(span, block.node);
  span.appendChild(block.node);
}
```

CSS 用 `box-shadow: inset 0 -2px 0 red` 做下划线效果，不影响原页面布局。

## 5. Background Service Worker

`extension/src/background/background.ts`。MV3 下 background 是可休眠的 Service Worker，不能持久内存状态，所有状态必须读写 `chrome.storage`。

### 5.1 消息路由

```typescript
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  switch (msg.type) {
    case "SCAN":         handleScan(msg, sender).then(sendResponse); return true;
    case "GET_HISTORY":  getHistory().then(sendResponse); return true;
    case "FEEDBACK":     sendFeedback(msg.payload).then(sendResponse); return true;
    case "CLEAR_HISTORY":clearHistory().then(sendResponse); return true;
  }
});
```

`return true` 声明异步响应，否则 port 会立刻关闭。

### 5.2 扫描编排（`handleScan`）

```typescript
async function handleScan(msg, sender) {
  const settings = await getSettings();
  const { job_id, stream_url } = await api.createScan({
    texts: msg.texts,
    image_urls: settings.sendImages ? msg.imageUrls : [],
  });

  setBadge(sender.tab!.id!, "…", "#888");

  const result = await api.pollJob(job_id, settings);   // 或 streamSSE
  await appendHistory({ url: sender.tab!.url, result, ts: Date.now() });

  const risk = computeRisk(result, settings.threshold);
  setBadge(sender.tab!.id!, String(result.flagged_count), riskColor(risk));

  chrome.tabs.sendMessage(sender.tab!.id!, { type: "SCAN_RESULT", result });
}
```

### 5.3 Badge 状态

扩展图标上的小徽章实时显示当前标签页风险：

- 无/未扫描 → 无徽章
- 扫描中 → `…` 灰色
- 完成 / 0 风险项 → `0` 绿色
- 完成 / 有风险 → `<count>` 红色

## 6. Popup（React）

`extension/src/popup/Popup.tsx` — 点扩展图标弹出的 360×480 面板。

```tsx
export default function Popup() {
  const [state, setState] = useState<"idle"|"scanning"|"done">("idle");
  const [result, setResult] = useState<ScanResponse | null>(null);

  async function onScan() {
    setState("scanning");
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.tabs.sendMessage(tab.id!, { type: "TRIGGER_SCAN" });
    // 等 content → background → response
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg.type === "SCAN_RESULT") { setResult(msg.result); setState("done"); }
    });
  }

  return (
    <div className="w-[360px] p-4 font-sans">
      <h1 className="text-lg font-bold">OTS Scanner</h1>
      <Button onClick={onScan} disabled={state === "scanning"}>
        {state === "scanning" ? "Scanning…" : "Scan this page"}
      </Button>
      {result && <ResultSummary data={result} />}
      <a href="#/options" onClick={() => chrome.runtime.openOptionsPage()}>Settings</a>
    </div>
  );
}
```

## 7. Options（React）

`extension/src/options/Options.tsx` — 右键扩展图标 → Options。

可配置项：

| 设置项               | 类型    | 默认                      | 说明                                  |
|----------------------|---------|---------------------------|---------------------------------------|
| `backendUrl`         | string  | `http://127.0.0.1:8080`   | FastAPI 根地址                        |
| `apiToken`           | string  | `dev-token-change-me`     | Bearer Token                          |
| `threshold`          | number  | `0.6`                     | 判定阈值（滑块）                      |
| `sendImages`         | bool    | `true`                    | 是否上传图片 URL（隐私开关）          |
| `autoScan`           | bool    | `false`                   | 页面加载后自动扫描                    |
| `connectionMode`     | enum    | `polling`                 | `polling` / `sse` / `websocket`       |
| `historyRetention`   | number  | `50`                      | 最多保留的历史条数                    |

**动态主机权限**：用户改 `backendUrl` 时：

```typescript
await chrome.permissions.request({
  origins: [new URL(newUrl).origin + "/*"],
});
```

避免 manifest 硬编码 `<all_urls>`，符合 Chrome Web Store 最小权限原则。

## 8. 反馈闭环（关键训练数据通道）

用户在悬浮卡片上点「🚩 False?」时：

```
悬浮卡片 onClick
  ↓ postMessage
Content Script
  ↓ chrome.runtime.sendMessage({ type: "FEEDBACK", payload })
Background
  ↓ POST /api/v1/feedback
FastAPI
  ↓ celery.send_task("ots.feedback.persist")
Worker → PostgreSQL/S3
```

Payload 结构（`backend/schemas/scan.py::FeedbackRequest`）：

```json
{
  "job_id": "abc123",
  "item_index": 2,
  "corrected_label": 0,
  "note": "这是反讽，不是攻击",
  "client_id": "uuid-stored-in-storage"
}
```

后端累积到一定量后进入下一轮 fine-tune 数据集，形成持续改进闭环。

## 9. 构建与分发

```bash
cd extension
npm install
npm run build      # 输出 extension/dist
npm run zip        # 输出 extension/ots-extension.zip（用于上传 Web Store）
```

`@crxjs/vite-plugin` 会自动：

- 处理 MV3 manifest 里的 `content_scripts` / `background.service_worker` 等字段
- HMR 开发（`npm run dev`）时自动刷新 content script
- 把 TS/TSX/CSS 打成符合 MV3 CSP 的 bundle（不用 eval，不用远程 import）

## 10. CSP 兼容性

MV3 禁止内联脚本和远程 JS，我们的构建确保：

- React 用预编译的 JSX，不触发 runtime eval
- Tailwind 已在构建期静态抽取，不做运行时 CSS-in-JS
- 所有请求走 `api.ts` 中的 `fetch`，目标域提前声明在 `host_permissions`

`manifest.json` 示例片段：

```json
{
  "manifest_version": 3,
  "permissions": ["storage", "activeTab", "scripting"],
  "optional_host_permissions": ["http://*/*", "https://*/*"],
  "content_security_policy": {
    "extension_pages": "script-src 'self'; object-src 'self'"
  },
  "action": { "default_popup": "src/popup/index.html" },
  "options_page": "src/options/index.html",
  "background": { "service_worker": "src/background/background.ts", "type": "module" },
  "content_scripts": [{ "matches": ["<all_urls>"], "js": ["src/content/content.ts"], "css": ["src/content/content.css"] }]
}
```

## 11. 可访问性与国际化

- 悬浮卡片支持键盘导航（`Tab` 循环焦点，`Esc` 关闭）。
- 所有交互元素带 `aria-label`；Badge 组件给颜色以外的文字区分（高/中/低），不依赖色盲敏感色。
- 文案用 `chrome.i18n.getMessage`，`_locales/zh_CN/messages.json` 和 `_locales/en/messages.json` 并存。

## 12. 性能预算

- 首屏 `Popup` JS 体积 < 50KB（gzip）
- `Content Script` 体积 < 30KB（gzip），注入延迟 < 10ms
- 扫描一页（30 文本 + 10 图）端到端 < 1.5s（本地后端）
- 空闲时 Service Worker 自动休眠，CPU/内存占用为 0
