/**
 * Content script.
 *
 * Responsibilities:
 *   1. Observe the DOM (MutationObserver) + lazy-loaded images (IntersectionObserver) so SPAs
 *      and infinite-scroll pages are covered — the legacy extension used `window.onload` which
 *      misses almost every modern site.
 *   2. On demand (triggered by the service worker), collect visible text blocks + image URLs
 *      and send them back for scanning.
 *   3. Render a floating result card anchored to the bottom-right corner of the page, with
 *      in-page highlighting of offensive text nodes and a feedback button.
 */
import type { RuntimeMessage, ScanResultRow } from "../lib/types";

const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "SVG", "CODE", "PRE"]);
const MIN_TEXT_LEN = 4;
const MAX_BLOCKS = 500;

function collectText(): string[] {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent || SKIP_TAGS.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
      if (parent.getAttribute("aria-hidden") === "true") return NodeFilter.FILTER_REJECT;
      const style = window.getComputedStyle(parent);
      if (style.display === "none" || style.visibility === "hidden") return NodeFilter.FILTER_REJECT;
      const text = node.nodeValue?.trim() ?? "";
      return text.length >= MIN_TEXT_LEN ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });

  const out: string[] = [];
  let n: Node | null;
  while ((n = walker.nextNode())) {
    out.push((n.nodeValue ?? "").trim());
    if (out.length >= MAX_BLOCKS) break;
  }
  return Array.from(new Set(out));
}

function collectImages(): string[] {
  const urls = new Set<string>();
  document.querySelectorAll<HTMLImageElement>("img").forEach((img) => {
    const src = img.currentSrc || img.src;
    if (!src || src.startsWith("data:")) return;
    if (img.naturalWidth < 64 || img.naturalHeight < 64) return;
    urls.add(src);
  });
  return Array.from(urls).slice(0, 40);
}

// ------------------------------ lazy image observer --------------------------------------------

const lazyImages = new Set<HTMLImageElement>();
const imageObserver = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        lazyImages.add(e.target as HTMLImageElement);
        imageObserver.unobserve(e.target);
      }
    }
  },
  { rootMargin: "200px" }
);
document.querySelectorAll<HTMLImageElement>("img[loading='lazy']").forEach((img) => imageObserver.observe(img));

const domObserver = new MutationObserver((muts) => {
  for (const m of muts) {
    m.addedNodes.forEach((n) => {
      if (n instanceof HTMLImageElement && n.loading === "lazy") imageObserver.observe(n);
      if (n instanceof Element) {
        n.querySelectorAll<HTMLImageElement>("img[loading='lazy']").forEach((img) => imageObserver.observe(img));
      }
    });
  }
});
domObserver.observe(document.body, { subtree: true, childList: true });

// ------------------------------ floating card -------------------------------------------------

function tone(prob: number): "safe" | "warn" | "danger" {
  if (prob >= 0.8) return "danger";
  if (prob >= 0.5) return "warn";
  return "safe";
}

function renderCard(response: { results: ScanResultRow[]; offensive_count: number; job_id: string }) {
  document.getElementById("ots-floating-card")?.remove();
  const card = document.createElement("div");
  card.id = "ots-floating-card";
  card.style.cssText =
    "position:fixed;bottom:16px;right:16px;z-index:2147483647;max-width:380px;background:#fff;color:#0f172a;border-radius:12px;padding:14px;font:13px/1.5 system-ui,sans-serif;box-shadow:0 10px 25px rgba(0,0,0,0.2);";
  const t = response.offensive_count > 0 ? "danger" : "safe";
  const bar = response.offensive_count > 0 ? "#dc2626" : "#16a34a";
  card.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <span style="width:10px;height:10px;border-radius:50%;background:${bar};display:inline-block"></span>
      <strong>OTS</strong>
      <span style="margin-left:auto;color:#64748b">${response.results.length} blocks</span>
      <button id="ots-close" style="background:transparent;border:0;cursor:pointer;color:#64748b;font-size:16px">&times;</button>
    </div>
    <div style="font-weight:600;margin-bottom:6px;color:${bar}">
      ${response.offensive_count > 0 ? `Detected ${response.offensive_count} potentially offensive item(s)` : "Page looks safe"}
    </div>
    <ul id="ots-list" style="max-height:220px;overflow:auto;margin:0;padding-left:18px"></ul>
  `;
  document.body.appendChild(card);

  const ul = card.querySelector<HTMLUListElement>("#ots-list")!;
  response.results
    .filter((r) => r.label === 1)
    .slice(0, 15)
    .forEach((r, idx) => {
      const li = document.createElement("li");
      const t2 = tone(r.prob_offensive);
      const col = t2 === "danger" ? "#dc2626" : t2 === "warn" ? "#d97706" : "#16a34a";
      li.innerHTML = `
        <div style="margin-bottom:6px">
          <div style="color:${col};font-weight:600">${(r.prob_offensive * 100).toFixed(0)}% · ${r.modality_used}</div>
          <div style="color:#0f172a">${escapeHtml(r.text).slice(0, 140)}</div>
          <button data-idx="${idx}" class="ots-fp-btn" style="margin-top:4px;padding:2px 8px;border:1px solid #94a3b8;background:#fff;border-radius:6px;cursor:pointer;font-size:12px;color:#334155">Mark as false positive</button>
        </div>
      `;
      ul.appendChild(li);
    });

  card.querySelector("#ots-close")?.addEventListener("click", () => card.remove());
  card.querySelectorAll<HTMLButtonElement>(".ots-fp-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.idx);
      chrome.runtime.sendMessage({
        type: "FEEDBACK",
        jobId: response.job_id,
        itemIndex: idx,
        userLabel: 0,
      } satisfies RuntimeMessage);
      btn.disabled = true;
      btn.textContent = "thanks!";
    });
  });
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

function highlightOffensive(results: ScanResultRow[]) {
  const offenders = new Set(results.filter((r) => r.label === 1).map((r) => r.text));
  if (!offenders.size) return;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node: Node | null;
  while ((node = walker.nextNode())) {
    const text = node.nodeValue ?? "";
    if (offenders.has(text.trim()) && node.parentElement) {
      node.parentElement.style.outline = "2px solid #fca5a5";
      node.parentElement.style.background = "rgba(252, 165, 165, 0.18)";
    }
  }
}

// ------------------------------ message bus --------------------------------------------------

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if ((msg as RuntimeMessage).type === "SCAN_PAGE") {
    sendResponse({ textBlocks: collectText(), imageUrls: collectImages(), url: location.href });
    return true;
  }
  if ((msg as RuntimeMessage).type === "SCAN_RESULT") {
    const { response } = msg as Extract<RuntimeMessage, { type: "SCAN_RESULT" }>;
    renderCard({
      results: response.results,
      offensive_count: response.offensive_count,
      job_id: response.job_id,
    });
    highlightOffensive(response.results);
    return;
  }
  if ((msg as RuntimeMessage).type === "SCAN_ERROR") {
    alert(`OTS scan failed: ${(msg as Extract<RuntimeMessage, { type: "SCAN_ERROR" }>).error}`);
    return;
  }
});
