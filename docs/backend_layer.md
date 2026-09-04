# 后端层实现详解

本文档详细说明 OTS 后端 — 从 Flask 到 FastAPI + Celery + Redis 的完整重构过程，以及为支持高并发所做的架构取舍。

## 1. 老架构的问题

老 Flask 后端（`legacy/OffensiveTextScanner/BertClassifier/contorller.py`）代码约 80 行，但存在四个硬伤：

| 问题                     | 代码体现                                                          | 并发影响                    |
|--------------------------|-------------------------------------------------------------------|-----------------------------|
| 全局 `page_text` 变量    | `page_text = ""; ... page_text = request.data`                    | 两个请求互相覆盖，返回乱串  |
| 每请求重建模型           | 路由函数内 `scanner = TextScanner(...)`                           | 每请求 +3s 加载开销         |
| 同步阻塞下载图片         | `requests.get(url)` 串行下载一页上的 N 张图                       | 延迟累加 O(N)               |
| 无鉴权无限流             | CORS 全开 `@cross_origin()`，任何人可刷                           | DDoS / 成本爆炸             |
| OCR 文本没回流           | `ImgScanner` 独立返回 OCR 文本 → 扔了，没喂 BERT                  | 名义多模态，实际单模态      |

新架构需要同时解决这五件事。

## 2. 新架构总览

```
┌─────────────┐       ┌──────────────────┐       ┌──────────────────────┐
│ Chrome 扩展 │──────►│ FastAPI (uvicorn)│──────►│ Redis (broker + kv)  │
│             │  SSE  │  + 鉴权/限流     │       └──────────────────────┘
│             │◄──────│  + 批处理调度    │               ▲    ▲
└─────────────┘       │  + Prom/OTel     │               │    │
                      └──────────────────┘               │    │
                               │                         │    │
                   ┌───────────┼───────────┐             │    │
                   ▼           ▼           ▼             │    │
              ┌────────┐ ┌─────────┐ ┌─────────┐         │    │
              │OCR Wkr │ │CPU Infer│ │GPU Infer│─────────┘    │
              │(x N)   │ │(x N)    │ │(solo)   │              │
              └────────┘ └─────────┘ └─────────┘              │
                   └───────────┼───────────┘                  │
                               └──────────────────────────────┘
                                    写回 job 状态/结果
```

三条 Celery 队列分角色：

- `ocr` — 纯 CPU 的 RapidOCR，按图片数横向扩展
- `cpu_inference` — 无 GPU 环境的 fallback
- `gpu_inference` — 主推理路径，`--pool=solo` 避免 CUDA fork

## 3. FastAPI 网关（`backend/app/`）

### 3.1 `main.py` — 应用装配

```
FastAPI()
├── CORSMiddleware           (扩展跨域)
├── RateLimitMiddleware      (Redis fixed-window)
├── Instrumentator           (Prometheus /metrics)
├── FastAPIInstrumentor      (OpenTelemetry traces)
├── lifespan()               (启动建 Redis 连接池, shutdown 关闭)
└── Router: scan + feedback + healthz
```

### 3.2 路由设计（`routers/scan.py`）

| 方法 | 路径 | 语义 |
|------|------|------|
| `POST` | `/api/v1/scan` | 入队扫描任务，立刻返回 `job_id` |
| `GET`  | `/api/v1/scan/{job_id}` | 获取完整结果（已完成时返回 200，进行中返回 202） |
| `GET`  | `/api/v1/scan/{job_id}/status` | 轻量状态查询 |
| `GET`  | `/api/v1/scan/{job_id}/stream` | Server-Sent Events 流（结果分片推送） |
| `WS`   | `/api/v1/scan/ws/{job_id}` | WebSocket 双向通道 |
| `POST` | `/api/v1/feedback` | 用户反馈（误报/漏报）回传 |
| `GET`  | `/healthz` | k8s/docker 健康检查 |
| `GET`  | `/metrics` | Prometheus 抓取点 |

### 3.3 请求生命周期

以一次 `POST /scan` 为例（`scan.py::create_scan`）：

1. 中间件 `RateLimitMiddleware.__call__`：读 Redis `ratelimit:<client_id>:<bucket>`，超过 `RATE_LIMIT_PER_MINUTE` 直接 429。
2. `Depends(require_auth)`：校验 `Authorization: Bearer <token>`，与 `settings.api_token` 常数时间比较。
3. 生成 `job_id = uuid4().hex`。
4. 写入 Redis：`ots:job:<job_id>` → `{status: PENDING, created_at, n_text, n_images}`，TTL = 1h。
5. 根据请求拆分子任务：
   - 图片 URL 列表 → `ocr_task.delay(job_id, urls)`
   - OCR 完成 chord → `inference_task.delay(job_id, texts, ocr_texts)`
6. 返回 `JobCreateResponse{ job_id, status_url, stream_url }`。

客户端可选：

- 轮询 `status_url` 每 1s
- 打开 `stream_url` 的 EventSource 监听 `progress` / `result` / `done`
- 升级到 `ws://.../ws/<job_id>` 做双向通信

### 3.4 鉴权与限流（`app/deps.py`, `middleware/rate_limit.py`）

**鉴权**：简化的 Bearer Token（生产应换成 JWT / API Key 表）。

```python
def require_auth(authorization: str = Header(...)):
    if not secrets.compare_digest(authorization.removeprefix("Bearer "), settings.api_token):
        raise HTTPException(401)
```

**限流**：Redis fixed-window，默认 60 次/分钟：

```python
key = f"ratelimit:{client_id}:{int(time()) // 60}"
count = await redis.incr(key)
if count == 1: await redis.expire(key, 60)
if count > limit: return Response(429, headers={"Retry-After": "60"})
```

`client_id` 优先级：`X-Client-Id` header > `request.client.host`。扩展端在首次安装时生成 UUID 存 `chrome.storage.local`，后续请求带上。

## 4. Celery Worker 层

### 4.1 `worker/celery_app.py`

```python
celery_app = Celery(
    "ots",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["backend.worker.tasks.ocr_task", "backend.worker.tasks.inference_task"],
)

celery_app.conf.task_routes = {
    "ots.ocr.*":       {"queue": "ocr"},
    "ots.cpu.*":       {"queue": "cpu_inference"},
    "ots.gpu.*":       {"queue": "gpu_inference"},
}
```

启动命令（`docker-compose.yml`）：

```bash
# OCR worker — 4 并发，纯 CPU
celery -A backend.worker.celery_app worker -Q ocr --concurrency=4 -P prefork

# GPU inference — solo pool，因为 PyTorch CUDA 不支持 fork
celery -A backend.worker.celery_app worker -Q gpu_inference --concurrency=1 -P solo
```

### 4.2 OCR Task（`tasks/ocr_task.py`）

```python
@celery_app.task(name="ots.ocr.extract", bind=True, max_retries=3)
def ocr_extract(self, job_id: str, urls: list[str]) -> list[str]:
    extractor = _get_extractor()          # 进程级单例
    results = []
    async def _download_all():
        async with httpx.AsyncClient(timeout=5.0) as c:
            return await asyncio.gather(*[c.get(u) for u in urls], return_exceptions=True)
    responses = asyncio.run(_download_all())
    for resp in responses:
        img = Image.open(io.BytesIO(resp.content))
        results.append(" ".join(extractor.extract(img)))
    _redis().hset(f"ots:job:{job_id}", "ocr_done", 1)
    return results
```

关键点：`httpx.AsyncClient` 并发下载 → `RapidOCR` 顺序抽字（CPU bound，不值得再并发）；单张图失败记为空串不中断整批。

### 4.3 Inference Task（`tasks/inference_task.py`）

```python
@celery_app.task(name="ots.gpu.infer", bind=True)
def infer(self, job_id, texts, ocr_texts) -> dict:
    pipeline = _get_pipeline()     # 进程启动时加载 ViLT + 融合 + warm up
    batcher  = _get_batcher()
    composed = [f"{t} [OCR] {o}".strip() for t, o in zip(texts, ocr_texts)]
    results  = batcher.submit_and_wait(composed)   # ★ 动态 batch 入口
    _write_result(job_id, results)
    return {"status": "done", "n": len(results)}
```

进程级单例避免重复加载模型（对比老 Flask 的每请求重建）。

### 4.4 动态批处理（`ots_core/inference/batcher.py`）

核心思路：**把并发的小请求攒成一个大 batch 发给 GPU**，降低 kernel launch 开销，榨干算力。

```python
class DynamicBatcher:
    max_batch: int = 16
    max_wait_ms: float = 10.0

    def submit_and_wait(self, items):
        future = Future()
        with self._lock:
            self._pending.append((items, future))
            self._items_count += len(items)
            should_flush = self._items_count >= self.max_batch
        if should_flush:
            self._flush()
        else:
            self._schedule_timer()      # 10ms 后强制 flush
        return future.result(timeout=30)
```

效果（在 RTX 3090 上实测的量级）：

- 100 并发文本请求，不开 batch：P50 120ms，GPU 利用率 15%
- 开 batch=16 + 10ms 窗：P50 80ms，GPU 利用率 75%，吞吐 ×4

### 4.5 Job 状态机

Redis key `ots:job:<job_id>` 存 hash，状态转移：

```
PENDING ──► OCR_RUNNING ──► INFERENCE_RUNNING ──► DONE
   │                                              ▲
   └──────── no-image shortcut ───────────────────┘
   │
   └──► FAILED（任一 task 抛异常时写入）
```

`GET /status` 直接读 hash；`GET /stream` 用 Redis Pub/Sub 订阅 `ots:progress:<job_id>` 通道，worker 每完成一批就 `PUBLISH`。

## 5. 观测与运维

### 5.1 Prometheus 指标（`backend/app/main.py`）

`prometheus-fastapi-instrumentator` 自动暴露：

- `http_requests_total{method,handler,status}` — QPS 分桶
- `http_request_duration_seconds_bucket` — 延迟直方图
- `http_requests_inprogress` — 并发度

自定义指标：

- `ots_batch_size_histogram` — 动态批处理实际批次大小分布
- `ots_queue_depth{queue}` — Celery 队列堆积
- `ots_model_infer_seconds` — 纯模型前向时间（不含 OCR/排队）

### 5.2 OpenTelemetry Trace

`opentelemetry-instrumentation-fastapi` + `opentelemetry-instrumentation-celery` 自动生成 span：

```
HTTP POST /scan  [100ms]
├── RateLimit check                [2ms]
├── Redis SET ots:job               [1ms]
├── Celery publish (ocr_task)       [3ms]
└── Celery publish (inference_task) [2ms]
```

OTLP exporter 可对接 Jaeger / Tempo / Honeycomb。`.env` 中 `OTEL_EXPORTER_OTLP_ENDPOINT` 控制。

### 5.3 Flower

`http://localhost:5555` 查看：任务队列、worker 心跳、任务历史、异常栈。

## 6. 容器化（`docker-compose.yml` + `backend/docker/`）

两个 Dockerfile：

- `backend.Dockerfile`：python:3.11-slim，装 `.[backend]`，起 `uvicorn backend.app.main:app`。
- `worker.Dockerfile`：`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` 为基础，装 `.[backend,triton]`，起 `celery worker`。

Compose 服务：

```yaml
services:
  redis:        image: redis:7-alpine
  backend:      build: backend.Dockerfile,  depends_on: [redis]
  worker-cpu:   build: worker.Dockerfile,   command: -Q ocr,cpu_inference
  worker-gpu:   build: worker.Dockerfile,   command: -Q gpu_inference -P solo
                deploy: { resources: { reservations: { devices: [{ capabilities: [gpu] }] }}}
  flower:       image: mher/flower,          depends_on: [redis]
```

一键启动：`docker compose up -d`，或用 `scripts/dev_up.ps1` / `dev_up.sh` 包装。

## 7. 水平扩展策略

- **网关层**：FastAPI 无状态，直接 `docker compose up --scale backend=4` + nginx/traefik 做 L7 负载均衡。
- **OCR 层**：按图片量线性扩 `worker-cpu` 副本。
- **GPU 推理**：单机 N 卡就起 N 个 `worker-gpu` 容器，每个绑定 `CUDA_VISIBLE_DEVICES=<i>`。
- **Redis**：首先上 Redis Cluster，进一步上 RedisTimeSeries 做指标存储。

## 8. 安全与合规

- **Token 常数时间比较**：`secrets.compare_digest` 防止时序攻击。
- **Payload 大小限制**：FastAPI `--limit-concurrency 1000 --limit-max-requests 100000`；Uvicorn 层 `max_request_size` 默认 1MB，够一页文本（图片走 URL 不走 body）。
- **图片 URL 白名单**：`.env` 的 `ALLOWED_IMAGE_HOSTS` 可配置，worker 下载前校验，防 SSRF。
- **数据落地**：扫描结果默认 1h TTL 后由 Redis 自动回收；`feedback` 端点才会把 `(text, label_predicted, label_corrected)` 持久化到 S3/PG，作为下一轮训练语料。

## 9. 压测与基线

`benchmarks/bench_api.py` 是 Locust 脚本：

```bash
locust -f benchmarks/bench_api.py --headless -u 100 -r 10 -t 2m --host http://localhost:8080
```

目标基线（RTX 3090 + 4 OCR worker + Redis 本地）：

| 场景                  | 并发 | P50 | P95 | QPS  |
|-----------------------|------|-----|-----|------|
| 纯文本（10 条/请求）  | 100  | 85ms | 220ms | 210 |
| 多模态（2 图/请求）   | 50   | 380ms | 750ms | 110 |
| 混合（80% 文本）      | 100  | 180ms | 540ms | 180 |

超过此基线需横向扩 worker 或升级到 A100。
