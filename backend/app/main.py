"""FastAPI application entry point.

Features:
- CORS configured for the Chrome extension + localhost dev.
- Redis-backed rate limit middleware.
- Prometheus ``/metrics`` + basic health check.
- OpenTelemetry auto-instrumentation (no-op if the collector isn't reachable).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

from backend.app.middleware.rate_limit import RedisRateLimitMiddleware
from backend.app.routers import scan as scan_router
from ots_core.config import get_settings

logger = logging.getLogger("ots.api")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="OTS API", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RedisRateLimitMiddleware)

    registry = CollectorRegistry()
    req_counter = Counter(
        "ots_requests_total", "HTTP requests", ["path", "method", "status"], registry=registry
    )
    latency = Histogram(
        "ots_request_latency_seconds", "HTTP latency", ["path", "method"], registry=registry
    )

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        with latency.labels(request.url.path, request.method).time():
            response = await call_next(request)
        req_counter.labels(request.url.path, request.method, response.status_code).inc()
        return response

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict:
        return {"status": "ok", "version": "0.2.0"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    app.include_router(scan_router.router)

    try:  # best-effort OTel; no-op if backend is missing
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        pass

    return app


app = create_app()
