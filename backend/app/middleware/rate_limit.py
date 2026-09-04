"""Redis-backed fixed-window rate limiter. Token bucket semantics:

- Per client_id (or IP fallback), we allow N requests per 60s window.
- A Redis INCR + EXPIRE pair is used so the limiter works across API replicas.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.deps import get_redis
from ots_core.config import get_settings


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)
        self._per_min = get_settings().rate_limit_per_min

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path.startswith(("/healthz", "/metrics")):
            return await call_next(request)

        client_key = request.headers.get("x-client-id") or (request.client.host if request.client else "anon")
        bucket = f"rl:{client_key}:{int(time.time() // 60)}"
        r = get_redis()
        try:
            cur = await r.incr(bucket)
            if cur == 1:
                await r.expire(bucket, 65)
        except Exception:
            return await call_next(request)  # fail-open if Redis is unavailable

        if cur > self._per_min:
            return Response(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content='{"detail":"rate limit exceeded"}',
                media_type="application/json",
            )
        return await call_next(request)
