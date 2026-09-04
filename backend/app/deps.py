"""Shared FastAPI dependencies: Redis client, auth, settings."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status
import redis.asyncio as redis

from ots_core.config import get_celery_settings, get_settings


@lru_cache(maxsize=1)
def _redis_pool() -> redis.ConnectionPool:
    url = get_celery_settings().redis_url
    return redis.ConnectionPool.from_url(url, decode_responses=True)


def get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=_redis_pool())


async def verify_token(authorization: str | None = Header(default=None)) -> str:
    settings = get_settings()
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.api_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    return token


Auth = Depends(verify_token)
