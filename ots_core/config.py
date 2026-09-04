"""Central runtime configuration for OTS.

All tunables live here so backend / workers / training / inference share the same source of truth.
Environment variables override defaults so `.env` works in docker.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="OTS_", extra="ignore")

    # model
    model_dir: Path = Field(default=Path("./checkpoints/vilt-offense"))
    base_vilt: str = "dandelin/vilt-b32-mlm"
    chinese_tokenizer: str = "bert-base-chinese"
    device: str = "cuda"
    num_labels: int = 2
    image_size: int = 384
    max_text_len: int = 128

    # inference / batching
    batch_size: int = 16
    batch_window_ms: int = 10
    use_fused_kernels: bool = True

    # ocr
    ocr_min_conf: float = 0.6
    ocr_min_chars: int = 2

    # api / auth
    api_token: str = "dev-token-change-me"
    rate_limit_per_min: int = 120
    cors_origins: str = "chrome-extension://*,http://localhost:*"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


class CelerySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"


@lru_cache(maxsize=1)
def get_settings() -> OTSSettings:
    return OTSSettings()


@lru_cache(maxsize=1)
def get_celery_settings() -> CelerySettings:
    return CelerySettings()
