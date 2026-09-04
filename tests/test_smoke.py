"""Smoke tests that every module at least imports.

These intentionally skip heavy-dep paths (torch/transformers/triton) when not installed.
"""
from __future__ import annotations

import importlib

import pytest


LIGHT_MODULES = [
    "ots_core",
    "ots_core.config",
    "ots_core.data.schema",
    "ots_core.data.clean.sanitize",
    "ots_core.data.clean.dedup",
    "backend.schemas.scan",
    "backend.app.middleware.rate_limit",
]


HEAVY_MODULES = [
    "ots_core.models.losses",
    "ots_core.models.heads",
    "ots_core.data.dataset",
    "ots_core.ocr.rapid_ocr",
    "ots_core.inference.pipeline",
    "ots_core.fusion",
    "ots_core.fusion.kernels.fused_qkv",
    "ots_core.fusion.kernels.fused_ln_linear",
    "ots_core.fusion.kernels.fused_attention",
    "ots_core.fusion.patcher",
    "backend.worker.celery_app",
    "backend.app.main",
]


@pytest.mark.parametrize("mod", LIGHT_MODULES)
def test_light_modules_import(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("mod", HEAVY_MODULES)
def test_heavy_modules_import(mod):
    try:
        importlib.import_module(mod)
    except ImportError as e:
        pytest.skip(f"missing optional dep for {mod}: {e}")
