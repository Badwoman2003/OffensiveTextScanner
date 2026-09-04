"""Shared pytest fixtures + import-time path hackery.

We deliberately avoid importing heavy deps (torch / transformers) at collection time so the
test suite can at least collect on bare environments.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
