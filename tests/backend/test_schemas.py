"""Schema sanity tests: model validation works without requiring Redis / Celery."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.schemas.scan import (
    FeedbackRequest,
    ScanRequest,
    ScanResponse,
    ScanResultRow,
)


def test_scan_request_defaults():
    req = ScanRequest()
    assert req.text_blocks == []
    assert req.image_urls == []
    assert req.threshold == 0.5


def test_scan_result_row_enforces_modality():
    with pytest.raises(ValidationError):
        ScanResultRow(text="hi", label=0, prob_offensive=0.1, modality_used="weird")


def test_scan_response_roundtrip():
    resp = ScanResponse(
        job_id="abc",
        results=[
            ScanResultRow(text="你好", ocr_text="", label=0, prob_offensive=0.02, modality_used="text_only"),
        ],
        offensive_count=0,
    )
    dumped = resp.model_dump_json()
    restored = ScanResponse.model_validate_json(dumped)
    assert restored.job_id == "abc"
    assert restored.results[0].prob_offensive == pytest.approx(0.02)


def test_feedback_request_label_range():
    FeedbackRequest(job_id="j", item_index=0, user_label=1)
    with pytest.raises(ValidationError):
        FeedbackRequest(job_id="j", item_index=0, user_label=2)  # type: ignore[arg-type]
