"""Pydantic v2 schemas shared by API + worker."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScanRequest(BaseModel):
    text_blocks: list[str] = Field(default_factory=list, description="Raw text snippets from the page")
    image_urls: list[str] = Field(default_factory=list, description="Absolute image URLs to download and OCR")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    client_id: str | None = None


class ScanResultRow(BaseModel):
    text: str
    ocr_text: str = ""
    label: int
    prob_offensive: float
    modality_used: Literal["text_only", "image_only", "both"]


class JobCreateResponse(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "success", "failed"]
    progress: float = 0.0
    message: str = ""


class ScanResponse(BaseModel):
    job_id: str
    status: Literal["success", "failed"] = "success"
    results: list[ScanResultRow] = Field(default_factory=list)
    offensive_count: int = 0
    summary: dict = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    job_id: str
    item_index: int
    user_label: Literal[0, 1]
    comment: str | None = None
