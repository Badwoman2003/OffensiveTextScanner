"""Scan endpoints: enqueue, poll, SSE stream, WebSocket stream, feedback."""
from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis

from backend.app.deps import Auth, get_redis
from backend.schemas.scan import (
    FeedbackRequest,
    JobCreateResponse,
    JobStatusResponse,
    ScanRequest,
    ScanResponse,
)
from backend.worker.celery_app import app as celery_app

router = APIRouter(prefix="/api/v1", tags=["scan"])

RESULT_KEY = "ots:result:{job_id}"
STATUS_KEY = "ots:status:{job_id}"
FEEDBACK_QUEUE = "ots:feedback"


@router.post("/scan", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[Auth])
async def create_scan(req: ScanRequest, redis: Redis = Depends(get_redis)) -> JobCreateResponse:
    job_id = uuid.uuid4().hex
    await redis.set(STATUS_KEY.format(job_id=job_id), "queued", ex=3600)

    # route to GPU queue if payload includes images, else CPU-only path
    queue = "gpu_inference" if req.image_urls else "cpu_inference"
    celery_app.send_task(
        "ots.scan",
        args=[job_id, req.model_dump()],
        queue=queue,
    )
    return JobCreateResponse(job_id=job_id)


@router.get("/scan/{job_id}/status", response_model=JobStatusResponse, dependencies=[Auth])
async def job_status(job_id: str, redis: Redis = Depends(get_redis)) -> JobStatusResponse:
    st = await redis.get(STATUS_KEY.format(job_id=job_id))
    if st is None:
        raise HTTPException(status_code=404, detail="unknown job")
    progress_raw = await redis.get(f"ots:progress:{job_id}")
    return JobStatusResponse(job_id=job_id, status=st, progress=float(progress_raw or 0.0))


@router.get("/scan/{job_id}", response_model=ScanResponse, dependencies=[Auth])
async def job_result(job_id: str, redis: Redis = Depends(get_redis)) -> ScanResponse:
    raw = await redis.get(RESULT_KEY.format(job_id=job_id))
    if raw is None:
        st = await redis.get(STATUS_KEY.format(job_id=job_id))
        if st is None:
            raise HTTPException(status_code=404, detail="unknown job")
        raise HTTPException(status_code=425, detail=f"result not ready: {st}")
    return ScanResponse.model_validate_json(raw)


@router.get("/scan/{job_id}/stream", dependencies=[Auth])
async def job_stream(job_id: str, redis: Redis = Depends(get_redis)) -> StreamingResponse:
    """Server-Sent Events: emit `status` then `partial` then `done` messages until the job
    reaches a terminal state."""
    async def gen():
        for _ in range(600):  # 5-minute cap
            st = await redis.get(STATUS_KEY.format(job_id=job_id))
            if st is None:
                yield "event: error\ndata: {\"detail\": \"unknown job\"}\n\n"
                return
            progress = await redis.get(f"ots:progress:{job_id}") or "0"
            yield f"event: status\ndata: {json.dumps({'status': st, 'progress': float(progress)})}\n\n"
            if st in {"success", "failed"}:
                raw = await redis.get(RESULT_KEY.format(job_id=job_id)) or "{}"
                yield f"event: done\ndata: {raw}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.websocket("/scan/ws/{job_id}")
async def job_ws(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    redis = get_redis()
    try:
        for _ in range(600):
            st = await redis.get(STATUS_KEY.format(job_id=job_id))
            if st is None:
                await ws.send_json({"type": "error", "detail": "unknown job"})
                break
            progress = float(await redis.get(f"ots:progress:{job_id}") or 0.0)
            await ws.send_json({"type": "status", "status": st, "progress": progress})
            if st in {"success", "failed"}:
                raw = await redis.get(RESULT_KEY.format(job_id=job_id))
                if raw:
                    await ws.send_text(raw)
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@router.post("/feedback", status_code=status.HTTP_202_ACCEPTED, dependencies=[Auth])
async def submit_feedback(req: FeedbackRequest, redis: Redis = Depends(get_redis)) -> dict:
    await redis.rpush(FEEDBACK_QUEUE, req.model_dump_json())
    return {"status": "accepted"}
