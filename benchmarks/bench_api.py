"""Locust load-test for the OTS API.

Usage::

    locust -f benchmarks/bench_api.py --headless -u 100 -r 10 --host http://localhost:8080 -t 2m

Targets from the plan:
- text-only requests:   >= 200 QPS on single GPU
- multimodal requests:  P95 < 800ms
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

DUMMY_TEXTS = [
    "今天天气真好，我们去公园玩吧",
    "这个人真是讨厌",
    "恭喜你获得了比赛冠军",
    "别再给我发这种消息了",
    "我超喜欢这个视频",
]

DUMMY_IMG = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Commons-logo.svg/120px-Commons-logo.svg.png"


class ScanUser(HttpUser):
    wait_time = between(0.1, 0.5)
    host = "http://localhost:8080"

    def on_start(self) -> None:
        self.client.headers.update(
            {"Authorization": "Bearer dev-token-change-me", "Content-Type": "application/json", "X-Client-Id": "locust"}
        )

    @task(5)
    def text_only(self) -> None:
        payload = {
            "text_blocks": random.sample(DUMMY_TEXTS, k=3),
            "image_urls": [],
            "threshold": 0.5,
            "client_id": "locust",
        }
        with self.client.post("/api/v1/scan", json=payload, catch_response=True) as r:
            if r.status_code != 202:
                r.failure(f"unexpected status: {r.status_code}")
                return
            job_id = r.json()["job_id"]
        self._wait_for(job_id)

    @task(1)
    def multimodal(self) -> None:
        payload = {
            "text_blocks": [random.choice(DUMMY_TEXTS)],
            "image_urls": [DUMMY_IMG],
            "threshold": 0.5,
            "client_id": "locust",
        }
        with self.client.post("/api/v1/scan", json=payload, catch_response=True) as r:
            if r.status_code != 202:
                r.failure(f"unexpected status: {r.status_code}")
                return
            job_id = r.json()["job_id"]
        self._wait_for(job_id, timeout=8.0)

    def _wait_for(self, job_id: str, timeout: float = 4.0) -> None:
        start = self.environment.runner.time() if self.environment.runner else 0
        _ = start  # avoid lint warning
        for _ in range(40):
            with self.client.get(f"/api/v1/scan/{job_id}/status", catch_response=True) as st:
                if st.status_code != 200:
                    st.failure(f"status {st.status_code}")
                    return
                body = st.json()
                if body["status"] in {"success", "failed"}:
                    break
            self.environment.runner.greenlet.sleep(0.1) if self.environment.runner else None
        self.client.get(f"/api/v1/scan/{job_id}")
