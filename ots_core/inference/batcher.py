"""Dynamic batcher: coalesce concurrent requests into fixed-size batches with a small time
window. Runs on its own background thread so both Celery tasks and direct in-process callers
can enqueue work transparently.

Contract::

    batcher = DynamicBatcher(pipeline.batch_scan, max_batch=16, max_wait_ms=10)
    future = batcher.submit(ScanItem(text=..., image=...))
    result: ScanResult = future.result(timeout=30)
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class _Entry(Generic[T, R]):
    item: T
    future: Future  # Future[R]


class DynamicBatcher(Generic[T, R]):
    def __init__(
        self,
        batch_fn: Callable[[list[T]], list[R]],
        max_batch: int = 16,
        max_wait_ms: int = 10,
    ) -> None:
        self._batch_fn = batch_fn
        self._max_batch = max_batch
        self._max_wait_s = max_wait_ms / 1000.0
        self._queue: Queue[_Entry[T, R]] = Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="ots-batcher", daemon=True)
        self._thread.start()

    def submit(self, item: T) -> Future:
        fut: Future = Future()
        self._queue.put(_Entry(item=item, future=fut))
        return fut

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except Empty:
                continue

            batch: list[_Entry[T, R]] = [first]
            deadline = time.monotonic() + self._max_wait_s
            while len(batch) < self._max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=timeout))
                except Empty:
                    break

            items = [e.item for e in batch]
            try:
                results = self._batch_fn(items)
                for entry, result in zip(batch, results):
                    entry.future.set_result(result)
            except Exception as exc:  # pragma: no cover - surface to callers
                for entry in batch:
                    if not entry.future.done():
                        entry.future.set_exception(exc)
