"""Polite scraper skeleton for Chinese meme + caption pairs.

This module intentionally does NOT hardcode any site-specific endpoints; instead it provides a
pluggable ``MemeSource`` interface + robots.txt-aware fetcher so integrators can implement their
own source (Weibo open API, Tieba search, Reddit r/chineselanguagememes, etc.).

Always honour target sites' robots.txt, ToS and copyright. Intended for research-only use.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Protocol
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import pandas as pd

from ots_core.data.schema import ModalityMask, Sample, Source


@dataclass
class RawItem:
    text: str
    image_url: str
    label: int | None  # None if requires downstream manual / weak labelling


class MemeSource(Protocol):
    source_tag: Source
    async def iter_items(self) -> AsyncIterator[RawItem]: ...


class RobotsGate:
    """Cache robots.txt parsers per netloc."""

    def __init__(self, user_agent: str = "OTS-Research/0.2") -> None:
        self.user_agent = user_agent
        self._cache: dict[str, RobotFileParser] = {}

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        netloc = urlparse(url).netloc
        rp = self._cache.get(netloc)
        if rp is None:
            rp = RobotFileParser()
            try:
                r = await client.get(f"https://{netloc}/robots.txt", timeout=5.0)
                rp.parse(r.text.splitlines())
            except Exception:
                rp.parse([])  # fail-open but still record
            self._cache[netloc] = rp
        return rp.can_fetch(self.user_agent, url)


async def _download_image(client: httpx.AsyncClient, url: str, out_dir: Path) -> Path | None:
    try:
        r = await client.get(url, timeout=15.0, follow_redirects=True)
        r.raise_for_status()
    except Exception:
        return None
    digest = hashlib.sha1(r.content).hexdigest()[:20]
    path = out_dir / f"{digest}.jpg"
    if not path.exists():
        path.write_bytes(r.content)
    return path


async def scrape(
    source: MemeSource,
    out_dir: Path,
    max_items: int = 10_000,
    concurrency: int = 4,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    gate = RobotsGate()
    rows: list[dict] = []

    async with httpx.AsyncClient(headers={"User-Agent": gate.user_agent}) as client:
        async def worker(item: RawItem) -> None:
            async with sem:
                if not await gate.allowed(client, item.image_url):
                    return
                img_path = await _download_image(client, item.image_url, img_dir)
                if img_path is None:
                    return
                if item.label is None:
                    return  # defer to manual labelling stage
                sample = Sample(
                    id="scr_" + img_path.stem,
                    text=item.text.strip(),
                    ocr_text="",
                    image_path=str(img_path.resolve()),
                    label=int(item.label),
                    source=source.source_tag,
                    modality_mask=ModalityMask.BOTH,
                )
                rows.append(sample.to_row())

        tasks: list[asyncio.Task[None]] = []
        count = 0
        async for raw in source.iter_items():
            if count >= max_items:
                break
            tasks.append(asyncio.create_task(worker(raw)))
            count += 1
        await asyncio.gather(*tasks)

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / f"{source.source_tag.value}.parquet", index=False)
    return df
