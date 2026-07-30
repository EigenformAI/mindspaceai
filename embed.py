from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from chunk import Chunk

API = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_MODEL = "nvidia/nemotron-3-embed-1b:free"
BATCH_SIZE = 128
MAX_ATTEMPTS = 4
WORKERS = 4


def load_api_key(env_path: Path | str = ".env") -> str:
    """Read OPENROUTER_API_KEY from the environment, falling back to .env."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    path = Path(env_path)
    if path.exists():
        match = re.search(r"^OPENROUTER_API_KEY=(.*)$", path.read_text(), re.M)
        if match and match.group(1).strip():
            return match.group(1).strip()
    raise RuntimeError(
        "OPENROUTER_API_KEY not found in the environment or .env — "
        "embedding cannot run without it"
    )


class Embedder:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or load_api_key()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(
            API,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = json.loads(resp.read())
                # The API may return rows out of order; index is authoritative.
                rows = sorted(payload["data"], key=lambda d: d["index"])
                if len(rows) != len(texts):
                    raise RuntimeError(
                        f"asked for {len(texts)} embeddings, got {len(rows)}"
                    )
                return [r["embedding"] for r in rows]

            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                # 4xx other than rate-limiting is a decision, not a hiccup:
                # a bad key, an unknown model, an exhausted budget. Retrying
                # wastes time and buries the one message that explains it.
                if exc.code != 429 and 400 <= exc.code < 500:
                    raise RuntimeError(f"HTTP {exc.code} from OpenRouter: {detail}") from exc
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")

            except (urllib.error.URLError, KeyError, TimeoutError) as exc:
                last_error = exc

            time.sleep(2 * (attempt + 1))

        raise RuntimeError(
            f"embedding failed after {MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


class EmbeddingStore:
    def __init__(self, root: Path | str = "data", index_version: str = "v1") -> None:
        self.dir = Path(root) / "embeddings" / index_version
        self.index_version = index_version

    def chunks_path(self, month: str) -> Path:
        return self.dir / f"{month}.chunks.jsonl"

    def vectors_path(self, month: str) -> Path:
        return self.dir / f"{month}.vectors.npy"

    def existing_hashes(self) -> set[str]:
        """Every content_hash already embedded under this index_version."""
        hashes: set[str] = set()
        if not self.dir.exists():
            return hashes
        for f in sorted(self.dir.glob("*.chunks.jsonl")):
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        hashes.add(json.loads(line)["content_hash"])
        return hashes

    def append(self, month: str, records: list[dict], vectors: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        vpath = self.vectors_path(month)
        if vpath.exists():
            vectors = np.vstack([np.load(vpath), vectors])
        np.save(vpath, vectors)

        with self.chunks_path(month).open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load(self, month: str) -> tuple[list[dict], np.ndarray]:
        records = [
            json.loads(line)
            for line in self.chunks_path(month).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return records, np.load(self.vectors_path(month))

    def load_window(self, months: list[str]) -> tuple[list[dict], np.ndarray]:
        all_records: list[dict] = []
        parts: list[np.ndarray] = []
        for month in months:
            if not self.chunks_path(month).exists():
                raise FileNotFoundError(f"no embeddings for {month} — run embed first")
            records, vectors = self.load(month)
            all_records.extend(records)
            parts.append(vectors)
        return all_records, np.vstack(parts)

    def available_months(self) -> list[str]:
        if not self.dir.exists():
            return []
        return sorted(p.name.split(".")[0] for p in self.dir.glob("*.chunks.jsonl"))

FLUSH_BATCHES = 10


def _embed_wave(
    embedder: Embedder,
    wave: list[list[Chunk]],
    on_batch_done=None,
) -> list[list[list[float]]]:
    results: list[list[list[float]] | None] = [None] * len(wave)
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        futures = {
            pool.submit(embedder.embed_batch, [c.text for c in batch]): i
            for i, batch in enumerate(wave)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            if on_batch_done:
                on_batch_done(len(wave[futures[future]]))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return results  # type: ignore[return-value]


def embed_chunks(
    chunks: Iterable[Chunk],
    store: EmbeddingStore,
    embedder: Embedder,
    month: str,
    progress: bool = True,
) -> dict[str, int]:
    already = store.existing_hashes()
    pending: list[Chunk] = []
    seen_in_batch: set[str] = set()
    skipped = 0

    for chunk in chunks:
        if chunk.content_hash in already or chunk.content_hash in seen_in_batch:
            skipped += 1
            continue
        seen_in_batch.add(chunk.content_hash)
        pending.append(chunk)

    if not pending:
        return {"embedded": 0, "cached": skipped, "dim": 0}

    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    embedded = 0
    dim = 0
    lock = threading.Lock()

    def on_batch_done(n: int) -> None:
        nonlocal embedded
        with lock:
            embedded += n
            if progress:
                print(f"  embedded {embedded}/{len(pending)}", end="\r", flush=True)

    for w in range(0, len(batches), FLUSH_BATCHES):
        wave = batches[w : w + FLUSH_BATCHES]
        wave_vectors = _embed_wave(embedder, wave, on_batch_done)

        records: list[dict] = []
        vectors: list[list[float]] = []
        for batch, batch_vectors in zip(wave, wave_vectors):
            vectors.extend(batch_vectors)
            if not dim and batch_vectors:
                dim = len(batch_vectors[0])
            for c in batch:
                rec = asdict(c)
                rec["model"] = embedder.model
                rec["index_version"] = store.index_version
                records.append(rec)
        store.append(month, records, np.asarray(vectors, dtype=np.float32))

    if progress:
        print(" " * 40, end="\r")
    return {"embedded": embedded, "cached": skipped, "dim": dim}
