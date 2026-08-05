"""Embeddings via OpenRouter.

One key for everything in this project — embedding, compression, synthesis,
labelling — so there is a single place for it to go wrong and a single bill.

`openai/text-embedding-3-small` returns 1536 dimensions, which is what the spec
means by "the original 1536-dim embeddings". Verified against the live endpoint
rather than assumed: OpenRouter's `/api/v1/models` listing does not mention any
embedding model at all, yet `/api/v1/embeddings` serves them. That listing is
not evidence either way.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from . import paths

_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/text-embedding-3-small"
_COST_PATH = paths.COST / "embed_cost.json"


def _record_spend(model: str, articles: int, tokens: int,
                  cost_usd: float, unpriced_batches: int) -> None:
    """Append this run's real spend to output/embed_cost.json.

    Appended, not overwritten: `run_embedding` saves in chunks and can be
    re-run after a failure, so a single corpus may be embedded across several
    invocations and the bill is their sum.
    """
    _COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    if _COST_PATH.exists():
        try:
            runs = json.loads(_COST_PATH.read_text()).get("runs", [])
        except (json.JSONDecodeError, AttributeError):
            # A damaged file is not a reason to lose this run's figure.
            runs = []
    runs.append({
        "model": model,
        "articles": articles,
        "prompt_tokens": tokens,
        "cost_usd": cost_usd,
        "batches_without_a_reported_cost": unpriced_batches,
    })
    _COST_PATH.write_text(json.dumps({
        "total_usd": sum(r["cost_usd"] for r in runs),
        "total_articles": sum(r["articles"] for r in runs),
        "runs": runs,
    }, indent=2))


def _resolve_model(model_name: str) -> str:
    """Accept the bare OpenAI name from config.yaml and route it via OpenRouter."""
    if not model_name:
        return _DEFAULT_MODEL
    if "/" in model_name:
        return model_name
    if model_name.startswith("text-embedding"):
        return f"openai/{model_name}"
    return _DEFAULT_MODEL


def _create_with_retry(client, model: str, batch: list[str], attempts: int = 4):
    """One embeddings call that rides out minutes of outage, not just seconds.

    The SDK already retries twice per call with backoff, which absorbs
    momentary hiccups. This outer loop exists for the unattended case — a
    refresh left running while nobody watches — where a rate-limit spell or a
    flapping connection lasting minutes would otherwise kill the whole stage.

    A 4xx other than 429 is a decision, not a hiccup — a bad key, an unknown
    model, an exhausted budget. Retrying those wastes time and buries the one
    message that explains the problem, so they raise immediately.
    """
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return client.embeddings.create(model=model, input=batch)
        except (APIConnectionError, APITimeoutError) as exc:
            last = exc
        except APIStatusError as exc:
            if exc.status_code != 429 and 400 <= exc.status_code < 500:
                raise
            last = exc
        if attempt == attempts - 1:
            break
        wait = 5 * (attempt + 1)
        print(f"[embed] batch failed ({last}); retry {attempt + 1}/{attempts - 1} "
              f"in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(
        f"embedding batch failed after {attempts} rounds: {last}"
    ) from last


def embed_articles(articles: list[dict], model_name: str,
                   batch_size: int = 64) -> np.ndarray:
    """One L2-normalised vector per article, in the order given.

    There is deliberately no local fallback. The previous version dropped to
    sentence-transformers whenever a key was missing, which would have produced
    384-dimension vectors — silently incompatible with everything downstream and
    with the spec. Failing loudly on a missing key is the smaller problem.
    """
    from openai import OpenAI

    from .sources import embed_text

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Everything in this project routes "
            "through OpenRouter — embedding, compression, synthesis, labelling — "
            "so there is nothing to fall back to."
        )

    texts = [embed_text(a.get("title", ""), a.get("content", "")) for a in articles]
    model = _resolve_model(model_name)
    client = OpenAI(api_key=api_key, base_url=_BASE_URL)

    print(f"[embed] OpenRouter {model}, {len(texts)} articles")

    all_vecs: list[list[float]] = []
    spent = 0.0
    tokens = 0
    unpriced = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
        batch = texts[i : i + batch_size]
        resp = _create_with_retry(client, model, batch)
        # Order by the response's own `index`; position is not guaranteed.
        all_vecs.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))

        # OpenRouter reports the real cost on every embeddings response —
        # unlike chat, it does so without being asked. Accumulated rather than
        # estimated from a price list, which would drift.
        usage = getattr(resp, "usage", None)
        cost = getattr(usage, "cost", None) if usage else None
        if cost is None:
            unpriced += 1
        else:
            spent += cost
        tokens += getattr(usage, "prompt_tokens", 0) or 0

    shown = f"${spent:.6f}" if spent < 0.01 else f"${spent:.4f}"
    print(f"[embed] {tokens:,} tokens, {shown}"
          + (f" ({unpriced} batch(es) reported no cost)" if unpriced else ""))
    _record_spend(model, len(texts), tokens, spent, unpriced)

    arr = np.array(all_vecs, dtype=np.float32)
    if arr.shape[0] != len(texts):
        raise RuntimeError(
            f"asked for {len(texts)} embeddings, got {arr.shape[0]} — refusing to "
            "return a misaligned array, which would attach vectors to the wrong "
            "articles with no error anywhere"
        )
    print(f"[embed] {arr.shape[0]} vectors, {arr.shape[1]} dimensions")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)
