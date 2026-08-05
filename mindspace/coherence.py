from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_MEMBERS = 2


def mean_pairwise_cosine(vectors: np.ndarray) -> float | None:
    n = vectors.shape[0]
    if n < MIN_MEMBERS:
        return None

    arr = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-12)

    sims = arr @ arr.T
    iu = np.triu_indices(n, k=1)
    return float(sims[iu].mean())


def corpus_tfidf(texts: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )
    matrix = vec.fit_transform(texts)
    return vec, matrix


def score_clusters(labels: np.ndarray,
                   embeddings: np.ndarray,
                   tfidf_matrix,
                   measure_map: str = "corrected") -> list[dict]:
    if measure_map not in ("corrected", "as-written"):
        raise ValueError(
            f"measure_map must be 'corrected' or 'as-written', got {measure_map!r}"
        )

    records = []
    for cid in sorted({int(x) for x in labels} - {-1}):
        idx = np.flatnonzero(labels == cid)

        by_meaning = mean_pairwise_cosine(embeddings[idx])
        by_words = mean_pairwise_cosine(np.asarray(tfidf_matrix[idx].todense()))

        if measure_map == "corrected":
            lexical, semantic = by_words, by_meaning
        else:
            lexical, semantic = by_meaning, by_words

        records.append({
            "cluster_id": cid,
            "size": int(idx.size),
            "lexical": lexical,
            "semantic": semantic,
            "cosine_over_embeddings": by_meaning,
            "cosine_over_tfidf": by_words,
            "measure_map": measure_map,
        })

    _add_name_gap(records)

    records.sort(key=lambda r: (r["name_gap"] is None, -(r["name_gap"] or 0.0)))
    return records


def _add_name_gap(records: list[dict]) -> None:
    usable = [r for r in records
              if r["lexical"] is not None and r["semantic"] is not None]

    for r in records:
        r["name_gap"] = None
        r["lexical_rank"] = None
        r["semantic_rank"] = None

    if len(usable) < 2:
        return

    n = len(usable)
    lex_rank = (rankdata([r["lexical"] for r in usable]) - 1) / (n - 1)
    sem_rank = (rankdata([r["semantic"] for r in usable]) - 1) / (n - 1)

    for r, lr, sr in zip(usable, lex_rank, sem_rank):
        r["lexical_rank"] = float(lr)
        r["semantic_rank"] = float(sr)
        r["name_gap"] = float(sr - lr)
        r["raw_difference"] = float(r["semantic"] - r["lexical"])


def write_name_gaps(records: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    usable = [r for r in records if r["name_gap"] is not None]
    dropped = len(records) - len(usable)

    path.write_text(json.dumps(usable, indent=2))

    positive = sum(1 for r in usable if r["name_gap"] > 0)
    print(f"[coherence] {len(usable)} clusters scored → {path}")
    print(f"[coherence] {positive} with a positive name gap "
          f"(semantically tighter than they are lexically)")
    if dropped:
        print(f"[coherence] {dropped} cluster(s) too small to measure "
              f"(< {MIN_MEMBERS} members) — excluded, not scored as zero")
    return path
