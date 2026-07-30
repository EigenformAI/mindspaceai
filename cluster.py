from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REDUCED_DIMS = 50
MIN_CLUSTER_SIZE = 15
N_MEDOIDS = 8
RANDOM_STATE = 42


@dataclass(slots=True)
class Cluster:
    cluster_id: int
    window: str
    algo_version: str
    size: int
    medoid_chunk_indices: list[int]     # rows in the window's vector matrix
    centroid: list[float] = field(default_factory=list)   # in reduced space
    top_documents: list[dict[str, Any]] = field(default_factory=list)


def reduce_dims(
    vectors: np.ndarray, n_components: int = REDUCED_DIMS, n_neighbors: int = 15
) -> np.ndarray:
    """Project onto ~50 dimensions with UMAP, using cosine distance.

    Cosine because the embedding vectors are normalised, so angle is the only
    thing that carries meaning.
    """
    import umap

    n_components = min(n_components, vectors.shape[0] - 2, vectors.shape[1])
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(vectors)


def project_2d(vectors: np.ndarray) -> np.ndarray:
    """A separate 2-D view, for eyeballing only. Never feeds a metric."""
    import umap

    return umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
    ).fit_transform(vectors)


def reduction_fingerprint(records: list[dict]) -> str:
    """Identity of the exact rows a reduction was computed over.

    Row count alone would miss a re-embedded corpus of the same size, and the
    reduction is only valid for the same vectors in the same order — HDBSCAN
    labels join back to records purely by position.
    """
    digest = hashlib.sha256()
    for r in records:
        digest.update(r["content_hash"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


class ReducedCache:
    def __init__(self, root: Path | str, index: str, dims: int, n_neighbors: int) -> None:
        self.dir = Path(root) / "reduced" / f"{index}--d{dims}-nn{n_neighbors}"

    def _paths(self, window: str) -> tuple[Path, Path]:
        return self.dir / f"{window}.npy", self.dir / f"{window}.meta.json"

    def load(self, window: str, fingerprint: str) -> np.ndarray | None:
        """The cached reduction, or None — including when the corpus changed."""
        npy, meta = self._paths(window)
        if not (npy.exists() and meta.exists()):
            return None
        if json.loads(meta.read_text())["fingerprint"] != fingerprint:
            return None
        return np.load(npy)

    def save(self, window: str, fingerprint: str, reduced: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        npy, meta = self._paths(window)
        np.save(npy, reduced)
        meta.write_text(json.dumps({"fingerprint": fingerprint, "rows": int(reduced.shape[0])}))


def cluster_vectors(
    reduced: np.ndarray,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int | None = None,
    selection: str = "eom",
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import HDBSCAN

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",       # UMAP output is euclidean, not cosine
        cluster_selection_method=selection,
    )
    labels = model.fit_predict(reduced)
    strengths = getattr(model, "probabilities_", np.ones(len(labels)))
    return labels, strengths


def build_clusters(
    records: list[dict],
    reduced: np.ndarray,
    labels: np.ndarray,
    window: str,
    algo_version: str,
    n_medoids: int = N_MEDOIDS,
) -> list[Cluster]:
    """Summarise each cluster: size, centroid, and its most central passages."""
    clusters: list[Cluster] = []

    for label in sorted({int(l) for l in labels if l >= 0}):
        idx = np.flatnonzero(labels == label)
        centroid = reduced[idx].mean(axis=0)

        # Medoids: the members closest to the centre. These are what the naming
        # models see — never the whole cluster, which would be far too large.
        distances = np.linalg.norm(reduced[idx] - centroid, axis=1)
        medoids = idx[np.argsort(distances)[:n_medoids]]

        seen: set[str] = set()
        top_docs = []
        for i in medoids:
            rec = records[i]
            if rec["document_id"] in seen:
                continue
            seen.add(rec["document_id"])
            top_docs.append(
                {
                    "document_id": rec["document_id"],
                    "published_at": rec["published_at"][:10],
                    "text": rec["text"][:300],
                }
            )

        clusters.append(
            Cluster(
                cluster_id=label,
                window=window,
                algo_version=algo_version,
                size=len(idx),
                medoid_chunk_indices=[int(i) for i in medoids],
                centroid=[float(x) for x in centroid],
                top_documents=top_docs,
            )
        )

    return clusters


class ClusterStore:
    def __init__(self, root: Path | str = "data", algo_version: str = "v1") -> None:
        self.dir = Path(root) / "clusters" / algo_version
        self.algo_version = algo_version

    def save(
        self,
        window: str,
        clusters: list[Cluster],
        labels: np.ndarray,
        strengths: np.ndarray,
        projection: np.ndarray | None = None,
        reduced: np.ndarray | None = None,
    ) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / f"{window}.clusters.jsonl").open("w", encoding="utf-8") as fh:
            for c in clusters:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(self.dir / f"{window}.labels.npy", labels)
        np.save(self.dir / f"{window}.strengths.npy", strengths)
        if projection is not None:
            np.save(self.dir / f"{window}.projection2d.npy", projection)
        if reduced is not None:
            # Storage requires caching these: re-running clustering with new
            # settings should not mean paying for the reduction again.
            np.save(self.dir / f"{window}.reduced.npy", reduced)

    def load(self, window: str) -> list[dict]:
        path = self.dir / f"{window}.clusters.jsonl"
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
