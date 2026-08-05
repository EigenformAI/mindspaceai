import re
from collections import defaultdict

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


def reduce_umap(embeddings: np.ndarray, n_neighbors: int, n_components: int,
                min_dist: float, metric: str) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(embeddings)


def cluster_hdbscan(coords: np.ndarray, min_cluster_size: int,
                    min_samples: int) -> np.ndarray:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(coords)


def _label_cluster(texts: list[str], top_n: int = 5) -> str:
    if not texts:
        return "unlabeled"
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            max_features=500,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
        )
        tfidf = vec.fit_transform(texts)
        scores = np.asarray(tfidf.sum(axis=0)).ravel()
        top_idx = scores.argsort()[-top_n:][::-1]
        terms = [vec.get_feature_names_out()[i] for i in top_idx]
        return " · ".join(terms)
    except Exception:
        return "unlabeled"


def build_clusters(articles: list[dict], labels: np.ndarray) -> list[dict]:
    from .sources import strip_html

    cluster_map: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        cluster_map[int(lbl)].append(i)

    clusters = []
    for lbl, indices in sorted(cluster_map.items()):
        texts = [
            (articles[i].get("title", "") + " " + strip_html(articles[i].get("content", "")))
            for i in indices
        ]
        name = "noise" if lbl == -1 else _label_cluster(texts)
        members = [
            {
                "title": articles[i].get("title", ""),
                "url": articles[i].get("url", ""),
                "source": articles[i].get("source", ""),
                "author": articles[i].get("author", ""),
                "published_at": articles[i].get("published_at"),
            }
            for i in indices
        ]
        clusters.append({
            "cluster_id": lbl,
            "label": name,
            "size": len(indices),
            "members": members,
        })

    return sorted(clusters, key=lambda c: (-c["size"], c["cluster_id"] == -1))
