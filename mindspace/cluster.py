import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


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


def project_anchored(embeddings: np.ndarray, ids: list[str], anchor: Path,
                     *, n_neighbors: int, n_components: int, min_dist: float,
                     metric: str, log=print) -> np.ndarray:
    """Coordinates on a map that is learned once and then kept.

    UMAP does not compute a position, it learns an arrangement, and it learns a
    different one from a different corpus. Fitting afresh each run therefore
    moved every document, including the ones that had not changed — measured on
    this corpus, adding a week (+10% of documents) left only 8 of 24 of the
    previous week's clusters ≥90% intact, and renumbered even those. Cluster
    ids are what names are addressed to, so every run invalidated every name.

    Keeping the fitted model fixes that at the root: documents already placed
    read their coordinates back unchanged, and only new ones are transformed
    onto the same map. Weekly clusterings over unchanged documents then repeat
    exactly, ids included.

    The map is never refitted while the anchor exists — that is the guarantee.
    It is also the expiry date: as the corpus drifts from what the map was
    learned on, new documents are placed increasingly badly, so the anchor
    should be deleted and relearned periodically, accepting that names expire
    once when it is.
    """
    import joblib
    import umap

    ids = [str(i) for i in ids]
    if anchor.exists():
        saved = joblib.load(anchor)
        placed = saved["coords"]
        fresh = [i for i in ids if i not in placed]
        log(f"[umap] anchor: {len(ids) - len(fresh)} documents keep their "
            f"coordinates, {len(fresh)} placed onto the same map")
        if fresh:
            at = {i: n for n, i in enumerate(ids)}
            new = saved["reducer"].transform(
                np.vstack([embeddings[at[i]] for i in fresh]).astype(np.float32))
            placed.update(zip(fresh, np.asarray(new)))
            joblib.dump(saved, anchor, compress=3)
        return np.vstack([placed[i] for i in ids])

    log(f"[umap] no anchor yet — learning the map from {len(ids)} documents "
        f"(minutes); later runs reuse it")
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                        min_dist=min_dist, metric=metric, random_state=42,
                        verbose=False)
    coords = np.asarray(reducer.fit_transform(embeddings))
    anchor.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"reducer": reducer, "coords": dict(zip(ids, coords))},
                anchor, compress=3)
    log(f"[umap] anchor written → {anchor.name} "
        f"({anchor.stat().st_size / 1048576:.0f} MB). Delete it to relearn the "
        f"map; doing so expires every generated name.")
    return coords


def cluster_hdbscan(coords: np.ndarray, min_cluster_size: int,
                    min_samples: int) -> np.ndarray:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(coords)


# sklearn's English list keeps the words that make a label unreadable: it drops
# "the" and "is" but not "maybe", "gets", "think" or "people", so a cluster can
# end up named "maybe · gets · people · like". These are the fillers that
# survive it — conversational verbs and hedges that say nothing about what a
# cluster is about, plus the contraction fragments the tokeniser leaves behind
# when it splits "doesn't".
_EXTRA_STOPWORDS = frozenset("""
    maybe gets getting kept keep like just really think thinking thing things
    way ways people person actually pretty lot lots new good little doing done
    used using make makes made want wants need needs know known say says said
    look looks looking come comes going went able sure yeah okay
    don doesn isn didn wasn aren won couldn shouldn wouldn haven hasn model models
""".split())

_STOPWORDS = sorted(ENGLISH_STOP_WORDS | _EXTRA_STOPWORDS)


def _vectorizer(**kw) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=_STOPWORDS,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
        **kw,
    )


def label_clusters(texts: list[str], labels, top_n: int = 5) -> dict[int, str]:
    """A TF-IDF name per cluster, scored against the whole document set.

    `_label_cluster` fits a vectoriser on one cluster's own documents, so IDF
    is computed inside the cluster and a word that is everywhere in the corpus
    still wins as long as it is everywhere in *this* cluster too. That is how
    "people", "think" and "like" became cluster names. Fitting once over every
    document makes the comparison the one that matters — frequent here AND
    rare elsewhere — so ubiquitous words lose on their own, without having to
    be listed by hand.

    `min_df=2` drops terms appearing in a single document: with thousands of
    documents those are typos and identifiers, never a theme.
    """
    labels = np.asarray(labels)
    out: dict[int, str] = {}
    if not texts:
        return out
    try:
        vec = _vectorizer(min_df=2, max_features=20000)
        matrix = vec.fit_transform(texts)
        terms = np.asarray(vec.get_feature_names_out())
    except Exception:
        return {int(l): "unlabeled" for l in set(labels.tolist())}

    for lbl in sorted({int(x) for x in labels}):
        rows = np.flatnonzero(labels == lbl)
        if rows.size == 0:
            continue
        mean = np.asarray(matrix[rows].mean(axis=0)).ravel()
        top = np.argsort(mean)[::-1][:top_n]
        out[lbl] = " · ".join(terms[top]) if mean[top[0]] > 0 else "unlabeled"
    return out


def _label_cluster(texts: list[str], top_n: int = 5) -> str:
    if not texts:
        return "unlabeled"
    try:
        vec = _vectorizer(max_features=500)
        tfidf = vec.fit_transform(texts)
        scores = np.asarray(tfidf.sum(axis=0)).ravel()
        top_idx = scores.argsort()[-top_n:][::-1]
        terms = [vec.get_feature_names_out()[i] for i in top_idx]
        return " · ".join(terms)
    except Exception:
        return "unlabeled"


def build_clusters(articles: list[dict], labels: np.ndarray) -> list[dict]:
    from .sources import text_for_tfidf

    cluster_map: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        cluster_map[int(lbl)].append(i)

    clusters = []
    for lbl, indices in sorted(cluster_map.items()):
        texts = [
            text_for_tfidf(articles[i].get("title", ""),
                           articles[i].get("content", ""))
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
