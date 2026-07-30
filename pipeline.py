#!/usr/bin/env python3
"""
mindspace pipeline: scrape → embed → cluster → output
Usage:
    python pipeline.py [--days N] [--skip-scrape] [--skip-embed]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _flush(items: list[dict]) -> int:
    import db
    saved = 0
    for a in items:
        url = a.get("url", "")
        if not url:
            continue
        try:
            db.upsert_article(
                source=a.get("source", ""),
                url=url,
                title=a.get("title", ""),
                content=a.get("content", ""),
                author=a.get("author"),
                published_at=a.get("published_at"),
            )
            saved += 1
        except Exception:
            pass
    return saved


def run_scrapers(cfg: dict, lookback_days: int) -> int:
    import db
    total = 0
    sources = cfg.get("sources", {})

    # RSS feeds
    from scrapers.rss import scrape_rss
    for feed in sources.get("rss", []):
        print(f"[scrape] RSS: {feed['name']}")
        try:
            items = scrape_rss(feed["name"], feed["url"], lookback_days)
            saved = _flush(items)
            print(f"         → {len(items)} items ({saved} new)")
            total += saved
        except Exception as exc:
            print(f"         ! error: {exc}", file=sys.stderr)

    # LessWrong
    lw_cfg = sources.get("lesswrong", {})
    if lw_cfg.get("enabled"):
        from scrapers.lesswrong import scrape_lesswrong
        print("[scrape] LessWrong")
        items = scrape_lesswrong(lw_cfg.get("limit", 40), lookback_days)
        saved = _flush(items)
        print(f"         → {len(items)} items ({saved} new)")
        total += saved

    # Alignment Forum
    af_cfg = sources.get("alignment_forum", {})
    if af_cfg.get("enabled"):
        from scrapers.lesswrong import scrape_alignment_forum
        print("[scrape] Alignment Forum")
        items = scrape_alignment_forum(af_cfg.get("limit", 40), lookback_days)
        saved = _flush(items)
        print(f"         → {len(items)} items ({saved} new)")
        total += saved

    # Hacker News
    hn_cfg = sources.get("hackernews", {})
    if hn_cfg.get("enabled"):
        from scrapers.hackernews import scrape_hackernews
        print("[scrape] Hacker News")
        items = scrape_hackernews(
            hn_cfg.get("queries", ["AI"]),
            hn_cfg.get("limit_per_query", 20),
            lookback_days,
        )
        saved = _flush(items)
        print(f"         → {len(items)} items ({saved} new, deduped)")
        total += saved

    # GitHub Trending
    gh_cfg = sources.get("github_trending", {})
    if gh_cfg.get("enabled"):
        from scrapers.github_trending import scrape_github_trending
        print("[scrape] GitHub Trending")
        items = scrape_github_trending()
        saved = _flush(items)
        print(f"         → {len(items)} items ({saved} new)")
        total += saved

    # HF Papers
    hf_cfg = sources.get("hf_papers", {})
    if hf_cfg.get("enabled"):
        from scrapers.hf_papers import scrape_hf_papers
        print("[scrape] HuggingFace Papers")
        items = scrape_hf_papers(hf_cfg.get("limit", 50), lookback_days)
        saved = _flush(items)
        print(f"         → {len(items)} items ({saved} new)")
        total += saved

    # Twitter via Grok API live search
    tw_cfg = sources.get("twitter", {})
    if tw_cfg.get("enabled"):
        xai_key = os.environ.get("XAI_API_KEY", "")
        if not xai_key:
            print("[scrape] Twitter: XAI_API_KEY not set — skipping", file=sys.stderr)
        else:
            from scrapers.grok_twitter import scrape_grok_twitter
            print("[scrape] Twitter (Grok live search)")
            items = scrape_grok_twitter(
                prompts=tw_cfg.get("prompts", []),
                api_key=xai_key,
                lookback_days=lookback_days,
                limit_per_prompt=tw_cfg.get("limit_per_prompt", 15),
            )
            saved = _flush(items)
            print(f"         → {len(items)} tweets ({saved} new)")
            total += saved

    return total


def run_embedding(cfg: dict):
    """Chunk the stored articles into paragraphs, then embed the paragraphs.

    Two differences from the article-level version this replaces, both from the
    method document's Stage 2 ("I'd embed paragraphs"):

      * The unit is a paragraph, not an article. One vector for a 40,000-word
        post lands at the average of everything it discusses, which is a point
        representing none of them.
      * Chunks and vectors live in files, not in the articles table, because
        chunks.jsonl row N has to describe vectors.npy row N and clustering
        joins them by position alone.

    Articles stay in SQLite. This is the boundary between collection and
    analysis, not a replacement for db.py.
    """
    import db
    from chunk import chunk_document
    from embed import Embedder, EmbeddingStore, embed_chunks, model_slug

    emb_cfg = cfg.get("embedding", {})
    model = emb_cfg.get("model", "qwen/qwen3-embedding-8b")
    index = model_slug(model)
    store = EmbeddingStore(cfg.get("data_dir", "data"), index)
    embedder = Embedder(model)
    print(f"[embed] index: {index}")

    months = db.months_present()
    if not months:
        print("[embed] no dated articles — run scrape first")
        return

    for month in months:
        skipped: dict = {}
        documents = db.read_documents(month, skipped=skipped)
        if not documents:
            continue
        chunks = [c for doc in documents for c in chunk_document(doc)]
        print(f"[embed] {month}: {len(documents)} documents -> {len(chunks)} chunks",
              flush=True)
        if skipped:
            # Reported, never silent: a dropped row reads as full coverage.
            print(f"[embed] {month}: skipped {skipped}")
        result = embed_chunks(chunks, store, embedder, month)
        print(f"[embed] {month}: embedded {result['embedded']}, "
              f"cached {result['cached']}", flush=True)


def run_clustering(cfg: dict):
    """Cluster the embedded paragraphs. Naming and export are separate stages.

    Clustering runs on a ~50-dimension reduction, not on the 2-D view. HDBSCAN
    loses contrast at full dimensionality, and distances in a 2-D projection are
    not meaningful enough to cluster on — that view is for looking at only.
    """
    from cluster import (
        ClusterStore,
        ReducedCache,
        build_clusters,
        cluster_vectors,
        reduce_dims,
        reduction_fingerprint,
    )
    from embed import EmbeddingStore, model_slug
    from export import export_projector

    data_dir = Path(cfg.get("data_dir", "data"))
    index = model_slug(cfg.get("embedding", {}).get("model", "qwen/qwen3-embedding-8b"))
    emb = EmbeddingStore(data_dir, index)

    import db

    months = db.months_present()
    if not months:
        print("[cluster] nothing embedded yet")
        return
    window = months[0] if len(months) == 1 else f"{months[0]}_{months[-1]}"

    records, vectors = emb.load_window(months)
    print(f"[cluster] window {window}: {len(records)} chunks, {vectors.shape[1]} dims")

    cl = cfg.get("clustering", {})
    dims = cl.get("dims", 50)
    n_neighbors = cl.get("n_neighbors", 15)
    min_cluster_size = cl.get("min_cluster_size", 5)
    min_samples = cl.get("min_samples", 3)
    selection = cl.get("selection", "eom")

    cache = ReducedCache(data_dir, index, dims, n_neighbors)
    fingerprint = reduction_fingerprint(records)
    reduced = cache.load(window, fingerprint)
    if reduced is None:
        print(f"[cluster] reducing to {dims} dims (UMAP)…", flush=True)
        reduced = reduce_dims(vectors, dims, n_neighbors)
        cache.save(window, fingerprint, reduced)
    else:
        print("[cluster] reduction cached — UMAP skipped")

    print("[cluster] HDBSCAN…", flush=True)
    labels, strengths = cluster_vectors(reduced, min_cluster_size, min_samples, selection)
    n_clusters = len({int(l) for l in labels if l >= 0})
    noise = int((labels < 0).sum())
    print(f"[cluster] {n_clusters} clusters, {noise} noise ({noise / len(labels):.0%})")

    # Derived from the settings, so two runs cannot overwrite each other and a
    # directory name always says what produced it.
    algo = (f"{index}--d{dims}-nn{n_neighbors}-mcs{min_cluster_size}"
            f"-ms{min_samples}-{selection}")
    clusters = build_clusters(records, reduced, labels, window, algo)
    ClusterStore(data_dir, algo).save(window, clusters, labels, strengths, None, reduced)
    print(f"[cluster] saved to {data_dir}/clusters/{algo}/")


def run_export(cfg: dict):
    """Write the TensorFlow Projector files. The deliverable.

    Runs after naming so `cluster_name` is populated; on a first pass with no
    names yet the column falls back to "(unnamed N)" rather than being blank —
    the Projector parses an empty cell to null and then calls .toString() on it,
    which takes down the whole view.
    """
    import db
    from cluster import ClusterStore
    from embed import EmbeddingStore, model_slug
    from export import export_projector
    from naming import NamingStore

    data_dir = Path(cfg.get("data_dir", "data"))
    index = model_slug(cfg.get("embedding", {}).get("model", ""))
    cl = cfg.get("clustering", {})
    dims = cl.get("dims", 50)
    algo = (f"{index}--d{dims}-nn{cl.get('n_neighbors', 15)}"
            f"-mcs{cl.get('min_cluster_size', 5)}-ms{cl.get('min_samples', 3)}"
            f"-{cl.get('selection', 'eom')}")

    months = db.months_present()
    window = months[0] if len(months) == 1 else f"{months[0]}_{months[-1]}"

    store = ClusterStore(data_dir, algo)
    clusters = store.load(window)
    if not clusters:
        print("[output] no clusters — run clustering first")
        return
    records, _ = EmbeddingStore(data_dir, index).load_window(months)
    labels = np.load(data_dir / "clusters" / algo / f"{window}.labels.npy")
    reduced = np.load(data_dir / "clusters" / algo / f"{window}.reduced.npy")

    names = {}
    for row in NamingStore(data_dir, window, algo).load():
        if row["names"] and row["cluster_id"] not in names:
            names[row["cluster_id"]] = row["names"][0]

    documents = {d["external_id"]: d for d in db.read_documents()}
    urls = {k: v["url"] for k, v in documents.items()}
    titles = {k: v["title"] for k, v in documents.items()}

    out_dir = Path(cfg.get("output", {}).get("dir", "./output")) / algo
    stats = export_projector(
        records, reduced, labels, clusters,
        out_dir=out_dir,
        cluster_names=names,
        urls=urls,
        titles=titles,
        binary=True,
        url_prefix="data",
    )
    print(f"[output] projector files -> {out_dir}  "
          f"({stats['points']} points, {len(names)} of {len(clusters)} named)")

def run_naming(cfg: dict, top: int | None = None):
    """Ask models to name each cluster, from its medoid passages.

    Every model sees the same passages, which is what makes agreement between
    them mean anything: they are reacting to identical evidence, so convergence
    reflects the concept rather than what each model was shown.

    Free models by default — they cost nothing and are not blocked by the
    account spend limit. Two different labs, because two models from one family
    agree because they were trained alike, not because a term has settled.
    """
    import db
    from cluster import ClusterStore
    from embed import load_api_key, model_slug
    from naming import NamingStore, name_many

    data_dir = Path(cfg.get("data_dir", "data"))
    index = model_slug(cfg.get("embedding", {}).get("model", ""))
    cl = cfg.get("clustering", {})
    algo = (f"{index}--d{cl.get('dims', 50)}-nn{cl.get('n_neighbors', 15)}"
            f"-mcs{cl.get('min_cluster_size', 5)}-ms{cl.get('min_samples', 3)}"
            f"-{cl.get('selection', 'eom')}")

    months = db.months_present()
    window = months[0] if len(months) == 1 else f"{months[0]}_{months[-1]}"

    clusters = ClusterStore(data_dir, algo).load(window)
    if not clusters:
        print("[naming] no clusters — run clustering first")
        return
    clusters.sort(key=lambda c: -c["size"])
    if top:
        clusters = clusters[:top]

    store = NamingStore(data_dir, window, algo)
    done = store.existing()
    key = load_api_key()
    models = cfg.get("naming", {}).get("models", ["google/gemma-4-31b-it:free"])

    for model in models:
        def on_result(cluster, variant, result, model=model):
            store.append(result)
            names = " | ".join(result.names) or "(nothing parsed)"
            print(f"[naming] {model.split('/')[-1]} c{cluster['cluster_id']:>3} "
                  f"({cluster['size']:>3}) {names}", flush=True)

        named, cached = name_many(clusters, model, key, ["label"], window, done, on_result)
        print(f"[naming] {model}: named {named}, cached {cached}")


def main():
    parser = argparse.ArgumentParser(description="mindspace AI topic pipeline")
    parser.add_argument("--days", type=int, default=None,
                        help="lookback window in days (overrides config)")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="skip scraping, use existing DB")
    parser.add_argument("--skip-naming", action="store_true",
                        help="do not ask models to name the clusters")
    parser.add_argument("--name-top", type=int, default=None,
                        help="name only the N largest clusters")
    parser.add_argument("--skip-embed", action="store_true",
                        help="skip embedding step")
    parser.add_argument("--scrape-only", action="store_true",
                        help="scrape and save to DB, then stop (no embed/cluster)")
    parser.add_argument("--config", default="config.yaml",
                        help="path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    lookback_days = args.days or cfg.get("lookback_days", 7)

    import db
    db.init_db()

    if not args.skip_scrape:
        print(f"\n── Scraping (lookback: {lookback_days}d) ──────────────────")
        new_count = run_scrapers(cfg, lookback_days)
        print(f"\n[db] {new_count} new articles saved | total: {db.article_count()}")

    if args.scrape_only:
        print("\n[done] scrape-only mode — skipping embed and cluster")
        return

    if not args.skip_embed:
        print("\n── Embedding ──────────────────────────────────────────")
        run_embedding(cfg)

    print("\n── Clustering ─────────────────────────────────────────")
    run_clustering(cfg)
    if not args.skip_naming:
        run_naming(cfg, args.name_top)
    run_export(cfg)


if __name__ == "__main__":
    main()
