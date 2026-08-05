#!/usr/bin/env python3
"""Viz #2: the 1-week projection, clustered as usual, labeled by pure TF-IDF.

    uv run python week_projector.py                    # week 1, no cap
    uv run python week_projector.py --week 2
    uv run python week_projector.py --max-clusters 15  # the hard threshold

Self-contained on purpose: it clusters ONE week's documents and nothing else —
no 91-day background, no slider frames. The method is the pipeline's own
(embedding -> UMAP -> HDBSCAN, same knobs from config.yaml), applied to the
week alone so the sphere is filled by this week's structure. Labels are the
TF-IDF top terms — no Fable, no Haiku, no API, $0.

Every source in the DB participates — no source filter.

Writes output/projector_week/: vectors.bytes (the 1536-dim embeddings, so the
projector's neighbour panel works in the true space), umap3d.bytes (the 3-D
map), metadata.tsv, projector_config.json, week_meta.json.
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml
from .. import paths

from ..cluster import _label_cluster, cluster_hdbscan, reduce_umap

_TAGS = re.compile(r"<[^>]+>")
MAX_WORDS = 1500    # LessWrong HTML can run to tens of thousands of words


def _clean(title: str, content: str) -> str:
    text = _TAGS.sub(" ", f"{title or ''} {content or ''}")
    return " ".join(text.split()[:MAX_WORDS])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=1,
                    help="window counting back from today, 1 = most recent")
    ap.add_argument("--days", type=int, default=None,
                    help="window size; default: modes.week.days in config")
    ap.add_argument("--as-of", type=date.fromisoformat, default=None,
                    metavar="YYYY-MM-DD",
                    help="end the window at this date instead of today, to "
                         "reproduce a past week exactly (exclusive bound)")
    ap.add_argument("--max-clusters", type=int, default=50,
                    help="keep the N largest clusters, fold the rest into "
                         "noise (the brief's hard threshold: 15)")
    ap.add_argument("--db", default=str(paths.DB))
    ap.add_argument("--config", default=str(paths.CONFIG))
    ap.add_argument("--out", default=str(paths.WEEK))
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    cl = cfg.get("clustering", {})
    wk = (cfg.get("modes") or {}).get("week") or {}

    days = args.days or wk.get("days", 7)
    # `--week N` counts back from the anchor, so the two compose: --as-of
    # 2026-07-15 --week 2 is the week before the one ending on the 15th.
    anchor = args.as_of or date.today()
    hi = anchor - timedelta(days=days * (args.week - 1))
    lo = hi - timedelta(days=days)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT source, url, title, content, published_at, embedding "
        "FROM articles WHERE published_at >= ? AND published_at < ? "
        "ORDER BY published_at",
        (lo.isoformat(), hi.isoformat())).fetchall()
    conn.close()

    docs = [dict(r) for r in rows if r["embedding"]]
    skipped = len(rows) - len(docs)
    print(f"week {args.week} [{lo}..{hi}): {len(docs)} documents"
          + (f", {skipped} without embedding skipped" if skipped else ""))
    if len(docs) < 50:
        print("too few documents to cluster meaningfully", file=sys.stderr)
        sys.exit(1)

    vectors = np.vstack([np.frombuffer(d["embedding"], dtype=np.float32)
                         for d in docs])
    texts = [_clean(d["title"], d["content"]) for d in docs]

    print(f"UMAP 3-D over {len(docs)} docs...", flush=True)
    coords = reduce_umap(vectors,
                         n_neighbors=cl.get("n_neighbors", 15),
                         n_components=3,
                         min_dist=cl.get("min_dist", 0.05),
                         metric=cl.get("metric", "cosine")).astype(np.float32)
    labels = cluster_hdbscan(coords,
                             wk.get("min_cluster_size", 3),
                             cl.get("min_samples", 2))

    # The hard threshold: the N largest keep their identity, the rest fold
    # into noise — reported, never silent.
    ids, sizes = np.unique(labels[labels >= 0], return_counts=True)
    keep = ids[np.argsort(sizes)[::-1][:args.max_clusters]]
    cut = [(int(i), int(s)) for i, s in zip(ids, sizes) if i not in keep]
    labels = np.where(np.isin(labels, keep), labels, -1)
    if cut:
        print(f"hard threshold {args.max_clusters}: {len(cut)} cluster(s) "
              f"({sum(s for _, s in cut)} docs) folded into noise: "
              + ", ".join(f"#{i}({s})" for i, s in cut))

    clusters = []
    for cid in sorted(keep, key=lambda c: -int((labels == c).sum())):
        members = [i for i, l in enumerate(labels) if l == cid]
        clusters.append({"cluster_id": int(cid), "size": len(members),
                         "keywords": _label_cluster([texts[i] for i in members])})
    noise = int((labels < 0).sum())
    print(f"{len(clusters)} clusters, {noise} noise ({noise / len(docs):.0%})")
    for c in clusters:
        print(f"  [{c['cluster_id']:>3}] {c['size']:>4}  {c['keywords']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vectors.tofile(out / "vectors.bytes")
    coords.tofile(out / "umap3d.bytes")

    kw = {c["cluster_id"]: c["keywords"] for c in clusters}
    with (out / "metadata.tsv").open("w") as fh:
        fh.write("label\tcluster\tsource\ttitle\turl\tpublished\n")
        for d, l in zip(docs, labels):
            # TSV is line-oriented: a newline inside a tweet title splits one
            # row into several and the projector then refuses the whole file.
            title = " ".join((d["title"] or "").split())[:160] or "-"
            row = [kw.get(int(l), "noise"),
                   str(int(l)) if l >= 0 else "noise",
                   d["source"],
                   title,
                   d["url"] or "-",
                   (d["published_at"] or "-")[:10]]
            fh.write("\t".join(v if v.strip() else "-" for v in row) + "\n")

    # Paths are resolved by the projector PAGE, whose data sits in data/ —
    # a bare filename here 404s even though the config itself loads fine.
    (out / "projector_config.json").write_text(json.dumps({"embeddings": [
        {"tensorName": f"Week {args.week} ({lo}..{hi}) — embeddings",
         "tensorShape": [len(docs), vectors.shape[1]],
         "tensorPath": "data/vectors.bytes",
         "metadataPath": "data/metadata.tsv"},
        {"tensorName": "UMAP 3-D cluster map",
         "tensorShape": [len(docs), 3],
         "tensorPath": "data/umap3d.bytes",
         "metadataPath": "data/metadata.tsv"},
    ]}, indent=2))
    (out / "week_meta.json").write_text(json.dumps({
        "week": args.week, "window": f"{lo}..{hi}", "documents": len(docs),
        "noise": noise, "max_clusters": args.max_clusters,
        "clusters": clusters, "cut_by_threshold": cut}, indent=2))
    print(f"saved -> {out}/")


if __name__ == "__main__":
    main()
