#!/usr/bin/env python3
"""Viz #1: six months of arXiv, clustered on embeddings, named by TF-IDF.

    python -m mindspace arxiv                      # the window from config
    python -m mindspace arxiv --start 2026-02 --end 2026-08
    python -m mindspace arxiv --min-cluster-size 25

Same method as the weekly view — embedding → UMAP → HDBSCAN, names from
TF-IDF top terms — applied to the paper corpus in its own database. The window
is months rather than days because arXiv's lead time is months: a paper
appearing today was being written when the discourse views were looking at
something else.

Two things differ from `week`, both because of scale. The 3-D projection is
cached against the exact set of papers, so re-clustering with another
`min_cluster_size` costs seconds instead of re-running UMAP over tens of
thousands of documents. And the browser tensor is capped: a full 1536-dim
matrix for 50,000 papers is ~300 MB, which no projector page will load, so a
deterministic subsample is exported and the number left out is printed.
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

from .. import db, paths
from ..cluster import _label_cluster, cluster_hdbscan, reduce_umap


def _month_range(start: str, end: str | None) -> tuple[date, date]:
    """'2026-02', '2026-08' -> (2026-02-01, 2026-09-01); end month inclusive."""
    lo = datetime.strptime(start, "%Y-%m").date()
    last = datetime.strptime(end or start, "%Y-%m").date()
    return lo, (last.replace(day=28) + timedelta(days=4)).replace(day=1)


def _cached_projection(cache: Path, ids: list[str]) -> np.ndarray | None:
    if not cache.exists():
        return None
    saved = np.load(cache, allow_pickle=True)
    if [str(x) for x in saved["article_ids"]] == ids:
        print("  reusing projection3d.npz — corpus unchanged")
        return saved["coords"]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mindspace arxiv")
    ap.add_argument("--start", metavar="YYYY-MM", help="first month, inclusive")
    ap.add_argument("--end", metavar="YYYY-MM", help="last month, inclusive")
    ap.add_argument("--days", type=int, help="a window ending today instead")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--max-points", type=int, default=None,
                    help="tensor rows to ship; default: modes.arxiv.max_points")
    ap.add_argument("--config", default=str(paths.CONFIG))
    ap.add_argument("--out", default=str(paths.ARXIV))
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    cl = cfg.get("clustering", {})
    ax = (cfg.get("modes") or {}).get("arxiv") or {}

    if args.start:
        lo, hi = _month_range(args.start, args.end)
    else:
        hi = date.today()
        lo = hi - timedelta(days=args.days or ax.get("days", 183))

    db.use(paths.ARXIV_DB)
    papers = [a for a in db.get_all_embedded()
              if lo.isoformat() <= (a["published_at"] or "") < hi.isoformat()]
    print(f"arxiv [{lo}..{hi}): {len(papers)} papers with vectors")
    if len(papers) < 50:
        print("too few embedded papers — run `arxiv scrape` then `arxiv embed`",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vectors = np.vstack([p["embedding"] for p in papers]).astype(np.float32)
    ids = [p["id"] for p in papers]

    cache = out / "projection3d.npz"
    coords = _cached_projection(cache, ids)
    if coords is None:
        print(f"  fitting 3-D UMAP over {len(papers)} papers "
              "(minutes; cached afterwards)…", flush=True)
        coords = reduce_umap(vectors,
                             n_neighbors=cl.get("n_neighbors", 15),
                             n_components=3,
                             min_dist=cl.get("min_dist", 0.05),
                             metric=cl.get("metric", "cosine"))
        np.savez(cache, coords=coords, article_ids=np.array(ids))
    coords = np.asarray(coords, dtype=np.float32)

    mcs = args.min_cluster_size or ax.get("min_cluster_size", 15)
    labels = cluster_hdbscan(coords, mcs, cl.get("min_samples", 2))
    n_clusters = len({int(l) for l in labels if l >= 0})
    noise = int((labels < 0).sum())
    print(f"{n_clusters} clusters, {noise} noise ({noise / len(papers):.0%}) "
          f"at min_cluster_size {mcs}")

    texts = [f"{p['title']} {p['content']}" for p in papers]
    clusters = []
    for cid in sorted({int(l) for l in labels if l >= 0}):
        members = [i for i, l in enumerate(labels) if l == cid]
        clusters.append({"cluster_id": cid, "size": len(members),
                         "keywords": _label_cluster([texts[i] for i in members])})
    clusters.sort(key=lambda c: -c["size"])
    for c in clusters[:15]:
        print(f"  [{c['cluster_id']:>4}] {c['size']:>5}  {c['keywords']}")
    if len(clusters) > 15:
        print(f"  … and {len(clusters) - 15} smaller clusters")

    # A 1536-dim tensor for the whole corpus is far too big for a browser, so
    # ship a deterministic sample — and say exactly how many were left out
    # rather than letting the page imply it shows everything.
    cap = args.max_points or ax.get("max_points") or len(papers)
    if len(papers) > cap:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(papers), cap, replace=False))
        print(f"  tensor capped at {cap}: {len(papers) - cap} papers not shipped "
              "to the browser (all of them are still clustered above)")
    else:
        keep = np.arange(len(papers))

    kw = {c["cluster_id"]: c["keywords"] for c in clusters}
    vectors[keep].tofile(out / "vectors.bytes")
    coords[keep].tofile(out / "umap3d.bytes")
    with (out / "metadata.tsv").open("w") as fh:
        fh.write("label\tcluster\tsource\ttitle\turl\tpublished\n")
        for i in keep:
            p, l = papers[i], int(labels[i])
            title = " ".join((p["title"] or "").split())[:160] or "-"
            row = [kw.get(l, "noise"), str(l) if l >= 0 else "noise",
                   p["source"] or "arXiv", title, p["url"] or "-",
                   (p["published_at"] or "-")[:10]]
            fh.write("\t".join(v if v.strip() else "-" for v in row) + "\n")

    n = len(keep)
    (out / "projector_config.json").write_text(json.dumps({"embeddings": [
        {"tensorName": f"arXiv {lo}..{hi} — embeddings",
         "tensorShape": [n, vectors.shape[1]],
         "tensorPath": "data/vectors.bytes", "metadataPath": "data/metadata.tsv"},
        {"tensorName": "UMAP 3-D cluster map", "tensorShape": [n, 3],
         "tensorPath": "data/umap3d.bytes", "metadataPath": "data/metadata.tsv"},
    ]}, indent=2))
    from collections import Counter
    months = Counter((p["published_at"] or "")[:7] for p in papers)
    last_day = hi - timedelta(days=1)
    month_end = (last_day.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    (out / "arxiv_meta.json").write_text(json.dumps({
        "window": f"{lo}..{hi}", "papers": len(papers), "shipped": n,
        "min_cluster_size": mcs, "noise": noise,
        "months": dict(sorted(months.items())),
        "partial_last_month": last_day < month_end,
        "clusters": clusters}, indent=2))
    print(f"saved -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
