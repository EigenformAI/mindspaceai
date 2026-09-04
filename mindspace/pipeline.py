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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from . import paths

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_config(path: str | None = None) -> dict:
    with open(path or paths.CONFIG) as f:
        return yaml.safe_load(f)


def as_of() -> date:
    """The date every window is measured back from: today, UTC.

    A DATE, not a timestamp. Anchoring on `datetime.now()` meant two runs an
    hour apart covered slightly different seven-day windows, which changed the
    clustering, which changed the cluster ids — and the Fable attractors are
    keyed to those ids, so a re-run silently attached expensive prose to the
    wrong documents. Nothing failed; the map just said the wrong thing.

    With a date, every run on the same day is identical and a redraw costs
    nothing. `--as-of` overrides it to reproduce an earlier day.
    """
    return _AS_OF or datetime.now(timezone.utc).date()


_AS_OF: date | None = None


def _cutoff(days: int) -> str:
    """ISO timestamp `days` before `as_of()`, to compare against `published_at`.

    A plain string comparison is safe because `db.upsert_article` normalises
    every publication date to ISO 8601 UTC on the way in, and ISO sorts
    lexicographically. It would not be safe on the raw scraper output — RFC 2822
    dates ("Fri, 24 Jul …") sort after every ISO date and would pass a lower
    bound while failing an upper one.
    """
    return (as_of() - timedelta(days=days)).isoformat()


def _flush(items: list[dict]) -> int:
    from . import db
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


def _run_source(name: str, fetch, failed: list[str], noun: str = "items") -> int:
    """Fetch one source, store what came back, and record it if it could not.

    A source that fails and a source with nothing new both used to print
    "0 items" and let the run finish clean. On a weekly pipeline that is how a
    feed goes dark for a month unnoticed — the totals only sag, and the first
    real symptom is a cluster that stops mentioning it.
    """
    from .sources import SourceUnavailable
    try:
        items = fetch()
    except SourceUnavailable as exc:
        failed.append(exc.source)
        print(f"         ! {exc.source} unavailable — nothing collected from it "
              f"this run", file=sys.stderr)
        return 0
    except Exception as exc:
        failed.append(name)
        print(f"         ! {name} failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 0
    saved = _flush(items)
    print(f"         → {len(items)} {noun} ({saved} new)")
    return saved


def run_scrapers(cfg: dict, lookback_days: int) -> tuple[int, list[str]]:
    from . import db
    total = 0
    failed: list[str] = []
    sources = cfg.get("sources", {})

    # Exclusive upper bound of the scrape window. Live runs use this moment;
    # with --as-of/--week it is midnight UTC of the anchor date, so a
    # historical scrape fills exactly the half-open [anchor-days, anchor)
    # that the clustering windows will read back.
    if _AS_OF:
        until = datetime(_AS_OF.year, _AS_OF.month, _AS_OF.day,
                         tzinfo=timezone.utc)
    else:
        until = datetime.now(timezone.utc)

    # RSS feeds
    from .sources.rss import scrape_rss
    for feed in sources.get("rss", []):
        print(f"[scrape] RSS: {feed['name']}")
        total += _run_source(
            feed["name"],
            lambda f=feed: scrape_rss(f["name"], f["url"], lookback_days,
                                      until=until),
            failed)

    # LessWrong — GraphQL first, then the site's agent API if it is refused.
    # GraphQL stays primary: one request for the whole window, with the date
    # bounds applied server-side. The fallback costs a request per post and its
    # 100-post ceiling reaches back only about five days, so it repairs a gap
    # rather than replacing the normal path.
    lw_cfg = sources.get("lesswrong", {})
    if lw_cfg.get("enabled"):
        from .sources.lesswrong import scrape_lesswrong
        print("[scrape] LessWrong")

        from .sources import SourceUnavailable

        def _lesswrong():
            try:
                return scrape_lesswrong(lw_cfg.get("limit", 40), lookback_days,
                                        until=until)
            except SourceUnavailable:
                from .sources.lesswrong_api import scrape_lesswrong_api
                print("         · GraphQL refused — falling back to the "
                      "documented agent API (/api/latest)", file=sys.stderr)
                return scrape_lesswrong_api(lookback_days, until=until)

        total += _run_source("LessWrong", _lesswrong, failed)

    # Alignment Forum
    af_cfg = sources.get("alignment_forum", {})
    if af_cfg.get("enabled"):
        from .sources.lesswrong import scrape_alignment_forum
        print("[scrape] Alignment Forum")
        total += _run_source(
            "Alignment Forum",
            lambda: scrape_alignment_forum(af_cfg.get("limit", 40),
                                           lookback_days, until=until),
            failed)

    # Hacker News
    hn_cfg = sources.get("hackernews", {})
    if hn_cfg.get("enabled"):
        from .sources.hackernews import scrape_hackernews
        print("[scrape] Hacker News")
        total += _run_source(
            "Hacker News",
            lambda: scrape_hackernews(
                hn_cfg.get("queries", ["AI"]),
                hn_cfg.get("limit_per_query", 50),
                lookback_days,
                min_points=hn_cfg.get("min_points", 10),
                until=until),
            failed, noun="items, deduped")

    # GitHub Trending
    gh_cfg = sources.get("github_trending", {})
    if gh_cfg.get("enabled"):
        if _AS_OF:
            # Trending is a snapshot of now — there is no API for "what was
            # trending in July". Scraping it into a historical window would
            # stamp today's repos with today's date and quietly misfile them.
            print("[scrape] GitHub Trending: no history exists — skipped for "
                  "an anchored window; it accumulates on live weekly runs",
                  file=sys.stderr)
        else:
            from .sources.github_trending import scrape_github_trending
            print("[scrape] GitHub Trending")
            # lookback_days picks the `since` window — GitHub's default is
            # `daily`, so a weekly run was seeing one day in seven.
            total += _run_source("GitHub Trending",
                                 lambda: scrape_github_trending(lookback_days),
                                 failed)

    # HF Papers
    hf_cfg = sources.get("hf_papers", {})
    if hf_cfg.get("enabled"):
        from .sources.hf_papers import scrape_hf_papers
        print("[scrape] HuggingFace Papers")
        total += _run_source(
            "HuggingFace Papers",
            lambda: scrape_hf_papers(
                hf_cfg.get("limit_per_day", hf_cfg.get("limit", 50)),
                lookback_days, until=until),
            failed)


    # Twitter via Grok API live search
    tw_cfg = sources.get("twitter", {})
    if tw_cfg.get("enabled"):
        xai_key = os.environ.get("XAI_API_KEY", "")
        if not xai_key:
            print("[scrape] Twitter: XAI_API_KEY not set — skipping", file=sys.stderr)
        else:
            from .sources.grok_twitter import scrape_grok_twitter
            print("[scrape] Twitter (Grok live search)")
            total += _run_source(
                "Twitter/Grok",
                lambda: scrape_grok_twitter(
                    prompts=tw_cfg.get("prompts", []),
                    api_key=xai_key,
                    lookback_days=lookback_days,
                    limit_per_prompt=tw_cfg.get("limit_per_prompt", 15),
                    until=until),
                failed, noun="tweets")

    return total, failed


def _month_range(start: str, end: str) -> tuple[date, date]:
    """'2026-02', '2026-08' -> (2026-02-01, 2026-09-01).

    The end month is inclusive, matching how a person says "February to
    August"; the returned upper bound is exclusive, matching every other
    window in this project.
    """
    lo = datetime.strptime(start, "%Y-%m").date()
    last = datetime.strptime(end or start, "%Y-%m").date()
    return lo, (last.replace(day=28) + timedelta(days=4)).replace(day=1)


def arxiv_scrape(argv: list[str] | None = None) -> int:
    """Fetch papers into their own database.

    Dates rather than a rolling window: arXiv is collected as history, in one
    or two passes over a range of months, not weekly like the discourse
    sources. `db.use` points every write at data/arxiv.db, which is what keeps
    a thousand papers a day out of a corpus that sees fifty documents.
    """
    from . import db
    from .sources.arxiv import scrape_arxiv

    ap = argparse.ArgumentParser(prog="mindspace arxiv scrape")
    ap.add_argument("--start", metavar="YYYY-MM",
                    help="first month, inclusive")
    ap.add_argument("--end", metavar="YYYY-MM",
                    help="last month, inclusive; defaults to --start")
    ap.add_argument("--days", type=int,
                    help="alternative to --start/--end: a window ending today")
    ap.add_argument("--categories", nargs="*",
                    help="default: sources.arxiv.categories in config")
    ap.add_argument("--config", default=str(paths.CONFIG))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ax = cfg.get("sources", {}).get("arxiv", {})
    if not ax.get("enabled", False):
        print("[arXiv] disabled in config (sources.arxiv.enabled)", file=sys.stderr)
        return 2

    if args.start:
        since, until = _month_range(args.start, args.end)
    else:
        days = args.days or (cfg.get("modes", {}).get("arxiv") or {}).get("days", 183)
        until = as_of()
        since = until - timedelta(days=days)

    categories = args.categories or ax.get("categories", ["cs.AI", "cs.LG", "stat.ML"])
    print(f"[arXiv] {since}..{until} · {', '.join(categories)} → {paths.ARXIV_DB.name}")

    db.use(paths.ARXIV_DB)
    db.init_db()
    items = scrape_arxiv(categories, since=since, until=until,
                         limit_per_slice=ax.get("limit_per_slice"))
    saved = _flush(items)
    print(f"\n[arXiv] {len(items)} fetched, {saved} written | "
          f"{db.article_count()} papers in {paths.ARXIV_DB.name}")
    return 0


def arxiv_embed(argv: list[str] | None = None) -> int:
    """Embed the paper corpus — the same step as `embed`, other database.

    Kept separate rather than given a --db flag so that the expensive default
    can never be aimed at the wrong corpus by a stray argument.
    """
    from . import db

    ap = argparse.ArgumentParser(prog="mindspace arxiv embed")
    ap.add_argument("--config", default=str(paths.CONFIG))
    args = ap.parse_args(argv)

    db.use(paths.ARXIV_DB)
    db.init_db()
    print(f"[arXiv] embedding into {paths.ARXIV_DB.name}")
    run_embedding(load_config(args.config))
    return 0


def run_embedding(cfg: dict):
    """Embed everything not yet embedded, saving as it goes.

    Saving per chunk rather than once at the end is what makes a failed run
    cheap. This previously embedded the whole backlog before writing a single
    row, so a rate limit or a dropped connection on the last request threw away
    every request before it. With incremental saves, re-running simply picks up
    from wherever it stopped — `get_articles_without_embeddings` already
    returns only what is missing.
    """
    from . import db
    from .embed import embed_articles

    pending = db.get_articles_without_embeddings()
    if not pending:
        print("[embed] all articles already embedded")
        return

    emb_cfg = cfg.get("embedding", {})
    model_name = emb_cfg.get("model", "text-embedding-3-small")
    batch_size = emb_cfg.get("batch_size", 100)
    chunk_size = max(batch_size * 5, batch_size)

    print(f"[embed] {len(pending)} articles to embed, saving every {chunk_size}")
    saved = 0
    try:
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]
            vectors = embed_articles(chunk, model_name=model_name,
                                     batch_size=batch_size)
            for article, vec in zip(chunk, vectors):
                db.save_embedding(article["id"], vec)
            saved += len(chunk)
            print(f"[embed] {saved}/{len(pending)} saved")
    except Exception as exc:
        print(f"[embed] stopped after {saved}/{len(pending)}: {exc}", file=sys.stderr)
        print("[embed] re-run to continue — what was saved is kept", file=sys.stderr)
        raise

    print(f"[embed] saved {saved} embeddings")


def run_cluster_and_output(cfg: dict, lookback_days: int):
    """Spec steps 1-4.

    1. cluster the past three months
    2. draw that background in greyscale
    3. cluster just this week, separately
    4. score this week's clusters, and write the name gaps

    ONE PROJECTION, FIT ONCE, SHARED BY BOTH CLUSTERINGS. UMAP is not stable
    under changes to its input — swapping out even a small fraction of the
    documents can rotate or reflect the whole embedding. Fitting it per window
    would therefore move every point each time the window slid, which is the
    "identifying continuity between basins of attraction" problem this design
    is meant to solve. Fit once and a basin keeps its coordinates forever;
    sliding the window only changes which points are drawn.
    """
    from . import db
    from .cluster import build_clusters, cluster_hdbscan, project_anchored
    from .coherence import corpus_tfidf, score_clusters, write_name_gaps
    from .sources import text_for_tfidf

    coh_cfg = cfg.get("coherence", {})
    q = (cfg.get("modes") or {}).get("quarter") or {}
    background_days = q.get("background_days", 91)
    week_days = q.get("week_days", 7)

    # The projection is fitted over the WHOLE corpus, not the three-month
    # window. Measured: refitting on identical input moves nothing, but drop 8%
    # of the documents — what one week of slide does — and points shift
    # (residual 0.17 after optimal rotation, distance correlation 0.96). A
    # per-window fit would therefore rearrange the map every time the slider
    # moved. Fitting once means a basin keeps its coordinates forever and any
    # window is just a subset of them.
    corpus = db.get_all_embedded()
    if not corpus:
        print("[cluster] no embedded articles — run scrape + embed first")
        return

    # Half-open [start, today) — the SAME window run_slider uses for its newest
    # stop. These two used to disagree: this one had no upper bound, so it
    # included documents published on the run day while the slider's window
    # excluded them. One run then wrote clusters.json with 72 clusters and
    # frames.json with 58, ids unaligned, and every Fable label was addressed
    # to groupings the display never showed.
    bg_cutoff = _cutoff(background_days)
    week_cutoff = _cutoff(week_days)
    today = as_of().isoformat()
    bg_idx = [i for i, a in enumerate(corpus)
              if bg_cutoff <= (a.get("published_at") or "") < today]
    week_idx = [i for i, a in enumerate(corpus)
                if week_cutoff <= (a.get("published_at") or "") < today]

    print(f"[cluster] corpus:     {len(corpus)} articles (projection fitted over all)")
    print(f"[cluster] background: {len(bg_idx)} in the last {background_days} days")
    print(f"[cluster] this week:  {len(week_idx)} in the last {week_days} days")
    if len(week_idx) < 2:
        print("[cluster] too few articles this week to cluster — stopping", file=sys.stderr)
        return

    embeddings = np.stack([a["embedding"] for a in corpus])
    cl_cfg = cfg.get("clustering", {})
    out_dir = Path(cfg.get("output", {}).get("dir", paths.QUARTER))
    out_dir.mkdir(parents=True, exist_ok=True)

    # The map is learned once and then kept, so a document's coordinates never
    # move again. The previous cache reused the projection only when the corpus
    # was byte-for-byte unchanged, which on a weekly run is never: one new week
    # refitted everything, and the weekly clusterings — and the cluster ids the
    # names are addressed to — came out different for weeks whose documents had
    # not changed at all.
    corpus_ids = [a["id"] for a in corpus]
    coords = project_anchored(
        embeddings, corpus_ids, out_dir / "umap_anchor.joblib",
        n_neighbors=cl_cfg.get("n_neighbors", 15),
        n_components=cl_cfg.get("n_components", 2),
        min_dist=cl_cfg.get("min_dist", 0.05),
        metric=cl_cfg.get("metric", "cosine"),
        log=lambda m: print(f"[cluster] {m}"))
    background = corpus

    def _cluster(where, what, min_cluster_size):
        labels = cluster_hdbscan(
            coords[where],
            min_cluster_size=min_cluster_size,
            min_samples=cl_cfg.get("min_samples", 2),
        )
        sizes = [int((labels == c).sum()) for c in set(labels) - {-1}]
        noise = int((labels == -1).sum())
        pct = 100 * noise / max(len(labels), 1)
        median = int(np.median(sizes)) if sizes else 0
        print(f"[cluster] {what}: {len(sizes)} clusters (median {median}, "
              f"largest {max(sizes) if sizes else 0}), {noise} noise ({pct:.0f}%)")
        return labels

    # Step 1 and step 3 — same coordinates, different subsets, and deliberately
    # different min_cluster_size: the background is twelve times the week, so
    # one value cannot mean the same thing for both.
    bg_labels = _cluster(np.array(bg_idx), "background (step 1)",
                         q.get("min_cluster_size", 15))
    week_labels = _cluster(np.array(week_idx), "this week (step 3)",
                           q.get("min_cluster_size_week", 8))

    # Widen the background labels back to corpus length so they line up with
    # `coords`. -2 marks a document outside the three-month window — distinct
    # from -1, which HDBSCAN uses for noise inside it. Collapsing the two would
    # make "we did not look at this" indistinguishable from "we looked and it
    # belonged nowhere".
    OUTSIDE = -2
    bg_labels_full = np.full(len(corpus), OUTSIDE, dtype=int)
    bg_labels_full[np.array(bg_idx)] = bg_labels

    # ── Step 4 ───────────────────────────────────────────────────────────────
    # TF-IDF is fitted over the WHOLE background, not just this week's subset:
    # "corpus-wide" in the spec, and the only way the numbers mean the same
    # thing from one cluster to the next.
    texts = [text_for_tfidf(a.get("title", ""), a.get("content", ""))
             for a in background]
    _, tfidf = corpus_tfidf(texts)

    week_articles = [background[i] for i in week_idx]
    records = score_clusters(
        labels=week_labels,
        embeddings=embeddings[week_idx],
        tfidf_matrix=tfidf[week_idx],
        measure_map=coh_cfg.get("measure_map", "corrected"),
    )
    write_name_gaps(records, out_dir / "name_gaps.json")

    week_clusters = build_clusters(week_articles, week_labels)
    label_by_id = {c["cluster_id"]: c["label"] for c in week_clusters}
    scored = [r for r in records if r["name_gap"] is not None]

    print("\n── This week's clusters, by name gap ──────────────────")
    print("   gap is rank(semantic) − rank(lexical): +1 means the most")
    print("   semantically tight and least lexically settled of the week\n")
    for r in scored[:8]:
        print(f"  [{r['size']:3d}] gap {r['name_gap']:+.2f}  "
              f"(lex rank {r['lexical_rank']:.2f} / sem rank {r['semantic_rank']:.2f})  "
              f"{label_by_id.get(r['cluster_id'], '')[:48]}")
    if len(scored) > 10:
        print("   …")
        for r in scored[-2:]:
            print(f"  [{r['size']:3d}] gap {r['name_gap']:+.2f}  "
                  f"(lex rank {r['lexical_rank']:.2f} / sem rank {r['semantic_rank']:.2f})  "
                  f"{label_by_id.get(r['cluster_id'], '')[:48]}")
    print("────────────────────────────────────────────────────────\n")

    # ── Outputs ──────────────────────────────────────────────────────────────
    # clusters.json holds THIS WEEK's clusters, because that is what
    # compress_clusters.py pairs with name_gaps.json.
    (out_dir / "clusters.json").write_text(
        json.dumps(week_clusters, indent=2, default=str))
    print(f"[output] saved → {out_dir / 'clusters.json'}")

    rows = [{"cluster_id": c["cluster_id"], "cluster_label": c["label"],
             "cluster_size": c["size"], **m}
            for c in week_clusters for m in c["members"]]
    pd.DataFrame(rows).to_csv(out_dir / "clusters.csv", index=False)
    print(f"[output] saved → {out_dir / 'clusters.csv'}")

    # coords.npz is what compress_clusters.py reloads to regenerate its viz, so
    # it has to line up with clusters.json — this week's slice, not the whole
    # background.
    np.savez(out_dir / "coords.npz",
             coords=coords[week_idx],
             labels=week_labels,
             article_ids=np.array([a["id"] for a in week_articles]))
    print(f"[output] saved → {out_dir / 'coords.npz'}")

    # The full projection, for anything that needs the background's geometry.
    np.savez(out_dir / "projection.npz",
             coords=coords,
             labels=bg_labels_full,
             published=np.array([(a.get("published_at") or "")[:10] for a in corpus]),
             article_ids=np.array([a["id"] for a in corpus]))
    print(f"[output] saved → {out_dir / 'projection.npz'}")

    # ── Step 2 ───────────────────────────────────────────────────────────────
    from .viz import build_background_viz
    build_background_viz(corpus, coords, bg_labels_full, set(week_idx), out_dir)

    # ── Steps 5 and 6 ────────────────────────────────────────────────────────
    run_slider(cfg, corpus, coords, tfidf, embeddings, out_dir)


def run_viz_only(cfg: dict) -> None:
    """Rebuild the HTML from saved results — no clustering, no API calls.

    Exists because re-running the pipeline is not free in a way that is easy to
    miss. The windows are measured from `now`, so an hour later the seven-day
    window holds a different set of documents; the clustering changes, the
    cluster ids change, and `name_gap_ai_labels.json` — which cost real money to
    produce — ends up attached to clusters that no longer contain the documents
    it was written about. Nothing errors. The map just says the wrong thing.

    So a change to the drawing code should redraw, not recompute.
    """
    from . import db
    from .viz import build_background_viz, build_slider_viz

    out_dir = Path(cfg.get("output", {}).get("dir", paths.QUARTER))
    proj_path, frames_path = out_dir / "projection.npz", out_dir / "frames.json"
    for p in (proj_path, frames_path):
        if not p.exists():
            print(f"[viz] {p} not found — run the full pipeline once first",
                  file=sys.stderr)
            return

    proj = np.load(proj_path, allow_pickle=True)
    coords, labels = proj["coords"], proj["labels"]
    saved_ids = [str(x) for x in proj["article_ids"]]

    corpus = db.get_all_embedded()
    if [a["id"] for a in corpus] != saved_ids:
        # The saved coordinates are positional. If the corpus has changed, row i
        # is no longer the same document, and every point would be mislabelled
        # without anything failing.
        print(f"[viz] the corpus has changed since projection.npz was written "
              f"({len(saved_ids)} saved, {len(corpus)} now) — "
              "re-run the full pipeline", file=sys.stderr)
        return

    frames = json.loads(frames_path.read_text())
    week_idx = set(frames[-1]["week_idx"]) if frames else set()

    print(f"[viz] rebuilding from saved results — {len(corpus)} documents, "
          f"{len(frames)} weekly stops")
    build_background_viz(corpus, coords, labels, week_idx, out_dir)
    build_slider_viz(frames, coords, corpus, out_dir)


def run_slider(cfg: dict, corpus: list[dict], coords, tfidf, embeddings,
               out_dir: Path) -> None:
    """Steps 5 and 6 — colour this week's clusters, one frame per week.

    Every frame reuses `coords`, which was fitted once over the whole corpus.
    Only membership and colour change between stops, so a basin sits in the
    same place at every stop and continuity is visible rather than inferred.

    Stops are limited to weeks whose three-month background is actually
    populated. Scraping only reaches back so far, and a stop whose background
    holds a few hundred documents would look like the field emptied out rather
    than like the collection did.
    """
    from .cluster import build_clusters, cluster_hdbscan
    from .coherence import score_clusters
    from .viz import build_slider_viz, coherence_color

    coh_cfg = cfg.get("coherence", {})
    cl_cfg = cfg.get("clustering", {})
    q = (cfg.get("modes") or {}).get("quarter") or {}
    background_days = q.get("background_days", 91)
    week_days = q.get("week_days", 7)
    min_bg = coh_cfg.get("slider_min_background", 1000)
    min_week = coh_cfg.get("slider_min_week", 50)

    published = np.array([(a.get("published_at") or "")[:10] for a in corpus])
    newest = as_of()

    frames, skipped = [], []
    for i in range(coh_cfg.get("slider_max_weeks", 26)):
        stop = newest - timedelta(days=7 * i)
        bg_lo = (stop - timedelta(days=background_days)).isoformat()
        wk_lo = (stop - timedelta(days=week_days)).isoformat()
        hi = stop.isoformat()

        bg_idx = np.flatnonzero((published >= bg_lo) & (published < hi))
        wk_idx = np.flatnonzero((published >= wk_lo) & (published < hi))
        if len(bg_idx) < min_bg or len(wk_idx) < min_week:
            skipped.append((stop, len(bg_idx), len(wk_idx)))
            continue

        wk_labels = cluster_hdbscan(
            coords[wk_idx],
            min_cluster_size=q.get("min_cluster_size_week", 8),
            min_samples=cl_cfg.get("min_samples", 2),
        )
        records = score_clusters(wk_labels, embeddings[wk_idx], tfidf[wk_idx],
                                 measure_map=coh_cfg.get("measure_map", "corrected"))
        by_id = {r["cluster_id"]: r for r in records}

        colors, bands = [], set()
        colour_of = {}
        for lbl in wk_labels:
            rec = by_id.get(int(lbl))
            if rec is None:
                # HDBSCAN noise: in the week's window but in no cluster, so
                # there is nothing to score. Drawn faint rather than dropped —
                # it is still a document that was published that week.
                colors.append("rgba(110,110,118,0.35)")
                continue
            color, band = coherence_color(rec["lexical_rank"], rec["semantic_rank"])
            colors.append(color)
            bands.add(band)
            colour_of[int(lbl)] = (color, band)

        # TF-IDF keyword labels, which exist for every week at no cost. The
        # Fable attractors only exist for whichever week compress_clusters last
        # ran on, so they are attached separately, by the viz.
        wk_articles = [corpus[i] for i in wk_idx]
        kw_of = {c["cluster_id"]: c["label"]
                 for c in build_clusters(wk_articles, wk_labels)}

        clusters_meta = []
        for r in records:
            cid = r["cluster_id"]
            color, band = colour_of.get(cid, ("rgba(110,110,118,0.35)", "uncoloured"))
            clusters_meta.append({
                "cluster_id": cid,
                "size": r["size"],
                "name_gap": r["name_gap"],
                "lexical_rank": r["lexical_rank"],
                "semantic_rank": r["semantic_rank"],
                "keywords": kw_of.get(cid, ""),
                "color": color,
                "band": band,
                # Positions within this frame's week_idx, so the viz can
                # highlight a cluster without recomputing anything.
                "points": np.flatnonzero(wk_labels == cid).tolist(),
            })

        frames.append({
            "week_end": hi,
            "background_idx": bg_idx.tolist(),
            "week_idx": wk_idx.tolist(),
            "week_colors": colors,
            "bands_present": sorted(bands),
            "n_clusters": len(records),
            "clusters": clusters_meta,
        })

    if not frames:
        print("[slider] no week has enough data — skipping", file=sys.stderr)
        return

    frames.reverse()   # oldest first, so the slider runs left to right
    print(f"\n[slider] {len(frames)} weekly stops, "
          f"{frames[0]['week_end']} .. {frames[-1]['week_end']}")
    if skipped:
        thin = [s for s in skipped if s[0].isoformat() > frames[0]["week_end"]]
        print(f"[slider] {len(skipped)} week(s) excluded for thin data — "
              f"the corpus does not reach further back"
              + (f" ({len(thin)} of them inside the covered range)" if thin else ""))

    # Everything the visualisation needs, including background_idx. Written in
    # full so `--viz-only` can rebuild the HTML without re-clustering: the week
    # windows are measured from `now`, so a second run hours later covers a
    # slightly different set of documents, gives different clusters, and
    # silently invalidates the Fable attractors keyed to the old cluster ids.
    (out_dir / "frames.json").write_text(json.dumps(frames))
    build_slider_viz(frames, coords, corpus, out_dir)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="mindspace AI topic pipeline")
    parser.add_argument("--days", type=int, default=None,
                        help="lookback window in days (overrides config)")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="skip scraping, use existing DB")
    parser.add_argument("--skip-embed", action="store_true",
                        help="skip embedding step")
    parser.add_argument("--scrape-only", action="store_true",
                        help="scrape and save to DB, then stop (no embed/cluster)")
    parser.add_argument("--embed-only", action="store_true",
                        help="embed articles missing vectors, then stop "
                             "(no scraping, no clustering)")
    parser.add_argument("--viz-only", action="store_true",
                        help="redraw the HTML from saved results; no clustering, "
                             "so cluster ids stay valid and the Fable attractors "
                             "keyed to them do not need regenerating")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None,
                        metavar="YYYY-MM-DD",
                        help="measure the windows back from this date instead of "
                             "today, to reproduce an earlier run exactly")
    parser.add_argument("--week", type=int, default=None, metavar="N",
                        help="which window counting back from today: 1 = the "
                             "most recent --days days ending today, 2 = the "
                             "window before that, and so on — same counting as "
                             "probe_grok.py; shorthand that computes --as-of")
    parser.add_argument("--config", default=str(paths.CONFIG),
                        help="path to config file")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    lookback_days = args.days or cfg.get("lookback_days", 7)

    global _AS_OF
    if args.week is not None and args.as_of:
        parser.error("--week and --as-of both set the window anchor — use one")
    if args.week is not None:
        if args.week < 1:
            parser.error("--week counts from 1 (the most recent window)")
        _AS_OF = (datetime.now(timezone.utc).date()
                  - timedelta(days=lookback_days * (args.week - 1)))
        print(f"[pipeline] week {args.week}: "
              f"{_AS_OF - timedelta(days=lookback_days)}..{_AS_OF}")
    else:
        _AS_OF = args.as_of
        if _AS_OF:
            print(f"[pipeline] windows measured back from {_AS_OF}")

    if args.viz_only:
        run_viz_only(cfg)
        return

    from . import db
    db.init_db()

    if args.embed_only:
        if args.scrape_only:
            parser.error("--embed-only and --scrape-only exclude each other")
        print("\n── Embedding ──────────────────────────────────────────")
        run_embedding(cfg)
        print("\n[done] embed-only mode — skipping scrape and cluster")
        return

    if not args.skip_scrape:
        print(f"\n── Scraping (lookback: {lookback_days}d) ──────────────────")
        new_count, failed = run_scrapers(cfg, lookback_days)
        print(f"\n[db] {new_count} new articles saved | total: {db.article_count()}")
        if failed:
            print(f"[db] INCOMPLETE — no data from: {', '.join(failed)}. "
                  f"This window is missing them permanently unless the scrape "
                  f"is repeated before clustering.", file=sys.stderr)

    if args.scrape_only:
        print("\n[done] scrape-only mode — skipping embed and cluster")
        return

    if not args.skip_embed:
        print("\n── Embedding ──────────────────────────────────────────")
        run_embedding(cfg)

    print("\n── Clustering ─────────────────────────────────────────")
    run_cluster_and_output(cfg, lookback_days)


if __name__ == "__main__":
    main()
