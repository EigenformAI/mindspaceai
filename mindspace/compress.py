#!/usr/bin/env python3
"""
Feed each cluster to multiple LLMs and compare their one-sentence theme compressions.

Usage:
    python compress_clusters.py [--top N] [--clusters output/clusters.json]

Everything routes through OpenRouter on OPENROUTER_API_KEY:

    compress   openai/gpt-4o-mini, x-ai/grok-4.5,
               anthropic/claude-haiku-4.5, google/gemini-2.5-flash
    synthesise anthropic/claude-fable-5
    label      anthropic/claude-haiku-4.5

Previously each compressor came from its own provider and only appeared if that
provider's key happened to be set, so the panel quietly shrank to whatever was
configured — while the synthesis prompt kept saying "Four different AI models".
"""
from . import paths
import argparse
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_COMPRESS_PROMPT = """\
Below is a cluster of AI-related content grouped by semantic similarity.

Keywords (TF-IDF): {keywords}

Titles and excerpts:
{samples}

In ONE short paragraph, name the single most specific and intellectually \
interesting conceptual theme unifying this cluster. Compress it into a single metaconcept if possible. Avoid generic phrases like \
"explores AI" or "discusses models". Capture what is distinctive within the AI field as it currently stands."""


_NAME_GAP_PROMPT = """\
Below is a cluster of AI-related content that is semantically tight but lexically scattered: \
the documents appear to describe the same underlying thing in different vocabularies, with no \
shared term for it yet.

Keywords (TF-IDF): {keywords}

Titles and excerpts:
{samples}

In ONE short paragraph, state plainly  the shared conceptual theme(s) that unite these documents. Be descriptive, not poetic. No metaphors, no coinages — just \
say what the shared referent is. Compress it into a single metaconcept if possible. Avoid generic phrases like \
"explores AI" or "discusses models". Capture what is distinctive within the AI field as it currently stands."""


def _get_content_map(db_path: str, urls: list[str]) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(urls))
    rows = conn.execute(
        f"SELECT url, title, content FROM articles WHERE url IN ({placeholders})", urls
    ).fetchall()
    conn.close()
    return {r[0]: (r[1] or "", r[2] or "") for r in rows}


def _build_samples(members: list[dict], content_map: dict, n: int = 8) -> str:
    lines = []
    for m in members[:n]:
        url = m.get("url", "")
        title, content = content_map.get(url, (m.get("title", ""), ""))
        snippet = content[:250].replace("\n", " ").strip()
        lines.append(f"- {title}\n  {snippet}")
    return "\n".join(lines)


_SPEND: list[dict] = []


def _call_openrouter(prompt: str, model: str, api_key: str, max_tokens: int = 80) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
        extra_body={"usage": {"include": True}},
    )
    u = getattr(resp, "usage", None)
    _SPEND.append({
        "model": model,
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "cost_usd": getattr(u, "cost", None),
    })
    return (resp.choices[0].message.content or "").strip()


def _write_spend(out_dir: Path, run_label: str) -> None:
    """Write what this run actually cost, per model and in total."""
    if not _SPEND:
        return
    priced = [c for c in _SPEND if c["cost_usd"] is not None]
    unpriced = len(_SPEND) - len(priced)

    by_model: dict[str, dict] = {}
    for c in _SPEND:
        m = by_model.setdefault(c["model"], {"calls": 0, "cost_usd": 0.0,
                                             "prompt_tokens": 0, "completion_tokens": 0})
        m["calls"] += 1
        m["cost_usd"] += c["cost_usd"] or 0.0
        m["prompt_tokens"] += c["prompt_tokens"] or 0
        m["completion_tokens"] += c["completion_tokens"] or 0

    total = sum(c["cost_usd"] or 0.0 for c in _SPEND)
    this_run = {
        "run": run_label,
        "total_usd": total,
        "total_calls": len(_SPEND),
        "calls_without_a_reported_cost": unpriced,
        "by_model": by_model,
        "calls": _SPEND,
    }

    path = out_dir / "cost.json"
    runs = []
    if path.exists():
        try:
            runs = json.loads(path.read_text()).get("runs", [])
        except (json.JSONDecodeError, AttributeError):
            runs = []
    runs.append(this_run)
    path.write_text(json.dumps({
        "total_usd": sum(r["total_usd"] for r in runs),
        "total_calls": sum(r["total_calls"] for r in runs),
        "runs": runs,
    }, indent=2))

    def usd(x: float) -> str:
        return f"${x:.6f}" if x < 0.01 else f"${x:.4f}"

    print("\n── Cost ───────────────────────────────────────────────")
    for m, v in sorted(by_model.items(), key=lambda kv: -kv[1]["cost_usd"]):
        print(f"  {m:<34} {v['calls']:>3} calls  {usd(v['cost_usd']):>11}"
              f"  ({v['prompt_tokens']:,} in / {v['completion_tokens']:,} out)")
    print(f"  {'TOTAL':<34} {len(_SPEND):>3} calls  {usd(total):>11}")
    if unpriced:
        print(f"  ({unpriced} call(s) reported no cost — excluded, not counted as free)")
    print(f"  saved → {path}")
    print("────────────────────────────────────────────────────────")


_LABEL_PROMPT = (
    "Distil this cluster description into a compact label of 2-5 words. "
    "Noun phrase only, no verbs, no punctuation.\n"
    # Both additions are responses to what the model actually did: it returned
    # several candidates separated by "or", in markdown, and max_tokens cut the
    # list mid-word. Four of nine labels came out as things like
    # '**Black box certification metrology**\n\nor alternatively:\n\n**'.
    "Give exactly ONE label. Do not offer alternatives. "
    "Plain text only — no markdown, no asterisks, no headings.\n\n"
    "Description: {text}"
)


def _clean_label(raw: str, limit: int = 60) -> str:
    """First candidate only, stripped of markdown.

    Belt as well as braces: the prompt now asks for one plain-text label, but a
    label is user-visible in the legend and a stray '**' there is worse than a
    few lines of defensive parsing here.
    """
    text = (raw or "").strip()
    # Keep only what precedes an offer of alternatives.
    for sep in ("\n\nor alternatively", "\n\nor\n", "\nor alternatively", "\nOr:"):
        idx = text.lower().find(sep.lower())
        if idx != -1:
            text = text[:idx]
    text = text.split("\n")[0]
    text = text.replace("*", "").replace("#", "").replace("`", "")
    text = text.strip().strip('"').strip("'").strip()
    return text[:limit]


# Model registry 
_COMPRESSORS = [
    ("gpt-4o-mini",       "openai/gpt-4o-mini"),
    ("grok-4.5",          "x-ai/grok-4.5"),
    ("claude-haiku-4.5",  "anthropic/claude-haiku-4.5"),
    ("gemini-2.5-flash",  "google/gemini-2.5-flash"),
]

_LABEL_MODEL = "anthropic/claude-haiku-4.5"


def _available_models() -> list[dict]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return []
    return [
        {"name": name,
         "fn": (lambda p, m=model, k=key: _call_openrouter(p, m, k, max_tokens=300))}
        for name, model in _COMPRESSORS
    ]


_SYNTH_PROMPT = """\
Four different AI models each independently compressed the same cluster of content into a single sentence.
Their descriptions are:

{compressions}

These sentences are all orbiting the same conceptual attractor — a deeper, more essential idea \
that all of them gesture toward but none has quite named directly.

Name the attractor. Not a description of the topic, but the underlying principle, tension, or \
structure that makes this cluster cohere as a thing. One compressed phrase or sentence — \
as precise, specific, and conceptually loaded as possible. \
Avoid generic AI framing. Think like a physicist naming a phenomenon."""


_NAME_GAP_SYNTH_PROMPT = """\
Several AI models each independently described the same cluster of documents. The cluster is \
semantically coherent but lexically scattered — evidence of a concept that exists in the \
discourse without a settled name. Their descriptions:

{compressions}

Name the attractor. Not a description of the topic, but the underlying principle, tension, or \
structure that makes this cluster cohere as a thing. One compressed phrase or sentence — \
as precise, specific, and conceptually loaded as possible. \
Avoid generic AI framing. Think like a physicist naming a phenomenon."""


def _synthesise(compressions: dict[str, str], synth_fn,
                prompt_template: str = _SYNTH_PROMPT) -> str:
    comp_text = "\n".join(f"  {k}: {v}" for k, v in compressions.items() if not v.startswith("[ERROR"))
    if not comp_text.strip():
        return "[no valid compressions to synthesise]"
    prompt = prompt_template.format(compressions=comp_text)
    try:
        return synth_fn(prompt)
    except Exception as exc:
        return f"[ERROR: {exc}]"


# ── Main ─────────────────────────────────────────────────────────────────────

def compress(clusters_path: str, top_n: int, db_path: str,
             synth_model: str | None = None, rank_by: str = "size",
             gaps_path: str | None = None) -> None:
    clusters = json.loads(Path(clusters_path).read_text())
    gap_by_id: dict[int, float] = {}
    if rank_by == "name-gap":
        gaps = json.loads(Path(gaps_path or paths.QUARTER / "name_gaps.json").read_text())
        gap_by_id = {g["cluster_id"]: g["name_gap"] for g in gaps}
        ranked_ids = [g["cluster_id"] for g in gaps if g["name_gap"] > 0][:top_n]
        by_id = {c["cluster_id"]: c for c in clusters}
        real_clusters = [by_id[cid] for cid in ranked_ids if cid in by_id]
    else:
        real_clusters = [c for c in clusters if c["cluster_id"] != -1][:top_n]

    compress_prompt = _NAME_GAP_PROMPT if rank_by == "name-gap" else _COMPRESS_PROMPT
    synth_prompt = _NAME_GAP_SYNTH_PROMPT if rank_by == "name-gap" else _SYNTH_PROMPT
    prefix = "name_gap_" if rank_by == "name-gap" else ""

    models = _available_models()
    if not models:
        print("OPENROUTER_API_KEY is not set — every model here routes through it.",
              file=sys.stderr)
        sys.exit(1)

    # Synthesis model (second-pass attractor naming)
    synth_fn = None
    if synth_model:
        or_key = os.environ.get("OPENROUTER_API_KEY", "")

        def synth_fn(prompt: str, _k=or_key, _m=synth_model) -> str:
            return _call_openrouter(prompt, _m, _k, max_tokens=1024)

        print(f"Synth model: {synth_model}")

    print(f"Models: {', '.join(m['name'] for m in models)}")
    print(f"Clusters: {len(real_clusters)}  (top {top_n} by {rank_by}, noise excluded)\n")

    # Pre-fetch all article content in one DB round-trip
    all_urls = [m["url"] for c in real_clusters for m in c["members"]]
    content_map = _get_content_map(db_path, all_urls)

    results: list[dict] = []

    for c in real_clusters:
        samples_text = _build_samples(c["members"], content_map)
        prompt = compress_prompt.format(
            keywords=c["label"],
            samples=samples_text,
        )

        # Fan out to all models in parallel
        compressions: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {pool.submit(m["fn"], prompt): m["name"] for m in models}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    compressions[name] = future.result()
                except Exception as exc:
                    compressions[name] = f"[ERROR: {exc}]"

        attractor = None
        if synth_fn:
            attractor = _synthesise(compressions, synth_fn, synth_prompt)

        results.append({
            "cluster_id": c["cluster_id"],
            "size": c["size"],
            "keywords": c["label"],
            "name_gap": gap_by_id.get(c["cluster_id"]),
            "compressions": compressions,
            "attractor": attractor,
        })

        # Print as we go
        print(f"── Cluster {c['cluster_id']} (n={c['size']}) ─── {c['label']}")
        for model_name in [m["name"] for m in models]:
            print(f"  {model_name:<22} {compressions.get(model_name, '')}")
        if attractor:
            print(f"  {'★ attractor':<22} {attractor}")
        print()

    # ── Label generation ─────────────────────────────────────────────────────
    # Ask Haiku to distill each attractor (or best compression) to 2-5 words
    label_fn = None
    if synth_fn:
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if or_key:
            label_fn = lambda p, k=or_key: _call_openrouter(p, _LABEL_MODEL, k, max_tokens=40)

    ai_labels: dict[int, str] = {}
    if label_fn:
        print("\nGenerating short cluster labels...")
        for r in results:
            src = r.get("attractor") or next(iter(r["compressions"].values()), "")
            if not src or src.startswith("[ERROR"):
                continue
            try:
                raw = label_fn(_LABEL_PROMPT.format(text=src))
                label = _clean_label(raw)
                ai_labels[r["cluster_id"]] = label
                print(f"  [{r['cluster_id']:3d}] {label}")
            except Exception as exc:
                print(f"  [{r['cluster_id']:3d}] [label error: {exc}]")

        if ai_labels:
            labels_path = paths.QUARTER / f"{prefix}ai_labels.json"
            labels_path.write_text(json.dumps(ai_labels, indent=2))
            print(f"Saved → {labels_path}")

            # Regenerate viz with AI labels if coords are available
            coords_path = paths.QUARTER / "coords.npz"
            if coords_path.exists():
                attractors = {r["cluster_id"]: r["attractor"]
                              for r in results if r.get("attractor")}
                _regen_viz(db_path, coords_path, ai_labels, attractors)

    # Save raw results as JSON for --labels-only re-runs
    json_out = paths.QUARTER / f"{prefix}compressions.json"
    json_out.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved → {json_out}")

    _write_spend(paths.COST, f"{prefix or 'size'}compress top{top_n}")

    # Save markdown report
    out_path = paths.QUARTER / f"{prefix}compressions.md"
    title = "Name-Gap Cluster Descriptions" if prefix else "Cluster Theme Compressions"
    lines = [f"# {title}\n"]
    for r in results:
        ai_lbl = ai_labels.get(r["cluster_id"], "")
        heading = f"## Cluster {r['cluster_id']} (n={r['size']})"
        if r.get("name_gap") is not None:
            heading += f" [gap={r['name_gap']}]"
        if ai_lbl:
            heading += f" — {ai_lbl}"
        lines.append(heading)
        lines.append(f"**Keywords:** {r['keywords']}\n")
        for model_name, text in r["compressions"].items():
            lines.append(f"**{model_name}:** {text}\n")
        if r.get("attractor"):
            lines.append(f"**★ Attractor ({synth_model}):** {r['attractor']}\n")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Saved → {out_path}")


def _regen_viz(db_path: str, coords_path: Path, ai_labels: dict[int, str],
               attractors: dict[int, str] | None = None) -> None:
    import numpy as np
    import sqlite3
    from .viz import build_viz
    from .cluster import build_clusters

    data = np.load(coords_path)
    coords = data["coords"]
    labels = data["labels"]
    article_ids = data["article_ids"].tolist()

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, source, url, title, content, author, published_at "
        "FROM articles WHERE id IN ({})".format(",".join("?" * len(article_ids))),
        article_ids,
    ).fetchall()
    conn.close()
    id_to_row = {r[0]: r for r in rows}
    articles = [
        {"id": aid, "source": id_to_row[aid][1], "url": id_to_row[aid][2],
         "title": id_to_row[aid][3], "content": id_to_row[aid][4],
         "author": id_to_row[aid][5], "published_at": id_to_row[aid][6]}
        for aid in article_ids if aid in id_to_row
    ]

    clusters = build_clusters(articles, labels)
    build_viz(articles, coords, labels, clusters, "output",
              ai_labels=ai_labels, attractors=attractors)
    print("[viz] regenerated with AI labels → output/viz.html")


def label_only(db_path: str, compressions_json: str) -> None:
    """Read saved compressions.json and run just the label generation + viz step."""
    results = json.loads(Path(compressions_json).read_text())

    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not or_key:
        print("OPENROUTER_API_KEY is not set — nothing can label.", file=sys.stderr)
        sys.exit(1)

    def label_fn(prompt: str) -> str:
        return _call_openrouter(prompt, _LABEL_MODEL, or_key, max_tokens=40)


    ai_labels: dict[int, str] = {}
    print("Generating short cluster labels...")
    for r in results:
        src = r.get("attractor") or next(iter(r.get("compressions", {}).values()), "")
        if not src or src.startswith("[ERROR"):
            continue
        try:
            raw = label_fn(_LABEL_PROMPT.format(text=src))
            label = _clean_label(raw)
            ai_labels[r["cluster_id"]] = label
            print(f"  [{r['cluster_id']:3d}] {label}")
        except Exception as exc:
            print(f"  [{r['cluster_id']:3d}] [label error: {exc}]")

    if ai_labels:
        labels_path = paths.QUARTER / "ai_labels.json"
        labels_path.write_text(json.dumps(ai_labels, indent=2))
        print(f"Saved → {labels_path}")
        coords_path = paths.QUARTER / "coords.npz"
        attractors = {r["cluster_id"]: r["attractor"]
                      for r in results if r.get("attractor")}
        if coords_path.exists():
            _regen_viz(db_path, coords_path, ai_labels, attractors)
        else:
            print("[viz] output/coords.npz not found — run pipeline.py --skip-scrape --skip-embed to generate it, then re-run with --labels-only")

    _write_spend(paths.COST, "labels-only")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=15, help="Max clusters to process (default 15)")
    parser.add_argument("--clusters", default=str(paths.QUARTER / "clusters.json"), help="Path to clusters.json")
    parser.add_argument("--db", default=str(paths.DB), help="Path to SQLite DB")
    parser.add_argument("--synth", metavar="MODEL",
                        default="anthropic/claude-fable-5",
                        help="OpenRouter model for attractor synthesis (default: anthropic/claude-fable-5). Pass '' to disable.")
    parser.add_argument("--labels-only", action="store_true",
                        help="Skip compress/synth; read output/compressions.json and regenerate labels + viz only")
    parser.add_argument("--rank-by", choices=["size", "name-gap"], default="size",
                        help="Cluster selection: 'size' (default) or 'name-gap' (requires output/name_gaps.json from name_gap.py)")
    parser.add_argument("--gaps", default=str(paths.QUARTER / "name_gaps.json"),
                        help="Path to name_gaps.json (used with --rank-by name-gap)")
    args = parser.parse_args(argv)
    if args.labels_only:
        label_only(args.db, str(paths.QUARTER / "compressions.json"))
    else:
        compress(args.clusters, args.top, args.db,
                 synth_model=args.synth or None,
                 rank_by=args.rank_by, gaps_path=args.gaps)


if __name__ == "__main__":
    main()
