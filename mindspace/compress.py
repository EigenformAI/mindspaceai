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


_BUZZWORD_PROMPT = """\
Below is a cluster of AI-related content that is lexically tight but semantically scattered: \
the documents keep reaching for the same term, but they are not talking about the same thing.

Keywords (TF-IDF): {keywords}

Titles and excerpts:
{samples}

In ONE short paragraph, say what the shared term is doing here — which distinct things it is \
being made to stand for across these documents. Do NOT invent a unifying concept. If there is \
no single referent, say so plainly and list the senses you can actually see. Be descriptive, \
not poetic. A term stretched across unrelated meanings is the finding; naming a false unity \
would hide it."""


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

# Clusters named concurrently. Six keeps well inside OpenRouter's rate
# limits while turning a ~2-hour eleven-week run into ~20 minutes.
CLUSTER_WORKERS = 6


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


_BUZZWORD_SYNTH_PROMPT = """\
Several AI models each independently described the same cluster of documents. The cluster is \
lexically coherent but semantically scattered — one term doing more work than the idea behind \
it. Their descriptions:

{compressions}

Name what the term has become, not what it once meant. One compressed phrase or sentence — \
as precise and specific as possible. Do not resolve the senses into a single idea; the \
spread IS the finding. If the models disagree about which senses the term now carries, that \
disagreement is itself evidence, and worth stating."""


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


# Main

def _name_batch(real_clusters: list[dict], db_path: str, models: list[dict],
                synth_fn, compress_prompt: str, synth_prompt: str,
                gap_by_id: dict) -> list[dict]:
    """Compress → synthesise → collect, for one set of clusters.

    Split out so a single week and a whole slider's worth of weeks run through
    exactly the same naming path; a second copy of this loop would be a second
    place for the panel, the prompts or the error handling to drift.
    """
    all_urls = [m["url"] for c in real_clusters for m in c["members"]]
    content_map = _get_content_map(db_path, all_urls) if all_urls else {}

    def one(c: dict) -> dict:
        prompt = compress_prompt.format(
            keywords=c["label"],
            samples=_build_samples(c["members"], content_map),
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

        attractor = _synthesise(compressions, synth_fn, synth_prompt) if synth_fn else None
        return {
            "cluster_id": c["cluster_id"],
            "size": c["size"],
            "keywords": c["label"],
            "name_gap": c.get("name_gap", gap_by_id.get(c["cluster_id"])),
            "compressions": compressions,
            "attractor": attractor,
        }

    # Clusters run concurrently, not just the panel within one cluster. Each
    # cluster is three sequential round trips (compress → synthesise → label),
    # so a serial loop spends most of its life waiting: eleven weeks measured
    # at roughly a minute per cluster, nearly two hours of mostly idle time.
    results: list[dict] = [None] * len(real_clusters)
    with ThreadPoolExecutor(max_workers=CLUSTER_WORKERS) as pool:
        futures = {pool.submit(one, c): i for i, c in enumerate(real_clusters)}
        for future in as_completed(futures):
            i = futures[future]
            c = real_clusters[i]
            try:
                results[i] = future.result()
            except Exception as exc:
                # One cluster failing must not lose the rest of the week's work.
                results[i] = {"cluster_id": c["cluster_id"], "size": c["size"],
                              "keywords": c["label"], "name_gap": c.get("name_gap"),
                              "compressions": {}, "attractor": f"[ERROR: {exc}]"}
            r = results[i]
            print(f"── Cluster {r['cluster_id']} (n={r['size']}) ─── {r['keywords']}")
            for model_name in [m["name"] for m in models]:
                print(f"  {model_name:<22} {r['compressions'].get(model_name, '')}")
            if r["attractor"]:
                print(f"  {'★ attractor':<22} {r['attractor']}")
            print()
    return results


def frame_clusters(frame: dict, db_path: str, top_n: int,
                   buzzwords: int = 0, floor: float = -0.3) -> list[dict]:
    """One week's clusters worth naming, in the shape `compress` already expects.

    frames.json stores membership as positions, not ids: `cluster["points"]`
    indexes into `frame["week_idx"]`, which indexes the corpus row order saved
    beside the projection. Resolving that chain here rather than re-clustering
    means the naming pass and the weekly view describe exactly the same
    documents — there is no second clustering to drift from the first.

    Two bands are selected, not one. The top `top_n` by positive gap are the
    emerging concepts. The `buzzwords` most negative below `floor` are the
    opposite finding — one word carrying several meanings — and until they were
    included the display had no buzzwords at all to show, which read as "there
    are none" when the truth was that nothing had ever asked for them.

    The floor exists because gaps sum to zero by construction: every week has a
    most-negative cluster whether or not anything is actually stretched. At
    -0.3 each of the eleven measured weeks has at least one real candidate
    (2.6 on average); at -0.5 one week has none.
    """
    import numpy as np

    ids = [str(x) for x in np.load(paths.QUARTER / "projection.npz",
                                   allow_pickle=True)["article_ids"]]
    week_idx = frame["week_idx"]

    scored = [c for c in frame["clusters"] if c.get("name_gap") is not None]
    ranked = sorted((c for c in scored if c["name_gap"] > 0),
                    key=lambda c: -c["name_gap"])[:top_n]
    if buzzwords:
        ranked += sorted((c for c in scored if c["name_gap"] < floor),
                         key=lambda c: c["name_gap"])[:buzzwords]
    if not ranked:
        return []

    wanted = {ids[week_idx[p]] for c in ranked for p in c["points"]}
    conn = sqlite3.connect(db_path)
    url_of = dict(conn.execute(
        f"SELECT id, url FROM articles WHERE id IN ({','.join('?' * len(wanted))})",
        list(wanted)).fetchall())
    conn.close()

    out = []
    for c in ranked:
        members = [{"url": url_of[ids[week_idx[p]]]}
                   for p in c["points"] if ids[week_idx[p]] in url_of]
        out.append({"cluster_id": c["cluster_id"], "size": c["size"],
                    "label": c["keywords"], "members": members,
                    "name_gap": c["name_gap"]})
    return out


def load_week_names(quarter_dir, latest_week: str | None = None):
    """Fable labels and attractor texts, keyed by week.

    Prefers the week-keyed files written by `fable --all-weeks`. Falls back to
    the flat pair, which only ever described a single week — attaching those to
    the newest week is what the display code used to assume implicitly, so an
    older output tree keeps rendering exactly as before.

    Returns ({week: {cluster_id: label}}, {week: {cluster_id: attractor}}).
    """
    from pathlib import Path
    quarter_dir = Path(quarter_dir)
    labels: dict[str, dict[int, str]] = {}
    attractors: dict[str, dict[int, str]] = {}

    lp = quarter_dir / "name_gap_ai_labels_by_week.json"
    if lp.exists():
        labels = {w: {int(k): v for k, v in d.items()}
                  for w, d in json.loads(lp.read_text()).items()}
    cp_ = quarter_dir / "name_gap_compressions_by_week.json"
    if cp_.exists():
        for w, rows in json.loads(cp_.read_text()).items():
            attractors[w] = {int(r["cluster_id"]): r["attractor"]
                             for r in rows if r.get("attractor")}

    if not labels and latest_week:
        flat = quarter_dir / "name_gap_ai_labels.json"
        if flat.exists():
            labels = {latest_week: {int(k): v
                                    for k, v in json.loads(flat.read_text()).items()}}
        flat_c = quarter_dir / "name_gap_compressions.json"
        if flat_c.exists():
            attractors = {latest_week: {int(r["cluster_id"]): r["attractor"]
                                        for r in json.loads(flat_c.read_text())
                                        if r.get("attractor")}}
    return labels, attractors


def compress_frames(top_n: int, db_path: str, synth_model: str | None,
                    only_week: str | None = None, buzzwords: int = 3,
                    force: bool = False) -> None:
    """Name every week's high-gap clusters, not just the newest one.

    The walkthrough in the brief names a cluster for week one, then for week
    two, and so on. The clustering for all of them already exists in
    frames.json; what was missing was that `compress` only ever read
    name_gaps.json, which holds the current week alone.

    Output is keyed by week because cluster ids restart with every weekly
    clustering: cluster 3 in May and cluster 3 in August are unrelated, and a
    flat map would let one silently overwrite the other.
    """
    frames_path = paths.QUARTER / "frames.json"
    if not frames_path.exists():
        print(f"{frames_path} not found — run `quarter` first", file=sys.stderr)
        sys.exit(1)
    frames = json.loads(frames_path.read_text())
    if only_week:
        frames = [f for f in frames if f["week_end"] == only_week]
        if not frames:
            print(f"no frame for week {only_week}", file=sys.stderr)
            sys.exit(1)

    models = _available_models()
    if not models:
        print("OPENROUTER_API_KEY is not set — every model here routes through it.",
              file=sys.stderr)
        sys.exit(1)

    synth_fn = label_fn = None
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if synth_model:
        def synth_fn(prompt: str, _k=or_key, _m=synth_model) -> str:
            return _call_openrouter(prompt, _m, _k, max_tokens=1024)
        def label_fn(prompt: str, _k=or_key) -> str:
            return _call_openrouter(prompt, _LABEL_MODEL, _k, max_tokens=40)

    labels_path = paths.QUARTER / "name_gap_ai_labels_by_week.json"
    comps_path = paths.QUARTER / "name_gap_compressions_by_week.json"
    # Resume rather than restart: naming is paid, and a run over eleven weeks
    # is long enough that it will sometimes be interrupted.
    by_week_labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    by_week_comps = json.loads(comps_path.read_text()) if comps_path.exists() else {}

    print(f"{len(frames)} week(s) · panel: {', '.join(m['name'] for m in models)}"
          + (f" · synth: {synth_model}" if synth_model else ""))

    for frame in frames:
        week = frame["week_end"]
        # Resume per cluster, not per week. Skipping a week that has any name
        # at all is right only while the selection never changes; the moment it
        # widens — a new band, a larger --top — every already-touched week is
        # skipped whole and the new clusters are never named, silently and for
        # free, which reads exactly like "there were none to add".
        done = set(by_week_labels.get(week, {}))
        candidates = frame_clusters(frame, db_path, top_n, buzzwords)
        clusters = (candidates if force else
                    [c for c in candidates if str(c["cluster_id"]) not in done])
        if not clusters:
            print(f"\n══ {week} — nothing new to name "
                  f"({len(done)} already named) ══")
            continue

        # The two bands are asked opposite questions, so they cannot share a
        # prompt: the emerging prompt tells the panel a shared referent exists
        # and to find it. Handed a buzzword cluster it obliges, fluently, and
        # invents the unity that the negative gap is evidence against.
        emerging = [c for c in clusters if c["name_gap"] > 0]
        buzz = [c for c in clusters if c["name_gap"] <= 0]
        print(f"\n══ {week} — {len(emerging)} emerging"
              + (f", {len(buzz)} buzzword" if buzz else "") + " ══\n")

        results = []
        if emerging:
            results += _name_batch(emerging, db_path, models, synth_fn,
                                   _NAME_GAP_PROMPT, _NAME_GAP_SYNTH_PROMPT, {})
        if buzz:
            results += _name_batch(buzz, db_path, models, synth_fn,
                                   _BUZZWORD_PROMPT, _BUZZWORD_SYNTH_PROMPT, {})

        labels: dict[str, str] = {}
        if label_fn:
            def distil(r: dict):
                src = r.get("attractor") or next(iter(r["compressions"].values()), "")
                # "[no valid compressions to synthesise]" is what _synthesise
                # returns when every model failed. It does not start with
                # "[ERROR", so an earlier guard let it through to be distilled
                # into a real-looking sidebar label.
                if not src or src.startswith("[ERROR") or src.startswith("[no valid"):
                    return r["cluster_id"], None
                try:
                    return r["cluster_id"], _clean_label(
                        label_fn(_LABEL_PROMPT.format(text=src)))
                except Exception as exc:
                    print(f"  [{r['cluster_id']:3d}] [label error: {exc}]")
                    return r["cluster_id"], None

            with ThreadPoolExecutor(max_workers=CLUSTER_WORKERS) as pool:
                for cid, label in pool.map(distil, results):
                    if label:
                        labels[str(cid)] = label
                        print(f"  [{cid:3d}] {label}")

        if force:
            by_week_labels[week] = labels
            by_week_comps[week] = results
        else:
            # Merge, because this run only ever holds the clusters that were
            # still missing — assigning would drop everything paid for before.
            by_week_labels.setdefault(week, {}).update(labels)
            by_week_comps.setdefault(week, []).extend(results)
        # Written after every week, so an interrupted run keeps what it paid
        # for — the ledger included. Recording spend only at the end meant a
        # run stopped halfway kept its names but lost the record of the money
        # that produced them.
        labels_path.write_text(json.dumps(by_week_labels, indent=2))
        comps_path.write_text(json.dumps(by_week_comps, indent=2, default=str))
        _write_spend(paths.COST, f"name_gap_weekly {week}")
        _SPEND.clear()

    named = sum(len(v) for v in by_week_labels.values())
    print(f"\n{named} cluster(s) named across {len(by_week_labels)} week(s)")
    print(f"Saved → {labels_path}\n        {comps_path}")
    # Spend was written per week above; nothing left to record here.


def compress(clusters_path: str, top_n: int, db_path: str,
             synth_model: str | None = None, rank_by: str = "size",
             gaps_path: str | None = None,
             frames: list[dict] | None = None) -> None:
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

    results = _name_batch(real_clusters, db_path, models, synth_fn,
                          compress_prompt, synth_prompt, gap_by_id)

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
    parser.add_argument("--all-weeks", action="store_true",
                        help="name the high-gap clusters of every weekly frame, "
                             "not just the newest week")
    parser.add_argument("--week", metavar="YYYY-MM-DD",
                        help="name one weekly frame by its week_end date")
    parser.add_argument("--gaps", default=str(paths.QUARTER / "name_gaps.json"),
                        help="Path to name_gaps.json (used with --rank-by name-gap)")
    parser.add_argument("--buzzwords", type=int, default=3, metavar="N",
                        help="also name each week's N most negative clusters — "
                             "one term carrying several meanings — from those "
                             "below -0.3 (default 3; 0 disables)")
    parser.add_argument("--force", action="store_true",
                        help="re-name clusters that already have a name. The "
                             "panel is not deterministic: two runs over the "
                             "same clustering produce different prose, so this "
                             "pays again and changes text that was reviewed")
    args = parser.parse_args(argv)
    if args.all_weeks or args.week:
        compress_frames(args.top, args.db, args.synth or None, args.week,
                        buzzwords=args.buzzwords, force=args.force)
    elif args.labels_only:
        label_only(args.db, str(paths.QUARTER / "compressions.json"))
    else:
        compress(args.clusters, args.top, args.db,
                 synth_model=args.synth or None,
                 rank_by=args.rank_by, gaps_path=args.gaps)


if __name__ == "__main__":
    main()
