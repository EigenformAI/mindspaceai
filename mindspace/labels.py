import argparse
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from . import paths

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os

from .compress import _call_openrouter, _clean_label, _write_spend, _LABEL_MODEL

_PROMPT = (
    "These keywords and titles come from one cluster of AI-related documents.\n"
    "Keywords: {keywords}\n"
    "Sample titles:\n{titles}\n\n"
    "Write a compact label of 2-5 words naming the cluster's topic. "
    "Noun phrase only, no verbs, no punctuation. "
    "Give exactly ONE label, plain text, no markdown."
)


def _titles_by_position(frames: list[dict], db_path: str) -> dict[int, str]:
    """corpus position → title, for every position any frame references."""
    proj = np.load(paths.QUARTER / "projection.npz", allow_pickle=True)
    ids = [str(x) for x in proj["article_ids"]]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    titles: dict[str, str] = {}
    chunk = 500
    for i in range(0, len(ids), chunk):
        part = ids[i : i + chunk]
        for r in conn.execute(
            f"SELECT id, title FROM articles WHERE id IN ({','.join('?' * len(part))})",
            part,
        ):
            titles[r["id"]] = r["title"] or ""
    conn.close()
    return {pos: titles.get(aid, "") for pos, aid in enumerate(ids)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true",
                    help="label only the newest week's clusters")
    ap.add_argument("--db", default=str(paths.DB))
    args = ap.parse_args(argv)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set — nothing can label.", file=sys.stderr)
        sys.exit(1)

    frames_path = paths.QUARTER / "frames.json"
    if not frames_path.exists():
        print(f"{frames_path} not found — run pipeline.py first.", file=sys.stderr)
        sys.exit(1)

    frames = json.loads(frames_path.read_text())
    title_of = _titles_by_position(frames, args.db)
    key = os.environ["OPENROUTER_API_KEY"]

    todo = []
    for f in (frames[-1:] if args.latest else frames):
        for c in f.get("clusters", []):
            if c.get("label"):        # already labelled; re-running is free
                continue
            member_titles = [
                title_of.get(f["week_idx"][p], "")
                for p in c.get("points", [])[:6]
                if p < len(f["week_idx"])
            ]
            prompt = _PROMPT.format(
                keywords=c.get("keywords", ""),
                titles="\n".join(f"- {t[:110]}" for t in member_titles if t),
            )
            todo.append((f["week_end"], c, prompt))

    if not todo:
        print("every cluster already has a label")
        return
    print(f"labelling {len(todo)} clusters "
          f"({'newest week' if args.latest else f'{len(frames)} stops'})…")

    def one(item):
        week, c, prompt = item
        return week, c, _clean_label(_call_openrouter(prompt, _LABEL_MODEL, key,
                                                      max_tokens=40))

    done = failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in as_completed(pool.submit(one, t) for t in todo):
            try:
                week, c, label = fut.result()
            except Exception as exc:
                failed += 1
                print(f"  ! {exc}", file=sys.stderr)
                continue
            if label:
                c["label"] = label
                done += 1
                print(f"  {week}  [{c['cluster_id']:>3}] {label}")
            else:
                failed += 1

    frames_path.write_text(json.dumps(frames))
    print(f"\n{done} labelled, {failed} failed → {frames_path}")
    if failed:
        print("failed clusters keep their TF-IDF keywords in the sidebar — "
              "re-run to fill them in", file=sys.stderr)
    _write_spend(paths.COST, "label-all" if not args.latest else "label-latest")
    print("next: pipeline.py --viz-only, then frontend/publish.sh")


if __name__ == "__main__":
    main()
