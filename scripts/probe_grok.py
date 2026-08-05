#!/usr/bin/env python3
"""Run each Grok prompt in isolation and report its real tool usage.

    uv run python probe_grok.py                 # 7-day window, config limit
    uv run python probe_grok.py --days 91
    uv run python probe_grok.py --limit 15      # cheaper probe, same counts

Answers "how many x_search tool calls does each prompt actually make?" from the
API's own accounting rather than guesswork: xAI returns
`server_side_tool_usage_details.x_search_calls`, `num_sources_used` and the
real cost on every response. Nothing is estimated here.

Each prompt is a paid call (measured $0.17–0.66 at limit 15), and this probe
does NOT store anything in the database — it is measurement only.
"""

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import yaml

# The probe lives in scripts/ but everything it touches lives in the package:
# sources, config.yaml, .env and the output tree. Anchor on the file's location
# so it runs identically from any working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from mindspace.sources.grok_twitter import _INSTRUCTIONS, _extract_text, _parse_posts
from mindspace import paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7,
                    help="window size to probe (default 7)")
    ap.add_argument("--week", type=int, default=1,
                    help="which window counting back from today: 1 = the most "
                         "recent --days days (default), 2 = the window before "
                         "that, and so on")
    ap.add_argument("--limit", type=int, default=None,
                    help="posts per prompt (default: config limit_per_prompt)")
    ap.add_argument("--config", default=str(paths.CONFIG))
    args = ap.parse_args()

    key = os.environ.get("XAI_API_KEY", "")
    if not key:
        print("XAI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI

    cfg = yaml.safe_load(open(args.config))["sources"]["twitter"]
    prompts = cfg.get("prompts", [])
    limit = args.limit or cfg.get("limit_per_prompt", 50)
    until = date.today() - timedelta(days=args.days * (args.week - 1))
    since = until - timedelta(days=args.days)
    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")

    print(f"window {since}..{until} ({args.days} days, week {args.week}), "
          f"limit {limit}/prompt, {len(prompts)} prompts\n")
    print(f"  {'prompt':<26}{'tool_calls':>11}{'sources':>9}{'posts':>7}"
          f"{'out_tok':>9}{'cost':>8}{'secs':>6}")

    rows = []
    for p in prompts:
        t0 = time.time()
        try:
            resp = client.responses.create(
                model="grok-4.5",
                instructions=_INSTRUCTIONS,
                input=[{"role": "user", "content":
                        f"{p['text']}\n\nReturn up to {limit} posts. "
                        "Return ONLY a JSON array, nothing else."}],
                tools=[{"type": "x_search",
                        "from_date": since.isoformat(),
                        "to_date": until.isoformat()}],
            )
        except Exception as exc:
            print(f"  {p['label']:<26} ERROR: {exc}", file=sys.stderr)
            continue

        u = resp.usage
        # xAI's extra usage fields can arrive as plain dicts rather than SDK
        # objects; getattr on a dict silently returns the default, so both
        # shapes have to be read or tool_calls comes back None on a paid call.
        det = getattr(u, "server_side_tool_usage_details", None)
        if isinstance(det, dict):
            calls = det.get("x_search_calls")
        else:
            calls = getattr(det, "x_search_calls", None)
        if calls is None and hasattr(u, "model_dump"):
            calls = (u.model_dump().get("server_side_tool_usage_details")
                     or {}).get("x_search_calls")
        sources = getattr(u, "num_sources_used", None)
        parsed = _parse_posts(_extract_text(resp))
        posts = len(parsed)
        # 1 USD = 10^10 ticks per docs.x.ai/developers/cost-tracking.
        cost = getattr(u, "cost_in_usd_ticks", 0) / 1e10
        secs = time.time() - t0
        rows.append({"label": p["label"], "tool_calls": calls,
                     "sources": sources, "posts": posts,
                     "output_tokens": u.output_tokens,
                     "cost_usd": cost, "seconds": round(secs, 1),
                     "posts_data": parsed,
                     "raw_usage": u.model_dump()
                     if hasattr(u, "model_dump") else None})
        print(f"  {p['label']:<26}{str(calls):>11}{str(sources):>9}{posts:>7}"
              f"{u.output_tokens:>9}{cost:>8.2f}{secs:>6.0f}")

    if not rows:
        sys.exit(1)

    print(f"\n  {'TOTAL':<26}{sum(r['tool_calls'] or 0 for r in rows):>11}"
          f"{sum(r['sources'] or 0 for r in rows):>9}"
          f"{sum(r['posts'] for r in rows):>7}"
          f"{sum(r['output_tokens'] for r in rows):>9}"
          f"{sum(r['cost_usd'] for r in rows):>8.2f}")

    name = (f"grok_probe_{args.days}d.json" if args.week == 1
            else f"grok_probe_{args.days}d_w{args.week}.json")
    # A probe is money spent, so its record sits with the ledgers, not with
    # the pipeline artefacts.
    out = paths.COST / name
    paths.ensure()
    with open(out, "w") as fh:
        json.dump({"window": f"{since}..{until}", "limit_per_prompt": limit,
                   "prompts": rows}, fh, indent=2)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
