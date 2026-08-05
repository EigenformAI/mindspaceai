#!/usr/bin/env python3
"""Per-source totals for mindspace.db.

    uv run python scripts/db_stats.py             # whole DB
    uv run python db_stats.py --week 2    # only the 7-day window ending 7 days ago

`--week` counts the same way as pipeline.py and probe_grok.py: week 1 is the
--days window ending today, week 2 the one before it, and so on. In week mode
undated articles never appear — the window filter compares published_at, and
NULL fails every comparison.
"""

import argparse
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

# Anchored on the repo root: this reports on the project's database whatever
# directory it is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindspace import paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=None, metavar="N",
                    help="restrict to one window counting back from today")
    ap.add_argument("--days", type=int, default=7,
                    help="window size for --week (default 7)")
    ap.add_argument("--db", default=str(paths.DB))
    args = ap.parse_args()

    where, params = "", []
    if args.week is not None:
        hi = date.today() - timedelta(days=args.days * (args.week - 1))
        lo = hi - timedelta(days=args.days)
        where = "WHERE published_at >= ? AND published_at < ?"
        params = [lo.isoformat(), hi.isoformat()]
        print(f"window week {args.week}: {lo}..{hi} (half-open)\n")

    conn = sqlite3.connect(args.db)
    rows = conn.execute(f"""
        SELECT source,
               COUNT(*)                                    AS total,
               SUM(embedding IS NOT NULL)                  AS embedded,
               SUM(published_at IS NULL OR published_at = '') AS undated,
               MIN(published_at), MAX(published_at)
        FROM articles {where}
        GROUP BY source ORDER BY total DESC
    """, params).fetchall()
    conn.close()

    if not rows:
        print("no articles in this window")
        return

    print(f"{'source':<22}{'total':>7}{'embedded':>10}{'undated':>9}"
          f"  {'oldest':<12}{'newest':<12}")
    t = e = u = 0
    for src, n, emb, und, oldest, newest in rows:
        t += n
        e += emb or 0
        u += und or 0
        print(f"{src:<22}{n:>7}{emb or 0:>10}{und or 0:>9}"
              f"  {(oldest or '—')[:10]:<12}{(newest or '—')[:10]:<12}")
    print(f"{'TOTAL':<22}{t:>7}{e:>10}{u:>9}")


if __name__ == "__main__":
    main()
