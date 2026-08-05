"""Hacker News via the Algolia search API.

A note on what a "story" is here: 38 of 40 sampled stories are links with no
text of their own — `story_text` is only set for Ask HN and Show HN. So most
documents from this source are titles. That is not a defect to work around;
an HN title is written to convey what the thing is, which is what clustering
needs. The article body lives on someone else's site.
"""

import sys
import time
from datetime import datetime, timedelta, timezone

import requests

_ALGOLIA = "https://hn.algolia.com/api/v1/search"

# Algolia refuses to page past 1000 hits for a query — it says so explicitly on
# page 1. The ceiling is per query, and both the time bounds and the points
# floor are part of the query, so narrowing either lowers the count. Measured
# for "AI": 91 days = 18324 hits, 7 days = 1505, 1 day = 139. Slicing the range
# into weeks keeps every query under the ceiling without a special case.
_ALGOLIA_MAX_HITS = 1000
_WEEK = timedelta(days=7)


def _week_windows(lookback_days: int,
                  until: datetime | None = None) -> list[tuple[datetime, datetime]]:
    """The range cut into windows of at most a week, oldest first."""
    until = until or datetime.now(timezone.utc)
    start = until - timedelta(days=lookback_days)
    out, cursor = [], start
    while cursor < until:
        stop = min(cursor + _WEEK, until)
        out.append((cursor, stop))
        cursor = stop
    return out


def scrape_hackernews(queries: list[str], limit_per_query: int = 50,
                      lookback_days: int = 7, min_points: int = 10,
                      until: datetime | None = None) -> list[dict]:
    """Stories matching each query, sampled evenly across the range.

    `limit_per_query` is a quota PER WEEK, not per run. Applied to the whole
    range it would take the top N of the newest end and leave the rest of the
    range unrepresented, which is the failure this module was fixed for.

    `min_points` is a noise floor, applied server-side. Without it "AI" matches
    18476 stories over 91 days, most of them incidental — the word appearing in
    a URL or body ("Airport sign fails to boot"). Above 10 points that falls to
    2057 and the results are actually about AI. It is deliberately not higher:
    a high score means already popular, which is in tension with looking for
    concepts that are still emerging.
    """
    windows = _week_windows(lookback_days, until)
    seen_urls: set[str] = set()
    articles = []
    total_available = 0

    for since, until in windows:
        cutoff = int(since.timestamp())
        ceiling = int(until.timestamp())
        numeric = f"created_at_i>{cutoff},created_at_i<{ceiling}"
        if min_points:
            numeric += f",points>{min_points}"

        for query in queries:
            try:
                resp = requests.get(
                    _ALGOLIA,
                    params={
                        "query": query,
                        "tags": "story",
                        "numericFilters": numeric,
                        "hitsPerPage": limit_per_query,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                payload = resp.json()
                hits = payload.get("hits", [])
            except Exception as exc:
                print(f"[HackerNews] query '{query}' error: {exc}", file=sys.stderr)
                continue

            # Algolia hands back the true match count for free, so what was left
            # behind can be stated rather than guessed at.
            available = payload.get("nbHits", len(hits))
            total_available += available
            if available > _ALGOLIA_MAX_HITS:
                print(f"[HackerNews] '{query}' {since.date()}..{until.date()}: "
                      f"{available} matches exceeds Algolia's {_ALGOLIA_MAX_HITS} "
                      "ceiling — narrow the window or raise min_points",
                      file=sys.stderr)

            for hit in hits:
                url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                created_ts = hit.get("created_at_i")
                published_at = (
                    datetime.fromtimestamp(created_ts, tz=timezone.utc).isoformat()
                    if created_ts else None
                )
                articles.append({
                    "source": "Hacker News",
                    "url": url,
                    "title": hit.get("title", ""),
                    "content": hit.get("story_text", "") or "",
                    "author": hit.get("author", ""),
                    "published_at": published_at,
                })

            time.sleep(0.3)

    if len(windows) > 1:
        print(f"[HackerNews] {len(articles)} stories from {len(windows)} weeks "
              f"({total_available} matched above {min_points} points; "
              f"{limit_per_query}/query/week is a weighting choice, not a ceiling)")
    return articles
