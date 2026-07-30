import sys
import time
from datetime import datetime, timedelta, timezone

import requests

_ALGOLIA = "https://hn.algolia.com/api/v1/search"


def scrape_hackernews(queries: list[str], limit_per_query: int = 20,
                      lookback_days: int = 7) -> list[dict]:
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
    seen_urls: set[str] = set()
    articles = []

    for query in queries:
        try:
            resp = requests.get(
                _ALGOLIA,
                params={
                    "query": query,
                    "tags": "story",
                    "numericFilters": f"created_at_i>{cutoff}",
                    "hitsPerPage": limit_per_query,
                },
                timeout=15,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except Exception as exc:
            print(f"[HackerNews] query '{query}' error: {exc}", file=sys.stderr)
            continue

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

    return articles
