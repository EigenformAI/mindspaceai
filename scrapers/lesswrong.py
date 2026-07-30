import sys
from datetime import datetime, timedelta, timezone

import requests

_GQL_QUERY = """
{
  posts(input: {terms: {view: "new", limit: %d}}) {
    results {
      _id
      title
      pageUrl
      postedAt
      author
      user { displayName }
      htmlBody
    }
  }
}
"""


def _fetch(endpoint: str, source_name: str, limit: int, lookback_days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    try:
        resp = requests.post(
            endpoint,
            json={"query": _GQL_QUERY % limit},
            timeout=20,
            headers={"Content-Type": "application/json"},
        )
        if not resp.ok:
            print(f"[{source_name}] HTTP {resp.status_code}: {resp.text[:600]}", file=sys.stderr)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[{source_name}] error: {exc}", file=sys.stderr)
        return []

    results = data.get("data", {}).get("posts", {}).get("results", []) or []
    articles = []
    for post in results:
        posted_raw = post.get("postedAt") or ""
        if posted_raw:
            try:
                posted_dt = datetime.fromisoformat(posted_raw.replace("Z", "+00:00"))
                if posted_dt < cutoff:
                    continue
            except ValueError:
                pass

        url = post.get("pageUrl") or ""
        if not url:
            continue

        articles.append({
            "source": source_name,
            "url": url,
            "title": post.get("title", ""),
            "content": post.get("htmlBody", ""),
            "author": post.get("author") or (post.get("user") or {}).get("displayName", ""),
            "published_at": posted_raw,
        })
    return articles


def scrape_lesswrong(limit: int = 40, lookback_days: int = 7) -> list[dict]:
    return _fetch(
        "https://www.lesswrong.com/graphql",
        "LessWrong",
        limit,
        lookback_days,
    )


def scrape_alignment_forum(limit: int = 40, lookback_days: int = 7) -> list[dict]:
    return _fetch(
        "https://www.alignmentforum.org/graphql",
        "Alignment Forum",
        limit,
        lookback_days,
    )
