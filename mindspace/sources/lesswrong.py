import sys
from datetime import datetime, timedelta, timezone

import requests

# `after` and `before` are sent to the server rather than filtering what comes
# back. Without them the query means "the newest N posts", and lookback_days
# only decides what to throw away afterwards — so it never bites:
#
#     lookback_days=7    -> 40 posts, 2026-07-31 .. 2026-08-03
#     lookback_days=90   -> 40 posts, 2026-07-31 .. 2026-08-03   identical
#     lookback_days=365  -> 40 posts, 2026-07-31 .. 2026-08-03   identical
#
# The 40 newest posts are all four days old, so every cutoff older than that
# keeps all of them. With the window on the server, 91 days returns 1590.
_GQL_QUERY = """
{
  posts(input: {terms: {view: "new", limit: %d, after: "%s", before: "%s"}}) {
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


def _fetch(endpoint: str, source_name: str, limit: int, lookback_days: int,
           until: datetime | None = None) -> list[dict]:
    until = until or datetime.now(timezone.utc)
    cutoff = until - timedelta(days=lookback_days)
    try:
        resp = requests.post(
            endpoint,
            json={"query": _GQL_QUERY % (limit, cutoff.date().isoformat(),
                                         until.date().isoformat())},
            timeout=60,
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

    # A full page is suspect. Measured on this API over 2026-05-04..08-03:
    # limit 500 returned exactly 500 and covered only 26 of the 91 days, while
    # limit 2000 and limit 5000 both returned 1590. Nothing in the response
    # says a window was cut short, so it has to be inferred and said out loud.
    if len(results) >= limit:
        print(f"[{source_name}] {len(results)} posts returned at the {limit} cap "
              f"for {cutoff.date()}..{until.date()} — the window is probably "
              f"truncated; raise the limit in config.yaml", file=sys.stderr)

    articles = []
    for post in results:
        posted_raw = post.get("postedAt") or ""
        if posted_raw:
            try:
                posted_dt = datetime.fromisoformat(posted_raw.replace("Z", "+00:00"))
                # Belt and braces: the server already filtered, but an upper
                # bound here keeps consecutive windows from overlapping.
                if not (cutoff <= posted_dt < until):
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


def scrape_lesswrong(limit: int = 40, lookback_days: int = 7,
                     until: datetime | None = None) -> list[dict]:
    return _fetch(
        "https://www.lesswrong.com/graphql",
        "LessWrong",
        limit,
        lookback_days,
        until,
    )


def scrape_alignment_forum(limit: int = 40, lookback_days: int = 7,
                           until: datetime | None = None) -> list[dict]:
    return _fetch(
        "https://www.alignmentforum.org/graphql",
        "Alignment Forum",
        limit,
        lookback_days,
        until,
    )
