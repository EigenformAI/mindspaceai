from datetime import datetime, timedelta, timezone

import feedparser


def scrape_rss(name: str, url: str, lookback_days: int = 30) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        published = None
        if getattr(entry, "published_parsed", None):
            published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if published_dt < cutoff:
                continue
            published = published_dt.isoformat()
        elif getattr(entry, "updated_parsed", None):
            updated_dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            if updated_dt < cutoff:
                continue
            published = updated_dt.isoformat()

        content = ""
        if hasattr(entry, "content") and entry.content:
            content = entry.content[0].value
        elif hasattr(entry, "summary"):
            content = entry.summary

        link = getattr(entry, "link", None)
        if not link:
            continue

        articles.append({
            "source": name,
            "url": link,
            "title": getattr(entry, "title", ""),
            "content": content,
            "author": getattr(entry, "author", ""),
            "published_at": published,
        })
    return articles
