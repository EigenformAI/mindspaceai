"""RSS and Atom feeds.

The reach of this source is set by feed depth, not by `lookback_days`, and the
depth varies enormously. Measured 2026-08-03:

    HuggingFace Blog   834 entries back to 2020-02-14
    Simon Willison      30 entries back to 2026-07-26   (~8 days)
    Latent Space        20 entries back to 2026-07-14
    Dwarkesh Patel      20 entries back to 2026-04-07

Simon Willison publishes about three pieces a day into a 30-entry feed, so a
91-day request gets 30 documents that all sit in the newest week and none in
the twelve before it — out of roughly 270 actually written. A cluster of his
writing can therefore look like it is emerging when only the source arrived.
The coverage line printed below exists so that shows up rather than being
discovered later as a finding.

Content availability differs too, and HuggingFace Blog is the odd one:

    Simon Willison     median  190 words
    Latent Space       median 2324 words
    Dwarkesh Patel     median 7324 words
    HuggingFace Blog   median    0 words   — every entry

Its feed carries only <title>, <pubDate>, <link> and <guid>: no <description>,
no <content:encoded>. Nothing here can recover that; the text is only on the
post's own page.
"""

import sys
from datetime import datetime, timedelta, timezone

import feedparser


def scrape_rss(name: str, url: str, lookback_days: int = 30,
               until: datetime | None = None) -> list[dict]:
    until = until or datetime.now(timezone.utc)
    cutoff = until - timedelta(days=lookback_days)
    feed = feedparser.parse(url)
    articles = []
    feed_dates = []

    for entry in feed.entries:
        published = None
        stamp = (getattr(entry, "published_parsed", None)
                 or getattr(entry, "updated_parsed", None))
        if stamp:
            when = datetime(*stamp[:6], tzinfo=timezone.utc)
            feed_dates.append(when)
            # Upper bound as well as lower: consecutive windows then chain
            # without a document falling into both.
            if not (cutoff <= when < until):
                continue
            published = when.isoformat()

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

    if feed_dates:
        oldest = min(feed_dates).date()
        note = ""
        if oldest > cutoff.date():
            missed = (oldest - cutoff.date()).days
            note = (f" — feed only reaches {oldest}, so the first {missed} day(s) "
                    "of this window have no coverage from it")
        print(f"[RSS] {name}: {len(feed.entries)} entries, oldest {oldest}, "
              f"{len(articles)} in window{note}",
              file=sys.stderr if note else sys.stdout)

    return articles
