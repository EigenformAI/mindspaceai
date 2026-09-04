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

import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

import feedparser

_LD = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
_META = re.compile(r'<meta[^>]+name="author"[^>]+content="([^"]*)"', re.I)
_UA = "mindspaceai/1.0"


def _ld_names(node) -> list[str]:
    """schema.org `author` is a Person, an Organization, or a list of either."""
    out = []
    for x in (node if isinstance(node, list) else [node]):
        if isinstance(x, dict) and x.get("name"):
            out.append(str(x["name"]).strip())
        elif isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def _page_author(link: str, timeout: int = 15) -> str:
    """The byline from the post's own page, for feeds that omit it.

    All four feeds here publish one, in one of two standard places, and none of
    it has to be inferred: HuggingFace names the people who wrote the post,
    Latent Space declares itself an Organization, Simon Willison and Dwarkesh
    use a plain <meta> tag. Guessing from the feed name instead would have been
    wrong for two of the four — "Latent Space" where the publication spells
    itself "Latent.Space", and the site credited for work by named people.

    Returns "" on any failure. A byline is worth one request, never a scrape.
    """
    try:
        req = urllib.request.Request(link, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    for m in _LD.finditer(html):
        try:
            doc = json.loads(m.group(1))
        except Exception:
            continue
        for d in (doc if isinstance(doc, list) else [doc]):
            if isinstance(d, dict) and d.get("author"):
                names = _ld_names(d["author"])
                if names:
                    return ", ".join(names)
    m = _META.search(html)
    return m.group(1).strip() if m else ""


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
            "author": getattr(entry, "author", "") or _page_author(link),
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
