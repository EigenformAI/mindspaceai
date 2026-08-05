"""HuggingFace daily papers — a curated pick per day, whose entries are arXiv
links. Title and abstract come from HuggingFace's own response, so arXiv itself
is never contacted.

TWO THINGS WORTH KNOWING BEFORE READING ANYTHING COMPUTED FROM THIS SOURCE.

**The daily pick is an editorial choice.** A share or a growth slope over it
measures what the curators found interesting, not what the field wrote.

**`published_at` is the arXiv publication date, while the loop walks curation
dates.** A paper picked in July but published in May is stored with the May
date, so roughly 1.6% of a window's results fall outside the window that was
asked for. They are kept rather than filtered — they are real papers with real
dates, and dropping them would discard valid data to tidy a boundary. The
consequence to remember: curation lags publication, so the most recent week
always under-represents this source (its share decays from about 40% to 16%).
"""

import sys
from datetime import datetime, timedelta, timezone

import requests

_API = "https://huggingface.co/api/daily_papers"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def scrape_hf_papers(limit_per_day: int | None = None, lookback_days: int = 7,
                     until: datetime | None = None) -> list[dict]:
    """Every paper curated in the last `lookback_days` days.

    `limit_per_day` caps each curation day, never the whole run — the name
    says so because the old total `limit` was enforced by a `break` that
    stopped the day loop: a 180-day request returned a full 50 papers from
    the first two days, reported success, and never asked for the other 178.
    A per-day cap cannot do that: every day in the range is still requested.
    HuggingFace curates about 26 papers a day, so any sane value never binds.
    Pass None for no cap.
    """
    articles = []
    seen: set[str] = set()
    days_seen = 0

    # `until` is an exclusive bound: a live run's last curation day is today,
    # an anchored run's is the day before the anchor date.
    until = until or datetime.now(timezone.utc)
    last_day = until - timedelta(microseconds=1)

    for day_offset in range(lookback_days):
        date = (last_day - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                _API,
                params={"date": date},
                headers=_HEADERS,
                timeout=15,
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            papers = resp.json()
        except Exception as exc:
            print(f"[HF Papers] {date} error: {exc}", file=sys.stderr)
            continue

        if not isinstance(papers, list):
            papers = papers.get("papers", []) if isinstance(papers, dict) else []

        kept_today = 0
        for paper in papers:
            info = paper.get("paper") or paper
            paper_id = info.get("id") or paper.get("id")
            if not paper_id or paper_id in seen:
                continue
            if limit_per_day is not None and kept_today >= limit_per_day:
                print(f"[HF Papers] {date}: {len(papers)} curated, keeping "
                      f"{limit_per_day} (limit_per_day)", file=sys.stderr)
                break
            seen.add(paper_id)
            kept_today += 1

            authors = info.get("authors", [])
            articles.append({
                "source": "HuggingFace Papers",
                "url": f"https://arxiv.org/abs/{paper_id}",
                "title": info.get("title", ""),
                "content": info.get("summary", ""),
                "author": ", ".join(
                    (a.get("name") or a) if isinstance(a, dict) else str(a)
                    for a in authors[:3]
                ),
                "published_at": info.get("publishedAt") or info.get("published_at"),
            })

        if kept_today:
            days_seen += 1

    if lookback_days > 1:
        print(f"[HF Papers] {len(articles)} papers across {days_seen}/{lookback_days} days")
    return articles
