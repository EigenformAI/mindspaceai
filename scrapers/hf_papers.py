import sys
from datetime import datetime, timedelta, timezone

import requests

_API = "https://huggingface.co/api/daily_papers"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def scrape_hf_papers(limit: int = 50, lookback_days: int = 7) -> list[dict]:
    articles = []
    seen: set[str] = set()

    for day_offset in range(lookback_days):
        date = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime("%Y-%m-%d")
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

        for paper in papers:
            paper_id = (paper.get("paper") or paper).get("id") or paper.get("id")
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)

            if len(articles) >= limit:
                break

            info = paper.get("paper") or paper
            title = info.get("title", "")
            abstract = info.get("summary", "")
            authors = info.get("authors", [])
            author_str = ", ".join(
                (a.get("name") or a) if isinstance(a, dict) else str(a)
                for a in authors[:3]
            )

            url = f"https://arxiv.org/abs/{paper_id}"
            published_at = info.get("publishedAt") or info.get("published_at")

            articles.append({
                "source": "HuggingFace Papers",
                "url": url,
                "title": title,
                "content": abstract,
                "author": author_str,
                "published_at": published_at,
            })

        if len(articles) >= limit:
            break

    return articles
