"""GitHub Trending.

THIS SOURCE CANNOT BE BACKFILLED. `?since=daily|weekly|monthly` all serve the
window ending now; an earlier week cannot be asked for. The page carries no date
markup, and the API's `created_at` / `pushed_at` are not what is wanted — a
repository created in 2015 and trending today would be filed under 2015.

So `published_at` is the day the entry was OBSERVED trending. That is not the
`scraped_at` substitution refused elsewhere in this project: there a real
publication date exists and fetch time would overwrite it, whereas here "this
repository was trending on date X" is the fact itself. Leaving it None, as it
was, meant every row was silently dropped by any publication-time filter — 15
documents thrown away per run.

The three windows are ALMOST DISJOINT, which is why `since` matters. Measured
2026-08-03:

    no param      15 repos   (identical to since=daily — GitHub's default)
    since=weekly  18 repos
    since=monthly 22 repos

    daily ∩ weekly = 2 | weekly ∩ monthly = 3 | daily ∩ monthly = 0 | union = 50

A day's spike, a week's momentum and a month's growth are different
repositories, not wider views of the same ones. A weekly run reading the
default `daily` therefore sees one day in seven, and what it misses is not
duplicates.
"""

import sys
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

_URL = "https://github.com/trending"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
_WINDOWS = ("daily", "weekly", "monthly")


def windows_for(lookback_days: int) -> list[str]:
    """The smallest `since` covering the request; all three when it cannot."""
    if lookback_days <= 1:
        return ["daily"]
    if lookback_days <= 7:
        return ["weekly"]
    if lookback_days <= 30:
        return ["monthly"]
    # Wider than any window on offer. The past is unreachable either way, so
    # take the widest snapshot of the present instead of one arbitrary slice.
    return list(_WINDOWS)


def _fetch_one(since: str, observed: datetime) -> list[dict]:
    try:
        resp = requests.get(_URL, params={"since": since},
                            headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[GitHub Trending] since={since} error: {exc}", file=sys.stderr)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    articles = []

    for repo in soup.select("article.Box-row"):
        h2 = repo.select_one("h2 a")
        if not h2:
            continue

        path = h2.get("href", "").strip("/")
        url = f"https://github.com/{path}"
        title = path.replace("/", " / ")

        desc_el = repo.select_one("p")
        description = desc_el.get_text(strip=True) if desc_el else ""

        lang_el = repo.select_one("[itemprop='programmingLanguage']")
        language = lang_el.get_text(strip=True) if lang_el else ""

        stars_el = repo.select_one("a[href$='/stargazers']")
        stars = stars_el.get_text(strip=True) if stars_el else ""

        content = description
        if language:
            content = f"[{language}] {content}"
        if stars:
            content += f" ★{stars}"

        articles.append({
            "source": "GitHub Trending",
            "url": url,
            "title": title,
            "content": content,
            "author": path.split("/")[0] if "/" in path else "",
            # Observed, not published — see the module docstring.
            "published_at": observed.isoformat(),
        })

    return articles


def scrape_github_trending(lookback_days: int = 7,
                           observed_at: datetime | None = None) -> list[dict]:
    observed = (observed_at or datetime.now(timezone.utc)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    since_list = windows_for(lookback_days)

    articles, seen = [], set()
    for since in since_list:
        for a in _fetch_one(since, observed):
            if a["url"] in seen:
                continue
            seen.add(a["url"])
            articles.append(a)

    if lookback_days > 30:
        print(f"[GitHub Trending] {len(articles)} repos from "
              f"{'+'.join(since_list)}, all dated {observed.date()} — this source "
              f"has no history, so the earlier weeks of a {lookback_days}-day "
              "window get nothing from it", file=sys.stderr)
    return articles
