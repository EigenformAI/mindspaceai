import sys

import requests
from bs4 import BeautifulSoup

_URL = "https://github.com/trending"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def scrape_github_trending() -> list[dict]:
    try:
        resp = requests.get(_URL, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[GitHub Trending] error: {exc}", file=sys.stderr)
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
            "published_at": None,
        })

    return articles
