#!/usr/bin/env python3
"""
Extract URLs from scraped article content, fetch each page, and store as new articles.

Usage:
    python expand_links.py [--depth N] [--limit N] [--source FILTER]

Options:
    --depth N       How many levels of links to follow (default: 1)
    --limit N       Max URLs to fetch per run (default: 200)
    --source FILTER Only expand links from articles matching this source substring
    --dry-run       Print URLs that would be fetched, don't store anything
"""
import argparse
import re
import sys
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import db

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mindspace-bot/1.0)"}
_TIMEOUT = 12
_DELAY = 0.5

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")

# Only fetch from these high-signal domains
_ALLOW_DOMAINS = {
    "arxiv.org",
    "reddit.com",
    "x.com", "twitter.com",
}


def _should_fetch(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if not p.scheme.startswith("http"):
        return False
    domain = p.netloc.lower().lstrip("www.")
    return any(domain == d or domain.endswith("." + d) for d in _ALLOW_DOMAINS)


def _extract_arxiv(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    title_tag = soup.find("h1", class_="title")
    title = title_tag.get_text(strip=True).removeprefix("Title:").strip() if title_tag else ""
    abstract_tag = soup.find("blockquote", class_="abstract")
    abstract = abstract_tag.get_text(strip=True).removeprefix("Abstract:").strip() if abstract_tag else ""
    if not title:
        title = soup.title.string.strip() if soup.title else url
    return title, abstract


def _extract_github(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    title = soup.title.string.strip() if soup.title else url
    readme = soup.find("article", {"data-testid": "readme"}) or soup.find("div", id="readme")
    content = readme.get_text(" ", strip=True)[:3000] if readme else ""
    return title, content


def _extract_reddit(url: str) -> tuple[str, str]:
    json_url = url.split("?")[0].rstrip("/") + ".json?limit=25"
    try:
        resp = requests.get(json_url, headers={**_HEADERS, "Accept": "application/json"},
                            timeout=_TIMEOUT)
        data = resp.json()
        post = data[0]["data"]["children"][0]["data"]
        title = post.get("title", "")
        body = post.get("selftext", "")
        comments = []
        for c in data[1]["data"]["children"][:10]:
            cd = c.get("data", {})
            if cd.get("body") and cd["body"] != "[deleted]":
                comments.append(cd["body"])
        content = body + "\n\n" + "\n---\n".join(comments)
        return title, content[:4000]
    except Exception:
        return "", ""


def _extract_generic(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    title = soup.title.string.strip() if soup.title else url
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    for selector in ["article", "main", '[role="main"]', ".post-content", ".entry-content", ".content"]:
        block = soup.select_one(selector)
        if block:
            return title, block.get_text(" ", strip=True)[:4000]
    return title, soup.get_text(" ", strip=True)[:3000]


def fetch_and_extract(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "html" not in ct:
            return None
    except Exception as exc:
        print(f"  ! {url[:70]} — {exc}", file=sys.stderr)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    domain = urlparse(url).netloc.lower()

    if "arxiv.org" in domain:
        canonical = re.sub(r"/pdf/", "/abs/", url).rstrip(".pdf")
        title, content = _extract_arxiv(soup, canonical)
        source = "web/arxiv"
    elif "reddit.com" in domain:
        title, content = _extract_reddit(url)
        source = "web/reddit"
    elif "x.com" in domain or "twitter.com" in domain:
        return None  # already scraped; skip
    else:
        title, content = _extract_generic(soup, url)
        source = "web/extracted"

    if not content.strip():
        return None

    return {
        "source": source,
        "url": url,
        "title": title,
        "content": content,
        "author": None,
        "published_at": None,
    }


def extract_urls_from_db(source_filter: str | None = None) -> list[str]:
    with db.get_conn() as conn:
        q = "SELECT content, url FROM articles"
        params = []
        if source_filter:
            q += " WHERE source LIKE ?"
            params.append(f"%{source_filter}%")
        rows = conn.execute(q, params).fetchall()

    existing = set()
    with db.get_conn() as conn:
        existing = {r[0] for r in conn.execute("SELECT url FROM articles").fetchall()}

    found: list[str] = []
    seen: set[str] = set()
    for content, parent_url in rows:
        for m in _URL_RE.finditer(content or ""):
            u = m.group(0).rstrip(".,;:)")
            if u not in seen and u not in existing and _should_fetch(u):
                seen.add(u)
                found.append(u)
    return found


def expand(depth: int = 1, limit: int = 200,
           source_filter: str | None = None, dry_run: bool = False) -> int:
    db.init_db()
    total_saved = 0
    queue = extract_urls_from_db(source_filter)
    print(f"[expand] found {len(queue)} candidate URLs in DB content")

    for level in range(depth):
        batch = queue[:limit]
        print(f"[expand] level {level + 1}/{depth}: fetching {len(batch)} URLs")
        next_queue: list[str] = []

        for i, url in enumerate(batch, 1):
            print(f"  ({i}/{len(batch)}) {url[:80]}", end=" … ", flush=True)
            if dry_run:
                print("(dry-run)")
                continue

            article = fetch_and_extract(url)
            if article is None:
                print("skip")
                time.sleep(_DELAY)
                continue

            db.upsert_article(**article)
            print(f"saved [{article['source']}]")
            total_saved += 1

            if level + 1 < depth:
                for m in _URL_RE.finditer(article["content"]):
                    u = m.group(0).rstrip(".,;:)")
                    if _should_fetch(u):
                        next_queue.append(u)

            time.sleep(_DELAY)

        queue = next_queue

    if not dry_run:
        print(f"\n[expand] done — {total_saved} new articles saved | total: {db.article_count()}")
    return total_saved


def main():
    parser = argparse.ArgumentParser(description="Expand links found in scraped articles")
    parser.add_argument("--depth", type=int, default=1, help="Link-follow depth (default 1)")
    parser.add_argument("--limit", type=int, default=200, help="Max URLs per level (default 200)")
    parser.add_argument("--source", type=str, default=None, help="Filter source articles by substring")
    parser.add_argument("--dry-run", action="store_true", help="List URLs without fetching")
    args = parser.parse_args()

    expand(
        depth=args.depth,
        limit=args.limit,
        source_filter=args.source,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
