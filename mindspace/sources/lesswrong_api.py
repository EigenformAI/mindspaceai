"""LessWrong's agent-facing Markdown API — a fallback when GraphQL is refused.

LessWrong publishes this interface for automated readers and documents it at
/api/SKILL.md. It matters here because the two live behind different rules:
measured 2026-08-14, /graphql and even /robots.txt answered 429 from this host
while every /api/ route answered 200. Using it is not a way around the block —
it is the door the site built for exactly this kind of client.

It is a fallback and not the primary path, for two reasons:

  * No date window. /api/latest returns the newest N posts and ignores
    before/after, so the window has to be applied here after fetching. At the
    measured rate of ~20 posts a day, the 100-post ceiling reaches back about
    five days — enough to repair a gap, short of a full week.
  * One request per post for the body. GraphQL returns a whole window in a
    single call; this costs a call per post, which is heavier on their servers
    and the reason it should not be used when GraphQL will answer.

Verified against the same population, not merely a similar one: on 2026-08-09,
the one day both sides covered completely, this route and the GraphQL scrape
returned the same eleven posts, title for title. /api/recent does NOT — it
lists roughly a quarter of them, being a curated view rather than the firehose.
"""

import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from . import SourceUnavailable

BASE = "https://www.lesswrong.com"
USER_AGENT = "mindspace/0.1 (research crawler; contact: eigenform.ai)"
LIST_LIMIT = 100          # the API's documented ceiling for list routes
REQUEST_PAUSE = 1.5       # seconds between post fetches

# "### [Title](/api/post/slug)\n By [Author](/users/x) \n 2026-08-09 15:58:07Z"
_ENTRY = re.compile(
    r"### \[(?P<title>.*?)\]\((?P<path>/api/post/[^)]+)\)\s*\n"
    r"\s*By \[(?P<author>[^\]]*)\][^\n]*\n"
    r"\s*(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2}:\d{2})Z",
    re.M)
_HTML_URL = re.compile(r"Post URL \(HTML\): \[(/posts/[^\]]+)\]")


def _get(path: str, timeout: int = 30) -> str:
    req = urllib.request.Request(
        BASE + path,
        headers={"User-Agent": USER_AGENT, "Accept": "text/markdown"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _body_after_header(markdown: str) -> str:
    """The post text, with the bullet-list header the API prefixes stripped.

    That header repeats the title, author, date, karma, tags and three URLs.
    Left in, every post would share a block of near-identical text, which is
    exactly the kind of shared vocabulary the lexical score is meant to detect
    — it would read as coherence that the documents do not have.
    """
    lines = markdown.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("*   ") or line.startswith("* "):
            start = i + 1
        elif start and line.strip():
            break
    return "\n".join(lines[start:]).strip()


def scrape_lesswrong_api(lookback_days: int = 7,
                         until: datetime | None = None,
                         limit: int = LIST_LIMIT) -> list[dict]:
    """Posts published in the half-open [until - lookback_days, until).

    Raises SourceUnavailable if the listing cannot be read; a single post that
    fails is skipped and reported, because losing one body is not a reason to
    lose the whole window.
    """
    until = until or datetime.now(timezone.utc)
    cutoff = until - timedelta(days=lookback_days)

    try:
        listing = _get(f"/api/latest?limit={limit}")
    except Exception as exc:
        raise SourceUnavailable("LessWrong", exc) from exc

    wanted = []
    oldest = None
    for m in _ENTRY.finditer(listing):
        stamp = f"{m['date']}T{m['time']}+00:00"
        try:
            posted = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        oldest = min(oldest or posted, posted)
        if cutoff <= posted < until:
            wanted.append({"path": m["path"], "title": m["title"].strip(),
                           "author": m["author"].strip(),
                           "published_at": posted.isoformat()})

    # The ceiling is silent: a full page that stops inside the window means
    # older posts exist and were never offered. Saying so is the difference
    # between a short week and a week that looks short.
    if oldest is not None and oldest > cutoff:
        print(f"[LessWrong/api] the {limit}-post ceiling reaches back only to "
              f"{oldest.date()}, but the window starts {cutoff.date()} — "
              f"posts before {oldest.date()} were not offered",
              file=sys.stderr)

    print(f"[LessWrong/api] {len(wanted)} posts in "
          f"{cutoff.date()}..{until.date()}; fetching bodies", flush=True)

    articles, failed = [], 0
    for i, p in enumerate(wanted):
        if i:
            time.sleep(REQUEST_PAUSE)
        try:
            md = _get(p["path"])
        except Exception as exc:
            failed += 1
            print(f"[LessWrong/api] {p['title'][:48]}: {type(exc).__name__}",
                  file=sys.stderr)
            continue
        # The canonical /posts/<id>/<slug> URL, so a post stored through this
        # route is the same row GraphQL would have written — not a second copy
        # under an /api/ path.
        m = _HTML_URL.search(md)
        url = BASE + m.group(1) if m else BASE + p["path"].replace("/api", "", 1)
        articles.append({
            "source": "LessWrong",
            "url": url,
            "title": p["title"],
            "content": _body_after_header(md),
            "author": p["author"],
            "published_at": p["published_at"],
        })

    if failed:
        print(f"[LessWrong/api] {failed} post(s) could not be read",
              file=sys.stderr)
    if wanted and not articles:
        raise SourceUnavailable("LessWrong", RuntimeError("no post bodies read"))
    return articles
