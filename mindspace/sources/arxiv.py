import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

API = "https://export.arxiv.org/api/query"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

MIN_INTERVAL = 5.0        # seconds between requests
PAGE_SIZE = 500           # well under the 2,000 ceiling
MAX_ATTEMPTS = 5          # per page, before giving up loudly
RATE_LIMIT_BACKOFF = 60   # a 429 needs minutes, not seconds
USER_AGENT = "mindspace/0.1 (research crawler; contact: eigenform.ai)"

_last_request = 0.0


def _throttle() -> None:
    global _last_request
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _get(params: dict) -> ET.Element:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return ET.fromstring(resp.read())
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                wait = RATE_LIMIT_BACKOFF * (attempt + 1)
                print(f"    [arXiv] 429, waiting {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            time.sleep(MIN_INTERVAL * (attempt + 1))
        except (urllib.error.URLError, ET.ParseError, TimeoutError) as exc:
            last_error = exc
            time.sleep(MIN_INTERVAL * (attempt + 1))

    raise RuntimeError(
        f"arXiv request failed after {MAX_ATTEMPTS} attempts ({last_error}): {url}\n"
        "Papers fetched before the failure are already stored — re-run to resume."
    ) from last_error


def _bare_id(entry_id: str) -> str:
    """http://arxiv.org/abs/2406.12345v2 -> 2406.12345

    The version is stripped so cross-listed duplicates and later revisions
    collapse onto one paper.
    """
    tail = entry_id.rstrip("/").rsplit("/", 1)[-1]
    return tail.split("v")[0] if "v" in tail else tail


def _month_slices(lo: date, hi: date) -> list[tuple[date, date]]:
    out, cursor = [], lo
    last = hi - timedelta(days=1)
    while cursor <= last:
        nxt = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((cursor, min(last, nxt - timedelta(days=1))))
        cursor = nxt
    return out


def _bounds(first: date, last: date) -> tuple[str, str]:
    """Inclusive days -> the minute range submittedDate expects."""
    start = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)
    end = datetime(last.year, last.month, last.day, tzinfo=timezone.utc) + timedelta(days=1)
    return start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")


def _to_article(entry: ET.Element, seen: set) -> dict | None:
    def text(tag: str) -> str:
        el = entry.find(tag, NS)
        return (el.text or "").strip() if el is not None else ""

    published = text("a:published")      # first version — never a:updated
    abstract = " ".join(text("a:summary").split())
    entry_id = text("a:id")
    if not (published and abstract and entry_id):
        return None

    bare = _bare_id(entry_id)
    if bare in seen:
        return None
    seen.add(bare)

    authors = [(a.findtext("a:name", "", NS) or "").strip()
               for a in entry.findall("a:author", NS)]
    return {
        "source": "arXiv",
        "url": f"https://arxiv.org/abs/{bare}",
        "title": " ".join(text("a:title").split()),
        "content": abstract,
        "author": ", ".join(a for a in authors[:3] if a),
        "published_at": published,
    }


def _fetch(category: str, first: date, last: date, seen: set,
           limit: int | None) -> list[dict]:
    lo, hi = _bounds(first, last)
    query = f"cat:{category} AND submittedDate:[{lo} TO {hi}]"
    articles: list[dict] = []
    start = fetched = 0
    expected: int | None = None

    while True:
        root = _get({"search_query": query, "start": start,
                     "max_results": PAGE_SIZE,
                     "sortBy": "submittedDate", "sortOrder": "ascending"})

        if expected is None:
            el = root.find("opensearch:totalResults", NS)
            expected = int(el.text.strip()) if el is not None and (el.text or "").strip() else None
            if expected is not None and expected >= 30_000:
                raise RuntimeError(
                    f"{category} {first}..{last}: {expected} results exceeds "
                    "arXiv's 30,000 cap. Slice finer than a month.")

        entries = root.findall("a:entry", NS)
        for entry in entries:
            article = _to_article(entry, seen)
            if article is not None:
                articles.append(article)
                if limit and len(articles) >= limit:
                    return articles
        fetched += len(entries)

        # Trust the reported total, not the page length: arXiv serves short and
        # empty pages mid-result-set, and treating one as the end silently
        # truncates the range, leaving a hole nothing downstream can detect.
        if expected is None:
            if not entries:
                return articles
        elif fetched >= expected:
            return articles
        elif not entries:
            raise RuntimeError(
                f"{category} {first}..{last}: empty page at offset {start} but "
                f"only {fetched} of {expected} results retrieved. Refusing to "
                "report a partial range as complete.")

        start += PAGE_SIZE


def scrape_arxiv(categories: list[str], since: date, until: date,
                 limit_per_slice: int | None = None) -> list[dict]:
    """Every abstract in `categories` published in the half-open [since, until).

    Deduped on the bare arXiv id, so a paper cross-listed across two of the
    requested categories is returned once.
    """
    lo, hi = since, until
    slices = _month_slices(lo, hi)

    seen: set = set()
    out: list[dict] = []
    for category in categories:
        for first, last in slices:
            got = _fetch(category, first, last, seen, limit_per_slice)
            out.extend(got)
            print(f"[arXiv] {category} {first}..{last}: {len(got)} new "
                  f"({len(out)} total)", flush=True)
    return out
