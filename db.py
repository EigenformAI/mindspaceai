import hashlib
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlencode, parse_qsl, urlunparse


_UTM_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term"}


def normalize_url(url: str) -> str:
    """Canonicalize URLs to collapse common duplicate forms."""
    try:
        p = urlparse(url.strip())
    except Exception:
        return url
    # arxiv: pdf → abs, strip version suffix, strip .pdf extension
    if "arxiv.org" in p.netloc:
        path = re.sub(r"/pdf/", "/abs/", p.path)
        path = re.sub(r"v\d+$", "", path.rstrip(".pdf"))
        return urlunparse(("https", "arxiv.org", path, "", "", ""))
    # strip UTM and tracking params universally
    qs = [(k, v) for k, v in parse_qsl(p.query) if k.lower() not in _UTM_PARAMS]
    # strip fragment, normalize scheme to https
    return urlunparse(("https", p.netloc.lower(), p.path.rstrip("/") or "/",
                       p.params, urlencode(qs), ""))

DB_PATH = Path("mindspace.db")

_SCHEMA = """
-- `embedding` is kept for compatibility with mindspacelocal but is never
-- written here: embedding works on paragraphs, and a paragraph's vector cannot
-- live on an article row. Chunks and vectors are files — see run_embedding.
CREATE TABLE IF NOT EXISTS articles (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    url         TEXT UNIQUE NOT NULL,
    title       TEXT,
    content     TEXT,
    author      TEXT,
    published_at TEXT,
    scraped_at  TEXT NOT NULL,
    embedding   BLOB
);
"""


def url_id(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()[:16]


def content_hash(title: str, content: str) -> str:
    text = (title or "").strip().lower() + "||" + (content or "")[:500].strip().lower()
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(_SCHEMA)


def upsert_article(source: str, url: str, title: str, content: str,
                   author: str = None, published_at: str = None) -> str:
    url = normalize_url(url)
    aid = url_id(url)
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO articles
                (id, source, url, title, content, author, published_at, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (aid, source, url, title, content, author, published_at, now),
        )
    return aid


def article_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]


def dedup_articles(dry_run: bool = False) -> int:
    """Remove duplicate articles, keeping the one with the most content.
    Deduplicates on: (1) normalized URL already handled by schema;
    (2) content hash (same title+content snippet from different URLs).
    Returns number of rows removed.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, url, title, content FROM articles"
        ).fetchall()

    seen: dict[str, str] = {}  # content_hash -> id to keep
    to_delete: list[str] = []

    # sort longest content first so we keep the richest version
    rows_sorted = sorted(rows, key=lambda r: len(r["content"] or ""), reverse=True)
    for r in rows_sorted:
        ch = content_hash(r["title"], r["content"])
        if ch in seen:
            to_delete.append(r["id"])
        else:
            seen[ch] = r["id"]

    if not dry_run and to_delete:
        with get_conn() as conn:
            conn.executemany("DELETE FROM articles WHERE id = ?",
                             [(aid,) for aid in to_delete])
    return len(to_delete)

# ---------------------------------------------------------------------------
# Reading for the analysis pipeline.
#
# The collection stage above stores one row per article. The analysis stage
# works on paragraphs, and keeps them in files rather than here: chunks.jsonl
# row N has to describe vectors.npy row N, and clustering joins the two by
# position alone. A BLOB per row gives no such guarantee, and rebuilding a
# matrix from thousands of separate reads is slower than one np.load.
#
# So this is the boundary: articles live in SQLite, chunks and vectors live in
# files, and the projector files are the output.
# ---------------------------------------------------------------------------

def normalize_published(value) -> str | None:
    """One ISO timestamp, or None when a row carries no usable date.

    Measured in this database on 2026-07-30, four shapes in one column:

        2026-07-29T22:17:35.487Z        LessWrong, HF Papers
        2026-07-29T15:01:35+00:00       RSS, Hacker News
        Fri, 24 Jul 2026 08:06:37 GMT   Twitter/Grok
        NULL                            GitHub Trending, all 17 rows

    Chunks are partitioned by `published_at[:7]`, so an unparsed RFC 2822
    string would land in a month called "Fri, 29". Undated rows are dropped
    rather than dated from `scraped_at` — fetch time is not publication time,
    and treating it as such quietly redates the corpus.
    """
    if not value or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        return (datetime.fromisoformat(text.replace("Z", "+00:00"))
                .astimezone(timezone.utc).isoformat())
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def source_id_for(source: str) -> str:
    """'Twitter/Grok' -> 'twitter-grok'.

    Names the source, not the pipeline that fetched it. If Hacker News is ever
    fetched some other way, the id stays `hackernews` and its document share
    stays continuous; a prefix like `mindspace-` would break that history.
    """
    return re.sub(r"[^a-z0-9]+", "-", (source or "").lower()).strip("-")


def read_documents(month: str | None = None, skipped: dict | None = None) -> list[dict]:
    """Articles as the dicts `chunk_document` expects, oldest first.

    `month` is 'YYYY-MM'; None reads everything. Rows without a date or without
    content are counted into `skipped` rather than dropped in silence — a
    quietly discarded row reads as full coverage.
    """
    documents = []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, source, url, title, content, author, published_at FROM articles"
        ).fetchall()

    for row in rows:
        iso = normalize_published(row["published_at"])
        if iso is None:
            if skipped is not None:
                key = f"{row['source']}: no date"
                skipped[key] = skipped.get(key, 0) + 1
            continue
        if month and iso[:7] != month:
            continue
        text = (row["content"] or "").strip()
        if not text:
            if skipped is not None:
                key = f"{row['source']}: empty content"
                skipped[key] = skipped.get(key, 0) + 1
            continue
        documents.append({
            "external_id": row["id"],
            "source_id": source_id_for(row["source"]),
            "url": row["url"],
            "title": (row["title"] or "").strip(),
            "published_at": iso,
            "text": text,
            "author": (row["author"] or "").strip() or None,
        })

    documents.sort(key=lambda d: d["published_at"])
    return documents


def months_present() -> list[str]:
    """Every month with at least one dated article, oldest first."""
    with get_conn() as conn:
        rows = conn.execute("SELECT published_at FROM articles").fetchall()
    months = set()
    for row in rows:
        iso = normalize_published(row["published_at"])
        if iso:
            months.add(iso[:7])
    return sorted(months)
