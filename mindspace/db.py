import hashlib
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse, urlencode, parse_qsl, urlunparse

import numpy as np
from . import paths

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

DB_PATH = paths.DB


def use(path) -> None:
    """Point every operation in this module at another database file.

    Both databases share this schema exactly, so all the helpers below work on
    either. Called once at the start of a command — the arXiv view and the
    discourse views never run in the same process, so there is no window in
    which the two could be confused.
    """
    global DB_PATH
    DB_PATH = Path(path)

_SCHEMA = """
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


def normalize_published(value) -> str | None:
    """One ISO 8601 UTC string, or None when there is no usable date.

    The scrapers hand back four different shapes:

        2026-07-29T22:17:35.487Z        LessWrong, HuggingFace Papers
        2026-07-29T15:01:35+00:00       RSS feeds, Hacker News
        Fri, 24 Jul 2026 08:06:37 GMT   Twitter/Grok
        None                            nothing usable

    SQLite compares this column as text, and "F" sorts after every digit, so an
    unnormalised RFC 2822 row passes a lower bound and fails an upper one:

        'Fri, 24 Jul 2026 08:06:37 GMT' >= '2026-05-04'  ->  1
        'Fri, 24 Jul 2026 08:06:37 GMT' <  '2026-08-03'  ->  0

    Every Grok row would therefore vanish from any date range without a word.
    Normalising on write makes the column one shape that sorts correctly, which
    is what lets `get_recent_articles` filter in SQL at all.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def upsert_article(source: str, url: str, title: str, content: str,
                   author: str = None, published_at: str = None) -> str:
    url = normalize_url(url)
    aid = url_id(url)
    now = datetime.now(timezone.utc).isoformat()
    published_at = normalize_published(published_at)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO articles
                (id, source, url, title, content, author, published_at, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title        = excluded.title,
                content      = excluded.content,
                author       = COALESCE(excluded.author, articles.author),
                published_at = COALESCE(articles.published_at, excluded.published_at)
            WHERE length(COALESCE(excluded.content, '')) >
                  length(COALESCE(articles.content, ''))
            """,
            (aid, source, url, title, content, author, published_at, now),
        )
    return aid


def get_articles_without_embeddings() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, content FROM articles WHERE embedding IS NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def save_embedding(article_id: str, embedding: np.ndarray):
    blob = embedding.astype(np.float32).tobytes()
    with get_conn() as conn:
        conn.execute(
            "UPDATE articles SET embedding = ? WHERE id = ?", (blob, article_id)
        )


def get_recent_articles(days: int) -> list[dict]:
    with get_conn() as conn:
        undated = conn.execute(
            "SELECT COUNT(*) FROM articles "
            "WHERE embedding IS NOT NULL AND (published_at IS NULL OR published_at = '')"
        ).fetchone()[0]
        if undated:
            print(f"[db] {undated} embedded article(s) have no publication date "
                  "and are excluded from every window")

        rows = conn.execute(
            """
            SELECT id, source, url, title, content, author, published_at, embedding
            FROM articles
            WHERE published_at >= datetime('now', ?)
              AND published_at IS NOT NULL
              AND embedding IS NOT NULL
            ORDER BY published_at DESC
            """,
            (f"-{days} days",),
        ).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        if d["embedding"]:
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy()
        results.append(d)
    return results


def get_all_embedded() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, source, url, title, content, author, published_at, embedding
            FROM articles
            WHERE embedding IS NOT NULL
              AND published_at IS NOT NULL AND published_at != ''
            ORDER BY published_at ASC
            """
        ).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy()
        out.append(d)
    return out


def article_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]


def dedup_articles(dry_run: bool = False) -> int:
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
