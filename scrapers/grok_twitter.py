import json
import sys
from datetime import datetime, timedelta, timezone

_INSTRUCTIONS = (
    "You are a research assistant that searches X (Twitter) for relevant posts. "
    "When asked, return ONLY a valid JSON array with no markdown fences and no prose. "
    'Each element: {"url": str, "author": str (handle, no @), "content": str, "published_date": str|null}'
)


def _extract_text(resp) -> str:
    if text := getattr(resp, "output_text", None):
        return text
    for item in getattr(resp, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", "") == "output_text":
                return block.text or ""
    return ""


def _parse_posts(raw: str) -> list[dict]:
    raw = raw.strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("["):
                raw = part
                break
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[Grok] JSON parse error: {exc} — snippet: {raw[:300]}", file=sys.stderr)
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def _citations_to_articles(resp) -> list[dict]:
    """Fallback: build stub articles from raw citation URLs if JSON parsing fails."""
    articles = []
    for url in getattr(resp, "citations", []) or []:
        url = str(url)
        if "x.com" not in url and "twitter.com" not in url:
            continue
        parts = url.rstrip("/").split("/")
        author = parts[-3] if len(parts) >= 3 and "status" in parts else ""
        articles.append({
            "source": "Twitter/Grok",
            "url": url,
            "title": f"@{author}: (tweet)" if author else url,
            "content": "",
            "author": author,
            "published_at": None,
        })
    return articles


def _call_prompt(client, label: str, prompt_text: str,
                 from_date: str, limit: int) -> list[dict]:
    full_prompt = (
        f"{prompt_text}\n\n"
        f"Return up to {limit} posts. "
        "Return ONLY a JSON array, nothing else."
    )
    try:
        resp = client.responses.create(
            model="grok-4.5",
            instructions=_INSTRUCTIONS,
            input=[{"role": "user", "content": full_prompt}],
            tools=[{"type": "x_search", "from_date": from_date}],
        )
    except Exception as exc:
        print(f"[Grok] '{label}' — API error: {exc}", file=sys.stderr)
        return []

    text = _extract_text(resp)
    posts = _parse_posts(text)

    if not posts:
        return _citations_to_articles(resp)

    articles = []
    seen: set[str] = set()
    for post in posts:
        url = str(post.get("url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        author = str(post.get("author", "")).lstrip("@")
        content = str(post.get("content", ""))
        articles.append({
            "source": "Twitter/Grok",
            "url": url,
            "title": f"@{author}: {content[:120]}",
            "content": content,
            "author": author,
            "published_at": post.get("published_date"),
        })
    return articles


def scrape_grok_twitter(prompts: list[dict], api_key: str,
                        lookback_days: int = 7,
                        limit_per_prompt: int = 15) -> list[dict]:
    """prompts: list of {"label": str, "text": str}"""
    try:
        from openai import OpenAI
    except ImportError:
        print("[Grok] openai not installed — run: pip install openai", file=sys.stderr)
        return []

    if not prompts:
        print("[Grok] no prompts configured", file=sys.stderr)
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    all_articles: list[dict] = []
    seen_urls: set[str] = set()

    for i, p in enumerate(prompts, 1):
        label = p.get("label", f"prompt-{i}")
        text = p.get("text", "").strip()
        if not text:
            continue
        print(f"[Grok] ({i}/{len(prompts)}) {label}")
        articles = _call_prompt(client, label, text, from_date, limit_per_prompt)
        fresh = [a for a in articles if a["url"] not in seen_urls]
        seen_urls.update(a["url"] for a in fresh)
        print(f"         → {len(fresh)} posts")
        all_articles.extend(fresh)

    return all_articles
