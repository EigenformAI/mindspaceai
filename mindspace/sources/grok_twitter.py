import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from .. import paths

_COST_PATH = paths.COST / "grok_cost.json"

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
    """The JSON array, wherever it sits in the text.

    At limit=15 the model returned bare JSON and a straight loads() worked. At
    limit=50 it reasons longer and narrates before the array ("I'll search X
    for posts… Searching more specifically…"), so parsing from character 0
    fails — and since the instructions were followed *eventually*, the array is
    in there, paid for, and thrown away. Scanning for it recovers those posts.
    """
    raw = raw.strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("["):
                raw = part
                break

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    # Scan every '[' for a parseable array of objects; keep the largest, since
    # narration can contain bracketed asides that also parse.
    decoder = json.JSONDecoder()
    best: list[dict] = []
    i = 0
    while True:
        j = raw.find("[", i)
        if j == -1:
            break
        try:
            value, end = decoder.raw_decode(raw, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if (isinstance(value, list) and value
                and all(isinstance(x, dict) for x in value)
                and len(value) > len(best)):
            best = value
        i = max(end, j + 1)

    if not best:
        print(f"[Grok] no JSON array found — snippet: {raw[:300]}", file=sys.stderr)
    return best


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


def _usage_row(resp, label: str, n_posts: int) -> dict:
    """Real cost and search count from xAI's own accounting on the response.

    Same extraction as probe_grok.py: the tool-usage details sometimes arrive
    as a plain dict, where getattr silently returns None.
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return {"label": label, "posts": n_posts}
    det = getattr(u, "server_side_tool_usage_details", None)
    if isinstance(det, dict):
        calls = det.get("x_search_calls")
    else:
        calls = getattr(det, "x_search_calls", None)
    if calls is None and hasattr(u, "model_dump"):
        calls = (u.model_dump().get("server_side_tool_usage_details")
                 or {}).get("x_search_calls")
    return {"label": label,
            "tool_calls": calls,
            "posts": n_posts,
            "output_tokens": getattr(u, "output_tokens", None),
            # 1 USD = 10^10 ticks (docs.x.ai/developers/cost-tracking): the
            # billed amount after cache discounts. Dividing by 1e9 instead
            # overstated every cost tenfold against the xAI console.
            "cost_usd": getattr(u, "cost_in_usd_ticks", 0) / 1e10}


def _write_ledger(rows: list[dict], from_date: str, to_date: str) -> None:
    """Append this run to output/grok_cost.json — append, never overwrite:
    each run is a real dollar amount, and replacing the file would erase the
    record of money already spent."""
    runs = []
    if _COST_PATH.exists():
        try:
            runs = json.loads(_COST_PATH.read_text()).get("runs", [])
        except (json.JSONDecodeError, AttributeError):
            runs = []
    runs.append({
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": f"{from_date}..{to_date}",
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 4),
        "prompts": rows,
    })
    _COST_PATH.parent.mkdir(exist_ok=True)
    _COST_PATH.write_text(json.dumps({
        "total_usd": round(sum(r["cost_usd"] for r in runs), 4),
        "runs": runs,
    }, indent=2))


def _call_prompt(client, label: str, prompt_text: str,
                 from_date: str, to_date: str,
                 limit: int) -> tuple[list[dict], dict]:
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
            tools=[{"type": "x_search",
                    "from_date": from_date, "to_date": to_date}],
        )
    except Exception as exc:
        print(f"[Grok] '{label}' — API error: {exc}", file=sys.stderr)
        return [], {"label": label, "error": str(exc)[:200]}

    text = _extract_text(resp)
    posts = _parse_posts(text)
    row = _usage_row(resp, label, len(posts))

    if not posts:
        return _citations_to_articles(resp), row

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
    return articles, row


def scrape_grok_twitter(prompts: list[dict], api_key: str,
                        lookback_days: int = 7,
                        limit_per_prompt: int = 15,
                        until: datetime | None = None) -> list[dict]:
    """prompts: list of {"label": str, "text": str}"""
    try:
        from openai import OpenAI
    except ImportError:
        print("[Grok] openai not installed — run: pip install openai", file=sys.stderr)
        return []

    if not prompts:
        print("[Grok] no prompts configured", file=sys.stderr)
        return []

    # x_search bounds are dates, not timestamps, and both ends are inclusive —
    # a post published exactly on a boundary date can be returned by two
    # adjacent weekly windows. The DB's URL upsert collapses those duplicates.
    until = until or datetime.now(timezone.utc)
    from_date = (until - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date = until.strftime("%Y-%m-%d")
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    all_articles: list[dict] = []
    seen_urls: set[str] = set()
    usage_rows: list[dict] = []

    for i, p in enumerate(prompts, 1):
        label = p.get("label", f"prompt-{i}")
        text = p.get("text", "").strip()
        if not text:
            continue
        print(f"[Grok] ({i}/{len(prompts)}) {label}")
        articles, row = _call_prompt(client, label, text, from_date, to_date,
                                     limit_per_prompt)
        usage_rows.append(row)
        fresh = [a for a in articles if a["url"] not in seen_urls]
        seen_urls.update(a["url"] for a in fresh)
        print(f"         → {len(fresh)} posts")
        all_articles.extend(fresh)

    if usage_rows:
        _write_ledger(usage_rows, from_date, to_date)
        spent = sum(r.get("cost_usd") or 0 for r in usage_rows)
        searches = sum(r.get("tool_calls") or 0 for r in usage_rows)
        print(f"[Grok] {from_date}..{to_date}: ${spent:.2f} across "
              f"{searches} x_search calls — logged to {_COST_PATH}")

    return all_articles
