import html
import re


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def embed_text(title: str, content: str) -> str:
    """Combine title + truncated content for embedding."""
    title = (title or "").strip()
    body = strip_html(content or "")[:800]
    return f"{title}\n\n{body}".strip()
