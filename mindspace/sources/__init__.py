import html
import re

class SourceUnavailable(Exception):
    """A source could not be reached — distinct from a source with nothing new.

    Both used to end as "0 items", so a source going dark looked exactly like a
    quiet week. On a pipeline run weekly against a fixed set of feeds, that is
    how a source disappears for a month before anyone notices, and the only
    evidence is clusters that quietly stop mentioning it.
    """

    def __init__(self, source: str, cause: Exception):
        super().__init__(f"{source}: {type(cause).__name__}")
        self.source = source
        self.cause = cause


_INVISIBLE_BLOCKS = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.S | re.I)


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = _INVISIBLE_BLOCKS.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_URLS = re.compile(r"https?://\S+|www\.\S+")


def text_for_tfidf(title: str, content: str) -> str:
    """Title and body as word statistics should see them: no markup, no links.

    Separate from embed_text on purpose. The embedding model is untroubled by a
    URL, and rewriting what it is given would mean documents embedded before
    and after this change were prepared differently — and, because the frozen
    UMAP anchor is keyed to those vectors, would expire every generated name.
    Only the lexical side needs this, so only the lexical side gets it.
    """
    body = _URLS.sub(" ", strip_html(content or ""))
    return f"{title or ''} {re.sub(r'\\s+', ' ', body)}".strip()


def embed_text(title: str, content: str) -> str:
    """Combine title + truncated content for embedding."""
    title = (title or "").strip()
    body = strip_html(content or "")[:800]
    return f"{title}\n\n{body}".strip()
