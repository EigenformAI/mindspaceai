from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

API = "https://openrouter.ai/api/v1/chat/completions"
MAX_ATTEMPTS = 4

PROMPTS = {
    "label": (
        "Below are passages from research papers that were grouped together by "
        "an automated clustering process.\n\n"
        "Give five short names for the common concept they share. Use terminology "
        "the field already uses.\n\n"
        "Reply with exactly five lines. Each line is one name, nothing else — no "
        "numbering, no explanation, no preamble.\n\n"
        "PASSAGES:\n{passages}"
    ),
    "compress": (
        "Below are passages from research papers that were grouped together by "
        "an automated clustering process.\n\n"
        "Compress what they have in common into a single term of two to four "
        "words. Prefer the term that most reduces the effort of describing all "
        "of these at once. If the field has no settled name for this yet, coin "
        "one rather than falling back on a vague existing label.\n\n"
        "Give five candidates. Reply with exactly five lines. Each line is one "
        "term, nothing else — no numbering, no explanation, no preamble.\n\n"
        "PASSAGES:\n{passages}"
    ),
}


@dataclass(slots=True)
class Naming:
    cluster_id: int
    window: str
    model: str
    lab: str
    prompt_version: str
    names: list[str]
    raw: str = ""


def build_passages(cluster: dict, n: int = 8, chars: int = 400) -> str:
    return "\n\n".join(
        f"[{i + 1}] {d['text'][:chars]}"
        for i, d in enumerate(cluster["top_documents"][:n])
    )


def _clean(line: str) -> str:
    line = line.strip()
    # Bold before bullets: the bullet pattern matches a leading "*", so running
    # it first eats one asterisk of "**Name**" and leaves "*Name**" behind.
    line = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line)          # bold markers
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)   # bullets and numbering
    return line.strip(' "\'.')


def parse_names(text: str, want: int = 5) -> list[str]:
    """Pull the candidate names out of a reply.

    Reasoning models prepend their thinking ("Okay, the user wants...").
    Anything sentence-length is not a two-to-four word term, so length is a
    reliable filter — no need to detect the reasoning itself.
    """
    names = []
    for raw in text.splitlines():
        name = _clean(raw)
        if not name or len(name.split()) > 8 or name.endswith(":"):
            continue
        if name.lower() not in {n.lower() for n in names}:
            names.append(name)
    return names[:want]


def name_cluster(
    cluster: dict,
    model: str,
    api_key: str,
    prompt_version: str = "label",
    window: str = "",
) -> Naming:
    prompt = PROMPTS[prompt_version].format(passages=build_passages(cluster))
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0,          # same cluster, same names — the backtest needs it
        }
    ).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read())
            text = (payload["choices"][0]["message"].get("content") or "").strip()
            return Naming(
                cluster_id=cluster["cluster_id"],
                window=window or cluster.get("window", ""),
                model=model,
                lab=model.split("/")[0],
                prompt_version=prompt_version,
                names=parse_names(text),
                raw=text[:2000],
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code != 429 and 400 <= exc.code < 500:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, KeyError, TimeoutError) as exc:
            last_error = exc
        time.sleep(5 * (attempt + 1))

    raise RuntimeError(f"naming failed after {MAX_ATTEMPTS} attempts: {last_error}")


WORKERS = 4    # concurrent naming calls; 429s are retried inside name_cluster


def name_many(
    clusters: list[dict],
    model: str,
    api_key: str,
    variants: list[str],
    window: str,
    done: set[tuple[int, str, str]],
    on_result,
) -> tuple[int, int]:
    tasks = [
        (cluster, variant)
        for cluster in clusters
        for variant in variants
        if (cluster["cluster_id"], model, variant) not in done
    ]
    cached = len(clusters) * len(variants) - len(tasks)
    if not tasks:
        return 0, cached

    named = 0
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        futures = {
            pool.submit(name_cluster, cluster, model, api_key, variant, window):
                (cluster, variant)
            for cluster, variant in tasks
        }
        for future in as_completed(futures):
            cluster, variant = futures[future]
            on_result(cluster, variant, future.result())
            named += 1
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return named, cached


class NamingStore:
    """Cached by (cluster_id, model, prompt_version) — a re-run costs nothing."""

    def __init__(
        self, root: Path | str = "data", window: str = "", algo_version: str = "v1"
    ) -> None:
        self.path = Path(root) / "names" / f"{algo_version}_{window}.jsonl"

    def existing(self) -> set[tuple[int, str, str]]:
        if not self.path.exists():
            return set()
        keys = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["cluster_id"], r["model"], r["prompt_version"]))
        return keys

    def append(self, naming: Naming) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(naming), ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
