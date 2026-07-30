# mindspace

A pipeline for surfacing cutting-edge AI concepts from across the web, clustering them by semantic similarity, and visualising the emerging "basins of attraction" in the idea space.

The goal isn't news — it's finding **proto-paradigms**: the conceptual frontier before it becomes a paper or a trend.

---

## How it works

```
scrape → expand links → dedup → embed → cluster → visualise
```

1. **Scrape** — pulls from RSS feeds, LessWrong, Alignment Forum, Hacker News, GitHub Trending, HuggingFace Papers, and X (via Grok live search with conceptual prompts)
2. **Expand links** — follows arxiv and Reddit links found inside scraped content to fetch full paper abstracts and discussion threads
3. **Dedup** — normalises URLs (arxiv `/pdf/` → `/abs/`, strips UTM params, etc.) and removes content-hash duplicates
4. **Embed** — encodes all articles into semantic vectors using `sentence-transformers`
5. **Cluster** — UMAP dimensionality reduction + HDBSCAN clustering, auto-labelled with TF-IDF
6. **Visualise** — interactive Plotly scatter plot saved to `output/`

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

# scraping only (lighter)
pip install -r requirements-scrape.txt

# full pipeline including embedding/clustering
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your xAI API key (needed for X/Twitter search):

```bash
cp .env.example .env
# edit .env and set XAI_API_KEY=xai-...
```

---

## Usage

### Full pipeline

```bash
python3 pipeline.py
```

### Scrape only (no embedding/clustering)

```bash
python3 pipeline.py --scrape-only
```

### Expand links found in scraped content

Fetches arxiv abstracts and Reddit threads linked from articles. Safe to re-run.

```bash
# preview what would be fetched
python3 expand_links.py --dry-run

# fetch links from all sources
python3 expand_links.py

# only follow links found in tweets (most signal-dense)
python3 expand_links.py --source "Twitter/Grok"

# follow two levels deep (can be large — use --limit)
python3 expand_links.py --depth 2 --limit 100
```

### Deduplicate the database

```bash
python3 -c "import db; n = db.dedup_articles(); print(f'removed {n}, remaining:', db.article_count())"
```

### Browse scraped content

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('mindspace.db')
cur = None
for src, title, url in conn.execute('SELECT source, title, url FROM articles ORDER BY source, scraped_at DESC'):
    if src != cur:
        print(f'\n\n=== {src} ==='); cur = src
    print(f'  {title[:90]}')
" | less
```

### Article counts by source

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('mindspace.db')
for r in conn.execute('SELECT source, count(*) FROM articles GROUP BY source ORDER BY count(*) DESC').fetchall():
    print(f'{r[1]:4d}  {r[0]}')
"
```

---

## Configuration

All behaviour is controlled by `config.yaml`:

| Section | Key | Description |
|---|---|---|
| `lookback_days` | — | How far back to scrape (default: 7) |
| `sources.rss` | `name`, `url` | RSS feeds to scrape |
| `sources.lesswrong` | `limit` | Posts to fetch from LessWrong |
| `sources.alignment_forum` | `limit` | Posts from Alignment Forum |
| `sources.hackernews` | `queries`, `limit_per_query` | HN Algolia search terms |
| `sources.hf_papers` | `limit` | HuggingFace Papers count |
| `sources.twitter` | `prompts`, `limit_per_prompt` | Grok search prompts (see below) |
| `embedding.model` | — | Sentence-transformer model name |
| `clustering` | — | UMAP + HDBSCAN parameters |

### Grok search prompts

The Twitter/X scraper is prompt-driven rather than handle-driven. Each prompt is a natural-language description of the kind of post you're looking for. The current prompts target:

- **naming-moment** — researchers mid-thought trying to name an emerging idea
- **cross-domain-unification** — the same structural principle appearing across multiple AI domains
- **inference-time-frontier** — what AI can do at inference time without retraining
- **empirical-mysteries** — findings researchers can't explain with current theory

Edit or add prompts in `config.yaml` under `sources.twitter.prompts`.

---

## Data

All scraped and extracted content is stored in `mindspace.db` (SQLite). The schema:

```
articles(id, source, url, title, content, author, published_at, scraped_at, embedding)
```

Source values: `Simon Willison`, `LessWrong`, `Alignment Forum`, `HackerNews`, `GitHub Trending`, `HuggingFace Papers`, `Twitter/Grok`, `web/arxiv`, `web/reddit`, `web/extracted`

The database is append-only by default — re-running scrapes will not overwrite existing articles.

---

## Files

```
pipeline.py          — main orchestration script
expand_links.py      — link extraction and fetching
db.py                — SQLite interface, URL normalisation, dedup
embed.py             — sentence-transformer embeddings
cluster.py           — UMAP + HDBSCAN clustering
viz.py               — Plotly visualisation
config.yaml          — all configuration
scrapers/
  rss.py             — RSS feed scraper
  lesswrong.py       — LessWrong/Alignment Forum GraphQL scraper
  hackernews.py      — HN Algolia API scraper
  github_trending.py — GitHub Trending scraper
  hf_papers.py       — HuggingFace Papers scraper
  grok_twitter.py    — xAI Grok Responses API + x_search tool
```
