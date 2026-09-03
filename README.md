# mindspaceai

> A research radar for AI: cluster web discourse and arXiv papers, score each cluster by how nameable it is, and map the result.

If you follow AI research, the ideas that end up mattering most often have no name yet: a dozen papers and posts are circling the same concept in different vocabularies, months before anyone coins the term everyone then uses. mindspaceai is built to find those.

Each week it scrapes AI discourse and arXiv, clusters the documents with sentence embeddings, UMAP and HDBSCAN, and scores every cluster by its *name gap*: how tightly its documents agree on one concept versus how little vocabulary they share. High-gap clusters are candidate **proto-paradigms**, concepts that exist in the discourse before they have a settled name. The output is a star-map of the idea space you can pan through week by week to watch clusters form and get named.

---

## The three views

| view | window | clustered on | named by | built here? |
|---|---|---|---|---|
| **week** | 7 days | its own projection | TF-IDF top terms | ✅ `mindspace week` |
| **quarter** | 3 months, each week highlighted in turn | one frozen projection | TF-IDF, or model panel → synthesis → distillation | ✅ `mindspace quarter` |
| **arxiv** | 6 months | its own projection | TF-IDF top terms | ✅ `mindspace arxiv` |

They read different clock speeds. Papers move in months, discourse moves in days, and the interesting thing is what crosses between them. All three cluster the same way (embeddings, UMAP, HDBSCAN) and differ in window, corpus, and how the clusters are named.

`week` and `quarter` are independent: the week stands on its own map so its structure fills the sphere, while the quarter pins every week onto coordinates that never move. The rendered page that presents these is a separate project; this repo's deliverable is the data pack.

The quarter comes in two flavours from the same clustering. Run it alone and every cluster carries its TF-IDF top terms, free and enough to read the map. Add the naming pass and the highest-gap clusters of **each** week get a panel-and-synthesis name instead, so moving through the weeks shows the names changing as well as the constellations.

---

## How it works

```
scrape → embed → cluster → score → name → export
```

1. **Scrape:** RSS, LessWrong, Alignment Forum, Hacker News, GitHub Trending, HuggingFace Papers, and X (via Grok's `x_search`). Every window is a half-open `[start, end)` range anchored on a date, so any past week can be collected on its own and re-collected safely.
2. **Embed:** `openai/text-embedding-3-small` via OpenRouter, 1536 dimensions, incremental and resumable.
3. **Cluster:** UMAP then HDBSCAN. For the quarter, UMAP is fitted **once over the whole corpus** and frozen.
4. **Score:** each weekly cluster gets a lexical (TF-IDF) and a semantic (embedding) coherence score; the gap between their ranks is the signal.
5. **Name:** the highest-gap clusters go to a panel of four models, whose descriptions are synthesised into one "attractor" sentence and distilled into a short label.
6. **Export:** tensors and metadata for the TensorFlow Embedding Projector.

[PIPELINE.md](PIPELINE.md) walks through each step with the reasoning and the measurements behind the settings.

---

## Quickstart

```bash
cp .env.example .env          # OPENROUTER_API_KEY, XAI_API_KEY
uv sync

uv run python -m mindspace                     # the command list

uv run python -m mindspace scrape --week 1     # a week into the database
uv run python -m mindspace embed               # vectors for what is new

uv run python -m mindspace week                # the weekly view (free)

uv run python -m mindspace arxiv scrape --start 2026-02 --end 2026-08
uv run python -m mindspace arxiv embed         # papers have their own database
uv run python -m mindspace arxiv               # the six-month paper view

uv run python -m mindspace quarter             # cluster the three-month map
uv run python -m mindspace export              # tensors + HTML, TF-IDF names

# optional: name the highest-gap clusters of every week, then re-export
uv run python -m mindspace fable --all-weeks   # panel + synthesis   [paid]
uv run python -m mindspace label               # cheap labels        [paid]
uv run python -m mindspace export
```

`fable` and `label` write names keyed to the cluster ids of the last `quarter` run, so they belong between `quarter` and `export` and nowhere else. Names are stored per week, because cluster ids restart with every weekly clustering: cluster 3 in May and cluster 3 in August are unrelated. `fable --week YYYY-MM-DD` names a single week; the run is resumable and skips weeks already named.

Leaving both naming steps out is a supported outcome, not a half-finished one: every view falls back to TF-IDF terms, and the whole quarter costs nothing.

Backfilling is week by week (`scrape --week 2`, `--week 3`, and so on) because several sources cannot be asked for a wide historical range in one call.

---

## What things cost

Every run records what it actually spent, taken from each API's own usage figures rather than a price list, and appended to `data/cost/`:

| step | typical |
|---|---|
| `scrape` (a week of X via Grok) | ~$1.30 |
| `embed` | ~$0.01 per 1,000 documents |
| `fable` (one week, 10–15 clusters) | ~$0.55 |
| `fable --all-weeks` (11 weeks, 116 clusters) | ~$6.15 |
| `label` | ~$0.25 |
| `arxiv embed` (six months of papers) | ~$0.30 |
| everything else | free, local computation |

---

## Layout

```
mindspace/          the pipeline
  cli.py            one entry point: python -m mindspace <command>
  sources/          one adapter per source
  export/           projector tensors, per view
scripts/            operator tools (db stats, a paid Grok probe)
data/               gitignored: the corpus, outputs per view, cost ledgers
```

`data/` holds paid scrapes that cannot be bought again, and `git clean -xdf` deletes ignored files, so it is worth a backup outside the repo. Set `MINDSPACE_DATA` to keep it somewhere else entirely.

---

## Design decisions worth knowing

### The projection is frozen

UMAP re-fitted on a shifted window rearranges the whole map, which makes it impossible to tell whether a region moved because the ideas moved or because the algorithm did. Fitting once over the corpus and caching the coordinates means a basin keeps its position; a window only chooses which points are lit.

### Windows are half-open and anchored on a date

`[start, end)` ranges chain without a document falling into two of them, and a date anchor (rather than "now") makes two runs on the same day identical, which matters because generated names are addressed to cluster ids, and a shifted window silently changes those ids.

### The name gap is a difference of ranks, not of values

TF-IDF cosines sit near 0 and embedding cosines near 0.4 for nearly every cluster, so a raw difference is positive for almost everything and separates nothing.

### One key, one bill

Every model call (embeddings, the panel, the synthesis, the labels) is routed through OpenRouter, so there is a single place for credentials to be wrong and a single ledger of what was spent.

---

## FAQ

**Is this a news feed or trend tracker?**
No. It deliberately ignores what is already trending. It looks for clusters of documents converging on one concept while using different words, which by definition are not yet news.

**What is the "name gap"?**
Per cluster, the difference between its lexical (TF-IDF) coherence rank and its semantic (embedding) coherence rank. A cluster whose documents mean the same thing but share almost no vocabulary ranks high, and those are the ones worth watching.

**Do I need paid API keys?**
Scraping X (via Grok) and embedding cost money; everything else is local computation. The naming pass (`fable` and `label`) is optional and paid. A full quarter with TF-IDF cluster names costs nothing beyond the scrape and embed. See [What things cost](#what-things-cost).

**Where is the visualization?**
This repo produces the data pack: tensors and metadata for the TensorFlow Embedding Projector. The rendered star-map page that presents it is a separate project.

**How far back can I collect?**
Any past week, one at a time. Windows are date-anchored and half-open, so re-collecting a week is safe and produces the same result.

**Can I use it for a domain other than AI?**
The pipeline (`scrape → embed → cluster → score → name`) is generic. The sources in `config.yaml` and the stopword and prompt tuning are AI-specific; swap the sources and retune and it works elsewhere.

---

Part of [Eigenform](https://github.com/EigenformAI) · licensed [MIT](LICENSE)
