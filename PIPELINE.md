# Mindspace AI Pipeline: Step-by-Step

The pipeline lives in `mindspace/`, operator scripts in `scripts/`, and everything it
collects or produces under `data/` (gitignored). The published page is a separate repo.

---

## Step 1 — Scrape

`mindspace/pipeline.py` pulls articles from multiple sources:

- RSS feeds (configurable in `config.yaml`)
- LessWrong & Alignment Forum (GraphQL API, with `after`/`before` date bounds)
- Hacker News (Algolia API, date-bounded, `points > 10` noise floor)
- GitHub Trending (scraper — no history; entries are dated by observation day)
- HuggingFace Papers (scraper, one curated list per day)
- X / Twitter (Grok API with the `x_search` tool)

Each article is stored in SQLite (`mindspace.db`) with title, content, URL, source, and its **publication** date, normalised to ISO 8601 UTC on write. Re-scrapes are safe: a URL already stored is only ever *improved* (longer content wins), never duplicated or degraded.

```bash
uv run python -m mindspace scrape --week 1     # the current week
uv run python -m mindspace scrape --week 5     # any past week, same counting
```

---

## Step 2 — Embed

Each article's title + content goes to `openai/text-embedding-3-small` **via OpenRouter**, returning a 1536-dimension vector, stored back in the DB.

Embedding is incremental and resumable: only rows without a vector are sent, progress is saved every 500 articles, and the real cost of every run (from the API's own usage figures, not an estimate) is appended to `data/cost/embed_cost.json`.

**Requires:** `OPENROUTER_API_KEY` in `.env` — the only key the whole pipeline needs.

---

## Step 3 — Cluster (spec steps 1 & 3)

`mindspace/pipeline.py` clusters twice, using the same process both times:

- **UMAP** — fitted **once over the whole corpus** and cached (`data/output/quarter/projection.npz`, reused as long as the corpus is unchanged). This is deliberate: UMAP re-fitted on a shifted window rearranges the whole map, which is exactly the "continuity between basins of attraction" problem the three-month anchor exists to solve. Fit once, and a basin keeps its coordinates forever; a window only chooses which points are shown.
- **HDBSCAN** — density clustering on the 2D coordinates: once over the **three-month background** (`modes.quarter.min_cluster_size: 15`) and once over **each week** separately (`modes.quarter.min_cluster_size_week: 8`). The two sizes differ because the windows differ by an order of magnitude.

Both windows are half-open `[start, today)`, measured from a **date** (not a timestamp), so every run on the same day sees identical windows and produces identical clusters — and `clusters.json` always describes the same grouping as `frames.json`. Documents published on the run day count from tomorrow.

Each cluster gets an automatic **TF-IDF keyword label** (e.g. `alignment · human · gpt-5 · value`).

**Output files:**
- `data/output/quarter/clusters.json` — this week's clusters (what the naming steps read)
- `data/output/quarter/name_gaps.json` — coherence scores per cluster (step 4)
- `data/output/quarter/frames.json` — clusters for every weekly slider stop
- `data/output/quarter/projection.npz` — the fixed 2D coordinates
- `data/output/quarter/background.html`, `data/output/quarter/slider.html` — the 2D views

---

## Step 4 — Score coherence (spec step 4)

`mindspace/coherence.py` gives every weekly cluster two mean-pairwise-cosine scores:

- **lexical** — cosine over a corpus-wide TF-IDF representation (do the documents share *words*?)
- **semantic** — cosine over the 1536-dim embeddings (do they share *meaning*?)

> Note: the original spec document had these two definitions the other way round. The code follows the meaning of the terms (and the spec's own interpretation table): TF-IDF dimensions *are* words, embeddings encode meaning. `coherence.measure_map: as-written` in `config.yaml` restores the literal reading.

The headline number is the **name gap**:

```
name_gap = rank(semantic) − rank(lexical)        (percentile ranks within the week)
```

Ranks, not raw differences — TF-IDF cosines sit near 0 and embedding cosines near 0.4 for every cluster, so a raw difference is positive for essentially everything and separates nothing.

A cluster with a high positive gap is *semantically tight but lexically scattered*: the documents describe the same thing in different vocabularies, with no settled term for it yet — a concept that exists before its name. A strongly negative gap is the opposite: one word, many meanings — a buzzword.

**Output:** `data/output/quarter/name_gaps.json`, sorted by gap, which is exactly the file step 5 selects from.

---

## Step 5 — Compress

`mindspace.compress --rank-by name-gap` takes the **top 15 clusters by name gap** (only positive gaps qualify) and, for each, sends titles and excerpts to 4 LLMs **in parallel** — all via OpenRouter on the one key:

- `openai/gpt-4o-mini`
- `x-ai/grok-4.5`
- `anthropic/claude-haiku-4.5`
- `google/gemini-2.5-flash`

with the name-gap prompt: describe plainly the shared referent these documents are circling *without a shared term*. All four compressions are saved to `data/output/quarter/name_gap_compressions.json` and `.md`.

---

## Step 6 — Synthesise

All 4 compressions per cluster are fed to **Claude Fable** (via OpenRouter), asked to name the single deeper "attractor" the four descriptions are circling — *"as precise, specific, and conceptually loaded as possible… think like a physicist naming a phenomenon."* One rich sentence per cluster.

---

## Step 7 — Label

Each Fable attractor is distilled by **Claude Haiku** into a 2–5 word noun phrase — the sidebar register: `Plausibility-verification gap`, `Simulacral competence`, `Competence Authority Decoupling`.

Saved to `data/output/quarter/name_gap_ai_labels.json`, keyed by cluster id. These labels are only ever valid for the clustering they were generated against — which is why they are a separate command, run only on a clustering that finished cleanly.

Separately, `mindspace/labels.py` gives **every** cluster at **every** weekly stop a cheap topical label (one Haiku call each, direct from keywords+titles, ≈ $0.25 for all ~270). Display surfaces prefer the Fable label where one exists and fall back to the cheap one, then to raw keywords.

---

## Step 8 — Visualise

Two views, answering different questions:

**TensorFlow Embedding Projector** (`data/output/quarter/projector/`) — the main view. `mindspace/export/projector.py` writes two tensors:

- `mindspaceai` (5005 × 1536) — the true semantic space. "Sphereize data" + PCA gives the spherical star-map look; nearest-neighbours are computed here.
- `mindspaceai — cluster map (UMAP 3D)` (5005 × 3) — the fixed cluster map; each island is one named cluster (turn Sphereize *off* for this one).

The stock projector build is patched (append-only; `index.html.upstream` keeps the pristine copy) with:
- a **week slider** — switches the per-week colour columns; the yellow→red→blue gradient encodes the name gap continuously (emerging → established → buzzword), background documents stay grey
- a **cluster sidebar** — Fable/Haiku names ranked by gap; click to see the full attractor and light up the cluster's points
- **hover cards** — mouse over any clustered point shows its cluster's name and description
- **in-diagram labels** — the toolbar "A" (3D labels mode) draws each cluster's name once, at its representative point
- collapsible side panels, clickable URLs, and a metadata card that hides the 22 per-week plumbing columns

**Plotly 2D** (`data/output/quarter/slider.html`) — the spec's step 5–6 rendered literally: the only view where the two windows *actually move* (documents enter and leave as the slider steps through 11 weeks) over coordinates that never change.

These files are the deliverable. The published page is built from them in a
separate repo, so nothing here serves HTTP.

---

## The weekly view (`mindspace week`)

Steps 1–8 build the three-month map. The weekly view answers a different
question — *what is getting attention right now* — and is deliberately its own
thing:

- **its own projection.** UMAP is fitted over that week's documents alone, so
  they fill the sphere instead of huddling in one corner of a corpus-wide map.
  It is explicitly not overlaid on the three-month projection.
- **TF-IDF names only.** Clustering is the usual embedding → UMAP → HDBSCAN;
  TF-IDF supplies the labels. No model is called, so the whole view costs
  nothing and can be regenerated as often as you like.
- **`modes.week.min_cluster_size: 3`** — the original prototype's value, which
  suits a few hundred documents. The quarter needs 15 for the same data at
  thirteen times the size; a single shared number cannot serve both, which is
  why the knobs live per mode.
- `--max-clusters N` folds all but the N largest clusters into noise, reported
  rather than silent. Measured on one week: a cap of 15 on top of
  `min_cluster_size: 3` sends 57% of the documents to noise, so the cap and the
  minimum size are worth choosing together.

Output: `data/output/week/` — the same projector file set as the quarter, sized
to the week.

---

## Running the Full Pipeline

```bash
uv run python -m mindspace            # the command list

# viz #3 — the three-month map, in the only safe order:
uv run python -m mindspace scrape --week 1     # a week into the database
uv run python -m mindspace embed               # vectors for what is new   (~$0.01/1k)
uv run python -m mindspace quarter             # cluster: background + week + slider frames
uv run python -m mindspace fable               # panel + Fable on the top name gaps  (~$0.80)
uv run python -m mindspace label               # cheap labels for the rest           (~$0.25)
uv run python -m mindspace export              # projector tensors
uv run python -m mindspace viz                 # redraw HTML without re-clustering

# viz #2 — one week on its own map, free, independent of the above:
uv run python -m mindspace week

# viz #1 — six months of papers, in their own database:
uv run python -m mindspace arxiv scrape --start 2026-02 --end 2026-08
uv run python -m mindspace arxiv embed         # ~$0.30 for 50k abstracts
uv run python -m mindspace arxiv
```

`fable` and `label` write names keyed to the cluster ids of the last `quarter`
run. Running them against a failed or re-run clustering attaches expensive
prose to the wrong documents, and nothing errors when it happens — so only
run them on a `quarter` that finished cleanly.

`viz` exists because re-clustering is never free: cluster ids change, and every
generated label is addressed to an id. Cosmetic changes should redraw, not
recompute.

## Environment Variables (`.env`)

| Key | Used for |
|-----|----------|
| `OPENROUTER_API_KEY` | Everything: embeddings, all 4 compressions, Fable synthesis, Haiku labelling |
| `XAI_API_KEY` | X/Twitter scraping (Grok `x_search`) |

One deliberate simplification versus the original repo: every model call is
routed through OpenRouter, so there is a single key, a single bill, and the
real cost of each run is recorded in `data/cost/*.json` from the API's own
usage figures.
