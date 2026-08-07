"""One entry point for the whole pipeline: `python -m mindspace <command>`.

The commands are the real steps, and the three views the brief asks for are
verbs of their own — `week`, `quarter`, `arxiv` — so the deliverables are
visible in the interface rather than buried in flag combinations.

Every flag after the command is passed straight through to the step, so
`python -m mindspace scrape --week 2` and `python -m mindspace week --week 2`
work exactly as the underlying modules document.
"""

import sys

USAGE = """usage: python -m mindspace <command> [options]

collect
  scrape [--week N | --days N | --as-of DATE]   fetch a window into the database
  embed                                        embed documents that lack a vector   [paid, ~$0.01/1k]

visualise
  week    [--week N | --as-of DATE]            viz #2: one week on its own map, TF-IDF names
  quarter [--as-of DATE]                       viz #3: 3-month map, weekly clusters, slider frames
  arxiv scrape --start YYYY-MM --end YYYY-MM   collect papers into data/arxiv.db
  arxiv embed                                  vectors for the paper corpus        [paid, ~$0.30/50k]
  arxiv   [--start YYYY-MM --end YYYY-MM]      viz #1: six months of papers, TF-IDF names

name  (only meaningful after `quarter`)
  fable [--top N] [--buzzwords N]              model panel -> Fable attractors     [paid, ~$0.80]
  label                                        cheap topical labels for the rest   [paid, ~$0.25]

publish
  export                                       projector tensors + the HTML views,
                                               from the last quarter run

Anything after the command goes to that step: `python -m mindspace scrape --week 3`.

Order matters in one place. `fable` and `label` write names keyed to the cluster
ids of the last `quarter` run, so running them after a failed or re-run
clustering attaches expensive prose to the wrong documents — silently, with
nothing failing. Run them only on a `quarter` that succeeded:

    scrape -> embed -> quarter -> fable -> label -> export
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    cmd, rest = argv[0], argv[1:]

    # Each visualisation is one pipeline invocation with its stages chosen;
    # the flags they inject are exactly what the old command lines used.
    if cmd == "scrape":
        from . import pipeline
        return pipeline.main(["--scrape-only", *rest]) or 0
    if cmd == "embed":
        from . import pipeline
        return pipeline.main(["--embed-only", *rest]) or 0
    if cmd == "quarter":
        from . import pipeline
        return pipeline.main(["--skip-scrape", "--skip-embed", *rest]) or 0
    if cmd == "week":
        from .export import week
        return week.main(rest) or 0
    if cmd == "fable":
        from . import compress
        return compress.main(["--rank-by", "name-gap",
                              "--synth", "anthropic/claude-fable-5", *rest]) or 0
    if cmd == "label":
        from . import labels
        return labels.main(rest) or 0
    if cmd == "export":
        from .export import projector
        projector.build()
        return 0
    if cmd == "arxiv":
        if rest and rest[0] == "scrape":
            from . import pipeline
            return pipeline.arxiv_scrape(rest[1:])
        if rest and rest[0] == "embed":
            from . import pipeline
            return pipeline.arxiv_embed(rest[1:])
        from .export import arxiv as arxiv_view
        return arxiv_view.main(rest) or 0

    print(f"unknown command: {cmd}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
