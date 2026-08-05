"""Every path this project reads or writes, resolved once from the repo root.

Anchored on this file's location, never on the working directory. A
cwd-relative path fails quietly: run from anywhere but the repo root and the
pipeline creates a second, empty output tree, finds no database, and reports
success. That is how a probe once had to be re-anchored by hand, and how a
directory reshuffle can silently orphan a corpus that took paid scrapes to
build.

Set MINDSPACE_DATA to keep the data outside the repo. Worth considering:
`git clean -xdf` deletes ignored files, and mindspace.db holds tweets that
cannot be bought again.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MINDSPACE_DATA") or ROOT / "data")

CONFIG = ROOT / "config.yaml"

DB = DATA / "mindspace.db"

# arXiv lives in its own database, same schema. Not tidiness: it arrives at
# roughly a thousand papers a day against the discourse corpus's fifty, it is
# clustered lexically and needs no embeddings, and its abs URLs collide with
# the HuggingFace Papers rows already in the main corpus. Keeping the files
# apart makes contamination impossible by construction rather than by
# remembering to filter.
ARXIV_DB = DATA / "arxiv.db"

# Regenerable artefacts, one directory per mode so the tree mirrors `modes:`
# in config.yaml and a third view can be added without rearranging anything.
# Safe to delete: a rerun rebuilds all of it.
OUTPUT = DATA / "output"
QUARTER = OUTPUT / "quarter"        # viz #3: clusters, frames, projection, HTML
WEEK = OUTPUT / "week"              # viz #2: the standalone weekly sphere
ARXIV = OUTPUT / "arxiv"            # viz #1: six months of papers
PROJECTOR = QUARTER / "projector"   # the quarter's TF Projector tensors

# Append-only ledgers of money actually spent. Deliberately NOT under OUTPUT,
# which gets wiped when the pipeline is rerun — a spend record that a rerun
# can erase is not a record.
COST = DATA / "cost"


def ensure() -> None:
    """Create the writable directories. Idempotent, cheap, safe to call often."""
    for d in (QUARTER, WEEK, ARXIV, COST):
        d.mkdir(parents=True, exist_ok=True)
