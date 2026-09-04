import json
import sys
from datetime import date, timedelta
from pathlib import Path
from .. import paths

import numpy as np

from .. import db

OUT = paths.PROJECTOR
TENSOR_NAME = "mindspaceai"


def _cell(value, limit: int = 200, empty: str = "(none)") -> str:
    """One TSV cell: never blank, never containing a tab or newline."""
    text = "" if value is None else str(value)
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
    return (text[:limit] or empty)


def build(background_days: int = 91, week_days: int = 7) -> None:
    articles = db.get_all_embedded()
    if not articles:
        print("no embedded articles — run the pipeline first", file=sys.stderr)
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)

    vectors = np.stack([a["embedding"] for a in articles]).astype("<f4")
    (OUT / "vectors.bytes").write_bytes(vectors.tobytes())

    cluster_of, label_of = {}, {}
    coords = None
    proj = paths.QUARTER / "projection.npz"
    if proj.exists():
        p = np.load(proj, allow_pickle=True)
        coords = p["coords"]
        for aid, lbl in zip([str(x) for x in p["article_ids"]], p["labels"]):
            cluster_of[aid] = int(lbl)

    def _plain(text: str) -> str:
        """Markdown stripped, whitespace collapsed — NOT truncated. The first
        version cut at 300 characters and the sidebar showed attractors ending
        mid-word ("…encoding, computi"). The sidebar scrolls; there is no
        reason to amputate the one paragraph that cost real money to write."""
        return " ".join((text or "").replace("*", "").split())

    def _cut(text: str, n: int) -> str:
        text = text or ""
        if len(text) <= n:
            return text
        return text[:n].rsplit(" ", 1)[0] + " …"

    # Names are per week: cluster ids restart with every weekly clustering, so
    # a flat map can only ever describe one of them.
    from ..compress import load_week_names
    _frames_peek = paths.QUARTER / "frames.json"
    _latest = (json.loads(_frames_peek.read_text())[-1]["week_end"]
               if _frames_peek.exists() else None)
    labels_by_week, attractors_by_week = load_week_names(paths.QUARTER, _latest)
    attractor_of: dict[int, str] = {
        cid: _plain(text)
        for cid, text in (attractors_by_week.get(_latest) or {}).items()}

    band_of, gap_of, desc_of = {}, {}, {}
    ai_labels: dict[int, str] = {}
    gaps_path, clusters_path = paths.QUARTER / "name_gaps.json", paths.QUARTER / "clusters.json"
    if gaps_path.exists() and clusters_path.exists():
        from ..viz import coherence_color
        gaps = {g["cluster_id"]: g for g in json.loads(gaps_path.read_text())}
        ai_labels = dict(labels_by_week.get(_latest) or {})
        for c in json.loads(clusters_path.read_text()):
            g = gaps.get(c["cluster_id"])
            if not g or g.get("lexical_rank") is None:
                continue
            band = coherence_color(g["lexical_rank"], g["semantic_rank"])[1]
            name = ai_labels.get(c["cluster_id"]) or c["label"]
            for m in c["members"]:
                band_of[m["url"]] = band
                gap_of[m["url"]] = g["name_gap"]
                label_of[m["url"]] = name
                if c["cluster_id"] in attractor_of:
                    desc_of[m["url"]] = attractor_of[c["cluster_id"]]

    today = date.today()
    bg_from = (today - timedelta(days=background_days)).isoformat()
    wk_from = (today - timedelta(days=week_days)).isoformat()

    def coherence_bucket(gap) -> str:
        if gap is None:
            return "background"
        b = round(float(gap) * 10) / 10
        word = ("emerging" if b > 0.33 else
                "buzzword" if b < -0.33 else "established")
        num = f"{b:+.1f}" if b != 0 else "+0.0"
        return f"{num} {word}"
    
    week_cols: list[tuple[str, dict[int, str]]] = []
    latest_names: dict[int, str] = {}
    frames_path = paths.QUARTER / "frames.json"
    if frames_path.exists():
        frames = json.loads(frames_path.read_text())
        aligned = all(max(f["week_idx"], default=0) < len(articles) for f in frames)
        if not aligned:
            print("frames.json indexes a different corpus — weekly columns "
                  "skipped; re-run the pipeline first", file=sys.stderr)
        else:
            weeks_manifest = []
            latest_week = frames[-1]["week_end"]
            latest_names: dict[int, str] = {}
            for f in frames:
                by_pos: dict[int, str] = {}
                names_by_pos: dict[int, str] = {}
                rep_by_pos: dict[int, str] = {}
                clusters_out = []
                for c in f.get("clusters", []):
                    bucket = coherence_bucket(c.get("name_gap"))
                    fable_name = (labels_by_week.get(f["week_end"], {})
                                  .get(c.get("cluster_id")))
                    name = _cell(fable_name or c.get("label")
                                 or c.get("keywords"), 60, empty="(unnamed)")
                    member_pos = [f["week_idx"][pnt] for pnt in c.get("points", [])
                                  if pnt < len(f["week_idx"])]
                    for pos in member_pos:
                        by_pos[pos] = bucket
                        names_by_pos[pos] = name
                    if member_pos and coords is not None:
                        pts = coords[member_pos]
                        centroid = pts.mean(axis=0)
                        rep = member_pos[
                            int(np.argmin(((pts - centroid) ** 2).sum(axis=1)))]
                        rep_by_pos[rep] = name
                    att = (attractor_of.get(c.get("cluster_id"))
                           if f["week_end"] == latest_week else None)
                    clusters_out.append({
                        "name": name, "gap": c.get("name_gap"),
                        "size": c.get("size"),
                        "desc": att or _cell(c.get("keywords"), 200, "(none)"),
                        "has_attractor": bool(att),
                        "pts": member_pos,
                    })
                clusters_out.sort(key=lambda x: -(x["gap"] or -9))
                if f["week_end"] == latest_week:
                    latest_names = names_by_pos
                weeks_manifest.append({"week": f["week_end"],
                                       "clusters": clusters_out})
                week_cols.append((f"coherence {f['week_end']}", by_pos))
                week_cols.append((f"cluster {f['week_end']}", names_by_pos))
                week_cols.append((f"label {f['week_end']}", rep_by_pos))
            (OUT / "weeks.json").write_text(json.dumps(weeks_manifest))
            print(f"  weekly columns: {len(week_cols)} for "
                  f"{len(weeks_manifest)} stops → weeks.json")

    header = (["title", "cluster_name", "coherence", "window", "source",
               "published", "author", "url", "description", "name_gap",
               "bg_cluster", "band"]
              + [name for name, _ in week_cols])
    rows, blanks = [], 0
    for a in articles:
        pub = (a.get("published_at") or "")[:10]
        window = ("this-week" if pub >= wk_from
                  else "background" if pub >= bg_from else "older")
        url = a.get("url", "")
        gap = gap_of.get(url)
        cells = [
            _cell(a.get("title"), 120, empty="(untitled)"),
            _cell(latest_names.get(len(rows)) or label_of.get(url), 60,
                  empty="(no cluster)"),
            _cell(coherence_bucket(gap)),
            _cell(window),
            _cell(a.get("source"), 40),
            _cell(pub, empty="(undated)"),
            _cell(a.get("author"), 60, empty="(unknown)"),
            _cell(url, 200, empty="(no url)"),
            _cell(_cut(desc_of.get(url), 300), 320, empty="(background)"),
            _cell(f"{gap:+.3f}" if gap is not None else None, empty="(n/a)"),
            _cell(cluster_of.get(a["id"], -2), empty="-2"),
            _cell(band_of.get(url), empty="(not scored)"),
        ]
        pos = len(rows)
        cells += [
            _cell(by_pos.get(pos, "\u200b"), empty="\u200b")
            if name.startswith("label ")
            else _cell(by_pos.get(pos, "background"))
            for name, by_pos in week_cols
        ]
        blanks += sum(1 for c in cells if not c)
        rows.append("\t".join(cells))

    (OUT / "metadata.tsv").write_text(
        "\t".join(header) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    from ..cluster import project_anchored
    ids = [a["id"] for a in articles]
    coords3 = project_anchored(
        vectors.astype(np.float32), ids,
        paths.QUARTER / "umap_anchor_3d.joblib",
        n_neighbors=15, n_components=3, min_dist=0.05, metric="cosine",
        log=lambda m: print(f"  {m}"))
    np.savez(paths.QUARTER / "projection3d.npz",
             coords=coords3, article_ids=np.array(ids))
    (OUT / "umap3d.bytes").write_bytes(
        np.asarray(coords3, dtype="<f4").tobytes())

    (OUT / "projector_config.json").write_text(json.dumps({
        "embeddings": [
            {
                "tensorName": TENSOR_NAME,
                "tensorShape": list(vectors.shape),
                "tensorPath": "data/vectors.bytes",
                "metadataPath": "data/metadata.tsv",
            },
            {
                "tensorName": f"{TENSOR_NAME} — cluster map (UMAP 3D)",
                "tensorShape": [int(coords3.shape[0]), 3],
                "tensorPath": "data/umap3d.bytes",
                "metadataPath": "data/metadata.tsv",
            },
        ]
    }, indent=2))

    size = (OUT / "vectors.bytes").stat().st_size / 1048576
    print(f"  vectors.bytes  {vectors.shape[0]} x {vectors.shape[1]}  {size:.1f} MB")
    print(f"  metadata.tsv   {len(rows)} rows, {len(header)} columns")
    print(f"  empty cells    {blanks}  (any non-zero here crashes the projector)")
    print(f"  scored         {len(band_of)} documents carry a coherence band")
    print(f"  → {OUT}")

    # The tensors and the two HTML views read the same clusters and the same
    # names, so writing one without the other leaves the deliverable describing
    # itself two different ways — and nothing fails when it happens. Keeping
    # them in one command removes the chance to forget the second.
    # Imported here rather than at module scope: pipeline reaches back into
    # this module, so a top-level import closes the loop.
    from ..pipeline import load_config, run_viz_only
    run_viz_only(load_config())


if __name__ == "__main__":
    build()
