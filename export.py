from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _clean(value: str, limit: int = 120) -> str:
    """Metadata is tab-separated, so tabs and newlines have to go."""
    return " ".join(str(value).split())[:limit]


def _label(cluster_names: dict[int, str], cid: int) -> str:
    if cid < 0:
        return "(no cluster)"
    return _clean(cluster_names.get(cid) or f"(unnamed {cid})", 60)


def export_projector(
    records: list[dict],
    vectors: np.ndarray,
    labels: np.ndarray,
    clusters: list[dict],
    out_dir: Path | str,
    cluster_names: dict[int, str] | None = None,
    urls: dict[str, str] | None = None,
    titles: dict[str, str] | None = None,
    max_points: int | None = None,
    binary: bool = False,
    url_prefix: str = "data",
) -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sizes = {c["cluster_id"]: c["size"] for c in clusters}
    cluster_names = cluster_names or {}
    urls = urls or {}
    titles = titles or {}

    idx = np.arange(len(records))
    if max_points and len(idx) > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))

    if binary:
        vectors[idx].astype("<f4").tofile(out / "vectors.bytes")
    else:
        with (out / "vectors.tsv").open("w", encoding="utf-8") as fh:
            for i in idx:
                fh.write("\t".join(f"{x:.5f}" for x in vectors[i]) + "\n")

    with (out / "metadata.tsv").open("w", encoding="utf-8") as fh:
        fh.write(
            "cluster_name\ttitle\tcluster_id\tsize\tmonth\tdocument\turl\ttext\n"
        )
        for i in idx:
            rec = records[i]
            cid = int(labels[i])
            fh.write(
                "\t".join(
                    [
                        _label(cluster_names, cid),
                        _clean(titles.get(rec["document_id"]) or "(no title)", 120),
                        "noise" if cid < 0 else str(cid),
                        str(sizes.get(cid, 0)),
                        rec["published_at"][:7],
                        rec["document_id"],
                        urls.get(rec["document_id"]) or "(no url)",
                        _clean(rec["text"], 160) or "(no text)",
                    ]
                )
                + "\n"
            )

    prefix = url_prefix.rstrip("/")
    config = {
        "embeddings": [
            {
                "tensorName": f"trend-detection {records[0]['published_at'][:7]}",
                "tensorShape": [len(idx), int(vectors.shape[1])],
                "tensorPath": f"{prefix}/vectors.bytes" if binary else f"{prefix}/vectors.tsv",
                "metadataPath": f"{prefix}/metadata.tsv",
            }
        ]
    }
    (out / "projector_config.json").write_text(json.dumps(config, indent=2))

    return {
        "points": len(idx),
        "dims": int(vectors.shape[1]),
        "clusters": len(sizes),
        "noise": int((labels[idx] < 0).sum()),
    }
