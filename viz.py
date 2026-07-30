from pathlib import Path

import numpy as np
import plotly.graph_objects as go


_NOISE_COLOR = "rgba(160,160,160,0.35)"

_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
]


def _color(cluster_id: int) -> str:
    if cluster_id == -1:
        return _NOISE_COLOR
    return _PALETTE[cluster_id % len(_PALETTE)]


def build_viz(articles: list[dict], coords: np.ndarray,
              labels: np.ndarray, clusters: list[dict],
              out_dir: str) -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_file = out_path / "viz.html"

    cluster_label_map = {c["cluster_id"]: c["label"] for c in clusters}

    traces = {}
    for i, (article, (x, y), lbl) in enumerate(zip(articles, coords, labels)):
        lbl = int(lbl)
        if lbl not in traces:
            traces[lbl] = {
                "x": [], "y": [],
                "text": [],
                "urls": [],
                "color": _color(lbl),
                "name": cluster_label_map.get(lbl, "noise") if lbl != -1 else "noise",
            }
        title = article.get("title", "")[:120]
        source = article.get("source", "")
        author = article.get("author", "") or ""
        url = article.get("url", "")
        pub = (article.get("published_at") or "")[:10]

        hover = (
            f"<b>{title}</b><br>"
            f"<i>{source}</i>"
            + (f" · {author}" if author else "")
            + (f" · {pub}" if pub else "")
            + f"<br><a href='{url}'>{url[:80]}</a>"
        )

        traces[lbl]["x"].append(x)
        traces[lbl]["y"].append(y)
        traces[lbl]["text"].append(hover)
        traces[lbl]["urls"].append(url)

    fig_traces = []
    for lbl in sorted(traces.keys(), key=lambda k: (k == -1, k)):
        t = traces[lbl]
        fig_traces.append(
            go.Scatter(
                x=t["x"],
                y=t["y"],
                mode="markers",
                marker=dict(
                    color=t["color"],
                    size=7 if lbl != -1 else 4,
                    opacity=0.85 if lbl != -1 else 0.4,
                    line=dict(width=0.5, color="white"),
                ),
                name=t["name"],
                text=t["text"],
                hovertemplate="%{text}<extra></extra>",
                customdata=t["urls"],
            )
        )

    layout = go.Layout(
        title="AI Topic Clusters",
        hovermode="closest",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e0e0e0"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(
            bgcolor="rgba(30,30,40,0.8)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    click_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
      var plot = document.querySelector('.js-plotly-plot');
      if (!plot) return;
      plot.on('plotly_click', function(data) {
        var pt = data.points[0];
        if (pt.customdata) window.open(pt.customdata, '_blank');
      });
    });
    </script>
    """

    html_content = (
        go.Figure(data=fig_traces, layout=layout).to_html(full_html=True, include_plotlyjs=True)
        + click_js
    )
    html_file.write_text(html_content, encoding="utf-8")
    print(f"[viz] saved → {html_file}")
    return str(html_file)
