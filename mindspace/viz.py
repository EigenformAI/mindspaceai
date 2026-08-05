import json
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


# ── Step 5: the coherence colour scale ───────────────────────────────────────
#
# A CONTINUOUS scale, not bands — the spec's revision (2026-08-03):
# "the colours need to be a sliding scale based on the semantic-lexical ratio
# going from yellow to red to blue progressively so that each cluster is a
# slightly different colour."
#
# The axis is name_gap = rank(semantic) − rank(lexical), in [−1, +1]:
#
#   +1  pale gold   meaning shared, vocabulary not yet — emerging
#    0  red         the two measures agree — established
#   −1  indigo      vocabulary shared, meaning not — buzzword
#
# Ranks rather than raw cosines, as before: TF-IDF cosine sits near 0 for every
# cluster and embedding cosine near 0.4, so only the position among the week's
# other clusters means anything.
_GAP_ANCHORS = [
    (-1.0, (67, 56, 202)),     # indigo — buzzword end
    (-0.5, (147, 51, 234)),    # purple
    (0.0, (220, 38, 38)),      # red — established
    (0.5, (255, 150, 40)),     # orange
    (1.0, (255, 224, 130)),    # pale gold — emerging end
]

_UNCOLOURED = "rgba(120,120,128,0.40)"


def gap_color(gap: float | None) -> str:
    """Piecewise-linear interpolation along the anchors above."""
    if gap is None:
        return _UNCOLOURED
    g = max(-1.0, min(1.0, float(gap)))
    for (g0, c0), (g1, c1) in zip(_GAP_ANCHORS, _GAP_ANCHORS[1:]):
        if g <= g1:
            t = (g - g0) / (g1 - g0)
            r, gr, b = (round(a + (b2 - a) * t) for a, b2 in zip(c0, c1))
            return f"rgb({r},{gr},{b})"
    r, gr, b = _GAP_ANCHORS[-1][1]
    return f"rgb({r},{gr},{b})"


# Click a point to open the article. Attached after render rather than at figure
# build time because `plotly_click` lives on the DOM element; the retry covers
# the case where the script runs before Plotly has finished drawing. The URL
# guard matters because traces without customdata can also be clicked.
_CLICK_JS = """
<script>
(function () {
  function wire() {
    var plot = document.querySelector('.js-plotly-plot');
    if (!plot || !plot.on) { return setTimeout(wire, 120); }
    plot.on('plotly_click', function (data) {
      var pt = data.points && data.points[0];
      var url = pt && pt.customdata;
      if (url) window.open(url, '_blank', 'noopener');
    });
  }
  wire();
})();
</script>
"""


def coherence_color(lex_rank: float | None, sem_rank: float | None) -> tuple[str, str]:
    """(css colour, coarse descriptor) for one cluster's pair of ranks.

    The colour is continuous; the descriptor is only a word for hover text and
    the projector export, cut at thirds of the gap axis.
    """
    if lex_rank is None or sem_rank is None:
        return _UNCOLOURED, "unscored"
    gap = sem_rank - lex_rank
    word = "emerging" if gap > 0.33 else "buzzword" if gap < -0.33 else "established"
    return gap_color(gap), word


def build_background_viz(articles: list[dict], coords: np.ndarray,
                         labels: np.ndarray, week_idx: set[int],
                         out_dir: str) -> str:
    """Spec step 2 — the three-month background, in greyscale.

    Deliberately not a presentation. Its job is to let someone look at step 1
    and judge whether the clustering produced anything sensible: are the basins
    separated, is the noise plausible, and is this week scattered across the map
    or piled in one corner. Colour would only get in the way of that.

    Written to background.html because compress_clusters.py owns viz.html.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_file = out_path / "background.html"

    labels = np.asarray(labels)
    xs, ys = coords[:, 0], coords[:, 1]

    # Greyscale by cluster: noise darkest and faintest, real clusters stepped
    # through light greys so the basins are distinguishable without implying
    # any of them is special. Which grey a cluster gets carries no meaning.
    #
    # -1 is noise inside the window, -2 is a document outside it entirely. They
    # are drawn differently on purpose: one was examined and placed nowhere,
    # the other was never in scope.
    real = sorted({int(v) for v in labels} - {-1, -2})
    grey_of = {}
    for i, cid in enumerate(real):
        level = 110 + int(110 * (i / max(len(real) - 1, 1)))
        grey_of[cid] = f"rgba({level},{level},{level},0.55)"
    grey_of[-1] = "rgba(70,70,70,0.30)"     # noise in the window
    grey_of[-2] = "rgba(45,45,52,0.18)"     # outside the window

    bg_x, bg_y, bg_c, bg_t, bg_u = [], [], [], [], []
    wk_x, wk_y, wk_t, wk_u = [], [], [], []

    for i, (x, y, lbl) in enumerate(zip(xs, ys, labels)):
        a = articles[i]
        hover = (
            f"<b>{(a.get('title') or '')[:110]}</b><br>"
            f"<i>{a.get('source', '')}</i>"
            + (f" · {(a.get('published_at') or '')[:10]}")
            + f"<br>cluster {int(lbl)}"
            + "<br><span style='color:#8a92a6'>click to open</span>"
        )
        url = a.get("url", "")
        if i in week_idx:
            wk_x.append(x); wk_y.append(y); wk_t.append(hover); wk_u.append(url)
        else:
            bg_x.append(x); bg_y.append(y); bg_t.append(hover); bg_u.append(url)
            bg_c.append(grey_of.get(int(lbl), "rgba(70,70,70,0.30)"))

    traces = [
        go.Scatter(
            x=bg_x, y=bg_y, mode="markers", name="three-month background",
            marker=dict(color=bg_c, size=4, line=dict(width=0)),
            text=bg_t, customdata=bg_u, hovertemplate="%{text}<extra></extra>",
        ),
        # This week is outlined, not filled: it marks which documents step 4
        # scored, without pre-empting the colour scale that will eventually
        # carry the coherence result.
        go.Scatter(
            x=wk_x, y=wk_y, mode="markers", name="this week",
            marker=dict(color="rgba(0,0,0,0)", size=8,
                        line=dict(width=1.1, color="rgba(255,255,255,0.85)")),
            text=wk_t, customdata=wk_u, hovertemplate="%{text}<extra></extra>",
        ),
    ]

    n_noise = int((labels == -1).sum())
    n_outside = int((labels == -2).sum())
    layout = go.Layout(
        title=(f"Three-month background — {len(articles) - n_outside} documents, "
               f"{len(real)} clusters, {n_noise} noise, "
               f"{len(wk_x)} from this week"
               + (f" ({n_outside} older documents shown faint)" if n_outside else "")),
        hovermode="closest",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e0e0e0"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(20,20,32,0.88)",
                    bordercolor="#444", borderwidth=1, font=dict(size=11)),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    html_file.write_text(
        go.Figure(data=traces, layout=layout).to_html(
            full_html=True, include_plotlyjs=True) + _CLICK_JS,
        encoding="utf-8")
    print(f"[viz] saved → {html_file}")
    return str(html_file)


def build_viz(articles: list[dict], coords: np.ndarray,
              labels: np.ndarray, clusters: list[dict],
              out_dir: str,
              ai_labels: dict[int, str] | None = None,
              attractors: dict[int, str] | None = None) -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_file = out_path / "viz.html"

    cluster_label_map = {c["cluster_id"]: c["label"] for c in clusters}
    has_ai = bool(ai_labels)

    traces = {}
    for i, (article, (x, y), lbl) in enumerate(zip(articles, coords, labels)):
        lbl = int(lbl)
        if has_ai and lbl != -1 and lbl not in ai_labels:
            continue
        if lbl not in traces:
            tfidf_name = cluster_label_map.get(lbl, "noise") if lbl != -1 else "noise"
            traces[lbl] = {
                "x": [], "y": [],
                "text": [],
                "urls": [],
                "color": _color(lbl),
                "name": tfidf_name,
                "tfidf_name": tfidf_name,
                "ai_name": (ai_labels.get(lbl, tfidf_name) if has_ai and lbl != -1 else tfidf_name),
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

    trace_order = sorted(traces.keys(), key=lambda k: (k == -1, k))
    fig_traces = []
    for lbl in trace_order:
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
                name=t["tfidf_name"],
                text=t["text"],
                hovertemplate="%{text}<extra></extra>",
                customdata=t["urls"],
            )
        )

    tfidf_names = [traces[lbl]["tfidf_name"] for lbl in trace_order]
    ai_names   = [traces[lbl]["ai_name"]    for lbl in trace_order]

    # Map AI short label → full attractor description for tooltip
    attractor_map: dict[str, str] = {}
    if has_ai and attractors:
        for lbl in trace_order:
            if lbl != -1 and lbl in attractors and lbl in (ai_labels or {}):
                attractor_map[ai_labels[lbl]] = attractors[lbl]

    toggle = [dict(
        type="buttons",
        direction="left",
        x=0.01, y=1.04,
        xanchor="left",
        yanchor="top",
        showactive=True,
        bgcolor="rgba(30,30,40,0.85)",
        bordercolor="#555",
        font=dict(color="#e0e0e0", size=12),
        buttons=[
            dict(label="Keywords", method="restyle",
                 args=[{"name": tfidf_names}]),
            dict(label="AI Labels", method="restyle",
                 args=[{"name": ai_names}]),
        ],
    )] if has_ai else []

    layout = go.Layout(
        title="AI Topic Clusters",
        hovermode="closest",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e0e0e0"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(20,20,32,0.88)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(size=11),
        ),
        updatemenus=toggle,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    attractor_json = json.dumps(attractor_map)

    click_js = f"""
    <style>
    #attractor-tip {{
      display: none;
      position: fixed;
      max-width: 380px;
      padding: 10px 14px;
      background: rgba(20,20,32,0.97);
      color: #d0d8f0;
      border: 1px solid #556;
      border-radius: 7px;
      font-size: 13px;
      line-height: 1.5;
      pointer-events: none;
      z-index: 9999;
      box-shadow: 0 4px 18px rgba(0,0,0,0.6);
    }}
    </style>
    <div id="attractor-tip"></div>
    <script>
    (function() {{
      var ATTRACTORS = {attractor_json};
      var tip = document.getElementById('attractor-tip');

      // Capture phase bypasses any Plotly stopPropagation on SVG elements.
      // Also walk up from tspan children to the .legendtext parent.
      function legendEl(target) {{
        var el = target;
        for (var i = 0; i < 4 && el; i++) {{
          if (el.classList && el.classList.contains('legendtext')) return el;
          el = el.parentElement;
        }}
        return null;
      }}

      document.addEventListener('mouseover', function(e) {{
        var el = legendEl(e.target);
        if (!el) {{ tip.style.display = 'none'; return; }}
        var name = el.getAttribute('data-unformatted') || el.textContent.trim();
        var desc = ATTRACTORS[name];
        if (!desc) {{ tip.style.display = 'none'; return; }}
        tip.textContent = desc;
        tip.style.display = 'block';
      }}, true);   /* capture=true — runs before Plotly handlers */

      document.addEventListener('mousemove', function(e) {{
        if (tip.style.display === 'none') return;
        var x = e.clientX + 18, y = e.clientY + 14;
        if (x + 400 > window.innerWidth)  x = e.clientX - 418;
        if (y + 120 > window.innerHeight) y = e.clientY - 80;
        tip.style.left = x + 'px';
        tip.style.top  = y + 'px';
      }}, true);

      document.addEventListener('mouseout', function(e) {{
        if (legendEl(e.target)) tip.style.display = 'none';
      }}, true);

      // Click-to-open for scatter points
      document.addEventListener('DOMContentLoaded', function() {{
        var plot = document.querySelector('.js-plotly-plot');
        if (!plot) return;
        plot.on('plotly_click', function(data) {{
          var pt = data.points[0];
          if (pt.customdata) window.open(pt.customdata, '_blank');
        }});
      }});
    }})();
    </script>
    """

    html_content = (
        go.Figure(data=fig_traces, layout=layout).to_html(full_html=True, include_plotlyjs=True)
        + click_js
    )

    html_file.write_text(html_content, encoding="utf-8")
    print(f"[viz] saved → {html_file}")
    return str(html_file)


# ── Step 6: the week-by-week slider ──────────────────────────────────────────

def build_slider_viz(frames_data: list[dict], coords: np.ndarray,
                     articles: list[dict], out_dir: str) -> str:
    """One frame per week; the slider moves both windows forward together.

    THE COORDINATES ARE THE SAME IN EVERY FRAME. That is the whole point, and
    it is why the projection is fitted once over the entire corpus rather than
    per window. Measured: refitting UMAP on identical input moves nothing, but
    drop 8% of the documents — one week of slide — and points shift (residual
    0.17 after optimal rotation). A per-window fit would rearrange the map at
    every stop, which is exactly the "continuity between basins of attraction"
    problem this is built to avoid. Here a basin stays put and only its colour
    changes, so continuity is something you see rather than something you have
    to reconstruct.

    Nothing animates for the same reason: there is no motion to animate. The
    spec allowed for that ("if this is hard/compute intensive we won't"); it
    turns out to be not hard but meaningless.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_file = out_path / "slider.html"

    # Colours are recomputed here from each cluster's stored name_gap rather
    # than read from the frame. The frames carry colours too, but those froze
    # whatever palette was in force when the pipeline ran — recomputing at draw
    # time means a palette change is a --viz-only redraw, not a re-cluster (and
    # not another Fable bill).
    for f in frames_data:
        n = len(f["week_idx"])
        colors = [_UNCOLOURED] * n
        for c in f.get("clusters", []):
            col = gap_color(c.get("name_gap"))
            c["color"] = col
            for pt in c.get("points", []):
                if 0 <= pt < n:
                    colors[pt] = col
        if f.get("clusters"):
            f["week_colors"] = colors

    xs, ys = coords[:, 0], coords[:, 1]
    hovers = [
        f"<b>{(a.get('title') or '')[:110]}</b><br>"
        f"<i>{a.get('source', '')}</i> · {(a.get('published_at') or '')[:10]}"
        "<br><span style='color:#8a92a6'>click to open</span>"
        for a in articles
    ]
    urls = [a.get("url", "") for a in articles]

    plotly_frames, slider_steps = [], []
    for f in frames_data:
        bg_idx = f["background_idx"]
        wk_idx = f["week_idx"]

        plotly_frames.append(go.Frame(
            name=f["week_end"],
            data=[
                go.Scatter(x=[xs[i] for i in bg_idx], y=[ys[i] for i in bg_idx],
                           mode="markers",
                           marker=dict(color="rgba(120,120,128,0.28)", size=3.5,
                                       line=dict(width=0)),
                           text=[hovers[i] for i in bg_idx],
                           # customdata rides along per frame, not just on the
                           # initial figure: Plotly replaces trace data wholesale
                           # when a frame is shown, so a click handler reading
                           # customdata finds nothing unless every frame carries
                           # it.
                           customdata=[urls[i] for i in bg_idx],
                           hovertemplate="%{text}<extra></extra>",
                           name="three-month background"),
                go.Scatter(x=[xs[i] for i in wk_idx], y=[ys[i] for i in wk_idx],
                           mode="markers",
                           marker=dict(color=f["week_colors"], size=8,
                                       line=dict(width=0.6, color="rgba(0,0,0,0.5)")),
                           text=[hovers[i] for i in wk_idx],
                           customdata=[urls[i] for i in wk_idx],
                           hovertemplate="%{text}<extra></extra>",
                           name="this week"),
            ],
        ))
        slider_steps.append(dict(
            method="animate", label=f["week_end"],
            args=[[f["week_end"]],
                  dict(mode="immediate",
                       frame=dict(duration=0, redraw=True),
                       transition=dict(duration=0))],
        ))

    first = plotly_frames[0]
    fig = go.Figure(
        data=list(first.data),
        frames=plotly_frames,
        layout=go.Layout(
            title=("Emerging concepts — three-month background, "
                   "this week coloured by coherence"),
            hovermode="closest",
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#e0e0e0"),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(20,20,32,0.9)",
                        bordercolor="#444", borderwidth=1, font=dict(size=10)),
            margin=dict(l=20, r=20, t=50, b=90),
            sliders=[dict(
                active=len(slider_steps) - 1,
                currentvalue=dict(prefix="week ending ", font=dict(size=13)),
                pad=dict(t=40), x=0.06, len=0.9,
                steps=slider_steps,
            )],
        ),
    )
    # Start on the most recent week, which is the one the slider defaults to.
    fig.update(data=list(plotly_frames[-1].data))

    # Attractors exist only for whichever week compress_clusters last ran on —
    # it reads clusters.json, which holds the current week. Attached to the last
    # frame alone rather than guessed at for the others, because a cluster id
    # means something different in every week's clustering, and reusing one
    # across weeks would put confident prose against the wrong documents.
    attractors = {}
    comp = out_path / "name_gap_compressions.json"
    if comp.exists():
        try:
            for r in json.loads(comp.read_text()):
                if r.get("attractor"):
                    attractors[int(r["cluster_id"])] = r["attractor"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fable-derived short labels outrank the cheap topical ones where they
    # exist (newest week's top clusters, per the user's decision 2026-08-03).
    ai_labels = {}
    lp = out_path / "name_gap_ai_labels.json"
    if lp.exists():
        try:
            ai_labels = {int(k): v for k, v in json.loads(lp.read_text()).items()}
        except (json.JSONDecodeError, ValueError):
            ai_labels = {}

    sidebar_frames = []
    for i, f in enumerate(frames_data):
        is_last = i == len(frames_data) - 1
        sidebar_frames.append({
            "week_end": f["week_end"],
            "n_points": len(f["week_idx"]),
            "clusters": [
                {**c,
                 "label": ((ai_labels.get(c["cluster_id"]) if is_last else None)
                           or c.get("label")),
                 "attractor": attractors.get(c["cluster_id"]) if is_last else None}
                for c in f.get("clusters", [])
            ],
        })
    if attractors:
        print(f"[viz] {len(attractors)} attractor(s) attached to "
              f"{frames_data[-1]['week_end']}")

    plot_html = fig.to_html(full_html=False, include_plotlyjs=True,
                            div_id="plot")
    page = _SLIDER_PAGE.replace("__PLOT__", plot_html).replace(
        "__FRAMES__", json.dumps(sidebar_frames))

    html_file.write_text(page, encoding="utf-8")
    print(f"[viz] saved \u2192 {html_file}  ({len(plotly_frames)} weekly stops)")
    return str(html_file)


_SLIDER_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Emerging concepts — weekly</title>
<style>
  :root { --surface:#0f1117; --panel:#141824; --line:#252a38;
          --ink:#e6e8ee; --dim:#9aa1b1; --mute:#6b7183; }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--surface);color:var(--ink);
    font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  #wrap{display:flex;height:100vh}
  #wrap > div:first-child{flex:1;min-width:0;height:100vh}
  #plot{width:100%;height:100%}
  #side{width:340px;flex:none;border-left:1px solid var(--line);
    background:var(--panel);overflow-y:auto;padding:18px 16px 40px}
  #side h2{font-size:12px;text-transform:uppercase;letter-spacing:.07em;
    color:var(--mute);margin:0 0 4px;font-weight:600}
  #wk{font-size:17px;font-weight:600;margin:0 0 4px}
  #meta{color:var(--dim);font-size:12.5px;margin:0 0 16px}
  .row{display:flex;gap:8px;padding:9px 8px;border-radius:7px;cursor:pointer;
    border:1px solid transparent}
  .row.noatt .nm{color:var(--dim)}
  .rk{width:19px;flex:none;text-align:right;color:var(--mute);font-size:11.5px;
    font-variant-numeric:tabular-nums;padding-top:2px}
  .row.noatt .rk{opacity:.55}
  .row:hover{background:#1a1f2e}
  .row.on{background:#1c2233;border-color:#39415a}
  .sw{width:11px;height:11px;border-radius:3px;flex:none;margin-top:4px}
  .nm{font-size:13px;line-height:1.4}
  .sc{color:var(--mute);font-size:11.5px;font-variant-numeric:tabular-nums;
    margin-top:2px}
  #att{margin-top:16px;padding:14px;border:1px solid var(--line);
    border-radius:8px;background:#10131c;font-size:13px;line-height:1.6;
    color:var(--dim);white-space:pre-wrap;display:none}
  #att b{color:var(--ink)}
  #grad{height:9px;border-radius:5px;margin:2px 0 4px;
    background:linear-gradient(90deg,#4338ca,#9333ea,#dc2626,#ff9628,#ffe082)}
  #gradlab{display:flex;justify-content:space-between;color:var(--mute);
    font-size:10.5px;margin-bottom:12px}
  #att .hd{font-size:11px;text-transform:uppercase;letter-spacing:.06em;
    color:var(--mute);margin-bottom:8px}
  .empty{color:var(--mute);font-size:13px;padding:8px 0}
</style></head>
<body><div id="wrap">__PLOT__<aside id="side">
  <h2>week ending</h2><div id="wk"></div><div id="meta"></div>
  <div id="grad"></div>
  <div id="gradlab"><span>buzzword</span><span>established</span><span>emerging</span></div>
  <div id="list"></div><div id="att"></div>
</aside></div>
<script>
(function(){
  var FRAMES = __FRAMES__;
  var side=document.getElementById('list'), wk=document.getElementById('wk'),
      meta=document.getElementById('meta'), att=document.getElementById('att');
  var cur = FRAMES.length - 1, N_WEEK = 0,
      plot = document.getElementById('plot');

  function fmt(x){ return (x>=0?'+':'') + x.toFixed(2); }

  // Trace 1 is this week's points; trace 0 is the background, and the band
  // legend entries come after. Restyling by index is safe because every frame
  // is built with the same two traces in the same order.
  var WEEK_TRACE = 1;

  function highlight(c){
    if(!N_WEEK) return;
    var lc = [], lw = [], sz = [];
    var mark = {};
    if(c){ c.points.forEach(function(p){ mark[p] = true; }); }
    for(var k = 0; k < N_WEEK; k++){
      var on = c && mark[k];
      lc.push(on ? '#ffffff' : 'rgba(0,0,0,0.5)');
      lw.push(on ? 2.2 : 0.6);
      sz.push(on ? 13 : 8);
    }
    Plotly.restyle(plot, {
      'marker.line.color':[lc], 'marker.line.width':[lw], 'marker.size':[sz]
    }, [WEEK_TRACE]);
  }

  function render(i){
    cur = i;
    N_WEEK = (FRAMES[i] && FRAMES[i].n_points) || 0;
    var f = FRAMES[i]; if(!f) return;
    wk.textContent = f.week_end;
    var n = f.clusters.length;
    var withAtt = f.clusters.filter(function(x){return x.attractor}).length;
    meta.textContent = n + ' cluster' + (n===1?'':'s') +
      ', ranked by name gap' + (withAtt ? ' · ' + withAtt + ' with an attractor' : '');
    att.style.display='none';
    side.innerHTML='';
    if(!n){ side.innerHTML='<div class="empty">no scored clusters this week</div>'; return; }
    f.clusters.forEach(function(c,k){
      var row=document.createElement('div');
      row.className='row' + (c.attractor ? '' : ' noatt');
      // The rank is the same order compress_clusters.py uses to pick its top N,
      // so a number here tells you why a cluster did or did not get an
      // attractor without having to know the flag it was run with.
      // Cheap Haiku label when label_clusters.py has run; TF-IDF keywords
      // otherwise, so a frame is never blank while waiting for labels.
      var nm = c.label || c.keywords || '(unnamed)';
      row.innerHTML='<span class="rk">'+(k+1)+'</span>'+
        '<span class="sw" style="background:'+c.color+'"></span>'+
        '<span><span class="nm">'+nm+'</span>'+
        '<div class="sc">gap '+fmt(c.name_gap)+' · n='+c.size+
        (c.attractor?' · attractor':'')+'</div></span>';
      row.onclick=function(){
        var was = row.classList.contains('on');
        [].forEach.call(side.children,function(e){e.classList.remove('on')});
        if(was){ highlight(null); att.style.display='none'; return; }
        row.classList.add('on');
        highlight(c);
        if(c.attractor){
          att.innerHTML='<div class="hd">attractor</div>'+
            c.attractor.replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>');
          att.style.display='block';
        } else {
          att.innerHTML='<div class="hd">keywords</div>'+(c.keywords||'(none)')+
            '<div class="hd" style="margin-top:10px">attractor</div>'+
            'Only the top clusters by name gap get one — see compress_clusters.py.';
          att.style.display='block';
        }
      };
      side.appendChild(row);
    });
  }

  function wire(){
    if(!plot || !plot.on){ return setTimeout(wire,120); }
    // plotly_sliderchange carries the step index; the slider is the only one
    // on this figure, so step.  is enough to identify the frame.
    plot.on('plotly_sliderchange', function(e){
      var name = e.step && e.step.label;
      var i = FRAMES.findIndex(function(f){ return f.week_end === name; });
      if(i >= 0){ render(i); highlight(null); }
    });
    plot.on('plotly_click', function(data){
      var pt = data.points && data.points[0];
      if(pt && pt.customdata) window.open(pt.customdata,'_blank','noopener');
    });
    render(cur);
  }
  wire();
})();
</script></body></html>
"""
