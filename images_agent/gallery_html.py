from __future__ import annotations

import html
import json
from pathlib import Path


def load_matches_json(path: Path) -> dict:
    raw = path.expanduser().resolve().read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("matches JSON must be a JSON object")
    return data


def _figure_block(path_str: str, score: float | None) -> str:
    p = Path(path_str).expanduser()
    try:
        uri = p.resolve().as_uri()
    except (OSError, ValueError):
        uri = ""
    uri_esc = html.escape(uri, quote=True)
    path_esc = html.escape(path_str)

    score_part = ""
    if score is not None:
        score_part = f'<span class="score">{html.escape(f"{score:.4f}")}</span>'

    img_tag = (
        f'<img src="{uri_esc}" alt="" loading="lazy" width="400" />'
        if uri_esc
        else '<span class="missing">(invalid path)</span>'
    )

    return (
        f'<figure class="card">'
        f'<a href="{uri_esc}" target="_blank" rel="noopener">{img_tag}</a>'
        f"<figcaption><code>{path_esc}</code>{score_part}</figcaption>"
        f"</figure>"
    )


def build_gallery_html(payload: dict) -> str:
    criteria_raw = payload.get("criteria", "")
    criteria = html.escape(str(criteria_raw)) if criteria_raw is not None else ""

    root_raw = payload.get("root", "")
    root_esc = html.escape(str(root_raw)) if root_raw is not None else ""

    paths = payload.get("paths")
    if not isinstance(paths, list):
        paths = []

    scores_raw = payload.get("scores")
    scores: list[float | None] | None = None
    if isinstance(scores_raw, list) and len(scores_raw) == len(paths):
        scores = []
        for s in scores_raw:
            if isinstance(s, (int, float)):
                scores.append(float(s))
            else:
                scores.append(None)

    rerank_used = payload.get("rerank_used")
    meta_bits: list[str] = []
    if rerank_used is True:
        pool = payload.get("rerank_pool")
        if isinstance(pool, int):
            meta_bits.append(html.escape(f"rerank pool: {pool}"))
        else:
            meta_bits.append(html.escape("vision rerank applied"))

    meta = f'<p class="meta">{" · ".join(meta_bits)}</p>' if meta_bits else ""

    figures: list[str] = []
    for i, path_str in enumerate(paths):
        if not isinstance(path_str, str):
            continue
        sc: float | None = scores[i] if scores is not None else None
        figures.append(_figure_block(path_str, sc))

    count = len(figures)
    gallery = "\n".join(figures) if figures else '<p class="empty">No paths in this result.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Matches — {criteria}</title>
  <style>
    :root {{
      font-family: system-ui, Segoe UI, Roboto, sans-serif;
      color: #1a1a1a;
      background: #f4f4f2;
    }}
    body {{
      margin: 0 auto;
      max-width: 1200px;
      padding: 1.5rem;
      line-height: 1.45;
    }}
    header {{
      margin-bottom: 1.5rem;
    }}
    header h1 {{
      font-size: 1.25rem;
      font-weight: 600;
      margin: 0 0 0.5rem;
    }}
    .root {{
      font-size: 0.85rem;
      color: #444;
      margin: 0;
    }}
    .meta {{
      font-size: 0.85rem;
      color: #555;
      margin: 0.5rem 0 0;
    }}
    .count {{
      font-size: 0.9rem;
      color: #333;
      margin: 0.25rem 0 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 1.25rem;
    }}
    .card {{
      margin: 0;
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgb(0 0 0 / 0.12);
      overflow: hidden;
    }}
    .card a {{
      display: block;
      background: #eee;
    }}
    .card img {{
      width: 100%;
      height: auto;
      vertical-align: middle;
      object-fit: contain;
      max-height: 320px;
    }}
    .card figcaption {{
      padding: 0.65rem 0.75rem;
      font-size: 0.75rem;
      word-break: break-all;
    }}
    .card code {{
      font-family: ui-monospace, Consolas, monospace;
      font-size: 0.72rem;
    }}
    .score {{
      display: block;
      margin-top: 0.35rem;
      font-weight: 600;
      color: #0b57d0;
    }}
    .empty, .missing {{
      color: #666;
      padding: 1rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{criteria}</h1>
    <p class="root">Library: <code>{root_esc}</code></p>
    <p class="count">{count} image(s)</p>
    {meta}
  </header>
  <div class="grid">
{gallery}
  </div>
</body>
</html>
"""


def write_gallery_from_matches(matches_json: Path, output_html: Path) -> None:
    payload = load_matches_json(matches_json)
    out = output_html.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_gallery_html(payload), encoding="utf-8")
