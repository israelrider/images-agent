from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional

import typer

from images_agent import config
from images_agent.gallery_html import write_gallery_from_matches
from images_agent.search import run_criteria_search, run_incremental_embedding_index, run_vector_search
from images_agent import albums as albums_mod
from images_agent import faces as faces_mod

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _cli() -> None:
    """CLIP vector search over a local image library (optional hybrid Ollama Cloud vision rerank)."""


def _resolve_root(root: Optional[Path]) -> Path:
    if root is not None:
        return root.expanduser().resolve()
    return config.image_library_root()


@app.command("index")
def index_cmd(
    root: Optional[Path] = typer.Option(
        None,
        "--root",
        "-r",
        help="Library root directory (default: IMAGE_LIBRARY_ROOT from .env)",
    ),
    sqlite: Optional[Path] = typer.Option(
        None,
        "--index",
        "-i",
        help="SQLite index path (default: IMAGE_INDEX_SQLITE or ~/.cache/images-agent/embeddings.sqlite)",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Drop all embeddings in the index file before rebuilding",
    ),
    vacuum: bool = typer.Option(
        False,
        "--vacuum",
        help="Run SQLite VACUUM after indexing",
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        min=1,
        help="Image paths per ONNX batch",
    ),
) -> None:
    """Embed all images under the library root into the local SQLite vector index."""
    resolved = _resolve_root(root)
    db = sqlite.expanduser().resolve() if sqlite is not None else config.image_index_sqlite()
    run_incremental_embedding_index(
        resolved,
        sqlite_path=db,
        rebuild=rebuild,
        do_vacuum=vacuum,
        batch_size=batch_size,
    )


@app.command("run")
def run_cmd(
    criteria: str = typer.Argument(
        ...,
        help='Natural-language description of what to find (CLIP text query, e.g. photos with a dog).',
    ),
    root: Optional[Path] = typer.Option(
        None,
        "--root",
        "-r",
        help="Library root directory (default: IMAGE_LIBRARY_ROOT from .env)",
    ),
    sqlite: Optional[Path] = typer.Option(
        None,
        "--index",
        "-i",
        help="SQLite vector index path (same as indexing; default IMAGE_INDEX_SQLITE)",
    ),
    limit_n: Optional[int] = typer.Option(
        None,
        "-n",
        "--limit",
        "--top-k",
        help="CLIP mode: return at most K best vector matches (default: IMAGE_AGENT_VECTOR_TOP_K or 100; "
        "use 0 for the full ranked library). "
        "Legacy (--legacy-full-vision): max images to send to the vision model (omit = all).",
    ),
    min_similarity: Optional[float] = typer.Option(
        None,
        "--min-similarity",
        help="Drop CLIP cosine scores below this (dot product after L2 normalize; roughly -1 … 1).",
    ),
    rerank: int = typer.Option(
        0,
        "--rerank",
        min=0,
        help="If >0, take this many top vector hits (after --top-k window, if set) and confirm with Ollama vision.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Vision model for legacy full scan or hybrid rerank (default: OPENAI_VISION_MODEL)",
    ),
    legacy_full_vision: bool = typer.Option(
        False,
        "--legacy-full-vision",
        help="Old behavior: send every image to the vision model (slow; no CLIP index required).",
    ),
    output: Path = typer.Option(
        Path("matches.json"),
        "--output",
        "-o",
        help="JSON file to write (paths of matching photos; includes scores in CLIP mode)",
    ),
) -> None:
    """Query the CLIP index (default) or run a full-library vision scan (legacy)."""
    resolved = _resolve_root(root)
    m = (model or "").strip() or config.ollama_vision_model()

    if legacy_full_vision:
        if not m:
            typer.echo(
                "No vision model configured. Set OPENAI_VISION_MODEL or pass --model.",
                err=True,
            )
            raise typer.Exit(code=2)
        run_criteria_search(resolved, criteria, model=m, limit=limit_n, output=output)
        return

    if rerank > 0 and not m:
        typer.echo(
            "Hybrid rerank needs a vision model. Set OPENAI_VISION_MODEL or pass --model.",
            err=True,
        )
        raise typer.Exit(code=2)

    db = sqlite.expanduser().resolve() if sqlite is not None else config.image_index_sqlite()
    if limit_n is None:
        top_k = config.default_vector_top_k()
        typer.echo(
            f"Note: Returning top {top_k} matches by CLIP similarity "
            "(override with -n/--limit; use -n 0 for all ranked images; "
            "or set IMAGE_AGENT_VECTOR_TOP_K).",
            err=True,
        )
    elif limit_n == 0:
        top_k = None
    else:
        top_k = limit_n

    run_vector_search(
        resolved,
        criteria,
        sqlite_path=db,
        top_k=top_k,
        min_similarity=min_similarity,
        rerank=rerank,
        vision_model=m,
        output=output,
    )


@app.command("gallery")
def gallery_cmd(
    matches_in: Path = typer.Option(
        Path("matches.json"),
        "--input",
        "-i",
        help="JSON from images-agent run (paths + optional scores)",
    ),
    output: Path = typer.Option(
        Path("gallery.html"),
        "--output",
        "-o",
        help="HTML gallery file to write",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the written HTML in the default browser",
    ),
) -> None:
    """Build a local HTML gallery from a matches.json file (file:// image URLs)."""
    m = matches_in.expanduser().resolve()
    if not m.is_file():
        typer.echo(f"No such file: {m}", err=True)
        raise typer.Exit(code=1)
    html_out = output.expanduser().resolve()
    write_gallery_from_matches(m, html_out)
    typer.echo(f"Wrote {html_out}")
    if open_browser:
        webbrowser.open(html_out.as_uri())


@app.command("faces")
def faces_cmd(
    root: Optional[Path] = typer.Option(None, "--root", "-r",
        help="Library root (default: IMAGE_LIBRARY_ROOT)"),
    sqlite: Optional[Path] = typer.Option(None, "--index", "-i",
        help="SQLite index path (shared with the CLIP index)"),
    rebuild: bool = typer.Option(False, "--rebuild",
        help="Drop all stored faces before detecting"),
    det_size: int = typer.Option(640, "--det-size", min=128,
        help="Face detector input size"),
    gpu: bool = typer.Option(False, "--gpu",
        help="Use GPU (ctx_id=0) for InsightFace; default CPU"),
    min_det_score: float = typer.Option(0.5, "--min-det-score",
        help="Drop weak face detections below this score"),
    cluster: bool = typer.Option(True, "--cluster/--no-cluster",
        help="Cluster faces into people after detecting"),
    eps: float = typer.Option(0.45, "--cluster-eps",
        help="DBSCAN cosine eps. Lower = stricter / more people"),
    min_samples: int = typer.Option(3, "--cluster-min-samples", min=1,
        help="Min faces to form a person cluster"),
) -> None:
    """Detect + embed faces, then cluster them into per-person groups."""
    resolved = _resolve_root(root)
    db = sqlite.expanduser().resolve() if sqlite is not None else config.image_index_sqlite()
    faces_mod.run_face_index(
        resolved, sqlite_path=db, rebuild=rebuild, det_size=det_size,
        use_gpu=gpu, min_det_score=min_det_score,
    )
    if cluster:
        faces_mod.run_face_clustering(sqlite_path=db, eps=eps, min_samples=min_samples)


@app.command("cluster")
def cluster_cmd(
    sqlite: Optional[Path] = typer.Option(None, "--index", "-i",
        help="SQLite index path (shared with the CLIP index)"),
    eps: float = typer.Option(0.45, "--cluster-eps",
        help="DBSCAN cosine eps. Lower = stricter / more people"),
    min_samples: int = typer.Option(3, "--cluster-min-samples", min=1,
        help="Min faces to form a person cluster"),
) -> None:
    """Re-cluster already-detected faces (cheap; tune --cluster-eps here)."""
    db = sqlite.expanduser().resolve() if sqlite is not None else config.image_index_sqlite()
    faces_mod.run_face_clustering(sqlite_path=db, eps=eps, min_samples=min_samples)


@app.command("albums")
def albums_cmd(
    albums_root: Path = typer.Argument(..., help="Folder where album folders are created"),
    root: Optional[Path] = typer.Option(None, "--root", "-r",
        help="Library root (default: IMAGE_LIBRARY_ROOT)"),
    sqlite: Optional[Path] = typer.Option(None, "--index", "-i",
        help="SQLite index path (shared with the CLIP index)"),
    scene_labels: Optional[Path] = typer.Option(None, "--scene-labels",
        help="Text file, one scene label per line (overrides built-in vocab)"),
    min_scene_score: Optional[float] = typer.Option(None, "--min-scene-score",
        help="Below this CLIP score a photo goes to 'uncategorized'"),
    no_scenes: bool = typer.Option(False, "--no-scenes",
        help="Person-only albums (no scene subfolders)"),
    op: str = typer.Option("copy", "--op",
        help="How files land in albums: copy | move | symlink"),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Plan + write manifest only; place no files"),
) -> None:
    """Build Person/Scene album folders, reusing the CLIP + face indexes."""
    resolved = _resolve_root(root)
    db = sqlite.expanduser().resolve() if sqlite is not None else config.image_index_sqlite()
    if op not in {"copy", "move", "symlink"}:
        typer.echo("--op must be one of: copy, move, symlink", err=True)
        raise typer.Exit(code=2)
    albums_mod.run_build_albums(
        resolved, sqlite_path=db, albums_root=albums_root.expanduser().resolve(),
        scene_labels_file=scene_labels, min_scene_score=min_scene_score,
        no_scenes=no_scenes, op=op, dry_run=dry_run,
    )


def main() -> None:
    app()


if __name__ == "__main__":  # entry point
    main()
