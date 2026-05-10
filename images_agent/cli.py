from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from images_agent import config
from images_agent.dupes_exact import find_duplicate_groups, format_dupes_human, format_dupes_json
from images_agent.router import RoutedTask, ambiguous_hint, route_task
from images_agent.sort_age import run_sort_age

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _resolve_root(root: Optional[Path]) -> Path:
    if root is not None:
        return root.expanduser().resolve()
    return config.image_library_root()


@app.command("dupes")
def dupes(
    root: Optional[Path] = typer.Option(
        None,
        "--root",
        "-r",
        help="Library root directory (default: IMAGE_LIBRARY_ROOT from .env)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit duplicate groups as JSON"),
):
    """Find byte-identical images under the library root (SHA-256 of file bytes)."""
    resolved = _resolve_root(root)
    groups = find_duplicate_groups(resolved)
    if json_out:
        typer.echo(format_dupes_json(groups), nl=False)
    else:
        typer.echo(format_dupes_human(groups), nl=False)


@app.command("sort-age")
def sort_age_cmd(
    root: Optional[Path] = typer.Option(
        None,
        "--root",
        "-r",
        help="Library root directory (default: IMAGE_LIBRARY_ROOT from .env)",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        help="Maximum number of images to send to the model (ignored with --no-limit)",
    ),
    no_limit: bool = typer.Option(
        False,
        "--no-limit",
        help="Analyze every image found under the root (may be slow or costly)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Vision model name (default: OLLAMA_VISION_MODEL)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit ranked rows as JSON"),
):
    """Estimate apparent child age per photo via Ollama Cloud vision model and sort ascending."""
    resolved = _resolve_root(root)
    lim = None if no_limit else limit
    m = (model or "").strip() or config.ollama_vision_model()
    run_sort_age(resolved, model=m, limit=lim, as_json=json_out)


@app.command("run")
def run_cmd(
    task: str = typer.Argument(..., help="Short natural-language description of what you want"),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
    limit: int = typer.Option(50, "--limit", "-n"),
    no_limit: bool = typer.Option(False, "--no-limit"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Route a short phrase to dupes or sort-age based on keywords."""
    routed = route_task(task)
    if routed == RoutedTask.DUPLICATES:
        dupes(root=root, json_out=json_out)
    elif routed == RoutedTask.SORT_AGE:
        sort_age_cmd(root=root, limit=limit, no_limit=no_limit, model=model, json_out=json_out)
    else:
        typer.echo(ambiguous_hint(), err=True)
        raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
