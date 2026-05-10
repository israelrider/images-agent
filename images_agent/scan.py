from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".heif",
        ".heic",
    }
)


def iter_image_paths(root: Path):
    """Yield existing files under root whose suffix matches IMAGE_EXTENSIONS (case-insensitive)."""
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf not in IMAGE_EXTENSIONS:
            continue
        yield path.resolve()
