from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from images_agent.scan import iter_image_paths

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of raw file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_paths_with_size(paths: Iterable[Path]):
    for path in paths:
        try:
            yield path, path.stat().st_size
        except OSError:
            continue


def find_duplicate_groups(root: Path) -> list[list[str]]:
    """
    Byte-identical files only: groups of two or more paths sharing the same SHA-256.
    Uses size bucketing before hashing to skip unique sizes.
    """
    size_buckets: dict[int, list[Path]] = defaultdict(list)
    for path, size in iter_paths_with_size(iter_image_paths(root)):
        size_buckets[size].append(path)

    hash_buckets: dict[str, list[Path]] = defaultdict(list)
    for _size, paths in size_buckets.items():
        if len(paths) < 2:
            continue
        for path in paths:
            try:
                h = sha256_file(path)
            except OSError:
                continue
            hash_buckets[h].append(path)

    groups: list[list[str]] = []
    for paths in hash_buckets.values():
        if len(paths) >= 2:
            groups.append(sorted(str(p) for p in paths))
    groups.sort(key=lambda g: g[0])
    return groups


def format_dupes_human(groups: list[list[str]]) -> str:
    if not groups:
        return "No duplicate image files (byte-identical) found.\n"
    lines: list[str] = []
    for i, paths in enumerate(groups, start=1):
        lines.append(f"Group {i} ({len(paths)} files, same SHA-256):")
        for p in paths:
            lines.append(f"  {p}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_dupes_json(groups: list[list[str]]) -> str:
    return json.dumps(groups, indent=2, ensure_ascii=False) + "\n"
