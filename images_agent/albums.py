"""
albums.py — materialize the library into layered Person/Scene album folders.

This is the step that turns analysis into the actual goal: a folder tree of
albums on disk. It reuses what the other modules already computed:

  * PERSON  comes from the `face` table (faces.py) -> one top folder per cluster.
  * SCENE   is derived from the CLIP image vectors ALREADY in the `embedding`
            table (index_store). We embed a fixed list of scene labels once with
            the text encoder and score each cached image vector against them.
            => scene tagging costs ZERO extra image embedding.

Layout produced:

    <albums_root>/
        Person_00/<scene>/<file>
        Person_01/<scene>/<file>
        No_People/<scene>/<file>      # images with no detected face
        Unsorted_Faces/<scene>/<file> # faces that didn't cluster (cluster -1)
        album_manifest.csv

Files are COPIED by default (an image with two people lands under both). A photo
can therefore appear more than once — that is intended.

This module is functionally complete: it depends only on infrastructure that
already exists and runs in this codebase. The only thing it needs is a populated
index (run `images-agent index` and `images-agent faces` first).
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

from images_agent import clip_pipeline, config, faces, index_store

DEFAULT_SCENE_LABELS = [
    "beach", "mountains", "city street", "indoor portrait",
    "group of people", "wedding", "birthday party", "food",
    "pets", "nature landscape", "document or screenshot",
    "car or vehicle", "sports", "concert", "sunset", "night photo",
]


def _safe(name: str) -> str:
    bad = '<>:"/\\|?*'
    s = "".join("_" if c in bad else c for c in str(name)).strip().replace(" ", "_")
    return s or "untitled"


def load_scene_labels(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_SCENE_LABELS)
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l] or list(DEFAULT_SCENE_LABELS)


def scene_for_each_path(
    paths: list[str],
    mat: np.ndarray,
    labels: list[str],
    min_scene_score: float | None,
) -> dict[str, str]:
    """Assign the best-matching scene label to each image vector (cached CLIP)."""
    label_vecs = np.stack([clip_pipeline.embed_text(f"a photo of {l}") for l in labels], axis=0)
    sims = mat @ label_vecs.T  # (n_images, n_labels)
    best_idx = np.argmax(sims, axis=1)
    best_score = sims[np.arange(sims.shape[0]), best_idx]
    out: dict[str, str] = {}
    for i, p in enumerate(paths):
        if min_scene_score is not None and float(best_score[i]) < min_scene_score:
            out[p] = "uncategorized"
        else:
            out[p] = labels[int(best_idx[i])]
    return out


def _unique_dest(dest_dir: Path, filename: str) -> Path:
    cand = dest_dir / filename
    if not cand.exists():
        return cand
    stem, suf = Path(filename).stem, Path(filename).suffix
    i = 1
    while (dest_dir / f"{stem}__{i}{suf}").exists():
        i += 1
    return dest_dir / f"{stem}__{i}{suf}"


def run_build_albums(
    root: Path,
    *,
    sqlite_path: Path,
    albums_root: Path,
    scene_labels_file: Path | None,
    min_scene_score: float | None,
    no_scenes: bool,
    op: str,            # "copy" | "move" | "symlink"
    dry_run: bool,
) -> None:
    root_resolved = root.resolve()
    conn = index_store._connect(sqlite_path)
    try:
        index_store.init_schema(conn)
        faces.init_face_schema(conn)
        pair = config.clip_model_pair_key()
        dim = clip_pipeline.text_embedding_dimension()
        paths, mat = index_store.load_normalized_matrix_under_root(
            conn, root_resolved, expected_dim=dim, expected_model_pair=pair
        )
        if not paths:
            sys.stderr.write(
                "No CLIP vectors for this library. Run `images-agent index` first.\n"
            )
            raise SystemExit(1)
        person_map = faces.person_clusters_by_path(conn, model=faces.FACE_MODEL_NAME)
    finally:
        conn.close()

    # scene per image (or none)
    if no_scenes:
        scene_map = {p: "" for p in paths}
    else:
        labels = load_scene_labels(scene_labels_file)
        sys.stderr.write(f"Scene vocabulary ({len(labels)}): {', '.join(labels)}\n")
        scene_map = scene_for_each_path(paths, mat, labels, min_scene_score)

    # any face row at all (to distinguish No_People from Unsorted_Faces)
    paths_with_faces = set(person_map.keys())

    albums_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[tuple] = []
    ops: list[tuple[Path, Path]] = []

    for p in paths:
        src = Path(p)
        scene = _safe(scene_map.get(p, "")) if not no_scenes else ""
        clusters = person_map.get(p, [])

        if not clusters:
            people = ["No_People"] if p not in paths_with_faces else ["Unsorted_Faces"]
        else:
            people = []
            for c in clusters:
                people.append("Unsorted_Faces" if c < 0 else f"Person_{c:02d}")
            people = sorted(set(people))

        for person in people:
            dest_dir = albums_root / person / scene if scene else albums_root / person
            ops.append((src, dest_dir / src.name))
        manifest_rows.append((p, len(clusters), ";".join(map(str, clusters)),
                              scene_map.get(p, "")))

    # manifest first (safety net, written even on dry run)
    manifest = albums_root / "album_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "num_people", "person_clusters", "scene"])
        w.writerows(manifest_rows)
    sys.stderr.write(f"Manifest: {manifest}\n")

    sys.stderr.write(f"Planned {len(ops)} {op} operations into {albums_root}.\n")
    if dry_run:
        sys.stderr.write("DRY RUN — nothing was written.\n")
        return

    done = 0
    for src, dest in ops:
        dest.parent.mkdir(parents=True, exist_ok=True)
        final = _unique_dest(dest.parent, dest.name)
        if op == "copy":
            shutil.copy2(src, final)
        elif op == "move":
            shutil.move(str(src), str(final))
        elif op == "symlink":
            try:
                final.symlink_to(src)
            except OSError as e:
                sys.stderr.write(f"symlink failed ({e}); copying instead: {src}\n")
                shutil.copy2(src, final)
        done += 1
        if done % 500 == 0:
            print(f"{done}/{len(ops)}\r", end="", flush=True)
    print()
    sys.stderr.write(f"Done: {done} files placed into albums under {albums_root}.\n")
