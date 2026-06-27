"""
faces.py — face detection, embedding, and clustering.

This module adds the one capability the original images-agent lacks: grouping
photos by *person*. It mirrors the existing CLIP pipeline's design:

  * its own SQLite table (`face`) living in the SAME index file as the CLIP
    `embedding` table, so everything stays in one place;
  * incremental work — a photo whose (mtime, size, model) already has rows is
    skipped on re-runs;
  * detection + embedding via InsightFace (buffalo_l), clustering via DBSCAN
    over cosine distance on the normalized embeddings.

Two entry points:
  * run_face_index(...)   -> detect + embed every image, store one row per face
  * run_face_clustering(...) -> cluster all stored face vectors into person ids

NOTE FOR THE IMPLEMENTING SESSION (Claude Code, on the real machine):
  The detection/clustering logic below is complete and intended to run as-is,
  but it has NOT been executed against the real library here (no GPU/photos in
  this environment). Run `images-agent faces` on a SMALL subfolder first, eyeball
  the cluster count, then tune --cluster-eps before the full 10k run. Points that
  most likely need tuning are marked `# TUNE`.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from images_agent import config, index_store
from images_agent.scan import iter_image_paths

FACE_MODEL_NAME = "buffalo_l"  # InsightFace model pack


# --------------------------------------------------------------------------- #
# Schema (shares the CLIP index sqlite file)
# --------------------------------------------------------------------------- #
def init_face_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS face (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT NOT NULL,
            mtime_ns    INTEGER NOT NULL,
            size        INTEGER NOT NULL,
            model       TEXT NOT NULL,
            bbox        TEXT NOT NULL,      -- json [x1,y1,x2,y2]
            det_score   REAL NOT NULL,
            dim         INTEGER NOT NULL,
            vector      BLOB NOT NULL,      -- normalized float32
            cluster_id  INTEGER,            -- filled by clustering; -1 = unsorted
            age         REAL,               -- buffalo_l genderage estimate; <0 = unknown
            sex         TEXT                -- buffalo_l gender ('M'/'F') or NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_face_path ON face(path)")
    # In-place upgrade for face tables created before age/sex existed.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(face)")}
    if "age" not in cols:
        conn.execute("ALTER TABLE face ADD COLUMN age REAL")
    if "sex" not in cols:
        conn.execute("ALTER TABLE face ADD COLUMN sex TEXT")
    conn.commit()


def path_already_indexed(conn, *, path: str, mtime_ns: int, size: int, model: str) -> bool:
    row = conn.execute(
        "SELECT mtime_ns, size FROM face WHERE path = ? AND model = ? LIMIT 1",
        (path, model),
    ).fetchone()
    return bool(row) and row[0] == mtime_ns and row[1] == size


def delete_faces_for_path(conn, path: str) -> None:
    conn.execute("DELETE FROM face WHERE path = ?", (path,))


def insert_face(conn, *, path, mtime_ns, size, model, bbox, det_score, dim, vector, age=-1.0, sex=None) -> None:
    conn.execute(
        """
        INSERT INTO face (path, mtime_ns, size, model, bbox, det_score, dim, vector, cluster_id, age, sex)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        (path, mtime_ns, size, model, json.dumps(bbox), float(det_score), dim, vector,
         float(age), sex),
    )


def load_all_face_vectors(conn, *, model: str) -> tuple[list[int], NDArray[np.float32]]:
    """Return (row_ids, matrix) for every stored face of the given model."""
    cur = conn.execute(
        "SELECT id, dim, vector FROM face WHERE model = ? ORDER BY id", (model,)
    )
    ids: list[int] = []
    vecs: list[np.ndarray] = []
    dim_seen = None
    for fid, dim, blob in cur.fetchall():
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != dim:
            continue
        if dim_seen is None:
            dim_seen = dim
        ids.append(int(fid))
        vecs.append(arr)
    if not ids:
        return [], np.empty((0, dim_seen or 512), dtype=np.float32)
    return ids, np.stack(vecs, axis=0)


def write_cluster_ids(conn, id_to_cluster: dict[int, int]) -> None:
    conn.executemany(
        "UPDATE face SET cluster_id = ? WHERE id = ?",
        [(cid, fid) for fid, cid in id_to_cluster.items()],
    )
    conn.commit()


def person_clusters_by_path(conn, *, model: str) -> dict[str, list[int]]:
    """Map each image path -> sorted list of distinct person cluster ids (>=0)."""
    cur = conn.execute(
        "SELECT path, cluster_id FROM face WHERE model = ? AND cluster_id IS NOT NULL",
        (model,),
    )
    out: dict[str, set[int]] = {}
    for path, cid in cur.fetchall():
        out.setdefault(path, set()).add(int(cid))
    return {p: sorted(c) for p, c in out.items()}


def face_attrs_by_path(conn, *, model: str) -> dict[str, list[tuple[int, float]]]:
    """Map each image path -> list of (cluster_id, age) per detected face.

    cluster_id may be -1 (unsorted); age is the buffalo_l estimate or <0 if
    unknown/missing. Unlike person_clusters_by_path this is NOT deduped, because
    age is a per-face occurrence (the same person can appear at several ages).
    """
    cur = conn.execute(
        "SELECT path, cluster_id, age FROM face WHERE model = ? AND cluster_id IS NOT NULL",
        (model,),
    )
    out: dict[str, list[tuple[int, float]]] = {}
    for path, cid, age in cur.fetchall():
        out.setdefault(path, []).append((int(cid), float(age) if age is not None else -1.0))
    return out


# --------------------------------------------------------------------------- #
# Detection + embedding
# --------------------------------------------------------------------------- #
def _load_face_app(det_size: int, ctx_id: int):
    """Lazy import so --help and non-face commands don't need InsightFace."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=FACE_MODEL_NAME)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def _read_image_bgr(path: Path):
    import cv2
    import numpy as np

    img = cv2.imread(str(path))
    if img is not None:
        return img
    # Fallback for formats cv2 can't open directly (e.g. HEIC).
    try:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def run_face_index(
    root: Path,
    *,
    sqlite_path: Path,
    rebuild: bool,
    det_size: int,
    use_gpu: bool,
    min_det_score: float,
) -> None:
    """Detect faces in every image under root; store one row per face."""
    conn = index_store._connect(sqlite_path)
    try:
        init_face_schema(conn)
        if rebuild:
            conn.execute("DELETE FROM face")
            conn.commit()
            sys.stderr.write("Cleared existing face rows (--rebuild).\n")

        paths = sorted(iter_image_paths(root), key=lambda p: str(p).lower())
        total = len(paths)
        if not total:
            sys.stderr.write("No images found under root.\n")
            return

        ctx_id = 0 if use_gpu else -1
        app = _load_face_app(det_size, ctx_id)

        processed = added = skipped = failed = 0
        for pth in paths:
            key = str(pth.resolve())
            try:
                st = pth.stat()
            except OSError as e:
                sys.stderr.write(f"stat fail {pth}: {e}\n")
                failed += 1
                continue
            if path_already_indexed(conn, path=key, mtime_ns=st.st_mtime_ns,
                                    size=st.st_size, model=FACE_MODEL_NAME):
                skipped += 1
                continue

            # changed file: drop stale rows then re-detect
            delete_faces_for_path(conn, key)
            img = _read_image_bgr(pth)
            if img is None:
                failed += 1
                continue
            try:
                faces = app.get(img)
            except Exception as e:
                sys.stderr.write(f"detect fail {pth}: {e}\n")
                failed += 1
                continue

            for f in faces:
                if float(getattr(f, "det_score", 1.0)) < min_det_score:  # TUNE
                    continue
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    continue
                v = np.asarray(emb, dtype=np.float32).reshape(-1)
                bbox = [float(x) for x in (f.bbox.tolist() if hasattr(f.bbox, "tolist") else f.bbox)]
                # buffalo_l's genderage sub-model fills these; default to unknown if absent.
                age = float(getattr(f, "age", -1))
                sex = getattr(f, "sex", None)
                sex = str(sex) if sex is not None else None
                insert_face(
                    conn,
                    path=key, mtime_ns=st.st_mtime_ns, size=st.st_size,
                    model=FACE_MODEL_NAME, bbox=bbox, det_score=float(getattr(f, "det_score", 1.0)),
                    dim=v.shape[0], vector=v.tobytes(), age=age, sex=sex,
                )
                added += 1
            processed += 1
            if processed % 100 == 0:
                conn.commit()
                print(f"{processed}/{total} images  (+{added} faces, {skipped} skipped)\r",
                      end="", flush=True)
        conn.commit()
        print()
        sys.stderr.write(
            f"Face index done: {processed} new images, {added} faces stored, "
            f"{skipped} unchanged, {failed} failed.\n"
        )
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def run_face_clustering(
    *,
    sqlite_path: Path,
    eps: float,
    min_samples: int,
) -> int:
    """Cluster all stored face vectors into people; write cluster_id back."""
    from sklearn.cluster import DBSCAN

    conn = index_store._connect(sqlite_path)
    try:
        init_face_schema(conn)
        ids, X = load_all_face_vectors(conn, model=FACE_MODEL_NAME)
        if not ids:
            sys.stderr.write("No faces stored. Run `images-agent faces` first.\n")
            return 0
        sys.stderr.write(f"Clustering {len(ids)} faces (eps={eps}, min_samples={min_samples})...\n")
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(X)  # TUNE
        write_cluster_ids(conn, {fid: int(lbl) for fid, lbl in zip(ids, labels)})
        n_people = len({int(l) for l in labels if l != -1})
        n_noise = int(np.sum(labels == -1))
        sys.stderr.write(f"Found {n_people} people; {n_noise} faces left unsorted (-1).\n")
        return n_people
    finally:
        conn.close()
