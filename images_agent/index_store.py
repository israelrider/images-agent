from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from numpy import float32
from numpy.typing import NDArray


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding (
            path TEXT PRIMARY KEY NOT NULL,
            mtime_ns INTEGER NOT NULL,
            size INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            model_pair TEXT NOT NULL,
            vector BLOB NOT NULL
        )
        """
    )
    conn.commit()


def upsert_embedding(
    conn: sqlite3.Connection,
    *,
    path: str,
    mtime_ns: int,
    size: int,
    dim: int,
    model_pair: str,
    vector: bytes,
) -> None:
    conn.execute(
        """
        INSERT INTO embedding (path, mtime_ns, size, dim, model_pair, vector)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          mtime_ns = excluded.mtime_ns,
          size = excluded.size,
          dim = excluded.dim,
          model_pair = excluded.model_pair,
          vector = excluded.vector
        """,
        (path, mtime_ns, size, dim, model_pair, vector),
    )


def row_matches_cache(
    conn: sqlite3.Connection,
    *,
    path: str,
    mtime_ns: int,
    size: int,
    model_pair: str,
) -> bool:
    row = conn.execute(
        """
        SELECT mtime_ns, size, model_pair
        FROM embedding
        WHERE path = ?
        """,
        (path,),
    ).fetchone()
    if row is None:
        return False
    rm, rs, rk = row
    return rm == mtime_ns and rs == size and rk == model_pair


def delete_all(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM embedding")
    conn.commit()


def vacuum(conn: sqlite3.Connection) -> None:
    conn.execute("VACUUM")
    conn.commit()


def prune_missing_paths(conn: sqlite3.Connection, paths: set[str]) -> int:
    cur = conn.execute("SELECT path FROM embedding")
    to_remove = [r[0] for r in cur.fetchall() if r[0] not in paths]
    if not to_remove:
        return 0
    conn.executemany("DELETE FROM embedding WHERE path = ?", [(p,) for p in to_remove])
    conn.commit()
    return len(to_remove)


def count_rows(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM embedding").fetchone()
    return int(row[0]) if row else 0


def path_under_root(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_normalized_matrix_under_root(
    conn: sqlite3.Connection,
    root: Path,
    *,
    expected_dim: int,
    expected_model_pair: str,
) -> tuple[list[str], NDArray[float32]]:
    root_res = root.resolve()
    cur = conn.execute(
        "SELECT path, dim, model_pair, vector FROM embedding WHERE dim = ? AND model_pair = ?",
        (expected_dim, expected_model_pair),
    )
    rows = cur.fetchall()
    paths: list[str] = []
    vectors: list[np.ndarray] = []
    for path_str, dim, pair, blob in rows:
        if dim != expected_dim or pair != expected_model_pair:
            continue
        p = Path(path_str)
        if not path_under_root(p, root_res):
            continue
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != expected_dim:
            continue
        paths.append(path_str)
        vectors.append(arr)
    if not paths:
        return [], np.empty((0, expected_dim), dtype=np.float32)
    mat = np.stack(vectors, axis=0)
    return paths, mat
