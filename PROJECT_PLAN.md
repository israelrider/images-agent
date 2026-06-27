# images-agent — project plan & continuation

## Goal

Organize ~10,000 mixed photos under `IMAGE_LIBRARY_ROOT` into **layered albums**:
one top folder per **person**, **scene** sub-folders inside, and (new) an
optional **age** layer. Denormalization is acceptable — a photo with several
people is copied under each person.

```
Albums/
├── Person_00/ <age?>/ <scene>/  ...
├── Person_01/ ...
├── No_People/ <scene>/ ...
├── Unsorted_Faces/ ...
└── album_manifest.csv
```

## Why this project won (vs. the deleted images-agent-2)

Both implemented the identical algorithm (InsightFace detect+embed → DBSCAN →
CLIP zero-shot scenes → copy). This project was kept because:

1. **Scales to 10k** — streams through a persistent **SQLite index**; the other
   held all records + embeddings in RAM and persisted nothing.
2. **Cheap tuning loop** — `faces` (expensive detect+embed) is separate from
   `cluster` (re-runs over stored vectors in seconds), so `--cluster-eps` can be
   re-tuned without re-detecting. The other forced a full re-detect each time.
3. **Incremental** — only new/changed files are reprocessed (mtime/size keyed).
4. **Near-free scene tagging** — scores scene-label text vectors against the
   CLIP image vectors already cached in `embedding`; no second image pass.
5. **Lighter stack** (FastEmbed ONNX vs. full torch) + faces as an optional extra.
6. **Packaged, versioned, modular**, with bonus `run` search and `gallery`.

## Architecture (already built)

- `index_store.py` — SQLite: `embedding` table (CLIP vectors) + (in `faces.py`)
  `face` table, same DB file.
- `clip_pipeline.py` — FastEmbed ONNX CLIP text/vision embeddings.
- `faces.py` — InsightFace `buffalo_l` detect+embed → `face` rows; DBSCAN
  clustering into person `cluster_id`.
- `albums.py` — reuses cached CLIP vectors for scene tagging; copies into the
  `Person/Scene` tree; writes `album_manifest.csv`.
- `cli.py` — `index`, `faces`, `cluster`, `albums`, plus `run`, `gallery`.

## Run order (verified end-to-end path)

```
pip install -e ".[faces]"                       # into the venv
images-agent index                              # CLIP index over the library
images-agent faces --root <small subfolder>     # smoke-test first, eyeball count
images-agent cluster --cluster-eps 0.40         # tune eps cheaply (stored vectors)
images-agent faces                              # full detect+embed+cluster of 10k
images-agent albums "<...>\_Albums" --dry-run   # writes manifest only, copies 0
images-agent albums "<...>\_Albums"             # builds Person/Scene tree (copy)
```

`--op copy` is the safe default (one photo can belong to several people).
`--op symlink` to save disk; `--op move` only if originals should be relocated.

---

## NEW FEATURE TO ADD: organize "by age"

InsightFace `buffalo_l` already returns a per-face **age estimate** (`face.age`,
plus `face.sex`/gender) at detection time — so age requires **no new model**,
only plumbing it through the existing pipeline.

### Design

Age is a property of a **face occurrence**, not of a person globally (the same
person appears at different ages across the library). So we store age per face
and apply it as an **optional layer between Person and Scene**.

1. **Schema** — add an `age REAL` column (and optionally `sex TEXT`) to the
   `face` table in `faces.init_face_schema`. Use `ALTER TABLE ... ADD COLUMN`
   guarded by a `PRAGMA table_info` check so existing indexes upgrade in place.
2. **Capture at detect time** — in `faces.run_face_index`, read
   `float(getattr(f, "age", -1))` (InsightFace populates it with the
   `genderage` sub-model in `buffalo_l`) and pass it to `insert_face`.
3. **Expose to albums** — extend `faces.person_clusters_by_path` (or add a
   sibling, e.g. `face_attrs_by_path`) to return, per path, the per-face
   `(cluster_id, age)` pairs so `albums.py` knows each person's age *in that
   photo*.
4. **Age buckets** — add `albums.age_bucket(age) -> str` mapping the estimate to
   a coarse, denormalization-friendly bucket. Default vocabulary:
   `baby (0–3)`, `child (4–12)`, `teen (13–19)`, `adult (20–39)`,
   `middle_aged (40–59)`, `senior (60+)`, and `age_unknown` when age < 0.
   Make boundaries overridable later if needed.
5. **CLI** — add an `--by-age` flag to `albums_cmd` (and pass through
   `run_build_albums`). When set, the destination becomes
   `Person_XX/<age_bucket>/<scene>/file`; when not set, behavior is unchanged
   (`Person_XX/<scene>/file`). Order is Person → Age → Scene so each person's
   folder stays the primary grouping. Combine cleanly with `--no-scenes`
   (→ `Person_XX/<age_bucket>/file`).
6. **Manifest** — add `age` and `age_bucket` columns to `album_manifest.csv`
   rows for auditability.

### Caveats to surface
- Age estimates are noisy (±5–10 yrs typical); buckets are intentionally coarse.
- `No_People` images have no age. `Unsorted_Faces` keep their per-face age.
- A person photographed across years will legitimately span several age buckets
  — expected, consistent with the copy-based denormalization model.

### Verification for the age feature
- `PRAGMA table_info(face)` shows the new `age` column; re-running `faces` on an
  already-indexed library fills age for new/changed files (or after `--rebuild`).
- `images-agent albums <dst> --by-age --dry-run` → manifest has populated
  `age`/`age_bucket`; spot-check a known child vs. adult photo lands in the right
  bucket.
- With `--by-age` off, the produced tree is byte-for-byte the prior layout.
