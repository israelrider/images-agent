# images-agent

A local, offline-first toolkit for a large personal photo library. It does two
things:

1. **Semantic search** — embed every photo with CLIP (FastEmbed ONNX) into a
   SQLite vector index, then rank by a natural-language query and view the hits
   as an HTML gallery.
2. **Album organization** — detect faces (InsightFace `buffalo_l`), cluster them
   into people, tag each photo with a scene, and build a denormalized
   `Person / age / scene` album tree by copying (or symlinking) the originals.

Everything runs locally. Only the optional *legacy / hybrid-rerank* vision path
calls out to an Ollama Cloud model; plain CLIP search and the whole album
pipeline are fully offline.

> Design notes and hardware sizing live in [`docs/`](docs/):
> [PROJECT_PLAN](docs/PROJECT_PLAN.md),
> [IMPLEMENTATION_PLAN](docs/IMPLEMENTATION_PLAN.md),
> [HARDWARE_NOTES](docs/HARDWARE_NOTES.md).

---

## Install

```bash
pip install -e .            # core: search + albums (CLIP, SQLite, scikit-learn)
pip install -e ".[faces]"   # adds face detection (insightface, onnxruntime, opencv, pillow-heif)
```

- The `[faces]` extra is required only for the `faces` / `cluster` / `albums`
  person-grouping pipeline. Plain search works with the core install.
- First runs download models: ~0.6 GB CLIP (FastEmbed) and ~0.3 GB `buffalo_l`
  (InsightFace), cached locally.
- **GPU (optional):** swap `onnxruntime` for `onnxruntime-gpu`, then pass `--gpu`
  to `faces`. Face detection is the only heavy step; everything else is CPU-fine.
- HEIC/iPhone photos are handled via `pillow-heif` (in the `[faces]` extra).

### Configuration (`.env`)

Copy [`.env.example`](.env.example) to `.env`:

| Variable | Purpose |
|---|---|
| `IMAGE_LIBRARY_ROOT` | Root folder of your photos (default `C:\Users\user\Pictures`). |
| `IMAGE_INDEX_SQLITE` | Index path (default `~/.cache/images-agent/embeddings.sqlite`). |
| `IMAGE_AGENT_VECTOR_TOP_K` | Default number of search hits (default 100). |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_VISION_MODEL` | Only for `--legacy-full-vision` or `--rerank`. |

If `images-agent` isn't on your PATH, every command also works as
`python -m images_agent.cli <command>`.

---

## Quick start — organize photos into person albums

This is the main pipeline. The CLIP index is shared between search and scene
tagging, so build it once.

```bash
# 1. Index the library with CLIP (also powers scene tags later)
images-agent index

# 2. Detect + embed faces, then cluster into people (start on a SMALL subfolder!)
images-agent faces --root "C:\Users\user\Pictures\<small subfolder>"

# 3. Tune clustering cheaply over stored vectors (no re-detection)
images-agent cluster --cluster-eps 0.45

# 4. Full face pass once eps looks right (the heavy, one-time step)
images-agent faces

# 5. Dry-run the albums — writes only album_manifest.csv, copies nothing
images-agent albums "C:\Users\user\Pictures\_Albums" --by-age --dry-run

# 6. Build for real (copy is the safe default)
images-agent albums "C:\Users\user\Pictures\_Albums" --by-age
```

Result:

```
_Albums/
├── Person_00/ <age_bucket>/ <scene>/ ...
├── Person_01/ ...
├── No_People/ <scene>/ ...          # photos with no detected face
├── Unsorted_Faces/ ...              # faces that didn't cluster (cluster -1)
└── album_manifest.csv               # one row per image: people, scene, age, age_bucket
```

A photo with several people is **copied under each person** (denormalized), so
`_Albums` can be larger than the library. With `--by-age`, the layer order is
`Person → age_bucket → scene`.

**Age buckets:** `child` (≤17) · `adult` (18–59) · `senior` (60+) ·
`age_unknown`. Omit `--by-age` for a plain `Person/scene` tree.

> ⚠️ **`--by-age` accuracy is poor on real-world libraries.** `buffalo_l`'s
> built-in age estimate is not just noisy — it fails *catastrophically* on dim /
> non-frontal / child faces (observed: a ~10-year-old tagged 47 and 60). It is
> usable only as the crudest split. For trustworthy per-person age organization
> we instead estimate age from the face crop with **MiVOLO** (a dedicated
> image-only age model), which is robust to those poses. See
> [docs/AGE_ESTIMATION.md](docs/AGE_ESTIMATION.md) for the full comparison,
> findings, and the per-person life-stage workflow.

---

## Command reference

### `index` — build the CLIP vector index
```bash
images-agent index [--root DIR] [--index PATH] [--rebuild] [--vacuum] [--batch-size N]
```
Embeds all images under the library root into SQLite. Incremental: only
new/changed files (keyed by mtime + size) are re-embedded. `--rebuild` starts
fresh.

### `run` — semantic search
```bash
images-agent run "photos with a dog" [-n K] [--rerank K] [--min-similarity F] [-o out.json]
```
Ranks the index by CLIP cosine similarity and writes `matches.json`
(`paths` + `scores`). By default returns the top 100 (`IMAGE_AGENT_VECTOR_TOP_K`);
`-n 0` exports the entire library ranked. `--rerank K` re-confirms the top *K*
hits with an Ollama vision model; `--legacy-full-vision` skips the index and
sends every image to the model (slow).

### `gallery` — HTML view of results
```bash
images-agent gallery [-i matches.json] [-o gallery.html] [--open]
```
Renders a thumbnail grid using `file://` URLs (loads offline). `--open` launches
your browser.

### `faces` — detect, embed, and cluster faces
```bash
images-agent faces [--root DIR] [--gpu] [--rebuild] [--det-size 640]
                   [--min-det-score 0.5] [--cluster/--no-cluster]
                   [--cluster-eps 0.45] [--cluster-min-samples 3]
```
Stores one row per face (vector, bbox, detection score, **age**, **sex**) in the
same SQLite file, then clusters into people. Incremental like `index`. Detection
is the heaviest step (~1.5–3 h for 10k photos on CPU) — smoke-test on a small
`--root` first.

### `cluster` — re-cluster only (cheap, for tuning)
```bash
images-agent cluster [--cluster-eps 0.45] [--cluster-min-samples 3]
```
Re-runs DBSCAN over already-stored face vectors in seconds. Use this to tune
`--cluster-eps` without re-detecting.

### `albums` — build the album tree
```bash
images-agent albums <ALBUMS_ROOT> [--by-age] [--no-scenes] [--people-only]
                    [--op copy|move|symlink] [--dry-run]
                    [--scene-labels FILE] [--min-scene-score F]
```
Reuses the cached CLIP vectors to tag scenes (zero extra image embedding) and the
face clusters to group people. `--dry-run` writes only the manifest.
`--people-only` keeps just real `Person_XX` folders, dropping `No_People` and
`Unsorted_Faces` (e.g. `albums DST --no-scenes --people-only` → one flat folder
of files per person). `--op symlink` avoids duplication (Windows symlinks need
Developer Mode/admin; falls back to copy if denied). `--op move` relocates
originals — use with care.
Default scene vocabulary (16): beach, mountains, city street, indoor portrait,
group of people, wedding, birthday party, food, pets, nature landscape, document
or screenshot, car or vehicle, sports, concert, sunset, night photo. Override
with `--scene-labels` (one label per line).

---

## Tuning reference

| Symptom | Knob |
|---|---|
| Duplicate folders for one person (over-split) | `--cluster-eps` up (e.g. 0.50) |
| Two different people merged into one folder | `--cluster-eps` down (e.g. 0.40) |
| A person needs fewer shots to count | `--cluster-min-samples` down |
| Weak / false-positive faces | `--min-det-score` up (e.g. 0.6) |
| Wrong scene buckets | edit the vocabulary / pass `--scene-labels` |
| Disk pressure from copies | `--op symlink` |

> **A note on `--cluster-eps`:** raising it does **not** merge fragments of one
> person — past ~0.5 it chains *distinct* people into one giant cluster (cosine
> DBSCAN density-linking). Lower is stricter. On a ~7k-photo library, `0.45`
> separated people cleanly (485 clusters); 0.60 collapsed most faces into a
> single blob. Tune downward, not up.

Person folders are **numbered, not named** — rename them after the first run.
Note that re-running `albums` regenerates numbered folders, so renames are
applied to the output tree, not preserved across rebuilds.

---

## How it fits together (modules)

- [`config.py`](images_agent/config.py) — `.env` loading and env accessors.
- [`scan.py`](images_agent/scan.py) — recursive, case-insensitive image scan.
- [`index_store.py`](images_agent/index_store.py) — SQLite `embedding` table (CLIP vectors).
- [`clip_pipeline.py`](images_agent/clip_pipeline.py) — FastEmbed ONNX CLIP text/vision embeddings.
- [`search.py`](images_agent/search.py) — vector search, incremental indexing, legacy vision scan.
- [`faces.py`](images_agent/faces.py) — InsightFace detect/embed → `face` table; DBSCAN clustering.
- [`albums.py`](images_agent/albums.py) — scene tagging + Person/age/scene tree builder + manifest.
- [`gallery_html.py`](images_agent/gallery_html.py) — HTML gallery from `matches.json`.
- [`cli.py`](images_agent/cli.py) — Typer CLI: `index`, `run`, `gallery`, `faces`, `cluster`, `albums`.
