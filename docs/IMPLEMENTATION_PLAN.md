# Implementation plan — add face grouping + album foldering

> Handoff brief for a Claude Code session running on the real machine.
> Read this first. The architecture decisions are already made; your job is to
> install, run against the real library, tune, and verify — not to redesign.

## Goal

Order ~10,000 photos under `IMAGE_LIBRARY_ROOT` (currently `C:\Users\user\Pictures`)
into **layered albums**: one top folder per person, scene sub-folders inside.

```
Albums/
├── Person_00/ beach/ , group_of_people/ ...
├── Person_01/ ...
├── No_People/ food/ , document_or_screenshot/ ...
├── Unsorted_Faces/ ...
└── album_manifest.csv
```

## What already existed (do not rebuild)

`images-agent` is a CLIP **search** tool: `index` builds a SQLite vector index
(FastEmbed ONNX CLIP ViT-B-32) over the library; `run` ranks by text query;
`gallery` renders results. It has no face capability and never writes albums.
We keep all of it and build *on top* of its SQLite index.

## What was added in this handoff (skeleton, needs real-machine run)

| File | Status | Notes |
|------|--------|-------|
| `images_agent/faces.py` | complete, **untested on real photos** | InsightFace detect+embed → `face` table; DBSCAN clustering |
| `images_agent/albums.py` | complete, depends only on existing infra | scene tagging reuses cached CLIP vectors; copies into Person/Scene tree |
| `images_agent/cli.py` | wired | new commands `faces`, `cluster`, `albums` |
| `pyproject.toml` | updated | `scikit-learn` core; `[faces]` extra for insightface/onnxruntime/opencv/pillow-heif |

Key design win: **scene tagging adds zero image embedding.** `albums.py` embeds
the scene label strings once with the existing text encoder and scores them
against the image vectors already cached in the `embedding` table. Faces are the
only genuinely new compute.

The `face` table lives in the **same** SQLite file as the CLIP `embedding` table.

## Your steps (in order)

1. **Install the face extra into the existing venv:**
   ```
   pip install -e ".[faces]"
   ```
   For GPU: replace `onnxruntime` with `onnxruntime-gpu` and install a CUDA build
   of nothing else needed (InsightFace uses onnxruntime). First run downloads the
   `buffalo_l` model pack (~300 MB).

2. **Make sure the CLIP index exists** (reused for scenes):
   ```
   images-agent index
   ```

3. **Smoke-test faces on a SMALL subfolder first** (do NOT start with 10k):
   ```
   images-agent faces --root "C:\Users\user\Pictures\<some small subfolder>"
   ```
   Check the printed people count looks sane.

4. **Tune clustering cheaply** (re-clusters stored vectors, no re-detection):
   ```
   images-agent cluster --cluster-eps 0.40
   images-agent cluster --cluster-eps 0.50
   ```
   - too many duplicate people → raise eps
   - different people merged → lower eps

5. **Full face index** once eps is chosen:
   ```
   images-agent faces            # detect+embed+cluster the whole library
   ```

6. **Dry-run the albums** (writes only `album_manifest.csv`, copies nothing):
   ```
   images-agent albums "C:\Users\user\Pictures\_Albums" --dry-run
   ```
   Inspect the manifest: person clusters + scene per file.

7. **Build for real:**
   ```
   images-agent albums "C:\Users\user\Pictures\_Albums"
   ```
   Default op is `copy` (safe). `--op symlink` to save disk; `--op move` only if
   the user explicitly wants originals relocated.

## Verification checklist (do this — don't skip)

- [ ] `python -m compileall images_agent` is clean.
- [ ] `images-agent --help` lists `faces`, `cluster`, `albums`.
- [ ] On a small subfolder, open 2–3 `Person_XX` folders and confirm the faces
      really are the same person.
- [ ] Confirm a photo with two people appears under both person folders.
- [ ] Confirm `No_People` holds face-less photos and `Unsorted_Faces` holds
      cluster `-1` faces.
- [ ] Row counts: `len(manifest)` ≈ number of indexed images.

## Known limits / decisions to surface to the user

- Person folders are **numbered, not named** — rename after the first run.
- Clustering is imperfect at scale; expect a tuning loop on `--cluster-eps`.
- Scene labels in `albums.DEFAULT_SCENE_LABELS` are generic; tailor them (or pass
  `--scene-labels file.txt`) to this library for better sub-albums.
- The user originally said "move"; we default to **copy** because one photo can
  belong to several people. Confirm before using `--op move`.
- HEIC/iPhone files need `pillow-heif` (in the `[faces]` extra).

## Tuning reference

| Symptom | Knob |
|---------|------|
| Duplicate people folders | `--cluster-eps` up (e.g. 0.50) |
| Two people merged | `--cluster-eps` down (e.g. 0.38) |
| Person needs fewer shots to count | `--cluster-min-samples` down |
| Weak/false faces | `--min-det-score` up (e.g. 0.6) |
| Wrong scene buckets | edit `DEFAULT_SCENE_LABELS` / `--scene-labels` |
| Disk pressure from copies | `--op symlink` |

> **Note (the `--cluster-eps` row above is misleading):** raising eps does *not*
> merge duplicate people — past ~0.5 it chains *distinct* people into one giant
> cluster. Tune **down** for over-split. `0.45` worked on ~7k photos (485 people).

## Status update — albums evolved beyond this brief

Since this handoff: added `albums --people-only` (real `Person_XX` folders only),
coarsened age buckets, and — most importantly — found `buffalo_l`'s `--by-age`
estimates **unusable** (catastrophic on children/dim faces). Per-person
age organization now uses **MiVOLO** (image-only age on face crops) to build
**life-stage** folders. Full details: [AGE_ESTIMATION.md](AGE_ESTIMATION.md);
project status: [PROJECT_PLAN.md](PROJECT_PLAN.md).
