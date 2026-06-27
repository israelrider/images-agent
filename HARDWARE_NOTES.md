# Hardware feasibility — running images-agent on ~10,000 photos

Target machine (**st7**):

- CPU: Intel Core i7-9750H, 6 cores / 12 threads, 2.6 GHz (2019 mobile)
- RAM: 32 GB
- GPU: NVIDIA Quadro P620 (4 GB, Pascal) + Intel UHD 630 (integrated)
- Storage: ~339 GB free of 932 GB
- OS: 64-bit Windows

**Verdict: yes — comfortably feasible as a one-time run.** Expect roughly
**2–4 hours total on CPU**; the expensive `faces` pass is paid only once.

## Per-stage reality check

| Stage | Compute | On st7 |
|---|---|---|
| `index` (CLIP, FastEmbed ONNX, CPU) | one pass over 10k | ~15–40 min; 6-core CPU is fine |
| `faces` (InsightFace `buffalo_l`) | one pass over 10k — heaviest step | CPU ~1.5–3 hrs; ~2–4× faster on the P620 |
| `cluster` (DBSCAN, cosine) | over stored face vectors | seconds–minutes (10k photos ≈ ~10k–40k faces) |
| scene tagging (in `albums`) | reuses cached CLIP vectors | near-instant, no second image pass |
| `albums` copy | disk I/O | minutes; disk-bound |

## Hardware notes

- **RAM (32 GB):** not a concern. Face vectors are ~2 KB each; sklearn DBSCAN
  with `metric="cosine"` streams distances in chunks, so no N×N matrix blowup
  even at ~40k faces.
- **GPU (Quadro P620, 4 GB):** usable and modestly helpful for the face pass.
  `buffalo_l` fits in 4 GB easily; it's a weak Pascal card but still ~2–4×
  faster than the CPU for detection. Install `onnxruntime-gpu` (instead of
  `onnxruntime`) and run `images-agent faces --gpu`. Keep FastEmbed CLIP on CPU
  — not worth the setup friction.
- **Disk (~339 GB free):** fine, but `albums` **copies by default**. Phone
  photos (~3–5 MB) → ~30–50 GB; multi-person photos copy once per person, so
  budget ~60–100 GB worst case. Use `--op symlink` to avoid duplication (Windows
  symlinks need Developer Mode/admin; the code falls back to copy if denied).
- **HEIC/iPhone:** handled — `pillow-heif` is in the `[faces]` extra.
- **First-run model downloads:** ~0.6 GB (FastEmbed CLIP) + ~0.3 GB
  (`buffalo_l`). Trivial against free space.

## Recommended run order on this machine

1. `pip install -e ".[faces]"`, then swap to `onnxruntime-gpu`.
2. Smoke-test: `images-agent faces --root <small subfolder> --gpu` — confirm the
   GPU path works and the cluster count looks sane.
3. Full `images-agent faces --gpu`, tune `images-agent cluster --cluster-eps`,
   then `images-agent albums <dst> --dry-run` before the real build.

Only real things to watch: disk space (if copying) and the P620's age — neither
is a blocker.
