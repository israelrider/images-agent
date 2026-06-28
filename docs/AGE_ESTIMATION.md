# Age estimation & per-person life-stage albums

Goal: organize one person's photos by their **age in each photo, inferred from
the image alone** (no EXIF dates, folder names, or other external data). This
records what we tried, what failed, and the approach that works.

## TL;DR

- **`buffalo_l` age (the `face.age` column): unusable.** Catastrophic on this
  library — a ~10-year-old child was tagged **47** and **60**. Fails on dim /
  dusk / non-frontal / occluded faces, which dominate casual photo collections.
- **CLIP zero-shot age: good but not safe.** Scoring face crops against
  `"a photo of a N year old"` prompts beat `buffalo_l` dramatically (children
  accurate; Spearman 0.79 vs. capture-year over 14 years). But it **under-ages**
  the top end and fails *confidently* on hard poses (sunglasses / downcast →
  collapsed ~19-year-olds to "child"). No cheap gate filters these: neither
  detector `det_score` nor CLIP's own softmax confidence distinguishes the wrong
  calls from right ones (proven empirically).
- **MiVOLO (chosen): robust.** A dedicated image-only age model
  (`iitolstykh/mivolo_v2`) run on the stored face crops. It correctly aged every
  hard-pose face that broke CLIP (the ~19-year-olds → 16–26 instead of "child").
  Residual bias: it **over-ages teens** by ~5 yr (a real 14 → ~20). Far gentler
  than CLIP's random catastrophic errors.

## How the three compared (same person A / person B faces, real age eyeballed)

| face (real age) | buffalo_l | CLIP zero-shot | MiVOLO |
|---|---|---|---|
| Person A, daylight (~5) | 7 | 5.6 | 6.8 |
| Person A, dusk (~10) | **47** | 9.5 | 9.7 |
| Person A, dark (~10) | **60** | 10.6 | 9.6 |
| Person A, teen (~14) | 24 | 14.1 | 20.3 |
| Person B, sunglasses (~19) | – | →child | 16.2 |
| Person B, close-up (~20) | – | →child | 25.9 |
| Person B, downcast (~19) | – | →child | 21.6–24.1 |

CLIP's failure mode is the dangerous one (confidently wrong, unfilterable);
MiVOLO's is a uniform over-aging at the teen range, absorbed by coarse buckets.

## MiVOLO setup (isolated, does not touch the shared venv)

MiVOLO pins `torch==2.5.1` / `timm==0.8.13.dev0`, which conflict with the shared
`.venv` (torch 2.12, used by sibling projects). So it lives in a **separate**
venv and runs as a subprocess:

- venv: `C:\Users\user\Documents\localhost\python\.venv-mivolo` (Python 3.13).
- deps: `timm==0.8.13.dev0`, `transformers`, `accelerate`, `safetensors`,
  `opencv-python-headless`, `scipy`, torch 2.12 CPU (the pinned 2.5.1 has no
  cp313 wheel — 2.12 works fine).
- the `mivolo` package is **not** `pip install`-able cleanly on Py3.13 (its
  `setup.py` needs `pkg_resources`, removed in setuptools ≥ 81). Instead clone it
  and put it on `PYTHONPATH`:
  `git clone https://github.com/WildChlamydia/MiVOLO.git` →
  `PYTHONPATH=...\MiVOLO`. (We only use `mivolo.model.create_timm_model`, not its
  detector — we already have face bboxes.)
- model: `AutoModelForImageClassification.from_pretrained("iitolstykh/mivolo_v2",
  trust_remote_code=True)` + `AutoImageProcessor`. Input = **384px face crops as
  numpy arrays**; body crop may be `[None]`. Read age from `output.age_output[i]`.
- Compute: CPU-only ViT at 384px; ~3,850 faces took tens of minutes.

## The per-person life-stage workflow (as run)

Operates on the flat `--people-only --no-scenes` album folders (one folder of
files per person, see [PROJECT_PLAN](PROJECT_PLAN.md)):

1. **Crop** each file's cluster face (bbox from the `face` table; the exact
   build de-collision naming — `name__1.jpg` — is replicated to map each output
   file back to its source).
2. **Age** each crop with MiVOLO (subprocess in `.venv-mivolo`).
3. **Move** into numbered life-stage folders (prefix keeps them in age order):
   `1-toddler` (<5) · `2-child` (5–9) · `3-preteen` (10–14) · `4-teen` (15–19) ·
   `5-young_adult` (20–24) · `6-adult` (25+) · `age_unknown`.

Because MiVOLO over-ages teens, for a child-to-teen subject the `4-teen` and
`5-young_adult` folders can be **merged** into one `4-teen_young_adult` to avoid
a boundary the model can't draw.

## Results (validated by eyeballing sampled face crops)

- **Person A** (cluster 1, 2,771 photos): `1-toddler 11 · 2-child 1343 · 3-preteen
  588 · 4-teen_young_adult 792 · 6-adult 23 · age_unknown 14`. Childhood-heavy,
  as expected for a child photographed into her teens.
- **Person B** (cluster 2, 1,080 photos): `3-preteen 3 · 4-teen 79 · 5-young_adult
  570 · 6-adult 428`. No `2-child` bucket — the misclassified young-adult faces
  CLIP dumped in "child" are correctly aged here.

## Caveats

- Single-photo age estimation has an irreducible error floor; expect reliable
  *life stages*, not exact years. Coarse/merged buckets are the honest output.
- MiVOLO over-ages teens ~5 yr; `buffalo_l` and CLIP have their own biases above.
- The MiVOLO pipeline scripts are exploratory (kept under the working scratchpad,
  not yet packaged into the CLI). Promote to a command if this becomes routine.
