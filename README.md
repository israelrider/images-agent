- **[pyproject.toml](pyproject.toml)** — `ollama`, `typer`, `python-dotenv`, `fastembed`, `numpy`; console script `images-agent` → `images_agent.cli:main`.
- **[.env.example](.env.example)** — `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_VISION_MODEL`, `IMAGE_LIBRARY_ROOT`, optional `IMAGE_INDEX_SQLITE` and CLIP FastEmbed model ids.
- **[images_agent/config.py](images_agent/config.py)** — `load_dotenv()` and env accessors (library root, Ollama, index path, CLIP model ids).
- **[images_agent/scan.py](images_agent/scan.py)** — recursive image scan (case-insensitive extensions).
- **[images_agent/index_store.py](images_agent/index_store.py)** — SQLite vector index (`embedding` rows: path, stats, normalized float32 blobs).
- **[images_agent/clip_pipeline.py](images_agent/clip_pipeline.py)** — FastEmbed ONNX CLIP text/vision embeddings (paired `Qdrant/clip-ViT-B-32-*` defaults).
- **[images_agent/ollama_cloud.py](images_agent/ollama_cloud.py)** — Ollama Cloud vision `chat` for legacy full scans and hybrid rerank.
- **[images_agent/search.py](images_agent/search.py)** — `run_vector_search`, `run_incremental_embedding_index`, legacy `run_criteria_search`; shared JSON writer.
- **[images_agent/gallery_html.py](images_agent/gallery_html.py)** — builds a local HTML page from `matches.json` (`file://` image URLs, optional scores / rerank metadata).
- **[images_agent/cli.py](images_agent/cli.py)** — `images-agent index` builds the SQLite index; `images-agent run` queries by CLIP (default) or `--legacy-full-vision`; `images-agent gallery` renders search results as HTML.

### Workflow (default CLIP vector search)

1. **Environment.** 

- Copy [.env.example](.env.example) to `.env`. 
- Set `OPENAI_BASE_URL` and 
- `OPENAI_API_KEY` if we use **hybrid rerank** or **legacy vision**; 
- plain CLIP search does not call Ollama.

2. **Install (editable)**

```bash
pip install -e .
```

3. **Index** (once per library, or rerun to refresh changed files):

```bash
images-agent index
```

First `index` may download ONNX models via FastEmbed (~0.34 GB vision + ~0.25 GB text for the defaults).

Optional: `--root`, `--index PATH`, `--rebuild`, `--vacuum`, `--batch-size N`.

4. **Query**

```bash
images-agent run "photos with a dog"
```

- CLIP mode loads vectors from SQLite, ranks by cosine (dot product of L2-normalized embeddings). 
- Output JSON includes **`paths`** and **`scores`**.


- By default, only the **top 100** vector hits are written 
- (so you get a bounded “best matches” list, not the whole library). 
- Override with `-n/--top-k`, set `IMAGE_AGENT_VECTOR_TOP_K` in `.env`, 
- or pass **`-n 0`** to export the **entire** library in ranked order (previous behavior).


- `--rerank K` confirms the top *K* vector hits (within the `--top-k` window when set) with the vision model;
- `--min-similarity FLOAT` drops low scores.

Use `-o/--output` to change the JSON path (default `matches.json`, gitignored).

5. **HTML gallery**

- open matches in a browser as a grid of thumbnails 
- (uses absolute `file://` URLs, so images load offline next to your library paths):

```bash
images-agent gallery
```

- Reads `matches.json` by default and writes `gallery.html`. 
- Customize paths with `-i` / `-o`. 
- Add `--open` to launch the default browser after writing.

```bash
images-agent gallery -i matches.json -o gallery.html --open
```

6. **Legacy full-library vision scan** 

(no index; every image goes to Ollama):

```bash
images-agent run "your criterion" --legacy-full-vision
```

Use `-n` to cap how many images are scanned.

**If `images-agent` isn’t on PATH:**

```bash
python -m images_agent.cli index
python -m images_agent.cli run "your criterion"
python -m images_agent.cli gallery --open
python -m images_agent.cli --help
```

```
..\.venv\Scripts\Activate.ps1

images-agent run "photos with a cat"
images-agent gallery --open
```