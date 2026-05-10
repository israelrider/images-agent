from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from images_agent import clip_pipeline, config, index_store
from images_agent.ollama_cloud import chat_vision_text, extract_json_object, make_client
from images_agent.scan import iter_image_paths


def _user_prompt(criteria: str) -> str:
    crit = criteria.strip()
    return (
        f'The user wants photos that match this criterion:\n\n"{crit}"\n\n'
        "Look at this image. Decide whether it satisfies the criterion using only what is visible.\n\n"
        "Return ONLY one JSON object with exactly this key:\n"
        '- "matches": boolean\n\n'
        "Do not wrap in markdown. No other text."
    )


def _matches_from_response(text: str, client, model: str, image_path: Path, criteria: str) -> bool | None:
    data = extract_json_object(text)
    if data is None:
        text2 = chat_vision_text(
            client,
            model,
            image_path,
            _user_prompt(criteria) + " Reply with raw JSON only; no markdown fences or prose.",
        )
        data = extract_json_object(text2)
    if data is None:
        return None
    m = data.get("matches")
    return m if isinstance(m, bool) else None


def match_image(client, model: str, image_path: Path, criteria: str) -> bool | None:
    text = chat_vision_text(client, model, image_path, _user_prompt(criteria))
    return _matches_from_response(text, client, model, image_path, criteria)


def search_disclaimer() -> str:
    return (
        "Note: Matches are based on the vision model's interpretation of your criterion; "
        "verify important picks manually.\n"
    )


def vector_search_disclaimer() -> str:
    return (
        "Note: Ranking uses local CLIP vector similarity vs your criteria; optionally refined with hybrid "
        "Ollama vision rerank.\nVerify important picks manually.\n"
    )


def write_matches_json(payload: dict, output: Path) -> None:
    out_path = output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_criteria_search(
    root: Path,
    criteria: str,
    *,
    model: str,
    limit: int | None,
    output: Path,
) -> None:
    sys.stderr.write(search_disclaimer())

    crit = criteria.strip()
    if not crit:
        sys.stderr.write("Criteria text is empty.\n")
        raise SystemExit(2)

    paths = sorted(iter_image_paths(root), key=lambda p: str(p).lower())
    if limit is not None:
        paths = paths[:limit]

    if not paths:
        sys.stderr.write("No images found under the given root.\n")
        raise SystemExit(1)

    client = make_client()
    matched: list[str] = []
    uncertain = 0
    total = len(paths)
    for i, path in enumerate(paths, start=1):
        print(f"{i} of {total}")
        decision = match_image(client, model, path, crit)
        if decision is True:
            matched.append(str(path))
        elif decision is None:
            uncertain += 1

    if uncertain:
        sys.stderr.write(
            f"Warning: could not parse a yes/no decision for {uncertain} image(s); "
            "those were skipped.\n"
        )

    payload = {
        "criteria": crit,
        "root": str(root.resolve()),
        "count": len(matched),
        "paths": matched,
    }
    write_matches_json(payload, output)
    sys.stderr.write(f"Wrote {len(matched)} path(s) to {output.expanduser().resolve()}\n")


def run_incremental_embedding_index(
    root: Path,
    *,
    sqlite_path: Path,
    rebuild: bool,
    do_vacuum: bool,
    batch_size: int,
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = index_store._connect(sqlite_path)
    try:
        index_store.init_schema(conn)
        pair = config.clip_model_pair_key()
        dim = clip_pipeline.text_embedding_dimension()

        paths = sorted(iter_image_paths(root), key=lambda p: str(p).lower())
        alive = {str(p) for p in paths}
        removed = index_store.prune_missing_paths(conn, alive)
        if removed:
            sys.stderr.write(f"Removed {removed} stale index row(s).\n")

        if rebuild:
            index_store.delete_all(conn)
            sys.stderr.write("Rebuilt index (cleared embeddings).\n")

        pending: list[Path] = []
        skipped = 0
        for pth in paths:
            try:
                st = pth.stat()
            except OSError as e:
                sys.stderr.write(f"Skip stat failure {pth}: {e}\n")
                continue
            key = str(pth.resolve())
            if index_store.row_matches_cache(conn, path=key, mtime_ns=st.st_mtime_ns, size=st.st_size, model_pair=pair):
                skipped += 1
                continue
            pending.append(pth)

        updated = 0
        failed = 0
        batches = max(1, batch_size)
        total_pending = len(pending)
        for i in range(0, len(pending), batches):
            chunk = pending[i : i + batches]
            for pth, vec in clip_pipeline.embed_images_batched(chunk, batch_size=len(chunk)):
                if vec is None or vec.shape[0] != dim:
                    sys.stderr.write(f"Embedding failed: {pth}\n")
                    failed += 1
                    continue
                try:
                    st = pth.stat()
                except OSError as e:
                    sys.stderr.write(f"Skip after embed (stat): {pth}: {e}\n")
                    failed += 1
                    continue
                index_store.upsert_embedding(
                    conn,
                    path=str(pth.resolve()),
                    mtime_ns=st.st_mtime_ns,
                    size=st.st_size,
                    dim=dim,
                    model_pair=pair,
                    vector=vec.astype(np.float32).tobytes(),
                )
                updated += 1
                print(f"{updated} indexed / {total_pending} pending (+{skipped} skipped)\r", end="", flush=True)
        conn.commit()
        print()
        sys.stderr.write(
            f"Index done: embedded {updated} file(s); skipped unchanged {skipped}; "
            f"embedding failures {failed}; total tracked {index_store.count_rows(conn)}.\n"
        )
        if do_vacuum:
            index_store.vacuum(conn)
            sys.stderr.write("VACUUM complete.\n")
    finally:
        conn.close()


def run_vector_search(
    root: Path,
    criteria: str,
    *,
    sqlite_path: Path,
    top_k: int | None,
    min_similarity: float | None,
    rerank: int,
    vision_model: str,
    output: Path,
) -> None:
    sys.stderr.write(vector_search_disclaimer())

    crit = criteria.strip()
    if not crit:
        sys.stderr.write("Criteria text is empty.\n")
        raise SystemExit(2)

    root_resolved = root.resolve()
    conn = index_store._connect(sqlite_path)
    try:
        index_store.init_schema(conn)
        pair = config.clip_model_pair_key()
        dim = clip_pipeline.text_embedding_dimension()
        paths, mat = index_store.load_normalized_matrix_under_root(
            conn, root_resolved, expected_dim=dim, expected_model_pair=pair
        )
    finally:
        conn.close()

    if not paths or mat.size == 0:
        sys.stderr.write(
            f"No indexed vectors for library under {root_resolved}. "
            "Run: images-agent index --root \"" + str(root_resolved) + "\"\n"
        )
        raise SystemExit(1)

    q = clip_pipeline.embed_text(crit)
    scores_f = mat @ q
    order = np.argsort(-scores_f)

    ranked: list[tuple[str, float]] = []
    for idx in order:
        sc = float(scores_f[int(idx)])
        if min_similarity is not None and sc < float(min_similarity):
            continue
        ranked.append((paths[int(idx)], sc))

    if rerank <= 0:
        if top_k is not None:
            ranked = ranked[: int(top_k)]
        out_paths = [p for p, _ in ranked]
        out_scores = [s for _, s in ranked]
        payload = {
            "criteria": crit,
            "root": str(root_resolved),
            "count": len(out_paths),
            "paths": out_paths,
            "scores": out_scores,
            "rerank_used": False,
        }
        write_matches_json(payload, output)
        sys.stderr.write(f"Wrote {len(out_paths)} path(s) to {output.expanduser().resolve()}\n")
        return

    if not vision_model:
        sys.stderr.write("Hybrid rerank needs a vision model. Set OPENAI_VISION_MODEL or pass --model.\n")
        raise SystemExit(2)

    basis = ranked if top_k is None else ranked[: int(top_k)]
    pool_cap = max(int(rerank), 1)
    pool = basis[:pool_cap]
    client = make_client()

    uncertain = 0
    confirmed: list[tuple[str, float]] = []
    total_pool = len(pool)
    sys.stderr.write(search_disclaimer())
    for i, (p_str, vec_score) in enumerate(pool, start=1):
        print(f"rerank {i} of {total_pool}")
        p = Path(p_str)
        decide = match_image(client, vision_model, p, crit)
        if decide is True:
            confirmed.append((p_str, vec_score))
        elif decide is None:
            uncertain += 1

    if uncertain:
        sys.stderr.write(
            f"Warning: could not parse a yes/no decision for {uncertain} rerank candidate(s).\n"
        )

    payload = {
        "criteria": crit,
        "root": str(root_resolved),
        "count": len(confirmed),
        "paths": [p for p, _ in confirmed],
        "scores": [s for _, s in confirmed],
        "rerank_used": True,
        "rerank_pool": pool_cap,
    }
    write_matches_json(payload, output)
    sys.stderr.write(f"Wrote {len(confirmed)} vision-confirmed path(s) to {output.expanduser().resolve()}\n")
