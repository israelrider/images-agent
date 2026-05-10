from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from images_agent.ollama_cloud import chat_vision_text, extract_json_object, make_client
from images_agent.scan import iter_image_paths

CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})


@dataclass
class AgeEstimateRow:
    path: str
    estimated_age_years: float | None
    confidence: Literal["low", "medium", "high"] | None
    has_clear_child: bool | None
    error: str | None = None


USER_PROMPT_SORT = """Look at this image. If one clear primary child is visible, estimate their approximate age in years as a decimal (e.g. 4.5). If no child is clear, use null.

Return ONLY one JSON object with exactly these keys:
- "estimated_age_years": number or null
- "confidence": one of "low", "medium", "high"
- "has_clear_child": boolean

Do not wrap in markdown. No other text."""


def _normalize_row(path: Path, data: dict | None) -> AgeEstimateRow:
    if data is None:
        return AgeEstimateRow(str(path), None, None, None, error="parse_error")

    age = data.get("estimated_age_years")
    if age is not None:
        try:
            age_f = float(age)
        except (TypeError, ValueError):
            age_f = None
    else:
        age_f = None

    conf = data.get("confidence")
    if isinstance(conf, str) and conf.lower() in CONFIDENCE_LEVELS:
        conf_norm: Literal["low", "medium", "high"] | None = conf.lower()  # type: ignore[assignment]
    else:
        conf_norm = None

    has_child = data.get("has_clear_child")
    if isinstance(has_child, bool):
        hc: bool | None = has_child
    elif has_child is None:
        hc = None
    else:
        hc = None

    return AgeEstimateRow(str(path), age_f, conf_norm, hc, error=None)


def estimate_age_for_image(client, model: str, path: Path) -> AgeEstimateRow:
    text = chat_vision_text(client, model, path, USER_PROMPT_SORT)
    data = extract_json_object(text)
    if data is None:
        text2 = chat_vision_text(
            client,
            model,
            path,
            USER_PROMPT_SORT + " Reply with raw JSON only; no markdown fences or prose.",
        )
        data = extract_json_object(text2)
    return _normalize_row(path, data)


def sort_age_disclaimer() -> str:
    return (
        "Note: Estimated ages are approximate guesses from the model, not facts. "
        "Use results for organization only.\n"
    )


def run_sort_age(
    root: Path,
    *,
    model: str,
    limit: int | None,
    as_json: bool,
) -> None:
    sys.stderr.write(sort_age_disclaimer())

    paths = sorted(iter_image_paths(root), key=lambda p: str(p).lower())
    if limit is not None:
        paths = paths[:limit]

    if not paths:
        sys.stderr.write("No images found under the given root.\n")
        raise SystemExit(1)

    client = make_client()
    rows: list[AgeEstimateRow] = []
    for path in paths:
        rows.append(estimate_age_for_image(client, model, path))

    def sort_key(r: AgeEstimateRow):
        age = r.estimated_age_years
        null_age = age is None
        return (null_age, age if age is not None else 0.0, r.path.lower())

    rows_sorted = sorted(rows, key=sort_key)

    if as_json:
        payload = [
            {
                "path": r.path,
                "estimated_age_years": r.estimated_age_years,
                "confidence": r.confidence,
                "has_clear_child": r.has_clear_child,
                "error": r.error,
            }
            for r in rows_sorted
        ]
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return

    for i, r in enumerate(rows_sorted, start=1):
        extra = f" err={r.error}" if r.error else ""
        sys.stdout.write(
            f"{i}. {r.path}\n"
            f"     age_years={r.estimated_age_years} confidence={r.confidence}"
            f" has_clear_child={r.has_clear_child}{extra}\n"
        )
