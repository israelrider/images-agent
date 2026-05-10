from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable {name}. See .env.example.")
    return value


def ollama_api_key() -> str:
    return _require_env("OPENAI_API_KEY")


def ollama_host() -> str:
    return os.environ.get("OPENAI_BASE_URL") or ""


def ollama_vision_model() -> str:
    return os.environ.get("OPENAI_VISION_MODEL") or ""


def image_library_root() -> Path:
    raw = os.environ.get("IMAGE_LIBRARY_ROOT", r"C:\Users\user\Pictures").strip()
    if not raw:
        raw = r"C:\Users\user\Pictures"
    return Path(raw).expanduser().resolve()


def image_index_sqlite() -> Path:
    raw = os.environ.get("IMAGE_INDEX_SQLITE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".cache" / "images-agent" / "embeddings.sqlite"


def clip_vision_embed_model() -> str:
    return os.environ.get("CLIP_VISION_EMBED_MODEL") or ""


def clip_text_embed_model() -> str:
    return os.environ.get("CLIP_TEXT_EMBED_MODEL") or ""


def clip_model_pair_key() -> str:
    return f"{clip_vision_embed_model()}|{clip_text_embed_model()}"


def default_vector_top_k() -> int:
    """Max paths returned by CLIP `run` when --limit/--top-k is omitted (see cli.run_cmd)."""
    raw = os.environ.get("IMAGE_AGENT_VECTOR_TOP_K", "").strip()
    if not raw:
        return 100
    try:
        n = int(raw)
    except ValueError as e:
        raise RuntimeError("IMAGE_AGENT_VECTOR_TOP_K must be a positive integer.") from e
    if n < 1:
        raise RuntimeError("IMAGE_AGENT_VECTOR_TOP_K must be at least 1.")
    return n
