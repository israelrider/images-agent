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
    return _require_env("OLLAMA_API_KEY")


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "https://ollama.com").strip() or "https://ollama.com"


def ollama_vision_model() -> str:
    return os.environ.get("OLLAMA_VISION_MODEL", "gemma3").strip() or "gemma3"


def image_library_root() -> Path:
    raw = os.environ.get("IMAGE_LIBRARY_ROOT", r"C:\Users\user\Pictures").strip()
    if not raw:
        raw = r"C:\Users\user\Pictures"
    return Path(raw).expanduser().resolve()
