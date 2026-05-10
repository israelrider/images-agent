from __future__ import annotations

import json
import re
from pathlib import Path

from ollama import Client, ResponseError

from images_agent import config


def make_client() -> Client:
    api_key = config.ollama_api_key()
    return Client(
        host=config.ollama_host(),
        headers={"Authorization": f"Bearer {api_key}"},
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)


def extract_json_object(text: str) -> dict | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = _FENCE_RE.sub("", raw, count=1)
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].rstrip()

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


VISION_SYSTEM = (
    "You analyze photographs. Respond with machine-readable facts only when asked for JSON."
)


def chat_vision_text(
    client: Client,
    model: str,
    image_path: Path,
    user_prompt: str,
) -> str:
    messages: list[dict] = [
        {"role": "system", "content": VISION_SYSTEM},
        {
            "role": "user",
            "content": user_prompt,
            "images": [str(image_path)],
        },
    ]
    try:
        response = client.chat(model=model, messages=messages, stream=False)
    except ResponseError as e:
        err_low = str(e.error).lower()
        if e.status_code == 404 or "not found" in err_low:
            raise RuntimeError(
                f"Ollama rejected vision model {model!r} ({e}). "
                "Use an exact model id with tag from https://ollama.com/api/tags "
                "(for example gemma4:31b, gemma3:12b, or qwen3-vl:235b-instruct)."
            ) from e
        raise
    msg = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    if msg is None:
        return ""
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    return content or ""
