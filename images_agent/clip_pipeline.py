from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from images_agent import config

_ie_model: str | None = None
_ie_instance = None

_te_model: str | None = None
_te_instance = None


def _reset_embedders_if_config_changed() -> None:
    global _ie_instance, _ie_model, _te_instance, _te_model
    vis = config.clip_vision_embed_model()
    tex = config.clip_text_embed_model()
    if _ie_model != vis:
        _ie_instance = None
        _ie_model = vis
    if _te_model != tex:
        _te_instance = None
        _te_model = tex


def _image_embedding_model():
    global _ie_instance
    _reset_embedders_if_config_changed()
    assert _ie_model is not None
    if _ie_instance is None:
        from fastembed import ImageEmbedding

        _ie_instance = ImageEmbedding(model_name=_ie_model)
    return _ie_instance


def _text_embedding_model():
    global _te_instance
    _reset_embedders_if_config_changed()
    assert _te_model is not None
    if _te_instance is None:
        from fastembed import TextEmbedding

        _te_instance = TextEmbedding(model_name=_te_model)
    return _te_instance


def l2_normalize(vector: NDArray[np.floating]) -> NDArray[np.float32]:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return (v / n).astype(np.float32)


def embed_text(query: str) -> NDArray[np.float32]:
    te = _text_embedding_model()
    raw = next(te.embed([query.strip() or " "]))
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    return l2_normalize(arr)


def text_embedding_dimension() -> int:
    return int(embed_text("probe").shape[0])


def embed_images_batched(paths: list[Path], batch_size: int = 16) -> Iterator[tuple[Path, NDArray[np.float32] | None]]:
    """Yield (path, normalized vector or None if embedding failed)."""
    ie = _image_embedding_model()
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        strs = [str(p) for p in batch]
        try:
            embs = list(ie.embed(strs))
            if len(embs) != len(batch):
                raise ValueError("embedding count mismatch")
            for pth, emb in zip(batch, embs, strict=True):
                try:
                    yield pth, l2_normalize(np.asarray(emb, dtype=np.float32))
                except Exception:
                    yield pth, None
        except Exception:
            for pth, single in zip(batch, strs, strict=True):
                try:
                    emb = next(ie.embed([single]))
                    yield pth, l2_normalize(np.asarray(emb, dtype=np.float32))
                except Exception:
                    yield pth, None
