"""Voice-embedding helpers: (de)serialization, cosine similarity, matching.

Pure numpy — no GPU or model needed, so this is unit-testable anywhere.
Embeddings are stored as raw float32 bytes in SQLite.
"""
from typing import Optional

import numpy as np


def to_bytes(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def is_valid(vec) -> bool:
    """pyannote returns NaN rows for clusters with too little speech."""
    arr = np.asarray(vec, dtype=np.float32)
    return arr.size > 0 and bool(np.all(np.isfinite(arr)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def best_match(
    embedding: np.ndarray,
    people: list[dict],
    threshold: float,
) -> tuple[Optional[int], Optional[str], float]:
    """Match one cluster embedding against known people.

    `people`: [{"id": int, "name": str, "voiceprints": [np.ndarray, ...]}].
    A person can have several voiceprints; we score by their *best* one, so a
    single good sample is enough to identify.

    Returns (person_id, name, score) or (None, None, best_score) if below
    threshold.
    """
    best_id: Optional[int] = None
    best_name: Optional[str] = None
    best_score = -1.0
    for p in people:
        for vp in p["voiceprints"]:
            s = cosine(embedding, vp)
            if s > best_score:
                best_score, best_id, best_name = s, p["id"], p["name"]
    if best_id is not None and best_score >= threshold:
        return best_id, best_name, best_score
    return None, None, max(best_score, 0.0)
