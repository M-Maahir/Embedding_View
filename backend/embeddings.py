"""Encode words or sentences, project to 3D, cluster, and find nearest neighbors."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_WORDS = 80
MAX_ITEMS = MAX_WORDS
SCENE_SCALE = 4.0
NEIGHBOR_K = 3
LABEL_LIMIT = 42
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z])")


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def parse_items(raw_items: list[str], kind: str = "word") -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = " ".join(item.split()).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= MAX_ITEMS:
            break
    if not cleaned:
        raise ValueError(f"Enter at least one {kind}.")
    return cleaned


def parse_words(raw_words: list[str]) -> list[str]:
    return parse_items(raw_words, kind="word")


def short_label(text: str, limit: int = LABEL_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def ensure_punkt() -> None:
    import nltk

    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            return
        except LookupError:
            continue
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)


def tokenize_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Enter at least one sentence.")
    try:
        ensure_punkt()
        from nltk.tokenize import sent_tokenize

        raw = sent_tokenize(stripped)
    except Exception:
        raw = _SENTENCE_SPLIT.split(stripped)
        if len(raw) <= 1:
            raw = [line.strip() for line in stripped.splitlines() if line.strip()]
    return parse_items(raw, kind="sentence")


def embed(words: list[str]) -> np.ndarray:
    model = get_model()
    vectors = model.encode(words, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float64)


def to_3d(vectors: np.ndarray) -> np.ndarray:
    n_samples = vectors.shape[0]
    if n_samples == 1:
        return np.zeros((1, 3), dtype=np.float64)

    n_components = min(3, n_samples, vectors.shape[1])
    coords = PCA(n_components=n_components, random_state=42).fit_transform(vectors)
    if n_components < 3:
        padded = np.zeros((n_samples, 3), dtype=np.float64)
        padded[:, :n_components] = coords
        coords = padded

    coords = coords - coords.mean(axis=0)
    max_abs = float(np.abs(coords).max()) or 1.0
    return coords / max_abs * SCENE_SCALE


def cluster_ids(vectors: np.ndarray) -> list[int]:
    n_samples = vectors.shape[0]
    if n_samples < 3:
        return [0] * n_samples
    k = min(6, max(2, n_samples // 3), n_samples)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
    return [int(label) for label in labels]


def nearest(vectors: np.ndarray, words: list[str], k: int = NEIGHBOR_K) -> dict[str, list[dict]]:
    n_samples = len(words)
    if n_samples <= 1:
        return {words[0]: []} if words else {}

    similarity = vectors @ vectors.T
    neighbor_k = min(k, n_samples - 1)
    neighbors: dict[str, list[dict]] = {}
    for i, word in enumerate(words):
        ranking = np.argsort(-similarity[i])
        picked: list[dict] = []
        for j in ranking:
            if int(j) == i:
                continue
            picked.append({"word": words[int(j)], "score": float(similarity[i, j])})
            if len(picked) >= neighbor_k:
                break
        neighbors[word] = picked
    return neighbors


def _payload_for(items: list[str], mode: str) -> dict:
    vectors = embed(items)
    coords = to_3d(vectors)
    clusters = cluster_ids(vectors)
    points = [
        {
            "word": item,
            "label": short_label(item),
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "z": float(coords[i, 2]),
            "cluster": clusters[i],
        }
        for i, item in enumerate(items)
    ]
    return {
        "mode": mode,
        "count": len(items),
        "points": points,
        "neighbors": nearest(vectors, items),
    }


def build_payload(raw_words: list[str]) -> dict:
    return _payload_for(parse_words(raw_words), mode="words")


def build_sentence_payload(text: str) -> dict:
    return _payload_for(tokenize_sentences(text), mode="sentences")
