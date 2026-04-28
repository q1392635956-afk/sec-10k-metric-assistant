"""
retriever.py
------------
Embed 10-K chunks using OpenAI text-embedding-3-small, cache them locally,
and retrieve the top-k most relevant chunks for a query using cosine similarity.

The embedding index is built once and saved to data/embeddings_cache.pkl.
Subsequent calls load from cache, which avoids repeated API calls.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
from openai import OpenAI

from ingest import load_chunks

CACHE_PATH = "data/embeddings_cache.pkl"
EMBED_MODEL = "text-embedding-3-small"


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


def _embed(text: str) -> list[float]:
    """Return embedding vector for a single string."""
    client = _get_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def build_or_load_index(
    data_path: str = "data/apple_2025_10k.txt",
) -> tuple[list[str], np.ndarray]:
    """Return (chunks, embeddings_matrix).

    Loads from cache if available; otherwise embeds all chunks and saves cache.
    """
    if os.path.exists(CACHE_PATH):
        print("[retriever] Loading embeddings from cache...")
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        return cache["chunks"], cache["embeddings"]

    print("[retriever] Building embeddings index (this may take a minute)...")
    chunks = load_chunks(data_path)

    embeddings: list[list[float]] = []
    for i, chunk in enumerate(chunks):
        embeddings.append(_embed(chunk))
        if (i + 1) % 50 == 0:
            print(f"[retriever]   {i + 1}/{len(chunks)} chunks embedded...")

    embeddings_array = np.array(embeddings, dtype=np.float32)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings_array}, f)
    print(f"[retriever] Saved cache to '{CACHE_PATH}'")

    return chunks, embeddings_array


def retrieve(
    query: str,
    chunks: list[str],
    embeddings: np.ndarray,
    top_k: int = 6,
) -> list[str]:
    """Return the top_k chunks most similar to query."""
    query_vec = np.array(_embed(query), dtype=np.float32)

    # Cosine similarity: (query · chunk) / (|query| * |chunk|)
    norms = np.linalg.norm(embeddings, axis=1) + 1e-10
    query_norm = np.linalg.norm(query_vec) + 1e-10
    scores = (embeddings @ query_vec) / (norms * query_norm)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]
