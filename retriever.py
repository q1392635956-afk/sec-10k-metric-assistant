"""
retriever.py
------------
Local TF-IDF retrieval over 10-K text chunks using scikit-learn.

No external API calls — runs entirely on-device.
The TF-IDF index is built once from the chunked text, cached to disk,
and reloaded on subsequent runs.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ingest import load_chunks

CACHE_PATH = "data/tfidf_cache.pkl"


def build_or_load_index(
    data_path: str = "data/apple_2025_10k.txt",
) -> tuple[list[str], TfidfVectorizer, np.ndarray]:
    """Return (chunks, vectorizer, tfidf_matrix).

    Loads from cache if available; otherwise builds the index and saves it.
    TF-IDF on ~55 KB of text is fast (< 1 second), but caching avoids
    repeated disk reads on every Streamlit rerun.
    """
    if os.path.exists(CACHE_PATH):
        print("[retriever] Loading TF-IDF index from cache...")
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        return cache["chunks"], cache["vectorizer"], cache["matrix"]

    print("[retriever] Building TF-IDF index...")
    chunks = load_chunks(data_path)

    # Unigrams + bigrams; sublinear TF dampens the effect of very frequent terms
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(chunks)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}, f)
    print(f"[retriever] Saved TF-IDF cache to '{CACHE_PATH}'")

    return chunks, vectorizer, matrix


def retrieve(
    query: str,
    chunks: list[str],
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 6,
) -> list[str]:
    """Return the top_k chunks most similar to the query by TF-IDF cosine similarity."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]
