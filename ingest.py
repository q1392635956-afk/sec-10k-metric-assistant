"""
ingest.py
---------
Load the Apple FY2025 10-K text file and split it into overlapping chunks
for TF-IDF retrieval.

Uses character-based chunking with newline-aware boundaries so financial
table rows are not split in the middle.
"""
from __future__ import annotations

import os

DATA_PATH = "data/apple_2025_10k.txt"

# ~1500 characters ≈ 375 tokens — keeps each financial statement table
# in its own chunk so TF-IDF retrieval scores stay focused
CHUNK_SIZE_CHARS = 1500
CHUNK_OVERLAP_CHARS = 150


def load_text(path: str = DATA_PATH) -> str:
    """Read raw 10-K text from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"10-K text file not found at '{path}'.\n"
            "Please follow the instructions in data/README.md to add the file."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping character-based chunks.

    Tries to break at a newline near the chunk boundary so financial
    table rows stay intact within a single chunk.
    """
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Break cleanly at a newline in the second half of the chunk
        if end < len(text):
            newline = text.rfind("\n", start + chunk_size // 2, end)
            if newline != -1:
                end = newline + 1  # include the newline itself

        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap

    return chunks


def load_chunks(path: str = DATA_PATH) -> list[str]:
    """Convenience wrapper: load text and return chunked list."""
    text = load_text(path)
    chunks = chunk_text(text)
    print(f"[ingest] Loaded {len(chunks)} chunks from '{path}'")
    return chunks
