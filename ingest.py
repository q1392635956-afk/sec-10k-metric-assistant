"""
ingest.py
---------
Load the Apple FY2025 10-K text file and split it into overlapping chunks
that can be embedded and retrieved.

Token-based chunking (via tiktoken) keeps chunks within the embedding
model's context window and preserves whole words at boundaries.
"""
from __future__ import annotations

import os
import tiktoken

DATA_PATH = "data/apple_2025_10k.txt"
CHUNK_SIZE = 800   # tokens per chunk — large enough to capture a full table row
CHUNK_OVERLAP = 100  # overlap so split sentences are still retrievable


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
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping token-based chunks.

    Returns a list of decoded string chunks.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        decoded = enc.decode(tokens[start:end])
        chunks.append(decoded)
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def load_chunks(path: str = DATA_PATH) -> list[str]:
    """Convenience wrapper: load text and return chunked list."""
    text = load_text(path)
    chunks = chunk_text(text)
    print(f"[ingest] Loaded {len(chunks)} chunks from '{path}'")
    return chunks
