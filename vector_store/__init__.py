"""Light‑weight FAISS + pickle wrapper.
The i‑th vector **is always** the i‑th text chunk, so we never babysit offsets.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss  # type: ignore

STORE_DIR = Path(os.getenv("STORE_DIR", "vector_store"))
INDEX_PATH = STORE_DIR / "papers.index"
TEXT_PATH = STORE_DIR / "chunks.pkl"

STORE_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# in‑memory singletons -------------------------------------------------------
# ---------------------------------------------------------------------------
if INDEX_PATH.exists():
    index = faiss.read_index(str(INDEX_PATH))
else:
    # we will re‑initialise lazily after we know the dimension
    index = None  # type: ignore

chunks: List[str]
if TEXT_PATH.exists():
    chunks = pickle.loads(TEXT_PATH.read_bytes())
else:
    chunks = []

# ---------------------------------------------------------------------------
# helpers --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _init_index(dim: int) -> None:
    global index
    index = faiss.IndexFlatIP(dim)


def add_embeddings(vecs, new_chunks: List[str]):
    """Add *aligned* (N,dim) vectors and matching text chunks."""
    global index, chunks

    if index is None:
        _init_index(vecs.shape[1])

    index.add(vecs)
    chunks.extend(new_chunks)

    # persist
    faiss.write_index(index, str(INDEX_PATH))
    TEXT_PATH.write_bytes(pickle.dumps(chunks))


def query(vec, k: int = 5) -> List[Tuple[str, float]]:
    """Return the top‑`k` (chunk, score) matches for a single query vector."""
    if index is None or index.ntotal == 0:
        return []

    scores, ids = index.search(vec, k)
    return [(chunks[i], float(scores[0, j])) for j, i in enumerate(ids[0]) if i != -1]

def add_to_store(doc: str):
    _store.append(doc)
    print(f"[vector_store] stored #{len(_store)} (chars={len(doc)})")

