"""FAISS index.

Functions
---------
add_to_store(text: str)  -> None
    Saves raw document text (or a chunk) in an in‑memory list.

search(query: str, k: int = 4) -> str
    Returns the most‑recent *k* stored documents concatenated.
    Ignores the query – this is only to keep the Flask app alive
    while you develop the actual similarity search.

stats() -> dict
    Small helper for debugging; returns how many docs / chars are
    currently cached.
"""
from __future__ import annotations

from typing import List

_store: List[str] = []

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def add_to_store(doc: str) -> None:
    """Append raw text (or a pre‑chunked string) to the store."""
    if doc:
        _store.append(doc)
        print(f"[vector_store] stored #{len(_store)} (chars={len(doc)})")


def search(query: str, k: int = 4) -> str:
    """Return concatenation of the latest *k* docs.

    Parameters
    ----------
    query : str
        Ignored in this stub. The signature matches what `app.py`
        expects, so you can drop in a real vector‑similarity search
        later without touching Flask.
    """
    if not _store:
        return ""
    result = "\n\n".join(_store[-k:])
    print(f"[vector_store] returning {len(result)} chars for query: {query[:40]}…")
    return result


def stats() -> dict:
    """Return current corpus stats for debugging."""
    return {
        "n_docs": len(_store),
        "chars": sum(map(len, _store)),
    }

# ---------------------------------------------------------------------------
# CLI test ------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    add_to_store("First doc ……")
    add_to_store("Second doc ……")
    print(search("dummy"))
    print(stats())
