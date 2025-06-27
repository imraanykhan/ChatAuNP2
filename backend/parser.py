# backend/parser.py
"""Light‑weight text‑to‑JSON converter for ChatAu answers.

The main entry‑point is ``convert_to_json`` which accepts a *raw* answer
string (Markdown or plain‑text) and returns a **Python dict** conforming to
the schema documented in ``SCHEMA_VERSION``.  The function is deliberately
stateless so it can be imported by FastAPI, unit‑tests, notebooks, etc.  A
``ParserError`` is raised if required top‑level sections are missing or
parsing fails catastrophically.

Design goals
============
* **Robust to minor typos** – flexible regexes and default fall‑backs.
* **Zero heavy deps** – only stdlib; upgrade to ChemNLP libs later if needed.
* **Explicit schema versioning** – eases future migrations.
* **Typed** – leverages PEP 484 for better IDE support.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

__all__ = ["convert_to_json", "ParserError"]

SCHEMA_VERSION = "1.0"


class ParserError(ValueError):
    """Raised when the input cannot be parsed into a valid protocol."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(header: str) -> str:
    """Make a snake‑case key from an arbitrary header label."""
    return re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")


def _split_top_sections(text: str) -> Dict[str, str]:
    """Return mapping {section_name: block_text}.  Section headings are
    assumed to be numbered Markdown headings like ``1. **Reagents**:``.
    """
    pattern = re.compile(r"^\s*\d+\.\s+\*\*(.+?)\*\*:", re.M)
    parts = pattern.split(text)
    # parts[0] → possible intro text we discard
    headers = parts[1::2]
    bodies = parts[2::2]
    return {h.strip(): b.strip() for h, b in zip(headers, bodies)}


_bullet_rx = re.compile(r"^\s*[\-–]\s+(.*)")


def _extract_bullets(block: str) -> List[str]:
    """Return list of bullet‑point lines with the leading dash stripped."""
    return [_bullet_rx.match(line).group(1).strip() for line in block.splitlines() if _bullet_rx.match(line)]


_amount_rx = re.compile(
    r"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg|µl|μl|ml|l|mL|L)\b",
    re.I,
)

_conc_rx = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mM|M|%\s*w\/v|%\s*v\/v)", re.I)

_volume_rx = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*mL", re.I)


@dataclass
class Reagent:
    description: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    concentration: Optional[str] = None
    final_volume_mL: Optional[float] = None

    def asdict(self) -> Dict[str, object]:
        return asdict(self)


def _parse_reagent(bullet: str) -> Reagent:
    """Very naive chemical line parser – good for controlled bullets."""
    amt_match = _amount_rx.search(bullet)
    conc_match = _conc_rx.search(bullet)
    vol_match = _volume_rx.search(bullet)

    amount = float(amt_match.group("qty")) if amt_match else None
    unit = amt_match.group("unit") if amt_match else None
    conc = conc_match.group(0) if conc_match else None
    vol = float(vol_match.group("val")) if vol_match else None

    return Reagent(
        description=bullet,
        amount=amount,
        unit=unit,
        concentration=conc,
        final_volume_mL=vol,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_to_json(raw: str) -> Dict[str, object]:
    """Convert free‑text *raw* protocol into a dict ready for ``json.dumps``.

    Parameters
    ----------
    raw : str
        The full answer text (Markdown or plain). Newlines are normalised.

    Returns
    -------
    dict
        JSON‑serialisable dict that follows the *ChatAu Protocol* schema.
    """
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")

    raw = textwrap.dedent(raw).strip()
    sections = _split_top_sections(raw)
    if not sections:
        raise ParserError("No numbered **Header** sections found – is the input formatted correctly?")

    # --- Build output -----------------------------------------------------
    out: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "title": "",
        "reagents": [],
        "procedure": [],
        "characterization": [],
        "storage": "",
    }

    # Title – use first line / fallback
    first_line = raw.split("\n", 1)[0]
    out["title"] = first_line.lstrip("# ")[:120]

    for header, block in sections.items():
        key = _slugify(header)
        bullets = _extract_bullets(block)

        if "reagent" in key:
            out["reagents"] = [_parse_reagent(b) .asdict() for b in bullets]
        elif "synthesis" in key or "procedure" in key:
            out["procedure"] = bullets
        elif "characterization" in key:
            out["characterization"] = bullets
        elif "storage" in key:
            out["storage"] = bullets[0] if bullets else block.strip()
        else:
            # Unknown section → store raw under its slug
            out[key] = bullets or block

    # Basic validation -----------------------------------------------------
    mandatory = ("reagents", "procedure")
    missing = [k for k in mandatory if not out.get(k)]
    if missing:
        raise ParserError(f"Missing required section(s): {', '.join(missing)}")

    return out


# ---------------------------------------------------------------------------
# CLI helper (``python backend/parser.py file.txt``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pathlib

    if len(sys.argv) != 2:
        print("Usage: python backend/parser.py <path_to_txt_file>")
        sys.exit(1)

    path = pathlib.Path(sys.argv[1])
    print(f"Parsing {path} …")
    data = convert_to_json(path.read_text(encoding="utf‑8"))
    print(json.dumps(data, indent=2, ensure_ascii=False))
