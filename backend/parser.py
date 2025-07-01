"""Light‑weight text‑to‑JSON converter for ChatAu answers.
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

__all__ = [
    "convert_to_json",
    "ParserError",
]

SCHEMA_VERSION = "1.0"


class ParserError(ValueError):
    """Raised when the protocol text cannot be parsed."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(header: str) -> str:
    """Turn a heading into a safe snake‑case key."""
    return re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")


def _split_sections(text: str) -> Dict[str, str]:
    """Return {section_title: body_text}. Expects headings like
    `1. **Reagents and Solutions**:` in Markdown.
    """
    pattern = re.compile(r"^\s*\d+\.\s+\*\*(.+?)\*\*:", re.M)
    parts = pattern.split(text)
    headers = parts[1::2]
    bodies = parts[2::2]
    return {h.strip(): b.strip() for h, b in zip(headers, bodies)}


_bullet_rx = re.compile(r"^\s*[\-–]\s+(.*)")
_amount_rx = re.compile(r"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg|µl|μl|ml|mL|l|L)\b", re.I)
_conc_rx   = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mM|M|%\s*w\/v|%\s*v\/v)", re.I)
_volume_rx = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*mL", re.I)


def _extract_bullets(block: str) -> List[str]:
    return [_bullet_rx.match(l).group(1).strip() for l in block.splitlines() if _bullet_rx.match(l)]


@dataclass
class Reagent:
    description: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    concentration: Optional[str] = None
    final_volume_mL: Optional[float] = None

    def asdict(self) -> Dict[str, object]:
        return asdict(self)


def _parse_reagent(line: str) -> Reagent:
    amt = _amount_rx.search(line)
    conc = _conc_rx.search(line)
    vol = _volume_rx.search(line)
    return Reagent(
        description=line,
        amount=float(amt.group("qty")) if amt else None,
        unit=amt.group("unit") if amt else None,
        concentration=conc.group(0) if conc else None,
        final_volume_mL=float(vol.group("val")) if vol else None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_to_json(raw: str) -> Dict[str, object]:
    """Convert *raw* protocol text to a JSON‑serialisable dict."""
    if not raw or not raw.strip():
        raise ParserError("Input text is empty.")

    raw = textwrap.dedent(raw).strip()
    sections = _split_sections(raw)
    if not sections:
        raise ParserError("No Markdown headings like `1. **Reagents**:` found.")

    out: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "title": raw.split("\n", 1)[0].lstrip("# ")[:120],
        "reagents": [],
        "procedure": [],
        "characterization": [],
        "storage": "",
    }

    for header, body in sections.items():
        key = _slugify(header)
        bullets = _extract_bullets(body)

        if "reagent" in key:
            out["reagents"] = [_parse_reagent(b).asdict() for b in bullets]
        elif any(k in key for k in ("synthesis", "procedure")):
            out["procedure"] = bullets
        elif "characterization" in key:
            out["characterization"] = bullets
        elif "storage" in key:
            out["storage"] = bullets[0] if bullets else body.strip()
        else:
            out[key] = bullets or body.strip()

    if not out["reagents"] or not out["procedure"]:
        raise ParserError("Missing required 'Reagents' or 'Procedure' sections.")

    return out

# ---------------------------------------------------------------------------
# CLI helper ---------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) == 2 else None
    if not path:
        print("Usage: python backend/parser.py <protocol.txt>")
        sys.exit(1)
    data = convert_to_json(open(path, encoding="utf-8").read())
    print(json.dumps(data, indent=2, ensure_ascii=False))
