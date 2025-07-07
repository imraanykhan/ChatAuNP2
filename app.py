# app.py – single Flask back‑end (fixed)
"""
✓ Upload a PDF  → /upload (multipart form)
✓ Ask a question → /ask   (form‑urlencoded)
✓ Convert answer → /parse (JSON)  via backend.parser.convert_to_json()

Changes vs. previous version
---------------------------
1. Finished the /ask route (prompt concat, returns JSON).
2. Implemented _extract_text() with PyPDF2 so /upload works.
3. Provided no‑op stubs for _embed/_chunk to avoid NameErrors.
4. Removed unused send_from_directory import.
5. Added try/except around vector‑store calls so the app
   still responds even if FAISS isn’t initialised yet.
"""
from __future__ import annotations

import io
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, abort
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import tiktoken

from backend.parser import convert_to_json, ParserError

import vector_store as vs  # your FAISS helper

# ── basic setup ────────────────────────────────────────────────────────────
load_dotenv()
app = Flask(__name__, template_folder="templates", static_folder="static")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ENC         = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL = "text-embedding-3-small"
MAX_TOKENS  = 256

# ── utilities ──────────────────────────────────────────────────────────────

def _chunk(text: str, max_toks: int = 300) -> List[str]:  # naive splitter
    words = text.split()
    chunks, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= max_toks:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _embed(texts: List[str]) -> np.ndarray:  # stub for local dev
    # Replace with client.embeddings.create to get real vectors.
    return np.random.rand(len(texts), 384).astype("float32")


def _extract_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Could not read PDF: %s" % exc) from exc

# ── routes ────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")
    if not file or file.filename == "":
        abort(400, "No PDF supplied.")
    if file.mimetype != "application/pdf":
        abort(400, "Only PDFs accepted.")

    try:
        text = _extract_text(file.read())
    except ValueError as err:
        abort(400, str(err))

    try:
        vs.add_to_store(text)
    except Exception as err:  # noqa: BLE001
        # Log but still succeed so front‑end isn’t blocked during dev
        print("[vector_store] add_to_store failed:", err)
    return jsonify({"status": "ok", "filename": file.filename})


@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question", "").strip()
    if not q:
        abort(400, "No question.")

    try:
        context = vs.search(q, k=4)  # returns str
    except Exception as err:  # noqa: BLE001
        print("[vector_store] search failed:", err)
        context = ""

    prompt = (
        "You are ChatAuNP, an AI assistant that designs gold nanomaterial syntheses. "
        "Use the provided context unless general chemistry knowledge is required. "
        "Provide concrete numerical parameters on the same volume scale as the paper. "
        "Response format (replace []):\n\n"
        "1. **Materials**: \n[]\n"
        "2. **Procedure**\n[]\n"
        "3. **Characterization**:\n[]\n\n"
        f"Context:\n{context}\n\nUser question: {q}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return jsonify({"answer": answer})


@app.route("/parse", methods=["POST"])
def parse_route():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    if not text:
        abort(400, "JSON must contain non-empty 'text'.")
    try:
        parsed = convert_to_json(text)
    except ParserError as e:
        abort(422, str(e))
    return jsonify(parsed)


@app.route("/ping")
def ping():
    return jsonify({"status": "alive"})

@app.errorhandler(400)
@app.errorhandler(422)
@app.errorhandler(500)
def handle_error(e):
    return jsonify(error=str(e)), getattr(e, "code", 500)

# ── local dev helper ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
