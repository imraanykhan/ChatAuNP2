from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from openai import OpenAI
from PyPDF2 import PdfReader
import tiktoken
import numpy as np

import vector_store as vs

# ---------------------------------------------------------------------------
# setup ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ENC = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL = "text-embedding-3-small"
MAX_TOKENS = 256  # chunk size (≈ 512 chars)

# ---------------------------------------------------------------------------
# utilities ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _embed(texts: List[str]):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([r.embedding for r in resp.data], dtype="float32")


def _chunk(text: str) -> List[str]:
    out, buf = [], []
    count = 0
    for tok in ENC.encode(text):
        buf.append(tok)
        count += 1
        if count >= MAX_TOKENS:
            out.append(ENC.decode(buf))
            buf, count = [], 0
    if buf:
        out.append(ENC.decode(buf))
    return out


def _extract_text(pdf_bytes: bytes) -> str:
    """Extract plain‑text layer only (no OCR)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "
".join(p.extract_text() or "" for p in reader.pages)

# ---------------------------------------------------------------------------
# routes ---------------------------------------------------------------------
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")
    if not file or not file.filename.lower().endswith(".pdf"):
        return redirect(url_for("home"))

    text = _extract_text(file.read())
    if not text.strip():
        return "No readable text found in PDF", 400

    chunks = _chunk(text)
    vecs = _embed(chunks)
    vs.add_embeddings(vecs, chunks)
    return redirect(url_for("home"))


@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question", "").strip()
    if not q:
        return jsonify({"error": "empty question"}), 400

    q_vec = _embed([q])
    top = vs.query(q_vec, k=6)

    context = "\n---\n".join(c for c, _ in top)
    prompt = (
        "You are ChatAuNP, an AI assistant that designs gold‑nanoparticle syntheses. "
        "Answer the user using only the information in the context unless general knowledge about the Turkevich method is required.\n\n"
        f"Context:\n{context}\n\nUser question: {q}"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap + fast; upgrade as needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content
    return jsonify({"answer": answer})


# ---------------------------------------------------------------------------
# CLI helper -----------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
