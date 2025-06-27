from __future__ import annotations

import io
import os
from typing import List

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from openai import OpenAI
from PyPDF2 import PdfReader
import tiktoken
import numpy as np

import vector_store as vs

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
load_dotenv()                                   # loads OPENAI_API_KEY from .env

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ENC = tiktoken.get_encoding("cl100k_base")      # same tokenizer OpenAI uses
EMBED_MODEL = "text-embedding-3-small"
MAX_TOKENS = 256                               


# utilities


def _embed(texts: List[str]) -> np.ndarray:
    """Call the OpenAI embeddings endpoint and return a (n, d) float32 array."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([r.embedding for r in resp.data], dtype="float32")


def _chunk(text: str) -> List[str]:
    """Split long text into ~MAX_TOKENS-sized chunks so FAISS can handle them."""
    out, buf, count = [], [], 0
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
    """Extract plain-text layer from the PDF (no OCR)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# routes


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
    return redirect(url_for("home"))             # refresh the UI


@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question", "").strip()
    if not q:
        return jsonify({"error": "empty question"}), 400

    q_vec = _embed([q])
    top = vs.query(q_vec, k=6)                

    context = "\n---\n".join(c for c, _ in top)
    prompt = (
        "You are ChatAuNP, an AI assistant that designs gold nanomaterials "
        "chemical syntheses. Answer the user *strictly* using the context unless general "
        "knowledge about synthetic methods is required. Assume that you will need to modify the experimental section "
        "in the provided context using the discussion and results of the uploaded context. Provide specific numerical "
        "parameters on the same volume scale as the paper. This includes, but is not limited to, concentrations, volumes, masses, and temperature. "
        "Do not only state modifications, also repeat the parts of the paper you believe should be kept the same. Every concentration should have a volume. "
        "Please respond exactly in the following format, replacing the [] with relevant information. For the procedure, do not separately produce a numbered list. Instead, produce a list within the **Procedure** block. \n\n "
"'''
1. **Materials**: 
[]
2. **Procedure**
[]
3. **Characterization**:
[]
''' \n\n "
        f"Context:\n{context}\n\nUser question: {q}"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",                     # cost
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content
    return jsonify({"answer": answer})


# CLI helper

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
