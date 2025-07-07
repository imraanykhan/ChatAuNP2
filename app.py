from __future__ import annotations

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from flask import (
    Flask, jsonify, render_template, request, abort, send_from_directory
)
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import tiktoken

from backend.parser import convert_to_json, ParserError

import vector_store as vs   # FAISS helper

# ── basic setup ────────────────────────────────────────────────────────────
load_dotenv()
app   = Flask(__name__, template_folder="templates", static_folder="static")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ENC         = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL = "text-embedding-3-small"
MAX_TOKENS  = 256

# ── utilities (unchanged) ─────────────────────────────────────────────────
def _embed(texts: List[str]) -> np.ndarray: ...
def _chunk(text: str) -> List[str]: ...
def _extract_text(pdf_bytes: bytes) -> str: ...

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
    text = _extract_text(file.read())
    vs.add_to_store(text)            # FAISS helper
    return jsonify({"status": "ok", "filename": file.filename})

@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question", "")
    if not q:
        abort(400, "No question.")
    context = vs.search(q, k=4)      # returns concatenated chunks
    prompt = (
        "You are ChatAuNP, an AI assistant that designs gold nanomaterials "
        "chemical syntheses. Answer the user *strictly* using the context unless general "
        "knowledge about synthetic methods is required. Assume that you will need to modify the experimental section "
        "in the provided context using the discussion and results of the uploaded context. Provide specific numerical "
        "parameters on the same volume scale as the paper. This includes, but is not limited to, concentrations, volumes, masses, and temperature. "
        "Do not only state modifications, also repeat the parts of the paper you believe should be kept the same. Every concentration should have a volume. "
        "Please respond exactly in the following format, replacing the [] with relevant information. For the procedure, do not separately produce a numbered list. Instead, produce a list within the **Procedure** block. \n\n "
        '''
        1. **Materials**: 
        []
        2. **Procedure**
        []
        3. **Characterization**:
        []
        '''
        )
    f"Context:\n{context}\n\nUser question: {q}"
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    )
    answer = response.choices[0].message.content
    return jsonify({"answer": answer})
    

#JSON exporter ---------------------------------------------------
@app.route("/parse", methods=["POST"])
def parse():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        abort(400, "JSON must contain non-empty 'text'.")
    try:
        payload = convert_to_json(text)
    except ParserError as e:
        abort(422, str(e))
    return jsonify(payload)

@app.route("/ping")
def ping():
    return jsonify({"status": "alive"})

# ── local dev helper ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
