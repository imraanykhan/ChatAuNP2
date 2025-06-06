"""
app.py – ChatAuNP v2
• Model: GPT-4o (text only; you can add image/audio later)
• Upload PDFs → FAISS vector DB (persisted to disk or Render volume)
• Retrieval-Augmented Generation (RAG) injected into the prompt
• Endpoints:
    /             – ask a question
    /upload       – POST a PDF (multipart-form) to grow the knowledge base
    /convert      – POST human-friendly procedure and get JSON back
"""

from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os, textwrap, datetime, pickle, uuid, io, pathlib

import faiss                # pip install faiss-cpu
import numpy as np
from PyPDF2 import PdfReader # pip install PyPDF2

# ---------------------------------------------------------------------
# ENV + OpenAI client
# ---------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"  # cheapest high-recall embedding model
GPT_MODEL  = "gpt-4o"                  # new flagship model :contentReference[oaicite:0]{index=0}

# ---------------------------------------------------------------------
# Vector-store helpers
# ---------------------------------------------------------------------
STORE_DIR      = pathlib.Path("vector_store")
INDEX_FILE     = STORE_DIR / "papers.index"
MAPPING_FILE   = STORE_DIR / "mapping.pkl"
CHUNK_SIZE_CHR = 1000                  # naïve char split; tweak later
K_RETRIEVE     = 8                     # how many chunks to feed GPT

STORE_DIR.mkdir(exist_ok=True)

def _load_index():
    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
    else:
        # use cosine similarity (inner product on normalized vectors)
        index = faiss.IndexFlatIP(1536)  # 1536 dims for text-embedding-3
    return index

def _load_mapping():
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def _save_index(index):
    faiss.write_index(index, str(INDEX_FILE))  # persistence :contentReference[oaicite:1]{index=1}

def _save_mapping(mapping):
    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(mapping, f)

index   = _load_index()
mapping = _load_mapping()

def _embed(texts: list[str]) -> np.ndarray:
    """Call OpenAI embeddings and return normalized np.array [n,1536]."""
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in res.data]).astype("float32")
    # cosine similarity needs normalized vectors
    faiss.normalize_L2(vecs)
    return vecs

def add_document(doc_id: str, chunks: list[str]):
    """Add chunks of one document to the FAISS index."""
    vecs = _embed(chunks)
    index.add(vecs)
    mapping[len(mapping)] = {"id": doc_id, "chunks": chunks}
    _save_index(index)
    _save_mapping(mapping)

def retrieve(query: str, k: int = K_RETRIEVE) -> str:
    """Return concatenated top-k chunk texts relevant to the query."""
    if index.ntotal == 0:
        return ""  # nothing uploaded yet
    q_vec = _embed([query])
    D, I = index.search(q_vec, k)
    results = []
    for idx in I[0]:
        if idx == -1:  # pad from FAISS
            continue
        # find which mapping bucket this idx belongs to
        # (since we add vectors in doc order, each mapping key covers len(chunks))
        offset = 0
        for key, info in mapping.items():
            n = len(info["chunks"])
            if idx < offset + n:
                chunk = info["chunks"][idx - offset]
                results.append(chunk)
                break
            offset += n
    return "\n".join(results)

# ---------------------------------------------------------------------
# Prompt scaffolding
# ---------------------------------------------------------------------
BASE_SYSTEM_PROMPT = textwrap.dedent(f"""
You are ChatAuNP, an expert gold-nanoparticle synthesis advisor.
When given a request, OUTPUT **only** a standardized procedure in this exact
human-friendly block format (Markdown fences are OK):

Title: <concise title>
Materials:
  - <material 1>
  - <material 2>
Apparatus:
  - <equipment 1>
Parameters:
  Temperature: <°C>
  pH: <value or range>
Steps:
  1. <imperative sentence>
  2. ...
Safety:
  - <key safety note>
""").strip()

def ask_chataunp(question: str) -> str:
    """RAG-augmented GPT-4o call."""
    retrieved = retrieve(question)
    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
    ]
    if retrieved:
        messages.append(
            {"role": "system",
             "content": f"### RELEVANT PAPERS\n{retrieved}\n---"}
        )
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------
# Flask app + routes
# ---------------------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index_route():
    answer = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer = ask_chataunp(question)
            with open("log.txt", "a") as f:
                ts = datetime.datetime.utcnow().isoformat()
                f.write(f"{ts}\nQ: {question}\nA: {answer}\n---\n")
    return render_template("index.html", answer=answer)

@app.route("/upload", methods=["POST"])
def upload_route():
    """Upload a PDF (multipart/form-data ‘file’); splits into chunks + indexes."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF accepted"}), 400
    doc_id = f"{uuid.uuid4()}"
    reader = PdfReader(file)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # naive fixed-size char chunks – good enough for demo
    chunks = [full_text[i : i + CHUNK_SIZE_CHR] for i in range(0, len(full_text), CHUNK_SIZE_CHR) if full_text[i : i + CHUNK_SIZE_CHR].strip()]
    add_document(doc_id, chunks)
    return jsonify({"status": "indexed", "chunks": len(chunks)})

@app.route("/convert", methods=["POST"])
def convert_route():
    """POST {procedure: <string>} → JSON; trivial parse using regex."""
    import re, json
    text = request.json.get("procedure", "")
    def _get_block(name):
        m = re.search(rf"{name}:(.*?)(\n[A-Z][a-zA-Z]+:|\Z)", text, re.S)
        return (m.group(1).strip() if m else "")
    out = {
        "title": _get_block("Title"),
        "materials": [line.strip("- • ") for line in _get_block("Materials").splitlines() if line.strip()],
        "apparatus": [line.strip("- • ") for line in _get_block("Apparatus").splitlines() if line.strip()],
        "parameters": {},
        "steps": [],
        "safety": [line.strip("- • ") for line in _get_block("Safety").splitlines() if line.strip()],
    }
    # parameters (temperature, pH) and numbered steps
    param_text = _get_block("Parameters")
    for line in param_text.splitlines():
        if ":" in line:
            k, v = [s.strip() for s in line.split(":", 1)]
            out["parameters"][k.lower()] = v
    step_text = _get_block("Steps")
    for i, line in enumerate(step_text.splitlines(), 1):
        clean = re.sub(r"^\d+\.\s*", "", line).strip()
        if clean:
            out["steps"].append({"step": i, "description": clean})
    return jsonify(out)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
