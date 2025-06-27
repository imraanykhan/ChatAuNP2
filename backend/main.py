# backend/main.py
"""Entry‑point for the ChatAu web service.

This module boots a FastAPI app that serves:
  • static front‑end (the HTML/JS in ./frontend)
  • /ask    → proxy to your LLM / RAG back‑end (placeholder stub)
  • /upload → PDF upload endpoint (stub)
  • /parse  → calls backend.parser.convert_to_json

Start locally with:
    uvicorn backend.main:app --reload

Render detects this file via `startCommand` in render.yaml.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

# ---------- local modules ----------
from backend.parser import convert_to_json, ParserError  # noqa: E402  (after import fix)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="ChatAu service")

# -- CORS (helpful if you ever split front/back origins) --
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# -- Serve static files ------------------------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

# ---------------------------------------------------------------------------
# Models & routes
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    """Stub that echoes back the question.
    Replace with call to your GPT‑4o / RAG engine.
    """
    # TODO: plug in your LLM logic here
    answer = f"[stub answer] You asked: {req.question}"
    return {"answer": answer}


@app.post("/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    """Accept a PDF, return simple acknowledgement.
    Extend with real embedding / DB‑insert logic.
    """
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    content = await pdf.read()  # bytes – store or process as needed
    size_kb = len(content) / 1024
    return {"status": "ok", "filename": pdf.filename, "size_kb": round(size_kb, 1)}


class ParseRequest(BaseModel):
    text: str


@app.post("/parse")
async def parse(req: ParseRequest):
    """Convert free‑text synthesis protocol → structured JSON."""
    try:
        data = convert_to_json(req.text)
        return JSONResponse(content=data)
    except ParserError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# Health check (optional) -----------------------------------------------------
# ---------------------------------------------------------------------------
@app.get("/ping")
async def ping():
    return {"status": "alive"}
