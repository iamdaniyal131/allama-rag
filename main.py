"""
Allama RAG - Main entry point (run from project root).
Serves the API and the classic web app. Uses src.core for logic.

Run from project root:
  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# Project root = directory where this file lives
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

# Web app lives next to main.py
WEB_DIR = os.path.join(ROOT, "web")
INDEX_HTML = os.path.join(WEB_DIR, "index.html")
ASSETS_DIR = os.path.join(WEB_DIR, "assets")

from src.core import initialize_clients, process_question

_qdrant = None
_genai = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qdrant, _genai
    try:
        _qdrant, _genai = initialize_clients()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize clients: {e}") from e
    yield
    _qdrant = None
    _genai = None


app = FastAPI(
    title="Allama RAG API",
    description="Search for relevant video segments by question.",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    results: list[dict]
    exact_results: list[dict]
    message: str = "ok"


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required and cannot be empty")
    if _qdrant is None or _genai is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        results, exact_results = process_question(request.query.strip(), _qdrant, _genai)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    def _plain(val):
        if hasattr(val, "item"):
            return val.item()
        if isinstance(val, dict):
            return {k: _plain(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_plain(x) for x in val]
        return val

    exact_serializable = [_plain(r) for r in exact_results]
    return SearchResponse(results=results, exact_results=exact_serializable)


@app.get("/api/health")
def health():
    return {"status": "ok", "clients_ready": _qdrant is not None and _genai is not None}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Avoid 404 when browser requests favicon."""
    return Response(status_code=204)


# Serve classic web app from ./web (same folder as main.py)
@app.get("/")
@app.get("/index.html")
def serve_index():
    if os.path.isfile(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    from fastapi.responses import HTMLResponse
    return HTMLResponse("<h1>Allama RAG</h1><p>Web folder not found.</p>", status_code=404)


if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
