# Allama RAG

Search for relevant video segments by question. Returns YouTube links with start/end times and transcript snippets.

## Structure

- **`main.py`** — Single API entry: FastAPI server (POST /api/search) + serves classic web app at `/`. Run with `uvicorn main:app`.
- **`src/core.py`** — Core RAG logic (Qdrant + Google GenAI, no LangChain). Used by `main.py` and `app_direct.py`.
- **`web/`** — Classic frontend (HTML/CSS/JS); served at `/` when you run `main:app`.
- **`app.py`** — Streamlit app **with LangChain** (QdrantVectorStore, LangChain embeddings/retriever).
- **`app_direct.py`** — Streamlit app **without LangChain** (uses `src.core` only).

## Environment

Use the **WORKforCompassion** conda environment. Set in `.env`:

- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `QDRANT_ENDPOINT`
- `QDRANT_API_KEY`

## Run

**Streamlit with LangChain:**
```bash
conda activate WORKforCompassion
streamlit run app.py
```

**Streamlit without LangChain (direct Qdrant + GenAI):**
```bash
conda activate WORKforCompassion
streamlit run app_direct.py
```

**API + classic website** (run from **project root** so `web/` is found):
```bash
conda activate WORKforCompassion
cd /path/to/allama_rag
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Then open http://localhost:8000 — same UI as Streamlit: search, exact match clip, main video, transcript, prev/next, view all results.

**API only (JSON)** (start the server with the command above first):
```bash
curl -X POST http://localhost:8000/api/search -H "Content-Type: application/json" -d '{"query": "How are humans created?"}'
```
Response: `{ "results": [{ "start", "end", "url", "transcript_snippet" }, ...], "exact_results": [...], "message": "ok" }`

## Install

```bash
conda activate WORKforCompassion
pip install -r requirements.txt
```
