"""
ASK with Allama - Core logic (no UI).
Uses Qdrant client and Google GenAI embeddings directly.
Can be used by Streamlit, FastAPI, or any other consumer.
"""
from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine
from google import genai
from google.genai import types

load_dotenv()

COLLECTION_NAME = "allama-rag-dev"
EMBEDDING_MODEL = "models/gemini-embedding-001"
SEARCH_K = 20
SCORE_THRESHOLD = 0.65

_qdrant_client: QdrantClient | None = None
_genai_client: Any = None


def get_embedding(text: str, client: Any, model: str | None = None) -> list[float]:
    """Get embedding for text using Google GenAI Client."""
    if client is None:
        raise RuntimeError("GenAI client not initialized")
    model = model or EMBEDDING_MODEL
    result = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    if not result.embeddings:
        raise ValueError("Empty embedding response")
    emb = result.embeddings[0]
    if hasattr(emb, "values"):
        return list(emb.values)
    if isinstance(emb, list):
        return emb
    raise ValueError("Unexpected embedding response format")


def fetch_datapoints_from_qdrant(file_name: str, qdrant_client: QdrantClient) -> list:
    results = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.file_name",
                    match=MatchValue(value=file_name),
                )
            ]
        ),
        limit=1000,
        with_vectors=True,
    )
    return results[0]


def cosine_similarity_scipy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(1 - cosine(vec1, vec2))


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def find_time_range_around_pinned(data_list: list[dict]) -> dict | None:
    pinned_idx = None
    for idx, item in enumerate(data_list):
        if item.get("pinned", False):
            pinned_idx = idx
            break

    if pinned_idx is None:
        return None

    left_idx = pinned_idx
    current_max = 0.0
    window_start = pinned_idx - 1

    while window_start >= 0:
        window_end = max(0, window_start - 2)
        window_indices = list(range(window_start, window_end - 1, -1))
        best_in_window = None
        best_similarity = current_max
        for i in window_indices:
            if data_list[i].get("pinned", False):
                continue
            similarity = data_list[i].get("cosine_similarity", -1.0)
            if similarity > best_similarity:
                best_similarity = similarity
                best_in_window = i
        if best_in_window is not None:
            left_idx = best_in_window
            current_max = best_similarity
            window_start -= 1
        else:
            break

    right_idx = pinned_idx
    current_max = 0.0
    window_start = pinned_idx + 1

    while window_start < len(data_list):
        window_end = min(len(data_list), window_start + 3)
        window_indices = list(range(window_start, window_end))
        best_in_window = None
        best_similarity = current_max
        for i in window_indices:
            if data_list[i].get("pinned", False):
                continue
            similarity = data_list[i].get("cosine_similarity", -1.0)
            if similarity > best_similarity:
                best_similarity = similarity
                best_in_window = i
        if best_in_window is not None:
            right_idx = best_in_window
            current_max = best_similarity
            window_start += 1
        else:
            break

    return {
        "start": data_list[left_idx]["start"] if left_idx is not None else None,
        "end": data_list[right_idx]["end"] if right_idx is not None else None,
        "start_idx": left_idx,
        "end_idx": right_idx,
        "pinned_item": data_list[pinned_idx],
    }


def initialize_clients() -> tuple[QdrantClient, Any]:
    """Initialize Qdrant and GenAI clients. Call once at startup."""
    global _qdrant_client, _genai_client
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env")
    _genai_client = genai.Client(api_key=api_key)
    _qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False,
    )
    return _qdrant_client, _genai_client


def get_clients() -> tuple[QdrantClient | None, Any]:
    """Return cached clients (if initialized)."""
    return _qdrant_client, _genai_client


def process_question(
    question: str,
    qdrant_client: QdrantClient,
    genai_client: Any,
) -> tuple[list[dict], list[dict]]:
    """
    Run retrieval and time-range expansion.
    Returns (results, exact_results).
    - results: list of { start, end, url, transcript_snippet }
    - exact_results: list of metadata dicts (start, end, url, file_name, ...) for exact match preview.
    """
    query_vector = get_embedding(question, client=genai_client)
    query_vector = np.array(query_vector, dtype=np.float32)

    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=SEARCH_K,
    )
    res = []
    for hit in hits:
        payload = hit.payload or {}
        metadata = payload.get("metadata", {})
        score = getattr(hit, "score", None)
        if score is not None and hasattr(score, "item"):
            score = float(score.item())
        elif score is not None:
            score = float(score)
        res.append({
            "metadata": metadata,
            "page_content": payload.get("page_content", ""),
            "score": score,
        })

    # Keep only hits with cosine similarity above threshold
    res = [r for r in res if r.get("score") is not None and r["score"] > SCORE_THRESHOLD]

    result_list = []
    exact_result_datapoint_list = []

    for selected_retrieved_doc in range(len(res)):
        meta = res[selected_retrieved_doc]["metadata"]
        file_name = meta.get("file_name")
        if not file_name:
            continue
        matched_data_points = fetch_datapoints_from_qdrant(file_name, qdrant_client)

        final_matched_data_points = []
        for i in matched_data_points:
            payload = i.payload or {}
            meta_i = payload.get("metadata", {})
            final_matched_data_points.append({
                "payload": payload.get("page_content", ""),
                "vector": np.array(i.vector),
                **meta_i,
            })

        for i in final_matched_data_points:
            if i["start"] == meta.get("start") and i["end"] == meta.get("end"):
                selected_embedding = i["vector"]
                i["pinned"] = True
            else:
                i["pinned"] = False

        final_matched_data_points.sort(key=lambda x: x["start"])

        for i in final_matched_data_points:
            i["cosine_similarity"] = cosine_similarity_scipy(selected_embedding, i["vector"])

        result = find_time_range_around_pinned(final_matched_data_points)
        if not result or "url" not in result["pinned_item"]:
            continue

        transcript_snippet = "\n".join(
            text["payload"]
            for text in final_matched_data_points[
                result["start_idx"] : result["end_idx"] + 1
            ]
        )
        hit_score = res[selected_retrieved_doc].get("score")
        result_list.append({
            "start": int(result["start"]),
            "end": int(result["end"]),
            "url": result["pinned_item"]["url"],
            "transcript_snippet": transcript_snippet,
            "score": hit_score,
        })
        exact_result_datapoint_list.append({**meta, "score": hit_score})

    return result_list, exact_result_datapoint_list


def get_youtube_embed_url(url: str, start: int | float, end: int | float) -> str:
    """YouTube embed URL (no autoplay)."""
    video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]
    return f"https://www.youtube.com/embed/{video_id}?start={int(start)}&end={int(end)}&autoplay=0"


def get_small_embed(url: str, start: int | float, end: int | float) -> str:
    """YouTube embed URL for small preview (autoplay)."""
    video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]
    return f"https://www.youtube.com/embed/{video_id}?start={int(start)}&end={int(end)}&autoplay=1"
