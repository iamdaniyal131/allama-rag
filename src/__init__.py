"""Allama RAG - core logic (used by main.py and app_direct.py)."""

from src.core import (
    process_question,
    initialize_clients,
    seconds_to_hhmmss,
    get_youtube_embed_url,
    get_small_embed,
)

__all__ = [
    "process_question",
    "initialize_clients",
    "seconds_to_hhmmss",
    "get_youtube_embed_url",
    "get_small_embed",
]
