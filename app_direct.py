"""
Allama RAG - Direct Streamlit entry point (uses src.core).
Run with: streamlit run app_direct.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.core import (
    initialize_clients,
    process_question,
    seconds_to_hhmmss,
    get_youtube_embed_url,
    get_small_embed,
)

if "results" not in st.session_state:
    st.session_state.results = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False

@st.cache_resource
def get_clients():
    return initialize_clients()

st.set_page_config(page_title="YouTube RAG Player (Direct)", layout="wide")
st.title("🎥 Ask question directly to Allama Sir")

try:
    qdrant_client, genai_client = get_clients()
except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.stop()

col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., How human are created, are they created from monkey?",
        key="question_input",
    )
with col2:
    st.write("")
    submit_button = st.button("🔍 Search", type="primary", use_container_width=True)

if submit_button and question:
    with st.spinner("Searching for relevant videos..."):
        try:
            results, exact_result_datapoint_list = process_question(
                question, qdrant_client, genai_client
            )
            st.session_state.results = results
            st.session_state.exact_result = exact_result_datapoint_list
            st.session_state.current_index = 0
            st.session_state.question_submitted = True
            if not results:
                st.warning("No results found for your question.")
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

if st.session_state.question_submitted and st.session_state.results:
    results = st.session_state.results
    exact_result_datapoint_list = st.session_state.exact_result
    idx = st.session_state.current_index

    st.markdown("---")
    st.markdown(f"### Video {idx + 1} of {len(results)}")

    if st.session_state.exact_result:
        st.markdown("### 🎯 Exact Datapoint Match")
        small_embed_url = get_small_embed(
            exact_result_datapoint_list[idx]["url"],
            exact_result_datapoint_list[idx]["start"],
            exact_result_datapoint_list[idx]["end"],
        )
        st.markdown(
            f'<iframe width="350" height="200" src="{small_embed_url}" '
            'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
            'encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"🔍 Exact Match: {seconds_to_hhmmss(exact_result_datapoint_list[idx]['start'])} → "
            f"{seconds_to_hhmmss(exact_result_datapoint_list[idx]['end'])}"
        )

    st.markdown("### 🎯 Explanation with context")
    current_video = results[idx]
    embed_url = get_youtube_embed_url(
        current_video["url"], current_video["start"], current_video["end"]
    )
    st.markdown(
        f'<iframe width="100%" height="500" src="{embed_url}" '
        'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
        'encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("📝 Transcript Snippet")
    st.text_area(
        label="Transcript",
        value=current_video.get("transcript_snippet", "No transcript available."),
        height=140,
        disabled=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"⏰ Start: {seconds_to_hhmmss(current_video['start'])}")
    with col2:
        st.info(f"⏱️ End: {seconds_to_hhmmss(current_video['end'])}")
    with col3:
        st.info(f"⏳ Duration: {current_video['end'] - current_video['start']}s")
    with col4:
        score = current_video.get("score")
        if score is not None:
            st.info(f"📊 Cosine similarity: {score:.3f}")
        else:
            st.info("📊 Similarity: —")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=(idx == 0), use_container_width=True):
            st.session_state.current_index -= 1
            st.rerun()
    with col2:
        st.markdown(
            f"<center><b>Video {idx + 1} / {len(results)}</b></center>",
            unsafe_allow_html=True,
        )
    with col3:
        if st.button(
            "Next ➡️", disabled=(idx >= len(results) - 1), use_container_width=True
        ):
            st.session_state.current_index += 1
            st.rerun()

    with st.expander("📋 View All Results"):
        for i, video in enumerate(results):
            status = "▶️ Currently Playing" if i == idx else ""
            score_str = f" (similarity {video.get('score', 0):.3f})" if video.get("score") is not None else ""
            st.markdown(
                f"**{i + 1}.** {video['url']} ({video['start']}s - {video['end']}s){score_str} {status}"
            )
else:
    st.info("👆 Enter a question above to search for relevant video segments")

with st.sidebar:
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Enter your question** in the search box
    2. **Click Search** to find relevant videos
    3. **Watch the video** with automatic start/end times
    4. **Navigate** using Previous/Next buttons
    5. Videos play automatically at the relevant timestamp
    """)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses **src.core** (direct Qdrant + Google embeddings).
    API and classic web app are in `src/api.py` and `web/`.
    """)
