import streamlit as st
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
from scipy.spatial.distance import cosine
import asyncio
import nest_asyncio
from langchain_core.prompts import PromptTemplate


# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'question_submitted' not in st.session_state:
    st.session_state.question_submitted = False

# Helper functions
def fetch_datapoints_from_qdrant(file_name, qdrant_client):
    results = qdrant_client.scroll(
        collection_name="allama-rag-dev",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.file_name",
                    match=MatchValue(value=file_name)
                )
            ]
        ),
        limit=1000,
        with_vectors=True
    )
    return results[0]

def cosine_similarity_scipy(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"

def find_time_range_around_pinned(data_list):
    pinned_idx = None
    for idx, item in enumerate(data_list):
        if item.get('pinned', False):
            pinned_idx = idx
            break
    
    if pinned_idx is None:
        return None
    
    # Search left with sliding window (stride = 1)
    left_idx = pinned_idx
    # current_max = data_list[pinned_idx].get('cosine_similarity', -1)
    current_max= 0
    window_start = pinned_idx - 1
    
    while window_start >= 0:
        # Define window of 3 points (or fewer if near boundary)
        window_end = max(0, window_start - 2)
        window_indices = list(range(window_start, window_end - 1, -1))
        
        # Find best point in current window that's greater than current_max
        # Skip any pinned points
        best_in_window = None
        best_similarity = current_max
        
        for i in window_indices:
            if data_list[i].get('pinned', False):
                continue  # Skip pinned points
            similarity = data_list[i].get('cosine_similarity', -1)
            if similarity > best_similarity:
                best_similarity = similarity
                best_in_window = i
        
        # If found a better point, update and move window by 1
        if best_in_window is not None:
            left_idx = best_in_window
            current_max = best_similarity
            window_start -= 1  # Move window by stride of 1
        else:
            # No better point found, stop searching
            break
    
    # Search right with sliding window (stride = 1)
    right_idx = pinned_idx
    # current_max = data_list[pinned_idx].get('cosine_similarity', -1)
    current_max= 0
    window_start = pinned_idx + 1
    
    while window_start < len(data_list):
        # Define window of 3 points (or fewer if near boundary)
        window_end = min(len(data_list), window_start + 3)
        window_indices = list(range(window_start, window_end))
        
        # Find best point in current window that's greater than current_max
        # Skip any pinned points
        best_in_window = None
        best_similarity = current_max
        
        for i in window_indices:
            if data_list[i].get('pinned', False):
                continue  # Skip pinned points
            similarity = data_list[i].get('cosine_similarity', -1)
            if similarity > best_similarity:
                best_similarity = similarity
                best_in_window = i
        
        # If found a better point, update and move window by 1
        if best_in_window is not None:
            right_idx = best_in_window
            current_max = best_similarity
            window_start += 1  # Move window by stride of 1
        else:
            # No better point found, stop searching
            break
    
    result = {
        'start': data_list[left_idx]['start'] if left_idx is not None else None,
        'end': data_list[right_idx]['end'] if right_idx is not None else None,
        'start_idx': left_idx,
        'end_idx': right_idx,
        'pinned_item': data_list[pinned_idx]
    }
    return result


@st.cache_resource
def initialize_qdrant():
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        chat_model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        qdrant_client = QdrantClient(
            url=os.getenv('QDRANT_ENDPOINT'), 
            api_key=os.getenv('QDRANT_API_KEY'),
            prefer_grpc=False  # Use REST API instead of gRPC
        )
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name="allama-rag-dev",
            embedding=embedding_model,
        )
        return qdrant_client, vector_store, chat_model, embedding_model
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        raise

def process_question(question, qdrant_client, vector_store, chat_model, embedding_model):
    
    with open('extract_query_ans.txt', 'r') as f:
        extraction_prompt_txt= f.read()
    template= PromptTemplate(
        template= extraction_prompt_txt,
        input_variables= ["QUERY", "CONTEXT"],
        validate_template= True
    )

    
    ques_retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5}
    )
    res = ques_retriever.invoke(question)
    
    result_list = []
    exact_result_datapoint_list= []
    
    for selected_retrieved_doc in range(len(res)):
        file_name = res[selected_retrieved_doc].metadata['file_name']
        matched_data_points = fetch_datapoints_from_qdrant(file_name, qdrant_client)
        
        final_matched_data_points = []
        for i in matched_data_points:
            final_matched_data_points.append({
                'payload': i.payload['page_content'],
                'vector': np.array(i.vector)
            } | i.payload['metadata'])
        
        for i in final_matched_data_points:
            if (i['start'] == res[selected_retrieved_doc].metadata['start'] and 
                i['end'] == res[selected_retrieved_doc].metadata['end']):
                selected_embedding = i['vector']
                selected_context= i['payload']
                i['pinned'] = True
            else:
                i['pinned'] = False
        
        final_matched_data_points = sorted(final_matched_data_points, key=lambda x: x['start'])

        # extraction_prompt= template.invoke({"QUERY": question, "CONTEXT": selected_context})
        # extracted_context = chat_model.invoke(extraction_prompt).content
        # extracted_context_embedding= np.array(embedding_model.embed_query(extracted_context))
        
        for i in final_matched_data_points:
            i['cosine_similarity'] = cosine_similarity_scipy(selected_embedding, i['vector']) 
            # i['cosine_similarity'] = cosine_similarity_scipy(extracted_context_embedding, i['vector'])
        
        result = find_time_range_around_pinned(final_matched_data_points)

        transcript_snippet= "\n".join([text['payload'] for text in final_matched_data_points[result['start_idx']: result['end_idx'] + 1]])
        
        if result and 'url' in result['pinned_item']:
            result_list.append({
                'start': int(result['start']),
                'end': int(result['end']),
                'url': result['pinned_item']['url'],
                'transcript_snippet': transcript_snippet
            })
        
            exact_result_datapoint_list.append(res[selected_retrieved_doc].metadata)
    # print(result_list)
        # print(res[selected_retrieved_doc].metadata)
    
    # print(result_list[0])
    # print(exact_result_datapoint_list[0])

    return result_list, exact_result_datapoint_list

def get_youtube_embed_url(url, start, end):
    """Convert YouTube URL to embed format with start and end times"""
    video_id = url.split('v=')[-1].split('&')[0] if 'v=' in url else url.split('/')[-1]
    return f"https://www.youtube.com/embed/{video_id}?start={start}&end={end}&autoplay=0"

def get_small_embed(url, start, end):
    video_id = url.split('v=')[-1].split('&')[0] if 'v=' in url else url.split('/')[-1]
    # print(f"https://www.youtube.com/embed/{video_id}?start={start}&end={end}&autoplay=1")
    return f"https://www.youtube.com/embed/{video_id}?start={int(start)}&end={int(end)}&autoplay=1"

# Streamlit UI
st.set_page_config(page_title="YouTube RAG Player", layout="wide")
st.title("🎥 YouTube RAG Video Player")

# Initialize Qdrant
try:
    qdrant_client, vector_store, chat_model, embedding_model = initialize_qdrant()
except Exception as e:
    st.error(f"Failed to initialize Qdrant: {str(e)}")
    st.stop()

# Question input
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., How human are created, are they created from monkey?",
        key="question_input"
    )
with col2:
    st.write("")  # Spacing
    submit_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Process question
if submit_button and question:
    with st.spinner("Searching for relevant videos..."):
        try:
            results, exact_result_datapoint_list = process_question(question, qdrant_client, vector_store, chat_model, embedding_model)
            st.session_state.results = results
            st.session_state.exact_result = exact_result_datapoint_list
            st.session_state.current_index = 0
            st.session_state.question_submitted = True
            
            if not results:
                st.warning("No results found for your question.")
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

# Display video player and navigation
if st.session_state.question_submitted and st.session_state.results:
    results = st.session_state.results
    exact_result_datapoint_list= st.session_state.exact_result
    idx = st.session_state.current_index
    
    st.markdown("---")
    
    # Video counter
    st.markdown(f"### Video {idx + 1} of {len(results)}")

######################
    # Small preview of exact datapoint
if 'exact_result' in st.session_state and st.session_state.exact_result:

    st.markdown("### 🎯 Exact Datapoint Match")

    small_embed_url = get_small_embed(
        exact_result_datapoint_list[idx]['url'],
        exact_result_datapoint_list[idx]['start'],
        exact_result_datapoint_list[idx]['end']
    )

    st.markdown(
        f'<iframe width="350" height="200" src="{small_embed_url}" '
        f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
        f'encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
        unsafe_allow_html=True
    )

    # Show small info
    st.caption(
        f"🔍 Exact Match: {seconds_to_hhmmss(exact_result_datapoint_list[idx]['start'])} → {seconds_to_hhmmss(exact_result_datapoint_list[idx]['end'])}"
    )
################

    st.markdown("### 🎯 Explanation with context")
    
    # Get current video details
    current_video = results[idx]
    embed_url = get_youtube_embed_url(
        current_video['url'],
        current_video['start'],
        current_video['end']
    )
    
    # Display video
    st.markdown(
        f'<iframe width="100%" height="500" src="{embed_url}" '
        f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
        f'encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    #display transcript snippet
    st.subheader("📝 Transcript Snippet")
    st.text_area(
        label="Transcript",
        value=current_video.get("transcript_snippet", "No transcript available."),
        height=140,
        disabled=True
    )
    
    # Video info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"⏰ Start: {seconds_to_hhmmss(current_video['start'])}")
    with col2:
        st.info(f"⏱️ End: {seconds_to_hhmmss(current_video['end'])}")
    with col3:
        st.info(f"⏳ Duration: {current_video['end'] - current_video['start']}s")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ Previous", disabled=(idx == 0), use_container_width=True):
            st.session_state.current_index -= 1
            st.rerun()
    
    with col2:
        st.markdown(f"<center><b>Video {idx + 1} / {len(results)}</b></center>", 
                   unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ➡️", disabled=(idx >= len(results) - 1), use_container_width=True):
            st.session_state.current_index += 1
            st.rerun()
    
    # Show all results in expandable section
    with st.expander("📋 View All Results"):
        for i, video in enumerate(results):
            status = "▶️ Currently Playing" if i == idx else ""
            st.markdown(f"**{i + 1}.** {video['url']} ({video['start']}s - {video['end']}s) {status}")

else:
    st.info("👆 Enter a question above to search for relevant video segments")

# Sidebar with instructions
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
    This app uses RAG (Retrieval-Augmented Generation) to find 
    the most relevant video segments from your question.
    """)