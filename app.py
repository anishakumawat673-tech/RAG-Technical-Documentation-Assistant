"""
DocuMind — AI Technical Documentation Assistant
===============================================
An interactive Streamlit interface for Python, Data Science, Machine Learning,
and Deep Learning documentation Q&A, powered by Sentence Transformers,
FAISS vector search, LangChain, and Google Gemini.

Knowledge Base:
Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, and TensorFlow
"""

import os
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    ConversationManager,
    initialize_rag_pipeline,
    answer_question,
)

# Page configuration
st.set_page_config(
    page_title="DocuMind — Python, Data Science & ML/DL Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 980px;
    }
    
    /* Header badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 0.8rem;
    }
    
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    
    .tech-badge.ml {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
    }
    
    .tech-badge.dl {
        background: linear-gradient(135deg, #db2777 0%, #be185d 100%);
    }
    
    /* Citation chip */
    .citation-card {
        background-color: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }
    
    /* Reformulation indicator */
    .reformulated-query {
        font-size: 0.85rem;
        color: #64748b;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "data", "index", "multilib_faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "data", "processed", "multilib_metadata.json")


@st.cache_resource(show_spinner="Loading FAISS multi-library vector index & AI models...")
def get_pipeline():
    """
    Cached loader for the multi-library RAG pipeline.
    Ensures the 1,282 FAISS vectors, SentenceTransformer model, and Gemini chains
    are loaded into memory once and shared across requests without re-embedding.
    """
    return initialize_rag_pipeline(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conv_mgr" not in st.session_state:
    st.session_state.conv_mgr = ConversationManager()

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None


# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🧠 DocuMind Controls")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.conv_mgr.clear()
        st.session_state.quick_prompt = None
        st.success("Conversation history cleared!")
        st.rerun()

    st.markdown("---")
    
    # Library Filtering
    st.markdown("#### 🎯 Knowledge Base Filter")
    library_options = [
        "All Libraries",
        "Python",
        "NumPy",
        "Pandas",
        "Matplotlib",
        "Scikit-learn",
        "PyTorch",
        "TensorFlow"
    ]
    selected_library = st.selectbox(
        "Filter documentation search:",
        library_options,
        index=0,
        help="Restrict semantic retrieval to a specific library or search across all documentation."
    )

    st.markdown("---")

    # System Status & Pipeline Info
    st.markdown("#### 🛠️ RAG Pipeline Status")
    st.markdown("""
    - **Knowledge Base**: 1,282 semantic chunks
    - **Libraries**: Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, TensorFlow
    - **Retriever**: FAISS `IndexFlatIP` (Cosine similarity)
    - **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions)
    - **LLM**: Google Gemini (`gemini-flash-lite-latest`)
    - **Memory**: Contextual Query Reformulation
    """)

    st.markdown("---")

    # Quick Example Prompts
    st.markdown("#### 💡 Example Questions")
    example_prompts = [
        "What is overfitting and how do I prevent it?",
        "How does GridSearchCV work in Scikit-learn?",
        "How do I create and manipulate a Pandas DataFrame?",
        "Explain Autograd and backpropagation in PyTorch",
        "How do I build a CNN in TensorFlow / Keras?",
        "What is gradient descent and which optimizers use it?"
    ]
    
    for prompt_text in example_prompts:
        if st.button(prompt_text, key=f"ex_{prompt_text}", use_container_width=True):
            st.session_state.quick_prompt = prompt_text
            st.rerun()

    st.markdown("---")
    st.markdown(
        "**DocuMind v2.0** — Powered by LangChain, FAISS & Google Gemini  \n"
        "Official Documentation Assistant for Python & Machine Learning"
    )


# Header Area
st.markdown("""
<div class="badge-container">
    <span class="tech-badge">Python</span>
    <span class="tech-badge">NumPy</span>
    <span class="tech-badge">Pandas</span>
    <span class="tech-badge">Matplotlib</span>
    <span class="tech-badge ml">Scikit-learn</span>
    <span class="tech-badge dl">PyTorch</span>
    <span class="tech-badge dl">TensorFlow</span>
</div>
""", unsafe_allow_html=True)

st.title("DocuMind — AI Technical Documentation Assistant")
st.markdown(
    "Ask conceptual and technical questions about **Python, Data Science, Machine Learning, and Deep Learning**. "
    "Answers are strictly grounded in official documentation from Scikit-learn, PyTorch, TensorFlow, Pandas, NumPy, Matplotlib, and Python."
)
st.markdown("---")


# Check Environment & API Key
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error(
        "⚠️ **Missing Gemini API Key**: Please ensure `GOOGLE_API_KEY` is configured in your `.env` file."
    )
    st.stop()


# Load the RAG Pipeline
try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"❌ **Failed to load RAG pipeline**: {str(e)}")
    st.stop()


# Display Conversation History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if msg.get("search_query") and msg["search_query"] != msg.get("original_question"):
                st.markdown(
                    f"<div class='reformulated-query'>🔍 Reformulated search query: "
                    f"<strong>\"{msg['search_query']}\"</strong></div>",
                    unsafe_allow_html=True
                )
            
            st.markdown(msg["content"])
            
            # Display citations
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 Source Citations & References ({len(sources)})", expanded=False):
                    for src in sources:
                        lib_tag = f"**[{src.get('library', 'Doc')}]** "
                        st.markdown(f"- {lib_tag}[{src['title']}]({src['url']}) — *{src.get('topic', '')}*")
                        
                    # Optional chunk inspect
                    chunks = msg.get("chunks", [])
                    if chunks:
                        st.markdown("---")
                        st.caption("Retrieved Documentation Chunks:")
                        for idx, chunk in enumerate(chunks, 1):
                            st.markdown(
                                f"**[Chunk {idx}]** *[{chunk.get('library', '')}] {chunk['doc_title']}* — "
                                f"Topic: `{chunk.get('topic', '')}` (Cosine Similarity: `{chunk['score']:.4f}`)"
                            )


# Handle User Input (from chat input or quick prompt)
user_query = st.chat_input("Ask a question about Python, Data Science, ML, or Deep Learning...")

# Check if a quick example prompt was clicked
if st.session_state.quick_prompt and not user_query:
    user_query = st.session_state.quick_prompt
    st.session_state.quick_prompt = None


if user_query:
    # 1. Render User Message
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        spinner_msg = (
            f"Searching {selected_library} documentation & generating grounded answer..."
            if selected_library != "All Libraries"
            else "Searching technical documentation & generating grounded answer..."
        )
        with st.spinner(spinner_msg):
            try:
                result = answer_question(
                    question=user_query,
                    conv_mgr=st.session_state.conv_mgr,
                    pipeline=pipeline,
                    top_k=4,
                    library=selected_library
                )

                # Show reformulated search query if it was modified
                if result["search_query"] != user_query:
                    st.markdown(
                        f"<div class='reformulated-query'>🔍 Reformulated search query: "
                        f"<strong>\"{result['search_query']}\"</strong></div>",
                        unsafe_allow_html=True
                    )

                # Render Answer
                st.markdown(result["answer"])

                # Render Citations
                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"📚 Source Citations & References ({len(sources)})", expanded=False):
                        for src in sources:
                            lib_tag = f"**[{src.get('library', 'Doc')}]** "
                            st.markdown(f"- {lib_tag}[{src['title']}]({src['url']}) — *{src.get('topic', '')}*")

                        chunks = result.get("chunks", [])
                        if chunks:
                            st.markdown("---")
                            st.caption("Retrieved Documentation Chunks:")
                            for idx, chunk in enumerate(chunks, 1):
                                st.markdown(
                                    f"**[Chunk {idx}]** *[{chunk.get('library', '')}] {chunk['doc_title']}* — "
                                    f"Topic: `{chunk.get('topic', '')}` (Cosine Similarity: `{chunk['score']:.4f}`)"
                                )

                # 3. Store in Session State
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_query
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "chunks": result.get("chunks", []),
                    "search_query": result["search_query"],
                    "original_question": user_query
                })

            except Exception as e:
                st.error(
                    "An error occurred while generating the answer. "
                    "Please check your API quota or network connection."
                )
                st.caption(f"Error details: {str(e)}")
