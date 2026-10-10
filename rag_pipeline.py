"""
DocuMind RAG Pipeline
=====================
Modular RAG pipeline powering the DocuMind Technical Documentation Assistant
for Python, Data Science, Machine Learning, and Deep Learning libraries:
- Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, TensorFlow
- ML/DL Foundations: Classification, Regression, Overfitting, Regularization,
  Bias-Variance, Gradient Descent, Backpropagation, CNNs, RNNs, Metrics, Preprocessing.

Default Knowledge Base:
- data/index/multilib_faiss.index
- data/processed/multilib_metadata.json
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


class ConversationManager:
    """Manages multi-turn conversation history in memory."""
    def __init__(self):
        self.history = []

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})

    def add_ai_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})

    def get_formatted_history(self) -> str:
        """Formats dialogue history into a string for LLM prompts."""
        if not self.history:
            return "No previous conversation history."
        lines = []
        for turn in self.history:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()

    def __len__(self):
        return len(self.history)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX_PATH = os.path.join(BASE_DIR, "data", "index", "multilib_faiss.index")
DEFAULT_METADATA_PATH = os.path.join(BASE_DIR, "data", "processed", "multilib_metadata.json")


def initialize_rag_pipeline(
    index_path: str = DEFAULT_INDEX_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model_name: str = "gemini-flash-lite-latest"
) -> dict:
    """
    Initializes and loads all RAG pipeline assets for the Python/DS/ML/DL knowledge base:
    1. Reads FAISS vector index (multilib_faiss.index)
    2. Loads chunk metadata (multilib_metadata.json)
    3. Loads SentenceTransformer embedding model
    4. Connects to Google Gemini API securely from environment
    5. Compiles LangChain LCEL contextualization and conversational chains
    """
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Gemini API key. Please ensure GOOGLE_API_KEY is configured in your .env file."
        )

    # Resolve relative paths against BASE_DIR if needed
    if not os.path.isabs(index_path) and not os.path.exists(index_path):
        alt_index = os.path.join(BASE_DIR, index_path)
        if os.path.exists(alt_index):
            index_path = alt_index

    if not os.path.isabs(metadata_path) and not os.path.exists(metadata_path):
        alt_meta = os.path.join(BASE_DIR, metadata_path)
        if os.path.exists(alt_meta):
            metadata_path = alt_meta

    # 1. Load FAISS index
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at: {index_path}")
    faiss_index = faiss.read_index(index_path)

    # 2. Load Chunk Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Chunk metadata not found at: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)

    # 3. Load Embedding Model
    embed_model = SentenceTransformer(embed_model_name)

    # 4. Instantiate Google Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=llm_model_name,
        temperature=0.2,
        google_api_key=api_key
    )

    # 5. Build Query Contextualization Chain
    contextualize_system_prompt = (
        "Given a conversation history and a user's follow-up question, rewrite the question into a "
        "clear, standalone search query that can be understood without the conversation history. "
        "Do NOT answer the question. Only return the reformulated search query. "
        "If the question is already clear and standalone, return it as-is."
    )
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        ("human", "Conversation History:\n{history}\n\nLatest Question: {question}\n\nStandalone Search Query:")
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    # 6. Build Conversational Grounded RAG Chain
    conv_system_prompt = """You are DocuMind, an expert AI Technical Documentation Assistant for Python, Data Science, Machine Learning, and Deep Learning (including Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, and TensorFlow).
Answer the user's question using the conversation history and ONLY the provided documentation context below.

CRITICAL INSTRUCTIONS:
1. Grounding: Rely strictly on the information in the provided context. Do NOT use outside knowledge or hallucinate APIs, functions, formulas, or parameters.
2. Conceptual & Code Explanations: Provide clear, accurate technical explanations and working Python code examples when supported by the context.
3. Conversational Continuity: Use conversation history to understand follow-up questions and context, while ensuring all facts remain grounded in the documentation context.
4. Insufficient Information & Abstention: If the provided documentation context does not contain enough information to answer the question accurately, respond with:
   "Based on the provided documentation, there is insufficient information to answer this question."
   Never invent unsupported syntax, functions, or theoretical claims.
5. Citations: At the end of your answer, list the exact source documentation titles, library names, and URLs referenced.

DOCUMENTATION CONTEXT:
{context}
"""
    conv_prompt = ChatPromptTemplate.from_messages([
        ("system", conv_system_prompt),
        ("human", "CONVERSATION HISTORY:\n{history}\n\nCURRENT QUESTION:\n{question}")
    ])
    conv_rag_chain = conv_prompt | llm | StrOutputParser()

    # Available libraries in the knowledge base
    available_libraries = sorted(list({c.get("library", "General") for c in chunk_metadata}))

    return {
        "faiss_index": faiss_index,
        "chunk_metadata": chunk_metadata,
        "embed_model": embed_model,
        "llm": llm,
        "llm_model_name": llm_model_name,
        "contextualize_chain": contextualize_chain,
        "conv_rag_chain": conv_rag_chain,
        "available_libraries": available_libraries
    }


def retrieve_context(
    query: str,
    pipeline: dict,
    top_k: int = 4,
    library: str = None
) -> list:
    """
    Retrieves top-k relevant documentation chunks from FAISS for a given search query,
    with optional library-level filtering.
    """
    embed_model = pipeline["embed_model"]
    faiss_index = pipeline["faiss_index"]
    chunk_metadata = pipeline["chunk_metadata"]

    query_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    # If library filter is applied, search a wider candidate pool and filter
    fetch_k = top_k * 4 if (library and library != "All Libraries") else top_k
    fetch_k = min(fetch_k, faiss_index.ntotal)

    scores, indices = faiss_index.search(query_vec, fetch_k)

    retrieved = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx == -1:
            continue
        chunk = chunk_metadata[idx]

        # Apply library filter if requested
        if library and library != "All Libraries":
            if chunk.get("library", "").lower() != library.lower():
                continue

        retrieved.append({
            "rank": len(retrieved) + 1,
            "score": float(score),
            "library": chunk.get("library", "Unknown"),
            "topic": chunk.get("topic", "General"),
            "doc_title": chunk.get("doc_title", chunk.get("title", "Documentation")),
            "source_url": chunk["source_url"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"]
        })

        if len(retrieved) >= top_k:
            break

    return retrieved


def format_context_for_prompt(chunks: list) -> str:
    """Formats retrieved chunks with headers, library tags, and metadata for the LLM prompt."""
    if not chunks:
        return "No relevant documentation chunks retrieved."

    formatted_blocks = []
    for i, c in enumerate(chunks, 1):
        block = (
            f"[Source {i}]: [{c['library']}] {c['doc_title']} | Topic: {c['topic']}\n"
            f"URL: {c['source_url']}\n"
            f"Content:\n{c['text']}"
        )
        formatted_blocks.append(block)
    return "\n\n" + ("-" * 60) + "\n\n".join(formatted_blocks)


def contextualize_user_query(question: str, history_str: str, pipeline: dict) -> str:
    """Reformulates follow-up queries using conversation history if present."""
    if not history_str or history_str == "No previous conversation history.":
        return question
    chain = pipeline["contextualize_chain"]
    standalone = str(chain.invoke({
        "history": history_str,
        "question": question
    })).strip()
    return standalone


def answer_question(
    question: str,
    conv_mgr: ConversationManager,
    pipeline: dict,
    top_k: int = 4,
    library: str = None
) -> dict:
    """
    Executes a complete conversational RAG turn:
    1. Contextualizes the query based on prior conversation
    2. Retrieves top-k documentation chunks from FAISS (with optional library filter)
    3. Invokes the conversational RAG chain
    4. Extracts source citations
    5. Appends the turn to the conversation history
    """
    history_str = conv_mgr.get_formatted_history()

    # 1. Contextualize query if history exists
    search_query = contextualize_user_query(question, history_str, pipeline)

    # 2. Retrieve documentation
    retrieved_chunks = retrieve_context(search_query, pipeline, top_k=top_k, library=library)
    context_text = format_context_for_prompt(retrieved_chunks)

    # 3. Generate grounded answer
    conv_rag_chain = pipeline["conv_rag_chain"]
    raw_answer = conv_rag_chain.invoke({
        "context": context_text,
        "history": history_str,
        "question": question
    })
    answer_text = str(raw_answer).strip()

    # 4. Extract unique source citations
    unique_sources = []
    seen_urls = set()
    for c in retrieved_chunks:
        if c["source_url"] not in seen_urls:
            unique_sources.append({
                "library": c["library"],
                "title": c["doc_title"],
                "topic": c["topic"],
                "url": c["source_url"]
            })
            seen_urls.add(c["source_url"])

    # 5. Append turn to conversation manager
    conv_mgr.add_user_message(question)
    conv_mgr.add_ai_message(answer_text)

    return {
        "question": question,
        "search_query": search_query,
        "answer": answer_text,
        "sources": unique_sources,
        "chunks": retrieved_chunks
    }
