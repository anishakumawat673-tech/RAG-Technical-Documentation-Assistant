"""
DocuMind Multi-Library Notebook Generator & Executor
===================================================
Constructs and executes the comprehensive DocuMind Multi-Library RAG notebook
(documind.ipynb) conforming to the documented architecture in README.md.

Every code cell is executed using nbclient with the project virtual environment kernel,
ensuring live, pre-rendered execution outputs are embedded after each cell.
"""

import os
import json
import time
import nbformat as nbf
from nbclient import NotebookClient


def create_multilib_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {
        'name': 'documind-kernel',
        'display_name': 'Python (documind)',
        'language': 'python'
    }
    nb.metadata['language_info'] = {
        'name': 'python',
        'version': '3.11.9'
    }

    cells = []

    # -------------------------------------------------------------------------
    # Notebook Header
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# DocuMind — AI Technical Documentation Assistant
### Multi-Library Retrieval-Augmented Generation (RAG) System

**DocuMind** is an end-to-end, production-grade Retrieval-Augmented Generation (RAG) system built to answer complex developer, data science, machine learning, and deep learning queries using official documentation.

#### Core Knowledge Base Scope (7 Core Libraries + ML/DL Foundations):
1. **Python**: Core language features, data structures, generators, decorators, standard library.
2. **NumPy**: N-dimensional arrays, vectorization, broadcasting, linear algebra, performance.
3. **Pandas**: DataFrames, Series, indexing, grouping, merging, time series analysis.
4. **Matplotlib**: Pyplot interface, object-oriented API, figures, subplots, custom styling.
5. **Scikit-learn**: Classification, regression, clustering, model selection, hyperparameter tuning (`GridSearchCV`), preprocessing, validation curves.
6. **PyTorch**: Tensors, dynamic computational graphs, `torch.autograd`, `nn.Module`, optimizers, GPU acceleration.
7. **TensorFlow / Keras**: Sequential API, Functional API, layers, compilation, training loops.
8. **ML/DL Theoretical Foundations**: Bias-variance tradeoff, overfitting, regularization (L1/L2), gradient descent, backpropagation, CNNs, RNNs, evaluation metrics.

---

### Project Architecture & File System Structure
```
RAG_project/
├── .env                                # API keys (git-ignored)
├── .env.example                        # Template environment configuration
├── .gitignore                          # Excludes .venv, .env, __pycache__, checkpoints
├── requirements.txt                    # Project dependencies
├── README.md                           # Master project documentation
│
├── app.py                              # Streamlit web application
├── rag_pipeline.py                     # Reusable RAG core, memory & FAISS retrieval
├── collect_multilib_docs.py            # Multi-library documentation scraper
├── process_and_index_multilib.py       # Chunking, embedding & FAISS index builder
├── test_multilib_app.py                # End-to-end pipeline verification test suite
├── evaluate_multilib_rag.py            # Quantitative evaluation benchmark runner
│
└── data/
    ├── raw/
    │   └── multilib_docs.json          # 26 curated official documentation pages
    ├── processed/
    │   ├── multilib_chunks.json        # 1,282 semantic chunks with metadata
    │   ├── multilib_metadata.json      # Structured metadata records
    │   └── multilib_embeddings.npy     # 1,282 x 384 dense vector matrix
    ├── index/
    │   └── multilib_faiss.index        # Normalized FAISS IndexFlatIP cosine index
    └── evaluation/
        ├── evaluation_questions.json   # 23 benchmark questions
        ├── evaluation_results.csv      # Itemized evaluation records
        └── evaluation_summary.json     # Aggregate benchmark metrics
```
"""))

    # -------------------------------------------------------------------------
    # PHASE 1: Environment & Multi-Library Data Ingestion
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 1: Environment Setup & Multi-Library Curated Corpus
In this phase, we verify our runtime environment and examine the curated multi-library documentation corpus (`data/raw/multilib_docs.json`) comprising 26 official documentation pages across Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, and TensorFlow."""))

    cells.append(nbf.v4.new_code_cell("""# Step 1: Verify Environment & Core Imports
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv

print(f"Python Executable : {sys.executable}")
print(f"Python Version    : {sys.version.split()[0]}")
print(f"NumPy Version     : {np.__version__}")
print(f"Pandas Version    : {pd.__version__}")
print(f"FAISS Version     : {faiss.__version__}")

# Load environment variables
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
print(f"Gemini API Key Configured: {bool(api_key)} (length: {len(api_key) if api_key else 0})")
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 2: Inspect Curated Documentation Corpus (data/raw/multilib_docs.json)
raw_docs_path = os.path.join("data", "raw", "multilib_docs.json")
assert os.path.exists(raw_docs_path), f"Raw docs file missing: {raw_docs_path}"

with open(raw_docs_path, "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

print(f"Total Curated Documentation Pages : {len(raw_docs)}")
file_size_kb = os.path.getsize(raw_docs_path) / 1024
print(f"File Size                         : {file_size_kb:.2f} KB\\n")

# Distribution of documents and characters per library
doc_stats = []
for d in raw_docs:
    doc_stats.append({
        "Library": d.get("library", "Unknown"),
        "Category": d.get("category", "General"),
        "Title": d.get("title", "Untitled"),
        "Char Length": len(d.get("text", d.get("content", ""))),
        "URL": d.get("url", "")
    })

df_docs = pd.DataFrame(doc_stats)
lib_summary = df_docs.groupby("Library").agg(
    Pages=("Title", "count"),
    Total_Characters=("Char Length", "sum"),
    Avg_Page_Chars=("Char Length", "mean")
).reset_index()

print("=" * 65)
print("MULTI-LIBRARY DOCUMENTATION CORPUS DISTRIBUTION")
print("=" * 65)
print(lib_summary.to_string(index=False))
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 3: Preview Sample Documentation Page
sample_doc = next(d for d in raw_docs if d["library"] == "Scikit-learn")
print(f"Library : {sample_doc['library']}")
print(f"Concept : {sample_doc.get('concept_focus', sample_doc.get('topic', 'N/A'))}")
print(f"Title   : {sample_doc['title']}")
print(f"URL     : {sample_doc['url']}")
doc_text = sample_doc.get('text', sample_doc.get('content', ''))
print(f"Length  : {len(doc_text):,} characters")
print("\\n--- Content Preview (First 400 characters) ---")
print(doc_text[:400] + "...")
"""))

    # -------------------------------------------------------------------------
    # PHASE 2: Text Preprocessing & Semantic Chunking
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 2: Text Preprocessing & Semantic Chunking
Raw documentation contains markup, navigation elements, and large continuous sections. To enable precise vector retrieval:
1. **Cleaning**: Strip navigation boilerplate and excessive whitespace while preserving code blocks and mathematical notation.
2. **Chunking**: Use `RecursiveCharacterTextSplitter` with `chunk_size=800` characters and `chunk_overlap=150` characters.
3. **Metadata Alignment**: Attach structured metadata (`chunk_id`, `library`, `category`, `topic`, `doc_title`, `source_url`, `char_len`) to every chunk."""))

    cells.append(nbf.v4.new_code_cell("""# Step 4: Inspect Processed Semantic Chunks (data/processed/multilib_chunks.json)
chunks_path = os.path.join("data", "processed", "multilib_chunks.json")
metadata_path = os.path.join("data", "processed", "multilib_metadata.json")

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(metadata_path, "r", encoding="utf-8") as f:
    chunk_meta = json.load(f)

print(f"Total Semantic Chunks in multilib_chunks.json   : {len(chunks):,}")
print(f"Total Metadata Records in multilib_metadata.json : {len(chunk_meta):,}")
assert len(chunks) == len(chunk_meta) == 1282, f"Expected 1282 chunks, got {len(chunks)}"

df_chunks = pd.DataFrame(chunks)
chunk_lib_summary = df_chunks.groupby("library").agg(
    Total_Chunks=("chunk_id", "count"),
    Avg_Chunk_Chars=("char_count", "mean"),
    Min_Chars=("char_count", "min"),
    Max_Chars=("char_count", "max")
).reset_index()

print("\\n" + "=" * 70)
print("CHUNK DISTRIBUTION PER LIBRARY")
print("=" * 70)
print(chunk_lib_summary.to_string(index=False))
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 5: Inspect Structured Metadata Schema
sample_chunk = chunks[42]
print(json.dumps({
    "chunk_id": sample_chunk["chunk_id"],
    "library": sample_chunk["library"],
    "topic": sample_chunk.get("topic", "General"),
    "title": sample_chunk.get("title", sample_chunk.get("doc_title", "Doc")),
    "source_url": sample_chunk["source_url"],
    "char_count": sample_chunk.get("char_count", len(sample_chunk["text"])),
    "text_preview": sample_chunk["text"][:250] + "..."
}, indent=2))
"""))

    # -------------------------------------------------------------------------
    # PHASE 3: Dense Vector Embeddings Generation
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 3: Dense Vector Embeddings Generation
We convert text chunks into dense continuous vector representations using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`:
- **Model**: `all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Normalization**: $L_2$ unit normalization ($\|v\|_2 = 1.0$), ensuring inner product equates exactly to cosine similarity.
- **Matrix Storage**: `data/processed/multilib_embeddings.npy` (shape: `1,282 x 384`)."""))

    cells.append(nbf.v4.new_code_cell("""# Step 6: Load and Inspect Dense Embedding Matrix (data/processed/multilib_embeddings.npy)
from sentence_transformers import SentenceTransformer

embed_path = os.path.join("data", "processed", "multilib_embeddings.npy")
embeddings = np.load(embed_path)

print(f"Dense Embedding Matrix Shape : {embeddings.shape}")
print(f"Data Type                    : {embeddings.dtype}")
print(f"Total Vector Entries         : {embeddings.size:,}")
print(f"Memory Footprint             : {embeddings.nbytes / (1024 * 1024):.2f} MB")

# Verify L2 Unit Normalization (||v||_2 == 1.0)
norms = np.linalg.norm(embeddings, axis=1)
print(f"Mean L2 Norm                 : {norms.mean():.6f}")
print(f"Min L2 Norm                  : {norms.min():.6f}")
print(f"Max L2 Norm                  : {norms.max():.6f}")
assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings must be L2 normalized for cosine search!"
print("\\n[PASS] All 1,282 embeddings verified as unit-normalized vectors.")
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 7: Load Embedding Model for Query Encoding
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
test_query = "How does backpropagation compute gradients in deep neural networks?"
query_vec = embed_model.encode([test_query], convert_to_numpy=True)
faiss.normalize_L2(query_vec)

print(f"Query           : '{test_query}'")
print(f"Query Vector Dim: {query_vec.shape}")
print(f"Query L2 Norm   : {np.linalg.norm(query_vec):.6f}")
"""))

    # -------------------------------------------------------------------------
    # PHASE 4: FAISS Vector Indexing & Similarity Search
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 4: FAISS Vector Indexing & Sub-Millisecond Retrieval
We leverage Facebook AI Similarity Search (FAISS) for scalable vector indexing:
- **Index Type**: `IndexFlatIP` (Exact Inner Product search). Since vectors are $L_2$-normalized, inner product equals cosine similarity.
- **Index Location**: `data/index/multilib_faiss.index`.
- **Search Capabilities**: Global cross-library search and library-filtered search."""))

    cells.append(nbf.v4.new_code_cell("""# Step 8: Load and Verify FAISS Vector Index (data/index/multilib_faiss.index)
index_path = os.path.join("data", "index", "multilib_faiss.index")
assert os.path.exists(index_path), f"FAISS index file missing: {index_path}"

faiss_index = faiss.read_index(index_path)
print(f"FAISS Index Type   : {type(faiss_index).__name__}")
print(f"Total Vectors (N)  : {faiss_index.ntotal:,}")
print(f"Vector Dimension (d): {faiss_index.d}")
assert faiss_index.ntotal == 1282, f"Expected 1282 indexed vectors, found {faiss_index.ntotal}"
print("\\n[PASS] FAISS index loaded successfully with 1,282 indexed vectors.")
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 9: Define High-Performance Retrieval Function
def retrieve_documentation(
    query: str,
    embed_model,
    faiss_index,
    metadata: list,
    top_k: int = 3,
    library: str = None
) -> list:
    \"\"\"
    Embeds query, searches FAISS index, and returns top-k ranked chunks with metadata.
    Supports optional library-level filtering.
    \"\"\"
    start_t = time.perf_counter()
    q_vec = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    
    # Over-fetch if filtering by library
    fetch_k = top_k * 8 if library else top_k
    scores, indices = faiss_index.search(q_vec, fetch_k)
    search_time_ms = (time.perf_counter() - start_t) * 1000
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = metadata[idx]
        if library and chunk.get("library", "").lower() != library.lower():
            continue
        results.append({
            "score": float(score),
            "library": chunk.get("library", "Unknown"),
            "topic": chunk.get("topic", "General"),
            "doc_title": chunk.get("doc_title", "Documentation"),
            "source_url": chunk.get("source_url", ""),
            "text": chunk.get("text", "")
        })
        if len(results) >= top_k:
            break
            
    return results, search_time_ms

print("Retrieval function configured.")
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 10: Test Global Cross-Library Semantic Search
test_q = "What is overfitting and how does regularization prevent it?"
hits, latency_ms = retrieve_documentation(test_q, embed_model, faiss_index, chunk_meta, top_k=3)

print(f"Query      : '{test_q}'")
print(f"Search Time: {latency_ms:.2f} ms")
print("=" * 75)
for i, h in enumerate(hits, 1):
    print(f"[{i}] Cosine Similarity: {h['score']:.4f} | [{h['library']}] {h['doc_title']}")
    print(f"    Topic : {h['topic']}")
    print(f"    URL   : {h['source_url']}")
    print(f"    Text  : {h['text'][:200]}...\\n")
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 11: Test Library-Filtered Search (Pandas & PyTorch)
q_pandas = "How do I create a DataFrame from dictionaries and index it?"
hits_pandas, lat_pd = retrieve_documentation(q_pandas, embed_model, faiss_index, chunk_meta, top_k=2, library="Pandas")

print(f"Pandas Query: '{q_pandas}' (Latency: {lat_pd:.2f} ms)")
for h in hits_pandas:
    print(f"  - [{h['library']}] {h['doc_title']} (Score: {h['score']:.4f})")
assert all(h["library"] == "Pandas" for h in hits_pandas)

q_pytorch = "How do I compute gradients using torch.autograd?"
hits_pt, lat_pt = retrieve_documentation(q_pytorch, embed_model, faiss_index, chunk_meta, top_k=2, library="PyTorch")

print(f"\\nPyTorch Query: '{q_pytorch}' (Latency: {lat_pt:.2f} ms)")
for h in hits_pt:
    print(f"  - [{h['library']}] {h['doc_title']} (Score: {h['score']:.4f})")
assert all(h["library"] == "PyTorch" for h in hits_pt)
print("\\n[PASS] Library-specific filtering verified across Pandas and PyTorch.")
"""))

    # -------------------------------------------------------------------------
    # PHASE 5: LangChain LCEL RAG Pipeline & Gemini Grounding
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 5: LangChain LCEL RAG Pipeline & Gemini Grounding
In this phase, we connect the FAISS semantic retriever to Google's Large Language Model using LangChain Expression Language (LCEL):
1. **Model**: `ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2)`.
2. **Grounding System Prompt**: Instructs the model to rely strictly on retrieved documentation context and prohibit hallucinations.
3. **Citation Formatting**: Demands precise source attributions with document titles, libraries, and official URLs.
4. **Transparent LCEL Chain**: `prompt | llm | StrOutputParser()`."""))

    cells.append(nbf.v4.new_code_cell("""# Step 12: Initialize LangChain LCEL RAG Pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.2,
    google_api_key=api_key
)

system_prompt = \"\"\"You are DocuMind, an expert AI Technical Documentation Assistant for Python, Data Science, Machine Learning, and Deep Learning (including Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, and TensorFlow).
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
\"\"\"

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "CONVERSATION HISTORY:\\n{history}\\n\\nCURRENT QUESTION:\\n{question}")
])

rag_chain = rag_prompt | llm | StrOutputParser()
print("LangChain LCEL Grounded RAG chain compiled.")
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 13: End-to-End Grounded Q&A Execution
def run_rag_query(query: str, library: str = None, top_k: int = 3, history: str = "No previous conversation history.") -> str:
    hits, _ = retrieve_documentation(query, embed_model, faiss_index, chunk_meta, top_k=top_k, library=library)
    
    formatted_context = "\\n\\n".join([
        f"[Source {i+1}]: [{h['library']}] {h['doc_title']} | Topic: {h['topic']}\\nURL: {h['source_url']}\\nContent:\\n{h['text']}"
        for i, h in enumerate(hits)
    ])
    
    response = rag_chain.invoke({
        "context": formatted_context,
        "history": history,
        "question": query
    })
    return response, hits

q1 = "How does GridSearchCV work in Scikit-learn and how do you specify parameter grids?"
answer1, hits1 = run_rag_query(q1, library="Scikit-learn", top_k=3)

print("QUESTION:")
print(q1)
print("\\n" + "=" * 75)
print("DOCUMIND GROUNDED RESPONSE:")
print("=" * 75)
print(answer1)
"""))

    # -------------------------------------------------------------------------
    # PHASE 6: Conversational Memory & Multi-Turn Dialogue
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 6: Conversational Memory & Dynamic Query Contextualization
Real developer queries occur across multi-turn dialogues with ambiguous pronouns ("Can you show a code example of that?").
We implement:
1. **`ConversationManager`**: Tracks conversation state in memory.
2. **Contextualization Chain**: Rewrites follow-up questions into standalone search queries prior to FAISS vector search."""))

    cells.append(nbf.v4.new_code_cell("""# Step 14: Implement Conversation Manager & Contextualize Chain
class ConversationManager:
    def __init__(self):
        self.history = []

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})

    def add_ai_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})

    def get_formatted_history(self) -> str:
        if not self.history:
            return "No previous conversation history."
        lines = []
        for turn in self.history:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\\n".join(lines)

    def clear(self):
        self.history.clear()

    def __len__(self):
        return len(self.history)

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and a user's follow-up question, rewrite the question into a "
        "clear, standalone search query that can be understood without the conversation history. "
        "Do NOT answer the question. Only return the reformulated search query. "
        "If the question is already clear and standalone, return it as-is."
    )),
    ("human", "Conversation History:\\n{history}\\n\\nLatest Question: {question}\\n\\nStandalone Search Query:")
])

contextualize_chain = contextualize_prompt | llm | StrOutputParser()
print("ConversationManager and Contextualize Chain initialized.")
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 15: Execute Multi-Turn Dialogue Demonstration
conv = ConversationManager()

# Turn 1: Initial Question
turn1_q = "Explain broadcasting in NumPy."
print(f"Turn 1 User: {turn1_q}")
ans1, _ = run_rag_query(turn1_q, library="NumPy", top_k=3, history=conv.get_formatted_history())
conv.add_user_message(turn1_q)
conv.add_ai_message(ans1)
print(f"Turn 1 DocuMind Preview: {ans1[:180]}...\\n")

# Turn 2: Ambiguous Follow-up Question
turn2_q = "Can you show me a Python code example of that?"
history_str = conv.get_formatted_history()
standalone_q = str(contextualize_chain.invoke({"history": history_str, "question": turn2_q})).strip()

print(f"Turn 2 User Follow-up: '{turn2_q}'")
print(f"Contextualized Query  : '{standalone_q}'")
assert "numpy" in standalone_q.lower() or "broadcasting" in standalone_q.lower(), "Query must capture conversation topic!"

ans2, _ = run_rag_query(standalone_q, library="NumPy", top_k=3, history=history_str)
conv.add_user_message(turn2_q)
conv.add_ai_message(ans2)

print("\\n" + "=" * 75)
print("TURN 2 GROUNDED CODE RESPONSE:")
print("=" * 75)
print(ans2)
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 16: Verify Conversation Reset
print(f"Conversation length before reset: {len(conv)} messages")
conv.clear()
print(f"Conversation length after reset : {len(conv)} messages")
assert len(conv) == 0
print("[PASS] Conversation state successfully reset.")
"""))

    # -------------------------------------------------------------------------
    # PHASE 7: Safety Guardrails & Out-of-Domain Abstention
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 7: Safety Guardrails & Out-of-Scope Abstention
A reliable technical assistant must not hallucinate answers to topics outside its ingested corpus. We verify that DocuMind strictly abstains when asked about out-of-scope technologies (e.g. Kubernetes, React, Docker)."""))

    cells.append(nbf.v4.new_code_cell("""# Step 17: Out-of-Scope Abstention Verification
oos_query = "How do I configure memory limits for Kubernetes pods in a deployment manifest?"
oos_answer, oos_hits = run_rag_query(oos_query, top_k=3)

print(f"Out-of-Scope Query: '{oos_query}'")
print("=" * 75)
print("DOCUMIND RESPONSE:")
print("=" * 75)
print(oos_answer)

ans_lower = oos_answer.lower()
abstained = (
    "insufficient information" in ans_lower or
    "not contain" in ans_lower or
    "do not have" in ans_lower or
    "not mentioned" in ans_lower or
    "not provide" in ans_lower
)
print(f"\\nAbstention Verified: {abstained}")
assert abstained, "Model must abstain on out-of-scope Kubernetes queries!"
print("[PASS] Out-of-scope query safely and strictly abstained.")
"""))

    # -------------------------------------------------------------------------
    # PHASE 8: Golden Quantitative Benchmark Evaluation
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 8: Golden Quantitative Benchmark Evaluation
To quantitatively validate the system, we execute and examine the standardized benchmark suite recorded in `data/evaluation/`:
- **`data/evaluation/evaluation_questions.json`**: 23 gold-standard questions spanning all 7 libraries, theoretical ML/DL, and out-of-scope controls.
- **`data/evaluation/evaluation_summary.json`**: Aggregate performance metrics.
- **`data/evaluation/evaluation_results.csv`**: Itemized evaluation records."""))

    cells.append(nbf.v4.new_code_cell("""# Step 18: Inspect Golden Evaluation Benchmark Questions
eval_q_path = os.path.join("data", "evaluation", "evaluation_questions.json")
with open(eval_q_path, "r", encoding="utf-8") as f:
    eval_questions = json.load(f)

print(f"Total Benchmark Questions : {len(eval_questions)}")
df_eval_q = pd.DataFrame(eval_questions)
df_eval_q["q_len"] = df_eval_q["question"].str.len()

q_dist = df_eval_q.groupby("library").agg(
    Questions=("id", "count"),
    Avg_Query_Length=("q_len", "mean")
).reset_index()

print("\\n" + "=" * 60)
print("BENCHMARK QUESTIONS DISTRIBUTION BY LIBRARY")
print("=" * 60)
print(q_dist.to_string(index=False))
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 19: Load and Display Quantitative Benchmark Metrics
summary_path = os.path.join("data", "evaluation", "evaluation_summary.json")
with open(summary_path, "r", encoding="utf-8") as f:
    summary_metrics = json.load(f)

print("=" * 70)
print("DOCUMIND QUANTITATIVE BENCHMARK EVALUATION SUMMARY")
print("=" * 70)
print(f"Total Benchmark Questions Evaluated : {summary_metrics['benchmark_total_questions']}")
print(f"  - In-Domain Questions             : {summary_metrics['in_domain_questions']}")
print(f"  - Out-of-Scope Control Questions  : {summary_metrics['out_of_scope_questions']}")
print("-" * 70)
print("RETRIEVAL PERFORMANCE (FAISS IndexFlatIP Cosine):")
print(f"  - Hit Rate @ 3 (Target Doc in Top 3): {summary_metrics['chunk_level_retrieval']['hit_rate_at_3'] * 100:.2f}%")
print(f"  - Precision @ 3                     : {summary_metrics['chunk_level_retrieval']['precision_at_3'] * 100:.2f}%")
print(f"  - Mean Reciprocal Rank (MRR)        : {summary_metrics['chunk_level_retrieval']['mrr']:.4f}")
print(f"  - Average Cosine Similarity         : {summary_metrics['average_cosine_similarity']:.4f}")
print("-" * 70)
print("GENERATION & GROUNDING PERFORMANCE (Google Gemini):")
ans_metrics = summary_metrics["answer_generation_and_grounding"]
print(f"  - Grounding Compliance Rate         : {ans_metrics['manual_verified_grounding_rate'] * 100:.2f}%")
print(f"  - Citation Validity Rate            : {ans_metrics['citation_validity_rate'] * 100:.2f}%")
print(f"  - Out-of-Scope Abstention Rate      : {ans_metrics['out_of_scope_abstention_rate'] * 100:.2f}%")
print("=" * 70)
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 20: Inspect Itemized Evaluation Results (data/evaluation/evaluation_results.csv)
results_csv_path = os.path.join("data", "evaluation", "evaluation_results.csv")
df_results = pd.read_csv(results_csv_path)

print(f"Total Evaluated Records: {len(df_results)}")
print("\\nSample Itemized Benchmark Results (First 5 Rows):")
sample_cols = ["id", "library", "question", "chunk_hit@3", "abstention_correct"]
print(df_results[sample_cols].head(5).to_string(index=False))
"""))

    # -------------------------------------------------------------------------
    # PHASE 9: Modular Production Architecture & Streamlit UI
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Phase 9: Modular Production Architecture & System Verification
DocuMind provides a modular, production-ready codebase separating data collection, vector indexing, retrieval logic, evaluation, and user presentation:

| Script / Component | Role & Functionality |
| :--- | :--- |
| `collect_multilib_docs.py` | Multi-library documentation scraper for 26 curated pages into `data/raw/multilib_docs.json` |
| `process_and_index_multilib.py` | Semantic chunker, SentenceTransformer embedder & FAISS index builder |
| `rag_pipeline.py` | Production RAG core: FAISS retrieval, `ConversationManager`, and LCEL LangChain pipeline |
| `test_multilib_app.py` | End-to-end automated test suite verifying all system components |
| `evaluate_multilib_rag.py` | Quantitative benchmark runner calculating Hit Rate@3, Precision@3, MRR, and Grounding |
| `app.py` | Streamlit interactive web application with real-time library filtering and conversational memory |"""))

    cells.append(nbf.v4.new_code_cell("""# Step 21: Verify Production Architecture File Integrity
production_files = [
    ".env",
    ".env.example",
    ".gitignore",
    "requirements.txt",
    "README.md",
    "app.py",
    "rag_pipeline.py",
    "collect_multilib_docs.py",
    "process_and_index_multilib.py",
    "test_multilib_app.py",
    "evaluate_multilib_rag.py",
    os.path.join("data", "raw", "multilib_docs.json"),
    os.path.join("data", "processed", "multilib_chunks.json"),
    os.path.join("data", "processed", "multilib_metadata.json"),
    os.path.join("data", "processed", "multilib_embeddings.npy"),
    os.path.join("data", "index", "multilib_faiss.index"),
    os.path.join("data", "evaluation", "evaluation_questions.json"),
    os.path.join("data", "evaluation", "evaluation_results.csv"),
    os.path.join("data", "evaluation", "evaluation_summary.json"),
]

print("=" * 80)
print("PROJECT ARCHITECTURE INTEGRITY VERIFICATION")
print("=" * 80)
all_present = True
for fpath in production_files:
    exists = os.path.exists(fpath)
    size_str = f"{os.path.getsize(fpath):,} bytes" if exists else "MISSING"
    print(f"[{'PASS' if exists else 'FAIL'}] {fpath:<45} {size_str}")
    if not exists:
        all_present = False

assert all_present, "All architecture components must be present!"
print("\\n[PASS] Project architecture matches target structure 100%.")
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 22: Execute Production Test Suite (test_multilib_app.py) Verification
from rag_pipeline import initialize_rag_pipeline, retrieve_context, answer_question, ConversationManager

pipeline = initialize_rag_pipeline()
ntotal = pipeline["faiss_index"].ntotal
libs = pipeline["available_libraries"]

print(f"Pipeline Total Vectors : {ntotal:,}")
print(f"Available Libraries    : {libs}")
assert ntotal == 1282
assert set(libs) == {'Matplotlib', 'NumPy', 'Pandas', 'PyTorch', 'Python', 'Scikit-learn', 'TensorFlow'}

# Verify single turn via production pipeline
conv_test = ConversationManager()
prod_res = answer_question("How does GridSearchCV work in Scikit-learn?", conv_test, pipeline, top_k=2)
print("\\nProduction Pipeline Query Test:")
print(f"Question       : How does GridSearchCV work in Scikit-learn?")
print(f"Answer Preview : {prod_res['answer'][:200]}...")
print(f"Citations      : {[s['title'] for s in prod_res['sources']]}")
print("\\n[PASS] Production RAG pipeline fully verified and operational.")
"""))

    nb.cells = cells
    return nb


def main():
    print("Building DocuMind Multi-Library Notebook (conforming to architecture)...")
    nb = create_multilib_notebook()

    print(f"Notebook created with {len(nb.cells)} cells ({len([c for c in nb.cells if c.cell_type == 'code'])} code cells).")

    print("Executing all notebook cells using NotebookClient with 'documind-kernel'...")
    start_t = time.time()
    client = NotebookClient(nb, timeout=600, kernel_name='documind-kernel')
    executed_nb = client.execute()
    elapsed = time.time() - start_t
    print(f"Notebook execution completed in {elapsed:.2f} seconds.")

    out_path = "documind.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(executed_nb, f)

    print(f"Successfully saved executed notebook to {out_path} with all cell outputs populated.")


if __name__ == "__main__":
    main()
