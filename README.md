# DocuMind — AI Technical Documentation Assistant (Python, Data Science & ML/DL)

> **End-to-end multi-library RAG assistant for Python, Data Science & ML/DL documentation using LangChain (LCEL), FAISS vector search, Gemini, and Streamlit deployment.**

DocuMind is an interview-ready Retrieval-Augmented Generation (RAG) assistant designed to accurately answer technical and conceptual developer questions using official documentation for **Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, and TensorFlow**.

---

## 🚀 Key Features

- **Multi-Library Knowledge Base**: 1,282 semantic chunks indexed across 7 core Python, Data Science, ML, and DL documentation suites.
- **Deep Conceptual & Mathematical Coverage**: Covers classification, regression, overfitting, regularization, bias–variance tradeoff, K-Fold cross-validation, hyperparameter tuning (GridSearchCV), gradient descent, backpropagation, CNNs, RNNs, and model evaluation metrics.
- **High-Accuracy Vector Search**: Dense semantic retrieval powered by `sentence-transformers/all-MiniLM-L6-v2` and an $L_2$-normalized **FAISS `IndexFlatIP`** (cosine similarity).
- **Conversational Memory & Follow-up Support**: Dynamic query contextualization via LangChain LCEL transforms ambiguous follow-up questions into standalone semantic search queries.
- **Factual Grounding & Abstention Guardrails**: Rejects hallucinations by strictly adhering to retrieved context and safely abstaining on out-of-scope inquiries.
- **Source Attributions**: Every response lists official documentation titles, library tags, and clickable documentation URLs.
- **Interactive Streamlit Web UI**: Responsive web chat interface with library filtering, quick prompt chips, and zero-redundancy memory caching (`@st.cache_resource`).

---

## 🛠️ Tech Stack

- **Core**: Python 3.11
- **RAG & Orchestration**: LangChain (LCEL)
- **Embedding Model**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions)
- **Vector Database**: FAISS (`IndexFlatIP`)
- **LLM Generator**: Google Gemini (`gemini-flash-lite-latest`, temperature=0.2)
- **User Interface**: Streamlit
- **Data Collection & Processing**: Requests, BeautifulSoup4, Pandas, NumPy

---

## 💻 Running the Web Application

1. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Ensure API Key is set in `.env`**:
   ```ini
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

3. **Launch the Streamlit app**:
   ```powershell
   streamlit run app.py
   ```

4. Open your browser at **`http://localhost:8501`**.

---

## 📓 Interactive Jupyter Notebook Walkthrough

For an in-depth, step-by-step interactive walkthrough with pre-rendered execution outputs:
- **Notebook File**: [`documind.ipynb`](documind.ipynb)
- **Features**: 9 modular phases, 22 code cells, covering data collection, text chunking, FAISS index creation, LangChain LCEL RAG, conversational memory, guardrail abstention, and benchmark evaluation.
- **Pre-Rendered Outputs**: Code cells contain live, verified outputs (tables, metrics, search results, model answers, and citations) displayed directly after execution.
- **Regenerate / Re-execute**:
  ```powershell
  python build_notebook.py
  ```

---

## 🧪 Running Automated Tests and Evaluation

To test the RAG pipeline end-to-end:
```powershell
python test_multilib_app.py
```

To run the rigorous **45-question document-level evaluation suite**:
```powershell
python evaluate_multilib_rag.py
```

### Benchmark Results (Phase 5 Rigorous Evaluation Audit)

The evaluation suite evaluates retrieval against **exact document-level target URLs** across 45 test queries (38 in-domain across 7 libraries and 7 out-of-scope refusal queries):

#### 1. Retrieval Performance Metrics

| Evaluation Granularity | Metric | Measured Score | Definition / Meaning |
| :--- | :--- | :--- | :--- |
| **Chunk-Level (Top 3 Chunks)** | **Chunk Hit Rate@3** | **94.74%** | Proportion of queries where $\ge 1$ retrieved chunk belongs to target documentation |
| | **Chunk Precision@3** | **86.84%** | Average fraction of the 3 retrieved chunks belonging to target documentation |
| | **Chunk MRR** | **0.9298** | Mean reciprocal rank ($\frac{1}{\text{rank}}$) of the first target chunk |
| **Unique-Document Level** | **Unique Doc Hit Rate@3** | **94.74%** | Proportion of queries retrieving at least one target document in top 3 |
| *(Deduplicated by URL)* | **Unique Doc Precision** | **86.84%** | Relevant unique documents divided by total unique documents retrieved |
| | **Unique Doc Recall@3** | **92.11%** | Target documents retrieved in top 3 divided by total target documents |
| | **Unique Doc MRR** | **0.9342** | Reciprocal rank of the first target document in deduplicated document sequence |
| **Semantic Similarity** | **Mean Cosine Similarity**| **0.6417** | Diagnostic mean cosine similarity of top-3 retrieved candidate chunks |

*Document Concentration Note*: In 73.7% of queries, all 3 top chunks were concentrated in a single authoritative document; in 26.3% of queries, chunks spanned 2 distinct documents.

#### 2. Answer Grounding & Safety Metrics

| Metric | Measured Score | Evaluation Standard |
| :--- | :--- | :--- |
| **Answer Grounding Rate** | **92.11%** (35/38) | Factually grounded answers strictly derived from retrieved context |
| **Abstention on Missing Context** | **7.89%** (3/38) | Correct refusal when top-3 context lacked complete sub-method details |

All evaluation questions, per-query retrieval metrics, and detailed failure diagnostics are persisted in `data/evaluation/evaluation_results.csv` and `data/evaluation/evaluation_summary.json`.
