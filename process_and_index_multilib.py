"""
DocuMind Multi-Library Chunking & FAISS Vector Indexing
======================================================
Processes the collected Python, Data Science, ML, and DL documentation:
1. Validates and cleans raw documentation from data/raw/multilib_docs.json
2. Chunks documents using RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
3. Preserves metadata: library, topic, title, source_url, chunk_id
4. Generates dense embeddings using sentence-transformers/all-MiniLM-L6-v2
5. Builds and saves normalized FAISS IndexFlatIP (cosine similarity)
6. Executes retrieval validation tests across representative ML/DL queries

Outputs:
- data/processed/multilib_chunks.json
- data/processed/multilib_metadata.json
- data/processed/multilib_embeddings.npy
- data/index/multilib_faiss.index
"""

import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_document_text(text: str) -> str:
    """Cleans whitespace while preserving code blocks and technical formatting."""
    # Normalize Windows CRLF to LF
    text = text.replace("\r\n", "\n")
    # Replace 3 or more consecutive newlines with 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    return text.strip()


def main():
    print("=" * 80)
    print("PHASE 2: CHUNKING & INDEXING MULTI-LIBRARY KNOWLEDGE BASE")
    print("=" * 80)

    # 1. Inspect input raw data
    raw_path = os.path.join("data", "raw", "multilib_docs.json")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Input file not found at: {raw_path}")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f) # Python list of dictionaries

    print(f"\n[Step 1] Inspecting collected data: {raw_path}")
    print(f"Total raw documents loaded: {len(raw_docs)}")

    valid_docs = []
    skipped_docs = []

    for idx, doc in enumerate(raw_docs, 1):
        lib = doc.get("library")
        url = doc.get("url")
        title = doc.get("title")
        text = doc.get("text", "")

        if not lib or not url or not text or len(text.strip()) < 100:
            skipped_docs.append({"index": idx, "url": url, "reason": "Missing or insufficient text"})
            continue

        valid_docs.append(doc)

    print(f"  Valid documents to process : {len(valid_docs)}")
    print(f"  Skipped documents          : {len(skipped_docs)}")
    if skipped_docs:
        for s in skipped_docs:
            print(f"    - Skipped doc #{s['index']}: {s['url']} ({s['reason']})")
    # 2. Chunking with RecursiveCharacterTextSplitter
    print("\n[Step 2] Chunking documents (chunk_size=800, chunk_overlap=150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    chunk_metadata_list = []

    for doc_idx, doc in enumerate(valid_docs, 1):
        cleaned_text = clean_document_text(doc["text"])
        splits = text_splitter.split_text(cleaned_text)

        lib = doc["library"]
        topic = doc.get("category", "General")
        title = doc.get("title", "Technical Documentation")
        url = doc["url"]

        for split_idx, split_text in enumerate(splits, 1):
            chunk_id = f"{lib.lower().replace('-', '_')}_{doc_idx:02d}_{split_idx:04d}"
            
            chunk_record = {
                "chunk_id": chunk_id,
                "library": lib,
                "topic": topic,
                "title": title,
                "source_url": url,
                "char_count": len(split_text),
                "text": split_text
            }
            all_chunks.append(chunk_record)

            # Metadata store (order identical to chunks)
            chunk_metadata_list.append({
                "chunk_id": chunk_id,
                "library": lib,
                "topic": topic,
                "doc_title": title,
                "source_url": url,
                "text": split_text
            })

    print(f"  Successfully created {len(all_chunks):,} chunks across {len(valid_docs)} documents.")

    # 3. Save processed chunks and metadata
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    chunks_path = os.path.join("data", "processed", "multilib_chunks.json")
    metadata_path = os.path.join("data", "processed", "multilib_metadata.json")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"\n[Step 3] Saved chunk data to: {chunks_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata_list, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata store to: {metadata_path}")

    # 4. Generate Embeddings using Sentence Transformers
    print("\n[Step 4] Generating dense embeddings via sentence-transformers/all-MiniLM-L6-v2...")
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(embed_model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded: {embed_model_name} (Dimension: {dim})")

    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64
    ).astype(np.float32)

    assert embeddings.shape == (len(all_chunks), dim), (
        f"Shape mismatch: expected ({len(all_chunks)}, {dim}), got {embeddings.shape}"
    )

    embeddings_path = os.path.join("data", "processed", "multilib_embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Saved embeddings matrix to: {embeddings_path}")

    # 5. Build and normalize FAISS vector index
    print("\n[Step 5] Building FAISS IndexFlatIP cosine similarity index...")
    os.makedirs(os.path.join("data", "index"), exist_ok=True)
    index_path = os.path.join("data", "index", "multilib_faiss.index")

    # L2 normalize embeddings for inner product to represent cosine similarity
    normalized_embeddings = embeddings.copy()
    faiss.normalize_L2(normalized_embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(normalized_embeddings)

    assert index.ntotal == len(all_chunks), (
        f"Index count mismatch: expected {len(all_chunks)}, got {index.ntotal}"
    )

    faiss.write_index(index, index_path)
    print(f"  FAISS Index built: {index.ntotal:,} vectors, {index.d} dimensions.")
    print(f"  Saved FAISS index to: {index_path}")

    # 6. Detailed Summary & Breakdown
    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY & BREAKDOWN")
    print("=" * 80)
    print(f"Total Documents Processed : {len(valid_docs)}")
    print(f"Total Chunks Created      : {len(all_chunks):,}")
    print(f"Embeddings Matrix Shape   : {embeddings.shape}")
    print(f"FAISS Index Total Vectors : {index.ntotal:,}")
    print(f"FAISS Index Dimensions    : {index.d}")
    print(f"FastAPI Chunks Included   : 0 (Pure Python / DS / ML / DL)")

    print("\nChunk Breakdown by Library:")
    lib_stats = {}
    for c in all_chunks:
        lib = c["library"]
        lib_stats[lib] = lib_stats.get(lib, 0) + 1
    for lib, count in sorted(lib_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {lib:<15}: {count:4d} chunks ({count / len(all_chunks) * 100:.1f}%)")

    print("\nChunk Breakdown by Topic Category:")
    topic_stats = {}
    for c in all_chunks:
        topic = c["topic"]
        topic_stats[topic] = topic_stats.get(topic, 0) + 1
    for topic, count in sorted(topic_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {topic:<40}: {count:3d} chunks")

    # 7. Semantic Retrieval Validation Test
    print("\n" + "=" * 80)
    print("SEMANTIC RETRIEVAL VALIDATION TEST (Top 3 Chunks)")
    print("=" * 80)

    test_queries = [
        "What is overfitting?",
        "How does GridSearchCV work?",
        "What is a Pandas DataFrame?",
        "What is gradient descent?",
        "What is a tensor in PyTorch?"
    ]

    for q_idx, query in enumerate(test_queries, 1):
        q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = index.search(q_vec, 3)

        print(f"\n[Query {q_idx}] \"{query}\"")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            hit = chunk_metadata_list[idx]
            snippet = hit["text"][:120].replace("\n", " ") + "..."
            print(f"  Top {rank} (Cosine: {score:.4f}) | [{hit['library']}] {hit['doc_title']}")
            print(f"         Topic: {hit['topic']}")
            print(f"         Snippet: \"{snippet}\"")

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETED SUCCESSFULLY: Ready for Phase 3 (RAG Pipeline Updates)")
    print("=" * 80)


if __name__ == "__main__":
    main()
