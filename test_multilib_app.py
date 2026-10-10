"""
Test Suite for DocuMind Multi-Library RAG Pipeline
==================================================
Tests:
1. FAISS Index & Metadata Loading
2. Cross-Library Retrieval & Filtering
3. End-to-End Grounded Q&A with Gemini
4. Conversational Follow-up Reformulation
5. Out-of-Scope Abstention
6. Conversation Reset
"""

import time
from rag_pipeline import (
    initialize_rag_pipeline,
    retrieve_context,
    ConversationManager,
    answer_question
)

print("=" * 75)
print("TEST 1: LOADING MULTI-LIBRARY RAG PIPELINE")
print("=" * 75)

pipeline = initialize_rag_pipeline()
ntotal = pipeline["faiss_index"].ntotal
libs = pipeline["available_libraries"]

print(f"Total Vectors Indexed : {ntotal:,}")
print(f"Available Libraries   : {libs}")
assert ntotal == 1282, f"Expected 1282 vectors, got {ntotal}"
assert "Scikit-learn" in libs and "PyTorch" in libs and "Pandas" in libs
print("[PASS] Test 1: Pipeline loaded successfully.")

print("\n" + "=" * 75)
print("TEST 2: RETRIEVAL WITH & WITHOUT LIBRARY FILTER")
print("=" * 75)

# Retrieval without filter
hits_global = retrieve_context("What is overfitting and regularization?", pipeline, top_k=2)
print("Global Search: 'What is overfitting and regularization?'")
for h in hits_global:
    print(f"  - [{h['library']}] {h['doc_title']} (Score: {h['score']:.4f})")
assert len(hits_global) == 2
assert hits_global[0]["library"] == "Scikit-learn"

# Retrieval with library filter
hits_filtered = retrieve_context("How do I create and index data?", pipeline, top_k=2, library="Pandas")
print("\nFiltered Search (library='Pandas'): 'How do I create and index data?'")
for h in hits_filtered:
    print(f"  - [{h['library']}] {h['doc_title']} (Score: {h['score']:.4f})")
assert all(h["library"] == "Pandas" for h in hits_filtered)
print("[PASS] Test 2: Retrieval and filtering working properly.")

print("\n" + "=" * 75)
print("TEST 3: END-TO-END GENERATION & GROUNDING")
print("=" * 75)

conv = ConversationManager()
q1 = "How does GridSearchCV work in Scikit-learn?"
res1 = answer_question(q1, conv, pipeline, top_k=3)
print(f"Q: {q1}")
print(f"Answer Preview: {res1['answer'][:180]}...")
print(f"Citations: {[s['title'] for s in res1['sources']]}")
assert len(res1["sources"]) > 0
assert "grid" in res1["answer"].lower() or "estimator" in res1["answer"].lower()
print("[PASS] Test 3: Grounded answer generated with citations.")

time.sleep(2)

print("\n" + "=" * 75)
print("TEST 4: CONVERSATIONAL FOLLOW-UP QUESTION")
print("=" * 75)

q2 = "Can you show me a Python code example of that?"
res2 = answer_question(q2, conv, pipeline, top_k=3)
print(f"Follow-up: {q2}")
print(f"Search Query Reformulated As: \"{res2['search_query']}\"")
print(f"Answer Preview: {res2['answer'][:180]}...")
assert res2["search_query"] != q2, "Search query should be contextualized"
assert "gridsearchcv" in res2["search_query"].lower() or "scikit" in res2["search_query"].lower()
print("[PASS] Test 4: Dynamic query contextualization working correctly.")

time.sleep(2)

print("\n" + "=" * 75)
print("TEST 5: OUT-OF-SCOPE ABSTENTION")
print("=" * 75)

# Test an unsupported topic outside Python/DS/ML/DL
conv_fresh = ConversationManager()
q_oos = "How do I configure memory limits for Kubernetes pods in a deployment manifest?"
res_oos = answer_question(q_oos, conv_fresh, pipeline, top_k=3)
print(f"Out-of-Scope Q: {q_oos}")
print(f"Answer: {res_oos['answer']}")
ans_lower = res_oos["answer"].lower()
abstained = (
    "insufficient information" in ans_lower or
    "not contain" in ans_lower or
    "do not have" in ans_lower or
    "not mentioned" in ans_lower or
    "not provide" in ans_lower
)
print(f"Abstention Verified: {abstained}")
assert abstained, "Model should abstain on Kubernetes question!"
print("[PASS] Test 5: Out-of-scope question safely abstained.")

print("\n" + "=" * 75)
print("TEST 6: CONVERSATION RESET")
print("=" * 75)

print(f"Length before clear: {len(conv)}")
conv.clear()
print(f"Length after clear : {len(conv)}")
assert len(conv) == 0
print("[PASS] Test 6: Conversation reset cleared history.")

print("\n" + "=" * 75)
print("ALL RAG PIPELINE TESTS PASSED SUCCESSFULLY!")
print("=" * 75)
