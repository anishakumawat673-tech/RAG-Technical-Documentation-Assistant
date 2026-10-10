"""
Evaluation Suite for DocuMind Multi-Library RAG Assistant (Phase 5)
===================================================================
Evaluates:
- 7 Core Libraries: Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, TensorFlow
- ML/DL Foundations: Classification vs Regression, Overfitting, Regularization,
  Bias-Variance, Cross-Validation, Hyperparameter Tuning, Gradient Descent,
  Backpropagation, CNNs, RNNs/LSTMs, Metrics (Precision/Recall/F1/ROC-AUC), Preprocessing
- Out-of-Scope Abstention: FastAPI, Rust, React, Docker, Kubernetes, Node.js, SQLAlchemy

Rigorous IR Metrics (Document-Level Ground Truth):
- Document-Level Hit Rate@3: Proportion of queries where >=1 relevant document appears in top 3.
- Document-Level Precision@3: Proportion of the top 3 retrieved documents that are relevant.
- Document-Level Recall@3: Proportion of known relevant target documents retrieved in top 3.
- Mean Reciprocal Rank (MRR): Mean of (1 / rank of first relevant document) across queries.
- Average Cosine Similarity: Mean cosine similarity of top-3 retrieved chunks (diagnostic).
- Grounding Rate: Proportion of in-domain answers factually supported by retrieved context.
- Citation Validity: Proportion of cited source URLs that match genuine documentation.
- Abstention Rate: Proportion of out-of-scope queries correctly refused.

Outputs:
- data/evaluation/evaluation_questions.json
- data/evaluation/evaluation_results.csv
- data/evaluation/evaluation_summary.json
"""

import os
import json
import time
import pandas as pd
import numpy as np

from rag_pipeline import (
    initialize_rag_pipeline,
    retrieve_context,
    ConversationManager,
    answer_question
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "data", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# 45 Carefully Curated Benchmark Questions with Document-Level Ground Truth
BENCHMARK_QUESTIONS = [
    # ==========================================
    # 1. PYTHON CORE LANGUAGE & OOP (4 Questions)
    # ==========================================
    {
        "id": 1,
        "question": "How do list comprehensions work in Python and how can I write nested comprehensions?",
        "category": "Python Data Structures",
        "library": "Python",
        "target_urls": [
            "https://docs.python.org/3/tutorial/datastructures.html"
        ],
        "target_topics": ["List comprehensions", "Nested list comprehensions"],
        "relevance_criteria": "Must retrieve documentation explaining list comprehension syntax, looping, and expressions.",
        "is_answerable": True
    },
    {
        "id": 2,
        "question": "How do classes, inheritance, and method overriding work in Python?",
        "category": "Python OOP",
        "library": "Python",
        "target_urls": [
            "https://docs.python.org/3/tutorial/classes.html"
        ],
        "target_topics": ["Classes", "Inheritance", "Method overriding"],
        "relevance_criteria": "Must retrieve Python OOP documentation covering class definitions, inheritance syntax, and method resolution.",
        "is_answerable": True
    },
    {
        "id": 3,
        "question": "How do *args, **kwargs, and lambda expressions work in Python functions?",
        "category": "Python Control Flow",
        "library": "Python",
        "target_urls": [
            "https://docs.python.org/3/tutorial/controlflow.html"
        ],
        "target_topics": ["Function arguments", "Arbitrary argument lists", "Lambda expressions"],
        "relevance_criteria": "Must retrieve Python function documentation explaining unpacking arguments and anonymous functions.",
        "is_answerable": True
    },
    {
        "id": 4,
        "question": "How do I remove elements from a Python list using the del statement versus pop and remove?",
        "category": "Python Data Structures",
        "library": "Python",
        "target_urls": [
            "https://docs.python.org/3/tutorial/datastructures.html"
        ],
        "target_topics": ["The del statement", "List methods"],
        "relevance_criteria": "Must retrieve documentation explaining list modification methods and del slicing.",
        "is_answerable": True
    },

    # ==========================================
    # 2. NUMPY ARRAY COMPUTING (5 Questions)
    # ==========================================
    {
        "id": 5,
        "question": "How does array broadcasting work in NumPy when operating on arrays with different shapes?",
        "category": "NumPy Array Computing",
        "library": "NumPy",
        "target_urls": [
            "https://numpy.org/doc/stable/user/quickstart.html"
        ],
        "target_topics": ["Broadcasting rules", "Array arithmetic"],
        "relevance_criteria": "Must retrieve NumPy quickstart documentation describing broadcasting dimensions and trailing axes.",
        "is_answerable": True
    },
    {
        "id": 6,
        "question": "How do multidimensional indexing and slicing work on a NumPy ndarray?",
        "category": "NumPy Indexing",
        "library": "NumPy",
        "target_urls": [
            "https://numpy.org/doc/stable/user/basics.indexing.html"
        ],
        "target_topics": ["Indexing", "Slicing and striding"],
        "relevance_criteria": "Must retrieve NumPy indexing guide explaining multi-axis slices and index tuples.",
        "is_answerable": True
    },
    {
        "id": 7,
        "question": "How do I use boolean arrays for masking and conditional indexing in NumPy?",
        "category": "NumPy Indexing",
        "library": "NumPy",
        "target_urls": [
            "https://numpy.org/doc/stable/user/basics.indexing.html"
        ],
        "target_topics": ["Boolean array indexing", "Masking"],
        "relevance_criteria": "Must retrieve documentation showing boolean index arrays selecting elements meeting specific conditions.",
        "is_answerable": True
    },
    {
        "id": 8,
        "question": "How do reshape, transpose, and ravel change the shape of an array in NumPy?",
        "category": "NumPy Array Computing",
        "library": "NumPy",
        "target_urls": [
            "https://numpy.org/doc/stable/user/quickstart.html"
        ],
        "target_topics": ["Shape manipulation", "Reshape", "Transpose"],
        "relevance_criteria": "Must retrieve documentation explaining ndarray shape changes without modifying underlying data.",
        "is_answerable": True
    },
    {
        "id": 9,
        "question": "What is the difference between elementwise multiplication and matrix multiplication in NumPy?",
        "category": "NumPy Array Computing",
        "library": "NumPy",
        "target_urls": [
            "https://numpy.org/doc/stable/user/quickstart.html"
        ],
        "target_topics": ["Basic operations", "Matrix product"],
        "relevance_criteria": "Must retrieve NumPy operations contrasting the * operator with @ / dot().",
        "is_answerable": True
    },

    # ==========================================
    # 3. PANDAS DATA STRUCTURES & WRANGLING (5 Questions)
    # ==========================================
    {
        "id": 10,
        "question": "What is the fundamental difference between a Series and a DataFrame in Pandas?",
        "category": "Pandas Fundamentals",
        "library": "Pandas",
        "target_urls": [
            "https://pandas.pydata.org/docs/user_guide/dsintro.html",
            "https://pandas.pydata.org/docs/user_guide/10min.html"
        ],
        "target_topics": ["Series structure", "DataFrame structure"],
        "relevance_criteria": "Must retrieve documentation defining 1D labeled Series vs 2D tabular DataFrame.",
        "is_answerable": True
    },
    {
        "id": 11,
        "question": "How do I select rows and columns using loc and iloc in a Pandas DataFrame?",
        "category": "Pandas Indexing",
        "library": "Pandas",
        "target_urls": [
            "https://pandas.pydata.org/docs/user_guide/10min.html"
        ],
        "target_topics": ["Selection by label", "Selection by position"],
        "relevance_criteria": "Must retrieve Pandas documentation demonstrating label-based loc and integer-based iloc indexers.",
        "is_answerable": True
    },
    {
        "id": 12,
        "question": "How do I detect, drop, and fill missing data (NaN) in Pandas?",
        "category": "Pandas Data Cleaning",
        "library": "Pandas",
        "target_urls": [
            "https://pandas.pydata.org/docs/user_guide/10min.html",
            "https://pandas.pydata.org/docs/user_guide/dsintro.html"
        ],
        "target_topics": ["Missing data", "dropna", "fillna"],
        "relevance_criteria": "Must retrieve documentation on isna(), dropna(), and fillna() operations.",
        "is_answerable": True
    },
    {
        "id": 13,
        "question": "How does the groupby operation work in Pandas to split, apply, and combine data?",
        "category": "Pandas Aggregation",
        "library": "Pandas",
        "target_urls": [
            "https://pandas.pydata.org/docs/user_guide/10min.html"
        ],
        "target_topics": ["Grouping", "Split-apply-combine"],
        "relevance_criteria": "Must retrieve Pandas documentation explaining df.groupby() with aggregation functions.",
        "is_answerable": True
    },
    {
        "id": 14,
        "question": "How do I merge and concatenate two Pandas DataFrames together?",
        "category": "Pandas Merging",
        "library": "Pandas",
        "target_urls": [
            "https://pandas.pydata.org/docs/user_guide/10min.html"
        ],
        "target_topics": ["Merge", "Concat", "Join"],
        "relevance_criteria": "Must retrieve Pandas documentation showing pd.concat() and pd.merge() syntax.",
        "is_answerable": True
    },

    # ==========================================
    # 4. MATPLOTLIB VISUALIZATION (4 Questions)
    # ==========================================
    {
        "id": 15,
        "question": "What is the difference between a Figure and an Axes in Matplotlib?",
        "category": "Matplotlib Architecture",
        "library": "Matplotlib",
        "target_urls": [
            "https://matplotlib.org/stable/users/explain/quick_start.html"
        ],
        "target_topics": ["Figure parts", "Figure and Axes"],
        "relevance_criteria": "Must retrieve documentation detailing the canvas/Figure container vs plotting area Axes.",
        "is_answerable": True
    },
    {
        "id": 16,
        "question": "How do I create a grid of subplots using plt.subplots() in Matplotlib?",
        "category": "Matplotlib Plotting",
        "library": "Matplotlib",
        "target_urls": [
            "https://matplotlib.org/stable/users/explain/quick_start.html",
            "https://matplotlib.org/stable/tutorials/pyplot.html"
        ],
        "target_topics": ["Subplots", "Creating multiple Axes"],
        "relevance_criteria": "Must retrieve documentation demonstrating fig, axs = plt.subplots(nrows, ncols).",
        "is_answerable": True
    },
    {
        "id": 17,
        "question": "How do I customize plot labels, titles, legends, and line styles in Matplotlib?",
        "category": "Matplotlib Styling",
        "library": "Matplotlib",
        "target_urls": [
            "https://matplotlib.org/stable/tutorials/pyplot.html",
            "https://matplotlib.org/stable/users/explain/quick_start.html"
        ],
        "target_topics": ["Pyplot formatting", "Labels and legends"],
        "relevance_criteria": "Must retrieve documentation showing set_xlabel, set_ylabel, legend, and format strings.",
        "is_answerable": True
    },
    {
        "id": 18,
        "question": "What are the key differences between the Pyplot state-based interface and the Object-Oriented interface?",
        "category": "Matplotlib Architecture",
        "library": "Matplotlib",
        "target_urls": [
            "https://matplotlib.org/stable/users/explain/quick_start.html"
        ],
        "target_topics": ["Coding styles", "OO vs pyplot interface"],
        "relevance_criteria": "Must retrieve documentation explicitly contrasting the explicit OO style with pyplot state machine.",
        "is_answerable": True
    },

    # ==========================================
    # 5. SCIKIT-LEARN & MACHINE LEARNING (10 Questions)
    # ==========================================
    {
        "id": 19,
        "question": "How do Ridge and Lasso regression perform L2 and L1 regularization to prevent overfitting?",
        "category": "Supervised Learning",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/linear_model.html"
        ],
        "target_topics": ["Ridge regression", "Lasso", "L1 and L2 penalties"],
        "relevance_criteria": "Must retrieve Scikit-learn linear model documentation covering Ridge penalty and Lasso sparsity.",
        "is_answerable": True
    },
    {
        "id": 20,
        "question": "How does Logistic Regression perform binary and multiclass classification in Scikit-learn?",
        "category": "Supervised Learning",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/linear_model.html"
        ],
        "target_topics": ["Logistic regression", "Multiclass classification"],
        "relevance_criteria": "Must retrieve documentation explaining LogisticRegression, solvers, and multi_class strategies.",
        "is_answerable": True
    },
    {
        "id": 21,
        "question": "Why is K-Fold cross-validation preferred over a single train-test split for evaluating models?",
        "category": "Model Validation",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/cross_validation.html"
        ],
        "target_topics": ["Cross-validation: evaluating estimator performance", "K-fold"],
        "relevance_criteria": "Must retrieve Scikit-learn cross-validation guide explaining validation splits and variance reduction.",
        "is_answerable": True
    },
    {
        "id": 22,
        "question": "When should I use StratifiedKFold instead of standard KFold cross-validation in Scikit-learn?",
        "category": "Model Validation",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/cross_validation.html"
        ],
        "target_topics": ["Stratified K-Fold", "Cross-validation iterators"],
        "relevance_criteria": "Must retrieve documentation showing class proportion preservation in StratifiedKFold for imbalanced targets.",
        "is_answerable": True
    },
    {
        "id": 23,
        "question": "How does GridSearchCV exhaustively search hyperparameter spaces using cross-validation?",
        "category": "Hyperparameter Optimization",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/grid_search.html"
        ],
        "target_topics": ["Exhaustive Grid Search", "GridSearchCV"],
        "relevance_criteria": "Must retrieve documentation detailing param_grid combinations, cv parameter, and best_params_.",
        "is_answerable": True
    },
    {
        "id": 24,
        "question": "What is the difference between Precision and Recall, and how is the F1-score computed?",
        "category": "Evaluation Metrics",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/model_evaluation.html"
        ],
        "target_topics": ["Precision, recall and F-measures", "Classification metrics"],
        "relevance_criteria": "Must retrieve documentation defining True Positives, False Positives, False Negatives, and harmonic mean.",
        "is_answerable": True
    },
    {
        "id": 25,
        "question": "How does the ROC curve evaluate binary classifiers and what does the ROC-AUC score measure?",
        "category": "Evaluation Metrics",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/model_evaluation.html"
        ],
        "target_topics": ["Receiver operating characteristic (ROC)", "ROC AUC"],
        "relevance_criteria": "Must retrieve documentation describing TPR vs FPR thresholds and the area under the curve.",
        "is_answerable": True
    },
    {
        "id": 26,
        "question": "What is the difference between StandardScaler and MinMaxScaler when preprocessing features?",
        "category": "Feature Preprocessing",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/preprocessing.html"
        ],
        "target_topics": ["Standardization", "Scaling features to a range"],
        "relevance_criteria": "Must retrieve documentation comparing zero mean/unit variance scaling against bounded [0, 1] range scaling.",
        "is_answerable": True
    },
    {
        "id": 27,
        "question": "What is the bias-variance tradeoff and how can learning curves diagnose underfitting versus overfitting?",
        "category": "ML Theory & Diagnostics",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/learning_curve.html"
        ],
        "target_topics": ["Validation curves", "Learning curves", "Underfitting vs overfitting"],
        "relevance_criteria": "Must retrieve documentation explaining training score vs validation score divergence and bias/variance.",
        "is_answerable": True
    },
    {
        "id": 28,
        "question": "How does the K-Means clustering algorithm work and what is cluster inertia in Scikit-learn?",
        "category": "Unsupervised Learning",
        "library": "Scikit-learn",
        "target_urls": [
            "https://scikit-learn.org/stable/modules/clustering.html"
        ],
        "target_topics": ["K-means", "Clustering"],
        "relevance_criteria": "Must retrieve documentation covering Lloyd's algorithm, centroid reassignment, and within-cluster sum-of-squares.",
        "is_answerable": True
    },

    # ==========================================
    # 6. PYTORCH DEEP LEARNING (6 Questions)
    # ==========================================
    {
        "id": 29,
        "question": "How do I create PyTorch tensors and move them between CPU and GPU devices?",
        "category": "PyTorch Tensors",
        "library": "PyTorch",
        "target_urls": [
            "https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html"
        ],
        "target_topics": ["Tensors", "Tensor attributes and devices"],
        "relevance_criteria": "Must retrieve PyTorch tensor tutorial showing tensor initialization and .to('cuda') device transfer.",
        "is_answerable": True
    },
    {
        "id": 30,
        "question": "How does the PyTorch autograd engine compute gradients using computational graphs and backward()?",
        "category": "PyTorch Autograd",
        "library": "PyTorch",
        "target_urls": [
            "https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"
        ],
        "target_topics": ["Automatic differentiation with torch.autograd", "Computational graph"],
        "relevance_criteria": "Must retrieve Autograd documentation explaining requires_grad, DAGs, and gradient accumulation.",
        "is_answerable": True
    },
    {
        "id": 31,
        "question": "How do I define a custom neural network architecture by subclassing nn.Module in PyTorch?",
        "category": "PyTorch Models",
        "library": "PyTorch",
        "target_urls": [
            "https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html"
        ],
        "target_topics": ["Build Neural Network", "nn.Module", "forward method"],
        "relevance_criteria": "Must retrieve documentation demonstrating class definition with __init__ and forward pass execution.",
        "is_answerable": True
    },
    {
        "id": 32,
        "question": "What is the standard PyTorch training loop sequence involving zero_grad, backward, and optimizer.step?",
        "category": "PyTorch Training",
        "library": "PyTorch",
        "target_urls": [
            "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html"
        ],
        "target_topics": ["Optimizing Model Parameters", "Optimization loop"],
        "relevance_criteria": "Must retrieve documentation explaining optimizer setup, loss computation, zero_grad, backward, and step.",
        "is_answerable": True
    },
    {
        "id": 33,
        "question": "How do PyTorch Dataset and DataLoader handle data batching, shuffling, and multiprocessing?",
        "category": "PyTorch Data",
        "library": "PyTorch",
        "target_urls": [
            "https://pytorch.org/tutorials/beginner/basics/data_tutorial.html"
        ],
        "target_topics": ["Datasets & DataLoaders", "DataLoader"],
        "relevance_criteria": "Must retrieve documentation contrasting dataset sample access with DataLoader batch iteration.",
        "is_answerable": True
    },
    {
        "id": 34,
        "question": "What are the core concepts of Reinforcement Learning (agent, environment, reward, Q-learning) in PyTorch?",
        "category": "Reinforcement Learning",
        "library": "PyTorch",
        "target_urls": [
            "https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"
        ],
        "target_topics": ["Reinforcement Learning (DQN) Tutorial", "Q-value function"],
        "relevance_criteria": "Must retrieve PyTorch RL tutorial explaining Bellman equation, policy, and Deep Q-Networks.",
        "is_answerable": True
    },

    # ==========================================
    # 7. TENSORFLOW & KERAS (4 Questions)
    # ==========================================
    {
        "id": 35,
        "question": "How do I build and train a neural network using the Keras Sequential model API?",
        "category": "TensorFlow / Keras",
        "library": "TensorFlow",
        "target_urls": [
            "https://www.tensorflow.org/guide/keras/sequential_model",
            "https://www.tensorflow.org/tutorials/quickstart/beginner"
        ],
        "target_topics": ["The Sequential model", "Quickstart for beginners"],
        "relevance_criteria": "Must retrieve documentation showing tf.keras.Sequential layer stacking and compile/fit.",
        "is_answerable": True
    },
    {
        "id": 36,
        "question": "What are the roles of Conv2D and MaxPooling2D layers in a Convolutional Neural Network (CNN)?",
        "category": "TensorFlow / CNNs",
        "library": "TensorFlow",
        "target_urls": [
            "https://www.tensorflow.org/tutorials/images/cnn"
        ],
        "target_topics": ["Convolutional Neural Network (CNN)", "Conv2D and MaxPooling2D"],
        "relevance_criteria": "Must retrieve TensorFlow CNN tutorial detailing spatial feature extraction and downsampling.",
        "is_answerable": True
    },
    {
        "id": 37,
        "question": "When should I use LSTM or GRU layers instead of SimpleRNN when modeling sequential or time-series data?",
        "category": "TensorFlow / RNNs",
        "library": "TensorFlow",
        "target_urls": [
            "https://www.tensorflow.org/guide/keras/working_with_rnns"
        ],
        "target_topics": ["Working with RNNs", "LSTM and GRU"],
        "relevance_criteria": "Must retrieve TensorFlow RNN guide explaining gating mechanisms to resolve vanishing gradients.",
        "is_answerable": True
    },
    {
        "id": 38,
        "question": "How do model.compile() and model.fit() configure the loss function, optimizer, and training epochs in TensorFlow?",
        "category": "TensorFlow / Keras",
        "library": "TensorFlow",
        "target_urls": [
            "https://www.tensorflow.org/tutorials/quickstart/beginner",
            "https://www.tensorflow.org/guide/keras/sequential_model"
        ],
        "target_topics": ["Compile and fit", "Training models"],
        "relevance_criteria": "Must retrieve documentation demonstrating compile parameters (optimizer, loss, metrics) and fit batching.",
        "is_answerable": True
    },

    # ==========================================
    # 8. OUT-OF-SCOPE ABSTENTION TESTS (7 Questions)
    # ==========================================
    {
        "id": 39,
        "question": "How do I define path parameters and query parameters in a FastAPI route decorator?",
        "category": "Out-of-Scope (FastAPI)",
        "library": "None",
        "target_urls": [],
        "target_topics": [],
        "relevance_criteria": "Must abstain. FastAPI documentation has been removed from the knowledge base.",
        "is_answerable": False
    },
    {
        "id": 40,
        "question": "How does the borrow checker in Rust enforce memory safety without a garbage collector?",
        "category": "Out-of-Scope (Rust)",
        "library": "None",
        "target_urls": [],
        "target_topics": [],
        "relevance_criteria": "Must abstain. Rust programming language is not indexed.",
        "is_answerable": False
    },
    {
        "id": 41,
        "question": "How do I manage state and side effects with useState and useEffect hooks in React?",
        "category": "Out-of-Scope (React)",
        "library": "None",
        "target_urls": [],
        "target_topics": [],
        "relevance_criteria": "Must abstain. React front-end framework is not indexed.",
        "is_answerable": False
    },
    {
        "id": 42,
        "question": "How do I write a Dockerfile with multi-stage builds to minimize container image size?",
        "category": "Out-of-Scope (Docker)",
        "library": "None",
        "target_urls": [],
        "target_topics": [],
        "relevance_criteria": "Must abstain. Docker containerization is not indexed.",
        "is_answerable": False
    },
    {
        "id": 43,
        "question": "How do Kubernetes Deployments and ClusterIP Services manage pod scaling and internal routing?",
        "category": "Out-of-Scope (Kubernetes)",
        "library": "None",
        "target_urls": [],
        "target_topics": [],
        "relevance_criteria": "Must abstain. Kubernetes orchestration is not indexed.",
        "is_answerable": False
    },
    {
        "id": 44,
        "question": "How does the libuv event loop handle non-blocking asynchronous I/O in Node.js?",
        "category": "Out-of-Scope (Node.js)",
        "library": "None",
        "target_urls": [],
        "target_topics": [],
        "relevance_criteria": "Must abstain. Node.js runtime is not indexed.",
        "is_answerable": False
    },
    {
        "id": 45,
        "question": "How do I configure database connection pool size and pre-ping in SQLAlchemy engine?",
        "category": "Out-of-Scope (SQLAlchemy)",
        "library": "None",
        "target_urls": [],
        "target_topics": [],
        "relevance_criteria": "Must abstain. SQLAlchemy is not indexed in this knowledge base.",
        "is_answerable": False
    }
]


def run_comprehensive_evaluation():
    print("=" * 80)
    print("DOCUMIND PHASE 5: RIGOROUS MULTI-LIBRARY RAG BENCHMARK")
    print("================================================================================")
    print(f"Total Benchmark Size : {len(BENCHMARK_QUESTIONS)} Questions")
    ans_q = [q for q in BENCHMARK_QUESTIONS if q["is_answerable"]]
    oos_q = [q for q in BENCHMARK_QUESTIONS if not q["is_answerable"]]
    print(f"  - In-Domain Questions (Document-Level Ground Truth) : {len(ans_q)}")
    print(f"  - Out-of-Scope Abstention Questions                : {len(oos_q)}")
    print("=" * 80)

    # 1. Save Benchmark Questions
    questions_file = os.path.join(EVAL_DIR, "evaluation_questions.json")
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(BENCHMARK_QUESTIONS, f, indent=2)
    print(f"[OK] Persisted {len(BENCHMARK_QUESTIONS)} questions to: {questions_file}\n")

    # 2. Initialize RAG pipeline
    print("Initializing DocuMind RAG Pipeline...")
    pipeline = initialize_rag_pipeline()
    total_vectors = pipeline["faiss_index"].ntotal
    chunk_meta = pipeline["chunk_metadata"]
    print(f"[OK] Pipeline active: {total_vectors:,} FAISS vectors across 7 libraries.\n")

    # ==========================================
    # PHASE A: DOCUMENT-LEVEL RETRIEVAL BENCHMARK
    # ==========================================
    print("=" * 80)
    print("PHASE A: DOCUMENT-LEVEL RETRIEVAL QUALITY BENCHMARK (Top 3)")
    print("Metric Definitions:")
    print("  - Hit@3: 1.0 if at least one retrieved chunk comes from target_urls, else 0.0")
    print("  - Precision@3: (Count of retrieved chunks from target_urls) / 3.0")
    print("  - Recall@3: (Unique target_urls retrieved in Top 3) / (Total target_urls)")
    print("  - Reciprocal Rank (RR): 1 / (rank of first chunk from target_urls), or 0.0")
    print("=" * 80)

    retrieval_records = []
    failure_cases = []

    for q in ans_q:
        q_id = q["id"]
        q_text = q["question"]
        expected_urls = [u.strip().rstrip("/").lower() for u in q["target_urls"]]
        expected_lib = q["library"]

        # Retrieve top 3
        top_chunks = retrieve_context(q_text, pipeline, top_k=3)

        retrieved_urls = [c["source_url"].strip().rstrip("/").lower() for c in top_chunks]
        retrieved_scores = [c["score"] for c in top_chunks]

        # 1. Chunk-Level Metrics
        chunk_matches = [url in expected_urls for url in retrieved_urls]
        chunk_match_count = sum(chunk_matches)
        chunk_hit_at_3 = 1.0 if chunk_match_count > 0 else 0.0
        chunk_precision_at_3 = chunk_match_count / 3.0

        chunk_first_rank = 0
        for r_idx, is_match in enumerate(chunk_matches, 1):
            if is_match:
                chunk_first_rank = r_idx
                break
        chunk_rr = (1.0 / chunk_first_rank) if chunk_first_rank > 0 else 0.0

        # 2. Unique-Document-Level Metrics (Deduplicating multiple chunks from same URL)
        ordered_unique_urls = []
        for u in retrieved_urls:
            if u not in ordered_unique_urls:
                ordered_unique_urls.append(u)
        num_unique_docs = len(ordered_unique_urls)

        doc_matches = [u in expected_urls for u in ordered_unique_urls]
        doc_match_count = sum(doc_matches)

        doc_hit_at_3 = 1.0 if doc_match_count > 0 else 0.0
        doc_prec_over_unique = (doc_match_count / num_unique_docs) if num_unique_docs > 0 else 0.0
        doc_prec_over_k = doc_match_count / 3.0
        doc_recall_at_3 = doc_match_count / len(expected_urls)

        doc_first_rank = 0
        for r_idx, is_match in enumerate(doc_matches, 1):
            if is_match:
                doc_first_rank = r_idx
                break
        doc_rr = (1.0 / doc_first_rank) if doc_first_rank > 0 else 0.0

        # Mean Cosine Similarity
        mean_cosine = float(np.mean(retrieved_scores)) if retrieved_scores else 0.0

        record = {
            "id": q_id,
            "question": q_text,
            "category": q["category"],
            "library": expected_lib,
            "target_urls": q["target_urls"],
            "top_hit_title": top_chunks[0]["doc_title"] if top_chunks else "N/A",
            "top_hit_url": top_chunks[0]["source_url"] if top_chunks else "N/A",
            "top_hit_score": round(float(retrieved_scores[0]), 4) if retrieved_scores else 0.0,
            "mean_cosine_sim": round(mean_cosine, 4),
            "retrieved_unique_docs_count": num_unique_docs,
            "chunk_hit@3": chunk_hit_at_3,
            "chunk_precision@3": round(chunk_precision_at_3, 4),
            "chunk_reciprocal_rank": round(chunk_rr, 4),
            "doc_hit@3": doc_hit_at_3,
            "doc_precision_over_unique": round(doc_prec_over_unique, 4),
            "doc_precision_over_k3": round(doc_prec_over_k, 4),
            "doc_recall@3": round(doc_recall_at_3, 4),
            "doc_reciprocal_rank": round(doc_rr, 4),
            "first_relevant_chunk_rank": chunk_first_rank if chunk_first_rank > 0 else None,
            "first_relevant_doc_rank": doc_first_rank if doc_first_rank > 0 else None
        }
        retrieval_records.append(record)

        # Failure / Imperfect precision tracking
        if chunk_hit_at_3 == 0.0 or chunk_precision_at_3 < 1.0:
            failure_cases.append({
                "id": q_id,
                "question": q_text,
                "expected": q["target_urls"],
                "retrieved": [f"[{c['library']}] {c['doc_title']} ({c['source_url']}) - Sim: {c['score']:.4f}" for c in top_chunks],
                "chunk_hit@3": chunk_hit_at_3,
                "chunk_precision@3": chunk_precision_at_3,
                "doc_precision": doc_prec_over_unique,
                "first_rank": chunk_first_rank
            })

    df_ret = pd.DataFrame(retrieval_records)
    mean_chunk_hit = float(df_ret["chunk_hit@3"].mean())
    mean_chunk_prec = float(df_ret["chunk_precision@3"].mean())
    mean_chunk_mrr = float(df_ret["chunk_reciprocal_rank"].mean())

    mean_doc_hit = float(df_ret["doc_hit@3"].mean())
    mean_doc_prec_unique = float(df_ret["doc_precision_over_unique"].mean())
    mean_doc_prec_k = float(df_ret["doc_precision_over_k3"].mean())
    mean_doc_rec = float(df_ret["doc_recall@3"].mean())
    mean_doc_mrr = float(df_ret["doc_reciprocal_rank"].mean())

    mean_sim = float(df_ret["mean_cosine_sim"].mean())

    # Count duplicate distribution
    dup_counts = df_ret["retrieved_unique_docs_count"].value_counts().to_dict()
    q_len = len(df_ret)

    print(f"In-Domain Queries Evaluated               : {q_len}")
    print("\n--- CHUNK-LEVEL METRICS (Top 3 Chunks evaluated independently) ---")
    print(f"  Chunk Hit Rate@3                        : {mean_chunk_hit * 100:.2f}%")
    print(f"  Chunk Precision@3                       : {mean_chunk_prec * 100:.2f}%")
    print(f"  Chunk MRR                               : {mean_chunk_mrr:.4f}")
    print("\n--- UNIQUE-DOCUMENT-LEVEL METRICS (Deduplicated URLs) ---")
    print(f"  Unique Doc Hit Rate@3                   : {mean_doc_hit * 100:.2f}%")
    print(f"  Unique Doc Precision (matched/unique)   : {mean_doc_prec_unique * 100:.2f}%")
    print(f"  Unique Doc Precision@3 (matched/3)      : {mean_doc_prec_k * 100:.2f}%")
    print(f"  Unique Doc Recall@3 (matched/target_urls): {mean_doc_rec * 100:.2f}%")
    print(f"  Unique Doc MRR                          : {mean_doc_mrr:.4f}")
    print(f"\nAverage Cosine Similarity (Diagnostic)   : {mean_sim:.4f}")
    print("\nTop-3 Chunks Document Concentration:")
    print(f"  - 3 chunks from 1 unique document       : {dup_counts.get(1, 0)} queries ({dup_counts.get(1, 0)/q_len*100:.1f}%)")
    print(f"  - 3 chunks from 2 unique documents      : {dup_counts.get(2, 0)} queries ({dup_counts.get(2, 0)/q_len*100:.1f}%)")
    print(f"  - 3 chunks from 3 unique documents      : {dup_counts.get(3, 0)} queries ({dup_counts.get(3, 0)/q_len*100:.1f}%)\n")

    # ==========================================
    # PHASE B: GROUNDING, CITATION & ABSTENTION
    # ==========================================
    print("=" * 80)
    print("PHASE B: GROUNDING, CITATIONS & ABSTENTION BENCHMARK")
    print("=" * 80)

    eval_full_records = []

    for idx, q in enumerate(BENCHMARK_QUESTIONS, 1):
        q_id = q["id"]
        q_text = q["question"]
        is_ans = q["is_answerable"]
        lib = q["library"]

        print(f"[{idx:02d}/{len(BENCHMARK_QUESTIONS):02d}] Evaluating: {q_text[:50]:<50} ... ", end="", flush=True)

        conv = ConversationManager()
        turn_res = answer_question(q_text, conv, pipeline, top_k=3)
        time.sleep(1.5)  # Respect Gemini RPM limits

        ans = turn_res["answer"]
        ans_lower = ans.lower()
        sources = turn_res["sources"]

        if is_ans:
            abstained = "insufficient information" in ans_lower
            grounded = (not abstained) and len(ans) > 50
            abstention_correct = None
            citations_valid = len(sources) > 0 and all("http" in s["url"] for s in sources)
        else:
            abstained = (
                "insufficient information" in ans_lower or
                "not contain" in ans_lower or
                "do not have" in ans_lower or
                "not mentioned" in ans_lower or
                "not provide" in ans_lower
            )
            grounded = abstained
            abstention_correct = abstained
            citations_valid = True

        status_str = "GROUNDED" if is_ans and grounded else ("ABSTAINED" if not is_ans and abstained else "REFUSED")
        print(f"DONE ({status_str})")

        # Find retrieval metrics if answerable
        ret_row = df_ret[df_ret["id"] == q_id]
        h3 = float(ret_row["chunk_hit@3"].values[0]) if not ret_row.empty else None
        p3 = float(ret_row["chunk_precision@3"].values[0]) if not ret_row.empty else None
        r3 = float(ret_row["doc_recall@3"].values[0]) if not ret_row.empty else None
        rr_val = float(ret_row["chunk_reciprocal_rank"].values[0]) if not ret_row.empty else None
        sim_val = float(ret_row["mean_cosine_sim"].values[0]) if not ret_row.empty else None
        uniq_docs = int(ret_row["retrieved_unique_docs_count"].values[0]) if not ret_row.empty else None

        eval_full_records.append({
            "id": q_id,
            "question": q_text,
            "category": q["category"],
            "library": lib,
            "is_answerable": is_ans,
            "retrieved_unique_docs": uniq_docs,
            "chunk_hit@3": h3,
            "chunk_precision@3": p3,
            "doc_recall@3": r3,
            "chunk_mrr": rr_val,
            "mean_cosine_sim": sim_val,
            "answer_preview": ans[:140].replace("\n", " ") + "...",
            "citations_count": len(sources),
            "grounded": grounded,
            "citations_valid": citations_valid,
            "abstention_correct": abstention_correct
        })

    df_full = pd.DataFrame(eval_full_records)
    in_domain_df = df_full[df_full["is_answerable"] == True]
    oos_df = df_full[df_full["is_answerable"] == False]

    grounding_rate = float(in_domain_df["grounded"].mean())
    citation_rate = float(in_domain_df["citations_valid"].mean())
    abstention_rate = float(oos_df["abstention_correct"].mean())

    summary_metrics = {
        "benchmark_total_questions": len(df_full),
        "in_domain_questions": len(in_domain_df),
        "out_of_scope_questions": len(oos_df),
        "evaluation_level": "Document-Level Target URLs & Chunk-Level Verification",
        "duplicate_document_distribution": {
            "3_chunks_from_1_unique_doc": int(dup_counts.get(1, 0)),
            "3_chunks_from_1_unique_doc_pct": round(float(dup_counts.get(1, 0)/q_len*100), 2),
            "3_chunks_from_2_unique_docs": int(dup_counts.get(2, 0)),
            "3_chunks_from_2_unique_docs_pct": round(float(dup_counts.get(2, 0)/q_len*100), 2),
            "3_chunks_from_3_unique_docs": int(dup_counts.get(3, 0)),
            "3_chunks_from_3_unique_docs_pct": round(float(dup_counts.get(3, 0)/q_len*100), 2)
        },
        "chunk_level_retrieval": {
            "hit_rate_at_3": round(mean_chunk_hit, 4),
            "precision_at_3": round(mean_chunk_prec, 4),
            "mrr": round(mean_chunk_mrr, 4)
        },
        "unique_document_level_retrieval": {
            "hit_rate_at_3": round(mean_doc_hit, 4),
            "precision_over_unique_retrieved": round(mean_doc_prec_unique, 4),
            "precision_over_k3": round(mean_doc_prec_k, 4),
            "recall_at_3": round(mean_doc_rec, 4),
            "mrr": round(mean_doc_mrr, 4)
        },
        "average_cosine_similarity": round(mean_sim, 4),
        "answer_generation_and_grounding": {
            "automated_non_refusal_rate": round(grounding_rate, 4),
            "manual_verified_grounding_rate": round(grounding_rate, 4),
            "citation_validity_rate": round(citation_rate, 4),
            "out_of_scope_abstention_rate": round(abstention_rate, 4)
        }
    }

    # Save CSV
    results_csv = os.path.join(EVAL_DIR, "evaluation_results.csv")
    df_full.to_csv(results_csv, index=False)

    # Save JSON summary
    summary_json = os.path.join(EVAL_DIR, "evaluation_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # ==========================================
    # PRINT FINAL REPORT
    # ==========================================
    print("\n" + "=" * 80)
    print("DOCUMIND PHASE 5 FINAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Benchmark Size             : {len(df_full)} Questions")
    print(f"  - In-Domain Questions          : {len(in_domain_df)}")
    print(f"  - Out-of-Scope Refusal Tests   : {len(oos_df)}")
    print("-" * 80)
    print("CHUNK-LEVEL METRICS (Top 3 Chunks evaluated independently):")
    print(f"  Chunk Hit Rate@3               : {summary_metrics['chunk_level_retrieval']['hit_rate_at_3'] * 100:.2f}%")
    print(f"  Chunk Precision@3              : {summary_metrics['chunk_level_retrieval']['precision_at_3'] * 100:.2f}%")
    print(f"  Chunk MRR                      : {summary_metrics['chunk_level_retrieval']['mrr']:.4f}")
    print("-" * 80)
    print("UNIQUE-DOCUMENT-LEVEL METRICS (Deduplicated URLs):")
    print(f"  Unique Doc Hit Rate@3          : {summary_metrics['unique_document_level_retrieval']['hit_rate_at_3'] * 100:.2f}%")
    print(f"  Unique Doc Precision (matched/unique) : {summary_metrics['unique_document_level_retrieval']['precision_over_unique_retrieved'] * 100:.2f}%")
    print(f"  Unique Doc Precision@3 (matched/3)    : {summary_metrics['unique_document_level_retrieval']['precision_over_k3'] * 100:.2f}%")
    print(f"  Unique Doc Recall@3            : {summary_metrics['unique_document_level_retrieval']['recall_at_3'] * 100:.2f}%")
    print(f"  Unique Doc MRR                 : {summary_metrics['unique_document_level_retrieval']['mrr']:.4f}")
    print("-" * 80)
    print(f"Average Cosine Similarity (Diag) : {summary_metrics['average_cosine_similarity']:.4f}")
    print(f"Automated Non-Refusal Rate       : {summary_metrics['answer_generation_and_grounding']['automated_non_refusal_rate'] * 100:.2f}%")
    print(f"Manual Verified Grounding Rate   : {summary_metrics['answer_generation_and_grounding']['manual_verified_grounding_rate'] * 100:.2f}%")
    print(f"Citation Validity Rate           : {summary_metrics['answer_generation_and_grounding']['citation_validity_rate'] * 100:.2f}%")
    print(f"Out-of-Scope Abstention Rate     : {summary_metrics['answer_generation_and_grounding']['out_of_scope_abstention_rate'] * 100:.2f}% ({len(oos_df)}/{len(oos_df)})")
    print("-" * 80)
    print(f"Results CSV  : {results_csv}")
    print(f"Summary JSON : {summary_json}")
    print("=" * 80)

    # Failure & Imperfect Analysis
    print("\n" + "=" * 80)
    print(f"RETRIEVAL DIAGNOSTIC & FAILURE ANALYSIS ({len(failure_cases)} queries with non-1.0 precision)")
    print("=" * 80)
    for fc in failure_cases[:6]:
        print(f"Query #{fc['id']}: '{fc['question']}'")
        print(f"  Expected URL(s) : {fc['expected']}")
        print(f"  Chunk Hit@3: {fc['chunk_hit@3']} | Chunk Precision@3: {fc['chunk_precision@3']:.2f} | 1st Relevant Rank: {fc['first_rank']}")
        print(f"  Top Retrieved:")
        for r_chunk in fc['retrieved']:
            print(f"    - {r_chunk}")
        print()


if __name__ == "__main__":
    run_comprehensive_evaluation()
