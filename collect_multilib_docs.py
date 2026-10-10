"""
DocuMind Multi-Library & ML/DL Documentation Collector
======================================================
Collects curated official documentation and conceptual guides for:
- Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, TensorFlow
- ML/DL Foundations: Supervised/Unsupervised/RL, Classification/Regression,
  Overfitting/Regularization, Bias-Variance, Cross-Validation, Gradient Descent,
  Backpropagation, CNNs, RNNs, Evaluation Metrics, and Feature Preprocessing.

Outputs: data/raw/multilib_docs.json
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup

# Curated registry of official documentation pages
DOCUMENTATION_SOURCES = [
    # --- 1. Python ---
    {
        "library": "Python",
        "category": "Core Language & Data Structures",
        "concept": "Data structures: lists, dicts, sets, comprehensions",
        "url": "https://docs.python.org/3/tutorial/datastructures.html"
    },
    {
        "library": "Python",
        "category": "Core Language & Control Flow",
        "concept": "Control flow, functions, arguments, lambdas",
        "url": "https://docs.python.org/3/tutorial/controlflow.html"
    },
    {
        "library": "Python",
        "category": "Object-Oriented Programming",
        "concept": "Classes, inheritance, methods, scopes",
        "url": "https://docs.python.org/3/tutorial/classes.html"
    },

    # --- 2. NumPy ---
    {
        "library": "NumPy",
        "category": "Array Computing",
        "concept": "ndarray, array creation, shape, broadcasting, operations",
        "url": "https://numpy.org/doc/stable/user/quickstart.html"
    },
    {
        "library": "NumPy",
        "category": "Array Computing",
        "concept": "Multidimensional indexing, slicing, boolean masks",
        "url": "https://numpy.org/doc/stable/user/basics.indexing.html"
    },

    # --- 3. Pandas ---
    {
        "library": "Pandas",
        "category": "Data Analysis & Manipulation",
        "concept": "10-minute tour: Series, DataFrame, indexing, operations",
        "url": "https://pandas.pydata.org/docs/user_guide/10min.html"
    },
    {
        "library": "Pandas",
        "category": "Data Analysis & Manipulation",
        "concept": "Series and DataFrame data structures, alignment, missing data",
        "url": "https://pandas.pydata.org/docs/user_guide/dsintro.html"
    },

    # --- 4. Matplotlib ---
    {
        "library": "Matplotlib",
        "category": "Data Visualization",
        "concept": "Figure, Axes, Artist architecture, object-oriented plotting",
        "url": "https://matplotlib.org/stable/users/explain/quick_start.html"
    },
    {
        "library": "Matplotlib",
        "category": "Data Visualization",
        "concept": "Pyplot interface, multiple figures, styling and labeling",
        "url": "https://matplotlib.org/stable/tutorials/pyplot.html"
    },

    # --- 5. Scikit-learn & Machine Learning Concepts ---
    {
        "library": "Scikit-learn",
        "category": "Supervised Learning",
        "concept": "Linear models, regression, classification, Ridge, Lasso regularization",
        "url": "https://scikit-learn.org/stable/modules/linear_model.html"
    },
    {
        "library": "Scikit-learn",
        "category": "ML Validation & Model Selection",
        "concept": "Train/val/test splits, K-Fold cross-validation, data leakage",
        "url": "https://scikit-learn.org/stable/modules/cross_validation.html"
    },
    {
        "library": "Scikit-learn",
        "category": "Hyperparameter Optimization",
        "concept": "GridSearchCV, RandomizedSearchCV, parameter spaces",
        "url": "https://scikit-learn.org/stable/modules/grid_search.html"
    },
    {
        "library": "Scikit-learn",
        "category": "Evaluation Metrics",
        "concept": "Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix",
        "url": "https://scikit-learn.org/stable/modules/model_evaluation.html"
    },
    {
        "library": "Scikit-learn",
        "category": "Feature Engineering & Preprocessing",
        "concept": "StandardScaler, MinMaxScaler, categorical encoding, feature scaling",
        "url": "https://scikit-learn.org/stable/modules/preprocessing.html"
    },
    {
        "library": "Scikit-learn",
        "category": "ML Theory & Diagnostics",
        "concept": "Overfitting, underfitting, bias-variance tradeoff, learning curves",
        "url": "https://scikit-learn.org/stable/modules/learning_curve.html"
    },
    {
        "library": "Scikit-learn",
        "category": "Unsupervised Learning",
        "concept": "Clustering, K-Means, DBSCAN, hierarchical clustering",
        "url": "https://scikit-learn.org/stable/modules/clustering.html"
    },

    # --- 6. PyTorch & Deep Learning Foundations ---
    {
        "library": "PyTorch",
        "category": "Deep Learning Tensors",
        "concept": "PyTorch tensors, creation, operations, GPU acceleration",
        "url": "https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html"
    },
    {
        "library": "PyTorch",
        "category": "Automatic Differentiation & Backprop",
        "concept": "Autograd, computational graphs, backpropagation, gradients",
        "url": "https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"
    },
    {
        "library": "PyTorch",
        "category": "Neural Network Architecture",
        "concept": "nn.Module, layers, activation functions (ReLU, Softmax)",
        "url": "https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html"
    },
    {
        "library": "PyTorch",
        "category": "Optimization & Training",
        "concept": "Loss functions, gradient descent, SGD, Adam, training loop",
        "url": "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html"
    },
    {
        "library": "PyTorch",
        "category": "Data Pipelines",
        "concept": "Dataset, DataLoader, batching, shuffling, parallel loading",
        "url": "https://pytorch.org/tutorials/beginner/basics/data_tutorial.html"
    },
    {
        "library": "PyTorch",
        "category": "Reinforcement Learning",
        "concept": "RL foundations: agent, environment, reward, Q-learning, DQN",
        "url": "https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"
    },

    # --- 7. TensorFlow & Advanced Deep Learning Architectures ---
    {
        "library": "TensorFlow",
        "category": "Deep Learning Fundamentals",
        "concept": "TensorFlow quickstart, model building, compilation, training",
        "url": "https://www.tensorflow.org/tutorials/quickstart/beginner"
    },
    {
        "library": "TensorFlow",
        "category": "Keras Model Architecture",
        "concept": "Keras Sequential model API, stacking layers, fit/evaluate",
        "url": "https://www.tensorflow.org/guide/keras/sequential_model"
    },
    {
        "library": "TensorFlow",
        "category": "Convolutional Neural Networks (CNNs)",
        "concept": "CNN architecture, Conv2D, MaxPooling2D, image feature extraction",
        "url": "https://www.tensorflow.org/tutorials/images/cnn"
    },
    {
        "library": "TensorFlow",
        "category": "Recurrent Neural Networks (RNNs)",
        "concept": "RNNs, LSTM, GRU, sequential data modeling, time-series",
        "url": "https://www.tensorflow.org/guide/keras/working_with_rnns"
    }
]


def extract_content_from_soup(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    """
    Locates the primary article container across various Sphinx, PyData,
    DevSite, and MkDocs documentation themes, removes chrome, and extracts
    structured text and code blocks.
    """
    # 1. Remove navigation, headers, footers, scripts, styles, forms, and anchor links
    for tag in soup.find_all(["header", "footer", "nav", "script", "style", "aside", "form"]):
        tag.decompose()
    for link in soup.find_all("a", class_=["headerlink", "anchor", "viewcode-link"]):
        link.decompose()

    # 2. Locate main documentation article container based on theme signatures
    container = (
        soup.find("article", class_=["bd-article", "pytorch-article", "devsite-article"]) or
        soup.find("div", class_=["devsite-article-body", "body", "document", "md-content"]) or
        soup.find("div", role="main") or
        soup.find("div", itemprop="articleBody") or
        soup.find("article") or
        soup.find("main") or
        soup.find("div", id="content")
    )
    if not container:
        container = soup.body

    # 3. Extract title
    h1 = container.find("h1") if container else soup.find("h1")
    if h1:
        title = h1.get_text(separator=" ", strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True).split("—")[0].split("-")[0].strip()
    else:
        title = "Technical Documentation"

    # 4. Extract blocks preserving code formatting
    blocks = []
    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li"]):
        # Skip nested elements already captured by parent pre or li
        if el.find_parent(["pre", "li"]):
            continue
        if el.name == "pre":
            code = el.get_text().strip()
            if code:
                blocks.append(f"```python\n{code}\n```")
        else:
            text = el.get_text(separator=" ", strip=True)
            if text and len(text) > 3:
                blocks.append(text)

    clean_text = "\n\n".join(blocks)
    return title, clean_text


def scrape_multilib_page(item: dict) -> dict:
    """
    Fetches and extracts a documentation page with error handling and metadata tags.
    """
    url = item["url"]
    headers = {
        "User-Agent": "DocuMind-MultiLib-Assistant/2.0 (Academic & Developer Study Project; Python/3.11)"
    }
    
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    title, clean_text = extract_content_from_soup(soup, url)

    if len(clean_text) < 250:
        raise ValueError(f"Extracted content too short ({len(clean_text)} chars) for {url}")

    return {
        "library": item["library"],
        "category": item["category"],
        "concept_focus": item["concept"],
        "title": title,
        "url": url,
        "status_code": response.status_code,
        "char_count": len(clean_text),
        "text": clean_text
    }


def main():
    print("=" * 80)
    print("DOCUMIND: MULTI-LIBRARY & ML/DL DOCUMENTATION COLLECTION")
    print("=" * 80)
    print(f"Target Libraries : Python, NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, TensorFlow")
    print(f"Planned Sources  : {len(DOCUMENTATION_SOURCES)} curated official guides\n")

    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    output_file = os.path.join("data", "raw", "multilib_docs.json")

    collected_docs = []
    seen_urls = set()
    failed_sources = []

    for idx, item in enumerate(DOCUMENTATION_SOURCES, 1):
        url = item["url"]
        lib = item["library"]
        concept = item["concept_focus" if "concept_focus" in item else "concept"]

        if url in seen_urls:
            print(f"[{idx:02d}/{len(DOCUMENTATION_SOURCES):02d}] Skipping duplicate URL: {url}")
            continue

        print(f"[{idx:02d}/{len(DOCUMENTATION_SOURCES):02d}] Fetching [{lib}] {concept[:42]:<42} ... ", end="", flush=True)

        try:
            doc_data = scrape_multilib_page(item)
            collected_docs.append(doc_data)
            seen_urls.add(url)
            print(f"SUCCESS ({doc_data['char_count']:,} chars | Title: \"{doc_data['title'][:30]}\")")
        except Exception as e:
            print(f"FAILED ({str(e)})")
            failed_sources.append({"item": item, "error": str(e)})

        # Polite delay between requests
        time.sleep(1)

    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY & VERIFICATION")
    print("=" * 80)
    print(f"Total Sources Attempted : {len(DOCUMENTATION_SOURCES)}")
    print(f"Successfully Collected  : {len(collected_docs)}")
    print(f"Failed Sources          : {len(failed_sources)}")

    if failed_sources:
        print("\nFailures:")
        for f in failed_sources:
            print(f" - [{f['item']['library']}] {f['item']['url']}: {f['error']}")

    # Save to disk
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(collected_docs, f, indent=2, ensure_ascii=False)

    print(f"\nPersisted {len(collected_docs)} documents to: {output_file}")
    print(f"Total Characters Collected: {sum(d['char_count'] for d in collected_docs):,} chars")

    # Breakdown by library
    print("\nBreakdown by Library:")
    lib_counts = {}
    for d in collected_docs:
        lib = d["library"]
        lib_counts[lib] = lib_counts.get(lib, 0) + 1
    for lib, count in sorted(lib_counts.items()):
        chars = sum(d["char_count"] for d in collected_docs if d["library"] == lib)
        print(f"  - {lib:<15}: {count:2d} pages ({chars:,} characters)")

    # Breakdown by concept category
    print("\nBreakdown by Concept Category:")
    cat_counts = {}
    for d in collected_docs:
        cat = d["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  - {cat:<40}: {count} page(s)")

    print("=" * 80)


if __name__ == "__main__":
    main()
