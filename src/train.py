"""
Train a Fake News Detection model:
  - TF-IDF vectorizer  (unigrams + bigrams, max 50 000 features)
  - Logistic Regression classifier

Saves trained artifacts to models/:
  - models/tfidf_vectorizer.pkl
  - models/logistic_regression.pkl
"""

import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import preprocess_series

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")


def load_dataset() -> pd.DataFrame:
    """Load Fake.csv + True.csv, assign labels, return combined DataFrame."""
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        raise FileNotFoundError(
            "Dataset files not found.\n"
            "Run  python data/download_data.py  to download,\n"
            "or place Fake.csv and True.csv in the data/ directory.\n"
            "You can also run  python src/generate_sample_data.py  to create "
            "a small demo dataset."
        )

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    fake_df["label"] = 0   # 0 = FAKE
    true_df["label"] = 1   # 1 = REAL

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text for richer signal
    df["content"] = df.get("title", pd.Series(dtype=str)).fillna("") + " " + \
                    df.get("text",  pd.Series(dtype=str)).fillna("")
    return df[["content", "label"]]


def train(test_size: float = 0.2, random_state: int = 42):
    print("Loading dataset...")
    df = load_dataset()
    print(f"  Total samples : {len(df)}")
    print(f"  Fake (0)      : {(df['label'] == 0).sum()}")
    print(f"  Real (1)      : {(df['label'] == 1).sum()}")

    print("\nPreprocessing text...")
    df["content_clean"] = preprocess_series(df["content"])

    X = df["content_clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train / Test  : {len(X_train)} / {len(X_test)}")

    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    clf_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"\nModels saved to {MODELS_DIR}/")
    return vectorizer, clf, X_test_tfidf, y_test


if __name__ == "__main__":
    train()
    print("\nTraining complete. Run  python src/evaluate.py  to see metrics.")
