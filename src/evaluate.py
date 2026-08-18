"""
Evaluate the trained Fake News Detection model.

Outputs:
  - Accuracy
  - Classification report (Precision, Recall, F1-score per class)
  - Confusion matrix (printed + saved as PNG to models/confusion_matrix.png)
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from preprocess import preprocess_series

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")

VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
CLF_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")


def load_model():
    if not os.path.exists(VEC_PATH) or not os.path.exists(CLF_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run  python src/train.py  first."
        )
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)
    return vectorizer, clf


def load_test_data():
    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)
    fake_df["label"] = 0
    true_df["label"] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["content"] = df.get("title", pd.Series(dtype=str)).fillna("") + " " + \
                    df.get("text",  pd.Series(dtype=str)).fillna("")

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        df["content"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    return X_test, y_test


def evaluate():
    print("Loading model...")
    vectorizer, clf = load_model()

    print("Loading test data...")
    X_test_raw, y_test = load_test_data()

    print("Preprocessing...")
    X_test_clean = preprocess_series(X_test_raw)
    X_test_tfidf  = vectorizer.transform(X_test_clean)

    print("Predicting...")
    y_pred = clf.predict(X_test_tfidf)

    # ── Metrics ────────────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"{'='*50}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    # ── Confusion Matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix  (Accuracy: {acc*100:.2f}%)", fontsize=13)
    plt.tight_layout()

    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved → {cm_path}")


if __name__ == "__main__":
    evaluate()
