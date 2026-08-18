"""
Interactive / CLI prediction script.

Usage:
  python src/predict.py
  python src/predict.py --text "Scientists confirm Earth is flat!"
"""

import argparse
import os
import pickle

from preprocess import clean_text

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
CLF_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")

LABEL_MAP = {0: "FAKE", 1: "REAL"}


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


def predict(text: str, vectorizer, clf) -> dict:
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    label   = clf.predict(vec)[0]
    proba   = clf.predict_proba(vec)[0]
    return {
        "prediction" : LABEL_MAP[label],
        "confidence" : f"{max(proba) * 100:.1f}%",
        "fake_prob"  : f"{proba[0] * 100:.1f}%",
        "real_prob"  : f"{proba[1] * 100:.1f}%",
    }


def interactive(vectorizer, clf):
    print("\nFake News Detector — type 'quit' to exit\n")
    while True:
        text = input("Enter news text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        result = predict(text, vectorizer, clf)
        print(f"  Prediction : {result['prediction']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  FAKE prob  : {result['fake_prob']}  |  REAL prob: {result['real_prob']}\n")


def main():
    parser = argparse.ArgumentParser(description="Predict fake vs real news.")
    parser.add_argument("--text", type=str, default=None, help="News text to classify.")
    args = parser.parse_args()

    vectorizer, clf = load_model()

    if args.text:
        result = predict(args.text, vectorizer, clf)
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']}")
        print(f"FAKE prob  : {result['fake_prob']}  |  REAL prob: {result['real_prob']}")
    else:
        interactive(vectorizer, clf)


if __name__ == "__main__":
    main()
