"""
Trains the model using the Kaggle dataset (if available) or the sample dataset,
and saves the artifacts to models/.

This script is called automatically on Streamlit Cloud via setup.sh.
Run manually:  python src/train_and_save.py
"""

import os
import sys
import pickle
import random

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_series

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
CLF_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")


# ── Sample data generator (fallback) ──────────────────────────────────────────
def generate_sample_data():
    random.seed(42)

    FAKE_TITLES = [
        "SHOCKING: {topic} EXPOSED by insider leak!!!",
        "You won't BELIEVE what {person} just admitted about {topic}",
        "BREAKING: Government hides truth about {topic} — wake up sheeple!",
        "Secret documents PROVE {person} controls {topic}",
        "{topic} is a HOAX designed to control the population",
        "Scientists BAFFLED by {topic} — mainstream media silent",
        "EXCLUSIVE: {person} caught lying about {topic} on live TV",
        "The {topic} agenda: what they don't want you to know",
        "BOMBSHELL: {person} admits {topic} was fake all along",
        "They are HIDING the truth about {topic} from you!!!",
    ]
    FAKE_BODIES = [
        "Sources close to the situation confirmed that {topic} has been manipulated "
        "for decades. {person} was seen at a secret meeting last week. Share before "
        "they delete this!!! The mainstream media refuses to cover this story.",
        "A whistleblower has come forward with damning evidence that {topic} is "
        "entirely fabricated. {person} denies involvement but leaked emails say "
        "otherwise. The truth is out there and they cannot hide it forever.",
        "Independent researchers have uncovered a massive cover-up involving {topic}. "
        "{person} is at the center of the conspiracy. Big media refuses to report "
        "this story because they are all part of the system.",
        "This shocking revelation about {topic} will change everything you know. "
        "{person} has been lying to the public for years. Wake up and share this "
        "before the government takes it down!!!",
    ]
    REAL_TITLES = [
        "Study finds new evidence linking {topic} to public health outcomes",
        "{person} announces policy changes on {topic} at press conference",
        "Experts weigh in on the latest developments in {topic}",
        "Report: {topic} shows significant improvement over previous quarter",
        "{person} and world leaders meet to discuss {topic} challenges",
        "Research team publishes peer-reviewed findings on {topic}",
        "Government releases official data on {topic} amid growing concerns",
        "Analysis: How {topic} is shaping the global economy this year",
        "New legislation proposed to address {topic} concerns",
        "{person} outlines five-year strategy for improving {topic}",
    ]
    REAL_BODIES = [
        "According to a peer-reviewed study published in the journal Nature, {topic} "
        "has measurable effects on communities worldwide. {person} commented that "
        "further research is needed to fully understand the long-term implications.",
        "At a press conference on Monday, {person} outlined new initiatives aimed at "
        "addressing {topic}. Officials say the policy is backed by extensive data "
        "collected over the past five years by independent researchers.",
        "Data released by the Department of Statistics shows that {topic} indicators "
        "have improved by 12% compared to last year. {person} called the results "
        "encouraging while cautioning against complacency in future planning.",
        "The latest quarterly report on {topic} confirms steady progress across key "
        "metrics. {person} stated that collaborative efforts between government and "
        "private sector have been instrumental in achieving these results.",
    ]
    TOPICS  = ["climate change", "vaccines", "election results", "economic policy",
               "immigration", "healthcare reform", "artificial intelligence",
               "space exploration", "renewable energy", "cybersecurity"]
    PERSONS = ["President Smith", "Senator Johnson", "Dr. Williams", "CEO Brown",
               "Mayor Davis", "Governor Wilson", "Secretary Lee", "Director Taylor",
               "Professor Chen", "Minister Patel"]

    def make_rows(titles, bodies, n):
        rows = []
        for _ in range(n):
            title = random.choice(titles).format(
                topic=random.choice(TOPICS), person=random.choice(PERSONS))
            text = random.choice(bodies).format(
                topic=random.choice(TOPICS), person=random.choice(PERSONS))
            rows.append({"title": title, "text": text})
        return rows

    fake_df = pd.DataFrame(make_rows(FAKE_TITLES, FAKE_BODIES, 500))
    true_df = pd.DataFrame(make_rows(REAL_TITLES, REAL_BODIES, 500))
    fake_df.to_csv(FAKE_CSV, index=False)
    true_df.to_csv(TRUE_CSV, index=False)
    print(f"Sample data generated: 500 fake + 500 real articles.")


# ── Load dataset ───────────────────────────────────────────────────────────────
def load_dataset():
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        print("Dataset not found. Generating sample data...")
        generate_sample_data()

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)
    fake_df["label"] = 0
    true_df["label"] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["content"] = df.get("title", pd.Series(dtype=str)).fillna("") + " " + \
                    df.get("text",  pd.Series(dtype=str)).fillna("")
    return df[["content", "label"]]


# ── Train & save ───────────────────────────────────────────────────────────────
def train_and_save():
    # Skip if models already exist
    if os.path.exists(VEC_PATH) and os.path.exists(CLF_PATH):
        print("Models already exist. Skipping training.")
        return

    print("Loading dataset...")
    df = load_dataset()
    print(f"  Total samples: {len(df)}  (Fake: {(df['label']==0).sum()}, Real: {(df['label']==1).sum()})")

    print("Preprocessing...")
    df["content_clean"] = preprocess_series(df["content"])

    X_train, _, y_train, _ = train_test_split(
        df["content_clean"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)

    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(CLF_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    train_and_save()
