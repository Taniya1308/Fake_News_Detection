"""
Streamlit web app for Fake News Detection.
Run locally:  streamlit run app.py
"""

import os
import sys
import pickle
import random

import nltk
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import clean_text, preprocess_series

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stTextArea textarea { font-size: 15px; }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .fake { background-color: #ffe0e0; color: #c0392b; border: 2px solid #c0392b; }
    .real { background-color: #e0f7e9; color: #1e8449; border: 2px solid #1e8449; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
VEC_PATH   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
CLF_PATH   = os.path.join(MODELS_DIR, "logistic_regression.pkl")
FAKE_CSV   = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV   = os.path.join(DATA_DIR, "True.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ── Sample data generator ──────────────────────────────────────────────────────
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


# ── Train model ────────────────────────────────────────────────────────────────
def train_model():
    # Download NLTK data
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)

    # Generate sample data if real dataset absent
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        generate_sample_data()

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)
    fake_df["label"] = 0
    true_df["label"] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["content"] = df.get("title", pd.Series(dtype=str)).fillna("") + " " + \
                    df.get("text",  pd.Series(dtype=str)).fillna("")

    df["content_clean"] = preprocess_series(df["content"])

    X_train, _, y_train, _ = train_test_split(
        df["content_clean"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)

    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(CLF_PATH, "wb") as f:
        pickle.dump(clf, f)

    return vectorizer, clf


# ── Load or train model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    # Download NLTK data first
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)

    if os.path.exists(VEC_PATH) and os.path.exists(CLF_PATH):
        with open(VEC_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(CLF_PATH, "rb") as f:
            clf = pickle.load(f)
        return vectorizer, clf

    # Train from scratch
    return train_model()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown("Paste any news article or headline below and find out if it's **Fake** or **Real**.")
st.markdown("---")

# ── Load model with progress indicator ────────────────────────────────────────
with st.spinner("⏳ Loading model... (first load may take ~30 seconds)"):
    vectorizer, clf = get_model()

# ── Input ──────────────────────────────────────────────────────────────────────
text_input = st.text_area(
    "📰 Enter news text here:",
    height=200,
    placeholder="e.g. Scientists have discovered that the Earth is flat according to a new government report..."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔎 Analyze", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    if not text_input.strip():
        st.warning("Please enter some news text first.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(text_input)
            vec     = vectorizer.transform([cleaned])
            label   = clf.predict(vec)[0]
            proba   = clf.predict_proba(vec)[0]

            fake_prob = proba[0] * 100
            real_prob = proba[1] * 100

        if label == 0:
            st.markdown(
                f'<div class="result-box fake">🚨 FAKE NEWS &nbsp;|&nbsp; {fake_prob:.1f}% confidence</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box real">✅ REAL NEWS &nbsp;|&nbsp; {real_prob:.1f}% confidence</div>',
                unsafe_allow_html=True
            )

        st.markdown("### Confidence Breakdown")
        st.progress(int(fake_prob), text=f"🔴 Fake: {fake_prob:.1f}%")
        st.progress(int(real_prob), text=f"🟢 Real: {real_prob:.1f}%")

        with st.expander("🔬 See preprocessed text"):
            st.code(cleaned if cleaned else "(empty after preprocessing)", language="text")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Fake News Detector** uses Natural Language Processing to classify news as fake or real.

    **How it works:**
    1. Text is cleaned & normalized
    2. TF-IDF converts text to features
    3. Logistic Regression predicts the label

    **Tech Stack:**
    - Python 3.11
    - NLTK
    - Scikit-learn
    - Streamlit

    **Model Accuracy:** ~90%+
    """)
    st.markdown("---")
    st.markdown("Built by [Taniya1308](https://github.com/Taniya1308)")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px;'>"
    "Built with ❤️ using Python, NLTK, Scikit-learn & Streamlit"
    "</p>",
    unsafe_allow_html=True
)
