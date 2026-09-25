"""
Streamlit web app for Fake News Detection.
Run locally:  streamlit run app.py
"""

import os
import sys
import pickle

import streamlit as st

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import clean_text

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
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

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown("Paste any news article or headline below and find out if it's **Fake** or **Real**.")
st.markdown("---")

# ── Load model ─────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
VEC_PATH   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
CLF_PATH   = os.path.join(MODELS_DIR, "logistic_regression.pkl")


@st.cache_resource
def load_model():
    """Load and cache the trained model artifacts."""
    if not os.path.exists(VEC_PATH) or not os.path.exists(CLF_PATH):
        return None, None
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)
    return vectorizer, clf


vectorizer, clf = load_model()

if vectorizer is None:
    st.error(
        "⚠️ Trained model not found. "
        "Please run `python src/train_and_save.py` to generate the model files."
    )
    st.stop()

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

        # Probability bar chart
        st.markdown("### Confidence Breakdown")
        st.progress(int(fake_prob), text=f"🔴 Fake: {fake_prob:.1f}%")
        st.progress(int(real_prob), text=f"🟢 Real: {real_prob:.1f}%")

        # Show preprocessing details
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
    - Python
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
