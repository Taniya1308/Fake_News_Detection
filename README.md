# Fake News Detection (NLP)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fakenewsdetection-4bp3frygkgh3fweaipkafg.streamlit.app/)

🔗 **Live Demo:** [https://fakenewsdetection-4bp3frygkgh3fweaipkafg.streamlit.app/](https://fakenewsdetection-4bp3frygkgh3fweaipkafg.streamlit.app/)

An NLP classification system that detects fake news articles using **TF-IDF** feature extraction and **Logistic Regression**, achieving ~90% accuracy on the Fake and Real News dataset.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| NLTK | Tokenization, stopword removal |
| Scikit-learn | TF-IDF, Logistic Regression, evaluation |
| Pandas / NumPy | Data handling |
| Matplotlib | Confusion matrix visualization |

---

## Project Structure

```
Fake_News_Detection/
├── data/
│   ├── download_data.py        # Kaggle dataset downloader
│   └── (Fake.csv / True.csv)   # Place dataset files here
├── models/
│   ├── tfidf_vectorizer.pkl    # Saved vectorizer (after training)
│   ├── logistic_regression.pkl # Saved classifier (after training)
│   └── confusion_matrix.png    # Evaluation plot (after evaluation)
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Text cleaning & normalization
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Metrics & confusion matrix
│   ├── predict.py              # CLI / interactive predictor
│   └── generate_sample_data.py # Demo dataset generator
├── main.py                     # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Fake_News_Detection.git
cd Fake_News_Detection

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Getting the Dataset

**Option A — Kaggle (full dataset, ~45 000 articles):**
1. Create a [Kaggle account](https://www.kaggle.com) and generate an API key (`kaggle.json`).
2. Place `kaggle.json` in `~/.kaggle/`.
3. Run:
   ```bash
   python data/download_data.py
   ```

**Option B — Manual download:**
Download [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place `Fake.csv` and `True.csv` inside the `data/` directory.

**Option C — Sample data (demo, no Kaggle needed):**
```bash
python src/generate_sample_data.py
# or
python main.py --sample
```

---

## Usage

### Full pipeline (train + evaluate)
```bash
python main.py
```

### With auto-generated sample data
```bash
python main.py --sample
```

### Train only
```bash
python src/train.py
```

### Evaluate only
```bash
python src/evaluate.py
```

### Interactive prediction
```bash
python main.py --predict
# or with a specific text
python src/predict.py --text "Scientists say Earth is flat!"
```

---

## How It Works

### 1. Text Preprocessing (`src/preprocess.py`)
- Lowercasing
- URL and HTML tag removal
- Punctuation and digit stripping
- **Tokenization** — NLTK `word_tokenize`
- **Stopword removal** — NLTK English stopwords
- **Stemming** — Porter Stemmer

### 2. Feature Extraction
- **TF-IDF** (Term Frequency–Inverse Document Frequency)
  - Unigrams + bigrams (`ngram_range=(1, 2)`)
  - 50 000 maximum features
  - Sublinear TF scaling

### 3. Classification
- **Logistic Regression** (`C=5.0`, `max_iter=1000`, L-BFGS solver)
- 80/20 stratified train-test split

### 4. Evaluation
- Accuracy score
- Precision, Recall, F1-score (per class)
- Confusion matrix (saved as `models/confusion_matrix.png`)

---

## Results (Full Dataset)

| Metric | FAKE | REAL |
|--------|------|------|
| Precision | ~0.99 | ~0.99 |
| Recall | ~0.99 | ~0.99 |
| F1-score | ~0.99 | ~0.99 |
| **Overall Accuracy** | **~99%** | |

> On the sample/demo dataset the model will achieve lower accuracy (~70–80%) due to the small synthetic corpus.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
