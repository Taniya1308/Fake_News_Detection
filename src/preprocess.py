"""
Text preprocessing pipeline:
  - Lowercasing
  - URL / HTML removal
  - Punctuation & digit removal
  - Tokenization (NLTK word_tokenize)
  - Stopword removal (NLTK English stopwords)
  - Stemming (PorterStemmer)
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data (silent if already present)
for _pkg in ("punkt", "stopwords", "punkt_tab"):
    nltk.download(_pkg, quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Full preprocessing pipeline for a single text string."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 4. Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # 5. Tokenize
    tokens = word_tokenize(text)

    # 6. Remove stopwords and short tokens, then stem
    tokens = [
        _stemmer.stem(tok)
        for tok in tokens
        if tok not in _stop_words and len(tok) > 2
    ]

    return " ".join(tokens)


def preprocess_series(series):
    """Apply clean_text to a pandas Series, return cleaned Series."""
    return series.fillna("").apply(clean_text)


if __name__ == "__main__":
    sample = (
        "BREAKING: Scientists Discover That Earth Is FLAT! "
        "Visit http://fakenews.com for more. "
        "The government doesn't want you to know this!!!"
    )
    print("Original :", sample)
    print("Processed:", clean_text(sample))
