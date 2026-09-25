#!/bin/bash
# Runs once before the Streamlit app starts on Streamlit Cloud.
# Trains and saves the model if it doesn't already exist.

echo "=== Running setup.sh ==="

# Download NLTK data
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
print('NLTK data downloaded.')
"

# Train and save model
python src/train_and_save.py

echo "=== Setup complete ==="
