"""
Script to download the Fake and Real News Dataset from Kaggle.

Dataset: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
  - Fake.csv  : fake news articles
  - True.csv  : real news articles

Instructions:
  1. Install kaggle CLI:  pip install kaggle
  2. Place your kaggle.json API key in ~/.kaggle/kaggle.json
  3. Run: python data/download_data.py

Alternatively, download manually from Kaggle and place Fake.csv and True.csv
inside the data/ directory.
"""

import subprocess
import sys
import os

DATASET = "clmentbisaillon/fake-and-real-news-dataset"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("kaggle package not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

    print(f"Downloading dataset '{DATASET}' into {DATA_DIR} ...")
    subprocess.check_call(
        ["kaggle", "datasets", "download", "-d", DATASET, "--unzip", "-p", DATA_DIR]
    )
    print("Download complete.")
    print(f"Files in {DATA_DIR}:")
    for f in os.listdir(DATA_DIR):
        print(f"  {f}")


if __name__ == "__main__":
    download()
