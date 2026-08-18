"""
main.py — End-to-end runner for the Fake News Detection pipeline.

Steps:
  1. (Optional) Generate sample data if real dataset is absent
  2. Train the model
  3. Evaluate and print metrics

Usage:
  python main.py                # full pipeline
  python main.py --sample       # generate sample data first, then run pipeline
  python main.py --predict      # launch interactive prediction (model must exist)
"""

import argparse
import os
import sys

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(description="Fake News Detection Pipeline")
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate a small sample dataset before training (for demo purposes)."
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Launch interactive prediction mode (requires a trained model)."
    )
    args = parser.parse_args()

    if args.predict:
        from predict import interactive, load_model
        vectorizer, clf = load_model()
        interactive(vectorizer, clf)
        return

    if args.sample:
        print("="*60)
        print("Step 0: Generating sample dataset...")
        print("="*60)
        import generate_sample_data  # noqa: F401 — runs on import

    print("="*60)
    print("Step 1: Training the model...")
    print("="*60)
    from train import train
    train()

    print("\n" + "="*60)
    print("Step 2: Evaluating the model...")
    print("="*60)
    from evaluate import evaluate
    evaluate()

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("  Models saved in  models/")
    print("  Run  python main.py --predict  to classify your own text.")
    print("="*60)


if __name__ == "__main__":
    main()
