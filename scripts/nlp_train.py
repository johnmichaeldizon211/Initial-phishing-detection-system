from __future__ import annotations

import argparse
from pathlib import Path

from backend.nlp import train_nlp_model


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression email model.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset.")
    parser.add_argument("--model-dir", default="models", help="Output directory for model files.")
    parser.add_argument("--text-col", default=None, help="Column name for email text.")
    parser.add_argument("--label-col", default=None, help="Column name for labels.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    args = parser.parse_args()

    info = train_nlp_model(
        data_path=Path(args.data),
        model_dir=Path(args.model_dir),
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
    )

    print("NLP training complete.")
    print(f"F1: {info['f1']:.3f} | ROC-AUC: {info['roc_auc']:.3f}")
    print(f"Model saved to: {Path(args.model_dir).resolve()}")


if __name__ == "__main__":
    main()
