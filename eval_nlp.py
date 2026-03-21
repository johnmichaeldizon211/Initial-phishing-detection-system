from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from backend.nlp import (
    _normalize_labels,
    _resolve_label_column,
    _resolve_text_column,
    load_nlp_model,
)
from backend.utils import clean_email_text, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NLP model on a dataset.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset.")
    parser.add_argument("--model-dir", default="models", help="Directory with saved NLP model.")
    parser.add_argument("--text-col", default=None, help="Column name for email text.")
    parser.add_argument("--label-col", default=None, help="Column name for labels.")
    parser.add_argument("--out-dir", default="outputs", help="Directory for evaluation outputs.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-misclassified", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    text_col = _resolve_text_column(df, args.text_col)
    label_col = _resolve_label_column(df, args.label_col)

    if text_col == "__subject_body__":
        texts = (
            df["subject"].fillna("").astype(str)
            + " "
            + df["body"].fillna("").astype(str)
        )
    else:
        texts = df[text_col].fillna("").astype(str)

    texts = texts.apply(clean_email_text)
    labels = _normalize_labels(df[label_col])

    vectorizer, classifier = load_nlp_model(args.model_dir)
    vec = vectorizer.transform(texts)
    probs = classifier.predict_proba(vec)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    report = classification_report(labels, preds, output_dict=True)
    cm = confusion_matrix(labels, preds)

    out_dir = ensure_dir(args.out_dir)
    eval_payload = {
        "f1": float(f1),
        "roc_auc": float(auc),
        "threshold": args.threshold,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "samples": int(len(labels)),
        "text_col": text_col,
        "label_col": label_col,
    }
    (out_dir / "nlp_eval.json").write_text(json.dumps(eval_payload, indent=2))

    pd.DataFrame(
        cm,
        index=["true_legit", "true_phish"],
        columns=["pred_legit", "pred_phish"],
    ).to_csv(out_dir / "nlp_confusion.csv", index=True)

    mis_idx = (preds != labels).nonzero()[0]
    if mis_idx.size > 0:
        subset = df.iloc[mis_idx].copy()
        subset["pred_prob"] = probs[mis_idx]
        subset["pred_label"] = preds[mis_idx]
        subset = subset.head(args.max_misclassified)
        subset.to_csv(out_dir / "nlp_misclassified.csv", index=False)

    print("NLP evaluation complete.")
    print(f"F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
