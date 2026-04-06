from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from backend.combine import combine_scores
from backend.nlp import load_nlp_model, predict_nlp_proba
from backend.vision import load_vision_model, predict_vision_proba
from backend.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate combined NLP+Vision scores on paired samples."
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="CSV with columns: email_text,label,screenshot_path",
    )
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--vision-model", default="models/vision_model.keras")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weight-nlp", type=float, default=0.5)
    parser.add_argument("--weight-vision", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.pairs)
    required = {"email_text", "label", "screenshot_path"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in pairs CSV: {missing}")

    vectorizer, classifier = load_nlp_model(args.model_dir)
    vision_model = load_vision_model(args.vision_model)

    labels = []
    probs = []

    for _, row in df.iterrows():
        label = int(row["label"])
        text = str(row["email_text"])
        screenshot = Path(row["screenshot_path"])

        nlp_score = predict_nlp_proba(text, vectorizer, classifier)
        vision_score = predict_vision_proba(screenshot, vision_model)
        combined = combine_scores(
            nlp_score, vision_score, args.weight_nlp, args.weight_vision
        )

        labels.append(label)
        probs.append(combined)

    preds = [1 if p >= args.threshold else 0 for p in probs]
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)

    out_dir = ensure_dir(args.out_dir)
    payload = {
        "f1": float(f1),
        "roc_auc": float(auc),
        "threshold": args.threshold,
        "confusion_matrix": cm.tolist(),
        "weights": {"nlp": args.weight_nlp, "vision": args.weight_vision},
        "samples": int(len(labels)),
    }
    (out_dir / "combined_eval.json").write_text(json.dumps(payload, indent=2))

    pd.DataFrame(
        cm,
        index=["true_legit", "true_phish"],
        columns=["pred_legit", "pred_phish"],
    ).to_csv(out_dir / "combined_confusion.csv", index=True)

    print("Combined evaluation complete.")
    print(f"F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
