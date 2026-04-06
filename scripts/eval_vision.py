from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from backend.vision import load_vision_model, predict_vision_proba
from backend.utils import ensure_dir

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _gather_images(data_dir: Path):
    items = []
    for label_name, label_value in [("phishing", 1), ("legit", 0)]:
        label_dir = data_dir / label_name
        if not label_dir.exists():
            continue
        for path in label_dir.glob("*"):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            items.append((path, label_value))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vision model on screenshot dataset.")
    parser.add_argument("--data-dir", required=True, help="Directory with phishing/legit folders.")
    parser.add_argument("--model-path", default="models/vision_model.keras")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-misclassified", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    items = _gather_images(data_dir)
    if not items:
        raise ValueError("No images found in phishing/legit folders.")

    model = load_vision_model(args.model_path)

    labels = []
    probs = []
    paths = []

    for path, label in items:
        prob = predict_vision_proba(path, model)
        labels.append(label)
        probs.append(prob)
        paths.append(str(path))

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
        "samples": int(len(labels)),
        "model_path": args.model_path,
    }
    (out_dir / "vision_eval.json").write_text(json.dumps(payload, indent=2))

    pd.DataFrame(
        cm,
        index=["true_legit", "true_phish"],
        columns=["pred_legit", "pred_phish"],
    ).to_csv(out_dir / "vision_confusion.csv", index=True)

    df = pd.DataFrame(
        {"path": paths, "label": labels, "pred_prob": probs, "pred_label": preds}
    )
    mis = df[df["label"] != df["pred_label"]].head(args.max_misclassified)
    if not mis.empty:
        mis.to_csv(out_dir / "vision_misclassified.csv", index=False)

    print("Vision evaluation complete.")
    print(f"F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
