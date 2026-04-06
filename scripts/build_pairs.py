from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from backend.nlp import _normalize_labels, _resolve_label_column, _resolve_text_column
from backend.utils import clean_email_text

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _gather_images(root: Path):
    items = {0: [], 1: []}
    for label_name, label_value in [("legit", 0), ("phishing", 1)]:
        folder = root / label_name
        if not folder.exists():
            continue
        for path in folder.glob("*"):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            items[label_value].append(path)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pairs.csv linking emails to screenshot images for combined eval."
    )
    parser.add_argument("--emails", required=True, help="CSV with email text + label")
    parser.add_argument("--text-col", default=None, help="Email text column name")
    parser.add_argument("--label-col", default=None, help="Label column name")
    parser.add_argument("--screenshots", default="data/screenshots")
    parser.add_argument("--out", default="data/pairs.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap for each class (defaults to number of screenshots).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    df = pd.read_csv(args.emails)
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

    labels = _normalize_labels(df[label_col])
    cleaned_texts = texts.apply(clean_email_text).tolist()

    email_by_label = {0: [], 1: []}
    for text, label in zip(cleaned_texts, labels):
        email_by_label[int(label)].append(text)

    screenshots = _gather_images(Path(args.screenshots))
    if not screenshots[0] or not screenshots[1]:
        raise ValueError("Screenshots folder must contain legit/ and phishing/ images.")

    rows = []
    for label_value in [0, 1]:
        images = screenshots[label_value]
        if args.max_per_class:
            images = images[: args.max_per_class]

        emails = email_by_label[label_value]
        if not emails:
            raise ValueError(f"No emails found for label {label_value}.")

        # Sample emails with replacement if needed
        for img_path in images:
            text = random.choice(emails)
            rows.append(
                {
                    "email_text": text,
                    "label": label_value,
                    "screenshot_path": img_path.as_posix(),
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved pairs to: {out_path.resolve()}")
    print(f"Total pairs: {len(rows)}")


if __name__ == "__main__":
    main()
