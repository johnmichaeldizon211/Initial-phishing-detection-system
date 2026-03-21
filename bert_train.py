from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from backend.nlp import _normalize_labels, _resolve_label_column, _resolve_text_column
from backend.utils import clean_email_text, ensure_dir


class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BERT-tiny email classifier.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset.")
    parser.add_argument("--text-col", default=None, help="Column name for email text.")
    parser.add_argument("--label-col", default=None, help="Column name for labels.")
    parser.add_argument("--model-name", default="prajjwal1/bert-tiny")
    parser.add_argument("--model-out", default="models/bert_tiny")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
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

    texts = texts.apply(clean_email_text).tolist()
    labels = _normalize_labels(df[label_col]).tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    # Use slow tokenizer to avoid fast-tokenizer backend issues on some setups.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    train_enc = tokenizer(
        x_train,
        truncation=True,
        max_length=args.max_length,
    )
    test_enc = tokenizer(
        x_test,
        truncation=True,
        max_length=args.max_length,
    )

    train_ds = EmailDataset(train_enc, y_train)
    test_ds = EmailDataset(test_enc, y_test)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        probs = _softmax(logits)[:, 1]
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(labels_np, preds)
        precision, recall, _, _ = precision_recall_fscore_support(
            labels_np, preds, average="binary", zero_division=0
        )
        auc = roc_auc_score(labels_np, probs)
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": auc,
        }

    output_dir = Path(args.model_out)
    ensure_dir(output_dir)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    preds_output = trainer.predict(test_ds)
    probs = _softmax(preds_output.predictions)[:, 1]
    preds = (probs >= 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    report_payload = {
        "model_name": args.model_name,
        "f1": float(metrics.get("eval_f1", 0.0)),
        "roc_auc": float(metrics.get("eval_roc_auc", 0.0)),
        "precision": float(metrics.get("eval_precision", 0.0)),
        "recall": float(metrics.get("eval_recall", 0.0)),
        "classification_report": report,
        "samples": len(labels),
        "text_col": text_col,
        "label_col": label_col,
    }

    (output_dir / "bert_report.json").write_text(json.dumps(report_payload, indent=2))

    print("BERT-tiny training complete.")
    print(
        f"F1: {report_payload['f1']:.3f} | ROC-AUC: {report_payload['roc_auc']:.3f}"
    )
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
