from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .utils import clean_email_text, ensure_dir

LABEL_MAP = {
    "phishing": 1,
    "phishing email": 1,
    "phish": 1,
    "malicious": 1,
    "spam": 1,
    "legit": 0,
    "legitimate": 0,
    "safe": 0,
    "safe email": 0,
    "ham": 0,
    "benign": 0,
}


def _resolve_text_column(df: pd.DataFrame, text_col: str | None) -> str:
    if text_col and text_col in df.columns:
        return text_col

    # Common columns found in phishing email datasets.
    candidates = [
        "text",
        "email_text",
        "body",
        "content",
        "message",
        "email",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    if "subject" in df.columns and "body" in df.columns:
        return "__subject_body__"

    raise ValueError(
        "Could not find a text column. Use --text-col to specify it explicitly."
    )


def _resolve_label_column(df: pd.DataFrame, label_col: str | None) -> str:
    if label_col and label_col in df.columns:
        return label_col

    candidates = [
        "label",
        "is_phishing",
        "target",
        "class",
        "category",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "Could not find a label column. Use --label-col to specify it explicitly."
    )


def _normalize_labels(series: Iterable) -> np.ndarray:
    normalized = []
    for item in series:
        if isinstance(item, str):
            key = item.strip().lower()
            if key in LABEL_MAP:
                normalized.append(LABEL_MAP[key])
                continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            value = 0
        normalized.append(1 if value == 1 else 0)
    return np.array(normalized, dtype=np.int32)


def train_nlp_model(
    data_path: str | Path,
    model_dir: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = pd.read_csv(data_path)
    resolved_text_col = _resolve_text_column(df, text_col)
    resolved_label_col = _resolve_label_column(df, label_col)

    if resolved_text_col == "__subject_body__":
        combined = (
            df["subject"].fillna("").astype(str)
            + " "
            + df["body"].fillna("").astype(str)
        )
        texts = combined
    else:
        texts = df[resolved_text_col]

    texts = texts.fillna("").astype(str).apply(clean_email_text)
    labels = _normalize_labels(df[resolved_label_col])

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
    )
    x_train_vec = vectorizer.fit_transform(x_train)

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(x_train_vec, y_train)

    x_test_vec = vectorizer.transform(x_test)
    probs = classifier.predict_proba(x_test_vec)[:, 1]
    preds = (probs >= 0.5).astype(int)

    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)

    model_dir = ensure_dir(model_dir)
    joblib.dump(vectorizer, model_dir / "tfidf.joblib")
    joblib.dump(classifier, model_dir / "nlp_model.joblib")

    info = {
        "text_col": resolved_text_col,
        "label_col": resolved_label_col,
        "test_size": test_size,
        "f1": f1,
        "roc_auc": auc,
        "classification_report": report,
    }
    (model_dir / "nlp_report.json").write_text(json.dumps(info, indent=2))
    return info


def load_nlp_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    vectorizer = joblib.load(model_dir / "tfidf.joblib")
    classifier = joblib.load(model_dir / "nlp_model.joblib")
    return vectorizer, classifier


def predict_nlp_proba(text: str, vectorizer, classifier) -> float:
    cleaned = clean_email_text(text)
    vec = vectorizer.transform([cleaned])
    proba = classifier.predict_proba(vec)[0, 1]
    return float(proba)


def explain_nlp(text: str, vectorizer, classifier, top_k: int = 8):
    if not hasattr(classifier, "coef_"):
        return []

    cleaned = clean_email_text(text)
    vec = vectorizer.transform([cleaned])
    row = vec.tocsr()
    if row.nnz == 0:
        return []

    coef = classifier.coef_[0]
    scores = row.data * coef[row.indices]
    if scores.size == 0:
        return []

    order = np.argsort(scores)[::-1]
    terms = vectorizer.get_feature_names_out()

    top_terms = []
    for idx in order:
        if len(top_terms) >= top_k:
            break
        if scores[idx] <= 0:
            break
        term = terms[row.indices[idx]]
        top_terms.append((term, float(scores[idx])))

    return top_terms
