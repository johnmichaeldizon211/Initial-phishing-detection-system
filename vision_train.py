from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.utils import ensure_dir
from backend.vision import IMAGE_SIZE, SKLEARN_IMAGE_SIZE


def _select_backend(choice: str) -> str:
    choice = (choice or "auto").lower().strip()
    if choice in {"tensorflow", "tf"}:
        return "tensorflow"
    if choice in {"sklearn", "classic"}:
        return "sklearn"

    try:
        import tensorflow as _  # noqa: F401

        return "tensorflow"
    except Exception:
        return "sklearn"


def _load_images_sklearn(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for label_name, label_value in [("phishing", 1), ("legit", 0)]:
        label_dir = data_dir / label_name
        if not label_dir.exists():
            continue
        for path in label_dir.glob("*"):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            img = Image.open(path).convert("RGB")
            img = img.resize(SKLEARN_IMAGE_SIZE)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            rows.append(arr.flatten())
            labels.append(label_value)

    if not rows:
        raise ValueError("No images found. Ensure phishing/ and legit/ folders contain images.")

    return np.array(rows), np.array(labels, dtype=np.int32)


def _train_sklearn(data_dir: Path, model_out: Path) -> dict:
    x, y = _load_images_sklearn(data_dir)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(x_train, y_train)

    probs = model.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "backend": "sklearn",
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "image_size": SKLEARN_IMAGE_SIZE,
        "samples": int(len(y)),
    }

    ensure_dir(model_out.parent)
    joblib.dump(model, model_out)
    report_path = model_out.with_suffix(".json")
    report_path.write_text(json.dumps(metrics, indent=2))

    print("Vision training complete (sklearn).")
    print(f"Saved: {model_out}")
    print(f"Metrics: {metrics}")
    return metrics


def _train_tensorflow(args, data_dir: Path, model_out: Path) -> dict:
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="binary",
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ]
    )

    weights = args.weights
    if isinstance(weights, str) and weights.strip().lower() in {"none", "null", "no"}:
        weights = None

    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    if args.fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:-args.fine_tune_layers]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=max(1, args.epochs // 2))

    results = model.evaluate(val_ds, verbose=0)
    metrics = dict(zip(model.metrics_names, [float(x) for x in results]))
    metrics["backend"] = "tensorflow"

    ensure_dir(model_out.parent)
    model.save(model_out)

    report_path = model_out.with_suffix(".json")
    report_path.write_text(json.dumps(metrics, indent=2))

    print("Vision training complete (tensorflow).")
    print(f"Saved: {model_out}")
    print(f"Metrics: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a vision model on website screenshots.")
    parser.add_argument("--data-dir", required=True, help="Directory with class subfolders (phishing/legit).")
    parser.add_argument("--model-out", default="models/vision_model.keras", help="Output model path.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--weights",
        default="imagenet",
        help="MobileNetV2 weights: 'imagenet' or 'none' to disable downloads (tensorflow only).",
    )
    parser.add_argument("--fine-tune", action="store_true", help="Enable fine-tuning after initial training.")
    parser.add_argument("--fine-tune-layers", type=int, default=20)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "tensorflow", "sklearn"],
        help="Training backend: auto (default), tensorflow, or sklearn.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    backend = _select_backend(args.backend)
    model_out = Path(args.model_out)

    if backend == "sklearn":
        if model_out.suffix in {".keras", ".h5", ".hdf5"}:
            model_out = model_out.with_name("vision_sklearn.joblib")
        _train_sklearn(data_dir, model_out)
    else:
        _train_tensorflow(args, data_dir, model_out)


if __name__ == "__main__":
    main()
