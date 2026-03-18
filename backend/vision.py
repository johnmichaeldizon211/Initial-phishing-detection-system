from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)
SKLEARN_IMAGE_SIZE = (96, 96)


def _load_tensorflow_model(path: Path):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not available. Use Python 3.11/3.12 or train the sklearn vision model."
        ) from exc
    return tf.keras.models.load_model(path)


def _load_sklearn_model(path: Path):
    return joblib.load(path)


def _preprocess_tensorflow(image_path: Path):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def _preprocess_sklearn(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(SKLEARN_IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def load_vision_model(model_path: str | Path) -> Tuple[str, object]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Vision model not found: {path}")

    if path.suffix in {".joblib", ".pkl"}:
        return "sklearn", _load_sklearn_model(path)

    return "tensorflow", _load_tensorflow_model(path)


def predict_vision_proba(image_path: str | Path, model_bundle) -> float:
    if isinstance(model_bundle, tuple) and len(model_bundle) == 2:
        backend, model = model_bundle
    else:
        backend, model = "tensorflow", model_bundle

    image_path = Path(image_path)

    if backend == "sklearn":
        vec = _preprocess_sklearn(image_path)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([vec])[0][1]
        else:
            score = model.decision_function([vec])[0]
            proba = 1 / (1 + np.exp(-score))
        return float(proba)

    arr = _preprocess_tensorflow(image_path)
    proba = model.predict(arr, verbose=0)[0][0]
    return float(proba)
