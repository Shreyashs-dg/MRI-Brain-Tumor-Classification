from functools import lru_cache

import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, MODEL_PATH
from .preprocess import preprocess_image


@lru_cache(maxsize=1)
def load_model():
    """Load the model once and reuse it."""
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_mri(image):
    """image: file path, uploaded file, or PIL image."""
    batch = preprocess_image(image)
    probabilities = load_model().predict(batch, verbose=0)[0]
    best = int(np.argmax(probabilities))

    return {
        "predicted_class": CLASS_NAMES[best],
        "confidence": round(float(probabilities[best]) * 100, 2),
        "probabilities": {
            name: round(float(p) * 100, 2) for name, p in zip(CLASS_NAMES, probabilities)
        },
    }