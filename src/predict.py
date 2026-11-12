
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "optimized_cnn_brain_tumor.keras"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

model = tf.keras.models.load_model(MODEL_PATH)


def predict_mri(image_path):
    # 1. Load and preprocess the MRI image
    image = Image.open(image_path).convert("L")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))

    # 2. Predict class probabilities
    probabilities = model.predict(image_array, verbose=0)[0]

    # 3. Find the predicted class
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)

    # 4. Return results
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": {name: round(float(prob * 100), 2)
            for name, prob in zip(CLASS_NAMES, probabilities)
        }
    }
