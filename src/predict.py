
from pathlib import Path
import urllib.request
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "optimized_cnn_brain_tumor.keras"

MODEL_URL = (
    "https://huggingface.co/shreyashs24/"
    "mri-brain-tumor-classifier/resolve/main/"
    "optimized_cnn_brain_tumor.keras"
)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print("Using locally available CNN model.")
        return

    print("Downloading optimized CNN model from Hugging Face...")

    temp_path = MODEL_PATH.with_suffix(".download")

    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "MRI-Brain-Tumor-Classifier"}
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with open(temp_path, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)

        temp_path.replace(MODEL_PATH)
        print("Model downloaded successfully.")

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


download_model()
model = tf.keras.models.load_model(MODEL_PATH)


def predict_mri(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))

    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": {
            name: round(float(prob * 100), 2)
            for name, prob in zip(CLASS_NAMES, probabilities)
        }
    }
