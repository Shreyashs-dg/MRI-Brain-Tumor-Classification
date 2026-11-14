from pathlib import Path

# Project root = the folder that contains src/, models/, app/
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "optimized_cnn_brain_tumor.keras"

IMG_SIZE = (224, 224)
CHANNELS = 1  # grayscale

# Same (alphabetical) order as in training: glioma=0, meningioma=1, notumor=2, pituitary=3
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]