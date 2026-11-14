import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

# Make `src` importable when running `streamlit run app/app.py` from the project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.predict import load_model, predict_mri  # noqa: E402

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}
LOW_CONFIDENCE = 70.0  # below this, show an extra warning

st.set_page_config(page_title="MRI Brain Tumor Classification", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


get_model()  # load once at startup, so the first prediction is fast

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "A convolutional neural network (CNN) that classifies a brain MRI scan "
        "into one of four categories:"
    )
    for label in DISPLAY_NAMES.values():
        st.markdown(f"- {label}")

    st.header("How to use")
    st.markdown(
        "1. Upload a single MRI slice (JPG or PNG)\n"
        "2. The prediction appears next to the image\n"
        "3. Check the confidence and the class probabilities"
    )

# ---------- Header ----------
st.title("🧠 MRI Brain Tumor Classification")
st.caption("Deep-learning classification of brain MRI scans")

st.warning(
    "**Educational project, not a medical diagnostic tool.** "
    "The results must not be used for diagnosis or treatment decisions. "
    "Always consult a qualified doctor."
)

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload an MRI brain scan (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an MRI image to get a prediction.")
    st.stop()

try:
    image = Image.open(uploaded_file)
    image.load()  # decode now, to catch corrupted files
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
    st.stop()

# ---------- Result ----------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Uploaded scan")
    st.image(image, width=360)

with right:
    st.subheader("Prediction")
    with st.spinner("Analysing MRI image..."):
        try:
            result = predict_mri(uploaded_file)
        except Exception:
            st.error("Something went wrong while analysing this image. Please try another one.")
            st.stop()

    st.info(f"### {DISPLAY_NAMES[result['predicted_class']]}")
    st.metric("Model confidence", f"{result['confidence']:.2f}%")

    if result["confidence"] < LOW_CONFIDENCE:
        st.warning("The model is not very confident about this image. Treat this result with extra caution.")

    st.markdown("**Class probabilities**")
    ranked = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
    for name, probability in ranked:
        st.progress(min(max(probability / 100, 0.0), 1.0), text=f"{DISPLAY_NAMES[name]}: {probability:.2f}%")

st.divider()
st.caption("Educational AI project. Not for clinical use.")