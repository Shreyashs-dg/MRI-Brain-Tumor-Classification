
import streamlit as st
import sys
from pathlib import Path
from PIL import Image

# Project path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import prediction function
from src.predict import predict_mri

# Page configuration
st.set_page_config(
    page_title="MRI Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# Application title
st.title("🧠 MRI Brain Tumor Classification")

st.write(
    "Upload an MRI brain scan to classify it "
    "using our trained deep learning model."
)

# About project
st.subheader("About the Project")

st.write(
    "This application uses a trained CNN model "
    "to classify MRI images into four categories:"
)

st.markdown("""
- Glioma
- Meningioma
- No Tumor
- Pituitary
""")

# MRI image upload
st.subheader("Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI brain scan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI Image",
        width=300
    )

    st.success("MRI image uploaded successfully!")

    if st.button("Predict Tumor Class"):

        with st.spinner("Analysing MRI image..."):
            result = predict_mri(uploaded_file)

        # Prediction result
        st.subheader("Prediction Result")

        st.success(
            f"Predicted Class: {result['predicted_class'].upper()}"
        )

        st.metric(
            label="Model Confidence",
            value=f"{result['confidence']:.2f}%"
        )

        # Probability visualization
        st.subheader("Class Probabilities")

        for class_name, probability in result["probabilities"].items():

            st.write(
                f"**{class_name.capitalize()}** — {probability:.2f}%"
            )

            st.progress(
                min(max(probability / 100, 0.0), 1.0)
            )

# Medical disclaimer
st.caption(
    "Educational AI project — not a medical diagnostic tool."
)
