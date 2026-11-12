# MRI Brain Tumor Classification using ANN and CNN

## Project Overview

This project applies Deep Learning techniques to classify brain MRI images into four categories:

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

Three models were developed or retained during the project:

- Artificial Neural Network (ANN)
- Baseline Convolutional Neural Network (CNN)
- Optimized Convolutional Neural Network (Optimized CNN)

The optimized CNN is integrated into a Streamlit application for MRI image prediction.

**Disclaimer:** This is an educational project, not a medical diagnostic tool. Predictions must not replace professional medical evaluation.

## Dataset

**Dataset:** Brain Tumor MRI Dataset by Masoud Nickparvar

[Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Dataset | Images |
|---|---:|
| Training | 5,600 |
| Testing | 1,600 |
| **Total** | **7,200** |

The dataset contains four equally represented classes:

| Class | Training | Testing | Total |
|---|---:|---:|---:|
| Glioma | 1,400 | 400 | 1,800 |
| Meningioma | 1,400 | 400 | 1,800 |
| No Tumor | 1,400 | 400 | 1,800 |
| Pituitary | 1,400 | 400 | 1,800 |
| **Total** | **5,600** | **1,600** | **7,200** |

## Technologies Used

- Python 3.13.3
- TensorFlow 2.21.0
- Keras
- NumPy 2.5.4
- Pandas 3.1.1
- Matplotlib 3.11.2
- Seaborn 0.13.2
- Scikit-learn 1.9.1
- Pillow 13.0.0
- Streamlit 1.67.0
- Jupyter Notebook
- Git and GitHub

## Image Preprocessing

- Image resizing: 224 × 224 pixels
- Grayscale image conversion
- Pixel normalization: 0 to 1
- Label encoding
- Training and validation splitting
- Image data augmentation

## Model Performance

The following results were obtained on the held-out test dataset of 1,600 images.

| Model | Test Accuracy | Correct Predictions |
|---|---:|---:|
| ANN | 74.50% | 1,192 / 1,600 |
| Baseline CNN | 83.13% | 1,330 / 1,600 |

The baseline CNN achieved an improvement of 8.63 percentage points over the ANN in test accuracy.

An optimized CNN was also developed through architecture, learning-rate, and regularization experiments. Its exact test performance is not reported here.

## Project Structure

```text
MRI-Brain-Tumor-Classification/
├── dataset/
│   ├── Training/
│   └── Testing/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ann_model.ipynb
│   ├── 04_cnn_model.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_model_optimization.ipynb
├── src/
│   └── predict.py
├── models/
├── app/
│   └── app.py
├── images/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shreyashs-dg/MRI-Brain-Tumor-Classification.git
```

### 2. Navigate to the project directory

```bash
cd MRI-Brain-Tumor-Classification
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment (Windows)

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the dataset

Download the dataset from Kaggle and place its Training and Testing folders inside the `dataset/` directory.

### 7. Model files

The trained model files are stored locally in the `models/` directory and are excluded from Git tracking. Make sure the required optimized CNN model is available at:

`models/optimized_cnn_brain_tumor.keras`

## Run the Streamlit Application

From the project root, run:

```bash
streamlit run app/app.py
```

Upload an MRI image to view the predicted class, confidence score, and class probabilities.

## Project Development Roadmap

1. Project Fundamentals & Setup
2. Data Collection & EDA
3. Image Preprocessing
4. ANN Implementation
5. CNN Implementation
6. Model Evaluation & Comparison
7. Model Optimization
8. Model Saving & Prediction
9. Application Development
10. Finalization & Deployment

## Author

Developed as a hands-on Deep Learning project.
