import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize
from sklearn.linear_model import Lasso
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Streamlit App Config
st.set_page_config(page_title="B-Mode Ultrasound NAFLD", layout="wide")

# Load pre-trained models
stacking_model = joblib.load("models/stacking_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
lasso = joblib.load("models/lasso_selector.pkl")

# Load MobileNetV2 for feature extraction
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-30]:
    layer.trainable = False
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Upload Image
st.sidebar.header("📤 Upload Ultrasound Image")
uploaded_file = st.sidebar.file_uploader("Upload an Ultrasound Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Save temporary image
    temp_image_path = "temp_ultrasound.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read Image
    image = cv2.imread(temp_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = resize(image_rgb, (224, 224)) / 255.0  # Normalize

    # Extract Features
    X_features = feature_extractor.predict(np.expand_dims(image_resized, axis=0))
    X_features = X_features.reshape(1, -1)  # Flatten features

    # Lasso Feature Selection
    important_features = np.abs(lasso.coef_) > 0.01
    if np.sum(important_features) < 45:
        top_features = np.argsort(np.abs(lasso.coef_))[-45:]
        important_features = np.zeros_like(lasso.coef_, dtype=bool)
        important_features[top_features] = True
    X_selected = X_features[:, important_features]

    # Predict NAFLD Classification
    stacking_pred = stacking_model.predict(X_selected)
    nafld_label = "Healthy" if stacking_pred[0] == 0 else "Fatty Liver (NAFLD) Detected"

    # Predict Fat Percentage
    fat_percentage = xgb_model.predict(X_selected)[0]

    # Display Results
    st.subheader("🩺 Prediction Results")
    st.info(f"**NAFLD Diagnosis:** {nafld_label}")
    st.success(f"**Estimated Fat Percentage:** {fat_percentage:.2f}%")

    # Show Uploaded Image
    st.image(image_rgb, caption="Uploaded Ultrasound", use_column_width=True)

st.markdown("---")
st.markdown("**ℹ Note:** The app automatically processes and classifies NAFLD from uploaded ultrasound images.")
