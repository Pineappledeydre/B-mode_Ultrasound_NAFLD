import streamlit as st
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize

# Load pre-trained models
stacking_model = joblib.load("models/stacking_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
pca_model = joblib.load("models/pca_model.pkl")
lasso_selector = joblib.load("models/lasso_selector.pkl")

# MobileNetV2 as Feature Extractor
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

st.set_page_config(page_title="Liver Ultrasound Analyzer", layout="wide")
st.title("🩺 Liver Ultrasound Analyzer")
uploaded_file = st.file_uploader("Upload a Liver Ultrasound Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)  # Grayscale to RGB
    
    img_resized = resize(img, (224, 224)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # Shape (1, 224, 224, 3)

    # Extract Features using MobileNetV2
    features = feature_extractor.predict(img_resized)
    features = features.reshape(features.shape[0], -1)  # Flatten features

    # Apply PCA Transformation
    features_pca = pca_model.transform(features)

    # Feature Selection using Lasso
    important_features = lasso_selector.coef_ > 0.01
    X_selected = features_pca[:, important_features]

    # Stacking Model Prediction (NAFLD Classification)
    stacking_pred = stacking_model.predict(X_selected)
    label = "Healthy" if stacking_pred[0] == 0 else "Fatty Liver (NAFLD)"

    # Fat Percentage Prediction using XGBoost
    fat_percentage = xgb_model.predict(stacking_pred.reshape(-1, 1))[0]

    # Show Results
    st.image(img, caption="Uploaded Liver Ultrasound", use_column_width=True)
    st.subheader(f"🩺 Diagnosis: **{label}**")
    st.subheader(f"📊 Estimated Fat Percentage: **{fat_percentage:.2f}%**")

    # Show Classification Probability
    st.progress(float(stacking_pred[0]))  # Visualize model confidence

    # **Show Processed Image**
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Liver Fat %: {fat_percentage:.2f}%")
    st.pyplot(fig)
