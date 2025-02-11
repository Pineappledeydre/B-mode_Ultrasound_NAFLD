import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize

stacking_model = joblib.load("models/stacking_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
lasso = joblib.load("models/lasso_selector.pkl")

# MobileNetV2 for feature extraction
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-30]:  # Keep last 30 layers trainable
    layer.trainable = False
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Streamlit UI Config
st.set_page_config(page_title="B-Mode Ultrasound NAFLD", layout="wide")
st.sidebar.header("📤 Upload Ultrasound Image")
uploaded_file = st.sidebar.file_uploader("Upload an Ultrasound Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    temp_image_path = "temp_ultrasound.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = cv2.imread(temp_image_path)
    if image is None:
        st.error(f"Could not load image: {temp_image_path}")
        st.stop()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = resize(image_rgb, (224, 224)) / 255.0  # Normalize

    # Extract features using MobileNetV2
    X_features = feature_extractor.predict(np.expand_dims(image_resized, axis=0))
    X_features = X_features.reshape(1, -1)  # Flatten

    # Apply Lasso Feature Selection
    important_features = np.abs(lasso.coef_) > 0.01
    if np.sum(important_features) < 45:
        top_features = np.argsort(np.abs(lasso.coef_))[-45:]
        important_features = np.zeros_like(lasso.coef_, dtype=bool)
        important_features[top_features] = True
    X_selected = X_features[:, important_features]

    # Ensure feature count matches model expectations
    expected_features = stacking_model.estimators_[0][1].n_features_in_
    if X_selected.shape[1] != expected_features:
        st.error(f"Feature shape mismatch! Expected {expected_features}, got {X_selected.shape[1]}")
        st.stop()

    # Predict NAFLD using Stacking Model
    stacking_pred_proba = stacking_model.predict_proba(X_selected)  # Probabilities
    stacking_pred = stacking_pred_proba.argmax(axis=1).reshape(-1, 1)  # Class Label

    st.write(f"**Stacking Model Raw Prediction Probabilities:** {stacking_pred_proba}")
    st.write(f"**Stacking Model Prediction Output:** {stacking_pred[0][0]}")

    nafld_label = "Healthy" if stacking_pred[0] == 0 else "Fatty Liver (NAFLD) Detected"

    # XGBoost for Classification Refinement (But NOT for Fat Prediction)
    if xgb_model.n_features_in_ == 1:
        st.write("Using **Only Stacking Prediction** for XGBoost Input")
        xgb_input = stacking_pred  # Use only stacking output
    else:
        st.write("sing **Stacking + Features** for XGBoost Input")
        xgb_input = np.hstack([stacking_pred_proba, X_selected])  # Use additional selected features

    xgb_refined_class = xgb_model.predict(xgb_input)[0]

    # Predict Final Fat Percentage with Lasso (Not XGBoost)
    st.write("**Final Fat Percentage Prediction with Lasso...**")
    fat_percentage_final = lasso.predict(X_selected)[0]  # Lasso as final predictor

    st.write(f"**Lasso Predicted Fat Percentage (Final):** {fat_percentage_final:.2f}%")

    # Display Results
    st.subheader("🩺 Prediction Results")
    st.info(f"**NAFLD Diagnosis:** {nafld_label}")
    st.success(f"**Estimated Fat Percentage:** {fat_percentage_final:.2f}%")
    st.image(image_rgb, caption="Uploaded Ultrasound", use_container_width=True)

st.markdown("---")
st.markdown("ℹ **Note:** The app automatically processes and classifies NAFLD from uploaded ultrasound images.")
