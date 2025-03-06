import streamlit as st
import numpy as np
import joblib
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize

st.set_page_config(page_title="B-Mode Ultrasound NAFLD", layout="wide")

# ✅ Load Models
try:
    stacking_model = joblib.load("models/stacking_model.pkl")  
    pca = joblib.load("models/pca_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")  

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ✅ MobileNetV2 Feature Extractor (Matches Training)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-30]:  
    layer.trainable = False
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer("block_6_expand_relu").output)

st.sidebar.header("📤 Upload Ultrasound Image")
uploaded_file = st.sidebar.file_uploader("Upload an Ultrasound Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.write("**Processing Uploaded Image...**")
    temp_image_path = "temp_ultrasound.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = cv2.imread(temp_image_path)
    if image is None:
        st.error("Could not load image!")
        st.stop()

    # ✅ Ensure Correct Image Preprocessing (Matches Training)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224)) / 127.5 - 1  # MobileNetV2 expects [-1,1] range

    # ✅ Extract Features
    X_features = feature_extractor.predict(np.expand_dims(image_resized, axis=0))
    X_features = X_features.reshape(1, -1)  # Flatten

    # ✅ PCA Transformation (Ensure Correct Processing Order)
    X_features_pca = pca.transform(X_features)

    # ✅ Ensure Feature Shape Matches Stacking Model
    expected_features = stacking_model.estimators_[0][1].n_features_in_
    if X_features_pca.shape[1] != expected_features:
        st.error(f"Feature shape mismatch! Expected {expected_features}, got {X_features_pca.shape[1]}")
        st.stop()

    # ✅ Predict NAFLD Classification (Stacking Model)
    stacking_pred_proba = stacking_model.predict_proba(X_features_pca)
    stacking_pred = stacking_pred_proba.argmax(axis=1).reshape(-1, 1)
    nafld_label = "Healthy" if stacking_pred[0] == 0 else "Fatty Liver (NAFLD) Detected"

    # ✅ Predict Fat Percentage (Using XGBoost Instead of Lasso)
    fats_pred = xgb_model.predict(stacking_pred_proba)[0]  

    # ✅ Display Results
    st.subheader("🩺 Prediction Results")
    if nafld_label == "Healthy":
        st.markdown(f'<p style="color:green; font-size:20px;"><b>🟢 Final NAFLD Diagnosis: {nafld_label}</b></p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:red; font-size:20px;"><b>🔴 Final NAFLD Diagnosis: {nafld_label}</b></p>', unsafe_allow_html=True)

    st.markdown(f'<p style="color:blue; font-size:20px;"><b>Estimated Fat Percentage: {fats_pred:.2f}%</b></p>', unsafe_allow_html=True)

    st.image(image_rgb, caption="Uploaded Ultrasound", use_container_width=True)

st.markdown("---")
st.markdown("ℹ **Note:** The app automatically processes and classifies NAFLD from uploaded ultrasound images.")
