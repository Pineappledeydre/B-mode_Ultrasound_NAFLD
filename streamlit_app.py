import streamlit as st
import numpy as np
import joblib
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize

st.set_page_config(page_title="B-Mode Ultrasound NAFLD", layout="wide")

# st.write("**Loading Models...**")
try:
    stacking_model = joblib.load("models/stacking_model.pkl")  
    lasso = joblib.load("models/lasso_selector.pkl")
    pca = joblib.load("models/pca_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")  

    #st.success("Models Loaded Successfully!")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# MobileNetV2 Feature Extractor
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

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = (resize(image_rgb, (224, 224)) - 0.5) * 2  # Normalize

    #st.write(f"**Image Resized Shape:** {image_resized.shape}")

    # Extract Features
    X_features = feature_extractor.predict(np.expand_dims(image_resized, axis=0))
    X_features = X_features.reshape(1, -1)  # Flatten

    #st.write(f"**Extracted Features Shape:** {X_features.shape}")

    # PCA before Lasso
    X_features_pca = pca.transform(X_features)      
    important_features = np.abs(lasso.coef_) > 0.01  # Select relevant features
    if np.sum(important_features) < 45:
        top_features = np.argsort(np.abs(lasso.coef_))[-45:]  # Ensure 45 features
        important_features = np.zeros_like(lasso.coef_, dtype=bool)
        important_features[top_features] = True

    X_selected = X_features_pca[:, important_features]  

    #st.write(f"**X_selected Shape Before Prediction:** {X_selected.shape}")

    expected_features = stacking_model.estimators_[0][1].n_features_in_
    if X_selected.shape[1] != expected_features:
        st.error(f"Feature shape mismatch! Expected {expected_features}, got {X_selected.shape[1]}")
        st.stop()

    stacking_pred_proba = stacking_model.predict_proba(X_selected)
    stacking_pred = stacking_pred_proba.argmax(axis=1).reshape(-1, 1)

    nafld_label = "Healthy" if stacking_pred[0] == 0 else "Fatty Liver (NAFLD) Detected"

    #st.write("📉 **Predicting Fat Percentage (Lasso First Pass)...**")
    fats_pred = lasso.predict(X_features_pca)[0]  # Use full PCA features
    #st.write(f"📉 **Predicted Fat Percentage:** {fats_pred:.2f}%")

    #st.write("**Refining NAFLD Diagnosis with XGBoost...**")
    # xgb_input = stacking_pred_proba  
    # xgb_final_pred = xgb_model.predict(xgb_input)
    #final_label = "Healthy" if xgb_final_pred[0] == 0 else "Fatty Liver (NAFLD) Detected"

    st.subheader("🩺 Prediction Results")
    st.info(f"**Final NAFLD Diagnosis :** {nafld_label}")
    st.success(f"**Estimated Fat Percentage:** {fats_pred:.2f}%")
    st.image(image_rgb, caption="Uploaded Ultrasound", use_container_width=True)

st.markdown("---")
st.markdown("ℹ **Note:** The app automatically processes and classifies NAFLD from uploaded ultrasound images.")
