import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize
import matplotlib.pyplot as plt

# Load trained models
stacking_model = joblib.load("models/stacking_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
lasso = joblib.load("models/lasso_selector.pkl")

# Load MobileNetV2 as Feature Extractor
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-30]:  # Keep last 30 layers trainable
    layer.trainable = False
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Streamlit App Config
st.set_page_config(page_title="B-Mode Ultrasound NAFLD", layout="wide")

# Upload Image
st.sidebar.header("📤 Upload Ultrasound Image")
uploaded_file = st.sidebar.file_uploader("Upload an Ultrasound Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Save Temporary Image
    temp_image_path = "temp_ultrasound.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load & Preprocess Image
    image = cv2.imread(temp_image_path)
    if image is None:
        st.error(f"❌ Could not load image: {temp_image_path}")
        st.stop()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = resize(image_rgb, (224, 224)) / 255.0  # Resize & normalize

    # Extract Features
    X_features = feature_extractor.predict(np.expand_dims(image_resized, axis=0))
    X_features = X_features.reshape(1, -1)  # Flatten

    # Apply Lasso Feature Selection
    important_features = np.abs(lasso.coef_) > 0.01
    if np.sum(important_features) < 45:
        top_features = np.argsort(np.abs(lasso.coef_))[-45:]
        important_features = np.zeros_like(lasso.coef_, dtype=bool)
        important_features[top_features] = True

    X_selected = X_features[:, important_features]

    # Ensure Feature Count Matches Training
    expected_features = stacking_model.estimators_[0][1].n_features_in_
    # st.write(f"**X_selected shape:** {X_selected.shape}")
    # st.write(f"**Expected features for Stacking Model:** {expected_features}")

    if X_selected.shape[1] != expected_features:
        st.error(f"❌ Feature shape mismatch! Expected {expected_features}, got {X_selected.shape[1]}")
        st.stop()

    # **NAFLD Classification Prediction**
    stacking_pred = stacking_model.predict(X_selected).reshape(-1, 1)
    # st.write(f"🔍 **Stacking Model Output:** {stacking_pred}")

    nafld_label = "Healthy" if stacking_pred[0] == 0 else "Healthy"
    fat_percentage = xgb_model.predict(stacking_pred)[0]
    st.subheader("🩺 Prediction Results")
    st.info(f"**NAFLD Diagnosis:** {nafld_label}")
    st.success(f"**Estimated Fat Percentage:** {fat_percentage:.2f}%")
    st.image(image_rgb, caption="Uploaded Ultrasound", use_container_width=True)

    # fig, ax = plt.subplots()
    # ax.imshow(image_rgb)
    # ax.set_title("Ultrasound Image")
    # ax.axis("off")  
    # st.pyplot(fig)

st.markdown("---")
st.markdown("**ℹ Note:** The app automatically processes and classifies NAFLD from uploaded ultrasound images.")
