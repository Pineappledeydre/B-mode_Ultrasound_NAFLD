import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize

# 🎯 Load trained models
stacking_model = joblib.load("models/stacking_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
lasso = joblib.load("models/lasso_selector.pkl")

# 🎯 Load MobileNetV2 as Feature Extractor
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-30]:  # Keep last 30 layers trainable
    layer.trainable = False
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# 🎯 Streamlit App Config
st.set_page_config(page_title="B-Mode Ultrasound NAFLD", layout="wide")

# 📤 Upload Image
st.sidebar.header("📤 Upload Ultrasound Image")
uploaded_file = st.sidebar.file_uploader("Upload an Ultrasound Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 📂 Save Temporary Image
    temp_image_path = "temp_ultrasound.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 🖼 Load & Preprocess Image
    image = cv2.imread(temp_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = resize(image_rgb, (224, 224)) / 255.0  # Resize & normalize

    # 🧠 Extract Features
    X_features = feature_extractor.predict(np.expand_dims(image_resized, axis=0))
    X_features = X_features.reshape(1, -1)  # Flatten

    # 📌 Apply Lasso Feature Selection
    important_features = np.abs(lasso.coef_) > 0.01
    if np.sum(important_features) < 45:
        top_features = np.argsort(np.abs(lasso.coef_))[-45:]
        important_features = np.zeros_like(lasso.coef_, dtype=bool)
        important_features[top_features] = True

    X_selected = X_features[:, important_features]

    # ✅ Ensure Feature Count Matches Training
    expected_features = stacking_model.estimators_[0][1].n_features_in_
    st.write(f"🔍 **X_selected shape:** {X_selected.shape}")
    st.write(f"🔍 **Expected features for Stacking Model:** {expected_features}")

    if X_selected.shape[1] != expected_features:
        st.error(f"❌ Feature shape mismatch! Expected {expected_features}, got {X_selected.shape[1]}")
        st.stop()

    # 🔍 **NAFLD Classification Prediction**
    stacking_pred_proba = stacking_model.predict(X_selected)
    st.write(f"🔍 **Stacking Model Output Shape:** {stacking_pred_proba.shape}")
    st.write(f"🔍 **Stacking Model Output:** {stacking_pred_proba}")

    # ✅ Convert to Single Value Prediction (NAFLD Diagnosis)
    stacking_pred = np.argmax(stacking_pred_proba, axis=1)  # Take highest probability class (0 or 1)
    st.write(f"🔍 **Final Stacking Prediction (Single Value):** {stacking_pred}")

    # 🩺 **NAFLD Diagnosis**
    nafld_label = "Healthy" if stacking_pred[0] == 0 else "Fatty Liver (NAFLD) Detected"

    # ✅ **Check Expected Features for XGBoost**
    xgb_expected_features = xgb_model.get_booster().num_features()
    st.write(f"🔍 **XGBoost Expected Features:** {xgb_expected_features}")
    st.write(f"🔍 **Stacking Prediction Shape Before XGBoost:** {stacking_pred.shape}")

    # 🔢 **Fat Percentage Prediction (Fix Input to XGBoost)**
    try:
        fat_percentage = xgb_model.predict(np.array(stacking_pred).reshape(-1, 1))[0]  # ✅ Pass single value
    except ValueError as e:
        st.error(f"❌ XGBoost Feature Mismatch: {e}")
        st.stop()

    # 🎯 **Display Results**
    st.subheader("🩺 Prediction Results")
    st.info(f"**NAFLD Diagnosis:** {nafld_label}")
    st.success(f"**Estimated Fat Percentage:** {fat_percentage:.2f}%")

    # 🖼 Show Uploaded Image
    st.image(image_rgb, caption="Uploaded Ultrasound", use_column_width=True)

st.markdown("---")
st.markdown("**ℹ Note:** The app automatically processes and classifies NAFLD from uploaded ultrasound images.")
