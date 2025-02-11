import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize
import matplotlib.pyplot as plt

# Streamlit App Config
st.set_page_config(page_title="B-Mode Ultrasound NAFLD", layout="wide")

# Load trained models with debugging
st.write("🔄 **Loading Models...**")
try:
    stacking_model = joblib.load("models/stacking_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    lasso = joblib.load("models/lasso_selector.pkl")
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")

# Load MobileNetV2 as Feature Extractor
st.write("🔄 **Initializing MobileNetV2 for Feature Extraction...**")
try:
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-30]:  # Keep last 30 layers trainable
        layer.trainable = False
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    st.success("✅ MobileNetV2 initialized!")
    
    # 🔍 Test the extractor on a dummy input
    test_input = np.random.rand(1, 224, 224, 3).astype("float32")
    test_output = feature_extractor.predict(test_input)
    st.write(f"⚡ **Test Output Shape from MobileNetV2:** {test_output.shape}")
except Exception as e:
    st.error(f"❌ Error initializing MobileNetV2: {e}")

# Upload Image
st.sidebar.header("📤 Upload Ultrasound Image")
uploaded_file = st.sidebar.file_uploader("Upload an Ultrasound Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.write("📥 **Processing Uploaded Image...**")
    
    # Save Temporary Image
    temp_image_path = "temp_ultrasound.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load & Preprocess Image
    image = cv2.imread(temp_image_path)

    if image is None:
        st.error(f"❌ Could not load image: {temp_image_path}")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = resize(image_rgb, (224, 224)).astype("float32") / 255.0  # Ensure float32 normalization

        st.write(f"📸 **Image Loaded Successfully** - Shape: {image_rgb.shape}")

        # Extract Features
        st.write("🔍 **Extracting Features...**")
        try:
            X_features = feature_extractor.predict(np.expand_dims(image_resized, axis=0))
            st.write(f"🔎 **Raw MobileNetV2 Output Shape:** {X_features.shape}")

            X_features = X_features.flatten().reshape(1, -1)  # Flatten
            st.write(f"📊 **Extracted Features Shape (After Flattening):** {X_features.shape}")
            st.write(f"🔎 **First 10 extracted features:** {X_features.flatten()[:10]}")

            # 🔥 Check for all zeros or NaN values
            if np.all(X_features == 0):
                st.error("🚨 **All extracted features are zero! MobileNetV2 is not working correctly.**")
            if np.isnan(X_features).any():
                st.error("🚨 **Extracted features contain NaN values!**")
        except Exception as e:
            st.error(f"❌ Feature extraction failed: {e}")

        # Apply Lasso Feature Selection
        st.write("🎯 **Applying Lasso Feature Selection...**")
        try:
            important_features = np.abs(lasso.coef_) > 0.01
            if np.sum(important_features) < 45:
                top_features = np.argsort(np.abs(lasso.coef_))[-45:]
                important_features = np.zeros_like(lasso.coef_, dtype=bool)
                important_features[top_features] = True
            X_selected = X_features[:, important_features]
            st.write(f"📊 **Selected Features Shape (After Lasso):** {X_selected.shape}")
            st.write(f"🔎 **First 10 selected features:** {X_selected.flatten()[:10]}")
        except Exception as e:
            st.error(f"❌ Lasso feature selection failed: {e}")

        # **NAFLD Classification Prediction**
        st.write("🧠 **Predicting NAFLD Diagnosis...**")
        try:
            stacking_pred_raw = stacking_model.predict_proba(X_selected)  # Get probabilities
            stacking_pred = stacking_pred_raw.argmax(axis=1).reshape(-1, 1)  # Convert to class label
            st.write(f"🔍 **Stacking Model Raw Prediction Probabilities:** {stacking_pred_raw}")
            st.write(f"🔍 **Stacking Model Prediction Output:** {stacking_pred}")

            nafld_label = "Healthy" if stacking_pred[0] == 0 else "Fatty Liver (NAFLD) Detected"
        except Exception as e:
            st.error(f"❌ NAFLD classification failed: {e}")
            stacking_pred = [[-1]]  # Placeholder if prediction fails

        # **Fat Percentage Prediction**
        st.write("📈 **Predicting Fat Percentage...**")
        try:
            fat_percentage_raw = xgb_model.predict_proba(stacking_pred)  # Get probabilities
            fat_percentage = xgb_model.predict(stacking_pred)[0]
            st.write(f"📉 **XGBoost Raw Prediction Probabilities:** {fat_percentage_raw}")
            st.write(f"📉 **Predicted Fat Percentage (XGBoost):** {fat_percentage:.2f}%")
        except Exception as e:
            st.error(f"❌ Fat percentage prediction failed: {e}")
            fat_percentage = -1  # Placeholder if prediction fails

        # **Final Output**
        st.subheader("🩺 Prediction Results")
        if stacking_pred[0] == -1:
            st.warning("⚠ Prediction was unsuccessful due to errors. Check logs above.")
        else:
            st.info(f"**NAFLD Diagnosis:** {nafld_label}")
            st.success(f"**Estimated Fat Percentage:** {fat_percentage:.2f}%")
        
        # Display Image
        st.image(image_rgb, caption="Uploaded Ultrasound", use_container_width=True)

st.markdown("---")
st.markdown("**ℹ Note:** The app automatically processes and clas
