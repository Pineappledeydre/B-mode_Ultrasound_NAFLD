import streamlit as st
st.set_page_config(page_title="B-Mode Ultrasound NAFLD", layout="wide") 

import numpy as np
import streamlit as st
import joblib
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize

# ✅ Load Models
stacking_model = joblib.load("models/stacking_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
lasso = joblib.load("models/lasso_selector.pkl")
pca = joblib.load("models/pca_model.pkl")  # ✅ Load PCA

# ✅ Load MobileNetV2 as Feature Extractor
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-30]:  
    layer.trainable = False
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# ✅ Streamlit UI
st.set_page_config(page_title="B-Mode Ultrasound NAFLD", layout="wide")
st.sidebar.header("📤 Upload Ultrasound Image")
uploaded_file = st.sidebar.file_uploader("Upload an Ultrasound Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # ✅ Process Image
    st.write("📊 **Processing Uploaded Image...**")
    
    temp_image_path = "temp_ultrasound.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = cv2.imread(temp_image_path)
    if image is None:
        st.error("❌ Could not load image!")
        st.stop()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = resize(image_rgb, (224, 224)) / 255.0  # Normalize

    st.write(f"✅ **Image Resized Shape:** {image_resized.shape}")

    # ✅ Extract Features
    X_features = feature_extractor.predict(np.expand_dims(image_resized, axis=0))
    X_features = X_features.reshape(1, -1)  # Flatten

    st.write(f"📊 **Extracted Features Shape:** {X_features.shape}")

    # ✅ Apply PCA before Lasso
    X_features_pca = pca.transform(X_features)  # **Apply PCA First!**
    
    # ✅ Apply Lasso Feature Selection
    important_features = np.abs(lasso.coef_) > 0.01  # Select relevant features
    if np.sum(important_features) < 45:
        top_features = np.argsort(np.abs(lasso.coef_))[-45:]  # Ensure 45 features
        important_features = np.zeros_like(lasso.coef_, dtype=bool)
        important_features[top_features] = True

    X_selected = X_features_pca[:, important_features]  # ✅ Use PCA Features

    st.write(f"✅ **X_selected Shape Before Prediction:** {X_selected.shape}")

    # ✅ Ensure Shape Matches Stacking Model
    expected_features = stacking_model.estimators_[0][1].n_features_in_
    if X_selected.shape[1] != expected_features:
        st.error(f"❌ Feature shape mismatch! Expected {expected_features}, got {X_selected.shape[1]}")
        st.stop()

    # **🔮 NAFLD Prediction**
    stacking_pred_proba = stacking_model.predict_proba(X_selected)
    stacking_pred = stacking_pred_proba.argmax(axis=1).reshape(-1, 1)

    nafld_label = "Healthy" if stacking_pred[0] == 0 else "Fatty Liver (NAFLD) Detected"

    # **📉 Predict Fat Percentage**
    st.write("📉 **Predicting Fat Percentage (Lasso First Pass)...**")
    fats_pred = lasso.predict(X_features_pca)[0]  # Use full PCA features

    st.write(f"📉 **Lasso Predicted Fat Percentage:** {fats_pred:.2f}%")

    # **📈 XGBoost Final Fat Prediction**
    st.write("📈 **Refining Fat Percentage Prediction (XGBoost Final Prediction)...**")

    if xgb_model.n_features_in_ == 1:
        xgb_input = stacking_pred_proba  # Use only stacking prediction
    else:
        xgb_input = np.hstack([stacking_pred_proba, np.array(fats_pred).reshape(-1, 1)])

    fat_percentage_final = xgb_model.predict(xgb_input)[0]

    st.subheader("🩺 Prediction Results")
    st.info(f"**NAFLD Diagnosis:** {nafld_label}")
    st.success(f"**Final Estimated Fat Percentage:** {fat_percentage_final:.2f}%")
    st.image(image_rgb, caption="Uploaded Ultrasound", use_container_width=True)

st.markdown("---")
st.markdown("ℹ **Note:** The app automatically processes and classifies NAFLD from uploaded ultrasound images.")
