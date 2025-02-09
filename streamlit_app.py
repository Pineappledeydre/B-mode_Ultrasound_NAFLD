import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize
from PIL import Image

# Load the trained model
model_path = "models/final_model.pkl"
final_model = joblib.load(model_path)

# Load feature extractor if needed
image_size = (224, 224)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Image preprocessing function
def preprocess_image(uploaded_file, target_size=(224, 224)):
    """Reads and preprocesses the uploaded ultrasound image."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = smart_resize(image, target_size)
    image_resized = preprocess_input(image_resized)  # Normalize input (for CNN)
    return image_resized

st.title("NAFLD & Fat Percentage Detection")
st.write("Upload an ultrasound image to detect fat percentage and NAFLD classification.")
uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["png", "jpg", "jpeg"])

from PIL import Image

if uploaded_file:
    # Save uploaded image temporarily
    temp_image_path = "temp_report.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Open image and process it
    img = Image.open(temp_image_path).convert("RGB")
    img_resized = np.array(resize(np.array(img), image_size)) / 255.0  # Normalize
    img_resized = img_resized.reshape(1, *image_size, 3)  # Expand dimensions

    # Extract features using MobileNetV2
    img_features = feature_extractor.predict(img_resized)
    img_features = img_features.reshape(1, -1)  # Flatten

    # Predict using Stacking Model + XGBoost
    prediction = final_model.predict(img_features)

    # Display the result
    st.sidebar.markdown(f"### 🏥 **NAFLD Risk Score:** `{prediction[0]:.3f}`")


