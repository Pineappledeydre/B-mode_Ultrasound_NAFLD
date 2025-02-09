import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.preprocessing.image import smart_resize

# the trained model
final_model = joblib.load("models/final_model.pkl")

# image preprocessing function
def preprocess_image(uploaded_file, target_size=(224, 224)):
    """Reads and preprocesses the uploaded ultrasound image."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # read as RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = smart_resize(image, target_size) / 255.0
    return image_resized

st.title("NAFLD & Fat Percentage Detection")
st.write("Upload an ultrasound image to detect fat percentage and NAFLD classification.")
uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    # Preprocess the image
    image = preprocess_image(uploaded_file)
    # Reshape to match model input
    image = image[np.newaxis, ...]  # Add batch dimension
    # Get predictions
    fat_percentage = final_model.predict(image)[0]  # Model returns fat % directly
    nafld_class = "Yes" if fat_percentage >= 5 else "No"

    # Display results
    st.subheader("Results")
    st.write(f"**Estimated Fat Percentage:** {fat_percentage:.2f}%")
    st.write(f"**NAFLD Classification:** {nafld_class}")

