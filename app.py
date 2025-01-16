import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Cache the model using the new st.cache_resource decorator
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/breast.h5')
    return model

# Load the model
model = load_model()

# App title and description
st.title("Breast Tumor Detection")
st.write("""
This app uses a pre-trained Deep Learning model to detect breast tumors from mammogram images.
Upload a mammogram image (e.g., JPG, PNG) and the app will predict whether a tumor is detected.
""")

# File uploader allows users to upload their image
uploaded_file = st.file_uploader("Choose a mammogram image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
   # st.write("Predicting...")
    
    # --- Preprocess the image ---
    # 1. Convert image to grayscale if not already
    image = ImageOps.grayscale(image)
    
    # 2. Resize image to the target size (64x64 as in the training pipeline)
    image = image.resize((64, 64))
    
    # 3. Convert the image to a NumPy array and normalize pixel values
    image_array = np.array(image).astype('float32') / 255.0
    
    # 4. Reshape the array to add the channel dimension and create a batch of size 1
    # The model expects input shape (64, 64, 1)
    image_array = image_array.reshape((1, 64, 64, 1))
    
    # --- Get model prediction ---
    prediction = model.predict(image_array)
    
    # The model outputs a probability. Using a threshold of 0.5:
    # - probability >= 0.5: Tumor Detected (class 1)
    # - probability < 0.5: No Tumor (class 0)
    class_label = "Tumor Detected" if prediction[0][0] >= 0.5 else "No Tumor"
    st.write(f"**Prediction:** {class_label}")
    st.write(f"**Prediction Confidence:** {prediction[0][0]:.4f}")
