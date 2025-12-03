import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

# Title and description
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and let AI classify it as a cat or dog!")

# Load model and labels
@st.cache_resource
def load_model():
    """Load the trained model and class labels"""
    try:
        model = tf.keras.models.load_model('cats_dogs_mobilenet.h5')
        with open('class_labels.json', 'r') as f:
            labels = json.load(f)
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Preprocess image for model
def preprocess_image(img):
    """Resize and normalize image for model input"""
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load model
model, class_labels = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of a cat or dog"
)

# Process uploaded image
if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        # Make prediction
        with st.spinner('🔍 Analyzing image...'):
            # Preprocess
            processed_img = preprocess_image(image)
            
            # Predict
            predictions = model.predict(processed_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            # Display results
            st.markdown("### 🎯 Prediction")
            label = class_labels[str(predicted_class)].upper()
            
            if label == "CAT":
                st.success(f"🐱 **{label}**")
            else:
                st.success(f"🐶 **{label}**")
            
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show all probabilities
            st.markdown("### 📊 Confidence Scores")
            for idx, prob in enumerate(predictions[0]):
                class_name = class_labels[str(idx)].capitalize()
                st.progress(
                    float(prob),
                    text=f"{class_name}: {prob*100:.1f}%"
                )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Powered by MobileNetV2 • Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)