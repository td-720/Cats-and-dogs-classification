import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and let AI classify it as a cat or dog!")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('cats_dogs_mobilenet.h5')
        with open('class_labels.json', 'r') as f:
            labels = json.load(f)
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

model, class_labels = load_model()

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of a cat or dog"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', width='stretch')
    
    with col2:
        with st.spinner('🔍 Analyzing image...'):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            st.markdown("### 🎯 Prediction")
            label = class_labels[str(predicted_class)].upper()
            
            if 'CAT' in label.upper():
                st.success(f"🐱 **{label}**")
            elif 'DOG' in label.upper():
                st.success(f"🐶 **{label}**")
            else:
                st.info(f"**{label}**")
            
            st.metric("Confidence", f"{confidence:.1f}%")
            
            st.markdown("### 📊 Confidence Scores")
            for idx, prob in enumerate(predictions[0]):
                class_name = class_labels[str(idx)].capitalize()
                st.progress(float(prob), text=f"{class_name}: {prob*100:.1f}%")

st.markdown("---")
st.markdown(
    "<div style='text-align: center'><p>Powered by MobileNetV2 • Built with Streamlit</p></div>",
    unsafe_allow_html=True
)