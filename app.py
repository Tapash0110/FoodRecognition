import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model (cached for performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Function to preprocess image
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((180, 180))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📖 About", "📩 Contact"])

# App title
st.markdown("<h1 style='text-align: center;'>🍏 Poshaq: Food Recognition App 🍏</h1>", unsafe_allow_html=True)

if page == "🏠 Home":
    st.markdown("<h2 style='text-align: center;'>📸 Upload Your Food Image</h2>", unsafe_allow_html=True)
    st.write("Upload an image of food, and our AI model will predict what it is!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        st.write("🔍 **Classifying...**")
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)

        # Extract predicted class
        predicted_class = np.argmax(prediction)
        class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
                       'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
                       'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
                       'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
                       'tomato', 'turnip', 'watermelon']
        
        # Ensure valid class prediction
        result = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"
        
        # Display Prediction
        st.markdown(f"<h3 style='text-align: center;'>🍽️ Prediction: **{result.capitalize()}**</h3>", unsafe_allow_html=True)

elif page == "📖 About":
    st.markdown("<h2>📖 About Us</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.write("""
        Welcome to **Poshaq**, your AI-powered food recognition platform! Our website leverages **machine learning** 
        to identify various fruits and vegetables from images, making food classification **easier and more efficient**. 
        
        Our **robust AI model** is trained on a diverse dataset to provide **accurate predictions** for food items. 
        Whether you're a **researcher, developer, or food enthusiast**, our tool offers a seamless experience.

        With growing interest in **nutrition, food tracking, and inventory management**, our platform simplifies 
        food identification through **AI-driven technology**. Try it out and explore the **power of AI** in food recognition!
        
