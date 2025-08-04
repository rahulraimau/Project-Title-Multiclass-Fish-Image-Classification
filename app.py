import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load your saved model
MODEL_PATH = 'best_fish_classifier_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (order must match training generator)
class_names = ['Salmon', 'Tuna', 'Trout', 'Cod', 'Mackerel']  # ← Replace with your actual class labels

st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")
st.title("🎣 Multiclass Fish Image Classifier")
st.markdown("Upload an image of a fish and the model will predict the species.")

# File uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show prediction
    st.subheader("🔎 Prediction")
    st.write(f"**Predicted Fish Category:** {predicted_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # Show confidence bar chart
    st.subheader("📊 Confidence Scores:")
    st.bar_chart({class_names[i]: float(prediction[i]) for i in range(len(class_names))})
