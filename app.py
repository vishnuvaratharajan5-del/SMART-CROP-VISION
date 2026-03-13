# app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 1. Load the saved model
model_path = 'models/plant_disease_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Trained model not found. Please run train_model.py first.")

# 2. Get class names based on your dataset/train subfolders
train_dir = 'dataset/train'
class_names = []

for entry in os.scandir(train_dir):
    if entry.is_dir():
        class_names.append(entry.name)

class_names.sort()

# Optional debug print
# print("Class names:", class_names)

# 3. Streamlit Web App UI
st.title("🌿 Smart Crop Vision")
st.write("Upload a plant leaf image to detect disease and get treatment suggestions.")

# 4. File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

# 5. If an image is uploaded
if uploaded_file is not None:
    # Load and display image
    img = image.load_img(uploaded_file, target_size=(256,256))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_class = class_names[pred_index]
    confidence = np.max(prediction)

    # Display results
    st.markdown(f"### 🌟 **Prediction:** {pred_class}")
    st.markdown(f"### 🔎 **Confidence:** {confidence*100:.2f}%")

    # 6. Treatment suggestions (basic example)
    if "Black_rot" in pred_class:
        st.write("💡 **Treatment Suggestion:** Prune infected branches and apply appropriate fungicide sprays.")
    elif "Late_blight" in pred_class:
        st.write("💡 **Treatment Suggestion:** Use fungicides containing chlorothalonil or copper-based sprays.")
    elif "healthy" in pred_class:
        st.write("✅ **Healthy:** No disease detected. Keep monitoring crop health regularly.")
    else:
        st.write("💡 **Treatment Suggestion:** No specific treatment data available. Consult an agronomist.")
