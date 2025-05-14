import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import cnn_model as cnn
import svm_model_crop as svm
import hybrid_model as hybrid
import os

# Page configuration
st.set_page_config(page_title="AI Disease Detection", layout="centered")
st.markdown("<h1 style='text-align: center; color: #0b3d0b;'>🩺 AI Disease Detection Tool</h1>", unsafe_allow_html=True)
st.markdown("---")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Model selector
model_choice = st.selectbox("Select model to use", ["SVM", "CNN", "Hybrid"])

# Display uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown("### Uploaded Image:")
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    st.pyplot(fig)

    # Temporary save image to disk for model input
    temp_path = os.path.join("temp_uploaded_image.jpg")
    image.save(temp_path)

    # Prediction
    if st.button("Check for Disease"):
        try:
            result = None
            if model_choice == "CNN":
                result = cnn.predict_image(temp_path)
            elif model_choice == "SVM":
                result = svm.use_svm_model(temp_path)
            elif model_choice == "Hybrid":
                result = hybrid.hybrid_model(temp_path)

            if result:
                st.success("✅ Disease Present")
            else:
                st.info("✅ Disease Free")
        except Exception as e:
            st.error(f"❌ Error: {e}")
