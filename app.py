import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model("papaya_disease_model.h5")
classes = ['Anthracnose', 'Black_Spot', 'Ring_Spot', 'Phytophthora', 'Powdery_Mildew', 'Good']

# Set the title and layout
st.set_page_config(page_title="Papaya Disease Detection", layout="wide")
st.title("🌿 Papaya Disease Detection")
st.write("Upload an image of a papaya  for prediction.")

# Add a sidebar for additional information
st.sidebar.header("About")
st.sidebar.write("This application uses a deep learning model to detect diseases in papaya .")
st.sidebar.write("Upload a clear image of a papaya  to get a prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display the prediction result
    st.subheader("Prediction Result")
    st.write(f"**Disease:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Add a button to clear the image
    if st.button("Clear Image"):
        st.experimental_rerun()

# Add footer
st.markdown("---")
st.write("Made with ❤️ by khubchand & irfan")