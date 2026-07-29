import streamlit as st
import pickle
from PIL import Image
import numpy as np
import base64

# Load the model and scaler
# Ensure these files (model.pkl, scaler.pkl) are in the same directory as your Streamlit app
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or scaler files not found. Make sure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Function to preprocess the image
def preprocess_image(image):
    img = image.convert('L') # Convert to grayscale
    img = img.resize((64, 64)) # Resize to the expected input size
    img_array = np.array(img).flatten() # Flatten the image to 1D array
    img_array = img_array.reshape(1, -1) # Reshape for scaler (1 sample, 4096 features)
    img_scaled = scaler.transform(img_array)
    return img_scaled

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{background-image: url(data:image/png;base64,{b64_encoded});
                 background-size: cover;}}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Streamlit App --- #
st.title("Smiling Face Detector")

# Set background image (assuming bgimg.png exists in the same directory)
set_background('bgimg.png')

st.sidebar.header("Choose Input Method")
choice = st.sidebar.radio(
    "Select input method:",
    ("Upload Image", "Use Camera", "Use Saved Image")
)

prediction_placeholder = st.empty()

if choice == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        if st.button("Classify Uploaded Image"):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            if prediction[0] == 1:
                prediction_placeholder.success("Prediction: Smiling Face!")
            else:
                prediction_placeholder.info("Prediction: Non-Smiling Face.")

elif choice == "Use Camera":
    st.warning("Camera access is not directly supported in all Streamlit environments. You may need a dedicated component or run locally.")
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption='Camera Input', use_column_width=True)
        st.write("")
        if st.button("Classify Camera Image"):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            if prediction[0] == 1:
                prediction_placeholder.success("Prediction: Smiling Face!")
            else:
                prediction_placeholder.info("Prediction: Non-Smiling Face.")

elif choice == "Use Saved Image":
    # For demonstration, let's assume a saved image path. 
    # In a real app, you might have a dropdown of available images or a text input.
    saved_image_path = st.text_input("Enter path to a saved image (e.g., 'test_image.jpg'):", "test_image.jpg")
    if saved_image_path:
        try:
            image = Image.open(saved_image_path)
            st.image(image, caption='Saved Image', use_column_width=True)
            st.write("")
            if st.button("Classify Saved Image"):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                if prediction[0] == 1:
                    prediction_placeholder.success("Prediction: Smiling Face!")
                else:
                    prediction_placeholder.info("Prediction: Non-Smiling Face.")
        except FileNotFoundError:
            st.error(f"Saved image not found at '{saved_image_path}'. Please ensure the path is correct.")
        except Exception as e:
            st.error(f"Error loading saved image: {e}")
