import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

# Try to import TensorFlow lazily and handle missing dependency gracefully
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    TF_AVAILABLE = True
    TF_IMPORT_ERROR = None
except Exception as e:
    tf = None
    preprocess_input = None
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(e)

# Page setup
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🫁 Pneumonia Detection from Chest X-Ray")

# Class labels
class_names = ["Normal", "Pneumonia"]

# Load model (safe)
@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        raise ImportError(f"TensorFlow is not available in the environment: {TF_IMPORT_ERROR}")
    model_path = Path("fine_tuning_with_resnet.keras")
    if not model_path.exists():
        # Optional: if you prefer automatic download, implement logic here to fetch the model from a URL
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Add the model to the repo, enable Git LFS, or provide a MODEL_URL to download at startup."
        )
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model

# Try loading model and show friendly errors instead of crashing
try:
    model = load_model()
except Exception as e:
    st.error("Model could not be loaded.")
    st.write("Troubleshooting steps:")
    st.write("- Ensure TensorFlow is listed in `requirements.txt` and compatible with the Streamlit deployment Python version.")
    st.write("- Ensure `fine_tuning_with_resnet.keras` is present in the repository or configure an automatic download.")
    st.exception(e)
    st.stop()

# Automatically get model input size
input_shape = model.input_shape  # (None, H, W, C)
input_h, input_w = input_shape[1], input_shape[2]
st.caption(f"Model input size: {input_h} × {input_w} — images will be resized and preprocessed automatically")

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((input_w, input_h))
    img_array = np.array(img).astype(np.float32)
    img_array = preprocess_input(img_array)  # ResNet preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)

    # Determine if binary or multi-class
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        # Binary model (single sigmoid output)
        score = float(prediction[0, 0])
        # Apply sigmoid if necessary
        if score < 0.0 or score > 1.0:
            score = 1.0 / (1.0 + np.exp(-score))
        predicted_label = class_names[1] if score >= 0.5 else class_names[0]
        confidence = score * 100.0
    else:
        # Multi-class model (softmax)
        probs = tf.nn.softmax(prediction, axis=-1).numpy()
        predicted_index = int(np.argmax(probs))
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(probs)) * 100.0

    # Display result
    st.subheader("Prediction")
    st.write(f"**{predicted_label}** with confidence: **{confidence:.2f}%**")

