import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

st.set_page_config(
    page_title="PCOS AI Diagnostics",
    page_icon="🏥",
    layout="centered"
)

MODEL_URL = "https://huggingface.co/bhargavi-2005/pcos-model/resolve/main/PCOS_Final_Deployment_Model.keras"


@st.cache_resource
def load_model():
    model_path = "PCOS_Final_Deployment_Model.keras"

    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL)
        with open(model_path, "wb") as f:
            f.write(response.content)

    model = tf.keras.models.load_model(model_path)
    return model


model = load_model()

st.title("🏥 PCOS AI Diagnostics")

uploaded_file = st.file_uploader(
    "Upload Ultrasound Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Ultrasound Image",
        use_container_width=True
    )

    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    try:
        prediction = model.predict(img, verbose=0)

        confidence = float(prediction[0][0])
if confidence < 0.5:
    result = "🩺 PCOS Detected"
    confidence_score = (1 - confidence) * 100
else:
    result = "✅ Normal"
    confidence_score = confidence * 100

        st.markdown("---")

        st.subheader(result)

        st.metric(
            label="Confidence Score",
            value=f"{confidence_score:.2f}%"
        )

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
