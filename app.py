import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import cv2

st.set_page_config(
    page_title="PCOS AI Diagnostics",
    page_icon="🏥",
    layout="centered"
)

MODEL_URL = "https://huggingface.co/bhargavi-2005/pcos-model/resolve/main/PCOS_Final_Deployment_Model.keras"


def preprocess_image(image):

    image = image.astype(np.uint8)

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))

    final = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2RGB
    )

    return final


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

    img = np.array(img)

    # SAME PREPROCESSING USED IN TRAINING
    img = preprocess_image(img)

    img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)

    try:

        prediction = model.predict(
            img,
            verbose=0
        )

        confidence = float(prediction[0][0])

        # DEBUG (remove later if desired)
        st.write("Raw Prediction:", confidence)

        # infected = PCOS (class 0)
        # noninfected = Normal (class 1)

        if confidence < 0.5:

            result = "🩺 PCOS Detected"

            confidence_score = (
                1 - confidence
            ) * 100

        else:

            result = "✅ Normal"

            confidence_score = (
                confidence
            ) * 100

        st.markdown("---")

        st.subheader(result)

        st.metric(
            "Confidence Score",
            f"{confidence_score:.2f}%"
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )
