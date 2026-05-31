import streamlit as st

st.set_page_config(
    page_title="PCOS AI Diagnostics",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 PCOS AI Diagnostics")

uploaded_file = st.file_uploader(
    "Upload Ultrasound Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.button("Analyze")
