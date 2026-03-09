import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
from utils.preprocess import prepare_features

# ---------------------------------------------------
# Page Setup
# ---------------------------------------------------
st.set_page_config(
    page_title="Indian Musical Instrument Recognition",
    page_icon="🎵",
    layout="centered"
)

# ---------------------------------------------------
# Custom Styling
# ---------------------------------------------------
st.markdown("""
<style>
.big-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:gray;
    font-size:18px;
}

.result-box {
    padding:20px;
    border-radius:10px;
    background-color:#0E1117;
    text-align:center;
    font-size:28px;
    border:1px solid #444;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🎵 Indian Musical Instrument Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an audio sample to identify the instrument</div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_model()

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------
def predict_instrument(features):
    pred = model.predict(features)[0]
    return label_encoder.inverse_transform([pred])[0]

# ---------------------------------------------------
# Upload Section
# ---------------------------------------------------
st.header("📁 Upload Audio File")

uploaded = st.file_uploader(
    "Upload .wav or .mp3 file",
    type=["wav", "mp3"]
)

if uploaded:

    st.success("Audio file uploaded successfully!")

    # Play audio
    st.audio(uploaded)

    # File details
    st.info(f"File Name: {uploaded.name}")

    st.divider()

    if st.button("🎯 Predict Instrument"):

        with st.spinner("Analyzing audio..."):

            try:
                features = prepare_features(uploaded)

                result = predict_instrument(features)

                st.markdown(
                    f"""
                    <div class="result-box">
                    🎯 Predicted Instrument <br>
                    <span style="color:#00FFAA;font-weight:bold">{result}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error("⚠ Error while processing audio.")
                st.write(e)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.divider()
st.caption("Developed using Machine Learning | Random Forest Model")
