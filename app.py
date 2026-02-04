import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import io
import tempfile
from utils.preprocess import prepare_features

# ---------------------------------------------------
# Page Setup
# ---------------------------------------------------
st.set_page_config(page_title="Indian Musical Instrument Classifier")

st.title("🎵 Indian Musical Instrument Recognition")
st.write("Upload an audio file or record audio for instrument classification.")

# ---------------------------------------------------
# Load Model + Encoder
# ---------------------------------------------------
model = joblib.load("models/random_forest_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ---------------------------------------------------
# Predict Function
# ---------------------------------------------------
def predict_instrument(features):
    pred = model.predict(features)[0]
    return label_encoder.inverse_transform([pred])[0]

# ---------------------------------------------------
# FILE UPLOAD SECTION
# ---------------------------------------------------
st.header("📁 Upload Audio File")

uploaded = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])

if uploaded:
    st.success("File uploaded!")

    if st.button("🎯 Predict Uploaded File"):
        features = prepare_features(uploaded)
        result = predict_instrument(features)

        st.markdown(
            f"""
            <h2 style='text-align:center;'>
            🎯 <span style='color:#FF5733;'>Predicted Instrument:</span> 
            <b><span style='color:#008000;'>{result}</span></b>
            </h2>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------
# LIVE AUDIO RECORDING SECTION
# ---------------------------------------------------
st.header("🎙️ Live Audio Recording")

audio_bytes = st.audio_input("Record here:")

if audio_bytes:
    st.success("Audio recorded!")

    if st.button("🎯 Predict Recorded Audio"):
        try:
            # Save recorded bytes into a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes.read())
                temp_path = temp_audio.name

            # Preprocess using your correct MFCC pipeline
            features = prepare_features(temp_path)

            # Predict
            result = predict_instrument(features)

            # Output
            st.markdown(
                f"""
                <h2 style='text-align:center;'>
                🎯 <span style='color:#FF5733;'>Predicted Instrument:</span> 
                <b><span style='color:#008000;'>{result}</span></b>
                </h2>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error("⚠ Could not process recorded audio.")
            st.write(e)
