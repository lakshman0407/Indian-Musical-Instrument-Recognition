import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import io

from utils.preprocess import prepare_features

# -------------------------------------------
# Page Configuration
# -------------------------------------------
st.set_page_config(
    page_title="Indian Musical Instrument Classifier",
    layout="centered"
)

st.title("🎵 Indian Musical Instrument Classification")
st.write("Upload audio (.wav /.mp3) or record live audio to predict the instrument.")

# -------------------------------------------
# Load Saved Models
# -------------------------------------------
model = joblib.load("models/random_forest_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# -------------------------------------------
# Prediction Function
# -------------------------------------------
def predict_instrument(audio_features):
    pred_class = model.predict(audio_features)[0]
    instrument_name = label_encoder.inverse_transform([pred_class])[0]
    return instrument_name


# ---------------------------------------------------------
# BLOCK 1 — FILE UPLOAD (WAV / MP3)
# ---------------------------------------------------------
st.header("📁 Upload Audio File")

uploaded_file = st.file_uploader(
    "Upload .wav or .mp3 file",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.success("Audio uploaded successfully!")

    # Preprocess uploaded file → MFCC → scaled features
    features = prepare_features(uploaded_file)

    # Predict instrument
    prediction = predict_instrument(features)

    st.subheader("🎯 Predicted Instrument")
    st.write(f"**{prediction}**")


# ---------------------------------------------------------
# BLOCK 2 — LIVE AUDIO RECORDING
# ---------------------------------------------------------
st.header("🎙️ Record Live Audio")

audio_bytes = st.audio_input("Click below to record audio")

if audio_bytes is not None:
    st.success("Live audio recorded successfully!")

    # Convert bytes → numpy audio using soundfile
    audio_np, sr = sf.read(io.BytesIO(audio_bytes))

    # librosa expects float32
    audio_np = audio_np.astype(np.float32)

    # Preprocess → MFCC → scaled
    features = prepare_features(io.BytesIO(audio_bytes))

    # Predict
    prediction = predict_instrument(features)

    st.subheader("🎯 Predicted Instrument")
    st.write(f"**{prediction}**")


# Footer
st.markdown("---")
st.write("Built with ❤️ for Indian Music Classification")
