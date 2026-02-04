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
# BLOCK 1 — FILE UPLOAD
# ---------------------------------------------------------
st.header("📁 Upload Audio File")

uploaded_file = st.file_uploader("Upload .wav or .mp3 file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.success("Audio uploaded successfully!")

    predict_file = st.button("🎯 Predict Uploaded Audio")

    if predict_file:
        features = prepare_features(uploaded_file)
        prediction = predict_instrument(features)

        st.markdown(
            f"""
            <h2 style='text-align:center;'>
                🎯 <span style='color:#FF5733;'>Predicted Instrument:</span> 
                <span style='color:#008000;'><b>{prediction}</b></span>
            </h2>
            """,
            unsafe_allow_html=True
        )




# ---------------------------------------------------------
# BLOCK 2 — LIVE AUDIO RECORDING
# ---------------------------------------------------------
st.header("🎙️ Record Live Audio")

audio_bytes = st.audio_input("Click below to record audio")

if audio_bytes is not None:
    st.success("Live audio recorded successfully!")

    predict_record = st.button("🎯 Predict Recorded Audio")

    if predict_record:

        try:
            # Convert UploadedFile → bytes → io buffer
            audio_file = io.BytesIO(audio_bytes.read())

            # Decode using soundfile
            audio_np, sr = sf.read(audio_file, dtype="float32", always_2d=False)

            # Preprocess → MFCC
            features = prepare_features(io.BytesIO(audio_bytes.read()))

            # Predict
            prediction = predict_instrument(features)

            # Colorful, big output
            st.markdown(
                f"""
                <h2 style='text-align:center;'>
                    🎯 <span style='color:#FF5733;'>Predicted Instrument:</span> 
                    <span style='color:#008000;'><b>{prediction}</b></span>
                </h2>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error("⚠ Unable to process recorded audio.")
            st.write(e)



# Footer
st.markdown("---")
st.write("Built with ❤️ for Indian Music Classification")
