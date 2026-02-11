import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import io
import tempfile
from utils.preprocess import prepare_features
import tensorflow as tf

# ---------------------------------------------------
# Page Setup
# ---------------------------------------------------
# Page setup
st.set_page_config(page_title="Indian Musical Instrument Classifier")

st.title("🎵 Indian Musical Instrument Recognition")
st.write("Upload an audio file or record audio for instrument classification.")

# ---------------------------------------------------
# Load Model + Encoder
# ---------------------------------------------------
model = joblib.load("models/random_forest_model.pkl")
# Load label encoder
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
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="unet_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Spectrogram function
def prepare_spectrogram(file):
    y, sr = librosa.load(file, sr=22050)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = spec / 255.0
    spec = np.expand_dims(spec, axis=-1)
    spec = np.expand_dims(spec, axis=0)
    return spec.astype(np.float32)

# Prediction
def predict_instrument(spec):
    interpreter.set_tensor(input_details[0]['index'], spec)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    return label_encoder.inverse_transform([np.argmax(pred)])[0]

# Upload file
st.header("📁 Upload Audio File")

uploaded = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])

if uploaded:
    st.success("File uploaded!")

if st.button("🎯 Predict Uploaded File"):
        features = prepare_features(uploaded)
        result = predict_instrument(features)
        spec = prepare_spectrogram(uploaded)
        result = predict_instrument(spec)

        st.markdown(
            f"""
            <h2 style='text-align:center;'>
            🎯 <span style='color:#FF5733;'>Predicted Instrument:</span> 
            <b><span style='color:#008000;'>{result}</span></b>
            </h2>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <h2 style='text-align:center;'>
        🎯 Predicted Instrument: <span style='color:green;'>{result}</span>
        </h2>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# LIVE AUDIO RECORDING SECTION
# ---------------------------------------------------
# Live Recording
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes.read())
            temp_path = temp_audio.name

        spec = prepare_spectrogram(temp_path)
        result = predict_instrument(spec)

        st.markdown(f"""
        <h2 style='text-align:center;'>
        🎯 Predicted Instrument: <span style='color:green;'>{result}</span>
        </h2>
        """, unsafe_allow_html=True)
