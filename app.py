import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import tempfile
from tflite_runtime.interpreter import Interpreter
from utils.preprocess import prepare_spectrogram   # YOUR MEL/STFT FUNCTION

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Indian Musical Instrument Classifier")
st.title("🎵 Indian Musical Instrument Recognition (U-Net Model)")

# ----------------------------
# Load Label Encoder
# ----------------------------
label_encoder = joblib.load("label_encoder.pkl")

# ----------------------------
# Load TFLite Model
# ----------------------------
interpreter = Interpreter(model_path="unet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# Predict Function
# ----------------------------
def predict_audio(file_path):
    spec = prepare_spectrogram(file_path)   # shape (128,128,1)
    spec = np.expand_dims(spec, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], spec)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output)

    return label_encoder.inverse_transform([pred_class])[0]

# ----------------------------
# Upload Section
# ----------------------------
st.header("📁 Upload Audio File")

uploaded = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])

if uploaded:
    st.success("File uploaded!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded.read())
        temp_path = temp_audio.name

    if st.button("🎯 Predict Uploaded File"):
        result = predict_audio(temp_path)

        st.markdown(
            f"""
            <h2 style='text-align:center;'>
            🎯 <span style='color:#FF5733;'>Predicted Instrument:</span>
            <b style='color:green;'>{result}</b>
            </h2>
            """,
            unsafe_allow_html=True
        )

# ----------------------------
# Live Recording
# ----------------------------
st.header("🎙️ Live Audio Recording")

audio_bytes = st.audio_input("Record here:")

if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes.read())
        temp_path = temp_audio.name

    if st.button("🎯 Predict Recorded Audio"):
        result = predict_audio(temp_path)
        st.success(f"Predicted Instrument: {result}")
