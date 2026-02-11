import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import tensorflow as tf

# Page setup
st.set_page_config(page_title="Indian Musical Instrument Classifier")
st.title("🎵 Indian Musical Instrument Recognition")

# Load label encoder
label_encoder = joblib.load("models/label_encoder.pkl")

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
        spec = prepare_spectrogram(uploaded)
        result = predict_instrument(spec)

        st.markdown(f"""
        <h2 style='text-align:center;'>
        🎯 Predicted Instrument: <span style='color:green;'>{result}</span>
        </h2>
        """, unsafe_allow_html=True)

# Live Recording
st.header("🎙️ Live Audio Recording")
audio_bytes = st.audio_input("Record here:")

if audio_bytes:
    st.success("Audio recorded!")
    if st.button("🎯 Predict Recorded Audio"):
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
