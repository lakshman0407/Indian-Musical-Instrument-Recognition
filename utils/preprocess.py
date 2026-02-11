import librosa
import numpy as np

# Fixed duration for model (30 seconds)
DESIRED_DURATION = 30
TARGET_SR = 44100  # Standard sampling rate used in training
def prepare_spectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

def load_audio(file_path_or_bytes):
    """
    Loads an audio file (wav/mp3) or recorded bytes.
    Converts to mono and resamples to TARGET_SR (44.1kHz).
    """
    # Load audio file (librosa handles wav, mp3, etc.)
    audio, sr = librosa.load(file_path_or_bytes, sr=TARGET_SR)
    # Trim silence
    audio, _ = librosa.effects.trim(audio)

    # Convert to mono
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    # Normalize
    audio = librosa.util.normalize(audio)

    return audio, TARGET_SR
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize/pad to 128x128
    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)

def trim_pad_audio(audio, sr):
    """
    Ensures audio is exactly DESIRED_DURATION seconds.
    Trims if too long, pads with zeros if too short.
    """
    desired_length = DESIRED_DURATION * sr
    
    if len(audio) > desired_length:
        audio = audio[:desired_length]   # Trim
    else:
        audio = np.pad(audio, (0, desired_length - len(audio)))  # Pad
    
    return audio

def extract_mfcc(audio, sr, n_mfcc=13):
    """
    Extracts MFCC features using the same parameters used during training.
    Window size = 25ms
    Hop length = same as window (no overlap)
    Returns flattened MFCC feature vector.
    """
    window_size_sec = 0.025
    n_fft = int(sr * window_size_sec)
    hop_length = n_fft  # no overlap

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc
    )

    # Flatten (convert 13 x N frame matrix to 1D vector)
    mfcc_flat = mfccs.flatten()
    return mfcc_flat

import joblib
import os

# Load scaler (ensure correct path in Streamlit app)
SCALER_PATH = os.path.join("models", "scaler.pkl")
scaler = joblib.load(SCALER_PATH)

def prepare_features(file_path_or_bytes):
    """
    Complete preprocessing pipeline:
    - Load audio (.wav/.mp3/microphone)
    - Convert to mono
    - Resample to 44.1 kHz
    - Trim/pad to 30 sec
    - Extract MFCC
    - Scale features
    """
    # 1. Load
    audio, sr = load_audio(file_path_or_bytes)

    # 2. Trim/Pad
    audio = trim_pad_audio(audio, sr)

    # 3. MFCC extraction
    mfcc_flat = extract_mfcc(audio, sr)

    # 4. Scale (reshape to 2D for scaler)
    mfcc_scaled = scaler.transform([mfcc_flat])

    return mfcc_scaled
    mel_db = mel_db.reshape(128,128,1)
    mel_db = mel_db / 255.0

    return mel_db
