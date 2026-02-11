import librosa
import numpy as np
import joblib
import os

DESIRED_DURATION = 30  # seconds

# Load scaler
scaler = joblib.load("models/scaler.pkl")

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)

    # Silence removal
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Normalize
    audio = librosa.util.normalize(audio)

    return audio, sr


def trim_pad_audio(audio, sr):
    desired_len = DESIRED_DURATION * sr

    if len(audio) > desired_len:
        audio = audio[:desired_len]
    else:
        audio = np.pad(audio, (0, desired_len - len(audio)))

    return audio


def extract_mfcc(audio, sr, n_mfcc=13):
    window_size_sec = 0.025
    n_fft = int(sr * window_size_sec)
    hop_length = n_fft   # no overlap

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc
    )

    return mfcc.flatten()


def prepare_features(file_path):
    # Load
    audio, sr = load_audio(file_path)

    # Trim/pad
    audio = trim_pad_audio(audio, sr)

    # Extract MFCC
    mfcc_flat = extract_mfcc(audio, sr)

    # Scale
    mfcc_scaled = scaler.transform([mfcc_flat])

    return mfcc_scaled
