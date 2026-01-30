import librosa
import numpy as np

# Fixed duration for model (30 seconds)
DESIRED_DURATION = 30
TARGET_SR = 44100  # Standard sampling rate used in training

def load_audio(file_path_or_bytes):
    """
    Loads an audio file (wav/mp3) or recorded bytes.
    Converts to mono and resamples to TARGET_SR (44.1kHz).
    """
    # Load audio file (librosa handles wav, mp3, etc.)
    audio, sr = librosa.load(file_path_or_bytes, sr=TARGET_SR)

    # Convert to mono
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)

    return audio, TARGET_SR
