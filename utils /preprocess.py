import librosa
import numpy as np

# Fixed duration for model (30 seconds)
DESIRED_DURATION = 30
TARGET_SR = 44100  # Standard sampling rate used in training
