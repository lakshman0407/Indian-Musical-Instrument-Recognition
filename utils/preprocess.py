import librosa
import numpy as np

def prepare_spectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    # Trim silence
    audio, _ = librosa.effects.trim(audio)

    # Normalize
    audio = librosa.util.normalize(audio)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize/pad to 128x128
    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)

    mel_db = mel_db.reshape(128,128,1)
    mel_db = mel_db / 255.0

    return mel_db
