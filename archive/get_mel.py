import librosa
import numpy as np

def get_mel_bank(sr=16000, n_fft=512, n_mels=26, fmin=0.0, fmax=8000.0):
    """Returns the Mel filterbank as a NumPy array."""
    # librosa.filters.mel creates a matrix of shape (n_mels, 1 + n_fft // 2)
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)