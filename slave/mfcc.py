import numpy as np
from config import *

from audio_weights import MEL_FILTER_BANK, DCT_MATRIX

HANN_WINDOW = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / (N_FFT - 1)))

def compute_manual_mfcc(audio):
    """Replicates computeMFCC() from Arduino"""
    audio = audio - np.mean(audio)
    mfcc_buf = np.zeros((N_FRAMES, N_MFCC))

    for frame_idx in range(N_FRAMES):
        start = frame_idx * HOP_LENGTH
        if start + N_FFT > len(audio):
            break

        frame = audio[start: start + N_FFT] * HANN_WINDOW
        fft_res = np.fft.rfft(frame, n=N_FFT)
        power_spectrum = np.abs(fft_res) ** 2
        mel_spectrum = np.dot(MEL_FILTER_BANK, power_spectrum)
        safe_mel = np.maximum(mel_spectrum, 1e-10)
        log_mel = 10.0 * np.log10(safe_mel)
        mfcc_buf[frame_idx] = np.dot(DCT_MATRIX, log_mel)

    return mfcc_buf

# def compute_mfcc(audio):
#     audio = audio - np.mean(audio)
#     out = np.zeros((N_FRAMES, N_MFCC))

#     for i in range(N_FRAMES):
#         start = i * HOP_LENGTH
#         if start + N_FFT > len(audio):
#             break

#         frame = audio[start:start+N_FFT] * HANN
#         fft = np.fft.rfft(frame)
#         power = np.abs(fft) ** 2

#         mel = np.dot(MEL_FILTER_BANK, power)
#         mel = np.maximum(mel, 1e-10)

#         log_mel = 10 * np.log10(mel)
#         out[i] = np.dot(DCT_MATRIX, log_mel)

#     return out