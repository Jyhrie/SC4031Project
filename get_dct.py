import librosa
import numpy as np

# Configuration
SR = 16000
N_FFT = 512
N_MELS = 26
N_MFCC = 13
FMIN = 0.0
FMAX = SR / 2.0

def generate_python_module():
    # 1. Generate Mel Filterbank
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX) #

    # 2. Generate DCT Matrix
    dct_matrix = np.zeros((N_MFCC, N_MELS)) #
    for i in range(N_MFCC):
        for j in range(N_MELS):
            dct_matrix[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N_MELS)) #
        if i == 0:
            dct_matrix[i, :] *= np.sqrt(1.0 / N_MELS) #
        else:
            dct_matrix[i, :] *= np.sqrt(2.0 / N_MELS) #

    # 3. Write to a .py file
    with open("audio_weights.py", "w") as f:
        f.write("import numpy as np\n\n")
        f.write(f"# Generated Weights for SR={SR}, N_FFT={N_FFT}\n")
        f.write(f"MEL_FILTER_BANK = np.array({mel_basis.tolist()})\n\n")
        f.write(f"DCT_MATRIX = np.array({dct_matrix.tolist()})\n")

    print("Success! Created 'audio_weights.py'")

if __name__ == "__main__":
    generate_python_module()