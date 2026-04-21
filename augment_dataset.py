import librosa
import numpy as np
import soundfile as sf
import random
import os

DURATION = 2.0 
SR = 16000
TARGET_SIZE = int(SR * DURATION)

def augment_audio(y, sr):
    # 1. Pitch and Speed (as before)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
    y = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))
    
    # 2. Random Start Time
    if len(y) < TARGET_SIZE:
        padding = TARGET_SIZE - len(y)
        offset = random.randint(0, padding)
        y = np.pad(y, (offset, padding - offset), mode='constant')
    else:
        y = y[:TARGET_SIZE]
    
    # 3. OVERLAY STATIC / BACKGROUND NOISE
    # We mix a random amount of 'hiss' (white noise) 
    # and 'brown noise' (deeper static)
    noise_level = random.uniform(0.0001, 0.0015) 
    static = np.random.normal(size=y.shape)
    y_with_noise = y + (noise_level * static)
    
    return y_with_noise

def process_and_multiply(src, dest, multiply_by=10):
    if not os.path.exists(dest): os.makedirs(dest)
    files = [f for f in os.listdir(src) if f.endswith(".wav") or f.endswith(".mp3")]
    
    for file in files:
        y, sr = librosa.load(os.path.join(src, file), sr=SR)
        for i in range(multiply_by):
            aug_y = augment_audio(y, sr)
            sf.write(f"{dest}/aug_{i}_{file.replace('.mp3', '.wav')}", aug_y, SR)

# Run the process
process_and_multiply("dataset/raw_hey_home", "dataset/hey_home", 10)
process_and_multiply("dataset/raw_other", "dataset/other", 10)