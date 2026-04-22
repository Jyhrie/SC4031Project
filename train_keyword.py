import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
DATA_PATH = "dataset/"
CLASSES = ["other", "hey_home"] 
SAMPLE_RATE = 16000
DURATION = 2  # The new "Safe" window
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256 
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION) # 28,000 samples

def augment_audio(audio):
    """Generates variations in RAM without cutting off the word ends."""
    vars = []
    # 1. Pitch Shifts (No length change)
    vars.append(librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=1))
    vars.append(librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=-1))
        
    return vars

def load_and_preprocess():
    X, y = [], []
    for idx, label in enumerate(CLASSES):
        folder = os.path.join(DATA_PATH, label)
        if not os.path.exists(folder): continue
            
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        print(f"Loading {len(files)} samples for {label}...")
        
        for f in files:
            path = os.path.join(folder, f)
            audio, _ = librosa.load(path, sr=SAMPLE_RATE)
            
            # audio = librosa.util.normalize(audio)
            audio = audio - np.mean(audio)

            if len(audio) < TARGET_SAMPLES:
                audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
            else:
                audio = audio[:TARGET_SAMPLES]

            # Original
            X.append(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
                                        n_fft=N_FFT, hop_length=HOP_LENGTH, center=False).T)
            y.append(idx)

            # Augment ON and OFF to balance the dataset
            if label in ["ON", "OFF", "NOISE_SILENCE"]:
                for v in augment_audio(audio):
                    mfcc = librosa.feature.mfcc(y=v, sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
                                                n_fft=N_FFT, hop_length=HOP_LENGTH, center=False)
                    X.append(mfcc.T)
                    y.append(idx)
            
    return np.array(X), np.array(y)

# 1. Load Data
X, y = load_and_preprocess()
X = X[..., np.newaxis] 

# Shuffle
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

unique, counts = np.unique(y, return_counts=True)
label_counts = dict(zip(CLASSES, counts))

print("--- Total Dataset Distribution ---")
for label, count in label_counts.items():
    percentage = (count / len(y)) * 100
    print(f"{label}: {count} samples ({percentage:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Model Architecture (Bottlenecked for RAM)
model = models.Sequential([
    # Input remains the same (MFCC frames, coefficients, 1 channel)
    layers.Input(shape=(124, 13, 1)),
    
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3), # Prevents overfitting to your specific Pi mic
    layers.Dense(1, activation='sigmoid') # Your single "Is this it?" output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=12, 
    restore_best_weights=True,
    verbose=1
)

# 3. Train
model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# 4. Quantize
def rep_data_gen():
    # Determine how many samples to pick (ensure we don't exceed dataset size)
    num_samples = min(len(X_train), 300)
    
    # Generate random indices without replacement
    random_indices = np.random.choice(len(X_train), num_samples, replace=False)
    
    for i in random_indices:
        # Extract the sample at the random index
        # [i:i+1] ensures the shape remains (1, frames, mfcc, 1)
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8 
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Success! Model Size: {len(tflite_model)/1024:.2f} KB")
print(f"Input Shape: {X.shape[1]} x {X.shape[2]}")