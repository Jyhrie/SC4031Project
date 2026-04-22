import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_PATH = "dataset/"
CLASSES = ["other", "hey_home"]
SAMPLE_RATE = 16000
DURATION = 2
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)


def augment_audio(audio):
    return [
        librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=1),
        librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=-1),
    ]


def load_and_preprocess():
    X, y = [], []

    for idx, label in enumerate(CLASSES):
        folder = os.path.join(DATA_PATH, label)
        if not os.path.exists(folder):
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        print(f"Loading {len(files)} samples for {label}...")

        for f in files:
            path = os.path.join(folder, f)
            audio, _ = librosa.load(path, sr=SAMPLE_RATE)

            audio = audio - np.mean(audio)

            if len(audio) < TARGET_SAMPLES:
                audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
            else:
                audio = audio[:TARGET_SAMPLES]

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                center=False
            ).T

            X.append(mfcc)
            y.append(idx)

            # augmentation
            if label == "hey_home":
                for v in augment_audio(audio):
                    mfcc_aug = librosa.feature.mfcc(
                        y=v,
                        sr=SAMPLE_RATE,
                        n_mfcc=N_MFCC,
                        n_fft=N_FFT,
                        hop_length=HOP_LENGTH,
                        center=False
                    ).T
                    X.append(mfcc_aug)
                    y.append(idx)

    return np.array(X), np.array(y)


# 1. Load dataset
X, y = load_and_preprocess()
X = X[..., np.newaxis]

# shuffle
idxs = np.arange(len(X))
np.random.shuffle(idxs)
X, y = X[idxs], y[idxs]

# ----------------------------
# SPLIT: train / val / test
# ----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# ----------------------------
# MODEL
# ----------------------------
model = models.Sequential([
    layers.Input(shape=(124, 13, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2), 
    
    layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.SeparableConv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(), 
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), 
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ----------------------------
# TRAIN
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32
)


# ----------------------------
# TEST EVALUATION
# ----------------------------
print("\nEvaluating test set...")
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int).reshape(-1)


# ----------------------------
# CLASSIFICATION REPORT
# ----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))


# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ----------------------------
# SAVE TFLITE (unchanged)
# ----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model)/1024:.2f} KB")