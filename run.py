import numpy as np
import sounddevice as sd
from python_speech_features import mfcc as psf_mfcc

# --- Configuration (Keep these identical to your PC) ---
MODEL_PATH = "model.tflite"
SAMPLE_RATE = 16000
WINDOW_SIZE = 32000  
STEP_SIZE = 8000     
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 26
CLASSES = ["ON", "OFF", "NOISE_SILENCE"] 
DEVICE_ID = 1  # Note: This often changes on RPi (usually 0 or 1)
CONFIDENCE_THRESHOLD = 0.85 

try:
    import tflite_runtime.interpreter as tflite
    tflite_interpreter = tflite.Interpreter
    print("Using LiteRT Engine")
except ImportError:
    import tensorflow as tf
    tflite_interpreter = tf.lite.Interpreter
    print("Using Full TensorFlow Engine")

# 1. Load Model with TFLite Runtime
interpreter = tflite_interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
_, expected_frames, expected_coeffs, _ = input_details['shape']

audio_buffer = np.zeros(WINDOW_SIZE, dtype='float32')

def predict(audio_data):
    # Standardize & Normalize
    audio_data = audio_data - np.mean(audio_data)
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    # Features
    mfcc = psf_mfcc(audio_data, samplerate=SAMPLE_RATE, winlen=N_FFT/SAMPLE_RATE,
                    winstep=HOP_LENGTH/SAMPLE_RATE, numcep=N_MFCC, nfilt=N_MELS,
                    nfft=N_FFT, preemph=0, winfunc=np.hanning)

    # Align frames
    if mfcc.shape[0] > expected_frames:
        mfcc = mfcc[:expected_frames, :]
    elif mfcc.shape[0] < expected_frames:
        mfcc = np.pad(mfcc, ((0, expected_frames - mfcc.shape[0]), (0, 0)), mode='constant')

    # Quantization
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    mfcc_quantized = (mfcc / input_scale) + input_zero_point
    mfcc_quantized = mfcc_quantized[np.newaxis, ..., np.newaxis].astype(np.int8)

    interpreter.set_tensor(input_details['index'], mfcc_quantized)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])
    probs = (output_data.astype(np.float32) - output_zero_point) * output_scale
    return np.argmax(probs), probs[0][np.argmax(probs)]

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status: print(status)
    
    audio_buffer = np.roll(audio_buffer, -len(indata))
    audio_buffer[-len(indata):] = indata[:, 0]

    class_idx, score = predict(audio_buffer)
    label = CLASSES[class_idx]

    if label != "NOISE_SILENCE" and score > CONFIDENCE_THRESHOLD:
        print(f">>> COMMAND: {label} ({score*100:.1f}%)")
        # Trigger Pi-specific action here (e.g., GPIO)

# --- Find your Microphone ---
print("Scanning for Audio Devices...")
print(sd.query_devices())

# Update DEVICE_ID based on the printed list if necessary
with sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE_ID, channels=1, 
                    callback=audio_callback, blocksize=STEP_SIZE):
    print(f"--- RPi Listener Active (Threshold: {CONFIDENCE_THRESHOLD}) ---")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")