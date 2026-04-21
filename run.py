import numpy as np
import sounddevice as sd
import librosa

# --- Configuration ---
MODEL_PATH = "model.tflite"
SAMPLE_RATE = 16000
WINDOW_SIZE = 32000  
STEP_SIZE = 8000     
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
# Note: Ensure CLASSES matches your train.py exactly!
CLASSES = ["hey_home", "other"] 
DEVICE_ID = 1  
CONFIDENCE_THRESHOLD = 0.85 

try:
    import tensorflow as tf
    tflite_interpreter = tf.lite.Interpreter
    print("Using Full TensorFlow Engine")
except ImportError:
    import tflite_runtime.interpreter as tflite
    tflite_interpreter = tflite.Interpreter
    print("Using LiteRT Engine")

# 1. Load Model
interpreter = tflite_interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
_, expected_frames, expected_coeffs, _ = input_details['shape']

audio_buffer = np.zeros(WINDOW_SIZE, dtype='float32')

def predict(audio_data):
    # 1. Preprocessing (Matching train.py)
    # Note: train.py used: audio - np.mean(audio)
    audio_data = audio_data - np.mean(audio_data)

    # 2. Feature Extraction with Librosa
    # We use center=False to match your training script's MFCC call
    mfcc = librosa.feature.mfcc(
        y=audio_data, 
        sr=SAMPLE_RATE, 
        n_mfcc=N_MFCC, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        center=False
    ).T  # Transpose to (frames, coefficients)

    # 3. Align frames (Crop or Pad)
    if mfcc.shape[0] > expected_frames:
        mfcc = mfcc[:expected_frames, :]
    elif mfcc.shape[0] < expected_frames:
        mfcc = np.pad(mfcc, ((0, expected_frames - mfcc.shape[0]), (0, 0)), mode='constant')

    # 4. Quantization (Handling the INT8 model)
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    mfcc_quantized = (mfcc / input_scale) + input_zero_point
    mfcc_quantized = mfcc_quantized[np.newaxis, ..., np.newaxis].astype(np.int8)

    # 5. Run Inference
    interpreter.set_tensor(input_details['index'], mfcc_quantized)
    interpreter.invoke()

    # 6. De-quantize Output
    output_data = interpreter.get_tensor(output_details['index'])
    probs = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    return np.argmax(probs), probs[0][np.argmax(probs)]

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status: print(status)
    
    # Slide the window
    audio_buffer = np.roll(audio_buffer, -len(indata))
    audio_buffer[-len(indata):] = indata[:, 0]

    class_idx, score = predict(audio_buffer)
    label = CLASSES[class_idx]

    # Adjust logic: check for your specific wake word
    if label == "hey_home" and score > CONFIDENCE_THRESHOLD:
        print(f">>> WAKE WORD DETECTED: {label} ({score*100:.1f}%)")

# --- Start Listening ---
print("Scanning for Audio Devices...")
print(sd.query_devices())

with sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE_ID, channels=1, 
                    callback=audio_callback, blocksize=STEP_SIZE):
    print(f"--- RPi Listener Active (Threshold: {CONFIDENCE_THRESHOLD}) ---")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")