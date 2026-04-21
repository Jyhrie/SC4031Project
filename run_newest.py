import numpy as np
import sounddevice as sd
from audio_weights import MEL_FILTER_BANK, DCT_MATRIX

# --- Configuration (Matches tinyml_proj.ino) ---
MODEL_PATH = "model.tflite"
SAMPLE_RATE = 16000
WINDOW_SIZE = 32000  # TARGET_SAMPLES
STEP_SIZE = 8000     
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
N_FRAMES = 124
CLASSES = ["ON", "OFF", "UNKNOWN"]
DEVICE_ID = 2
CONFIDENCE_THRESHOLD = 0.80



# Pre-compute Hann Window once
HANN_WINDOW = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / (N_FFT - 1)))
print("Using Device ID:", DEVICE_ID)

try:
    import tflite_runtime.interpreter as tflite
    tflite_interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    tflite_interpreter = tf.lite.Interpreter

# 1. Initialize TFLite
interpreter = tflite_interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

audio_buffer = np.zeros(WINDOW_SIZE, dtype='float32')

def compute_manual_mfcc(audio):
    """Replicates computeMFCC() from Arduino [cite: 32-45]"""
    # 1. DC Offset Removal [cite: 33]
    audio = audio - np.mean(audio)
    
    mfcc_buf = np.zeros((N_FRAMES, N_MFCC))
    
    # 2. Frame Processing Loop [cite: 35]
    for frame_idx in range(N_FRAMES):
        start = frame_idx * HOP_LENGTH
        if start + N_FFT > len(audio): break
            
        # Step 1: Apply Hann Window [cite: 36]
        # Audio is already float32 (-1.0 to 1.0), matching Arduino scaling [cite: 36]
        frame = audio[start : start + N_FFT] * HANN_WINDOW
        
        # Step 2: FFT and Power Spectrum [cite: 37, 38]
        fft_res = np.fft.rfft(frame, n=N_FFT)
        power_spectrum = np.abs(fft_res)**2
        
        # Step 3: Apply Mel Filter Bank [cite: 40, 41]
        mel_spectrum = np.dot(MEL_FILTER_BANK, power_spectrum)
        
        # Step 4: Log Scale [cite: 42]
        safe_mel = np.maximum(mel_spectrum, 1e-10)
        log_mel = 10.0 * np.log10(safe_mel)
        
        # Step 5: DCT Matrix Multiplication 
        mfcc_buf[frame_idx] = np.dot(DCT_MATRIX, log_mel)
        
    return mfcc_buf

def predict(audio_data):
    # Get Manual MFCCs
    mfcc = compute_manual_mfcc(audio_data)
    print(f"Mean MFCC: {np.mean(mfcc):.2f}, Std Dev: {np.std(mfcc):.2f}")
    mfcc = (mfcc + 11.5) * (110.0 / 65.0) - 30.0

    # Quantization logic for INT8 [cite: 66, 68]
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    mfcc_quantized = (mfcc / input_scale) + input_zero_point
    mfcc_quantized = np.clip(mfcc_quantized, -128, 127)
    mfcc_quantized = mfcc_quantized[np.newaxis, ..., np.newaxis].astype(np.int8)

    # Run Inference [cite: 70]
    interpreter.set_tensor(input_details['index'], mfcc_quantized)
    interpreter.invoke()

    # De-quantize Output [cite: 73, 74]
    output_data = interpreter.get_tensor(output_details['index'])
Since you are moving to a more complex architecture and using a Sigmoid output on the Raspberry Pi, you need to update your run_new.py to handle the single-output logic and take advantage of the Pi's resources.

Here is the updated inference script. I have adjusted the prediction logic, added the Sigmoid thresholding, and included the Batch Normalization compatibility.

Updated run_new.py
Python
import numpy as np
import sounddevice as sd
import time
from audio_weights import MEL_FILTER_BANK, DCT_MATRIX

# --- Configuration ---
MODEL_PATH = "model_pi.tflite" # Update to your new complex model
SAMPLE_RATE = 16000
WINDOW_SIZE = 32000  
STEP_SIZE = 8000     
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
N_FRAMES = 124 # Matches the dimension fix we discussed
DEVICE_ID = 2  # Hardcoded to 'hw:2,0' for your Razer Seiren Mini
CONFIDENCE_THRESHOLD = 0.85 # Sigmoid probability threshold

# Pre-compute Hann Window
HANN_WINDOW = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / (N_FFT - 1)))

try:
    import tflite_runtime.interpreter as tflite
    tflite_interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    tflite_interpreter = tf.lite.Interpreter

# 1. Initialize TFLite
interpreter = tflite_interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

audio_buffer = np.zeros(WINDOW_SIZE, dtype='float32')

def compute_manual_mfcc(audio):
    """Replicates the Pi-specific preprocessing"""
    # 1. DC Offset Removal
    audio = audio - np.mean(audio)
    
    # 2. Peak Normalization (To match your new training data)
    peak = np.max(np.abs(audio))
    if peak > 0.001:
        audio = audio / peak
    
    mfcc_buf = np.zeros((N_FRAMES, N_MFCC))
    
    for frame_idx in range(N_FRAMES):
        start = frame_idx * HOP_LENGTH
        end = start + N_FFT
        if end > len(audio): break
            
        frame = audio[start : end] * HANN_WINDOW
        
        # FFT Magnitude (Using Magnitude instead of Power for stability)
        fft_res = np.abs(np.fft.rfft(frame, n=N_FFT))
        
        # Apply Weights
        mel_spectrum = np.dot(MEL_FILTER_BANK, fft_res)
        log_mel = np.log(mel_spectrum + 1e-9) # Using natural log for standard scaling
        
        mfcc_buf[frame_idx] = np.dot(DCT_MATRIX, log_mel)
        
    return mfcc_buf

def predict(audio_data):
    mfcc = compute_manual_mfcc(audio_data)

    # Quantization logic
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    mfcc_quantized = (mfcc / input_scale) + input_zero_point
    mfcc_quantized = np.clip(mfcc_quantized, -128, 127)
    mfcc_quantized = mfcc_quantized[np.newaxis, ..., np.newaxis].astype(np.int8)

    # Run Inference
    interpreter.set_tensor(input_details['index'], mfcc_quantized)
    interpreter.invoke()

    # --- SIGMOID LOGIC ---
    output_data = interpreter.get_tensor(output_details['index'])
    # De-quantize the single probability value
    prob = (output_data[0][0].astype(np.float32) - output_zero_point) * output_scale
    
    return prob

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status: print(status)
    
    # Roll and update buffer
    audio_buffer = np.roll(audio_buffer, -len(indata))
    audio_buffer[-len(indata):] = indata[:, 0]

    # Quick volume check to avoid unnecessary CPU usage
    if np.sqrt(np.mean(audio_buffer**2)) > 0.01:
        score = predict(audio_buffer)
        print(f"Score: {score}")
        if score > CONFIDENCE_THRESHOLD:
            print(f">>> KEYWORD DETECTED: Hey Home ({score*100:.1f}%)")

# --- Scanning & Listening ---
print("Scanning Devices...")
print(sd.query_devices())

with sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE_ID, channels=1, 
                    callback=audio_callback, blocksize=STEP_SIZE):
    print(f"--- RPi Manual Listener Active ---")
    try:
        while True: sd.sleep(1000)
    except KeyboardInterrupt: print("\nStopped.")