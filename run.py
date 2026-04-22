import numpy as np
import sounddevice as sd
from slave.audio_weights import MEL_FILTER_BANK, DCT_MATRIX

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
        print(f"Predicted Score: {score:.4f}")
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