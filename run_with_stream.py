import asyncio
import numpy as np
import sounddevice as sd
from audio_weights import MEL_FILTER_BANK, DCT_MATRIX
import websockets
import time

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
CONFIDENCE_THRESHOLD = 0.90

# Pre-compute Hann Window once
HANN_WINDOW = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / (N_FFT - 1)))
print("Using Device ID:", DEVICE_ID)

SERVER_URL = "ws://192.168.18.73:8000/ws/audio" 
SAMPLE_RATE = 16000
STREAM_SECONDS = 4  # How long to stream after detection
audio_queue = asyncio.Queue()

is_streaming = False
stream_timeout = 0
loop = asyncio.get_event_loop() # Initialize the loop here

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
    global is_streaming, stream_timeout
    global audio_buffer, loop
    
    if status: 
        print(f"Status: {status}")
    
    # 1. Update the rolling buffer for the TFLite model
    audio_buffer = np.roll(audio_buffer, -len(indata))
    audio_buffer[-len(indata):] = indata[:, 0]

    # 2. Logic while NOT streaming (Looking for Wake Word)
    if not is_streaming:
        # Quick volume gate to save CPU
        rms = np.sqrt(np.mean(audio_buffer**2))
        if rms > 0.01:
            score = predict(audio_buffer)
            
            # Inverse logic: low score (< 0.15) means "Hey Home" was detected
            if score < (1.0 - CONFIDENCE_THRESHOLD): 
                print(f">>> KEYWORD DETECTED! Confidence: {(1.0 - score)*100:.1f}%")
                is_streaming = True
                stream_timeout = time.time() + STREAM_SECONDS
    
    # 3. Logic while streaming (Sending data to Server)
    if is_streaming:
        # Send raw bytes to the async queue
        # indata contains the most recent chunk of audio
        try:
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy().tobytes())
        except Exception as e:
            print(f"Queue Error: {e}")

        # Check if we should stop streaming
        if time.time() > stream_timeout:
            print(">>> Streaming Window Closed.")
            is_streaming = False

async def websocket_sender():
    """The async loop that manages the connection"""
    global is_streaming
    print(f"Connecting to server at {SERVER_URL}...")
    
    while True:
        try:
            async with websockets.connect(SERVER_URL) as ws:
                print("Connected! Waiting for keyword...")
                while True:
                    # Wait for data to appear in the queue
                    audio_bytes = await audio_queue.get()
                    await ws.send(audio_bytes)
                    
        except Exception as e:
            print(f"Connection lost ({e}). Retrying in 3s...")
            await asyncio.sleep(3)

# --- Scanning & Listening ---
print("Scanning Devices...")
print(sd.query_devices())

with sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE_ID, channels=1, 
                    callback=audio_callback, blocksize=STEP_SIZE):
    print(f"--- RPi Manual Listener Active ---")
    try:
        while True: sd.sleep(1000)
    except KeyboardInterrupt: print("\nStopped.")