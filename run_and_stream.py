import asyncio
import queue
import time

import numpy as np
import sounddevice as sd
from audio_weights import MEL_FILTER_BANK, DCT_MATRIX
import websockets


DEVICE_ID = 2

# --- Configuration (Matches tinyml_proj.ino) ---
MODEL_PATH = "model.tflite"
SAMPLE_RATE = 16000
WINDOW_SIZE = 32000  # TARGET_SAMPLES
STEP_SIZE = 8000     
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
N_FRAMES = 124
CONFIDENCE_THRESHOLD = 0.80
SAMPLES_SINCE_LAST_DETECTION = 0
MIN_DETECTION_GAP = 8000  # 1.0 second cooldown (at 16kHz)
INITIAL_WARMUP_SAMPLES = 0
STREAMING_BLOCKS_REMAINING = 0
BLOCKS_TO_STREAM = 8

PC_IP = "192.168.18.73"
PORT = 8000
WS_URL = f"ws://{PC_IP}:{PORT}" # Match the new direct listener

# Pre-compute Hann Window once
HANN_WINDOW = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / (N_FFT - 1)))
print("Using Device ID:", DEVICE_ID)

event_queue = asyncio.Queue()
audio_queue = queue.Queue()
stream_start_time = None
f_stream_enabled = False

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
    global audio_buffer, f_stream_enabled, stream_start_time
    if status: print(status)
    if not f_stream_enabled:
        # Roll and update buffer
        audio_buffer = np.roll(audio_buffer, -len(indata))
        audio_buffer[-len(indata):] = indata[:, 0]
        

        # Quick volume check to avoid unnecessary CPU usage
        if np.sqrt(np.mean(audio_buffer**2)) > 0.01:
            score = predict(audio_buffer)
            print(f"Predicted Score: {score:.4f}")
            if score > CONFIDENCE_THRESHOLD:
                print(f">>> KEYWORD DETECTED: Hey Home ({score*100:.1f}%)")
                f_stream_enabled = True
                stream_start_time = time.time()

    if f_stream_enabled:
        audio_queue.put(indata.tobytes())
        if f_stream_enabled and (time.time() - stream_start_time) > 4:
            f_stream_enabled = False
            print(">>> Stream Ended")
            

async def websocket_sender():
    global f_stream_enabled, stream_start_time
    print(f"Connecting to server at {WS_URL}...")
    async with websockets.connect(WS_URL) as ws:
        print("✅ WebSocket Connected!")
        while True:
            try:
                # Non-blocking get from queue
                data = audio_queue.get_nowait()
                await ws.send(data)
            except queue.Empty:
                # Small sleep to yield to the event loop
                await asyncio.sleep(0.01)

# async def main():
#     global loop
#     loop = asyncio.get_running_loop()

#     # --- Scanning & Listening ---
#     print("Scanning Devices...")
#     print(sd.query_devices())

#     # Start the Microphone Stream
#     stream = sd.InputStream(
#         samplerate=SAMPLE_RATE, 
#         device=DEVICE_ID, 
#         channels=1, 
#         callback=audio_callback, 
#         blocksize=STEP_SIZE
#     )

#     with stream:
#         print("--- RPi Local Inference + WebSocket Active ---")
#         # Run the sender task
#         await websocket_sender()


with sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE_ID, channels=1, 
                    callback=audio_callback, blocksize=STEP_SIZE):
    print(f"--- RPi Manual Listener Active ---")
    try:
        asyncio.run(websocket_sender())
        while True: sd.sleep(1000)
    except KeyboardInterrupt: print("\nStopped.")
    
# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\nStopped.")

# with sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE_ID, channels=1, 
#                     callback=audio_callback, blocksize=STEP_SIZE):
#     print(f"--- RPi Manual Listener Active ---")
#     try:
#         while True: sd.sleep(1000)
#     except KeyboardInterrupt: print("\nStopped.")