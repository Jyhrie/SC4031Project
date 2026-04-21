import numpy as np
import sounddevice as sd
import asyncio
import websockets
import threading
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

PC_IP = "192.168.18.73"
PORT = 8000
STATION_ID = "device_2"
WS_URL = f"ws://{PC_IP}:{PORT}/{STATION_ID}"

STREAM_DURATION = 4       # seconds to stream after keyword
STREAM_SAMPLES = SAMPLE_RATE * STREAM_DURATION  # 64000 samples

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

# --- State flags ---
is_streaming = False          # True while capturing/sending post-keyword audio
stream_buffer = []            # Accumulates raw samples during streaming phase
stream_samples_collected = 0  # How many samples collected so far


def compute_manual_mfcc(audio):
    """Replicates computeMFCC() from Arduino"""
    audio = audio - np.mean(audio)
    mfcc_buf = np.zeros((N_FRAMES, N_MFCC))

    for frame_idx in range(N_FRAMES):
        start = frame_idx * HOP_LENGTH
        if start + N_FFT > len(audio):
            break

        frame = audio[start: start + N_FFT] * HANN_WINDOW
        fft_res = np.fft.rfft(frame, n=N_FFT)
        power_spectrum = np.abs(fft_res) ** 2
        mel_spectrum = np.dot(MEL_FILTER_BANK, power_spectrum)
        safe_mel = np.maximum(mel_spectrum, 1e-10)
        log_mel = 10.0 * np.log10(safe_mel)
        mfcc_buf[frame_idx] = np.dot(DCT_MATRIX, log_mel)

    return mfcc_buf


def predict(audio_data):
    mfcc = compute_manual_mfcc(audio_data)
    print(f"Mean MFCC: {np.mean(mfcc):.2f}, Std Dev: {np.std(mfcc):.2f}")
    mfcc = (mfcc + 11.5) * (110.0 / 65.0) - 30.0

    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    mfcc_quantized = (mfcc / input_scale) + input_zero_point
    mfcc_quantized = np.clip(mfcc_quantized, -128, 127)
    mfcc_quantized = mfcc_quantized[np.newaxis, ..., np.newaxis].astype(np.int8)

    interpreter.set_tensor(input_details['index'], mfcc_quantized)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])
    prob = (output_data[0][0].astype(np.float32) - output_zero_point) * output_scale

    return prob


async def send_audio_over_websocket(audio_samples: np.ndarray):
    """Connect to PC, send raw float32 audio bytes, then disconnect."""
    audio_bytes = audio_samples.astype(np.float32).tobytes()
    print(f"Connecting to {WS_URL} ...")
    try:
        async with websockets.connect(WS_URL) as ws:
            print(f"Connected. Sending {len(audio_samples)} samples ({len(audio_bytes)} bytes)...")
            await ws.send(audio_bytes)
            print("Audio sent. Closing connection.")
    except Exception as e:
        print(f"WebSocket error: {e}")


def stream_and_resume(audio_samples: np.ndarray):
    """Run the async WebSocket send in a dedicated thread, then re-enable inference."""
    global is_streaming, stream_buffer, stream_samples_collected

    asyncio.run(send_audio_over_websocket(audio_samples))

    # Reset streaming state and re-enable inference
    stream_buffer = []
    stream_samples_collected = 0
    is_streaming = False
    print("--- Inference resumed ---")


def audio_callback(indata, frames, time_info, status):
    global audio_buffer, is_streaming, stream_buffer, stream_samples_collected

    if status:
        print(status)

    chunk = indata[:, 0].copy()

    # ── STREAMING PHASE ──────────────────────────────────────────────────────
    if is_streaming:
        remaining = STREAM_SAMPLES - stream_samples_collected
        to_take = min(len(chunk), remaining)
        stream_buffer.append(chunk[:to_take])
        stream_samples_collected += to_take

        if stream_samples_collected >= STREAM_SAMPLES:
            # Got all 4 seconds — fire off the WebSocket send in a background thread
            full_audio = np.concatenate(stream_buffer)
            print(f"Captured {len(full_audio)} samples. Streaming to PC...")
            t = threading.Thread(target=stream_and_resume, args=(full_audio,), daemon=True)
            t.start()
        return  # Skip inference while streaming

    # ── INFERENCE PHASE ──────────────────────────────────────────────────────
    audio_buffer = np.roll(audio_buffer, -len(chunk))
    audio_buffer[-len(chunk):] = chunk

    if np.sqrt(np.mean(audio_buffer ** 2)) > 0.01:
        score = predict(audio_buffer)
        print(f"Predicted Score: {score:.4f}")

        if score > CONFIDENCE_THRESHOLD:
            print(f">>> KEYWORD DETECTED: Hey Home ({score * 100:.1f}%) — pausing inference, streaming next {STREAM_DURATION}s")
            audio_buffer[:] = 0  # <-- purge stale audio before streaming phase
            is_streaming = True
            stream_buffer = []
            stream_samples_collected = 0


# --- Scanning & Listening ---
print("Scanning Devices...")
print(sd.query_devices())

with sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE_ID, channels=1,
                    callback=audio_callback, blocksize=STEP_SIZE):
    print(f"--- RPi Manual Listener Active ---")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")