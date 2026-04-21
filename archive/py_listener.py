import numpy as np
import sounddevice as sd
import asyncio
import websockets
import threading
from audio_weights import MEL_FILTER_BANK, DCT_MATRIX

# --- Configuration ---
MODEL_PATH = "model.tflite"
SAMPLE_RATE = 16000
WINDOW_SIZE = 32000
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
WS_URL = f"ws://{PC_IP}:{PORT}"

DEVICE_NAME = "rpi_01"  # change per device

STREAM_DURATION = 4
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

# Initialize TFLite
interpreter = tflite_interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

audio_buffer = np.zeros(WINDOW_SIZE, dtype='float32')

# --- State flags ---
is_streaming = False
stream_buffer = []
stream_samples_collected = 0

# --- Persistent WebSocket state ---
ws_connection = None        # the live websocket object
ws_loop = None              # the event loop running in the WS thread
ws_ready = threading.Event()  # signals when connection + ID handshake is done


async def ws_manager():
    """Runs in a dedicated thread. Maintains a persistent WebSocket connection."""
    global ws_connection

    while True:
        try:
            print(f"Connecting to {WS_URL} as '{DEVICE_NAME}'...")
            async with websockets.connect(WS_URL) as ws:
                # Send device ID immediately on connect
                await ws.send(DEVICE_NAME)
                ws_connection = ws
                ws_ready.set()
                print(f"Connected and identified as '{DEVICE_NAME}'.")

                # Keep connection alive until it drops
                await ws.wait_closed()

        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            ws_connection = None
            ws_ready.clear()
            print("Reconnecting in 2s...")
            await asyncio.sleep(2)


def start_ws_thread():
    """Start the WebSocket manager in a background thread with its own event loop."""
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(ws_manager())


def send_audio(audio_samples: np.ndarray):
    """Send audio over the persistent connection. Called from a threading.Thread."""
    global is_streaming, stream_buffer, stream_samples_collected

    if ws_connection is None:
        print("No WebSocket connection — dropping audio.")
    else:
        audio_bytes = audio_samples.astype(np.float32).tobytes()
        future = asyncio.run_coroutine_threadsafe(
            ws_connection.send(audio_bytes),
            ws_loop
        )
        try:
            future.result(timeout=5)
            print(f"Audio sent ({len(audio_bytes)} bytes).")
        except Exception as e:
            print(f"Send error: {e}")

    stream_buffer = []
    stream_samples_collected = 0
    is_streaming = False
    print("--- Inference resumed ---")


def compute_manual_mfcc(audio):
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
            full_audio = np.concatenate(stream_buffer)
            print(f"Captured {len(full_audio)} samples. Sending to PC...")
            t = threading.Thread(target=send_audio, args=(full_audio,), daemon=True)
            t.start()
        return

    # ── INFERENCE PHASE ──────────────────────────────────────────────────────
    audio_buffer = np.roll(audio_buffer, -len(chunk))
    audio_buffer[-len(chunk):] = chunk

    if np.sqrt(np.mean(audio_buffer ** 2)) > 0.01:
        score = predict(audio_buffer)
        print(f"Predicted Score: {score:.4f}")

        if score > CONFIDENCE_THRESHOLD:
            print(f">>> KEYWORD DETECTED: Hey Home ({score * 100:.1f}%) — pausing inference, streaming next {STREAM_DURATION}s")
            audio_buffer[:] = 0
            is_streaming = True
            stream_buffer = []
            stream_samples_collected = 0


# --- Start WebSocket thread ---
print("Scanning Devices...")
print(sd.query_devices())

ws_thread = threading.Thread(target=start_ws_thread, daemon=True)
ws_thread.start()

print("Waiting for WebSocket connection...")
ws_ready.wait()  # Block until connected and ID sent

with sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE_ID, channels=1,
                    callback=audio_callback, blocksize=STEP_SIZE):
    print(f"--- RPi Manual Listener Active ---")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")