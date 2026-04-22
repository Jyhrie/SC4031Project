import sounddevice as sd
import threading
import time

from config import *
from state import State
from audio import create_audio_callback
from network import start_ws
from ui import ui

print("Starting system...")

state = State()
state.audio_buffer = __import__("numpy").zeros(WINDOW_SIZE, dtype="float32")

# start websocket thread
threading.Thread(target=start_ws, args=(state,), daemon=True).start()

time.sleep(2)

# audio stream
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    device=DEVICE_ID,
    channels=1,
    blocksize=STEP_SIZE,
    callback=create_audio_callback(state)
):
    print("Running smart home node...")

    while True:
        sd.sleep(1000)

ui.start()
