import sounddevice as sd
import threading
import time
import numpy as np

from config import *
from state import State
from audio import create_audio_callback
from network import start_ws
from ui import ui   # <-- ENABLE THIS

print("Starting system...")

state = State()
state.audio_buffer = np.zeros(WINDOW_SIZE, dtype="float32")

# ----------------------------
# START WEBSOCKET THREAD
# ----------------------------
threading.Thread(
    target=start_ws,
    args=(state,),
    daemon=True
).start()

time.sleep(2)

# ----------------------------
# AUDIO THREAD (IMPORTANT FIX)
# ----------------------------
# def audio_loop():
#     with sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         device=DEVICE_ID,
#         channels=1,
#         blocksize=STEP_SIZE,
#         callback=create_audio_callback(state)
#     ):
#         print("Running smart home node...")

#         while True:
#             sd.sleep(1000)

# threading.Thread(target=audio_loop, daemon=True).start()
# ----------------------------
# UI MUST RUN IN MAIN THREAD
# ----------------------------
ui.start()