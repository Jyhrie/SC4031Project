import numpy as np
import sounddevice as sd
import threading

from config import *
from model import predict
from network import send_audio

from ui import ui

def create_audio_callback(state):

    def callback(indata, frames, time, status):
        chunk = indata[:, 0].copy()
        
        # streaming mode
        if state.streaming:
            remaining = STREAM_SAMPLES - state.stream_count
            take = min(len(chunk), remaining)

            state.stream_buf.append(chunk[:take])
            state.stream_count += take

            if state.stream_count >= STREAM_SAMPLES:
                audio = np.concatenate(state.stream_buf)

                state.streaming = False
                state.stream_buf = []
                state.stream_count = 0

                ui.set_processing()
                threading.Thread(
                    target=send_audio,
                    args=(state, audio),
                    daemon=True
                ).start()

            return

        # inference buffer
        state.audio_buffer = np.roll(state.audio_buffer, -len(chunk))
        state.audio_buffer[-len(chunk):] = chunk

        if np.sqrt(np.mean(state.audio_buffer ** 2)) > 0.01:
            score = predict(state.audio_buffer)

            print("score:", score)

            if score > CONFIDENCE_THRESHOLD:
                ui.set_listening()
                print("KEYWORD DETECTED")

                state.audio_buffer[:] = 0
                state.streaming = True
                state.stream_buf = []
                state.stream_count = 0

    return callback