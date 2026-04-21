import asyncio
import websockets
import sounddevice as sd
import numpy as np
import queue
import sys

# --- CONFIG ---
PC_IP = "192.168.18.73"
PORT = 8000
WS_URL = f"ws://{PC_IP}:{PORT}" # Match the new direct listener
SAMPLERATE = 16000 
CHANNELS = 1
DEVICE_ID = 2 
BLOCKSIZE = 1600 # 0.1 seconds of audio per packet

# Thread-safe queue to hold audio data
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    # Put a copy of the audio data into the queue
    audio_queue.put(indata.tobytes())

async def stream_mic():
    print(f"Connecting to PC at {WS_URL}...")
    try:
        async with websockets.connect(WS_URL) as ws:
            print("✅ Connected! Streaming audio...")

            # Start the microphone stream
            with sd.InputStream(samplerate=SAMPLERATE, 
                                device=DEVICE_ID,
                                channels=CHANNELS, 
                                dtype='int16', 
                                blocksize=BLOCKSIZE,
                                callback=callback):
                
                while True:
                    # Check if there is data in the queue
                    try:
                        # Non-blocking get from queue
                        data = audio_queue.get_nowait()
                        await ws.send(data)
                    except queue.Empty:
                        # Small sleep to yield to the event loop
                        await asyncio.sleep(0.01)
                        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(stream_mic())
    except KeyboardInterrupt:
        print("\nStopping...")