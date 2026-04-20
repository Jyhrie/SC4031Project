import asyncio
import websockets
import sounddevice as sd
import numpy as np

# --- CONFIG ---
PC_IP = "192.168.18.73"  # <--- REPLACE WITH YOUR PC'S IP
PORT = 8000
WS_URL = f"ws://{PC_IP}:{PORT}/ws/rpi_unit_01"

# Audio Settings (Match what Whisper expects)
SAMPLERATE = 16000 
CHANNELS = 1
DEVICE_ID = 0 # Your Razer Seiren Mini ID

async def stream_mic():
    print(f"Connecting to PC at {WS_URL}...")
    try:
        async with websockets.connect(WS_URL) as ws:
            print("✅ Connected! Streaming audio...")

            # Define the callback to send data over the socket
            def callback(indata, frames, time, status):
                if status:
                    print(status)
                # Convert to 16-bit PCM bytes and send
                asyncio.run_coroutine_threadsafe(
                    ws.send(indata.tobytes()), loop
                )

            with sd.InputStream(samplerate=SAMPLERATE, device=DEVICE_ID,
                                channels=CHANNELS, dtype='int16', 
                                callback=callback):
                while True:
                    await asyncio.sleep(1)
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(stream_mic())