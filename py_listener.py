import asyncio
import websockets
import numpy as np
import sounddevice as sd

# Audio Settings
SAMPLERATE = 16000
CHANNELS = 1

print(f"🔊 PC Speakers Ready. Waiting for RPi audio on port 8000...")

async def audio_handler(websocket):
    print(f"🔗 Remote device connected!")
    try:
        async for message in websocket:
            # 1. Receive the raw 16-bit PCM bytes
            # 2. Convert to float32 for sounddevice
            audio_array = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 3. Play instantly
            sd.play(audio_array, samplerate=SAMPLERATE, blocking=False)
            
    except websockets.exceptions.ConnectionClosed:
        print("❌ Device disconnected.")
    except Exception as e:
        print(f"⚠️ Error: {e}")

async def main():
    # '0.0.0.0' allows connections from your local network
    async with websockets.serve(audio_handler, "0.0.0.0", 8000):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())