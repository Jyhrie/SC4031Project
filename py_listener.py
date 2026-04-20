import asyncio
import websockets
import numpy as np
import sounddevice as sd

# --- CONFIG ---
SAMPLERATE = 16000
CHANNELS = 1
# Increase this if you hear "crackling" (latency vs stability tradeoff)
BLOCKSIZE = 1600 

print(f"🔊 PC Speakers Ready. Listening on port 8000...")

async def audio_handler(websocket):
    print(f"🔗 Remote device connected!")
    
    # Open the output stream once and keep it open for the duration of the connection
    # 'dtype=float32' matches our normalized conversion below
    with sd.OutputStream(samplerate=SAMPLERATE, 
                         channels=CHANNELS, 
                         dtype='float32',
                         blocksize=BLOCKSIZE) as stream:
        try:
            async for message in websocket:
                # 1. Convert incoming 16-bit PCM bytes to a NumPy array
                audio_int16 = np.frombuffer(message, dtype=np.int16)
                
                # 2. Normalize to float32 (-1.0 to 1.0)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                # 3. Write directly to the stream
                # This will wait (block) just long enough to keep the buffer full
                stream.write(audio_float32)
                
                # Optional: Debugging to see if data is actually flowing
                # print(f"Received {len(message)} bytes", end='\r')

        except websockets.exceptions.ConnectionClosed:
            print("\n❌ Device disconnected.")
        except Exception as e:
            print(f"\n⚠️ Error: {e}")

async def main():
    # '0.0.0.0' listens to all network interfaces on your PC
    async with websockets.serve(audio_handler, "0.0.0.0", 8000):
        await asyncio.Future()  # Keep the server running forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping server...")