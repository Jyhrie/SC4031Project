import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

# Audio Settings (Must match the RPi settings)
SAMPLERATE = 16000
CHANNELS = 1

print(f"🔊 PC Speakers Ready. Waiting for RPi audio...")

@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()
    print(f"🔗 Linked to {device_id}. Playing audio...")
    
    try:
        while True:
            # 1. Receive the raw 16-bit PCM bytes
            data = await websocket.receive_bytes()
            
            # 2. Convert to float32 (the format sounddevice likes)
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 3. Play it instantly
            # 'blocking=False' is critical so it doesn't freeze the websocket
            sd.play(audio_array, samplerate=SAMPLERATE, blocking=False)

    except WebSocketDisconnect:
        print(f"❌ {device_id} disconnected.")
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)