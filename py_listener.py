import whisper
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

# 1. Load your trained NLP model (The "Brain")
try:
    intent_model = joblib.load("intent_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Intent Model Ready.")
except:
    print("⚠️ Warning: intent_model.pkl not found. Classification will fail.")

# 2. Load Whisper (The "Ear")
print("⏳ Loading Whisper (Base)...")
stt_model = whisper.load_model("base")
print("✅ Whisper Ready. Waiting for RPi...")

@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()
    print(f"🔗 Established link with: {device_id}")
    
    # Use a small buffer to collect chunks before transcribing
    audio_buffer = []

    try:
        while True:
            # Receive raw 16-bit PCM bytes
            data = await websocket.receive_bytes()
            audio_buffer.append(np.frombuffer(data, dtype=np.int16))

            # Transcribe every 3 seconds of audio (approx 48000 samples)
            if sum(len(x) for x in audio_buffer) > 48000:
                # Flatten buffer and normalize to float32 for Whisper
                full_audio = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
                audio_buffer = [] # Clear buffer

                # A. Speech to Text
                result = stt_model.transcribe(full_audio, fp16=False)
                text = result['text'].strip().lower()
                
                if text:
                    print(f"🗣️ Heard: {text}")
                    
                    # B. Intent Classification
                    X = vectorizer.transform([text])
                    intent = intent_model.predict(X)[0]
                    print(f"🧠 Intent: {intent}")
                    
                    # Send feedback back to the Pi
                    await websocket.send_json({"intent": intent, "text": text})

    except WebSocketDisconnect:
        print(f"❌ {device_id} disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)