import asyncio
import websockets
import numpy as np
import whisper
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PORT = 8000
WHISPER_MODEL = "medium"

print(f"Loading Whisper '{WHISPER_MODEL}'...")
model = whisper.load_model(WHISPER_MODEL)
print(f"✅ Whisper ready. Listening on port {PORT}...")

DEVICE_ROUTES = {
    "AC": ["AC"],
    "LIGHTS": ["LIGHTS"],
    "FAN": ["FAN"]
}

transcription_queue = asyncio.Queue()
active_clients = {}

class IntentCNN(nn.Module):
    def __init__(self, vocab_size, num_actions, num_devices, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, 3)
        self.conv2 = nn.Conv1d(128, 128, 5)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)

        self.action_head = nn.Linear(32, num_actions)   
        self.device_head = nn.Linear(32, num_devices) 

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.action_head(x), self.device_head(x)


# load tokenizer + encoders
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    action_encoder, device_encoder = pickle.load(f)

MAX_LEN = 15
MAX_WORDS = 5000

model_nlp = IntentCNN(MAX_WORDS,
                  len(action_encoder.classes_),
                  len(device_encoder.classes_)).to(DEVICE)

model_nlp.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model_nlp.eval()
print("✅ NLP model ready")

async def transcription_worker():
    """Single worker — one Whisper instance, processes jobs sequentially."""
    while True:
        audio_float32, device_id = await transcription_queue.get()
        print(f"🎙️  Transcribing job from {device_id} ({transcription_queue.qsize()} remaining in queue)...")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_float32, language="en")
        )
        transcript = result["text"].strip()
        print(f"[{device_id}] 📝 {transcript}")

        action, device_name, a_conf, d_conf = predict_intent(transcript)
        print(f"[{device_id}] 🤖 {action}_{device_name} ({a_conf:.2f}, {d_conf:.2f})")

        # Send the result back to the specific device
        response = json.dumps({
            "command": "nlp_result",
            "text": transcript,
            "action": action,
            "device": device_name,
            "action_confidence": a_conf,
            "device_confidence": d_conf
        })

        targets = DEVICE_ROUTES.get(device_name, [])

        for target in targets:
            if target in active_clients:
                try:
                    await active_clients[target].send(response)
                    print(f"[{device_id}] 📤 Sent to {target}")
                except Exception as e:
                    print(f"[{device_id}] ⚠️ Failed sending to {target}: {e}")
                    
        if device_id in active_clients:
            await active_clients[device_id].send(response)
            print(f"[{device_id}] 📤 Sent feedback to self")

        transcription_queue.task_done()

def predict_intent(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    x = torch.tensor(padded, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        action_logits, device_logits = model_nlp(x)

        action_probs = torch.softmax(action_logits, dim=1)[0].cpu().numpy()
        device_probs = torch.softmax(device_logits, dim=1)[0].cpu().numpy()

    action_idx = action_probs.argmax()
    device_idx = device_probs.argmax()

    action = action_encoder.inverse_transform([action_idx])[0]
    device = device_encoder.inverse_transform([device_idx])[0]

    return action, device, float(action_probs[action_idx]), float(device_probs[device_idx])


async def audio_handler(websocket):
    client_ip = websocket.remote_address
    
    # 1. Extract device ID correctly
    device_id = websocket.request.path.strip("/") 
    if not device_id:
        device_id = "unknown_device"

    try:
        # 2. Save the connection using the device_id as the key
        active_clients[device_id] = websocket
        print(f"🔗 Connected: {device_id} ({client_ip[0]})")
        
        async for message in websocket:
            if isinstance(message, bytes):
                print(f"[{device_id}] 📦 {len(message)} bytes — queued...")
                
                # 3. Add .copy() to make the array writable and clear the PyTorch warning
                audio_float32 = np.frombuffer(message, dtype=np.float32).copy()
                
                # 4. Pass the exact same device_id into the queue
                await transcription_queue.put((audio_float32, device_id))
                
            elif isinstance(message, str):
                print(f"[{device_id}] 💬 Text received: {message}")

    except websockets.exceptions.ConnectionClosed:
        print(f"❌ Disconnected: {device_id}")
    except Exception as e:
        print(f"⚠️ [{device_id}] Error: {e}")
    finally:
        # Clean up the dictionary when they drop
        if device_id in active_clients:
            del active_clients[device_id]


async def main():
    asyncio.create_task(transcription_worker())
    async with websockets.serve(audio_handler, "0.0.0.0", PORT):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping server...")