import asyncio
import websockets
import numpy as np
import whisper
import json

PORT = 8000
WHISPER_MODEL = "medium"

print(f"Loading Whisper '{WHISPER_MODEL}'...")
model = whisper.load_model(WHISPER_MODEL)
print(f"✅ Whisper ready. Listening on port {PORT}...")

transcription_queue = asyncio.Queue()

# Global dictionary to track live connections
active_clients = {}

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

        # Send the result back to the specific device
        if device_id in active_clients:
            ws = active_clients[device_id]
            try:
                response = json.dumps({
                    "command": "transcription_result", 
                    "text": transcript
                })
                await ws.send(response)
                print(f"[{device_id}] 📤 Sent transcript back to device.")
            except Exception as e:
                print(f"[{device_id}] ⚠️ Failed to send transcript: {e}")
        else:
            print(f"[{device_id}] ⚠️ Device disconnected before transcript could be sent.")

        transcription_queue.task_done()


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