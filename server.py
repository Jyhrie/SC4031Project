import asyncio
import websockets
import numpy as np
import wave
import io
import whisper

PORT = 8000
WHISPER_MODEL = "medium"

print(f"Loading Whisper '{WHISPER_MODEL}'...")
model = whisper.load_model(WHISPER_MODEL)
print(f"✅ Whisper ready. Listening on port {PORT}...")

active_clients = {}

transcription_queue = asyncio.Queue()


async def transcription_worker():
    """Single worker — one Whisper instance, processes jobs sequentially."""
    while True:
        audio_float32, client_id = await transcription_queue.get()
        print(f"🎙️  Transcribing job from {client_id} ({transcription_queue.qsize()} remaining in queue)...")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_float32, language="en")
        )
        transcript = result["text"].strip()
        print(f"[{client_id}] 📝 {transcript}")

        transcription_queue.task_done()


async def audio_handler(websocket):
    client_id = websocket.remote_address

    device_id = websocket.request.path.strip("/")

    if not device_id:
        device_id = "unknown_device"
    try:
        async for message in websocket:
            print(f"[{client_id}] 📦 {len(message)} bytes — queued...")
            audio_float32 = np.frombuffer(message, dtype=np.float32)
            await transcription_queue.put((audio_float32, device_id))

    except websockets.exceptions.ConnectionClosed:
        print(f"❌ Disconnected: {client_id}")
    except Exception as e:
        print(f"⚠️ [{client_id}] Error: {e}")
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