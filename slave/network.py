import asyncio
import websockets
import json
import threading
from state import State
from config import WS_URL

state = None


async def ws_loop():
    global state
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                state.ws = ws
                print("Connected to server")

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        print(data)
                        if data.get("command") == "transcription_result":
                            print("[PC]:", data["text"])
                    except:
                        pass

        except Exception as e:
            print("WS error:", e)
            await asyncio.sleep(2)


def start_ws(state_obj):
    global state
    state = state_obj

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    state.loop = loop

    loop.run_until_complete(ws_loop())


def send_audio(state, audio):
    if not state.ws:
        return

    data = audio.astype("float32").tobytes()

    future = asyncio.run_coroutine_threadsafe(
        state.ws.send(data),
        state.loop
    )

    future.result()