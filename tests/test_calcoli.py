import asyncio
import json
import os

import pytest
import websockets


pytestmark = pytest.mark.asyncio

WS_URL = os.getenv("GIDEON_WS_URL", "ws://127.0.0.1:8001/ws")

async def test():
    uri = WS_URL
    async with websockets.connect(uri) as ws:
        tests = [
            "quanto fa 100 diviso 4",
            "calcola 15 per 3",
            "quanto fa 50 meno 18",
            "radice di 144",
            "quanto fa 8 elevato alla 2"
        ]
        for t in tests:
            await ws.send(json.dumps({"type": "text_message", "payload": {"text": t}}))
            response = await ws.recv()
            result = json.loads(response)
            answer = result["payload"].get("text", "N/A")
            print(f"{t} -> {answer}")
        
if __name__ == "__main__":
    asyncio.run(test())
