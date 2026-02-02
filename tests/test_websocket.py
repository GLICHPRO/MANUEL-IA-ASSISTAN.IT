"""
Test WebSocket connection per Gideon 2.0
"""
import asyncio
import json
import os

import pytest
import websockets


pytestmark = pytest.mark.asyncio

WS_URL = os.getenv("GIDEON_WS_URL", "ws://127.0.0.1:8002/ws")

async def test_websocket():
    uri = WS_URL
    print(f"Testing WebSocket connection to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket CONNESSO!")
            
            # Invia un messaggio di test
            test_msg = {
                "type": "message",
                "content": "Ciao Gideon!"
            }
            await websocket.send(json.dumps(test_msg))
            print(f"📤 Messaggio inviato: {test_msg}")
            
            # Ricevi risposta
            response = await websocket.recv()
            print(f"📥 Risposta ricevuta: {response}")
            
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        print(f"   Tipo errore: {type(e).__name__}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
