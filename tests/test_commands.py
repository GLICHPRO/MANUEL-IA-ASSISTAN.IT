#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for new Gideon commands"""

import asyncio
import json
import os
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pytest
import websockets


pytestmark = pytest.mark.asyncio

WS_URL = os.getenv("GIDEON_WS_URL", "ws://127.0.0.1:8001/ws")
TEST_COMMANDS = [
    "che ore sono",
    "che giorno è oggi",
    "stato batteria",
    "ciao gideon",
    "come stai",
]


async def _send_command(command: str, timeout: int = 15) -> dict:
    """Invia un comando via WebSocket e restituisce l'esito grezzo."""
    async with websockets.connect(WS_URL) as ws:
        message = {"type": "text_message", "payload": {"text": command}}
        await ws.send(json.dumps(message))
        response = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return json.loads(response)


@pytest.mark.parametrize("command", TEST_COMMANDS)
async def test_command(command: str):
    """Verifica che ogni comando riceva una risposta valida dal WebSocket."""
    data = await _send_command(command)
    assert data.get("type") in {"message_result", "command_result"}
    payload = data.get("payload", {})
    assert payload.get("text") or payload.get("response"), "La risposta non contiene testo"


async def main():
    print("=" * 50)
    print("🧪 TEST NUOVI COMANDI GIDEON")
    print("=" * 50)

    for cmd in TEST_COMMANDS:
        print(f"\n📤 Comando: {cmd}")
        try:
            data = await _send_command(cmd)
            if data.get("type") in {"message_result", "command_result"}:
                payload = data.get("payload", {})
                print(f"✅ Risposta: {payload.get('text', payload.get('response', 'Nessuna risposta'))}")
            else:
                print(f"📨 Risposta inattesa: {data}")
        except Exception as exc:  # pragma: no cover - output diagnostico solo in esecuzione manuale
            print(f"❌ Errore durante il comando '{cmd}': {exc}")
        await asyncio.sleep(1)
    
    print("\n" + "=" * 50)
    print("✅ Test completati!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
