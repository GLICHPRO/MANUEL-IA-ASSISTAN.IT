#!/usr/bin/env python3
"""Test WebSocket GIDEON"""

import asyncio
import json
import os

import pytest
import websockets


pytestmark = pytest.mark.asyncio

WS_URL = os.getenv("GIDEON_WS_URL", "ws://127.0.0.1:8002/ws")

async def test_gideon():
    uri = WS_URL
    
    try:
        async with websockets.connect(uri, ping_timeout=None) as websocket:
            print('✅ Connected to GIDEON!')
            print()
            
            message = {
                'type': 'text_message',
                'payload': {
                    'text': 'Spiegami in breve la differenza tra machine learning e deep learning'
                }
            }
            
            print('📤 Q:', message['payload']['text'])
            print()
            print('⏳ Waiting for response...')
            
            await websocket.send(json.dumps(message))
            
            response = await asyncio.wait_for(websocket.recv(), timeout=120)
            result = json.loads(response)
            
            print()
            print('📥 Response type:', result.get('type'))
            print()
            
            if result.get('type') == 'message_result':
                payload = result.get('payload', {})
                print('✅ SUCCESS!')
                print()
                print('=== RISPOSTA GIDEON ===')
                print(payload.get('text', payload.get('response', 'No text')))
                print()
                print('Source:', payload.get('data', {}).get('source', 'unknown'))
                print('Confidence:', payload.get('confidence', 'N/A'))
            else:
                print('Full response:')
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
    except asyncio.TimeoutError:
        print('❌ TIMEOUT: No response after 120s')
    except Exception as e:
        print(f'❌ Error: {type(e).__name__}: {e}')

if __name__ == '__main__':
    asyncio.run(test_gideon())
