#!/usr/bin/env python3
"""Test script per verificare l'API di GIDEON"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001"

def test_chat():
    """Test del chat endpoint"""
    url = f"{BASE_URL}/api/chat/send"
    
    # Domanda complessa per testare l'AI
    payload = {
        "message": "Spiegami la differenza tra machine learning e deep learning",
        "user_id": "test_user"
    }
    
    print("🧪 Test GIDEON Chat API")
    print("=" * 50)
    print(f"📤 Messaggio: {payload['message']}")
    print("⏳ Invio richiesta...")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        print(f"\n📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Risposta ricevuta:")
            print("-" * 50)
            
            if isinstance(data, dict):
                if 'response' in data:
                    print(f"📝 Risposta: {data['response'][:500]}...")
                elif 'message' in data:
                    print(f"📝 Message: {data['message'][:500]}...")
                else:
                    print(json.dumps(data, indent=2, ensure_ascii=False)[:1000])
            else:
                print(str(data)[:1000])
        else:
            print(f"\n❌ Errore HTTP: {response.status_code}")
            print(f"Body: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Errore: Server non raggiungibile")
        print("   Assicurati che il server sia in esecuzione su http://127.0.0.1:8001")
    except requests.exceptions.Timeout:
        print("\n❌ Errore: Timeout - la richiesta ha impiegato troppo tempo")
    except Exception as e:
        print(f"\n❌ Errore: {e}")

if __name__ == "__main__":
    test_chat()
