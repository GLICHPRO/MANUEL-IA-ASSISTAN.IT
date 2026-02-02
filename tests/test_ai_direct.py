#!/usr/bin/env python3
"""Direct test of AI generation via API"""

import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def test_direct_chat():
    """Test direct chat with AI provider through chat endpoint"""
    print("=" * 60)
    print("Testing /api/chat/send endpoint")
    print("=" * 60)
    
    payload = {
        "message": "Ciao! Qual e la capitale d'Italia?",
        "user_id": "direct_test"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/chat/send",
        json=payload,
        timeout=120
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
def test_assistant_command():
    """Test via assistant command endpoint"""
    print("\n" + "=" * 60)
    print("Testing /api/command endpoint")
    print("=" * 60)
    
    payload = {
        "command": "Spiegami la differenza tra machine learning e deep learning"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/command",
        json=payload,
        timeout=120
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)[:2000]}")

def test_ai_status():
    """Check AI provider status"""
    print("\n" + "=" * 60)
    print("Testing /api/ai/status endpoint")  
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/ai/status", timeout=30)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"AI Status: {json.dumps(data, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    test_ai_status()
    test_direct_chat()
    test_assistant_command()
