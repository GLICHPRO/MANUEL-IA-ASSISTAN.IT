"""
GIDEON TTS Service - WebSocket Server per Text-to-Speech

Riceve messaggi dal backend GIDEON e li riproduce vocalmente
usando Microsoft Edge Neural TTS (gratuito)
"""

import asyncio
import json
import websockets
import subprocess
import tempfile
import os
from pathlib import Path
from voice.gideon_tts import GideonTTS

# Configurazione
CONFIG = {
    "backend_url": "ws://127.0.0.1:8001/ws",
    "auto_speak": True,
    "speak_welcome": True,
    "reconnect_interval": 5,
}

# TTS instance
tts = GideonTTS(voice="diego", pitch="-10Hz")


def play_audio(file_path: str):
    """Riproduce file audio su Windows usando Windows Media Player"""
    try:
        # Usa PowerShell con Windows Media Player (supporta MP3)
        ps_script = f'''
Add-Type -AssemblyName PresentationCore
$mediaPlayer = New-Object System.Windows.Media.MediaPlayer
$mediaPlayer.Open([System.Uri]::new("{file_path}"))
$mediaPlayer.Play()
Start-Sleep -Milliseconds 500
while ($mediaPlayer.HasAudio -and $mediaPlayer.Position -lt $mediaPlayer.NaturalDuration.TimeSpan) {{
    Start-Sleep -Milliseconds 100
}}
$mediaPlayer.Close()
'''
        subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            check=True,
            timeout=60
        )
    except subprocess.TimeoutExpired:
        print("⚠️ Timeout riproduzione audio")
    except Exception as e:
        print(f"❌ Errore riproduzione: {e}")
        # Fallback: prova con start (apre app predefinita)
        try:
            subprocess.run(["cmd", "/c", "start", "/wait", "", file_path], 
                          capture_output=True, timeout=30)
        except:
            pass


async def speak(text: str):
    """Sintetizza e riproduce testo"""
    print(f'\n🎤 GIDEON dice: "{text[:60]}{"..." if len(text) > 60 else ""}"')
    
    try:
        # Genera audio
        audio_path = await tts.synthesize(text)
        
        # Riproduci (in thread separato per non bloccare)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, play_audio, audio_path)
        
    except Exception as e:
        print(f"❌ Errore TTS: {e}")


async def handle_message(message: dict):
    """Gestisce messaggi dal backend"""
    msg_type = message.get("type", "")
    print(f"📨 Ricevuto: {msg_type}")
    
    if msg_type in ["message_result", "response"]:
        payload = message.get("payload", {})
        text = (payload.get("text") or 
                payload.get("response") or 
                message.get("text") or 
                message.get("response"))
        
        if text and CONFIG["auto_speak"]:
            await speak(text)
            
    elif msg_type == "status":
        print(f"📊 Status: {message.get('payload')}")
        
    elif msg_type == "error":
        print(f"❌ Errore: {message.get('payload')}")
        await speak("Si è verificato un errore")
        
    else:
        print(f"📦 Messaggio non gestito: {msg_type}")


async def connect():
    """Connette al backend GIDEON"""
    while True:
        try:
            print(f"\n🔌 Connessione a GIDEON Backend...")
            
            async with websockets.connect(CONFIG["backend_url"]) as ws:
                print("✅ Connesso a GIDEON Backend!")
                
                if CONFIG["speak_welcome"]:
                    await speak("Connessione stabilita. Gideon TTS attivo.")
                
                # Ascolta messaggi
                async for message in ws:
                    try:
                        data = json.loads(message)
                        await handle_message(data)
                    except json.JSONDecodeError as e:
                        print(f"❌ Errore parsing: {e}")
                        
        except websockets.exceptions.ConnectionClosed:
            print("🔌 Disconnesso da GIDEON")
        except Exception as e:
            print(f"❌ Errore: {e}")
        
        print(f"🔄 Riconnessione in {CONFIG['reconnect_interval']}s...")
        await asyncio.sleep(CONFIG["reconnect_interval"])


async def main():
    """Entry point"""
    print("\n" + "="*50)
    print("   GIDEON TTS - Text-to-Speech Service")
    print("   Microsoft Edge Neural Voice (Gratuito)")
    print("="*50 + "\n")
    
    # Lista voci disponibili
    GideonTTS.list_voices()
    
    print(f"\n🎯 Voce attiva: {tts.voice_name}")
    print(f"   Rate: {tts.rate}, Pitch: {tts.pitch}")
    print()
    
    # Test TTS
    print("🧪 Test voce...")
    await speak("Sistema TTS pronto.")
    
    # Connetti al backend
    await connect()


if __name__ == "__main__":
    asyncio.run(main())
