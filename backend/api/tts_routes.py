"""
GIDEON TTS API - Endpoint per la voce di Gideon

Genera audio con Edge TTS e lo invia al frontend per la riproduzione.
La voce È Gideon - non c'è distinzione.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import hashlib
from pathlib import Path
from edge_tts import Communicate
import re
import os

router = APIRouter(prefix="/api/tts", tags=["TTS - Voce Gideon"])

# Directory cache
CACHE_DIR = Path(__file__).parent.parent / "voice" / "audio_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configurazione voce GIDEON
GIDEON_VOICE = {
    "name": "it-IT-GiuseppeNeural",  # VOCE UFFICIALE DI GIDEON
    "rate": "+0%",
    "pitch": "+0Hz",
    "volume": "+0%"
}


class TTSRequest(BaseModel):
    text: str
    

def clean_text_for_speech(text: str) -> str:
    """Pulisce il testo per la sintesi vocale"""
    # Rimuovi markdown
    text = re.sub(r'[*_`#]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Rimuovi emoji
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]', '', text)
    
    # Normalizza spazi
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_cache_path(text: str) -> Path:
    """Genera path cache per il testo"""
    clean = clean_text_for_speech(text)
    key = hashlib.md5(f"{clean}_{GIDEON_VOICE['name']}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.mp3"


@router.post("/speak")
async def speak_text(request: TTSRequest):
    """
    Genera audio dalla voce di Gideon
    
    Ritorna il file audio MP3 da riprodurre nel browser.
    """
    text = request.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Testo vuoto")
    
    clean_text = clean_text_for_speech(text)
    if not clean_text:
        raise HTTPException(status_code=400, detail="Testo non valido")
    
    # Check cache
    cache_path = get_cache_path(text)
    
    if not cache_path.exists():
        try:
            # Genera audio con Edge TTS
            communicate = Communicate(
                text=clean_text,
                voice=GIDEON_VOICE["name"],
                rate=GIDEON_VOICE["rate"],
                pitch=GIDEON_VOICE["pitch"],
                volume=GIDEON_VOICE["volume"]
            )
            await communicate.save(str(cache_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore sintesi: {str(e)}")
    
    return FileResponse(
        path=str(cache_path),
        media_type="audio/mpeg",
        filename="gideon_voice.mp3"
    )


@router.post("/speak/stream")
async def speak_text_stream(request: TTSRequest):
    """
    Genera audio in streaming (per risposte lunghe)
    """
    text = request.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Testo vuoto")
    
    clean_text = clean_text_for_speech(text)
    
    async def audio_stream():
        communicate = Communicate(
            text=clean_text,
            voice=GIDEON_VOICE["name"],
            rate=GIDEON_VOICE["rate"],
            pitch=GIDEON_VOICE["pitch"],
            volume=GIDEON_VOICE["volume"]
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    
    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg"
    )


@router.get("/voice-info")
async def get_voice_info():
    """Informazioni sulla voce di Gideon"""
    return {
        "name": "Gideon",
        "voice_engine": "Microsoft Edge Neural TTS",
        "voice_id": GIDEON_VOICE["name"],
        "language": "it-IT",
        "quality": "Neural (alta qualità)"
    }
