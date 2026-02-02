"""
GIDEON SPEAKER - Voce Naturale Umana

Fa parlare Gideon come una VERA persona:
- Voce profonda e autorevole (Diego)
- Pause naturali e respiri
- Intonazione dinamica ed emotiva
- Variazione nel ritmo parlato
- Enfasi sulle parole chiave
"""

import asyncio
import subprocess
import re
import random
from pathlib import Path
from typing import Optional
from edge_tts import Communicate
import hashlib
import os

# Configurazione voce Gideon - OTTIMIZZATA per naturalezza
GIDEON_VOICE_CONFIG = {
    "voice": "it-IT-GiuseppeNeural",  # VOCE UFFICIALE DI GIDEON
    "rate": "+0%",                     # Velocità normale (più naturale)
    "pitch": "+0Hz",                   # Tono naturale
    "volume": "+0%",
}

# Parole su cui mettere enfasi (più espressività)
EMPHASIS_WORDS = [
    "importante", "attenzione", "perfetto", "ottimo", "eccellente",
    "problema", "errore", "successo", "fatto", "pronto", "certo",
    "sicuro", "assolutamente", "naturalmente", "ovviamente"
]

# Interiezioni naturali da aggiungere occasionalmente
NATURAL_INTERJECTIONS = [
    "Allora,", "Dunque,", "Ecco,", "Bene,", "Vediamo,"
]

class GideonSpeaker:
    """
    Sistema vocale di Gideon - parla come una persona VERA
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.voice = GIDEON_VOICE_CONFIG["voice"]
        self.rate = GIDEON_VOICE_CONFIG["rate"]
        self.pitch = GIDEON_VOICE_CONFIG["pitch"]
        self.volume = GIDEON_VOICE_CONFIG["volume"]
        
        # Cache directory
        self.cache_dir = Path(__file__).parent / "audio_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Stato
        self.is_speaking = False
        self.enabled = True  # TTS abilitato
        
        self._initialized = True
        print("🎤 GIDEON Speaker inizializzato (voce umana naturale)")
    
    def _get_cache_key(self, text: str) -> str:
        """Genera hash per cache"""
        key = f"{text}_{self.voice}_{self.rate}_{self.pitch}_v3"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _add_natural_pauses(self, text: str) -> str:
        """Aggiunge pause naturali come farebbe una persona"""
        # Pausa più lunga dopo il punto (come quando si riflette)
        text = re.sub(r'\.(\s+)([A-Z])', r'. \2', text)
        
        # Pausa media dopo virgola
        text = re.sub(r',(\s*)', r', ', text)
        
        # Pausa dopo due punti (introduce qualcosa)
        text = re.sub(r':(\s*)', r': ', text)
        
        return text
    
    def _prepare_text_for_speech(self, text: str) -> str:
        """
        Prepara il testo per sembrare più naturale e umano
        """
        # Rimuovi markdown
        text = re.sub(r'[*_`#]', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [link](url) → link
        text = re.sub(r'<[^>]*>', '', text)  # Rimuovi HTML
        
        # Rimuovi emoji
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]', '', text)
        
        # Normalizza spazi
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Aggiungi pause naturali
        text = self._add_natural_pauses(text)
        
        return text
    
    def _build_expressive_ssml(self, text: str) -> str:
        """
        Costruisce SSML AVANZATO per voce espressiva e umana:
        - Stile conversazionale
        - Pause naturali come respirazione
        - Enfasi emotiva
        - Variazione di ritmo
        """
        clean_text = self._prepare_text_for_speech(text)
        
        # Aggiungi micro-pause dopo punteggiatura (simula respirazione)
        # Punto = pausa lunga (riflessione)
        clean_text = re.sub(r'\.(\s+)', r'. <break time="500ms"/> ', clean_text)
        # Punto esclamativo = pausa media con enfasi
        clean_text = re.sub(r'\!(\s+)', r'! <break time="350ms"/> ', clean_text)
        # Punto interrogativo = pausa per risposta
        clean_text = re.sub(r'\?(\s+)', r'? <break time="400ms"/> ', clean_text)
        # Virgola = pausa breve (respiro)
        clean_text = re.sub(r',(\s+)', r', <break time="200ms"/> ', clean_text)
        # Due punti = pausa introduttiva
        clean_text = re.sub(r':(\s+)', r': <break time="300ms"/> ', clean_text)
        # Punto e virgola = pausa media
        clean_text = re.sub(r';(\s+)', r'; <break time="250ms"/> ', clean_text)
        
        # Enfasi su parole chiave (le rende più espressive)
        for word in EMPHASIS_WORDS:
            pattern = rf'\b({word})\b'
            replacement = rf'<emphasis level="moderate">\1</emphasis>'
            clean_text = re.sub(pattern, replacement, clean_text, flags=re.IGNORECASE)
        
        # SSML con stile conversazionale Microsoft (mstts:express-as)
        ssml = f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                   xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="it-IT">
            <voice name="{self.voice}">
                <mstts:express-as style="chat" styledegree="1.5">
                    <prosody rate="{self.rate}" pitch="{self.pitch}">
                        {clean_text}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>'''
        
        return ssml
    
    async def synthesize(self, text: str) -> Optional[str]:
        """Sintetizza testo in audio MP3 con voce UMANA"""
        if not text or not text.strip():
            return None
        
        clean_text = self._prepare_text_for_speech(text)
        if not clean_text:
            return None
        
        # Check cache
        cache_key = self._get_cache_key(clean_text)
        cache_path = self.cache_dir / f"{cache_key}.mp3"
        
        if cache_path.exists():
            return str(cache_path)
        
        try:
            # Metodo 1: Usa SSML espressivo (più naturale)
            ssml_text = self._build_expressive_ssml(text)
            
            # Edge-tts supporta SSML parziale, proviamo con testo processato
            processed_text = self._prepare_text_for_speech(text)
            
            # Genera audio con Edge TTS
            communicate = Communicate(
                text=processed_text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
                volume=self.volume
            )
            
            await communicate.save(str(cache_path))
            return str(cache_path)
            
        except Exception as e:
            print(f"❌ Errore sintesi: {e}")
            return None
    
    def _play_audio_windows(self, file_path: str) -> bool:
        """Riproduce audio su Windows"""
        try:
            # Usa PowerShell con MediaPlayer
            ps_script = f'''
$ErrorActionPreference = "SilentlyContinue"
Add-Type -AssemblyName PresentationCore
$player = New-Object System.Windows.Media.MediaPlayer
$player.Open([System.Uri]::new("{file_path}"))
Start-Sleep -Milliseconds 200
$player.Play()
while ($player.HasAudio -and $player.Position -lt $player.NaturalDuration.TimeSpan) {{
    Start-Sleep -Milliseconds 50
}}
Start-Sleep -Milliseconds 100
$player.Close()
'''
            subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                timeout=60
            )
            return True
            
        except Exception as e:
            print(f"❌ Errore riproduzione: {e}")
            return False
    
    async def speak(self, text: str) -> bool:
        """
        FA PARLARE GIDEON!
        
        Sintetizza e riproduce il testo come voce naturale.
        Usa cache per risposte comuni.
        """
        if not self.enabled:
            return False
        
        if self.is_speaking:
            print("⏸️ Già in parlata, skip...")
            return False
        
        self.is_speaking = True
        
        try:
            # Sintetizza
            audio_path = await self.synthesize(text)
            
            if audio_path:
                # Riproduci in thread separato per non bloccare
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._play_audio_windows, audio_path)
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Errore speak: {e}")
            return False
            
        finally:
            self.is_speaking = False
    
    def speak_sync(self, text: str) -> bool:
        """Versione sincrona di speak"""
        return asyncio.run(self.speak(text))
    
    def enable(self):
        """Abilita TTS"""
        self.enabled = True
        print("🔊 TTS abilitato")
    
    def disable(self):
        """Disabilita TTS"""
        self.enabled = False
        print("🔇 TTS disabilitato")
    
    def set_voice(self, voice_name: str):
        """Cambia voce"""
        self.voice = voice_name
        print(f"🎤 Voce cambiata: {voice_name}")
    
    def set_speed(self, percent: int):
        """Cambia velocità (-50 a +100)"""
        val = max(-50, min(100, percent))
        self.rate = f"{'+' if val >= 0 else ''}{val}%"
    
    def set_pitch(self, hz: int):
        """Cambia tono (-50 a +50)"""
        val = max(-50, min(50, hz))
        self.pitch = f"{'+' if val >= 0 else ''}{val}Hz"


# Singleton globale
gideon_speaker = GideonSpeaker()


async def gideon_speak(text: str) -> bool:
    """
    🎤 FUNZIONE PRINCIPALE - Fa parlare Gideon
    
    Uso:
        await gideon_speak("Ciao! Come posso aiutarti?")
    """
    return await gideon_speaker.speak(text)


def gideon_speak_sync(text: str) -> bool:
    """Versione sincrona"""
    return gideon_speaker.speak_sync(text)


# Test
if __name__ == "__main__":
    async def test():
        print("\n🧪 Test Gideon Speaker\n")
        
        await gideon_speak("Ciao! Sono Gideon, il tuo assistente personale.")
        await asyncio.sleep(0.5)
        await gideon_speak("Come posso aiutarti oggi?")
        
        print("\n✅ Test completato!")
    
    asyncio.run(test())
