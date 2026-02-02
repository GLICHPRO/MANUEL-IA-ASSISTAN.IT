"""
GIDEON TTS - Text-to-Speech con Microsoft Edge (gratuito)

Usa edge-tts per voce naturale italiana senza costi
"""

import asyncio
from edge_tts import Communicate
import hashlib
import os
from pathlib import Path
from typing import Optional

# Voci italiane disponibili
ITALIAN_VOICES = {
    # VOCE DI GIDEON (principale)
    "gideon": "it-IT-GiuseppeNeural",   # VOCE UFFICIALE DI GIDEON
    
    # Alternative maschili
    "diego": "it-IT-DiegoNeural",       # Voce profonda
    "giuseppe": "it-IT-GiuseppeNeural", # Alias di gideon
    "benigno": "it-IT-BenignoNeural",   # Voce maschile matura
    "calimero": "it-IT-CalimeroNeural", # Voce giovane
    "cataldo": "it-IT-CataldoNeural",   # Voce maschile calma
    "gianni": "it-IT-GianniNeural",     # Voce cordiale
    "lisandro": "it-IT-LisandroNeural", # Voce seria
    "rinaldo": "it-IT-RinaldoNeural",   # Voce formale
    
    # Femminili
    "elsa": "it-IT-ElsaNeural",         # Voce femminile standard
    "isabella": "it-IT-IsabellaNeural", # Voce alternativa
    "fabiola": "it-IT-FabiolaNeural",   # Voce energica
    "fiamma": "it-IT-FiammaNeural",     # Voce vivace
    "imelda": "it-IT-ImeldaNeural",     # Voce professionale
    "irma": "it-IT-IrmaNeural",         # Voce matura
    "palmira": "it-IT-PalmiraNeural",   # Voce gentile
    "pierina": "it-IT-PierinaNeural",   # Voce anziana
}

class GideonTTS:
    """Text-to-Speech per GIDEON usando Microsoft Edge Neural voices"""
    
    def __init__(
        self,
        voice: str = "gideon",  # Voce ufficiale di GIDEON
        rate: str = "+20%",     # ⚡ Velocità AUMENTATA per latenza minima
        pitch: str = "+0Hz",    # Tono naturale
        volume: str = "+0%",    # Volume
        cache_dir: Optional[str] = None
    ):
        self.voice_name = ITALIAN_VOICES.get(voice.lower(), voice)
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        
        # Directory cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent / "audio_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ GIDEON TTS inizializzato")
        print(f"   Voce: {self.voice_name}")
        print(f"   Rate: {self.rate}, Pitch: {self.pitch}")
    
    def _get_cache_key(self, text: str) -> str:
        """Genera hash per cache"""
        key = f"{text}_{self.voice_name}_{self.rate}_{self.pitch}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """Pulisce il testo per la sintesi"""
        import re
        text = re.sub(r'[*_`#]', '', text)  # Rimuovi markdown
        text = re.sub(r'<[^>]*>', '', text)  # Rimuovi HTML
        text = re.sub(r'\s+', ' ', text)     # Normalizza spazi
        return text.strip()
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Sintetizza testo in audio MP3
        
        Args:
            text: Testo da sintetizzare
            output_path: Path output (opzionale, usa cache se None)
            
        Returns:
            Path del file audio generato
        """
        clean_text = self._clean_text(text)
        if not clean_text:
            raise ValueError("Testo vuoto")
        
        # Controlla cache
        cache_key = self._get_cache_key(clean_text)
        cache_path = self.cache_dir / f"{cache_key}.mp3"
        
        if output_path is None and cache_path.exists():
            print("📦 Audio dalla cache")
            return str(cache_path)
        
        # Path di output
        out_path = output_path or str(cache_path)
        
        print(f"🎤 Sintetizzando: \"{clean_text[:50]}{'...' if len(clean_text) > 50 else ''}\"")
        
        # Crea comunicatore Edge TTS
        communicate = Communicate(
            text=clean_text,
            voice=self.voice_name,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume
        )
        
        # Salva audio
        await communicate.save(out_path)
        
        print(f"✅ Audio salvato: {out_path}")
        return out_path
    
    def speak(self, text: str) -> str:
        """Sintetizza testo (sync wrapper)"""
        return asyncio.run(self.synthesize(text))
    
    async def synthesize_to_stream(self, text: str):
        """Genera audio in streaming (per riproduzione immediata)"""
        clean_text = self._clean_text(text)
        communicate = Communicate(
            text=clean_text,
            voice=self.voice_name,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume
        )
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    
    def set_voice(self, voice: str):
        """Cambia voce"""
        self.voice_name = ITALIAN_VOICES.get(voice.lower(), voice)
        print(f"🎤 Voce cambiata: {self.voice_name}")
    
    def set_speed(self, percent: int):
        """Cambia velocità (-50 a +100)"""
        val = max(-50, min(100, percent))
        self.rate = f"{'+' if val >= 0 else ''}{val}%"
        print(f"⚡ Velocità: {self.rate}")
    
    def set_pitch(self, hz: int):
        """Cambia tono (-50 a +50 Hz)"""
        val = max(-50, min(50, hz))
        self.pitch = f"{'+' if val >= 0 else ''}{val}Hz"
        print(f"🎵 Tono: {self.pitch}")
    
    def clear_cache(self):
        """Pulisce la cache audio"""
        files = list(self.cache_dir.glob("*.mp3"))
        for f in files:
            f.unlink()
        print(f"🗑️ Cache pulita ({len(files)} file rimossi)")
    
    @staticmethod
    def list_voices():
        """Lista voci disponibili"""
        print("\n🎤 Voci italiane disponibili:\n")
        
        print("MASCHILI:")
        for key, name in ITALIAN_VOICES.items():
            if key in ["diego", "giuseppe", "benigno", "calimero", "cataldo", "gianni", "lisandro", "rinaldo"]:
                print(f"  {key:12} → {name}")
        
        print("\nFEMMINILI:")
        for key, name in ITALIAN_VOICES.items():
            if key not in ["diego", "giuseppe", "benigno", "calimero", "cataldo", "gianni", "lisandro", "rinaldo"]:
                print(f"  {key:12} → {name}")
        
        return ITALIAN_VOICES


async def test_tts():
    """Test del TTS"""
    print("\n🧪 Test GIDEON TTS (Microsoft Edge)\n")
    
    tts = GideonTTS(voice="diego", pitch="-15Hz")
    
    text = "Ciao! Sono Gideon, il tuo assistente personale. Come posso aiutarti oggi?"
    audio_path = await tts.synthesize(text, "test_output.mp3")
    
    size = os.path.getsize(audio_path) / 1024
    print(f"\n✅ Test completato!")
    print(f"   Audio: {audio_path}")
    print(f"   Dimensione: {size:.2f} KB")
    
    return True


if __name__ == "__main__":
    asyncio.run(test_tts())
