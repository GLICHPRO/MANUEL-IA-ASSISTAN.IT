"""
🤖 GIDEON UNIFIED - Sistema AI Unificato

G.I.D.E.O.N. = Generative Intelligence for Dynamic Executive Operations Network

Un unico assistente che integra:
- Cognitive Engine (analisi, predizioni, simulazioni)
- Executive Engine (decisioni, esecuzione, automazione)
- Voice & Vision (TTS, riconoscimento, analisi immagini)
- Memory & Learning (contesto, apprendimento continuo)

Architettura:
┌─────────────────────────────────────────────────────────────┐
│                      GIDEON UNIFIED                          │
│              (Un Solo Assistente Intelligente)               │
├─────────────────────────────────────────────────────────────┤
│  🧠 COGNITIVE      │  ⚡ EXECUTIVE     │  🎤 INTERFACE       │
│  - Analyzer        │  - Intent Parser  │  - Voice (TTS)      │
│  - Predictor       │  - Decision Maker │  - Vision AI        │
│  - Simulator       │  - Action Runner  │  - Chat             │
│  - Ranker          │  - Security       │  - Commands         │
└─────────────────────────────────────────────────────────────┘
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from loguru import logger
import time


class GideonMode(Enum):
    """Modalità operative di GIDEON"""
    PASSIVE = "passive"      # Solo analisi e suggerimenti
    COPILOT = "copilot"      # Suggerisce e chiede conferma
    PILOT = "pilot"          # Esecuzione autonoma
    EXECUTIVE = "executive"  # Orchestrazione completa


class GideonLevel(Enum):
    """Livelli di risposta"""
    NORMAL = "normal"        # Amichevole, breve
    ADVANCED = "advanced"    # Tecnico, dettagliato
    EXPERT = "expert"        # Massima profondità


class GideonUnified:
    """
    🤖 GIDEON - Sistema AI Unificato
    
    Un unico punto di accesso a tutte le funzionalità AI:
    - process() - Pipeline principale (input → analisi → decisione → risposta)
    - quick() - Risposta rapida per comandi semplici
    - analyze() - Analisi approfondita
    - execute() - Esecuzione azione
    - vision() - Analisi immagini
    - speak() - Text-to-Speech
    """
    
    def __init__(self):
        # Stato
        self.mode = GideonMode.COPILOT
        self.level = GideonLevel.NORMAL
        self.is_initialized = False
        
        # Componenti (lazy loading)
        self._assistant = None      # Brain principale
        self._gideon_core = None    # Modulo cognitivo
        self._jarvis_core = None    # Modulo esecutivo
        self._orchestrator = None   # Pipeline coordinator
        self._mode_manager = None   # Gestione modalità
        self._automation = None     # Smart actions
        self._tts_service = None    # Text-to-Speech
        
        # Cache e performance
        self._response_cache = {}
        self._cache_ttl = 60  # secondi
        
        # Personalità
        self.personality = {
            "name": "GIDEON",
            "full_name": "Generative Intelligence for Dynamic Executive Operations Network",
            "voice": "it-IT-GiuseppeNeural",
            "style": "friendly_professional",
            "emoji_enabled": True
        }
        
        # Statistiche
        self.stats = {
            "requests_processed": 0,
            "actions_executed": 0,
            "errors": 0,
            "started_at": None
        }
    
    async def initialize(self):
        """Inizializza tutti i componenti"""
        logger.info("🤖 Inizializzazione GIDEON Unified...")
        
        try:
            # Import componenti
            from brain.assistant import GideonAssistant
            from gideon import GideonCore
            from jarvis import JarvisSupervisor
            from automation import AutomationLayer
            from core.mode_manager import ModeManager
            from core.orchestrator import Orchestrator
            
            # Inizializza assistant (brain principale)
            self._assistant = GideonAssistant()
            await self._assistant.initialize()
            
            # Mode manager
            self._mode_manager = ModeManager()
            
            # Gideon Core (cognitivo)
            self._gideon_core = GideonCore()
            
            # Automation layer
            self._automation = AutomationLayer()
            
            # Jarvis (esecutivo) - collegato a Gideon e Automation
            self._jarvis_core = JarvisSupervisor(
                gideon_core=self._gideon_core,
                automation_layer=self._automation,
                mode_manager=self._mode_manager
            )
            
            # Orchestrator - coordina tutto
            self._orchestrator = Orchestrator(
                gideon_core=self._gideon_core,
                jarvis_core=self._jarvis_core,
                automation_layer=self._automation,
                mode_manager=self._mode_manager
            )
            
            self.is_initialized = True
            self.stats["started_at"] = datetime.now()
            
            logger.info("✅ GIDEON Unified inizializzato")
            logger.info(f"   Mode: {self.mode.value}")
            logger.info(f"   Level: {self.level.value}")
            
        except Exception as e:
            logger.error(f"❌ Errore inizializzazione GIDEON: {e}")
            raise
    
    async def process(
        self,
        text: str,
        context: Dict = None,
        mode: str = None,
        include_voice: bool = False,
        response_mode: str = "fast"
    ) -> Dict[str, Any]:
        """
        🎯 Pipeline principale GIDEON
        
        Processa qualsiasi richiesta: domande, comandi, analisi, azioni.
        
        Args:
            text: Input dell'utente
            context: Contesto aggiuntivo
            mode: Override modalità (passive/copilot/pilot)
            include_voice: Genera anche audio TTS
            response_mode: eco/fast/deep
            
        Returns:
            Risposta completa con testo, azioni, audio opzionale
        """
        start_time = time.time()
        self.stats["requests_processed"] += 1
        
        # Aggiorna modalità se specificata
        if mode:
            self.set_mode(mode)
        
        # Token limits per response mode
        token_config = {
            "eco": {"max_tokens": 150, "temperature": 0.3},
            "fast": {"max_tokens": 300, "temperature": 0.5},
            "deep": {"max_tokens": 800, "temperature": 0.7}
        }.get(response_mode, {"max_tokens": 300, "temperature": 0.5})
        
        try:
            # Usa l'assistant per processare
            if self._assistant and self._assistant.is_ready():
                result = await self._assistant.process_command(
                    text=text,
                    mode="text",
                    context=context,
                    max_tokens=token_config["max_tokens"],
                    temperature=token_config["temperature"]
                )
                
                response_text = result.get("text", result.get("response", ""))
                confidence = result.get("confidence", 0.9)
                
            else:
                # Fallback locale
                response_text = self._local_response(text)
                confidence = 0.7
                result = {"text": response_text}
            
            # Genera audio se richiesto
            audio_base64 = None
            if include_voice:
                audio_base64 = await self._generate_voice(response_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "response": response_text,
                "confidence": confidence,
                "mode": self.mode.value,
                "level": self.level.value,
                "processing_time_ms": round(processing_time, 2),
                "audio_base64": audio_base64,
                "actions": result.get("data", {}).get("actions", []),
                "source": result.get("data", {}).get("source", "gideon")
            }
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"GIDEON process error: {e}")
            return {
                "success": False,
                "response": f"⚠️ Errore nell'elaborazione: {str(e)}",
                "confidence": 0.0,
                "mode": self.mode.value,
                "error": str(e)
            }
    
    async def quick(self, text: str) -> Dict[str, Any]:
        """
        ⚡ Risposta rapida per comandi semplici
        
        Usa pattern matching locale per risposte immediate.
        """
        text_lower = text.lower()
        
        # Time
        if any(w in text_lower for w in ["ora", "che ore", "orario"]):
            now = datetime.now()
            return {
                "success": True,
                "response": f"🕐 Sono le {now.hour}:{now.minute:02d}",
                "intent": "time"
            }
        
        # Date
        if any(w in text_lower for w in ["data", "oggi", "giorno"]):
            now = datetime.now()
            giorni = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
            mesi = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
                   "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"]
            return {
                "success": True,
                "response": f"📅 {giorni[now.weekday()]} {now.day} {mesi[now.month-1]} {now.year}",
                "intent": "date"
            }
        
        # Greeting
        if any(w in text_lower for w in ["ciao", "salve", "buongiorno", "buonasera"]):
            hour = datetime.now().hour
            if hour < 12:
                greeting = "Buongiorno"
            elif hour < 18:
                greeting = "Buon pomeriggio"
            else:
                greeting = "Buonasera"
            return {
                "success": True,
                "response": f"👋 {greeting}! Sono GIDEON, come posso aiutarti?",
                "intent": "greeting"
            }
        
        # Help
        if any(w in text_lower for w in ["aiuto", "help", "cosa puoi"]):
            return {
                "success": True,
                "response": """🤖 Sono **GIDEON** - Il tuo assistente AI. Posso:

• 💬 **Conversare** - Rispondere a domande
• 🔍 **Analizzare** - Immagini, dati, situazioni
• ⚙️ **Eseguire** - Azioni e automazioni
• 📊 **Ragionare** - Valutare scenari complessi
• 🎤 **Parlare** - Con voce naturale

Prova a chiedermi qualcosa!""",
                "intent": "help"
            }
        
        # Non è un quick command - usa pipeline completa
        return await self.process(text)
    
    async def analyze(self, text: str, context: Dict = None) -> Dict[str, Any]:
        """
        🔍 Analisi approfondita
        
        Usa il modulo cognitivo per analisi dettagliate.
        """
        if self._gideon_core:
            try:
                analysis = await self._gideon_core.analyze(text, context)
                return {
                    "success": True,
                    "analysis": analysis,
                    "source": "gideon_cognitive"
                }
            except Exception as e:
                logger.error(f"Analyze error: {e}")
        
        # Fallback a process
        return await self.process(text, context, response_mode="deep")
    
    async def vision(
        self,
        image_base64: str = None,
        image_path: str = None,
        question: str = "Analizza questa immagine in dettaglio"
    ) -> Dict[str, Any]:
        """
        👁️ Analisi immagini con AI Vision
        
        Args:
            image_base64: Immagine in base64
            image_path: Path al file immagine
            question: Domanda sull'immagine
        """
        try:
            from automation.smart_actions import smart_actions
            
            if image_path:
                result = await smart_actions.analyze_image(image_path, question)
            elif image_base64:
                # Salva temporaneamente
                import tempfile
                import base64
                from pathlib import Path
                
                temp_path = Path(tempfile.gettempdir()) / f"gideon_vision_{int(time.time())}.jpg"
                
                # Rimuovi header se presente
                if "," in image_base64:
                    image_base64 = image_base64.split(",")[1]
                
                temp_path.write_bytes(base64.b64decode(image_base64))
                result = await smart_actions.analyze_image(str(temp_path), question)
            else:
                return {"success": False, "error": "Fornire image_base64 o image_path"}
            
            return result
            
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return {"success": False, "error": str(e)}
    
    async def speak(self, text: str) -> Optional[str]:
        """
        🔊 Genera audio TTS
        
        Returns:
            Audio in base64
        """
        return await self._generate_voice(text)
    
    async def execute(self, action: str, params: Dict = None) -> Dict[str, Any]:
        """
        ⚡ Esegue un'azione
        
        Args:
            action: Nome dell'azione
            params: Parametri
        """
        self.stats["actions_executed"] += 1
        
        try:
            if self._automation:
                result = await self._automation.execute(action, params or {})
                return result
            
            return {
                "success": False,
                "error": "Automation layer non disponibile"
            }
            
        except Exception as e:
            logger.error(f"Execute error: {e}")
            return {"success": False, "error": str(e)}
    
    # ========== CONFIGURATION ==========
    
    def set_mode(self, mode: str):
        """Imposta modalità operativa"""
        try:
            self.mode = GideonMode(mode.lower())
            if self._mode_manager:
                self._mode_manager.set_mode(mode)
            logger.info(f"🎛️ GIDEON mode: {self.mode.value}")
        except ValueError:
            logger.warning(f"Modalità non valida: {mode}")
    
    def set_level(self, level: str):
        """Imposta livello risposta"""
        try:
            self.level = GideonLevel(level.lower())
            logger.info(f"📊 GIDEON level: {self.level.value}")
        except ValueError:
            logger.warning(f"Livello non valido: {level}")
    
    def get_status(self) -> Dict[str, Any]:
        """Ritorna stato corrente"""
        return {
            "initialized": self.is_initialized,
            "mode": self.mode.value,
            "level": self.level.value,
            "stats": {
                **self.stats,
                "started_at": self.stats["started_at"].isoformat() if self.stats["started_at"] else None
            },
            "components": {
                "assistant": self._assistant is not None and self._assistant.is_ready(),
                "gideon_core": self._gideon_core is not None,
                "jarvis_core": self._jarvis_core is not None,
                "orchestrator": self._orchestrator is not None,
                "automation": self._automation is not None
            }
        }
    
    # ========== PRIVATE METHODS ==========
    
    def _local_response(self, text: str) -> str:
        """Risposta locale di fallback"""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["ciao", "salve"]):
            return "👋 Ciao! Sono GIDEON, come posso aiutarti?"
        
        if any(w in text_lower for w in ["grazie", "thanks"]):
            return "😊 Prego! Sono sempre qui per aiutarti."
        
        return "🤖 Ricevuto. Come posso aiutarti?"
    
    async def _generate_voice(self, text: str) -> Optional[str]:
        """Genera audio TTS"""
        try:
            from tts_service import EdgeTTSService
            
            if not hasattr(self, '_tts') or self._tts is None:
                self._tts = EdgeTTSService()
            
            audio_bytes = await self._tts.synthesize(text)
            if audio_bytes:
                import base64
                return base64.b64encode(audio_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
        
        return None
    
    async def shutdown(self):
        """Cleanup"""
        logger.info("🛑 Shutdown GIDEON Unified...")
        
        if self._assistant:
            await self._assistant.shutdown()
        
        self.is_initialized = False


# Singleton instance
_gideon_instance: Optional[GideonUnified] = None


def get_gideon() -> GideonUnified:
    """Ottieni istanza GIDEON (singleton)"""
    global _gideon_instance
    if _gideon_instance is None:
        _gideon_instance = GideonUnified()
    return _gideon_instance


async def init_gideon() -> GideonUnified:
    """Inizializza e ritorna GIDEON"""
    gideon = get_gideon()
    if not gideon.is_initialized:
        await gideon.initialize()
    return gideon
