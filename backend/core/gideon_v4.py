"""
🤖 GIDEON 4.0 - Ultimate Unified AI System
═══════════════════════════════════════════════════════════════════════════════

G.I.D.E.O.N. = Generative Intelligence for Dynamic Executive Operations Network

╔═══════════════════════════════════════════════════════════════════════════════╗
║                           GIDEON VERSION HISTORY                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  v1.0 - Basic Assistant       : Simple Q&A, basic commands                    ║
║  v2.0 - Brain Integration     : NLP, Memory, Reasoning, AI Providers         ║
║  v3.0 - Cognitive Module      : Predictor, Analyzer, Simulator, Meta-Cognition║
║  v4.0 - Ultimate Unified      : All versions + Executive AI + Full Autonomy  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Architecture v4.0:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GIDEON 4.0 UNIFIED                                 │
│                    (Ultimate AI System - Single Entry Point)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  🧠 BRAIN (v2.0)        │  🔮 COGNITIVE (v3.0)    │  ⚡ EXECUTIVE (Jarvis)      │
│  ├─ NLP Processor       │  ├─ Predictor           │  ├─ Intent Interpreter      │
│  ├─ Memory Manager      │  ├─ Analyzer            │  ├─ Decision Engine         │
│  ├─ Reasoning Engine    │  ├─ Simulator           │  ├─ Execution Controller    │
│  ├─ AI Providers        │  ├─ Risk Analyzer       │  ├─ Security Validator      │
│  └─ Personality         │  ├─ Meta-Cognition      │  └─ Audit Logger            │
│                         │  ├─ Self-Correction     │                             │
│                         │  ├─ Identity Core       │                             │
│                         │  └─ Goal Management     │                             │
├─────────────────────────┴─────────────────────────┴─────────────────────────────┤
│  🎤 INTERFACE           │  🔧 AUTOMATION          │  💾 MEMORY                  │
│  ├─ Voice (TTS)         │  ├─ Smart Actions       │  ├─ Episodic Memory         │
│  ├─ Vision AI           │  ├─ Dev Automations     │  ├─ Decision Memory         │
│  ├─ Chat                │  ├─ Workflow Engine     │  ├─ Context System          │
│  └─ WebSocket           │  └─ Plugin System       │  └─ Continuous Learning     │
└─────────────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.gideon_v4 import Gideon4
    
    gideon = await Gideon4.create()
    
    # Single unified method for everything
    result = await gideon.process("Analizza questa situazione e suggerisci azioni")
    
    # Or use specific capabilities
    result = await gideon.analyze(data)
    result = await gideon.predict(context)
    result = await gideon.execute(action, params)

Author: Gideon Development Team
Version: 4.0.0
Date: 2026-02-10
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum, auto
from dataclasses import dataclass, field
from loguru import logger
import time
import traceback


# ═══════════════════════════════════════════════════════════════════════════════
#                               ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class GideonMode(Enum):
    """Modalità operative di GIDEON"""
    PASSIVE = "passive"       # Solo analisi e suggerimenti
    COPILOT = "copilot"       # Suggerisce e chiede conferma
    PILOT = "pilot"           # Esecuzione autonoma
    EXECUTIVE = "executive"   # Orchestrazione completa


class GideonLevel(Enum):
    """Livelli di risposta/profondità"""
    QUICK = "quick"           # Risposte immediate
    NORMAL = "normal"         # Amichevole, breve
    ADVANCED = "advanced"     # Tecnico, dettagliato
    EXPERT = "expert"         # Massima profondità


class ResponseMode(Enum):
    """Modalità di risposta"""
    ECO = "eco"               # Minimo token, veloce
    FAST = "fast"             # Bilanciato
    DEEP = "deep"             # Analisi approfondita


class CapabilityType(Enum):
    """Tipi di capability disponibili"""
    CHAT = "chat"
    ANALYZE = "analyze"
    PREDICT = "predict"
    SIMULATE = "simulate"
    EXECUTE = "execute"
    VISION = "vision"
    VOICE = "voice"
    MEMORY = "memory"
    REASONING = "reasoning"


@dataclass
class GideonConfig:
    """Configurazione unificata Gideon 4.0"""
    mode: GideonMode = GideonMode.COPILOT
    level: GideonLevel = GideonLevel.NORMAL
    response_mode: ResponseMode = ResponseMode.FAST
    voice_enabled: bool = True
    vision_enabled: bool = True
    auto_learn: bool = True
    personality: str = "friendly_professional"
    language: str = "it-IT"
    voice_model: str = "it-IT-GiuseppeNeural"
    max_tokens: int = 500
    temperature: float = 0.7


@dataclass
class GideonStats:
    """Statistiche runtime"""
    started_at: datetime = None
    requests_processed: int = 0
    actions_executed: int = 0
    predictions_made: int = 0
    analyses_completed: int = 0
    errors: int = 0
    avg_response_time_ms: float = 0.0
    
    def update_avg_response_time(self, new_time_ms: float):
        if self.requests_processed == 0:
            self.avg_response_time_ms = new_time_ms
        else:
            self.avg_response_time_ms = (
                (self.avg_response_time_ms * (self.requests_processed - 1) + new_time_ms) 
                / self.requests_processed
            )


@dataclass
class GideonResult:
    """Risultato standardizzato"""
    success: bool
    response: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    mode: str = "copilot"
    level: str = "normal"
    source: str = "gideon_v4"
    processing_time_ms: float = 0.0
    actions: List[Dict] = field(default_factory=list)
    audio_base64: Optional[str] = None
    error: Optional[str] = None
    version: str = "4.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "response": self.response,
            "data": self.data,
            "confidence": self.confidence,
            "mode": self.mode,
            "level": self.level,
            "source": self.source,
            "processing_time_ms": self.processing_time_ms,
            "actions": self.actions,
            "audio_base64": self.audio_base64,
            "error": self.error,
            "version": self.version
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                              GIDEON 4.0 CORE
# ═══════════════════════════════════════════════════════════════════════════════

class Gideon4:
    """
    🤖 GIDEON 4.0 - Ultimate Unified AI System
    
    Single entry point that integrates:
    - Brain v2.0 (NLP, Memory, Reasoning, AI Providers)
    - Cognitive v3.0 (Predictor, Analyzer, Simulator, Meta-Cognition)
    - Executive (Jarvis - Intent, Decisions, Execution)
    - Automation (Smart Actions, Workflows)
    - Interface (Voice, Vision, Chat)
    
    Usage:
        gideon = await Gideon4.create()
        result = await gideon.process("Your message here")
    """
    
    VERSION = "4.0.0"
    
    def __init__(self, config: GideonConfig = None):
        """Initialize Gideon 4.0 - use create() for async init"""
        self.config = config or GideonConfig()
        self.stats = GideonStats()
        self.is_initialized = False
        
        # Core components (lazy loaded)
        self._brain = None           # Brain/Assistant v2.0
        self._cognitive = None       # GideonCore v3.0
        self._executive = None       # JarvisSupervisor
        self._automation = None      # AutomationLayer
        self._orchestrator = None    # Pipeline Coordinator
        self._mode_manager = None    # Mode Management
        self._memory_system = None   # Memory System
        self._tts_service = None     # Text-to-Speech
        
        # Capability registry
        self._capabilities: Dict[CapabilityType, bool] = {
            cap: False for cap in CapabilityType
        }
        
        # Personality
        self.identity = {
            "name": "GIDEON",
            "full_name": "Generative Intelligence for Dynamic Executive Operations Network",
            "version": self.VERSION,
            "personality": self.config.personality,
            "language": self.config.language,
            "voice": self.config.voice_model
        }
        
        logger.info(f"🤖 GIDEON {self.VERSION} instance created")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                           INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    async def create(cls, config: GideonConfig = None) -> 'Gideon4':
        """Factory method per creare e inizializzare Gideon 4.0"""
        instance = cls(config)
        await instance.initialize()
        return instance
    
    async def initialize(self) -> bool:
        """
        Inizializza tutti i componenti di Gideon 4.0
        
        Carica in ordine:
        1. Brain (NLP, Memory, AI Providers)
        2. Cognitive (Analyzer, Predictor, Simulator)
        3. Executive (Intent, Decisions, Execution)
        4. Automation (Actions, Workflows)
        """
        if self.is_initialized:
            return True
            
        logger.info("═" * 60)
        logger.info("🤖 Inizializzazione GIDEON 4.0 Ultimate")
        logger.info("═" * 60)
        
        init_start = time.time()
        
        try:
            # 1. Brain v2.0
            await self._init_brain()
            
            # 2. Cognitive v3.0
            await self._init_cognitive()
            
            # 3. Executive (Jarvis)
            await self._init_executive()
            
            # 4. Automation Layer
            await self._init_automation()
            
            # 5. Orchestrator
            await self._init_orchestrator()
            
            # 6. Additional services
            await self._init_services()
            
            self.is_initialized = True
            self.stats.started_at = datetime.now()
            
            init_time = (time.time() - init_start) * 1000
            
            logger.info("═" * 60)
            logger.info(f"✅ GIDEON 4.0 inizializzato in {init_time:.0f}ms")
            logger.info(f"   Mode: {self.config.mode.value}")
            logger.info(f"   Level: {self.config.level.value}")
            logger.info(f"   Capabilities: {sum(self._capabilities.values())}/{len(self._capabilities)}")
            logger.info("═" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore inizializzazione GIDEON 4.0: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _init_brain(self):
        """Inizializza Brain v2.0 (NLP, Memory, Reasoning, AI)"""
        try:
            from brain.assistant import GideonAssistant
            
            self._brain = GideonAssistant()
            await self._brain.initialize()
            
            self._capabilities[CapabilityType.CHAT] = True
            self._capabilities[CapabilityType.REASONING] = True
            self._capabilities[CapabilityType.MEMORY] = True
            
            logger.info("  ✓ Brain v2.0 initialized")
            
        except Exception as e:
            logger.warning(f"  ⚠ Brain v2.0 not available: {e}")
    
    async def _init_cognitive(self):
        """Inizializza Cognitive v3.0 (Predictor, Analyzer, Simulator)"""
        try:
            from gideon import GideonCore
            
            self._cognitive = GideonCore()
            
            self._capabilities[CapabilityType.ANALYZE] = True
            self._capabilities[CapabilityType.PREDICT] = True
            self._capabilities[CapabilityType.SIMULATE] = True
            
            logger.info("  ✓ Cognitive v3.0 initialized")
            
        except Exception as e:
            logger.warning(f"  ⚠ Cognitive v3.0 not available: {e}")
    
    async def _init_executive(self):
        """Inizializza Executive (Jarvis)"""
        try:
            from jarvis import JarvisSupervisor
            from core.mode_manager import ModeManager
            
            self._mode_manager = ModeManager()
            
            self._executive = JarvisSupervisor(
                gideon_core=self._cognitive,
                automation_layer=self._automation,
                mode_manager=self._mode_manager
            )
            
            self._capabilities[CapabilityType.EXECUTE] = True
            
            logger.info("  ✓ Executive (Jarvis) initialized")
            
        except Exception as e:
            logger.warning(f"  ⚠ Executive not available: {e}")
    
    async def _init_automation(self):
        """Inizializza Automation Layer"""
        try:
            from automation import AutomationLayer
            
            self._automation = AutomationLayer()
            
            # Link to executive if available
            if self._executive:
                self._executive.link_automation(self._automation)
            
            logger.info("  ✓ Automation Layer initialized")
            
        except Exception as e:
            logger.warning(f"  ⚠ Automation Layer not available: {e}")
    
    async def _init_orchestrator(self):
        """Inizializza Orchestrator"""
        try:
            from core.orchestrator import Orchestrator
            
            self._orchestrator = Orchestrator(
                gideon_core=self._cognitive,
                jarvis_core=self._executive,
                automation_layer=self._automation,
                mode_manager=self._mode_manager
            )
            
            logger.info("  ✓ Orchestrator initialized")
            
        except Exception as e:
            logger.warning(f"  ⚠ Orchestrator not available: {e}")
    
    async def _init_services(self):
        """Inizializza servizi aggiuntivi (TTS, Vision, etc.)"""
        try:
            # TTS Service
            if self.config.voice_enabled:
                from tts_service import EdgeTTSService
                self._tts_service = EdgeTTSService()
                self._capabilities[CapabilityType.VOICE] = True
                logger.info("  ✓ TTS Service initialized")
        except Exception as e:
            logger.warning(f"  ⚠ TTS Service not available: {e}")
        
        try:
            # Vision capability
            if self.config.vision_enabled:
                from automation.smart_actions import smart_actions
                self._capabilities[CapabilityType.VISION] = True
                logger.info("  ✓ Vision Service available")
        except Exception as e:
            logger.warning(f"  ⚠ Vision Service not available: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                          MAIN PROCESSING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def process(
        self,
        text: str,
        context: Dict = None,
        mode: str = None,
        level: str = None,
        include_voice: bool = False,
        response_mode: str = None
    ) -> GideonResult:
        """
        🎯 Main processing pipeline - Single entry point for all requests
        
        Handles:
        - Questions and conversations
        - Commands and actions
        - Analysis requests
        - Predictions and simulations
        
        Args:
            text: User input
            context: Additional context
            mode: Override mode (passive/copilot/pilot/executive)
            level: Override level (quick/normal/advanced/expert)
            include_voice: Generate TTS audio
            response_mode: eco/fast/deep
            
        Returns:
            GideonResult with response, actions, confidence, etc.
        """
        start_time = time.time()
        self.stats.requests_processed += 1
        
        # Update mode/level if specified
        if mode:
            self._set_mode(mode)
        if level:
            self._set_level(level)
        
        # Response mode config
        resp_mode = ResponseMode(response_mode) if response_mode else self.config.response_mode
        token_config = self._get_token_config(resp_mode)
        
        try:
            # Check for quick patterns first
            quick_result = await self._try_quick_response(text)
            if quick_result:
                return quick_result
            
            # Use brain for main processing
            if self._brain and self._brain.is_ready():
                result = await self._brain.process_command(
                    text=text,
                    mode="text",
                    context=context,
                    max_tokens=token_config["max_tokens"],
                    temperature=token_config["temperature"]
                )
                
                response_text = result.get("text", result.get("response", ""))
                confidence = result.get("confidence", 0.9)
                actions = result.get("data", {}).get("actions", [])
                source = result.get("data", {}).get("source", "brain_v2")
                
            else:
                # Fallback
                response_text = self._local_response(text)
                confidence = 0.7
                actions = []
                source = "fallback"
            
            # Generate voice if requested
            audio_base64 = None
            if include_voice and self._tts_service:
                audio_base64 = await self._generate_voice(response_text)
            
            processing_time = (time.time() - start_time) * 1000
            self.stats.update_avg_response_time(processing_time)
            
            return GideonResult(
                success=True,
                response=response_text,
                confidence=confidence,
                mode=self.config.mode.value,
                level=self.config.level.value,
                source=source,
                processing_time_ms=round(processing_time, 2),
                actions=actions,
                audio_base64=audio_base64
            )
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"GIDEON 4.0 process error: {e}")
            
            return GideonResult(
                success=False,
                response=f"⚠️ Errore: {str(e)}",
                confidence=0.0,
                mode=self.config.mode.value,
                error=str(e),
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
    
    async def _try_quick_response(self, text: str) -> Optional[GideonResult]:
        """Try to handle quick patterns locally"""
        text_lower = text.lower().strip()
        
        # Time
        if any(w in text_lower for w in ["ora", "che ore", "orario"]):
            now = datetime.now()
            return GideonResult(
                success=True,
                response=f"🕐 Sono le {now.hour}:{now.minute:02d}",
                confidence=1.0,
                source="quick"
            )
        
        # Date
        if any(w in text_lower for w in ["data", "oggi", "giorno"]) and len(text_lower) < 20:
            now = datetime.now()
            giorni = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
            mesi = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
                   "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"]
            return GideonResult(
                success=True,
                response=f"📅 {giorni[now.weekday()]} {now.day} {mesi[now.month-1]} {now.year}",
                confidence=1.0,
                source="quick"
            )
        
        # Greeting
        if text_lower in ["ciao", "salve", "hey", "buongiorno", "buonasera", "hello"]:
            hour = datetime.now().hour
            if hour < 12:
                greeting = "Buongiorno"
            elif hour < 18:
                greeting = "Buon pomeriggio"
            else:
                greeting = "Buonasera"
            return GideonResult(
                success=True,
                response=f"👋 {greeting}! Sono GIDEON 4.0, come posso aiutarti?",
                confidence=1.0,
                source="quick"
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                         SPECIFIC CAPABILITIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def analyze(self, data: Union[str, Dict], context: Dict = None) -> GideonResult:
        """
        🔍 Analisi approfondita
        
        Uses the Cognitive v3.0 module for detailed analysis.
        """
        start_time = time.time()
        self.stats.analyses_completed += 1
        
        try:
            if self._cognitive:
                if isinstance(data, str):
                    # Text analysis
                    analysis = await self._cognitive.analyze({"text": data, "context": context})
                else:
                    analysis = await self._cognitive.analyze(data)
                
                return GideonResult(
                    success=True,
                    response="Analisi completata.",
                    data={"analysis": analysis},
                    confidence=0.9,
                    source="cognitive_v3",
                    processing_time_ms=round((time.time() - start_time) * 1000, 2)
                )
            else:
                # Fallback to process
                return await self.process(f"Analizza: {data}", context, response_mode="deep")
                
        except Exception as e:
            logger.error(f"Analyze error: {e}")
            return GideonResult(
                success=False,
                response=f"Errore analisi: {e}",
                error=str(e)
            )
    
    async def predict(self, context: Dict, horizon: str = "short") -> GideonResult:
        """
        🔮 Previsioni basate sul contesto
        
        Uses the Cognitive v3.0 Predictor.
        """
        start_time = time.time()
        self.stats.predictions_made += 1
        
        try:
            if self._cognitive:
                prediction = await self._cognitive.predict(context)
                
                return GideonResult(
                    success=True,
                    response="Previsione generata.",
                    data={"prediction": prediction, "horizon": horizon},
                    confidence=prediction.get("confidence", 0.8),
                    source="cognitive_v3",
                    processing_time_ms=round((time.time() - start_time) * 1000, 2)
                )
            else:
                return GideonResult(
                    success=False,
                    response="Modulo predittivo non disponibile.",
                    error="Cognitive module not initialized"
                )
                
        except Exception as e:
            logger.error(f"Predict error: {e}")
            return GideonResult(success=False, response=f"Errore previsione: {e}", error=str(e))
    
    async def simulate(
        self,
        scenario: Dict,
        method: str = "monte_carlo",
        iterations: int = 1000
    ) -> GideonResult:
        """
        🎲 Simulazione scenari
        
        Supports: monte_carlo, bayesian, what_if
        """
        start_time = time.time()
        
        try:
            if self._cognitive:
                if method == "monte_carlo":
                    result = await self._cognitive.simulate_monte_carlo(scenario, iterations)
                elif method == "what_if":
                    modifications = scenario.get("modifications", [])
                    result = await self._cognitive.what_if(scenario, modifications)
                else:
                    result = await self._cognitive.simulate(scenario)
                
                return GideonResult(
                    success=True,
                    response=f"Simulazione {method} completata.",
                    data={"simulation": result, "method": method, "iterations": iterations},
                    confidence=0.85,
                    source="cognitive_v3",
                    processing_time_ms=round((time.time() - start_time) * 1000, 2)
                )
            else:
                return GideonResult(
                    success=False,
                    response="Modulo simulazione non disponibile.",
                    error="Cognitive module not initialized"
                )
                
        except Exception as e:
            logger.error(f"Simulate error: {e}")
            return GideonResult(success=False, response=f"Errore simulazione: {e}", error=str(e))
    
    async def execute(self, action: str, params: Dict = None) -> GideonResult:
        """
        ⚡ Esegue un'azione
        
        Uses the Automation Layer and Executive module.
        """
        start_time = time.time()
        self.stats.actions_executed += 1
        
        # Check mode allows execution
        if self.config.mode == GideonMode.PASSIVE:
            return GideonResult(
                success=False,
                response="⚠️ Esecuzione non permessa in modalità PASSIVE.",
                error="Mode does not allow execution"
            )
        
        try:
            if self._automation:
                result = await self._automation.execute(action, params or {})
                
                return GideonResult(
                    success=result.get("success", False),
                    response=result.get("message", f"Azione '{action}' eseguita."),
                    data=result,
                    confidence=0.95,
                    source="automation",
                    actions=[{"action": action, "params": params, "result": result}],
                    processing_time_ms=round((time.time() - start_time) * 1000, 2)
                )
            else:
                return GideonResult(
                    success=False,
                    response="Automation layer non disponibile.",
                    error="Automation not initialized"
                )
                
        except Exception as e:
            logger.error(f"Execute error: {e}")
            return GideonResult(success=False, response=f"Errore esecuzione: {e}", error=str(e))
    
    async def vision(
        self,
        image_base64: str = None,
        image_path: str = None,
        question: str = "Analizza questa immagine in dettaglio"
    ) -> GideonResult:
        """
        👁️ Analisi immagini con AI Vision
        """
        start_time = time.time()
        
        if not self._capabilities.get(CapabilityType.VISION):
            return GideonResult(
                success=False,
                response="Vision AI non disponibile.",
                error="Vision capability not initialized"
            )
        
        try:
            from automation.smart_actions import smart_actions
            
            if image_path:
                result = await smart_actions.analyze_image(image_path, question)
            elif image_base64:
                import tempfile
                import base64
                from pathlib import Path
                
                temp_path = Path(tempfile.gettempdir()) / f"gideon_vision_{int(time.time())}.jpg"
                
                if "," in image_base64:
                    image_base64 = image_base64.split(",")[1]
                
                temp_path.write_bytes(base64.b64decode(image_base64))
                result = await smart_actions.analyze_image(str(temp_path), question)
            else:
                return GideonResult(
                    success=False,
                    response="Fornire image_base64 o image_path",
                    error="No image provided"
                )
            
            return GideonResult(
                success=True,
                response=result.get("analysis", "Analisi completata."),
                data=result,
                confidence=0.9,
                source="vision_ai",
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
            
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return GideonResult(success=False, response=f"Errore vision: {e}", error=str(e))
    
    async def speak(self, text: str) -> GideonResult:
        """
        🔊 Genera audio TTS
        """
        try:
            audio_base64 = await self._generate_voice(text)
            
            if audio_base64:
                return GideonResult(
                    success=True,
                    response=text,
                    audio_base64=audio_base64,
                    source="tts"
                )
            else:
                return GideonResult(
                    success=False,
                    response="TTS non disponibile.",
                    error="TTS service not available"
                )
                
        except Exception as e:
            logger.error(f"Speak error: {e}")
            return GideonResult(success=False, response=f"Errore TTS: {e}", error=str(e))
    
    async def recommend(self, query: str, context: Dict = None) -> GideonResult:
        """
        💡 Ottiene raccomandazione completa
        
        Pipeline: analizza → prevede → simula → classifica → suggerisce
        """
        start_time = time.time()
        
        try:
            if self._cognitive:
                recommendation = await self._cognitive.get_recommendation(query, context)
                
                return GideonResult(
                    success=True,
                    response=f"Raccomandazione per: {query}",
                    data=recommendation,
                    confidence=recommendation.get("confidence", 0.85),
                    source="cognitive_v3_full",
                    processing_time_ms=round((time.time() - start_time) * 1000, 2)
                )
            else:
                return await self.process(query, context, response_mode="deep")
                
        except Exception as e:
            logger.error(f"Recommend error: {e}")
            return GideonResult(success=False, response=f"Errore raccomandazione: {e}", error=str(e))
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                            CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _set_mode(self, mode: str):
        """Set operating mode"""
        try:
            self.config.mode = GideonMode(mode.lower())
            if self._mode_manager:
                self._mode_manager.set_mode(mode)
            logger.info(f"🎛️ GIDEON mode: {self.config.mode.value}")
        except ValueError:
            logger.warning(f"Invalid mode: {mode}")
    
    def _set_level(self, level: str):
        """Set response level"""
        try:
            self.config.level = GideonLevel(level.lower())
            logger.info(f"📊 GIDEON level: {self.config.level.value}")
        except ValueError:
            logger.warning(f"Invalid level: {level}")
    
    def set_mode(self, mode: str):
        """Public method to set mode"""
        self._set_mode(mode)
    
    def set_level(self, level: str):
        """Public method to set level"""
        self._set_level(level)
    
    def _get_token_config(self, mode: ResponseMode) -> Dict:
        """Get token configuration for response mode"""
        configs = {
            ResponseMode.ECO: {"max_tokens": 150, "temperature": 0.3},
            ResponseMode.FAST: {"max_tokens": 300, "temperature": 0.5},
            ResponseMode.DEEP: {"max_tokens": 800, "temperature": 0.7}
        }
        return configs.get(mode, configs[ResponseMode.FAST])
    
    def _local_response(self, text: str) -> str:
        """Local fallback response"""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["ciao", "salve", "hello"]):
            return "👋 Ciao! Sono GIDEON 4.0, come posso aiutarti?"
        
        if any(w in text_lower for w in ["grazie", "thanks"]):
            return "😊 Prego! Sono sempre qui per aiutarti."
        
        return "🤖 Ricevuto. Elaboro la tua richiesta..."
    
    async def _generate_voice(self, text: str) -> Optional[str]:
        """Generate TTS audio"""
        try:
            if self._tts_service:
                audio_bytes = await self._tts_service.synthesize(text)
                if audio_bytes:
                    import base64
                    return base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                              STATUS & INFO
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "version": self.VERSION,
            "initialized": self.is_initialized,
            "mode": self.config.mode.value,
            "level": self.config.level.value,
            "identity": self.identity,
            "capabilities": {k.value: v for k, v in self._capabilities.items()},
            "stats": {
                "started_at": self.stats.started_at.isoformat() if self.stats.started_at else None,
                "requests_processed": self.stats.requests_processed,
                "actions_executed": self.stats.actions_executed,
                "predictions_made": self.stats.predictions_made,
                "analyses_completed": self.stats.analyses_completed,
                "errors": self.stats.errors,
                "avg_response_time_ms": round(self.stats.avg_response_time_ms, 2)
            },
            "components": {
                "brain": self._brain is not None and self._brain.is_ready() if self._brain else False,
                "cognitive": self._cognitive is not None,
                "executive": self._executive is not None,
                "automation": self._automation is not None,
                "orchestrator": self._orchestrator is not None,
                "tts": self._tts_service is not None
            }
        }
    
    def get_capabilities(self) -> List[str]:
        """Get list of active capabilities"""
        return [cap.value for cap, active in self._capabilities.items() if active]
    
    def get_help(self) -> str:
        """Get help text"""
        caps = self.get_capabilities()
        return f"""🤖 **GIDEON 4.0** - Ultimate Unified AI System

**Versione:** {self.VERSION}
**Modalità:** {self.config.mode.value}
**Livello:** {self.config.level.value}

**Capabilities attive:** {', '.join(caps)}

**Comandi principali:**
• `process(text)` - Pipeline principale
• `analyze(data)` - Analisi approfondita
• `predict(context)` - Previsioni
• `simulate(scenario)` - Simulazioni
• `execute(action)` - Esecuzione azioni
• `vision(image)` - Analisi immagini
• `speak(text)` - Sintesi vocale
• `recommend(query)` - Raccomandazioni

**Modalità:**
• `passive` - Solo analisi
• `copilot` - Suggerimenti + conferma
• `pilot` - Autonomo
• `executive` - Orchestrazione completa
"""
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                              SHUTDOWN
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def shutdown(self):
        """Cleanup all resources"""
        logger.info("🛑 Shutdown GIDEON 4.0...")
        
        if self._brain:
            await self._brain.shutdown()
        
        if self._orchestrator and hasattr(self._orchestrator, 'shutdown'):
            await self._orchestrator.shutdown()
        
        self.is_initialized = False
        logger.info("✅ GIDEON 4.0 shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
#                            SINGLETON MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

_gideon_v4_instance: Optional[Gideon4] = None


def get_gideon_v4() -> Gideon4:
    """Get Gideon 4.0 singleton instance"""
    global _gideon_v4_instance
    if _gideon_v4_instance is None:
        _gideon_v4_instance = Gideon4()
    return _gideon_v4_instance


async def init_gideon_v4(config: GideonConfig = None) -> Gideon4:
    """Initialize and return Gideon 4.0"""
    global _gideon_v4_instance
    if _gideon_v4_instance is None:
        _gideon_v4_instance = await Gideon4.create(config)
    elif not _gideon_v4_instance.is_initialized:
        await _gideon_v4_instance.initialize()
    return _gideon_v4_instance


# ═══════════════════════════════════════════════════════════════════════════════
#                        BACKWARD COMPATIBILITY ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

# Alias for GideonUnified compatibility
GideonUnified = Gideon4
get_gideon = get_gideon_v4
init_gideon = init_gideon_v4
