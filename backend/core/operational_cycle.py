"""
🔄 GIDEON 3.0 - Operational Cycle Manager

Ciclo Operativo Completo:
1. Attivazione tramite comando vocale (wake word detection)
2. Ascolto continuo e monitoraggio contesto
3. Intent detection (Jarvis)
4. Processing pipeline
5. Feedback e risposta
"""

import asyncio
import inspect
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


# ============ ENUMS ============

class CycleState(Enum):
    """Stati del ciclo operativo"""
    DORMANT = "dormant"           # Sistema spento/sleep
    STANDBY = "standby"           # In attesa wake word
    AWAKENING = "awakening"       # Risveglio in corso
    LISTENING = "listening"       # Ascolto attivo
    PROCESSING = "processing"     # Elaborazione in corso
    RESPONDING = "responding"     # Risposta in corso
    CONFIRMING = "confirming"     # Attesa conferma
    EXECUTING = "executing"       # Esecuzione azione
    COOLDOWN = "cooldown"         # Post-risposta, pronto per follow-up
    ERROR = "error"               # Stato errore


class ActivationType(Enum):
    """Tipi di attivazione"""
    WAKE_WORD = "wake_word"       # Attivato da wake word
    BUTTON = "button"             # Attivato da pulsante
    HOTKEY = "hotkey"             # Attivato da shortcut tastiera
    API = "api"                   # Attivato via API
    SCHEDULED = "scheduled"       # Attivato da scheduler
    CONTINUOUS = "continuous"     # Ascolto continuo (hands-free)
    CONTEXT = "context"           # Attivato da cambio contesto


class ListeningMode(Enum):
    """Modalità di ascolto"""
    WAKE_WORD_ONLY = "wake_word_only"     # Solo wake word
    CONTINUOUS = "continuous"              # Ascolto continuo
    PUSH_TO_TALK = "push_to_talk"         # Premi per parlare
    AUTO_DETECT = "auto_detect"           # Rilevamento automatico


class ContextType(Enum):
    """Tipi di contesto monitorato"""
    AUDIO = "audio"
    VISUAL = "visual"
    SYSTEM = "system"
    USER = "user"
    TEMPORAL = "temporal"
    LOCATION = "location"


# ============ DATA CLASSES ============

@dataclass
class WakeWordConfig:
    """Configurazione wake word"""
    primary: str = "gideon"
    alternatives: List[str] = field(default_factory=lambda: ["jarvis", "assistente", "hey gideon"])
    sensitivity: float = 0.7  # 0-1, più alto = più sensibile
    cooldown_ms: int = 500    # Tempo minimo tra attivazioni
    require_prefix: bool = False  # Se True, wake word deve essere all'inizio


@dataclass
class ListeningConfig:
    """Configurazione ascolto"""
    mode: ListeningMode = ListeningMode.WAKE_WORD_ONLY
    timeout_seconds: float = 10.0
    silence_threshold_ms: int = 1500      # Silenzio per considerare frase finita
    min_phrase_length: int = 2            # Lunghezza minima frase (caratteri)
    language: str = "it-IT"
    continuous_timeout_seconds: float = 30.0  # Timeout per modalità continua
    auto_restart: bool = True             # Riavvia ascolto automaticamente


@dataclass
class ContextSnapshot:
    """Snapshot del contesto corrente"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Audio context
    audio_level: float = 0.0
    is_speech_detected: bool = False
    background_noise_level: float = 0.0
    
    # System context
    active_application: str = ""
    system_idle_time_ms: int = 0
    clipboard_content_type: str = ""
    
    # User context
    user_activity: str = "unknown"
    last_interaction: Optional[datetime] = None
    session_duration_minutes: float = 0.0
    
    # Temporal context
    time_of_day: str = ""  # morning, afternoon, evening, night
    day_of_week: str = ""
    
    # Custom context
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "audio": {
                "level": self.audio_level,
                "speech_detected": self.is_speech_detected,
                "background_noise": self.background_noise_level
            },
            "system": {
                "active_app": self.active_application,
                "idle_time_ms": self.system_idle_time_ms,
                "clipboard_type": self.clipboard_content_type
            },
            "user": {
                "activity": self.user_activity,
                "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
                "session_minutes": self.session_duration_minutes
            },
            "temporal": {
                "time_of_day": self.time_of_day,
                "day_of_week": self.day_of_week
            },
            "custom": self.custom
        }


@dataclass
class CycleEvent:
    """Evento del ciclo operativo"""
    id: str
    timestamp: datetime
    event_type: str
    state_from: CycleState
    state_to: CycleState
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "state_from": self.state_from.value,
            "state_to": self.state_to.value,
            "details": self.details,
            "duration_ms": self.duration_ms
        }


@dataclass
class IntentResult:
    """Risultato intent detection"""
    name: str
    confidence: float
    entities: Dict[str, Any]
    text: str
    category: str = "general"
    requires_confirmation: bool = False
    is_actionable: bool = True
    suggested_response: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "entities": self.entities,
            "text": self.text,
            "category": self.category,
            "requires_confirmation": self.requires_confirmation,
            "is_actionable": self.is_actionable,
            "suggested_response": self.suggested_response
        }


@dataclass
class CycleSession:
    """Sessione del ciclo operativo"""
    id: str
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    
    # Stati
    current_state: CycleState = CycleState.DORMANT
    previous_state: CycleState = CycleState.DORMANT
    
    # Attivazione
    activation_type: Optional[ActivationType] = None
    activation_text: str = ""
    
    # Contesto
    context: Optional[ContextSnapshot] = None
    
    # Intent
    intent: Optional[IntentResult] = None
    
    # Interazioni
    turns: int = 0
    events: List[CycleEvent] = field(default_factory=list)
    
    # Metriche
    total_listening_ms: float = 0.0
    total_processing_ms: float = 0.0
    
    @property
    def duration(self) -> timedelta:
        end = self.ended_at or datetime.now()
        return end - self.started_at
    
    @property
    def is_active(self) -> bool:
        return self.current_state not in [CycleState.DORMANT, CycleState.ERROR]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "current_state": self.current_state.value,
            "activation_type": self.activation_type.value if self.activation_type else None,
            "turns": self.turns,
            "duration_seconds": self.duration.total_seconds(),
            "is_active": self.is_active
        }


# ============ WAKE WORD DETECTOR ============

class WakeWordDetector:
    """
    Rileva wake word nel testo.
    Supporta pattern multipli e varianti fonetiche.
    """
    
    def __init__(self, config: WakeWordConfig = None):
        self.config = config or WakeWordConfig()
        self._patterns: List[re.Pattern] = []
        self._last_activation: Optional[datetime] = None
        self._build_patterns()
    
    def _build_patterns(self):
        """Costruisce pattern regex"""
        all_words = [self.config.primary] + self.config.alternatives
        
        for word in all_words:
            # Pattern base
            pattern = re.escape(word.lower())
            
            # Varianti con prefissi
            prefixes = ["hey", "ehi", "ok", "ciao"]
            pattern_with_prefix = f"({'|'.join(prefixes)}\\s+)?{pattern}"
            
            if self.config.require_prefix:
                self._patterns.append(re.compile(f"^{pattern_with_prefix}", re.IGNORECASE))
            else:
                self._patterns.append(re.compile(pattern_with_prefix, re.IGNORECASE))
    
    def detect(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Rileva wake word nel testo.
        
        Returns:
            (detected, wake_word_found, command_after_wake_word)
        """
        # Check cooldown
        if self._last_activation:
            elapsed = (datetime.now() - self._last_activation).total_seconds() * 1000
            if elapsed < self.config.cooldown_ms:
                return False, None, None
        
        text_lower = text.lower().strip()
        
        for pattern in self._patterns:
            match = pattern.search(text_lower)
            if match:
                wake_word = match.group()
                command = text[match.end():].strip()
                
                self._last_activation = datetime.now()
                
                logger.debug(f"Wake word detected: '{wake_word}' in '{text}'")
                return True, wake_word, command if command else None
        
        return False, None, None
    
    def add_wake_word(self, word: str):
        """Aggiunge wake word"""
        if word.lower() not in [w.lower() for w in self.config.alternatives]:
            self.config.alternatives.append(word)
            self._build_patterns()
    
    def remove_wake_word(self, word: str):
        """Rimuove wake word"""
        word_lower = word.lower()
        self.config.alternatives = [w for w in self.config.alternatives 
                                    if w.lower() != word_lower]
        self._build_patterns()
    
    def set_sensitivity(self, sensitivity: float):
        """Imposta sensibilità (0-1)"""
        self.config.sensitivity = max(0, min(1, sensitivity))


# ============ CONTEXT MONITOR ============

class ContextMonitor:
    """
    Monitora il contesto del sistema per decisioni intelligenti.
    """
    
    def __init__(self):
        self._current_context = ContextSnapshot()
        self._context_history: List[ContextSnapshot] = []
        self._max_history = 100
        self._monitors: Dict[ContextType, Callable] = {}
        self._update_interval = 1.0  # secondi
        self._running = False
        
        # Setup default monitors
        self._setup_default_monitors()
    
    def _setup_default_monitors(self):
        """Configura monitor default"""
        self._monitors[ContextType.TEMPORAL] = self._monitor_temporal
        self._monitors[ContextType.USER] = self._monitor_user
    
    def _monitor_temporal(self) -> Dict[str, Any]:
        """Monitora contesto temporale"""
        now = datetime.now()
        hour = now.hour
        
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 18:
            time_of_day = "afternoon"
        elif 18 <= hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        return {
            "time_of_day": time_of_day,
            "day_of_week": now.strftime("%A").lower(),
            "hour": hour,
            "minute": now.minute
        }
    
    def _monitor_user(self) -> Dict[str, Any]:
        """Monitora contesto utente"""
        return {
            "activity": "active" if self._current_context.last_interaction and 
                       (datetime.now() - self._current_context.last_interaction).seconds < 60 
                       else "idle",
            "last_interaction": self._current_context.last_interaction
        }
    
    async def start(self):
        """Avvia monitoraggio continuo"""
        self._running = True
        logger.info("Context monitor started")
        
        while self._running:
            await self._update_context()
            await asyncio.sleep(self._update_interval)
    
    async def stop(self):
        """Ferma monitoraggio"""
        self._running = False
        logger.info("Context monitor stopped")
    
    async def _update_context(self):
        """Aggiorna contesto"""
        new_context = ContextSnapshot(timestamp=datetime.now())
        
        # Esegui tutti i monitor
        for context_type, monitor_func in self._monitors.items():
            try:
                if inspect.iscoroutinefunction(monitor_func):
                    result = await monitor_func()
                else:
                    result = monitor_func()
                
                # Applica risultati al contesto
                if context_type == ContextType.TEMPORAL:
                    new_context.time_of_day = result.get("time_of_day", "")
                    new_context.day_of_week = result.get("day_of_week", "")
                elif context_type == ContextType.USER:
                    new_context.user_activity = result.get("activity", "unknown")
                
            except Exception as e:
                logger.error(f"Error in context monitor {context_type}: {e}")
        
        # Salva contesto
        self._current_context = new_context
        self._context_history.append(new_context)
        
        # Cleanup history
        if len(self._context_history) > self._max_history:
            self._context_history = self._context_history[-self._max_history:]
    
    def get_context(self) -> ContextSnapshot:
        """Ottiene contesto corrente"""
        return self._current_context
    
    def record_interaction(self):
        """Registra interazione utente"""
        self._current_context.last_interaction = datetime.now()
    
    def add_monitor(self, context_type: ContextType, monitor_func: Callable):
        """Aggiunge monitor custom"""
        self._monitors[context_type] = monitor_func
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Riassunto contesto per decisioni"""
        ctx = self._current_context
        return {
            "time_context": ctx.time_of_day,
            "user_state": ctx.user_activity,
            "session_duration": ctx.session_duration_minutes,
            "recent_interaction": ctx.last_interaction is not None and 
                                 (datetime.now() - ctx.last_interaction).seconds < 60
        }


# ============ OPERATIONAL CYCLE MANAGER ============

class OperationalCycleManager:
    """
    Manager del Ciclo Operativo Completo.
    
    Ciclo:
    1. STANDBY → Attende wake word
    2. AWAKENING → Attivazione rilevata
    3. LISTENING → Ascolto input utente
    4. PROCESSING → Jarvis interpreta intent
    5. RESPONDING/EXECUTING → Risponde o esegue
    6. COOLDOWN → Pronto per follow-up
    7. STANDBY → Torna in attesa
    """
    
    def __init__(self, jarvis_core=None, mode_manager=None):
        # Componenti
        self.jarvis = jarvis_core
        self.mode_manager = mode_manager
        self.wake_detector = WakeWordDetector()
        self.context_monitor = ContextMonitor()
        
        # Configurazioni
        self.wake_config = WakeWordConfig()
        self.listening_config = ListeningConfig()
        
        # Stato
        self._current_session: Optional[CycleSession] = None
        self._sessions_history: List[CycleSession] = []
        self._state = CycleState.DORMANT
        self._event_counter = 0
        
        # Timers
        self._listening_start: Optional[datetime] = None
        self._cooldown_duration = 5.0  # secondi per follow-up senza wake word
        
        # Callbacks
        self._on_state_change: List[Callable] = []
        self._on_wake_detected: List[Callable] = []
        self._on_intent_detected: List[Callable] = []
        self._on_cycle_complete: List[Callable] = []
        
        # Intent handlers
        self._intent_handlers: Dict[str, Callable] = {}
        
        # Stats
        self._stats = {
            "total_activations": 0,
            "total_sessions": 0,
            "total_intents": 0,
            "successful_interactions": 0,
            "errors": 0
        }
    
    # === State Management ===
    
    @property
    def state(self) -> CycleState:
        return self._state
    
    @property
    def is_listening(self) -> bool:
        return self._state in [CycleState.LISTENING, CycleState.COOLDOWN]
    
    @property
    def is_active(self) -> bool:
        return self._state not in [CycleState.DORMANT, CycleState.ERROR]
    
    async def _transition_to(self, new_state: CycleState, details: Dict = None):
        """Transizione di stato"""
        if self._state == new_state:
            return
        
        old_state = self._state
        self._state = new_state
        
        # Log evento
        event = CycleEvent(
            id=f"evt_{self._event_counter:06d}",
            timestamp=datetime.now(),
            event_type="state_change",
            state_from=old_state,
            state_to=new_state,
            details=details or {}
        )
        self._event_counter += 1
        
        if self._current_session:
            self._current_session.previous_state = old_state
            self._current_session.current_state = new_state
            self._current_session.events.append(event)
        
        logger.info(f"Cycle state: {old_state.value} → {new_state.value}")
        
        # Notifica callbacks
        for callback in self._on_state_change:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(old_state, new_state, details)
                else:
                    callback(old_state, new_state, details)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    # === Session Management ===
    
    def _create_session(self, activation_type: ActivationType) -> CycleSession:
        """Crea nuova sessione"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._stats['total_sessions']}"
        
        session = CycleSession(
            id=session_id,
            activation_type=activation_type,
            context=self.context_monitor.get_context()
        )
        
        self._current_session = session
        self._stats["total_sessions"] += 1
        
        logger.info(f"New session created: {session_id}")
        return session
    
    def _end_session(self):
        """Termina sessione corrente"""
        if self._current_session:
            self._current_session.ended_at = datetime.now()
            self._sessions_history.append(self._current_session)
            
            # Cleanup history
            if len(self._sessions_history) > 100:
                self._sessions_history = self._sessions_history[-100:]
            
            logger.info(f"Session ended: {self._current_session.id}")
            self._current_session = None
    
    # === Main Cycle Operations ===
    
    async def start(self):
        """Avvia il ciclo operativo"""
        logger.info("Starting operational cycle...")
        
        # Avvia context monitor
        asyncio.create_task(self.context_monitor.start())
        
        # Vai in standby
        await self._transition_to(CycleState.STANDBY)
        
        logger.info("✅ Operational cycle started - waiting for activation")
    
    async def stop(self):
        """Ferma il ciclo operativo"""
        logger.info("Stopping operational cycle...")
        
        await self.context_monitor.stop()
        
        if self._current_session:
            self._end_session()
        
        await self._transition_to(CycleState.DORMANT)
        
        logger.info("✅ Operational cycle stopped")
    
    async def process_audio_input(self, text: str, 
                                   source: ActivationType = ActivationType.WAKE_WORD) -> Dict[str, Any]:
        """
        Processa input audio/testuale.
        Entry point principale per il ciclo.
        
        Args:
            text: Testo trascritto o digitato
            source: Tipo di attivazione
        
        Returns:
            Risultato del ciclo
        """
        result = {
            "success": False,
            "state": self._state.value,
            "intent": None,
            "response": None,
            "requires_action": False
        }
        
        try:
            # Registra interazione
            self.context_monitor.record_interaction()
            
            # === CHECK WAKE WORD ===
            if self._state == CycleState.STANDBY:
                detected, wake_word, command = self.wake_detector.detect(text)
                
                if detected:
                    # Attivazione!
                    await self._handle_activation(wake_word, command, source)
                    
                    if command:
                        # Comando inline dopo wake word
                        text = command
                    else:
                        # Solo wake word, attendi comando
                        result["success"] = True
                        result["state"] = self._state.value
                        result["response"] = "Dimmi"
                        return result
                else:
                    # Nessun wake word, ignora
                    return result
            
            # === CHECK COOLDOWN (follow-up senza wake word) ===
            elif self._state == CycleState.COOLDOWN:
                # Accetta input diretto durante cooldown
                pass
            
            # === LISTENING/PROCESSING ===
            if self._state in [CycleState.LISTENING, CycleState.COOLDOWN, CycleState.AWAKENING]:
                return await self._process_command(text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            await self._transition_to(CycleState.ERROR, {"error": str(e)})
            self._stats["errors"] += 1
            result["error"] = str(e)
            return result
    
    async def _handle_activation(self, wake_word: str, command: str, 
                                  source: ActivationType):
        """Gestisce attivazione"""
        self._stats["total_activations"] += 1
        
        # Crea sessione
        session = self._create_session(source)
        session.activation_text = wake_word
        
        # Transizione
        await self._transition_to(CycleState.AWAKENING, {
            "wake_word": wake_word,
            "has_command": bool(command)
        })
        
        # Notifica callbacks
        for callback in self._on_wake_detected:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(wake_word, command)
                else:
                    callback(wake_word, command)
            except Exception as e:
                logger.error(f"Wake callback error: {e}")
        
        # Se nessun comando, vai in ascolto
        if not command:
            await self._transition_to(CycleState.LISTENING)
            self._listening_start = datetime.now()
        
        logger.info(f"Activation: wake_word='{wake_word}', command='{command}'")
    
    async def _process_command(self, text: str) -> Dict[str, Any]:
        """Processa comando utente"""
        result = {
            "success": False,
            "state": self._state.value,
            "intent": None,
            "response": None,
            "requires_action": False
        }
        
        # Aggiorna stato
        await self._transition_to(CycleState.PROCESSING)
        
        # === INTENT DETECTION (Jarvis) ===
        intent = await self._detect_intent(text)
        result["intent"] = intent.to_dict() if intent else None
        
        if self._current_session:
            self._current_session.intent = intent
            self._current_session.turns += 1
        
        self._stats["total_intents"] += 1
        
        # Notifica callbacks
        for callback in self._on_intent_detected:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(intent)
                else:
                    callback(intent)
            except Exception as e:
                logger.error(f"Intent callback error: {e}")
        
        # === HANDLE INTENT ===
        if intent and intent.is_actionable:
            # Cerca handler
            handler = self._intent_handlers.get(intent.name)
            
            if handler:
                await self._transition_to(CycleState.EXECUTING)
                
                try:
                    if inspect.iscoroutinefunction(handler):
                        handler_result = await handler(intent)
                    else:
                        handler_result = handler(intent)
                    
                    result["response"] = handler_result.get("response", "")
                    result["requires_action"] = handler_result.get("action_taken", False)
                    result["success"] = True
                    
                except Exception as e:
                    logger.error(f"Intent handler error: {e}")
                    result["error"] = str(e)
            else:
                # Nessun handler, usa Jarvis
                if self.jarvis and hasattr(self.jarvis, 'process'):
                    await self._transition_to(CycleState.RESPONDING)
                    jarvis_result = await self.jarvis.process(text)
                    result["response"] = jarvis_result.get("response", "")
                    result["success"] = True
                else:
                    result["response"] = intent.suggested_response or "Non ho capito come aiutarti con questo."
        else:
            result["response"] = "Non ho capito. Puoi ripetere?"
        
        # === COMPLETION ===
        await self._transition_to(CycleState.COOLDOWN)
        self._stats["successful_interactions"] += 1
        
        # Callback ciclo completato
        for callback in self._on_cycle_complete:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Cycle complete callback error: {e}")
        
        # Timer per tornare in standby
        asyncio.create_task(self._cooldown_timer())
        
        return result
    
    async def _detect_intent(self, text: str) -> Optional[IntentResult]:
        """Rileva intent usando Jarvis"""
        if self.jarvis and hasattr(self.jarvis, 'interpreter'):
            try:
                jarvis_intent = self.jarvis.interpreter.interpret(text)
                
                return IntentResult(
                    name=jarvis_intent.name,
                    confidence=jarvis_intent.confidence,
                    entities=jarvis_intent.entities,
                    text=text,
                    category=str(jarvis_intent.category) if hasattr(jarvis_intent, 'category') else "general",
                    requires_confirmation=jarvis_intent.confidence < 0.7,
                    is_actionable=jarvis_intent.confidence >= 0.5,
                    suggested_response=getattr(jarvis_intent, 'suggested_response', '')
                )
            except Exception as e:
                logger.error(f"Intent detection error: {e}")
        
        # Fallback: intent base
        return IntentResult(
            name="unknown",
            confidence=0.3,
            entities={},
            text=text,
            is_actionable=False,
            suggested_response="Non ho capito la richiesta."
        )
    
    async def _cooldown_timer(self):
        """Timer per tornare in standby dopo cooldown"""
        await asyncio.sleep(self._cooldown_duration)
        
        if self._state == CycleState.COOLDOWN:
            await self._transition_to(CycleState.STANDBY)
            self._end_session()
    
    # === Manual Controls ===
    
    async def activate_manually(self, source: ActivationType = ActivationType.BUTTON):
        """Attivazione manuale (pulsante/API)"""
        await self._handle_activation("manual", None, source)
        await self._transition_to(CycleState.LISTENING)
    
    async def set_continuous_listening(self, enabled: bool):
        """Abilita/disabilita ascolto continuo"""
        self.listening_config.mode = ListeningMode.CONTINUOUS if enabled else ListeningMode.WAKE_WORD_ONLY
        
        if enabled:
            await self._transition_to(CycleState.LISTENING)
            logger.info("Continuous listening enabled")
        else:
            await self._transition_to(CycleState.STANDBY)
            logger.info("Continuous listening disabled")
    
    async def cancel_current(self):
        """Annulla operazione corrente"""
        if self._current_session:
            self._end_session()
        await self._transition_to(CycleState.STANDBY)
    
    # === Handlers & Callbacks ===
    
    def register_intent_handler(self, intent_name: str, handler: Callable):
        """Registra handler per intent specifico"""
        self._intent_handlers[intent_name] = handler
        logger.debug(f"Registered intent handler: {intent_name}")
    
    def on_state_change(self, callback: Callable):
        """Registra callback per cambio stato"""
        self._on_state_change.append(callback)
    
    def on_wake_detected(self, callback: Callable):
        """Registra callback per wake word detection"""
        self._on_wake_detected.append(callback)
    
    def on_intent_detected(self, callback: Callable):
        """Registra callback per intent detection"""
        self._on_intent_detected.append(callback)
    
    def on_cycle_complete(self, callback: Callable):
        """Registra callback per ciclo completato"""
        self._on_cycle_complete.append(callback)
    
    # === Configuration ===
    
    def configure_wake_word(self, **kwargs):
        """Configura wake word"""
        for key, value in kwargs.items():
            if hasattr(self.wake_config, key):
                setattr(self.wake_config, key, value)
        self.wake_detector = WakeWordDetector(self.wake_config)
    
    def configure_listening(self, **kwargs):
        """Configura ascolto"""
        for key, value in kwargs.items():
            if hasattr(self.listening_config, key):
                setattr(self.listening_config, key, value)
    
    def set_cooldown_duration(self, seconds: float):
        """Imposta durata cooldown"""
        self._cooldown_duration = seconds
    
    # === Status & Stats ===
    
    def get_status(self) -> Dict[str, Any]:
        """Stato corrente"""
        return {
            "state": self._state.value,
            "is_active": self.is_active,
            "is_listening": self.is_listening,
            "listening_mode": self.listening_config.mode.value,
            "current_session": self._current_session.to_dict() if self._current_session else None,
            "context": self.context_monitor.get_context_summary(),
            "wake_words": [self.wake_config.primary] + self.wake_config.alternatives
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche"""
        return {
            **self._stats,
            "sessions_in_memory": len(self._sessions_history),
            "intent_handlers": list(self._intent_handlers.keys())
        }
    
    def get_session_history(self, limit: int = 20) -> List[Dict]:
        """Storico sessioni"""
        return [s.to_dict() for s in self._sessions_history[-limit:]]


# ============ FACTORY ============

def create_operational_cycle(jarvis_core=None, mode_manager=None) -> OperationalCycleManager:
    """Crea istanza del ciclo operativo"""
    return OperationalCycleManager(
        jarvis_core=jarvis_core,
        mode_manager=mode_manager
    )
