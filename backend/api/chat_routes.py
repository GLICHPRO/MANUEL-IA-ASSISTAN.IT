"""
💬 Chat API Routes - GIDEON 3.0

API per chat testuale tipo WhatsApp:
- Storico completo conversazioni
- Indicatore livello operativo
- Modalità multi-turno con contesto
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import asyncio
import json
import uuid
import hashlib
import base64
from pathlib import Path
from loguru import logger

router = APIRouter(prefix="/chat", tags=["Chat"])

# TTS imports for combined endpoint
try:
    from edge_tts import Communicate
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge_tts not available for combined chat+voice")

# Reference to global assistant instance (set from main.py)
_chat_assistant_instance = None

def set_chat_assistant_instance(assistant):
    """Set the global assistant instance for chat routes"""
    global _chat_assistant_instance
    _chat_assistant_instance = assistant
    logger.info("✅ Chat routes: Assistant instance linked")


# ============ ENUMS ============

class OperatingLevel(str, Enum):
    """Livello operativo del sistema"""
    PASSIVE = "passive"     # Solo risponde
    COPILOT = "copilot"     # Suggerisce e chiede conferma
    PILOT = "pilot"         # Esegue senza conferma


class ResponseMode(str, Enum):
    """Modalità di risposta AI"""
    ECO = "eco"           # Base: economico, risposte brevi
    FAST = "fast"         # Veloce: ottimizzato per velocità  
    DEEP = "deep"         # Approfondito: risposte dettagliate


# Configurazioni per modalità risposta
RESPONSE_MODE_CONFIG = {
    "eco": {
        "max_tokens": 150,
        "temperature": 0.3,
        "context_limit": 2,
        "model_preference": "fast",  # modelli più leggeri
        "description": "💚 ECO - Risposte brevi e veloci"
    },
    "fast": {
        "max_tokens": 300,
        "temperature": 0.5,
        "context_limit": 4,
        "model_preference": "balanced",
        "description": "⚡ FAST - Bilanciato velocità/qualità"
    },
    "deep": {
        "max_tokens": 800,
        "temperature": 0.7,
        "context_limit": 8,
        "model_preference": "quality",  # modelli più potenti
        "description": "🧠 DEEP - Risposte approfondite"
    }
}


class MessageRole(str, Enum):
    """Ruolo del messaggio"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageStatus(str, Enum):
    """Stato del messaggio"""
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    ERROR = "error"


# ============ MODELS ============

class ChatMessage(BaseModel):
    """Singolo messaggio chat"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    status: MessageStatus = MessageStatus.DELIVERED
    mode: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatSession(BaseModel):
    """Sessione di chat"""
    id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: List[ChatMessage] = Field(default_factory=list)
    mode: OperatingLevel = OperatingLevel.COPILOT
    context: Dict[str, Any] = Field(default_factory=dict)
    turn_count: int = 0
    is_active: bool = True


class SendMessageRequest(BaseModel):
    """Richiesta invio messaggio"""
    message: str
    session_id: Optional[str] = None
    mode: Optional[str] = "copilot"
    include_context: bool = True
    context_limit: int = 10


class SendMessageResponse(BaseModel):
    """Risposta a messaggio"""
    message_id: str
    session_id: str
    response: str
    mode: str
    confidence: float
    reasoning_steps: List[str] = []
    actions_taken: List[str] = []
    follow_up: Optional[str] = None
    timestamp: datetime


class ChatHistoryRequest(BaseModel):
    """Richiesta storico"""
    session_id: str
    limit: int = 50
    offset: int = 0
    include_metadata: bool = True


# ============ SESSION MANAGER ============

class ChatSessionManager:
    """
    Gestisce le sessioni di chat con storico completo
    e supporto multi-turno
    """
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.context_window = 10  # Numero di messaggi per contesto
    
    def create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Crea nuova sessione"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        session = ChatSession(id=session_id)
        
        # Messaggio di benvenuto
        welcome = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="👋 Ciao! Sono GIDEON, il tuo assistente AI. Come posso aiutarti?",
            mode="copilot",
            confidence=1.0
        )
        session.messages.append(welcome)
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Ottiene sessione esistente o ne crea una nuova"""
        if session_id not in self.sessions:
            return self.create_session(session_id)
        return self.sessions[session_id]
    
    def add_message(
        self, 
        session_id: str, 
        role: MessageRole, 
        content: str,
        mode: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> ChatMessage:
        """Aggiunge messaggio alla sessione"""
        session = self.get_session(session_id)
        
        message = ChatMessage(
            role=role,
            content=content,
            mode=mode,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        session.messages.append(message)
        session.updated_at = datetime.now()
        
        if role == MessageRole.USER:
            session.turn_count += 1
        
        return message
    
    def get_context(self, session_id: str, limit: int = None) -> List[Dict]:
        """Ottiene contesto conversazione per multi-turno"""
        session = self.get_session(session_id)
        limit = limit or self.context_window
        
        recent_messages = session.messages[-limit:]
        
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in recent_messages
        ]
    
    def get_history(
        self, 
        session_id: str, 
        limit: int = 50, 
        offset: int = 0,
        include_metadata: bool = True
    ) -> Dict:
        """Ottiene storico completo"""
        session = self.get_session(session_id)
        
        messages = session.messages[offset:offset + limit]
        
        history = []
        for msg in messages:
            item = {
                "id": msg.id,
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "status": msg.status.value
            }
            
            if msg.mode:
                item["mode"] = msg.mode
            if msg.confidence is not None:
                item["confidence"] = msg.confidence
            if include_metadata and msg.metadata:
                item["metadata"] = msg.metadata
            
            history.append(item)
        
        return {
            "session_id": session_id,
            "messages": history,
            "total_count": len(session.messages),
            "turn_count": session.turn_count,
            "current_mode": session.mode.value,
            "created_at": session.created_at.isoformat(),
            "is_active": session.is_active
        }
    
    def set_mode(self, session_id: str, mode: str) -> OperatingLevel:
        """Imposta modalità operativa"""
        session = self.get_session(session_id)
        
        try:
            session.mode = OperatingLevel(mode)
        except ValueError:
            session.mode = OperatingLevel.COPILOT
        
        # Aggiunge messaggio di sistema
        self.add_message(
            session_id,
            MessageRole.SYSTEM,
            f"Modalità cambiata a {session.mode.value.upper()}",
            mode=session.mode.value
        )
        
        return session.mode
    
    def clear_session(self, session_id: str) -> bool:
        """Cancella sessione"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Statistiche globali"""
        total_messages = sum(len(s.messages) for s in self.sessions.values())
        total_turns = sum(s.turn_count for s in self.sessions.values())
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": sum(1 for s in self.sessions.values() if s.is_active),
            "total_messages": total_messages,
            "total_turns": total_turns,
            "avg_messages_per_session": total_messages / max(len(self.sessions), 1)
        }
    
    # WebSocket management
    async def connect(self, websocket: WebSocket, session_id: str):
        """Connette WebSocket a sessione"""
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
        
        # Invia stato iniziale
        session = self.get_session(session_id)
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "mode": session.mode.value,
            "history_count": len(session.messages)
        })
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Disconnette WebSocket"""
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
    
    async def broadcast(self, session_id: str, message: Dict):
        """Broadcast a tutti i client della sessione"""
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass


# ============ GLOBAL INSTANCE ============

chat_manager = ChatSessionManager()


# ============ RESPONSE GENERATOR ============

class ResponseGenerator:
    """Genera risposte intelligenti basate su contesto usando AI"""
    
    def __init__(self):
        self.mode_prefixes = {
            "passive": "📖",
            "copilot": "🤖",
            "pilot": "🚀",
            "analyzing": "🔍"
        }
    
    async def generate(
        self,
        message: str,
        context: List[Dict],
        mode: str,
        response_mode: str = "fast",
        mode_config: Dict = None
    ) -> Dict:
        """
        Genera risposta usando l'assistant AI (OpenRouter/OpenAI/etc)
        
        Args:
            message: Messaggio utente
            context: Contesto conversazione
            mode: Modalità operativa (passive/copilot/pilot)
            response_mode: Modalità risposta (eco/fast/deep)
            mode_config: Configurazione modalità con max_tokens, temperature, etc.
        """
        global _chat_assistant_instance
        
        # Default config se non fornito
        if mode_config is None:
            mode_config = RESPONSE_MODE_CONFIG.get(response_mode, RESPONSE_MODE_CONFIG["fast"])
        
        logger.info(f"💬 Chat generate [{response_mode}]: {message[:50]}...")
        
        # Try to use the AI assistant if available
        if _chat_assistant_instance and _chat_assistant_instance.is_ready():
            try:
                logger.info(f"🤖 Using AI assistant [{response_mode}] (max_tokens={mode_config['max_tokens']}, temp={mode_config['temperature']})...")
                result = await _chat_assistant_instance.process_command(
                    text=message,
                    mode="text",
                    max_tokens=mode_config.get("max_tokens", 300),
                    temperature=mode_config.get("temperature", 0.5)
                )
                
                response_text = result.get("text", result.get("response", ""))
                confidence = result.get("confidence", 0.9)
                source = result.get("data", {}).get("source", "ai")
                
                # Ensure reasoning_steps is always a list
                raw_steps = result.get("data", {}).get("reasoning_steps", ["AI response generated"])
                if isinstance(raw_steps, list):
                    reasoning_steps = raw_steps
                elif isinstance(raw_steps, int):
                    reasoning_steps = [f"Completed {raw_steps} reasoning steps"]
                else:
                    reasoning_steps = [str(raw_steps)] if raw_steps else ["AI response generated"]
                
                logger.info(f"✅ AI response generated ({source}): {len(response_text)} chars")
                
                return {
                    "response": response_text,
                    "confidence": confidence,
                    "reasoning_steps": reasoning_steps,
                    "actions": result.get("data", {}).get("actions", []),
                    "follow_up": None,
                    "source": source
                }
            except Exception as e:
                logger.warning(f"AI assistant error: {e}, falling back to local response")
        
        # Fallback to simple local response
        logger.info("📝 Using local fallback response")
        normalized_mode = (mode or "copilot").lower()
        prefix = self.mode_prefixes.get(normalized_mode, "🤖")
        lower_msg = message.lower()
        
        reasoning_steps = []
        actions = []
        confidence = 0.7
        follow_up = None
        
        # Simple pattern matching for basic responses
        if any(word in lower_msg for word in ["ciao", "salve", "buongiorno", "hey"]):
            response = f"{prefix} Ciao! Come posso aiutarti oggi?"
            confidence = 0.95
        elif "ore" in lower_msg or "tempo" in lower_msg:
            now = datetime.now()
            response = f"{prefix} Sono le **{now.hour}:{now.minute:02d}**."
            confidence = 1.0
        elif any(word in lower_msg for word in ["aiuto", "cosa puoi", "funzionalità"]):
            response = f"""{prefix} Posso aiutarti con:

• **Informazioni**: rispondere a domande
• **Azioni**: aprire programmi, file, siti web  
• **Automazioni**: eseguire workflow e macro
• **Analisi**: analizzare dati e fornire insights

    La mia autonomia dipende dalla modalità: **{normalized_mode.upper()}**."""
            confidence = 0.95
        elif any(word in lower_msg for word in ["apri", "avvia", "lancia", "esegui", "start"]):
            app_name = self._extract_app_name(message)
            actions = [f"launch_app:{app_name.lower()}"]
            if normalized_mode == "pilot":
                response = f"{prefix} Richiesta eseguita: ho aperto {app_name} in modalità PILOT."
                confidence = 0.92
                reasoning_steps = [
                    "Modalità PILOT → esecuzione diretta",
                    f"Azione completata: apertura di {app_name}"
                ]
            elif normalized_mode == "copilot":
                response = f"{prefix} Vuoi che apra {app_name}? Dimmi solo una conferma e procedo subito."
                follow_up = f"Posso aprire {app_name} adesso, confermi?"
                confidence = 0.82
                reasoning_steps = [
                    "Modalità COPILOT → richiesta conferma",
                    f"Azione proposta: apertura di {app_name}"
                ]
            else:
                response = f"{prefix} In modalità passiva passo la richiesta su {app_name} senza eseguirla. Cambia modalità per procedere."
                confidence = 0.7
                reasoning_steps = [
                    "Modalità PASSIVE → nessuna esecuzione",
                    f"Richiesta passata per {app_name}"
                ]
        else:
            response = f"{prefix} Ricevuto il tuo messaggio. Per risposte più intelligenti, assicurati che l'AI sia configurata correttamente."
            confidence = 0.5
            reasoning_steps = ["AI non disponibile", "Risposta locale generata"]
        
        return {
            "response": response,
            "confidence": confidence,
            "reasoning_steps": reasoning_steps,
            "actions": actions,
            "follow_up": follow_up,
            "source": "local_fallback"
        }
    
    def _analyze_context(self, context: List[Dict]) -> Dict:
        """Analizza contesto per multi-turno"""
        summary = {
            "has_previous_question": False,
            "last_topic": None,
            "user_messages_count": 0
        }
        
        for msg in context:
            if msg.get("role") == "user":
                summary["user_messages_count"] += 1
                summary["last_topic"] = msg.get("content", "")[:50]
                if "?" in msg.get("content", ""):
                    summary["has_previous_question"] = True
        
        return summary
    
    def _extract_app_name(self, message: str) -> str:
        """Estrae nome applicazione dal messaggio"""
        apps = ["Chrome", "Firefox", "Word", "Excel", "PowerPoint", 
                "Notepad", "Calculator", "Explorer", "Terminal", "VS Code"]
        
        for app in apps:
            if app.lower() in message.lower():
                return app
        
        # Prova a estrarre dopo "apri"
        words = message.split()
        for i, word in enumerate(words):
            if word.lower() in ["apri", "avvia", "lancia"] and i + 1 < len(words):
                return words[i + 1].capitalize()
        
        return "l'applicazione"


response_generator = ResponseGenerator()


# ============ REST ENDPOINTS ============

@router.post("/send", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    """
    Invia messaggio e ricevi risposta
    
    Supporta:
    - Multi-turno con contesto
    - Diversi livelli operativi
    - Storico conversazione
    """
    # Ottieni o crea sessione
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    session = chat_manager.get_session(session_id)
    
    # Imposta modalità
    if request.mode:
        chat_manager.set_mode(session_id, request.mode)
    
    # Aggiungi messaggio utente
    user_msg = chat_manager.add_message(
        session_id,
        MessageRole.USER,
        request.message
    )
    
    # Ottieni contesto
    context = []
    if request.include_context:
        context = chat_manager.get_context(session_id, request.context_limit)
    
    # Genera risposta
    result = await response_generator.generate(
        request.message,
        context,
        request.mode or "copilot"
    )
    
    # Aggiungi risposta
    assistant_msg = chat_manager.add_message(
        session_id,
        MessageRole.ASSISTANT,
        result["response"],
        mode=request.mode,
        confidence=result["confidence"],
        metadata={
            "reasoning_steps": result["reasoning_steps"],
            "actions": result["actions"]
        }
    )
    
    # Broadcast WebSocket
    await chat_manager.broadcast(session_id, {
        "type": "new_message",
        "message": {
            "id": assistant_msg.id,
            "role": "assistant",
            "content": result["response"],
            "mode": request.mode,
            "confidence": result["confidence"]
        }
    })
    
    return SendMessageResponse(
        message_id=assistant_msg.id,
        session_id=session_id,
        response=result["response"],
        mode=request.mode or "copilot",
        confidence=result["confidence"],
        reasoning_steps=result["reasoning_steps"],
        actions_taken=result["actions"],
        follow_up=result["follow_up"],
        timestamp=datetime.now()
    )


# ============ FAST COMBINED ENDPOINT ============

class SendWithVoiceRequest(BaseModel):
    """Richiesta chat + voice combinata per zero lag"""
    message: str
    session_id: Optional[str] = None
    mode: Optional[str] = "copilot"
    response_mode: Optional[str] = "fast"  # eco, fast, deep
    include_voice: bool = True


class SendWithVoiceResponse(BaseModel):
    """Risposta con testo + audio base64"""
    message_id: str
    session_id: str
    response: str
    mode: str
    confidence: float
    audio_base64: Optional[str] = None  # Audio MP3 in base64
    audio_size: int = 0
    timestamp: datetime


# Cache dir for TTS
TTS_CACHE_DIR = Path(__file__).parent.parent / "voice" / "audio_cache"
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Gideon voice config - ULTRA FAST settings
GIDEON_VOICE = "it-IT-GiuseppeNeural"
GIDEON_RATE = "+20%"  # ⚡ Parlata veloce per latenza minima


def clean_for_tts(text: str) -> str:
    """Pulisce testo per TTS"""
    import re
    text = re.sub(r'[*_`#]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def generate_tts_audio(text: str) -> bytes:
    """Genera audio TTS e ritorna bytes"""
    if not EDGE_TTS_AVAILABLE:
        return b""
    
    clean_text = clean_for_tts(text)
    if not clean_text:
        return b""
    
    # Check cache
    cache_key = hashlib.md5(f"{clean_text}_{GIDEON_VOICE}".encode()).hexdigest()
    cache_path = TTS_CACHE_DIR / f"{cache_key}.mp3"
    
    if cache_path.exists():
        return cache_path.read_bytes()
    
    # Generate new audio with FAST rate
    try:
        communicate = Communicate(text=clean_text, voice=GIDEON_VOICE, rate=GIDEON_RATE)
        await communicate.save(str(cache_path))
        return cache_path.read_bytes()
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return b""


@router.post("/send-with-voice", response_model=SendWithVoiceResponse)
async def send_message_with_voice(request: SendWithVoiceRequest):
    """
    🚀 ENDPOINT ULTRA-VELOCE: Chat + Voice in un'unica richiesta
    
    Genera risposta AI e audio TTS, ritorna tutto insieme.
    Supporta modalità: eco (breve), fast (bilanciato), deep (approfondito)
    """
    import time
    start_time = time.time()
    
    # Ottieni configurazione modalità risposta
    response_mode = (request.response_mode or "fast").lower()
    mode_config = RESPONSE_MODE_CONFIG.get(response_mode, RESPONSE_MODE_CONFIG["fast"])
    
    logger.info(f"⚡ Response mode: {response_mode} - {mode_config['description']}")
    
    # Ottieni o crea sessione
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    session = chat_manager.get_session(session_id)
    
    # Imposta modalità
    if request.mode:
        chat_manager.set_mode(session_id, request.mode)
    
    # Aggiungi messaggio utente
    user_msg = chat_manager.add_message(
        session_id,
        MessageRole.USER,
        request.message
    )
    
    # ⚡ Ottieni contesto in base alla modalità
    context_limit = mode_config["context_limit"]
    context = chat_manager.get_context(session_id, context_limit)
    
    # === GENERA RISPOSTA AI con parametri modalità ===
    logger.info(f"⚡ {response_mode.upper()}: generando risposta AI (max_tokens={mode_config['max_tokens']})...")
    result = await response_generator.generate(
        request.message,
        context,
        request.mode or "copilot",
        response_mode=response_mode,
        mode_config=mode_config
    )
    
    ai_time = time.time() - start_time
    logger.info(f"⚡ AI ({response_mode}): {ai_time:.2f}s")
    
    # Aggiungi risposta
    assistant_msg = chat_manager.add_message(
        session_id,
        MessageRole.ASSISTANT,
        result["response"],
        mode=request.mode,
        confidence=result["confidence"]
    )
    
    # === GENERA AUDIO ===
    audio_base64 = None
    audio_size = 0
    
    if request.include_voice and EDGE_TTS_AVAILABLE:
        logger.info(f"⚡ Fast chat: generating TTS audio...")
        tts_start = time.time()
        
        audio_bytes = await generate_tts_audio(result["response"])
        
        if audio_bytes:
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_size = len(audio_bytes)
        
        tts_time = time.time() - tts_start
        logger.info(f"⚡ TTS audio in {tts_time:.2f}s ({audio_size} bytes)")
    
    total_time = time.time() - start_time
    logger.info(f"⚡ Total send-with-voice: {total_time:.2f}s")
    
    return SendWithVoiceResponse(
        message_id=assistant_msg.id,
        session_id=session_id,
        response=result["response"],
        mode=request.mode or "copilot",
        confidence=result["confidence"],
        audio_base64=audio_base64,
        audio_size=audio_size,
        timestamp=datetime.now()
    )


@router.get("/response-modes")
async def get_response_modes():
    """
    📊 Ottieni le modalità di risposta disponibili
    
    Returns:
        Lista delle modalità con descrizioni e configurazioni
    """
    return {
        "modes": [
            {
                "id": "eco",
                "name": "💚 ECO",
                "description": "Risposte brevi e veloci, minimo consumo",
                "max_tokens": 150,
                "speed": "ultra-fast",
                "detail": "low"
            },
            {
                "id": "fast", 
                "name": "⚡ FAST",
                "description": "Bilanciato tra velocità e qualità",
                "max_tokens": 300,
                "speed": "fast",
                "detail": "medium"
            },
            {
                "id": "deep",
                "name": "🧠 DEEP",
                "description": "Risposte approfondite e dettagliate",
                "max_tokens": 800,
                "speed": "normal",
                "detail": "high"
            }
        ],
        "default": "fast"
    }


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    include_metadata: bool = True
):
    """Ottiene storico completo della chat"""
    return chat_manager.get_history(session_id, limit, offset, include_metadata)


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Informazioni sulla sessione"""
    session = chat_manager.get_session(session_id)
    
    return {
        "session_id": session.id,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "message_count": len(session.messages),
        "turn_count": session.turn_count,
        "current_mode": session.mode.value,
        "is_active": session.is_active,
        "context_summary": chat_manager.get_context(session_id, 3)
    }


@router.post("/session/{session_id}/mode")
async def set_session_mode(session_id: str, mode: str):
    """Imposta modalità operativa della sessione"""
    new_mode = chat_manager.set_mode(session_id, mode)
    
    # Broadcast cambio modalità
    await chat_manager.broadcast(session_id, {
        "type": "mode_changed",
        "mode": new_mode.value
    })
    
    return {
        "session_id": session_id,
        "mode": new_mode.value,
        "description": {
            "passive": "Solo risponde, nessuna azione",
            "copilot": "Suggerisce e chiede conferma",
            "pilot": "Esegue senza conferma"
        }.get(new_mode.value, "")
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Elimina sessione"""
    success = chat_manager.clear_session(session_id)
    
    if success:
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions")
async def list_sessions(active_only: bool = True):
    """Lista tutte le sessioni"""
    sessions = []
    
    for session_id, session in chat_manager.sessions.items():
        if active_only and not session.is_active:
            continue
        
        sessions.append({
            "id": session_id,
            "created_at": session.created_at.isoformat(),
            "message_count": len(session.messages),
            "turn_count": session.turn_count,
            "mode": session.mode.value,
            "is_active": session.is_active
        })
    
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


@router.get("/stats")
async def get_chat_stats():
    """Statistiche globali chat"""
    return chat_manager.get_stats()


# ============ WEBSOCKET ============

@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket per chat in tempo reale
    
    Messaggi supportati:
    - send: invia messaggio
    - mode: cambia modalità
    - history: richiedi storico
    - typing: indicatore digitazione
    """
    await chat_manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "send":
                # Messaggio utente
                message = data.get("message", "")
                mode = data.get("mode", "copilot")
                
                # Aggiungi messaggio utente
                chat_manager.add_message(session_id, MessageRole.USER, message)
                
                # Invia stato "typing"
                await websocket.send_json({
                    "type": "typing",
                    "status": True
                })
                
                # Genera risposta
                context = chat_manager.get_context(session_id)
                result = await response_generator.generate(message, context, mode)
                
                # Invia reasoning steps
                for step in result["reasoning_steps"]:
                    await websocket.send_json({
                        "type": "reasoning_step",
                        "step": step
                    })
                    await asyncio.sleep(0.3)
                
                # Aggiungi e invia risposta
                assistant_msg = chat_manager.add_message(
                    session_id,
                    MessageRole.ASSISTANT,
                    result["response"],
                    mode=mode,
                    confidence=result["confidence"]
                )
                
                await websocket.send_json({
                    "type": "response",
                    "message_id": assistant_msg.id,
                    "text": result["response"],
                    "mode": mode,
                    "confidence": result["confidence"],
                    "actions": result["actions"],
                    "follow_up": result["follow_up"]
                })
                
                # Fine typing
                await websocket.send_json({
                    "type": "typing",
                    "status": False
                })
            
            elif msg_type == "mode":
                # Cambio modalità
                mode = data.get("mode", "copilot")
                new_mode = chat_manager.set_mode(session_id, mode)
                
                await websocket.send_json({
                    "type": "mode_changed",
                    "mode": new_mode.value
                })
            
            elif msg_type == "history":
                # Richiesta storico
                limit = data.get("limit", 50)
                history = chat_manager.get_history(session_id, limit)
                
                await websocket.send_json({
                    "type": "history",
                    **history
                })
            
            elif msg_type == "typing":
                # Broadcast typing indicator
                await chat_manager.broadcast(session_id, {
                    "type": "user_typing",
                    "status": data.get("status", False)
                })
    
    except WebSocketDisconnect:
        chat_manager.disconnect(websocket, session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        chat_manager.disconnect(websocket, session_id)
