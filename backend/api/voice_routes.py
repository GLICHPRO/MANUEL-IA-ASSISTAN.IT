"""
🗣️ Voice & Chat API Routes - GIDEON 3.0

Endpoints per:
- Interazione vocale naturale
- Chat testuale
- Sintesi vocale (TTS)
- Riconoscimento vocale (STT)
- Gestione conversazione
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from enum import Enum
import asyncio
import json
import logging
from datetime import datetime

# Import voice system
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from voice.natural_voice import (
    NaturalVoiceEngine,
    ResponseComposer,
    ConversationManager,
    VoiceEmotion,
    VoiceProfile,
    ReasoningContext,
    ComposedResponse,
    SpeechOutput
)

# Logger
logger = logging.getLogger("voice_api")

# Router
router = APIRouter(prefix="/api/voice", tags=["voice"])

# Global instances
_voice_engine: Optional[NaturalVoiceEngine] = None
_composer: Optional[ResponseComposer] = None
_conversation: Optional[ConversationManager] = None
_active_conversations: Dict[str, ConversationManager] = {}


def get_voice_engine() -> NaturalVoiceEngine:
    """Get or create voice engine"""
    global _voice_engine
    if _voice_engine is None:
        _voice_engine = NaturalVoiceEngine()
    return _voice_engine


def get_composer() -> ResponseComposer:
    """Get or create response composer"""
    global _composer
    if _composer is None:
        _composer = ResponseComposer(get_voice_engine())
    return _composer


def get_conversation(session_id: str = "default") -> ConversationManager:
    """Get or create conversation manager for session"""
    global _active_conversations
    if session_id not in _active_conversations:
        _active_conversations[session_id] = ConversationManager(get_composer())
    return _active_conversations[session_id]


# === MODELS ===

class TTSRequest(BaseModel):
    """Richiesta sintesi vocale"""
    text: str
    emotion: Optional[str] = None
    rate: Optional[float] = 1.0
    include_ssml: bool = False


class ChatRequest(BaseModel):
    """Richiesta chat"""
    message: str
    session_id: str = "default"
    include_voice: bool = True
    reasoning_steps: Optional[List[str]] = None
    confidence: Optional[float] = None


class ReasoningRequest(BaseModel):
    """Richiesta con contesto ragionamento"""
    query: str
    response_text: str
    reasoning_steps: List[str] = []
    confidence: float = 0.8
    action_taken: Optional[str] = None
    include_reasoning_summary: bool = False
    session_id: str = "default"


class QuickResponseRequest(BaseModel):
    """Risposta rapida senza ragionamento"""
    text: str
    emotion: Optional[str] = None
    session_id: str = "default"


class EmotionRequest(BaseModel):
    """Richiesta con emozione specifica"""
    emotion: str


# === REST ENDPOINTS ===

@router.get("/status")
async def get_voice_status():
    """Get voice system status"""
    engine = get_voice_engine()
    conv = get_conversation()
    
    return {
        "voice_engine_ready": engine is not None,
        "active_sessions": len(_active_conversations),
        "default_session_stats": conv.get_stats()
    }


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Converte testo in output vocale strutturato.
    
    Genera segmenti con pause, intonazioni e modificatori
    per sintesi vocale naturale.
    """
    engine = get_voice_engine()
    
    # Parse emotion
    emotion = None
    if request.emotion:
        try:
            emotion = VoiceEmotion(request.emotion)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid emotion: {request.emotion}. Valid: {[e.value for e in VoiceEmotion]}"
            )
    
    # Process text
    output = engine.process_text(request.text, emotion)
    
    result = output.to_dict()
    
    # Add SSML if requested
    if request.include_ssml:
        result["ssml"] = output.get_ssml()
    
    return JSONResponse(content=result)


@router.post("/tts/ssml")
async def get_ssml(request: TTSRequest):
    """Genera solo SSML dal testo"""
    engine = get_voice_engine()
    
    emotion = None
    if request.emotion:
        try:
            emotion = VoiceEmotion(request.emotion)
        except ValueError:
            pass
    
    output = engine.process_text(request.text, emotion)
    
    return {
        "ssml": output.get_ssml(),
        "duration_estimate": output.total_duration_estimate
    }


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Endpoint principale chat.
    
    Gestisce input utente e genera risposta completa
    dopo eventuale ragionamento.
    """
    conv = get_conversation(request.session_id)
    
    # Add user input
    conv.add_user_input(request.message)
    
    # If reasoning steps provided, use them
    if request.reasoning_steps:
        ctx = conv.start_reasoning(request.message)
        for step in request.reasoning_steps:
            conv.add_reasoning_step(step)
        
        # Complete with generic response (actual response should come from AI)
        response = conv.complete_reasoning(
            request.message,  # Placeholder - real implementation gets AI response
            confidence=request.confidence or 0.8,
            include_reasoning_summary=len(request.reasoning_steps) > 2
        )
    else:
        # Quick response
        response = conv.quick_response(request.message, VoiceEmotion.FRIENDLY)
    
    result = response.to_dict()
    
    # Add conversation context
    result["session_id"] = request.session_id
    result["conversation_stats"] = conv.get_stats()
    
    return JSONResponse(content=result)


@router.post("/respond")
async def compose_response(request: ReasoningRequest):
    """
    Compone risposta finale dopo ragionamento completo.
    
    Questo è l'endpoint principale per generare
    la risposta unica finale dopo che il ragionamento
    è stato completato.
    """
    conv = get_conversation(request.session_id)
    
    # Build reasoning context
    ctx = ReasoningContext(
        query=request.query,
        reasoning_steps=request.reasoning_steps,
        confidence=request.confidence,
        action_taken=request.action_taken
    )
    
    # Set as current reasoning
    conv.current_reasoning = ctx
    
    # Complete and compose
    response = conv.complete_reasoning(
        request.response_text,
        confidence=request.confidence,
        action_taken=request.action_taken,
        include_reasoning_summary=request.include_reasoning_summary
    )
    
    return JSONResponse(content=response.to_dict())


@router.post("/quick")
async def quick_response(request: QuickResponseRequest):
    """
    Risposta rapida senza ragionamento.
    
    Per risposte semplici che non richiedono
    elaborazione complessa.
    """
    conv = get_conversation(request.session_id)
    
    emotion = None
    if request.emotion:
        try:
            emotion = VoiceEmotion(request.emotion)
        except ValueError:
            emotion = VoiceEmotion.FRIENDLY
    
    response = conv.quick_response(request.text, emotion)
    
    return JSONResponse(content=response.to_dict())


@router.get("/emotions")
async def list_emotions():
    """Lista emozioni vocali disponibili"""
    return {
        "emotions": [
            {
                "value": e.value,
                "description": {
                    "neutral": "Tono neutro, informativo",
                    "friendly": "Tono amichevole, cordiale",
                    "serious": "Tono serio, formale",
                    "excited": "Tono entusiasta, energico",
                    "concerned": "Tono preoccupato, attento",
                    "confident": "Tono sicuro, deciso",
                    "curious": "Tono curioso, interrogativo",
                    "calm": "Tono calmo, rilassato",
                    "urgent": "Tono urgente, pressante"
                }.get(e.value, "")
            }
            for e in VoiceEmotion
        ]
    }


@router.get("/conversation/{session_id}")
async def get_conversation_info(session_id: str):
    """Ottieni info conversazione"""
    conv = get_conversation(session_id)
    
    return {
        "session_id": session_id,
        "stats": conv.get_stats(),
        "context_summary": conv.get_context_summary(),
        "last_user_query": conv.get_last_user_query(),
        "history_length": len(conv.history)
    }


@router.get("/conversation/{session_id}/history")
async def get_conversation_history(session_id: str, limit: int = 10):
    """Ottieni storia conversazione"""
    conv = get_conversation(session_id)
    
    history = conv.history[-limit:] if limit else conv.history
    
    return {
        "session_id": session_id,
        "turns": [
            {
                "role": t.role,
                "text": t.text,
                "timestamp": t.timestamp.isoformat(),
                "emotion": t.emotion.value if t.emotion else None,
                "confidence": t.confidence
            }
            for t in history
        ]
    }


@router.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Pulisce storia conversazione"""
    conv = get_conversation(session_id)
    conv.clear_history()
    
    return {"success": True, "session_id": session_id, "message": "Conversation cleared"}


@router.post("/conversation/{session_id}/start-reasoning")
async def start_reasoning(session_id: str, query: str):
    """Inizia fase di ragionamento"""
    conv = get_conversation(session_id)
    ctx = conv.start_reasoning(query)
    
    return {
        "session_id": session_id,
        "reasoning_started": True,
        "query": query,
        "is_processing": conv.is_processing
    }


@router.post("/conversation/{session_id}/add-step")
async def add_reasoning_step(session_id: str, step: str):
    """Aggiunge step al ragionamento corrente"""
    conv = get_conversation(session_id)
    
    if not conv.is_processing:
        raise HTTPException(
            status_code=400,
            detail="No active reasoning. Call start-reasoning first."
        )
    
    conv.add_reasoning_step(step)
    
    return {
        "session_id": session_id,
        "step_added": step,
        "total_steps": len(conv.current_reasoning.reasoning_steps) if conv.current_reasoning else 0
    }


# === WEBSOCKET FOR REAL-TIME VOICE ===

@router.websocket("/ws/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket per interazione vocale real-time.
    
    Comandi supportati:
    - user_input: Input testuale utente
    - start_reasoning: Inizia ragionamento
    - add_step: Aggiunge step ragionamento
    - complete: Completa e genera risposta
    - quick: Risposta rapida
    - get_history: Ottieni storia
    - clear: Pulisci conversazione
    """
    await websocket.accept()
    conv = get_conversation(session_id)
    
    logger.info(f"Voice WebSocket connected: {session_id}")
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "stats": conv.get_stats()
        })
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
                response = await handle_voice_command(data, conv, session_id)
                await websocket.send_json(response)
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")
        await websocket.close(code=1011)


async def handle_voice_command(data: Dict, conv: ConversationManager, session_id: str) -> Dict:
    """Gestisce comando WebSocket"""
    cmd = data.get("command", "")
    
    try:
        if cmd == "user_input":
            text = data.get("text", "")
            conv.add_user_input(text)
            return {
                "type": "input_received",
                "text": text,
                "stats": conv.get_stats()
            }
        
        elif cmd == "start_reasoning":
            query = data.get("query", "")
            conv.start_reasoning(query)
            return {
                "type": "reasoning_started",
                "query": query,
                "is_processing": True
            }
        
        elif cmd == "add_step":
            step = data.get("step", "")
            conv.add_reasoning_step(step)
            return {
                "type": "step_added",
                "step": step,
                "total_steps": len(conv.current_reasoning.reasoning_steps) if conv.current_reasoning else 0
            }
        
        elif cmd == "complete":
            response_text = data.get("response", "")
            confidence = data.get("confidence", 0.8)
            action = data.get("action_taken")
            include_summary = data.get("include_reasoning", False)
            
            response = conv.complete_reasoning(
                response_text, confidence, action, include_summary
            )
            
            return {
                "type": "response_ready",
                "response": response.to_dict()
            }
        
        elif cmd == "quick":
            text = data.get("text", "")
            emotion_str = data.get("emotion")
            emotion = VoiceEmotion(emotion_str) if emotion_str else None
            
            response = conv.quick_response(text, emotion)
            
            return {
                "type": "response_ready",
                "response": response.to_dict()
            }
        
        elif cmd == "get_history":
            limit = data.get("limit", 10)
            history = conv.history[-limit:]
            
            return {
                "type": "history",
                "turns": [
                    {
                        "role": t.role,
                        "text": t.text,
                        "emotion": t.emotion.value if t.emotion else None
                    }
                    for t in history
                ]
            }
        
        elif cmd == "clear":
            conv.clear_history()
            return {
                "type": "cleared",
                "session_id": session_id
            }
        
        elif cmd == "get_stats":
            return {
                "type": "stats",
                "stats": conv.get_stats()
            }
        
        else:
            return {
                "type": "error",
                "error": f"Unknown command: {cmd}"
            }
    
    except Exception as e:
        logger.error(f"Error handling voice command: {e}")
        return {
            "type": "error",
            "error": str(e)
        }


# === HELPER ENDPOINTS ===

@router.get("/profile")
async def get_voice_profile():
    """Ottieni profilo vocale corrente"""
    engine = get_voice_engine()
    profile = engine.profile
    
    return {
        "name": profile.name,
        "base_rate": profile.base_rate,
        "base_pitch": profile.base_pitch,
        "base_volume": profile.base_volume,
        "pause_multiplier": profile.pause_multiplier,
        "emotion_settings": profile.emotion_settings
    }


@router.post("/profile")
async def update_voice_profile(
    rate: Optional[int] = None,
    pitch: Optional[int] = None,
    volume: Optional[float] = None,
    pause_multiplier: Optional[float] = None
):
    """Aggiorna profilo vocale"""
    engine = get_voice_engine()
    
    if rate is not None:
        engine.profile.base_rate = max(100, min(300, rate))
    if pitch is not None:
        engine.profile.base_pitch = max(-12, min(12, pitch))
    if volume is not None:
        engine.profile.base_volume = max(0.1, min(1.0, volume))
    if pause_multiplier is not None:
        engine.profile.pause_multiplier = max(0.5, min(2.0, pause_multiplier))
    
    return {
        "success": True,
        "profile": {
            "rate": engine.profile.base_rate,
            "pitch": engine.profile.base_pitch,
            "volume": engine.profile.base_volume,
            "pause_multiplier": engine.profile.pause_multiplier
        }
    }


@router.get("/sessions")
async def list_sessions():
    """Lista sessioni attive"""
    return {
        "sessions": [
            {
                "id": sid,
                "stats": conv.get_stats()
            }
            for sid, conv in _active_conversations.items()
        ],
        "total": len(_active_conversations)
    }
