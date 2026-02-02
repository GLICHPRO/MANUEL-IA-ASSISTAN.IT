"""
🤖 GIDEON Unified API Routes

Endpoint unificati per l'assistente GIDEON
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from loguru import logger
import asyncio

# Import GIDEON
from core.gideon_unified import get_gideon, init_gideon

router = APIRouter(prefix="/gideon-unified", tags=["GIDEON Unified"])


# ========== REQUEST MODELS ==========

class ProcessRequest(BaseModel):
    text: str
    context: Optional[Dict] = None
    mode: Optional[str] = None  # passive/copilot/pilot
    include_voice: bool = False
    response_mode: str = "fast"  # eco/fast/deep


class VisionRequest(BaseModel):
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    question: str = "Analizza questa immagine in dettaglio"


class SpeakRequest(BaseModel):
    text: str


class ExecuteRequest(BaseModel):
    action: str
    params: Optional[Dict] = {}


class ModeRequest(BaseModel):
    mode: str  # passive/copilot/pilot/executive


# ========== ENDPOINTS ==========

@router.post("/process")
async def process(request: ProcessRequest):
    """
    🎯 Pipeline principale GIDEON
    
    Processa qualsiasi richiesta: domande, comandi, analisi.
    """
    try:
        gideon = get_gideon()
        if not gideon.is_initialized:
            await gideon.initialize()
        
        result = await gideon.process(
            text=request.text,
            context=request.context,
            mode=request.mode,
            include_voice=request.include_voice,
            response_mode=request.response_mode
        )
        
        return result
        
    except Exception as e:
        logger.error(f"GIDEON process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick")
async def quick(request: ProcessRequest):
    """
    ⚡ Risposta rapida per comandi semplici
    """
    try:
        gideon = get_gideon()
        if not gideon.is_initialized:
            await gideon.initialize()
        
        result = await gideon.quick(request.text)
        return result
        
    except Exception as e:
        logger.error(f"GIDEON quick error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze(request: ProcessRequest):
    """
    🔍 Analisi approfondita
    """
    try:
        gideon = get_gideon()
        if not gideon.is_initialized:
            await gideon.initialize()
        
        result = await gideon.analyze(request.text, request.context)
        return result
        
    except Exception as e:
        logger.error(f"GIDEON analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision")
async def vision(request: VisionRequest):
    """
    👁️ Analisi immagini con AI Vision
    """
    try:
        gideon = get_gideon()
        if not gideon.is_initialized:
            await gideon.initialize()
        
        result = await gideon.vision(
            image_base64=request.image_base64,
            image_path=request.image_path,
            question=request.question
        )
        
        return result
        
    except Exception as e:
        logger.error(f"GIDEON vision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speak")
async def speak(request: SpeakRequest):
    """
    🔊 Genera audio TTS
    """
    try:
        gideon = get_gideon()
        if not gideon.is_initialized:
            await gideon.initialize()
        
        audio_base64 = await gideon.speak(request.text)
        
        return {
            "success": True,
            "audio_base64": audio_base64
        }
        
    except Exception as e:
        logger.error(f"GIDEON speak error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute(request: ExecuteRequest):
    """
    ⚡ Esegue un'azione
    """
    try:
        gideon = get_gideon()
        if not gideon.is_initialized:
            await gideon.initialize()
        
        result = await gideon.execute(request.action, request.params)
        return result
        
    except Exception as e:
        logger.error(f"GIDEON execute error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mode")
async def set_mode(request: ModeRequest):
    """
    🎛️ Imposta modalità operativa
    """
    try:
        gideon = get_gideon()
        gideon.set_mode(request.mode)
        
        return {
            "success": True,
            "mode": gideon.mode.value
        }
        
    except Exception as e:
        logger.error(f"GIDEON mode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def status():
    """
    📊 Stato GIDEON
    """
    try:
        gideon = get_gideon()
        return gideon.get_status()
        
    except Exception as e:
        logger.error(f"GIDEON status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """
    ❤️ Health check GIDEON
    """
    gideon = get_gideon()
    
    return {
        "status": "healthy" if gideon.is_initialized else "initializing",
        "name": "GIDEON",
        "version": "3.0",
        "mode": gideon.mode.value,
        "initialized": gideon.is_initialized
    }
