"""
🤖 GIDEON 4.0 - Unified API Routes

Single endpoint that connects all Gideon versions (2.0, 3.0, 4.0)
providing a unified interface for all capabilities.

Endpoints:
    POST /gideon          - Main unified endpoint
    POST /gideon/process  - Full processing pipeline
    POST /gideon/quick    - Quick responses
    POST /gideon/analyze  - Deep analysis
    POST /gideon/predict  - Predictions
    POST /gideon/simulate - Simulations
    POST /gideon/execute  - Action execution
    POST /gideon/vision   - Image analysis
    POST /gideon/speak    - Text-to-Speech
    POST /gideon/recommend - Full recommendation
    GET  /gideon/status   - System status
    GET  /gideon/health   - Health check
    POST /gideon/mode     - Set operating mode
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from loguru import logger
import asyncio
import traceback

# Import Gideon 4.0
from core.gideon_v4 import (
    Gideon4, GideonConfig, GideonMode, GideonLevel, ResponseMode,
    get_gideon_v4, init_gideon_v4, GideonResult
)

router = APIRouter(prefix="/gideon4", tags=["GIDEON 4.0 Unified"])


# ═══════════════════════════════════════════════════════════════════════════════
#                            REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedRequest(BaseModel):
    """Richiesta unificata - gestisce tutti i tipi di input"""
    text: str = Field(..., description="Testo da processare")
    context: Optional[Dict] = Field(None, description="Contesto aggiuntivo")
    mode: Optional[str] = Field(None, description="Modalità: passive/copilot/pilot/executive")
    level: Optional[str] = Field(None, description="Livello: quick/normal/advanced/expert")
    response_mode: Optional[str] = Field("fast", description="eco/fast/deep")
    include_voice: bool = Field(False, description="Includi audio TTS")
    capability: Optional[str] = Field(None, description="Forza capability specifica")


class AnalyzeRequest(BaseModel):
    """Richiesta analisi"""
    data: Any = Field(..., description="Dati da analizzare (testo o dict)")
    context: Optional[Dict] = None


class PredictRequest(BaseModel):
    """Richiesta previsione"""
    context: Dict = Field(..., description="Contesto per la previsione")
    horizon: str = Field("short", description="short/medium/long")


class SimulateRequest(BaseModel):
    """Richiesta simulazione"""
    scenario: Dict = Field(..., description="Scenario da simulare")
    method: str = Field("monte_carlo", description="monte_carlo/bayesian/what_if")
    iterations: int = Field(1000, description="Numero iterazioni (per Monte Carlo)")


class ExecuteRequest(BaseModel):
    """Richiesta esecuzione"""
    action: str = Field(..., description="Azione da eseguire")
    params: Optional[Dict] = Field(default_factory=dict)


class VisionRequest(BaseModel):
    """Richiesta analisi immagine"""
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    question: str = Field("Analizza questa immagine in dettaglio")


class SpeakRequest(BaseModel):
    """Richiesta TTS"""
    text: str


class ModeRequest(BaseModel):
    """Cambio modalità"""
    mode: str = Field(..., description="passive/copilot/pilot/executive")
    level: Optional[str] = Field(None, description="quick/normal/advanced/expert")


class RecommendRequest(BaseModel):
    """Richiesta raccomandazione"""
    query: str = Field(..., description="Query per la raccomandazione")
    context: Optional[Dict] = None


# ═══════════════════════════════════════════════════════════════════════════════
#                            HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def get_initialized_gideon() -> Gideon4:
    """Get initialized Gideon 4.0 instance"""
    gideon = get_gideon_v4()
    if not gideon.is_initialized:
        await gideon.initialize()
    return gideon


def result_to_response(result: GideonResult) -> Dict[str, Any]:
    """Convert GideonResult to API response"""
    return result.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
#                            MAIN UNIFIED ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("")
@router.post("/")
async def unified_endpoint(request: UnifiedRequest):
    """
    🎯 GIDEON 4.0 - Main Unified Endpoint
    
    Single entry point for all Gideon capabilities.
    Routes automatically based on input or explicit capability.
    
    Capabilities:
    - chat: Conversazione normale
    - analyze: Analisi approfondita
    - predict: Previsioni
    - simulate: Simulazioni
    - execute: Esecuzione azioni
    - vision: Analisi immagini
    - recommend: Raccomandazioni complete
    """
    try:
        gideon = await get_initialized_gideon()
        
        # Route to specific capability if specified
        if request.capability:
            capability = request.capability.lower()
            
            if capability == "analyze":
                result = await gideon.analyze(request.text, request.context)
            elif capability == "predict":
                result = await gideon.predict(request.context or {"query": request.text})
            elif capability == "recommend":
                result = await gideon.recommend(request.text, request.context)
            else:
                # Default to process
                result = await gideon.process(
                    text=request.text,
                    context=request.context,
                    mode=request.mode,
                    level=request.level,
                    include_voice=request.include_voice,
                    response_mode=request.response_mode
                )
        else:
            # Standard processing
            result = await gideon.process(
                text=request.text,
                context=request.context,
                mode=request.mode,
                level=request.level,
                include_voice=request.include_voice,
                response_mode=request.response_mode
            )
        
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON unified error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#                         SPECIFIC ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/process")
async def process(request: UnifiedRequest):
    """
    🎯 Full processing pipeline
    
    Handles any type of request through the main GIDEON pipeline.
    """
    try:
        gideon = await get_initialized_gideon()
        
        result = await gideon.process(
            text=request.text,
            context=request.context,
            mode=request.mode,
            level=request.level,
            include_voice=request.include_voice,
            response_mode=request.response_mode
        )
        
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick")
async def quick(request: UnifiedRequest):
    """
    ⚡ Quick response for simple commands
    
    Uses pattern matching for fast local responses.
    """
    try:
        gideon = await get_initialized_gideon()
        
        # Try quick response first
        result = await gideon._try_quick_response(request.text)
        
        if result is None:
            # Fallback to fast process
            result = await gideon.process(
                text=request.text,
                context=request.context,
                response_mode="eco"
            )
        
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON quick error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    🔍 Deep analysis
    
    Uses Cognitive v3.0 for detailed analysis.
    """
    try:
        gideon = await get_initialized_gideon()
        
        result = await gideon.analyze(request.data, request.context)
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict(request: PredictRequest):
    """
    🔮 Generate predictions
    
    Uses Cognitive v3.0 Predictor.
    """
    try:
        gideon = await get_initialized_gideon()
        
        result = await gideon.predict(request.context, request.horizon)
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate")
async def simulate(request: SimulateRequest):
    """
    🎲 Run simulation
    
    Supports monte_carlo, bayesian, what_if.
    """
    try:
        gideon = await get_initialized_gideon()
        
        result = await gideon.simulate(
            scenario=request.scenario,
            method=request.method,
            iterations=request.iterations
        )
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON simulate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute(request: ExecuteRequest):
    """
    ⚡ Execute action
    
    Uses Automation Layer and Executive module.
    """
    try:
        gideon = await get_initialized_gideon()
        
        result = await gideon.execute(request.action, request.params)
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON execute error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision")
async def vision(request: VisionRequest):
    """
    👁️ Analyze image
    
    Uses Vision AI for image analysis.
    """
    try:
        gideon = await get_initialized_gideon()
        
        result = await gideon.vision(
            image_base64=request.image_base64,
            image_path=request.image_path,
            question=request.question
        )
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON vision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speak")
async def speak(request: SpeakRequest):
    """
    🔊 Text-to-Speech
    
    Generate audio from text.
    """
    try:
        gideon = await get_initialized_gideon()
        
        result = await gideon.speak(request.text)
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON speak error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend")
async def recommend(request: RecommendRequest):
    """
    💡 Get full recommendation
    
    Pipeline: analyze → predict → simulate → rank → suggest
    """
    try:
        gideon = await get_initialized_gideon()
        
        result = await gideon.recommend(request.query, request.context)
        return result_to_response(result)
        
    except Exception as e:
        logger.error(f"GIDEON recommend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURATION & STATUS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/mode")
async def set_mode(request: ModeRequest):
    """
    🎛️ Set operating mode
    
    Modes:
    - passive: Analysis only
    - copilot: Suggest + confirm
    - pilot: Autonomous
    - executive: Full orchestration
    """
    try:
        gideon = await get_initialized_gideon()
        
        gideon.set_mode(request.mode)
        if request.level:
            gideon.set_level(request.level)
        
        return {
            "success": True,
            "mode": gideon.config.mode.value,
            "level": gideon.config.level.value
        }
        
    except Exception as e:
        logger.error(f"GIDEON mode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """
    📊 Get system status
    
    Returns complete status of Gideon 4.0.
    """
    try:
        gideon = get_gideon_v4()
        
        if not gideon.is_initialized:
            return {
                "initialized": False,
                "version": Gideon4.VERSION,
                "message": "GIDEON 4.0 not initialized. Call any endpoint to auto-initialize."
            }
        
        return gideon.get_status()
        
    except Exception as e:
        logger.error(f"GIDEON status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    ❤️ Health check
    
    Quick health check for monitoring.
    """
    gideon = get_gideon_v4()
    
    return {
        "status": "healthy" if gideon.is_initialized else "not_initialized",
        "version": Gideon4.VERSION,
        "initialized": gideon.is_initialized,
        "capabilities": gideon.get_capabilities() if gideon.is_initialized else []
    }


@router.get("/capabilities")
async def get_capabilities():
    """
    🔧 Get available capabilities
    
    Returns list of active capabilities.
    """
    gideon = get_gideon_v4()
    
    return {
        "version": Gideon4.VERSION,
        "capabilities": gideon.get_capabilities() if gideon.is_initialized else [],
        "help": gideon.get_help() if gideon.is_initialized else "Initialize GIDEON first"
    }


@router.get("/help")
async def get_help():
    """
    ❓ Get help
    
    Returns usage information.
    """
    try:
        gideon = await get_initialized_gideon()
        
        return {
            "version": Gideon4.VERSION,
            "help": gideon.get_help(),
            "endpoints": [
                {"path": "/gideon", "method": "POST", "description": "Main unified endpoint"},
                {"path": "/gideon/process", "method": "POST", "description": "Full pipeline"},
                {"path": "/gideon/quick", "method": "POST", "description": "Quick responses"},
                {"path": "/gideon/analyze", "method": "POST", "description": "Deep analysis"},
                {"path": "/gideon/predict", "method": "POST", "description": "Predictions"},
                {"path": "/gideon/simulate", "method": "POST", "description": "Simulations"},
                {"path": "/gideon/execute", "method": "POST", "description": "Actions"},
                {"path": "/gideon/vision", "method": "POST", "description": "Image analysis"},
                {"path": "/gideon/speak", "method": "POST", "description": "Text-to-Speech"},
                {"path": "/gideon/recommend", "method": "POST", "description": "Recommendations"},
                {"path": "/gideon/status", "method": "GET", "description": "System status"},
                {"path": "/gideon/health", "method": "GET", "description": "Health check"},
                {"path": "/gideon/mode", "method": "POST", "description": "Set mode"}
            ]
        }
        
    except Exception as e:
        logger.error(f"GIDEON help error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#                              WEBSOCKET
# ═══════════════════════════════════════════════════════════════════════════════

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    🔌 WebSocket for real-time communication
    
    Send JSON messages with same format as unified endpoint.
    """
    await websocket.accept()
    
    try:
        gideon = await get_initialized_gideon()
        
        await websocket.send_json({
            "type": "connected",
            "version": Gideon4.VERSION,
            "message": "GIDEON 4.0 WebSocket connected"
        })
        
        while True:
            data = await websocket.receive_json()
            
            try:
                result = await gideon.process(
                    text=data.get("text", ""),
                    context=data.get("context"),
                    mode=data.get("mode"),
                    level=data.get("level"),
                    include_voice=data.get("include_voice", False),
                    response_mode=data.get("response_mode", "fast")
                )
                
                await websocket.send_json({
                    "type": "response",
                    **result_to_response(result)
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info("GIDEON WebSocket disconnected")
    except Exception as e:
        logger.error(f"GIDEON WebSocket error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#                         LEGACY COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

# These routes maintain compatibility with older API calls

@router.post("/chat")
async def chat_legacy(request: UnifiedRequest):
    """Legacy chat endpoint - redirects to unified"""
    return await unified_endpoint(request)


@router.post("/ask")
async def ask_legacy(request: UnifiedRequest):
    """Legacy ask endpoint - redirects to unified"""
    return await unified_endpoint(request)
