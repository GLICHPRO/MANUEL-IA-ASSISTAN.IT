"""
🚀 GIDEON AI Hub + Workflow API Routes

Endpoint per:
- AI Hub (tutti i servizi AI in uno)
- Workflow Engine (automazioni n8n style)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from loguru import logger

router = APIRouter(prefix="/hub", tags=["AI Hub & Workflows"])


# ============================================================
#                      REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str
    provider: str = "auto"
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7


class TTSRequest(BaseModel):
    text: str
    provider: str = "auto"
    voice: Optional[str] = None
    model: Optional[str] = None


class ImageRequest(BaseModel):
    prompt: str
    provider: str = "auto"
    model: Optional[str] = None
    size: str = "1024x1024"
    style: Optional[str] = None


class VideoRequest(BaseModel):
    prompt: Optional[str] = None
    image_base64: Optional[str] = None
    avatar_id: Optional[str] = None
    script: Optional[str] = None
    provider: str = "auto"


# New Video AI Request Models
class VeoVideoRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    model: str = "veo-3.1-generate-preview"
    aspect_ratio: str = "16:9"
    resolution: str = "720p"


class SoraVideoRequest(BaseModel):
    prompt: str
    duration: int = 5
    model: str = "sora-2"
    aspect_ratio: str = "16:9"


class SeedreamImageRequest(BaseModel):
    prompt: str
    model: str = "bytedance/seedream-3.0"
    aspect_ratio: str = "1:1"
    num_outputs: int = 1


class SeedVRVideoRequest(BaseModel):
    prompt: str
    model: str = "seedvr-360"
    resolution: str = "4k"


class LumaVideoRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    model: str = "dream-machine"


class PikaVideoRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    model: str = "pika-1.0"


class MusicRequest(BaseModel):
    prompt: str
    style: Optional[str] = None
    duration: int = 30


class PresentationRequest(BaseModel):
    topic: str
    slides: int = 10
    style: str = "professional"


class PodcastRequest(BaseModel):
    content: str
    title: str = "GIDEON Podcast"
    hosts: int = 2
    duration_minutes: int = 5


class WorkflowCreateRequest(BaseModel):
    name: str
    description: str = ""
    nodes: Optional[List[Dict]] = None


class WorkflowExecuteRequest(BaseModel):
    input_data: Optional[Dict] = None


class NodeCreateRequest(BaseModel):
    node_type: str
    name: str
    config: Optional[Dict] = None
    position: Optional[Dict] = None


class NodeConnectRequest(BaseModel):
    source_id: str
    target_id: str


# ============================================================
#                      AI HUB ENDPOINTS
# ============================================================

@router.get("/services")
async def list_services():
    """
    📦 Lista tutti i servizi AI disponibili
    """
    from integrations.ai_hub import get_ai_hub
    
    hub = get_ai_hub()
    return {
        "services": hub.get_available_services(),
        "stats": hub.get_stats()
    }


@router.post("/chat")
async def ai_chat(request: ChatRequest):
    """
    💬 Chat con AI (multi-provider)
    
    Providers: openrouter, openai, anthropic, google
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.chat(
            message=request.message,
            provider=request.provider,
            model=request.model,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return result
        
    except Exception as e:
        logger.error(f"AI Hub chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts")
async def ai_tts(request: TTSRequest):
    """
    🎤 Text-to-Speech (multi-provider)
    
    Providers: elevenlabs, openai_tts, edge_tts
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.text_to_speech(
            text=request.text,
            provider=request.provider,
            voice=request.voice,
            model=request.model
        )
        
        return result
        
    except Exception as e:
        logger.error(f"AI Hub TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image")
async def ai_image(request: ImageRequest):
    """
    🎨 Genera immagini con AI
    
    Providers: dalle, stability, replicate
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_image(
            prompt=request.prompt,
            provider=request.provider,
            model=request.model,
            size=request.size,
            style=request.style
        )
        
        return result
        
    except Exception as e:
        logger.error(f"AI Hub image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video")
async def ai_video(request: VideoRequest):
    """
    🎬 Genera video con AI
    
    Providers: heygen, runway, kling
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_video(
            prompt=request.prompt,
            image=request.image_base64,
            avatar_id=request.avatar_id,
            script=request.script,
            provider=request.provider
        )
        
        return result
        
    except Exception as e:
        logger.error(f"AI Hub video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
#                NEW VIDEO AI ENDPOINTS
# ============================================================

@router.post("/video/veo")
async def ai_video_veo(request: VeoVideoRequest):
    """
    📹 Google Veo 3.1 Video Generation
    
    Models: veo-3.1-generate-preview, veo-3.1-fast-preview, veo-2
    Aspect Ratios: 16:9 (landscape), 9:16 (portrait)
    Resolutions: 720p, 1080p, 4k
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_video_veo(
            prompt=request.prompt,
            image=request.image,
            model=request.model,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Veo video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/sora")
async def ai_video_sora(request: SoraVideoRequest):
    """
    🎥 OpenAI Sora 2 Video Generation
    
    Models: sora-2, sora-turbo
    Duration: 5-60 seconds
    Note: API may not be publicly available yet
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_video_sora(
            prompt=request.prompt,
            duration=request.duration,
            model=request.model,
            aspect_ratio=request.aspect_ratio
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Sora video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image/seedream")
async def ai_image_seedream(request: SeedreamImageRequest):
    """
    🌸 ByteDance Seedream 3.0 Image Generation
    
    Uses Replicate API
    Aspect Ratios: 1:1, 16:9, 9:16, 4:3, 3:4
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_image_seedream(
            prompt=request.prompt,
            model=request.model,
            aspect_ratio=request.aspect_ratio,
            num_outputs=request.num_outputs
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Seedream image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/seedvr")
async def ai_video_seedvr(request: SeedVRVideoRequest):
    """
    🌐 SeedVR 360°/VR Video Generation
    
    Uses Replicate API
    Formats: 360, VR
    Resolutions: 2k, 4k
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_video_seedvr(
            prompt=request.prompt,
            model=request.model,
            resolution=request.resolution
        )
        
        return result
        
    except Exception as e:
        logger.error(f"SeedVR video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/luma")
async def ai_video_luma(request: LumaVideoRequest):
    """
    ✨ Luma AI Dream Machine Video Generation
    
    Models: dream-machine, luma-video
    Supports text-to-video and image-to-video
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_video_luma(
            prompt=request.prompt,
            image=request.image,
            model=request.model
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Luma video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/pika")
async def ai_video_pika(request: PikaVideoRequest):
    """
    🎞️ Pika Labs Video Generation
    
    Models: pika-1.0, pika-turbo
    Supports text-to-video and image-to-video
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_video_pika(
            prompt=request.prompt,
            image=request.image,
            model=request.model
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Pika video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/music")
async def ai_music(request: MusicRequest):
    """
    🎵 Genera musica con AI
    
    Providers: suno (coming soon)
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_music(
            prompt=request.prompt,
            style=request.style,
            duration=request.duration
        )
        
        return result
        
    except Exception as e:
        logger.error(f"AI Hub music error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/presentation")
async def ai_presentation(request: PresentationRequest):
    """
    📊 Genera presentazioni (Gamma AI style)
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_presentation(
            topic=request.topic,
            slides=request.slides,
            style=request.style
        )
        
        return result
        
    except Exception as e:
        logger.error(f"AI Hub presentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/podcast")
async def ai_podcast(request: PodcastRequest):
    """
    🎙️ Genera podcast da contenuti (NotebookLM style)
    """
    from integrations.ai_hub import get_ai_hub
    
    try:
        hub = get_ai_hub()
        await hub.initialize()
        
        result = await hub.generate_podcast(
            content=request.content,
            title=request.title,
            hosts=request.hosts,
            duration_minutes=request.duration_minutes
        )
        
        return result
        
    except Exception as e:
        logger.error(f"AI Hub podcast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
#                    WORKFLOW ENDPOINTS
# ============================================================

@router.get("/workflows")
async def list_workflows():
    """
    📋 Lista tutti i workflow
    """
    from core.workflow_engine import get_workflow_engine
    
    engine = get_workflow_engine()
    return {
        "workflows": engine.list_workflows()
    }


@router.post("/workflows")
async def create_workflow(request: WorkflowCreateRequest):
    """
    ➕ Crea un nuovo workflow
    """
    from core.workflow_engine import get_workflow_engine
    
    try:
        engine = get_workflow_engine()
        workflow = engine.create_workflow(
            name=request.name,
            description=request.description,
            nodes=request.nodes
        )
        
        return {
            "success": True,
            "workflow_id": workflow.id,
            "name": workflow.name
        }
        
    except Exception as e:
        logger.error(f"Create workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """
    📄 Ottieni dettagli workflow
    """
    from core.workflow_engine import get_workflow_engine
    
    engine = get_workflow_engine()
    workflow = engine.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow non trovato")
    
    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "enabled": workflow.enabled,
        "nodes": {
            node_id: {
                "id": node.id,
                "type": node.type.value,
                "name": node.name,
                "config": node.config,
                "next_nodes": node.next_nodes,
                "position": node.position
            }
            for node_id, node in workflow.nodes.items()
        },
        "variables": workflow.variables,
        "last_run": workflow.last_run.isoformat() if workflow.last_run else None,
        "run_count": workflow.run_count
    }


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    🗑️ Elimina un workflow
    """
    from core.workflow_engine import get_workflow_engine
    
    engine = get_workflow_engine()
    success = engine.delete_workflow(workflow_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Workflow non trovato")
    
    return {"success": True}


@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: WorkflowExecuteRequest):
    """
    ▶️ Esegui un workflow
    """
    from core.workflow_engine import get_workflow_engine
    
    try:
        engine = get_workflow_engine()
        execution = await engine.execute_workflow(
            workflow_id=workflow_id,
            input_data=request.input_data
        )
        
        return {
            "execution_id": execution.id,
            "status": execution.status,
            "context": execution.context,
            "logs": execution.logs,
            "error": execution.error
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Execute workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows/{workflow_id}/nodes")
async def add_node(workflow_id: str, request: NodeCreateRequest):
    """
    ➕ Aggiungi nodo a workflow
    """
    from core.workflow_engine import get_workflow_engine
    
    engine = get_workflow_engine()
    node = engine.add_node(
        workflow_id=workflow_id,
        node_type=request.node_type,
        name=request.name,
        config=request.config,
        position=request.position
    )
    
    if not node:
        raise HTTPException(status_code=404, detail="Workflow non trovato")
    
    return {
        "success": True,
        "node_id": node.id
    }


@router.post("/workflows/{workflow_id}/connect")
async def connect_nodes(workflow_id: str, request: NodeConnectRequest):
    """
    🔗 Collega due nodi
    """
    from core.workflow_engine import get_workflow_engine
    
    engine = get_workflow_engine()
    success = engine.connect_nodes(
        workflow_id=workflow_id,
        source_id=request.source_id,
        target_id=request.target_id
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Impossibile collegare i nodi")
    
    return {"success": True}


@router.post("/workflows/presets/{preset_name}")
async def create_preset_workflow(preset_name: str):
    """
    📦 Crea workflow da preset
    
    Presets disponibili:
    - daily_summary: Riassunto giornaliero
    - voice_assistant: Assistente vocale
    - content_creator: Creatore contenuti
    - email_responder: Risponditore email
    """
    from core.workflow_engine import get_workflow_engine
    
    engine = get_workflow_engine()
    workflow = engine.create_preset_workflow(preset_name)
    
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' non trovato")
    
    return {
        "success": True,
        "workflow_id": workflow.id,
        "name": workflow.name
    }


@router.get("/workflows/presets")
async def list_presets():
    """
    📋 Lista preset disponibili
    """
    return {
        "presets": [
            {
                "name": "daily_summary",
                "description": "Genera un riassunto giornaliero alle 9:00",
                "nodes": 4
            },
            {
                "name": "voice_assistant",
                "description": "Risponde a comandi vocali",
                "nodes": 3
            },
            {
                "name": "content_creator",
                "description": "Crea contenuti multimediali da un topic",
                "nodes": 5
            },
            {
                "name": "email_responder",
                "description": "Risponde automaticamente alle email",
                "nodes": 3
            }
        ]
    }


# ============================================================
#                    QUICK ACTIONS
# ============================================================

@router.post("/quick/summarize")
async def quick_summarize(text: str, max_length: int = 200):
    """
    📝 Riassumi testo rapidamente
    """
    from integrations.ai_hub import get_ai_hub
    
    hub = get_ai_hub()
    await hub.initialize()
    
    result = await hub.chat(
        message=f"Riassumi questo testo in massimo {max_length} caratteri:\n\n{text}",
        max_tokens=300
    )
    
    return result


@router.post("/quick/translate")
async def quick_translate(text: str, target_language: str = "en"):
    """
    🌍 Traduci testo rapidamente
    """
    from integrations.ai_hub import get_ai_hub
    
    hub = get_ai_hub()
    await hub.initialize()
    
    result = await hub.chat(
        message=f"Traduci in {target_language}:\n\n{text}",
        max_tokens=500
    )
    
    return result


@router.post("/quick/code")
async def quick_code(task: str, language: str = "python"):
    """
    💻 Genera codice rapidamente
    """
    from integrations.ai_hub import get_ai_hub
    
    hub = get_ai_hub()
    await hub.initialize()
    
    result = await hub.chat(
        message=f"Scrivi codice {language} per: {task}",
        system_prompt="Sei un esperto programmatore. Rispondi solo con codice funzionante e commenti.",
        max_tokens=1000
    )
    
    return result


@router.post("/quick/explain")
async def quick_explain(topic: str, level: str = "simple"):
    """
    📚 Spiega un argomento
    """
    from integrations.ai_hub import get_ai_hub
    
    hub = get_ai_hub()
    await hub.initialize()
    
    level_prompt = {
        "simple": "Spiega come se parlassi a un bambino di 10 anni",
        "medium": "Spiega in modo chiaro e accessibile",
        "expert": "Spiega in modo tecnico e dettagliato"
    }.get(level, "Spiega in modo chiaro")
    
    result = await hub.chat(
        message=f"{level_prompt}: {topic}",
        max_tokens=800
    )
    
    return result
