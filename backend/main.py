"""
GIDEON 3.0 + JARVIS CORE - Main Application Entry Point
Sistema Cognitivo Autonomo con Pipeline Intelligente
"""

# Fix encoding for Windows console
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ⚠️ IMPORTANTE: Carica .env PRIMA di qualsiasi altro import
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"✅ .env loaded from {env_path}")

import asyncio
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Legacy routes
from api.routes import main_router, voice_router, analysis_router, pilot_router, memory_router, logs_router, actions_router, ai_router, social_router, set_assistant_instance

# NEW: Gideon 3.0 + Jarvis Core routes
from api.gideon_routes import gideon_router, jarvis_router, system_router, set_system_instances

# NEW: Chat routes assistant setter
from api.chat_routes import set_chat_assistant_instance

from core.config import settings
from database.database import init_db
from voice.manager import VoiceManager
# TTS ora gestito via API endpoint /api/tts/speak
from brain.assistant import GideonAssistant

# Configure logging
logger.remove()
logger.add(sys.stderr, level=settings.LOG_LEVEL)
logger.add("logs/gideon_{time}.log", rotation="500 MB", retention="10 days", level="INFO")
logger.add("logs/conversations_{time}.log", rotation="100 MB", retention="30 days", level="DEBUG", 
           filter=lambda record: "CONVERSATION" in record["message"])

# Global instances - Legacy
voice_manager: VoiceManager = None
assistant: GideonAssistant = None
active_connections: list[WebSocket] = []
current_level: str = "normal"

# Global instances - NEW: Gideon 3.0 + Jarvis Core
orchestrator = None
gideon_core = None
jarvis_core = None
automation_layer = None
mode_manager = None
emergency_system = None
action_logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global voice_manager, assistant
    global orchestrator, gideon_core, jarvis_core, mode_manager, emergency_system, action_logger
    
    # Startup
    logger.info("🚀 Starting GIDEON 3.0 + JARVIS CORE...")
    
    # Initialize database (optional)
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.warning(f"⚠️ Database init skipped: {e}")
    
    # Initialize voice system (optional)
    try:
        voice_manager = VoiceManager()
        await voice_manager.initialize()
        logger.info("✅ Voice system initialized")
    except Exception as e:
        logger.warning(f"⚠️ Voice system init skipped: {e}")
    
    # Initialize legacy assistant brain
    try:
        assistant = GideonAssistant()
        await assistant.initialize()
        set_assistant_instance(assistant)
        set_chat_assistant_instance(assistant)  # Also set for chat routes
        logger.info("✅ Legacy Assistant brain initialized")
    except Exception as e:
        logger.warning(f"⚠️ Legacy Assistant brain init skipped: {e}")
    
    # ========== NEW: Initialize GIDEON 3.0 + JARVIS CORE ==========
    try:
        # Mode Manager
        from core.mode_manager import ModeManager
        mode_manager = ModeManager()
        logger.info("✅ Mode Manager initialized")
        
        # Emergency System
        from core.emergency import EmergencySystem
        emergency_system = EmergencySystem()
        logger.info("✅ Emergency System initialized")
        
        # Action Logger
        from core.action_logger import ActionLogger
        action_logger = ActionLogger()
        logger.info("✅ Action Logger initialized")
        
        # Gideon Core (Analitico)
        from gideon import GideonCore
        gideon_core = GideonCore()
        logger.info("✅ Gideon Core initialized (Cognitive/Predictive Module)")
        
        # Automation Layer (Esecutivo)
        from automation import AutomationLayer
        automation_layer = AutomationLayer()
        logger.info("✅ Automation Layer initialized (Executive Module)")
        
        # Jarvis Executive AI
        from jarvis import JarvisSupervisor
        jarvis_core = JarvisSupervisor(
            gideon_core=gideon_core,
            automation_layer=automation_layer,
            mode_manager=mode_manager
        )
        logger.info("✅ Jarvis Executive AI initialized (understand → decide → orchestrate → execute)")
        
        # Orchestrator
        from core.orchestrator import Orchestrator
        orchestrator = Orchestrator(
            gideon_core=gideon_core,
            jarvis_core=jarvis_core,
            automation_layer=automation_layer,
            mode_manager=mode_manager,
            emergency_system=emergency_system,
            action_logger=action_logger
        )
        logger.info("✅ Orchestrator initialized (Pipeline Coordinator)")
        
        # Set instances in routes
        set_system_instances(
            orchestrator=orchestrator,
            gideon=gideon_core,
            jarvis=jarvis_core,
            automation=automation_layer,
            mode_manager=mode_manager,
            emergency=emergency_system
        )
        logger.info("✅ System instances linked to API routes")
        
    except Exception as e:
        logger.error(f"❌ GIDEON 3.0 init error: {e}")
        import traceback
        traceback.print_exc()
    
    # Initialize Smart Actions
    try:
        from automation.smart_actions import smart_actions
        smart_actions.initialize(
            openrouter_key=settings.OPENROUTER_API_KEY,
            openai_key=getattr(settings, 'OPENAI_API_KEY', None)
        )
        logger.info("✅ Smart Actions initialized (Timer, WhatsApp, Email, Vision)")
    except Exception as e:
        logger.warning(f"⚠️ Smart Actions init error: {e}")
    
    logger.info("🎉 GIDEON 3.0 System is ready!")
    logger.info(f"📡 API running on http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📚 Docs available at http://{settings.HOST}:{settings.PORT}/api/docs")
    logger.info("")
    logger.info("🏗️  Architecture:")
    logger.info("    JARVIS (Executive AI) → understand, decide, orchestrate, execute")
    logger.info("    ├── GIDEON (Cognitive) → analysis, predictions, simulations")
    logger.info("    └── AUTOMATION (Executive) → actions, workflows, macros")
    logger.info("")
    logger.info(f"🧠 Gideon API: /api/gideon/*")
    logger.info(f"⚡ Jarvis API: /api/jarvis/*")
    logger.info(f"🎛️ System API: /api/system/*")
    logger.info(f"🤖 Smart Actions API: /api/smart/*")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down GIDEON 3.0...")
    
    # Stop Smart Actions
    try:
        from automation.smart_actions import smart_actions
        smart_actions.shutdown()
    except:
        pass
    
    if voice_manager:
        try:
            await voice_manager.shutdown()
        except:
            pass
    
    if assistant:
        try:
            await assistant.shutdown()
        except:
            pass
    
    logger.info("👋 GIDEON 3.0 stopped")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="GIDEON 3.0 + JARVIS CORE API",
    description="Sistema Cognitivo Autonomo - Analytical + Executive Modules",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    logger.info(f"✅ Frontend mounted from {frontend_path}")
else:
    logger.warning(f"⚠️ Frontend path not found: {frontend_path}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import psutil
    
    return {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu": psutil.cpu_percent(interval=0.1),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent
        },
        "services": {
            "legacy_brain": assistant.is_ready() if assistant else False,
            "voice": voice_manager is not None,
            "gideon_core": gideon_core is not None,
            "jarvis_core": jarvis_core is not None,
            "orchestrator": orchestrator is not None,
            "mode_manager": mode_manager is not None,
            "emergency": emergency_system is not None
        },
        "mode": mode_manager.get_mode_info() if mode_manager else None
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "GIDEON 3.0 + JARVIS CORE",
        "version": "3.0.0",
        "status": "active",
        "architecture": {
            "gideon": "Analytical Module (predict, analyze, simulate)",
            "jarvis": "Executive Module (decide, execute, control)"
        },
        "endpoints": {
            "docs": "/api/docs",
            "frontend": "/static/gideon_unified.html",
            "chat": "/static/chat.html",
            "health": "/health",
            "websocket": "/ws",
            "gideon_api": "/api/gideon",
            "jarvis_api": "/api/jarvis",
            "system_api": "/api/system"
        }
    }


# Include legacy routers
app.include_router(main_router, prefix="/api")
app.include_router(voice_router, prefix="/api/voice")
app.include_router(analysis_router, prefix="/api/analysis")
app.include_router(pilot_router, prefix="/api/pilot")
app.include_router(memory_router, prefix="/api/memory")
app.include_router(logs_router, prefix="/api/logs")
app.include_router(actions_router, prefix="/api/actions")
app.include_router(ai_router, prefix="/api/ai")
app.include_router(social_router, prefix="/api/social")

# NEW: Include GIDEON 3.0 + JARVIS CORE routers
app.include_router(gideon_router, prefix="/api")
app.include_router(jarvis_router, prefix="/api")
app.include_router(system_router, prefix="/api")

# NEW: GIDEON 4.0 UNIFIED routes (Ultimate Single Entry Point)
try:
    from api.gideon_v4_routes import router as gideon_v4_router
    app.include_router(gideon_v4_router, prefix="/api")
    logger.info("✅ GIDEON 4.0 routes loaded (Ultimate Unified AI)")
except ImportError as e:
    logger.warning(f"⚠️ GIDEON 4.0 routes not loaded: {e}")

# GIDEON UNIFIED routes (Legacy - Assistente Unificato)
try:
    from api.gideon_unified_routes import router as gideon_unified_router
    app.include_router(gideon_unified_router, prefix="/api")
    logger.info("✅ GIDEON Unified routes loaded (Assistente Unificato)")
except ImportError as e:
    logger.warning(f"⚠️ GIDEON Unified routes not loaded: {e}")

# TTS Routes - Voce di Gideon
try:
    from api.tts_routes import router as tts_router
    app.include_router(tts_router)
    logger.info("✅ TTS routes loaded (Voce Gideon)")
except ImportError as e:
    logger.warning(f"⚠️ TTS routes not loaded: {e}")

# NEW: Chat & Voice routes (WhatsApp-style)
try:
    from api.chat_routes import router as chat_router
    app.include_router(chat_router, prefix="/api")
    logger.info("✅ Chat routes loaded")
except ImportError as e:
    logger.warning(f"⚠️ Chat routes not loaded: {e}")

try:
    from api.voice_routes import router as voice_routes_v2
    app.include_router(voice_routes_v2, prefix="/api")
    logger.info("✅ Voice routes v2 loaded")
except ImportError as e:
    logger.warning(f"⚠️ Voice routes v2 not loaded: {e}")

# NEW: Security routes (Kill-switch, Rollback, Permissions)
try:
    from api.security_routes import router as security_router
    app.include_router(security_router, prefix="/api")
    logger.info("✅ Security routes loaded")
except ImportError as e:
    logger.warning(f"⚠️ Security routes not loaded: {e}")

# Dev Automations routes (standalone)
try:
    from api.dev_routes import router as dev_router
    app.include_router(dev_router, prefix="/api")
    logger.info("✅ Dev automation routes loaded")
except ImportError as e:
    logger.warning(f"⚠️ Dev routes not loaded: {e}")

# NEW: Smart Actions routes (Timer, WhatsApp, Email, Vision)
try:
    from api.smart_routes import router as smart_router
    from automation.smart_actions import smart_actions
    app.include_router(smart_router)
    logger.info("✅ Smart Actions routes loaded (Timer, WhatsApp, Email, Vision)")
except ImportError as e:
    logger.warning(f"⚠️ Smart Actions routes not loaded: {e}")

# NEW: GitHub Integration
try:
    from api.github_routes import router as github_router
    app.include_router(github_router, prefix="/api")
    logger.info("✅ GitHub routes loaded")
except ImportError as e:
    logger.warning(f"⚠️ GitHub routes not loaded: {e}")

# NEW: AI Hub + Workflow Engine routes
try:
    from api.hub_routes import router as hub_router
    app.include_router(hub_router, prefix="/api")
    logger.info("✅ AI Hub + Workflow routes loaded (Multi-AI, Automazioni)")
except ImportError as e:
    logger.warning(f"⚠️ AI Hub routes not loaded: {e}")

# NEW: CAM - Crisis Automation Mode routes
try:
    from api.cam_routes import router as cam_router
    app.include_router(cam_router, prefix="/api")
    logger.info("✅ CAM Crisis Automation Mode routes loaded")
except ImportError as e:
    logger.warning(f"⚠️ CAM routes not loaded: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    global current_level
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"🔌 New WebSocket connection. Total: {len(active_connections)}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")
            payload = data.get("payload", {})
            
            logger.debug(f"📨 Received: {message_type}")
            
            # NEW: Handle cognitive pipeline requests
            if message_type == "cognitive_process":
                user_text = payload.get("text")
                context = payload.get("context", {})
                
                logger.info(f"CONVERSATION [COGNITIVE] USER: {user_text}")
                
                if orchestrator:
                    result = await orchestrator.process(user_text, context)
                    logger.info(f"CONVERSATION [COGNITIVE] GIDEON: {result.get('response', '')[:200]}")
                    
                    await websocket.send_json({
                        "type": "cognitive_result",
                        "payload": result
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "payload": {"error": "Orchestrator not initialized"}
                    })
            
            # NEW: Handle mode changes
            elif message_type == "set_mode":
                mode_name = payload.get("mode")
                if mode_manager:
                    from core.mode_manager import OperatingMode
                    mode_map = {
                        "passive": OperatingMode.PASSIVE,
                        "copilot": OperatingMode.COPILOT,
                        "pilot": OperatingMode.PILOT,
                        "executive": OperatingMode.EXECUTIVE
                    }
                    if mode_name in mode_map:
                        mode_manager.set_mode(mode_map[mode_name])
                        await websocket.send_json({
                            "type": "mode_changed",
                            "payload": mode_manager.get_mode_info()
                        })
            
            # Legacy handlers
            elif message_type == "voice_command":
                user_text = payload.get("text")
                # Log conversation for traceability
                logger.info(f"CONVERSATION [VOICE] [LEVEL:{current_level}] USER: {user_text}")
                
                # Process voice command
                result = await assistant.process_command(
                    text=user_text,
                    mode="voice"
                )
                
                # Log response
                response_text = result.get('text', result.get('response', ''))[:200]
                logger.info(f"CONVERSATION [VOICE] [LEVEL:{current_level}] GIDEON: {response_text}")
                
                result['level'] = current_level
                await websocket.send_json({
                    "type": "command_result",
                    "payload": result
                })
            
            elif message_type == "text_message":
                user_text = payload.get("text")
                # Log conversation for traceability
                logger.info(f"CONVERSATION [TEXT] [LEVEL:{current_level}] USER: {user_text}")
                
                # Process text message
                result = await assistant.process_command(
                    text=user_text,
                    mode="text"
                )
                
                # Log response
                response_text = result.get('text', result.get('response', ''))[:200]
                logger.info(f"CONVERSATION [TEXT] [LEVEL:{current_level}] GIDEON: {response_text}")
                
                result['level'] = current_level
                await websocket.send_json({
                    "type": "message_result",
                    "payload": result
                })
            
            elif message_type == "set_level":
                # Update current operation level
                current_level = payload.get("level", "normal")
                logger.info(f"CONVERSATION [LEVEL_CHANGE] New level: {current_level}")
                await websocket.send_json({
                    "type": "level_updated",
                    "payload": {"level": current_level}
                })
            
            elif message_type == "pilot_command":
                # Handle pilot mode command - full system control
                user_text = payload.get("text")
                execute = payload.get("execute", True)
                
                logger.info(f"CONVERSATION [PILOT] USER: {user_text}")
                
                # Process as pilot command with enhanced capabilities
                result = await assistant.process_command(
                    text=user_text,
                    mode="pilot",
                    pilot_execute=execute
                )
                
                response_text = result.get('text', result.get('response', ''))[:200]
                logger.info(f"CONVERSATION [PILOT] GIDEON: {response_text}")
                
                result['level'] = 'pilot'
                result['pilot_mode'] = True
                
                await websocket.send_json({
                    "type": "pilot_result",
                    "payload": result
                })
            
            elif message_type == "analysis_request":
                logger.info(f"CONVERSATION [ANALYSIS] [LEVEL:{current_level}] Target: {payload.get('target')}")
                # Trigger analysis
                analysis_result = await assistant.analyze_system(
                    target=payload.get("target")
                )
                logger.info(f"CONVERSATION [ANALYSIS] [LEVEL:{current_level}] Result: Analysis completed")
                await websocket.send_json({
                    "type": "analysis_result",
                    "payload": analysis_result
                })
            
            elif message_type == "ping":
                # Keepalive
                await websocket.send_json({"type": "pong", "payload": {"timestamp": payload.get("timestamp", 0)}})
            
            elif message_type == "handshake":
                # Client initial connection
                session_id = payload.get("sessionId", "unknown")
                mode = payload.get("mode", "copilot")
                logger.info(f"🤝 WebSocket handshake from session: {session_id}, mode: {mode}")
                await websocket.send_json({
                    "type": "handshake_ack",
                    "payload": {
                        "status": "connected",
                        "session_id": session_id,
                        "server_version": "3.0.0",
                        "features": ["chat", "voice", "cognitive", "actions", "modes"]
                    }
                })
            
            elif message_type == "message_received":
                # Acknowledgment from client (no response needed)
                pass
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Total: {len(active_connections)}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


async def broadcast_event(event_type: str, data: dict):
    """Broadcast event to all connected clients"""
    message = {
        "type": event_type,
        "payload": data
    }
    
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.error(f"Failed to broadcast to connection: {e}")


# ============ ENTRY POINT ============
if __name__ == "__main__":
    logger.info("🚀 Starting GIDEON Backend Server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
