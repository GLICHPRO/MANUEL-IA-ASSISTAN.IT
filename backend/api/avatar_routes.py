"""
🎭 Avatar API Routes - Endpoints per controllo Avatar 3D

Endpoints per:
- Stato avatar in tempo reale
- Controllo espressioni
- Lip sync con audio
- Animazioni predefinite
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

# Import avatar controller
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.avatar_controller import (
    AvatarController, 
    AvatarState, 
    Expression, 
    GazeTarget,
    AvatarAnimations,
    Viseme,
    AvatarFeedbackSystem,
    OperatingModeColors,
    HUDIndicator,
    CalculationAnimation
)

# Logger
logger = logging.getLogger("avatar_api")

# Router
router = APIRouter(prefix="/api/avatar", tags=["avatar"])

# Global avatar controller instance
_avatar_controller: Optional[AvatarController] = None
_feedback_system: Optional[AvatarFeedbackSystem] = None
_connected_clients: List[WebSocket] = []
_update_task: Optional[asyncio.Task] = None


def get_avatar_controller() -> AvatarController:
    """Get or create avatar controller"""
    global _avatar_controller
    if _avatar_controller is None:
        _avatar_controller = AvatarController()
    return _avatar_controller


def get_feedback_system() -> AvatarFeedbackSystem:
    """Get or create feedback system"""
    global _feedback_system
    if _feedback_system is None:
        _feedback_system = AvatarFeedbackSystem(get_avatar_controller())
    return _feedback_system


# === MODELS ===

class ExpressionRequest(BaseModel):
    expression: str
    intensity: float = 1.0


class StateRequest(BaseModel):
    state: str


class GazeRequest(BaseModel):
    target: str  # "user", "left", "right", "up", "down", "random"
    offset_x: float = 0.0
    offset_y: float = 0.0


class HeadPoseRequest(BaseModel):
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


class LipSyncRequest(BaseModel):
    text: str
    duration: Optional[float] = None


class LipSyncFramesRequest(BaseModel):
    frames: List[Dict]  # [{t: float, v: string, i: float}, ...]


class AnimationRequest(BaseModel):
    animation: str  # "nod", "shake", "thinking", "alert"


class OperatingModeRequest(BaseModel):
    mode: str  # "passive", "copilot", "pilot", "executive"


class ActivityRequest(BaseModel):
    activity: str  # "idle", "analyzing", "processing", "executing", "alert"
    progress: float = 0.0


class HUDUpdateRequest(BaseModel):
    indicator_id: str
    value: Optional[float] = None
    color: Optional[str] = None
    visible: Optional[bool] = None


class ConfidenceRequest(BaseModel):
    confidence: float


class DataPointRequest(BaseModel):
    text: str


# === REST ENDPOINTS ===

@router.get("/status")
async def get_avatar_status():
    """Get current avatar status"""
    controller = get_avatar_controller()
    return JSONResponse(content=controller.get_status())


@router.get("/state")
async def get_avatar_state():
    """Get full render state for 3D avatar"""
    controller = get_avatar_controller()
    return JSONResponse(content=controller.get_render_state())


@router.post("/expression")
async def set_expression(request: ExpressionRequest):
    """Set avatar expression"""
    controller = get_avatar_controller()
    
    try:
        expression = Expression(request.expression)
        controller.set_expression(expression, request.intensity)
        
        # Broadcast to connected clients
        await broadcast_state()
        
        return {"success": True, "expression": request.expression}
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid expression: {request.expression}. Valid: {[e.value for e in Expression]}"
        )


@router.post("/state")
async def set_state(request: StateRequest):
    """Set avatar state"""
    controller = get_avatar_controller()
    
    try:
        state = AvatarState(request.state)
        controller.set_state(state)
        
        await broadcast_state()
        
        return {"success": True, "state": request.state}
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state: {request.state}. Valid: {[s.value for s in AvatarState]}"
        )


@router.post("/gaze")
async def set_gaze(request: GazeRequest):
    """Set gaze target"""
    controller = get_avatar_controller()
    
    try:
        target = GazeTarget(request.target)
        controller.set_gaze_target(target, (request.offset_x, request.offset_y))
        
        await broadcast_state()
        
        return {"success": True, "gaze": request.target}
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid gaze target: {request.target}"
        )


@router.post("/head")
async def set_head_pose(request: HeadPoseRequest):
    """Set head pose"""
    controller = get_avatar_controller()
    controller.set_head_pose(request.pitch, request.yaw, request.roll)
    
    await broadcast_state()
    
    return {"success": True, "head": controller.head.to_dict()}


@router.post("/look")
async def look_at(x: float, y: float):
    """Direct eye look at coordinates"""
    controller = get_avatar_controller()
    controller.look_at(x, y)
    
    await broadcast_state()
    
    return {"success": True, "looking_at": {"x": x, "y": y}}


@router.post("/lipsync/text")
async def start_lip_sync_from_text(request: LipSyncRequest):
    """Start lip sync from text"""
    controller = get_avatar_controller()
    controller.start_lip_sync(request.text, request.duration)
    
    await broadcast_state()
    
    return {
        "success": True, 
        "text": request.text,
        "estimated_duration": request.duration or len(request.text) * 0.08
    }


@router.post("/lipsync/frames")
async def load_lip_sync_frames(request: LipSyncFramesRequest):
    """Load pre-generated lip sync frames"""
    controller = get_avatar_controller()
    controller.load_lip_sync_data(request.frames)
    
    await broadcast_state()
    
    return {"success": True, "frames_loaded": len(request.frames)}


@router.post("/lipsync/stop")
async def stop_lip_sync():
    """Stop lip sync"""
    controller = get_avatar_controller()
    controller.stop_lip_sync()
    
    await broadcast_state()
    
    return {"success": True}


@router.post("/animation/{animation_name}")
async def play_animation(animation_name: str):
    """Play predefined animation"""
    animations = {
        "nod": AvatarAnimations.nod,
        "shake": AvatarAnimations.shake_head,
        "thinking": AvatarAnimations.thinking,
        "alert": AvatarAnimations.alert
    }
    
    if animation_name not in animations:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid animation: {animation_name}. Valid: {list(animations.keys())}"
        )
    
    # Get animation
    animation = animations[animation_name]()
    
    return {
        "success": True,
        "animation": animation.to_dict()
    }


@router.get("/expressions")
async def list_expressions():
    """List available expressions"""
    return {
        "expressions": [e.value for e in Expression]
    }


@router.get("/states")
async def list_states():
    """List available states"""
    return {
        "states": [s.value for s in AvatarState]
    }


@router.get("/visemes")
async def list_visemes():
    """List available visemes"""
    return {
        "visemes": [v.value for v in Viseme]
    }


# === FEEDBACK SYSTEM ENDPOINTS ===

@router.get("/feedback")
async def get_feedback_state():
    """Get current feedback system state with HUD indicators"""
    feedback = get_feedback_system()
    return JSONResponse(content=feedback.get_feedback_state())


@router.get("/feedback/full")
async def get_full_render_state():
    """Get complete render state (avatar + feedback) for frontend"""
    feedback = get_feedback_system()
    return JSONResponse(content=feedback.get_full_render_state())


@router.post("/feedback/mode")
async def set_operating_mode(request: OperatingModeRequest):
    """
    Set operating mode and update avatar colors/expressions.
    
    Modes:
    - passive: Blue (observation mode)
    - copilot: Cyan (assistance mode)  
    - pilot: Green (full control mode)
    - executive: Purple (supervision mode)
    """
    feedback = get_feedback_system()
    
    valid_modes = ["passive", "copilot", "pilot", "executive"]
    if request.mode.lower() not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Valid: {valid_modes}"
        )
    
    feedback.set_operating_mode(request.mode)
    await broadcast_state()
    
    return {
        "success": True,
        "mode": request.mode,
        "glow_color": feedback.avatar.hologram.glow_color
    }


@router.post("/feedback/activity")
async def set_activity_state(request: ActivityRequest):
    """
    Set activity state and update visual feedback.
    
    Activities:
    - idle: Normal state (uses mode color)
    - analyzing: Yellow glow, ring animation
    - processing: Orange glow, particle animation
    - executing: Mode color, focused expression
    - alert: Red-orange glow, flicker effect
    """
    feedback = get_feedback_system()
    
    valid_activities = ["idle", "analyzing", "processing", "executing", "alert"]
    if request.activity.lower() not in valid_activities:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid activity: {request.activity}. Valid: {valid_activities}"
        )
    
    feedback.set_activity(request.activity, request.progress)
    await broadcast_state()
    
    return {
        "success": True,
        "activity": request.activity,
        "progress": request.progress,
        "glow_color": feedback.avatar.hologram.glow_color
    }


@router.post("/feedback/progress")
async def update_progress(progress: float):
    """Update activity progress (0-1)"""
    feedback = get_feedback_system()
    feedback.update_progress(progress)
    await broadcast_state()
    
    return {"success": True, "progress": progress}


@router.post("/feedback/confidence")
async def set_confidence(request: ConfidenceRequest):
    """Update confidence indicator (affects color)"""
    feedback = get_feedback_system()
    feedback.set_confidence(request.confidence)
    await broadcast_state()
    
    return {
        "success": True, 
        "confidence": request.confidence,
        "color": feedback.hud_indicators["confidence"].color
    }


@router.post("/feedback/hud")
async def update_hud_indicator(request: HUDUpdateRequest):
    """Update specific HUD indicator"""
    feedback = get_feedback_system()
    feedback.update_hud(
        request.indicator_id, 
        request.value, 
        request.color, 
        request.visible
    )
    await broadcast_state()
    
    return {"success": True, "indicator": request.indicator_id}


@router.post("/feedback/data")
async def add_data_point(request: DataPointRequest):
    """Add data point to visual stream"""
    feedback = get_feedback_system()
    feedback.add_data_point(request.text)
    
    return {"success": True, "text": request.text}


@router.post("/feedback/cpu")
async def simulate_cpu(load: float):
    """Simulate CPU activity for visual effect"""
    feedback = get_feedback_system()
    feedback.simulate_cpu_activity(load)
    await broadcast_state()
    
    return {"success": True, "cpu_load": load}


@router.get("/feedback/colors")
async def get_mode_colors():
    """Get all operating mode colors"""
    return {
        "colors": {
            "pilot": OperatingModeColors.PILOT,
            "copilot": OperatingModeColors.COPILOT,
            "passive": OperatingModeColors.PASSIVE,
            "executive": OperatingModeColors.EXECUTIVE,
            "analyzing": OperatingModeColors.ANALYZING,
            "processing": OperatingModeColors.PROCESSING,
            "alert": OperatingModeColors.ALERT,
            "idle": OperatingModeColors.IDLE
        }
    }


# === WEBSOCKET FOR REAL-TIME UPDATES ===

@router.websocket("/ws")
async def avatar_websocket(websocket: WebSocket):
    """WebSocket for real-time avatar state updates"""
    await websocket.accept()
    _connected_clients.append(websocket)
    
    controller = get_avatar_controller()
    
    logger.info(f"Avatar WebSocket connected. Total clients: {len(_connected_clients)}")
    
    try:
        # Start update loop if first client
        global _update_task
        if _update_task is None or _update_task.done():
            _update_task = asyncio.create_task(avatar_update_loop())
        
        # Send initial state
        await websocket.send_json(controller.get_render_state())
        
        # Listen for commands
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                await handle_websocket_command(data, websocket)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
                
    except WebSocketDisconnect:
        logger.info("Avatar WebSocket disconnected")
    except Exception as e:
        logger.error(f"Avatar WebSocket error: {e}")
    finally:
        if websocket in _connected_clients:
            _connected_clients.remove(websocket)
        
        # Stop update loop if no clients
        if not _connected_clients and _update_task:
            _update_task.cancel()


async def handle_websocket_command(data: Dict, websocket: WebSocket):
    """Handle WebSocket command"""
    controller = get_avatar_controller()
    command = data.get("command")
    
    try:
        if command == "set_expression":
            expression = Expression(data.get("expression", "neutral"))
            intensity = data.get("intensity", 1.0)
            controller.set_expression(expression, intensity)
            
        elif command == "set_state":
            state = AvatarState(data.get("state", "idle"))
            controller.set_state(state)
            
        elif command == "set_gaze":
            target = GazeTarget(data.get("target", "user"))
            offset = data.get("offset", (0, 0))
            controller.set_gaze_target(target, tuple(offset))
            
        elif command == "set_head":
            controller.set_head_pose(
                data.get("pitch", 0),
                data.get("yaw", 0),
                data.get("roll", 0)
            )
            
        elif command == "look_at":
            controller.look_at(data.get("x", 0), data.get("y", 0))
            
        elif command == "start_lipsync":
            controller.start_lip_sync(
                data.get("text", ""),
                data.get("duration")
            )
            
        elif command == "stop_lipsync":
            controller.stop_lip_sync()
            
        elif command == "get_state":
            await websocket.send_json({
                "type": "state",
                "data": controller.get_render_state()
            })
            return
        
        # Broadcast updated state
        await broadcast_state()
        
        # Acknowledge command
        await websocket.send_json({
            "type": "ack",
            "command": command,
            "success": True
        })
        
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "command": command,
            "error": str(e)
        })


async def avatar_update_loop():
    """Background loop to update avatar state"""
    controller = get_avatar_controller()
    last_time = asyncio.get_event_loop().time()
    
    try:
        while _connected_clients:
            current_time = asyncio.get_event_loop().time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update controller
            state = controller.update(dt, current_time)
            
            # Broadcast to all clients
            for client in _connected_clients[:]:
                try:
                    await client.send_json({
                        "type": "state",
                        "data": state
                    })
                except Exception:
                    if client in _connected_clients:
                        _connected_clients.remove(client)
            
            # ~30 FPS update rate
            await asyncio.sleep(1/30)
            
    except asyncio.CancelledError:
        logger.info("Avatar update loop cancelled")


async def broadcast_state():
    """Broadcast current state to all connected clients"""
    if not _connected_clients:
        return
    
    controller = get_avatar_controller()
    state = controller.get_render_state()
    
    for client in _connected_clients[:]:
        try:
            await client.send_json({
                "type": "state",
                "data": state
            })
        except Exception:
            if client in _connected_clients:
                _connected_clients.remove(client)
