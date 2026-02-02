"""
🎯 CAM API ROUTES
==================
API endpoints per Crisis Automation Mode.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cam", tags=["CAM - Crisis Automation Mode"])


# ============ MODELS ============
class CAMStatusResponse(BaseModel):
    status: str = "MONITORING"
    crisis_level: str = "NONE"
    autonomy_percentage: int = 100
    control_state: str = "NORMAL"
    layers_active: int = 6
    last_check: str = ""
    active_automations: int = 0


class EmergencyStopResponse(BaseModel):
    success: bool
    message: str
    stopped_automations: int = 0
    timestamp: str


class LayerStatusResponse(BaseModel):
    layer_name: str
    status: str
    details: Dict[str, Any]


class AuditLogEntry(BaseModel):
    timestamp: str
    action: str
    layer: str
    details: str
    user: str = "system"


# ============ STATE ============
class CAMState:
    def __init__(self):
        self.status = "MONITORING"
        self.crisis_level = "NONE"
        self.autonomy_percentage = 100
        self.control_state = "NORMAL"
        self.emergency_stopped = False
        self.audit_log: List[Dict] = []
        self.layers_status = {
            "detection": {"status": "Active", "last_signal": None},
            "control": {"status": "Active", "autonomy": 100},
            "reasoning": {"status": "Ready", "confidence": 95},
            "automation": {"status": "Ready", "active_tasks": 0},
            "human": {"status": "Active", "pending_approvals": 0},
            "recovery": {"status": "Standby", "lessons_learned": 0}
        }
        self._add_audit("SYSTEM", "CAM initialized", "system")
    
    def _add_audit(self, action: str, details: str, layer: str = "orchestrator"):
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "layer": layer,
            "details": details
        })
        # Keep last 100 entries
        if len(self.audit_log) > 100:
            self.audit_log = self.audit_log[-100:]


# Global state
cam_state = CAMState()


# ============ ENDPOINTS ============
@router.get("/status", response_model=CAMStatusResponse)
async def get_cam_status():
    """Get current CAM status and all layer states."""
    return CAMStatusResponse(
        status=cam_state.status,
        crisis_level=cam_state.crisis_level,
        autonomy_percentage=cam_state.autonomy_percentage,
        control_state=cam_state.control_state,
        layers_active=sum(1 for l in cam_state.layers_status.values() 
                         if l["status"] in ["Active", "Ready"]),
        last_check=datetime.now().isoformat(),
        active_automations=cam_state.layers_status["automation"]["active_tasks"]
    )


@router.post("/emergency-stop", response_model=EmergencyStopResponse)
async def emergency_stop():
    """
    🛑 EMERGENCY STOP - Ferma tutte le automazioni immediatamente.
    """
    logger.warning("🛑 EMERGENCY STOP TRIGGERED!")
    
    # Stop all automations
    stopped_count = cam_state.layers_status["automation"]["active_tasks"]
    
    # Update state
    cam_state.status = "STOPPED"
    cam_state.emergency_stopped = True
    cam_state.autonomy_percentage = 0
    cam_state.control_state = "EMERGENCY_STOPPED"
    
    # Update all layers
    for layer in cam_state.layers_status:
        cam_state.layers_status[layer]["status"] = "STOPPED"
    
    cam_state._add_audit("EMERGENCY_STOP", f"All automations stopped. Count: {stopped_count}", "orchestrator")
    
    return EmergencyStopResponse(
        success=True,
        message="🛑 EMERGENCY STOP completato. Tutte le automazioni fermate.",
        stopped_automations=stopped_count,
        timestamp=datetime.now().isoformat()
    )


@router.post("/resume")
async def resume_operations():
    """Resume normal operations after emergency stop."""
    if not cam_state.emergency_stopped:
        return {"success": False, "message": "Sistema non in stato di emergenza"}
    
    cam_state.status = "MONITORING"
    cam_state.emergency_stopped = False
    cam_state.autonomy_percentage = 100
    cam_state.control_state = "NORMAL"
    
    # Restore layer statuses
    cam_state.layers_status["detection"]["status"] = "Active"
    cam_state.layers_status["control"]["status"] = "Active"
    cam_state.layers_status["reasoning"]["status"] = "Ready"
    cam_state.layers_status["automation"]["status"] = "Ready"
    cam_state.layers_status["human"]["status"] = "Active"
    cam_state.layers_status["recovery"]["status"] = "Standby"
    
    cam_state._add_audit("RESUME", "Operations resumed after emergency stop", "orchestrator")
    
    return {"success": True, "message": "Operazioni riprese normalmente"}


@router.get("/layer/{layer_name}", response_model=LayerStatusResponse)
async def get_layer_status(layer_name: str):
    """Get detailed status of a specific CAM layer."""
    if layer_name not in cam_state.layers_status:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_name}' not found")
    
    layer_info = cam_state.layers_status[layer_name]
    
    layer_details = {
        "detection": {
            "description": "🔍 Threat Detection - Analisi continua minacce e anomalie",
            "signals_monitored": ["system_health", "user_behavior", "external_events"],
            "threshold": "adaptive"
        },
        "control": {
            "description": "⚙️ Controlled Response - Governa autonomia e potenza",
            "autonomy_level": cam_state.autonomy_percentage,
            "clamp_active": False
        },
        "reasoning": {
            "description": "🧠 Multi-Path Reasoning - Analisi decisionale avanzata",
            "paths_evaluated": 3,
            "uncertainty_handling": "explicit"
        },
        "automation": {
            "description": "🤖 Smart Automation - Automazioni con limiti di sicurezza",
            "active_tasks": layer_info.get("active_tasks", 0),
            "max_parallel": 5
        },
        "human": {
            "description": "👤 Human Interface - Anti-panico, ruoli, cognitive load",
            "pending_approvals": layer_info.get("pending_approvals", 0),
            "ui_mode": "normal"
        },
        "recovery": {
            "description": "🔄 Recovery Protocol - Post-crisi e lessons learned",
            "lessons_learned": layer_info.get("lessons_learned", 0),
            "restore_mode": "gradual"
        }
    }
    
    return LayerStatusResponse(
        layer_name=layer_name,
        status=layer_info["status"],
        details=layer_details.get(layer_name, {})
    )


@router.get("/audit-log")
async def get_audit_log(limit: int = 50):
    """Get CAM audit log entries."""
    entries = cam_state.audit_log[-limit:]
    return {
        "total_entries": len(cam_state.audit_log),
        "returned": len(entries),
        "entries": entries
    }


@router.post("/simulate-crisis")
async def simulate_crisis(level: str = "LOW"):
    """Simulate a crisis for testing CAM response."""
    valid_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if level.upper() not in valid_levels:
        raise HTTPException(status_code=400, detail=f"Invalid level. Use: {valid_levels}")
    
    cam_state.crisis_level = level.upper()
    
    if level.upper() == "CRITICAL":
        cam_state.status = "CRISIS"
        cam_state.autonomy_percentage = 50
        cam_state.control_state = "RESTRICTED"
    elif level.upper() == "HIGH":
        cam_state.status = "ALERT"
        cam_state.autonomy_percentage = 75
        cam_state.control_state = "CAUTIOUS"
    elif level.upper() == "MEDIUM":
        cam_state.status = "ALERT"
        cam_state.autonomy_percentage = 90
    else:
        cam_state.status = "MONITORING"
        cam_state.autonomy_percentage = 100
        cam_state.control_state = "NORMAL"
    
    cam_state._add_audit("SIMULATE_CRISIS", f"Crisis level set to {level.upper()}", "detection")
    
    return {
        "success": True,
        "message": f"Crisis level simulated: {level.upper()}",
        "new_status": cam_state.status
    }


@router.post("/clear-crisis")
async def clear_crisis():
    """Clear crisis state and return to normal monitoring."""
    cam_state.crisis_level = "NONE"
    cam_state.status = "MONITORING"
    cam_state.autonomy_percentage = 100
    cam_state.control_state = "NORMAL"
    
    cam_state._add_audit("CLEAR_CRISIS", "Crisis cleared, returning to normal", "recovery")
    
    return {"success": True, "message": "Crisis cleared. Normal operations resumed."}
