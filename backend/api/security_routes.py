"""
🔐 Security API Routes - GIDEON 3.0

API per sistema di sicurezza:
- Kill switch management
- Rollback operations
- Permission management
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from core.security import (
    SecuritySystem,
    SecurityLevel,
    UserRole,
    KillSwitchState,
    RollbackTrigger,
    create_security_system
)

router = APIRouter(prefix="/security", tags=["Security"])


# ============ GLOBAL INSTANCE ============

_security_system: Optional[SecuritySystem] = None


def get_security() -> SecuritySystem:
    """Ottieni istanza SecuritySystem"""
    global _security_system
    if _security_system is None:
        _security_system = create_security_system()
    return _security_system


def set_security_instance(security: SecuritySystem):
    """Imposta istanza SecuritySystem"""
    global _security_system
    _security_system = security


# ============ MODELS ============

class KillSwitchRequest(BaseModel):
    """Richiesta kill switch"""
    reason: str = "Manual activation"
    code: Optional[str] = None


class ReviveRequest(BaseModel):
    """Richiesta revive"""
    code: Optional[str] = None


class RollbackRequest(BaseModel):
    """Richiesta rollback"""
    point_id: Optional[str] = None
    count: int = 1
    reason: str = "Manual rollback"


class CreateUserRequest(BaseModel):
    """Richiesta creazione utente"""
    username: str
    role: str = "user"


class PermissionCheckRequest(BaseModel):
    """Richiesta verifica permesso"""
    user_id: str
    action: str
    context: Dict[str, Any] = {}


class PermissionModifyRequest(BaseModel):
    """Richiesta modifica permesso"""
    user_id: str
    action: str


class SecurityLevelRequest(BaseModel):
    """Richiesta cambio livello sicurezza"""
    level: str


class UserLockRequest(BaseModel):
    """Richiesta blocco utente"""
    user_id: str
    minutes: int = 30


# ============ KILL SWITCH ENDPOINTS ============

@router.get("/status")
async def get_security_status():
    """
    Stato completo del sistema di sicurezza
    """
    security = get_security()
    return {
        "success": True,
        "status": security.get_status(),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/kill-switch/soft")
async def soft_kill(request: KillSwitchRequest):
    """
    🟡 Soft Kill - Ferma nuove azioni, completa quelle in corso
    """
    security = get_security()
    result = await security.kill_switch.soft_kill(request.reason)
    return result


@router.post("/kill-switch/hard")
async def hard_kill(request: KillSwitchRequest):
    """
    🔴 Hard Kill - Ferma TUTTO immediatamente
    """
    security = get_security()
    result = await security.kill_switch.hard_kill(request.reason)
    return result


@router.post("/kill-switch/emergency")
async def emergency_kill(request: KillSwitchRequest):
    """
    🚨 Emergency Kill - Blocco totale con notifiche
    """
    security = get_security()
    result = await security.kill_switch.emergency_kill(request.reason, request.code)
    return result


@router.post("/kill-switch/revive")
async def revive_system(request: ReviveRequest):
    """
    ✅ Revive - Riattiva sistema dopo kill switch
    """
    security = get_security()
    result = await security.kill_switch.revive(code=request.code)
    return result


@router.get("/kill-switch/status")
async def get_kill_switch_status():
    """
    Stato kill switch
    """
    security = get_security()
    return security.kill_switch.get_status()


@router.get("/kill-switch/history")
async def get_kill_switch_history(limit: int = 20):
    """
    Storico eventi kill switch
    """
    security = get_security()
    return {
        "history": security.kill_switch.get_history(limit),
        "count": len(security.kill_switch.get_history(limit))
    }


# ============ ROLLBACK ENDPOINTS ============

@router.post("/rollback")
async def execute_rollback(request: RollbackRequest):
    """
    🔄 Esegui rollback
    
    - Se point_id fornito: rollback punto specifico
    - Altrimenti: rollback ultime N azioni
    """
    security = get_security()
    
    if request.point_id:
        result = await security.rollback.rollback(
            request.point_id,
            RollbackTrigger.MANUAL,
            request.reason
        )
        return result
    else:
        results = await security.rollback.rollback_last(request.count)
        return {
            "success": True,
            "rolled_back": len([r for r in results if r.get("success")]),
            "total": request.count,
            "results": results
        }


@router.get("/rollback/points")
async def get_rollback_points(limit: int = 50):
    """
    Lista punti di rollback disponibili
    """
    security = get_security()
    return {
        "points": security.rollback.get_points(limit),
        "stats": security.rollback.get_stats()
    }


@router.get("/rollback/history")
async def get_rollback_history(limit: int = 50):
    """
    Storico rollback eseguiti
    """
    security = get_security()
    return {
        "history": security.rollback.get_history(limit)
    }


@router.post("/rollback/settings")
async def update_rollback_settings(
    auto_enabled: Optional[bool] = None,
    risk_threshold: Optional[float] = None
):
    """
    Aggiorna impostazioni rollback
    """
    security = get_security()
    
    if auto_enabled is not None:
        security.rollback.enable_auto_rollback(auto_enabled)
    
    if risk_threshold is not None:
        security.rollback.set_risk_threshold(risk_threshold)
    
    return {
        "success": True,
        "settings": {
            "auto_rollback_enabled": security.rollback._auto_rollback_enabled,
            "risk_threshold": security.rollback._risk_threshold
        }
    }


# ============ PERMISSION ENDPOINTS ============

@router.post("/permissions/check")
async def check_permission(request: PermissionCheckRequest):
    """
    🔑 Verifica permesso utente per azione
    """
    security = get_security()
    result = security.permissions.check_permission(
        request.user_id,
        request.action,
        request.context
    )
    return result


@router.post("/permissions/authorize")
async def authorize_action(request: PermissionCheckRequest):
    """
    🔐 Autorizza azione (check completo con kill-switch e livello sicurezza)
    """
    security = get_security()
    result = await security.authorize_action(
        request.user_id,
        request.action,
        request.context
    )
    return result


@router.get("/permissions/user/{user_id}")
async def get_user_permissions(user_id: str):
    """
    Lista permessi utente
    """
    security = get_security()
    user = security.permissions.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user": user.to_dict(),
        "permissions": security.permissions.get_user_permissions(user_id)
    }


@router.post("/permissions/grant")
async def grant_permission(request: PermissionModifyRequest):
    """
    ➕ Concedi permesso specifico a utente
    """
    security = get_security()
    success = security.permissions.grant_permission(request.user_id, request.action)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "success": True,
        "message": f"Permission {request.action} granted to {request.user_id}"
    }


@router.post("/permissions/revoke")
async def revoke_permission(request: PermissionModifyRequest):
    """
    ➖ Revoca permesso specifico a utente
    """
    security = get_security()
    success = security.permissions.revoke_permission(request.user_id, request.action)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "success": True,
        "message": f"Permission {request.action} revoked from {request.user_id}"
    }


# ============ USER MANAGEMENT ENDPOINTS ============

@router.post("/users")
async def create_user(request: CreateUserRequest):
    """
    👤 Crea nuovo utente
    """
    security = get_security()
    
    try:
        role = UserRole(request.role)
    except ValueError:
        role = UserRole.USER
    
    user = security.permissions.create_user(request.username, role)
    
    return {
        "success": True,
        "user": user.to_dict()
    }


@router.get("/users")
async def list_users():
    """
    Lista tutti gli utenti
    """
    security = get_security()
    users = [u.to_dict() for u in security.permissions._users.values()]
    
    return {
        "users": users,
        "count": len(users)
    }


@router.get("/users/{user_id}")
async def get_user(user_id: str):
    """
    Dettagli utente
    """
    security = get_security()
    user = security.permissions.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user": user.to_dict(),
        "permissions": security.permissions.get_user_permissions(user_id)
    }


@router.post("/users/lock")
async def lock_user(request: UserLockRequest):
    """
    🔒 Blocca utente
    """
    security = get_security()
    success = security.permissions.lock_user(request.user_id, request.minutes)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "success": True,
        "message": f"User {request.user_id} locked for {request.minutes} minutes"
    }


@router.post("/users/{user_id}/unlock")
async def unlock_user(user_id: str):
    """
    🔓 Sblocca utente
    """
    security = get_security()
    success = security.permissions.unlock_user(user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "success": True,
        "message": f"User {user_id} unlocked"
    }


# ============ SECURITY LEVEL ENDPOINTS ============

@router.get("/level")
async def get_security_level():
    """
    Livello sicurezza corrente
    """
    security = get_security()
    return {
        "level": security._security_level.name,
        "value": security._security_level.value
    }


@router.post("/level")
async def set_security_level(request: SecurityLevelRequest):
    """
    Imposta livello sicurezza
    """
    security = get_security()
    
    try:
        level = SecurityLevel[request.level.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid security level: {request.level}")
    
    result = security.set_security_level(level)
    return {
        "success": True,
        **result
    }


@router.get("/levels")
async def list_security_levels():
    """
    Lista livelli sicurezza disponibili
    """
    return {
        "levels": [
            {"name": level.name, "value": level.value}
            for level in SecurityLevel
        ]
    }


# ============ EMERGENCY ENDPOINTS ============

@router.post("/emergency-stop")
async def emergency_stop(reason: str = "Emergency stop requested"):
    """
    🚨 EMERGENCY STOP - Blocca tutto e rollback
    """
    security = get_security()
    result = await security.emergency_stop(reason)
    return result


# ============ AUDIT & EVENTS ENDPOINTS ============

@router.get("/audit-log")
async def get_audit_log(limit: int = 100):
    """
    Audit log sicurezza
    """
    security = get_security()
    return {
        "log": security.get_audit_log(limit),
        "count": len(security.get_audit_log(limit))
    }


@router.get("/security-events")
async def get_security_events(limit: int = 100):
    """
    Eventi sicurezza (tentativi bloccati, etc.)
    """
    security = get_security()
    return {
        "events": security.permissions.get_security_events(limit)
    }


# ============ BLACKLIST ENDPOINTS ============

@router.post("/blacklist/add")
async def add_to_blacklist(action: str):
    """
    Aggiungi azione a blacklist globale
    """
    security = get_security()
    security.permissions.add_to_blacklist(action)
    return {
        "success": True,
        "message": f"Action {action} added to blacklist"
    }


@router.post("/blacklist/remove")
async def remove_from_blacklist(action: str):
    """
    Rimuovi azione da blacklist
    """
    security = get_security()
    security.permissions.remove_from_blacklist(action)
    return {
        "success": True,
        "message": f"Action {action} removed from blacklist"
    }


@router.get("/blacklist")
async def get_blacklist():
    """
    Lista azioni in blacklist
    """
    security = get_security()
    return {
        "blacklist": list(security.permissions._global_blacklist)
    }


# ============ ROLES ENDPOINTS ============

@router.get("/roles")
async def list_roles():
    """
    Lista ruoli disponibili
    """
    return {
        "roles": [
            {
                "name": role.name,
                "value": role.value,
                "description": {
                    "GUEST": "Solo lettura",
                    "USER": "Azioni base",
                    "POWER_USER": "Azioni avanzate",
                    "ADMIN": "Amministratore",
                    "SUPERADMIN": "Super amministratore",
                    "SYSTEM": "Sistema interno"
                }.get(role.name, "")
            }
            for role in UserRole
        ]
    }


@router.get("/roles/{role_name}/permissions")
async def get_role_permissions(role_name: str):
    """
    Permessi per ruolo
    """
    security = get_security()
    
    try:
        role = UserRole[role_name.upper()]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Role not found: {role_name}")
    
    permissions = security.permissions._role_permissions.get(role, set())
    
    return {
        "role": role.value,
        "permissions": list(permissions),
        "count": len(permissions)
    }
