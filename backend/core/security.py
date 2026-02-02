"""
🔐 GIDEON 3.0 - Security System
Sistema di sicurezza completo con:
- Kill-switch globale avanzato
- Rollback automatico per errori/rischi
- Permessi granulari per utenti e azioni
"""

import asyncio
import inspect
import hashlib
import json
import os
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============ ENUMS ============

class SecurityLevel(Enum):
    """Livelli di sicurezza del sistema"""
    OPEN = 0            # Nessuna restrizione (dev only)
    STANDARD = 1        # Sicurezza standard
    ELEVATED = 2        # Sicurezza elevata
    HIGH = 3            # Alta sicurezza
    MAXIMUM = 4         # Massima sicurezza
    LOCKDOWN = 5        # Blocco totale


class UserRole(Enum):
    """Ruoli utente"""
    GUEST = "guest"             # Solo lettura
    USER = "user"               # Azioni base
    POWER_USER = "power_user"   # Azioni avanzate
    ADMIN = "admin"             # Amministratore
    SUPERADMIN = "superadmin"   # Super amministratore
    SYSTEM = "system"           # Sistema interno


class ActionCategory(Enum):
    """Categorie di azioni"""
    READ = "read"               # Lettura dati
    WRITE = "write"             # Scrittura dati
    EXECUTE = "execute"         # Esecuzione comandi
    SYSTEM = "system"           # Azioni di sistema
    NETWORK = "network"         # Azioni di rete
    FILE = "file"               # Operazioni file
    PROCESS = "process"         # Gestione processi
    CONFIG = "config"           # Configurazione
    ADMIN = "admin"             # Amministrazione
    CRITICAL = "critical"       # Azioni critiche


class RiskLevel(Enum):
    """Livello di rischio azione"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class KillSwitchState(Enum):
    """Stato del kill-switch"""
    INACTIVE = "inactive"
    SOFT_KILL = "soft_kill"     # Ferma nuove azioni
    HARD_KILL = "hard_kill"     # Ferma tutto immediatamente
    EMERGENCY = "emergency"      # Emergenza totale


class RollbackTrigger(Enum):
    """Trigger per rollback automatico"""
    MANUAL = "manual"
    ERROR = "error"
    RISK_THRESHOLD = "risk_threshold"
    TIMEOUT = "timeout"
    ANOMALY = "anomaly"
    KILL_SWITCH = "kill_switch"


# ============ DATA CLASSES ============

@dataclass
class Permission:
    """Singola permission"""
    action: str
    category: ActionCategory
    allowed_roles: Set[UserRole]
    risk_level: RiskLevel
    requires_confirmation: bool = False
    requires_2fa: bool = False
    max_per_hour: int = -1  # -1 = unlimited
    cooldown_seconds: int = 0
    description: str = ""


@dataclass
class User:
    """Utente del sistema"""
    id: str
    username: str
    role: UserRole
    permissions_override: Dict[str, bool] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    action_counts: Dict[str, int] = field(default_factory=dict)
    action_timestamps: Dict[str, List[datetime]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_locked(self) -> bool:
        """Verifica se utente è bloccato"""
        if self.locked_until and datetime.now() < self.locked_until:
            return True
        return False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role.value,
            "is_active": self.is_active,
            "is_locked": self.is_locked(),
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat()
        }


@dataclass
class RollbackPoint:
    """Punto di rollback"""
    id: str
    timestamp: datetime
    action_id: str
    action_type: str
    original_state: Dict[str, Any]
    rollback_command: str
    rollback_params: Dict[str, Any]
    is_reversible: bool
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action_id": self.action_id,
            "action_type": self.action_type,
            "rollback_command": self.rollback_command,
            "is_reversible": self.is_reversible,
            "priority": self.priority
        }


@dataclass
class SecurityEvent:
    """Evento di sicurezza"""
    id: str
    timestamp: datetime
    event_type: str
    severity: str
    user_id: Optional[str]
    action: str
    details: Dict[str, Any]
    risk_score: float
    blocked: bool
    reason: str
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity,
            "user_id": self.user_id,
            "action": self.action,
            "risk_score": self.risk_score,
            "blocked": self.blocked,
            "reason": self.reason
        }


# ============ KILL SWITCH MANAGER ============

class KillSwitchManager:
    """
    🛑 Kill Switch Manager
    
    Gestisce il kill-switch globale del sistema con diversi livelli:
    - SOFT_KILL: Ferma nuove azioni, completa quelle in corso
    - HARD_KILL: Ferma tutto immediatamente
    - EMERGENCY: Blocco totale con notifiche
    """
    
    def __init__(self):
        self._state = KillSwitchState.INACTIVE
        self._activated_at: Optional[datetime] = None
        self._activated_by: Optional[str] = None
        self._reason: str = ""
        self._active_tasks: List[asyncio.Task] = []
        self._callbacks: List[Callable] = []
        self._history: List[Dict] = []
        self._auto_revive_task: Optional[asyncio.Task] = None
        
        # Codici di emergenza
        self._emergency_codes = {
            "PANIC123": "panic_code",
            "STOP_ALL": "stop_code",
            "EMERGENCY": "emergency_code"
        }
        
        # Whitelist azioni sempre permesse
        self._whitelist = {"get_status", "health_check", "revive", "get_logs"}
    
    @property
    def is_active(self) -> bool:
        return self._state != KillSwitchState.INACTIVE
    
    @property
    def state(self) -> KillSwitchState:
        return self._state
    
    async def soft_kill(self, reason: str, user_id: str = "system") -> Dict:
        """
        Soft kill - ferma nuove azioni
        """
        return await self._activate(KillSwitchState.SOFT_KILL, reason, user_id)
    
    async def hard_kill(self, reason: str, user_id: str = "system") -> Dict:
        """
        Hard kill - ferma tutto immediatamente
        """
        result = await self._activate(KillSwitchState.HARD_KILL, reason, user_id)
        
        # Cancella tutti i task attivi
        cancelled = 0
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
                cancelled += 1
        
        result["tasks_cancelled"] = cancelled
        return result
    
    async def emergency_kill(self, reason: str, code: str = None) -> Dict:
        """
        🚨 Emergency kill - blocco totale con notifiche
        """
        # Verifica codice se fornito
        if code and code not in self._emergency_codes:
            return {"success": False, "error": "Invalid emergency code"}
        
        result = await self._activate(KillSwitchState.EMERGENCY, reason, "EMERGENCY")
        
        # Cancella tutti i task
        cancelled = 0
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
                cancelled += 1
        
        # Notifica emergenza
        await self._notify_emergency(reason)
        
        result["tasks_cancelled"] = cancelled
        result["emergency"] = True
        return result
    
    async def _activate(self, state: KillSwitchState, reason: str, user_id: str) -> Dict:
        """Attiva kill switch"""
        self._state = state
        self._activated_at = datetime.now()
        self._activated_by = user_id
        self._reason = reason
        
        event = {
            "event": "KILL_SWITCH_ACTIVATED",
            "state": state.value,
            "reason": reason,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        self._history.append(event)
        
        # Notifica callbacks
        for callback in self._callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback("kill_switch", event)
                else:
                    callback("kill_switch", event)
            except Exception as e:
                logger.error(f"Kill switch callback error: {e}")
        
        logger.warning(f"🛑 KILL SWITCH ACTIVATED: {state.value} - {reason}")
        
        return {
            "success": True,
            "state": state.value,
            "reason": reason,
            "timestamp": self._activated_at.isoformat(),
            "message": f"⛔ Kill switch attivato: {state.value}"
        }
    
    async def revive(self, user_id: str = "system", code: str = None) -> Dict:
        """
        Riattiva il sistema
        """
        if not self.is_active:
            return {"success": False, "error": "Kill switch non attivo"}
        
        # Per EMERGENCY serve codice
        if self._state == KillSwitchState.EMERGENCY:
            if code not in self._emergency_codes:
                return {"success": False, "error": "Codice emergenza richiesto"}
        
        old_state = self._state
        self._state = KillSwitchState.INACTIVE
        
        event = {
            "event": "KILL_SWITCH_DEACTIVATED",
            "previous_state": old_state.value,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        self._history.append(event)
        
        # Notifica callbacks
        for callback in self._callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback("revive", event)
                else:
                    callback("revive", event)
            except Exception as e:
                logger.error(f"Revive callback error: {e}")
        
        logger.info(f"✅ Sistema riattivato da {user_id}")
        
        return {
            "success": True,
            "previous_state": old_state.value,
            "message": "✅ Sistema riattivato"
        }
    
    async def schedule_auto_revive(self, minutes: int) -> Dict:
        """Schedula auto-revive"""
        if self._auto_revive_task:
            self._auto_revive_task.cancel()
        
        async def auto_revive():
            await asyncio.sleep(minutes * 60)
            if self.is_active and self._state != KillSwitchState.EMERGENCY:
                await self.revive("auto_revive")
        
        self._auto_revive_task = asyncio.create_task(auto_revive())
        
        return {
            "success": True,
            "auto_revive_in_minutes": minutes
        }
    
    def can_execute(self, action: str) -> bool:
        """Verifica se azione può essere eseguita"""
        if action in self._whitelist:
            return True
        
        if self._state == KillSwitchState.INACTIVE:
            return True
        
        return False
    
    async def _notify_emergency(self, reason: str):
        """Notifica emergenza"""
        # In produzione: email, SMS, notifiche push, etc.
        logger.critical(f"🚨 EMERGENCY: {reason}")
    
    def register_task(self, task: asyncio.Task):
        """Registra task per kill-switch"""
        self._active_tasks.append(task)
        self._active_tasks = [t for t in self._active_tasks if not t.done()]
    
    def on_event(self, callback: Callable):
        """Registra callback per eventi"""
        self._callbacks.append(callback)
    
    def get_status(self) -> Dict:
        """Stato kill switch"""
        return {
            "state": self._state.value,
            "is_active": self.is_active,
            "activated_at": self._activated_at.isoformat() if self._activated_at else None,
            "activated_by": self._activated_by,
            "reason": self._reason,
            "active_tasks": len([t for t in self._active_tasks if not t.done()])
        }
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Storico eventi"""
        return self._history[-limit:]


# ============ ROLLBACK MANAGER ============

class RollbackManager:
    """
    🔄 Rollback Manager
    
    Gestisce rollback automatici per:
    - Errori durante esecuzione
    - Superamento soglie di rischio
    - Timeout operazioni
    - Anomalie rilevate
    """
    
    def __init__(self, max_history: int = 1000):
        self._rollback_points: List[RollbackPoint] = []
        self._max_history = max_history
        self._auto_rollback_enabled = True
        self._risk_threshold = 0.7
        self._callbacks: List[Callable] = []
        self._rollback_history: List[Dict] = []
        
        # Handlers per tipi di rollback
        self._handlers: Dict[str, Callable] = {}
        
        # Statistiche
        self._stats = {
            "total_points": 0,
            "rollbacks_executed": 0,
            "rollbacks_failed": 0,
            "auto_rollbacks": 0
        }
    
    def create_rollback_point(
        self,
        action_id: str,
        action_type: str,
        original_state: Dict[str, Any],
        rollback_command: str,
        rollback_params: Dict[str, Any] = None,
        is_reversible: bool = True,
        priority: int = 1,
        metadata: Dict[str, Any] = None
    ) -> RollbackPoint:
        """
        Crea punto di rollback
        """
        point = RollbackPoint(
            id=f"rb_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            action_id=action_id,
            action_type=action_type,
            original_state=original_state,
            rollback_command=rollback_command,
            rollback_params=rollback_params or {},
            is_reversible=is_reversible,
            priority=priority,
            metadata=metadata or {}
        )
        
        self._rollback_points.append(point)
        self._stats["total_points"] += 1
        
        # Mantieni limite storia
        if len(self._rollback_points) > self._max_history:
            self._rollback_points = self._rollback_points[-self._max_history:]
        
        logger.debug(f"Rollback point created: {point.id}")
        return point
    
    async def rollback(
        self,
        point_id: str,
        trigger: RollbackTrigger = RollbackTrigger.MANUAL,
        reason: str = ""
    ) -> Dict:
        """
        Esegue rollback di un punto specifico
        """
        point = next((p for p in self._rollback_points if p.id == point_id), None)
        
        if not point:
            return {"success": False, "error": "Rollback point not found"}
        
        if not point.is_reversible:
            return {"success": False, "error": "Action is not reversible"}
        
        try:
            # Esegui handler se registrato
            if point.action_type in self._handlers:
                handler = self._handlers[point.action_type]
                if inspect.iscoroutinefunction(handler):
                    result = await handler(point)
                else:
                    result = handler(point)
            else:
                # Rollback generico
                result = await self._execute_generic_rollback(point)
            
            # Log
            self._rollback_history.append({
                "point_id": point_id,
                "trigger": trigger.value,
                "reason": reason,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
            
            self._stats["rollbacks_executed"] += 1
            if trigger != RollbackTrigger.MANUAL:
                self._stats["auto_rollbacks"] += 1
            
            # Notifica callbacks
            for callback in self._callbacks:
                try:
                    if inspect.iscoroutinefunction(callback):
                        await callback("rollback_completed", point.to_dict())
                    else:
                        callback("rollback_completed", point.to_dict())
                except Exception:
                    pass
            
            logger.info(f"🔄 Rollback executed: {point_id} - {trigger.value}")
            
            return {
                "success": True,
                "point_id": point_id,
                "action_type": point.action_type,
                "trigger": trigger.value,
                "result": result
            }
        
        except Exception as e:
            self._stats["rollbacks_failed"] += 1
            logger.error(f"Rollback failed: {point_id} - {e}")
            
            return {
                "success": False,
                "point_id": point_id,
                "error": str(e)
            }
    
    async def rollback_last(self, count: int = 1) -> List[Dict]:
        """
        Rollback degli ultimi N punti
        """
        results = []
        
        reversible_points = [p for p in reversed(self._rollback_points) if p.is_reversible]
        
        for point in reversible_points[:count]:
            result = await self.rollback(point.id, RollbackTrigger.MANUAL, "Batch rollback")
            results.append(result)
        
        return results
    
    async def auto_rollback_on_error(
        self,
        action_id: str,
        error: Exception
    ) -> Optional[Dict]:
        """
        Rollback automatico su errore
        """
        if not self._auto_rollback_enabled:
            return None
        
        point = next(
            (p for p in reversed(self._rollback_points) if p.action_id == action_id),
            None
        )
        
        if point:
            return await self.rollback(
                point.id,
                RollbackTrigger.ERROR,
                f"Error: {str(error)}"
            )
        
        return None
    
    async def auto_rollback_on_risk(
        self,
        action_id: str,
        risk_score: float
    ) -> Optional[Dict]:
        """
        Rollback automatico su soglia rischio superata
        """
        if not self._auto_rollback_enabled:
            return None
        
        if risk_score < self._risk_threshold:
            return None
        
        point = next(
            (p for p in reversed(self._rollback_points) if p.action_id == action_id),
            None
        )
        
        if point:
            return await self.rollback(
                point.id,
                RollbackTrigger.RISK_THRESHOLD,
                f"Risk score {risk_score} exceeded threshold {self._risk_threshold}"
            )
        
        return None
    
    async def _execute_generic_rollback(self, point: RollbackPoint) -> Dict:
        """
        Esecuzione rollback generico
        """
        # Implementazione generica basata su comando
        command = point.rollback_command
        params = point.rollback_params
        
        # Qui si integra con il sistema di esecuzione
        logger.info(f"Generic rollback: {command} with {params}")
        
        return {
            "command": command,
            "params": params,
            "executed": True
        }
    
    def register_handler(self, action_type: str, handler: Callable):
        """Registra handler per tipo di azione"""
        self._handlers[action_type] = handler
    
    def on_rollback(self, callback: Callable):
        """Registra callback per rollback"""
        self._callbacks.append(callback)
    
    def set_risk_threshold(self, threshold: float):
        """Imposta soglia rischio per auto-rollback"""
        self._risk_threshold = max(0.0, min(1.0, threshold))
    
    def enable_auto_rollback(self, enabled: bool = True):
        """Abilita/disabilita auto-rollback"""
        self._auto_rollback_enabled = enabled
    
    def get_points(self, limit: int = 50) -> List[Dict]:
        """Lista punti di rollback"""
        return [p.to_dict() for p in self._rollback_points[-limit:]]
    
    def get_stats(self) -> Dict:
        """Statistiche rollback"""
        return {
            **self._stats,
            "auto_rollback_enabled": self._auto_rollback_enabled,
            "risk_threshold": self._risk_threshold,
            "current_points": len(self._rollback_points)
        }
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Storico rollback"""
        return self._rollback_history[-limit:]


# ============ PERMISSION MANAGER ============

class PermissionManager:
    """
    🔑 Permission Manager
    
    Gestisce permessi granulari per:
    - Utenti con ruoli diversi
    - Azioni con categorie e rischi
    - Rate limiting e cooldown
    - Override e whitelist/blacklist
    """
    
    def __init__(self):
        self._users: Dict[str, User] = {}
        self._permissions: Dict[str, Permission] = {}
        self._role_permissions: Dict[UserRole, Set[str]] = {
            UserRole.GUEST: set(),
            UserRole.USER: set(),
            UserRole.POWER_USER: set(),
            UserRole.ADMIN: set(),
            UserRole.SUPERADMIN: set(),
            UserRole.SYSTEM: set()
        }
        self._global_blacklist: Set[str] = set()
        self._callbacks: List[Callable] = []
        self._security_events: List[SecurityEvent] = []
        
        # Inizializza permessi default
        self._init_default_permissions()
    
    def _init_default_permissions(self):
        """Inizializza permessi default"""
        
        # Permessi base per tutti
        base_permissions = [
            Permission("get_status", ActionCategory.READ, 
                      {UserRole.GUEST, UserRole.USER, UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.NONE, description="Visualizza stato sistema"),
            Permission("get_time", ActionCategory.READ,
                      {UserRole.GUEST, UserRole.USER, UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.NONE, description="Ottieni ora corrente"),
            Permission("search_web", ActionCategory.NETWORK,
                      {UserRole.USER, UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.LOW, description="Ricerca web"),
        ]
        
        # Permessi utente standard
        user_permissions = [
            Permission("open_url", ActionCategory.NETWORK,
                      {UserRole.USER, UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.LOW, description="Apri URL nel browser"),
            Permission("open_app", ActionCategory.EXECUTE,
                      {UserRole.USER, UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.MEDIUM, description="Apri applicazione"),
            Permission("send_notification", ActionCategory.WRITE,
                      {UserRole.USER, UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.NONE, description="Invia notifica"),
            Permission("set_reminder", ActionCategory.WRITE,
                      {UserRole.USER, UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.NONE, description="Imposta promemoria"),
        ]
        
        # Permessi power user
        power_permissions = [
            Permission("execute_macro", ActionCategory.EXECUTE,
                      {UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.MEDIUM, requires_confirmation=True, description="Esegui macro"),
            Permission("execute_workflow", ActionCategory.EXECUTE,
                      {UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.MEDIUM, requires_confirmation=True, description="Esegui workflow"),
            Permission("file_read", ActionCategory.FILE,
                      {UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.LOW, description="Leggi file"),
            Permission("file_write", ActionCategory.FILE,
                      {UserRole.POWER_USER, UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.MEDIUM, requires_confirmation=True, description="Scrivi file"),
        ]
        
        # Permessi admin
        admin_permissions = [
            Permission("manage_users", ActionCategory.ADMIN,
                      {UserRole.ADMIN, UserRole.SUPERADMIN},
                      RiskLevel.HIGH, requires_confirmation=True, description="Gestisci utenti"),
            Permission("change_mode", ActionCategory.SYSTEM,
                      {UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.MEDIUM, description="Cambia modalità operativa"),
            Permission("execute_command", ActionCategory.EXECUTE,
                      {UserRole.ADMIN, UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.HIGH, requires_confirmation=True, description="Esegui comando sistema"),
            Permission("manage_config", ActionCategory.CONFIG,
                      {UserRole.ADMIN, UserRole.SUPERADMIN},
                      RiskLevel.MEDIUM, description="Modifica configurazione"),
        ]
        
        # Permessi superadmin/sistema
        super_permissions = [
            Permission("kill_switch", ActionCategory.CRITICAL,
                      {UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.CRITICAL, requires_2fa=True, description="Attiva kill switch"),
            Permission("rollback", ActionCategory.CRITICAL,
                      {UserRole.SUPERADMIN, UserRole.SYSTEM},
                      RiskLevel.HIGH, requires_confirmation=True, description="Esegui rollback"),
            Permission("manage_permissions", ActionCategory.ADMIN,
                      {UserRole.SUPERADMIN},
                      RiskLevel.HIGH, requires_2fa=True, description="Gestisci permessi"),
            Permission("system_shutdown", ActionCategory.CRITICAL,
                      {UserRole.SUPERADMIN},
                      RiskLevel.CRITICAL, requires_2fa=True, requires_confirmation=True, 
                      description="Spegni sistema"),
        ]
        
        # Registra tutti i permessi
        all_permissions = (base_permissions + user_permissions + 
                          power_permissions + admin_permissions + super_permissions)
        
        for perm in all_permissions:
            self._permissions[perm.action] = perm
            for role in perm.allowed_roles:
                self._role_permissions[role].add(perm.action)
    
    def create_user(
        self,
        username: str,
        role: UserRole = UserRole.USER,
        user_id: str = None
    ) -> User:
        """Crea nuovo utente"""
        user_id = user_id or f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        user = User(
            id=user_id,
            username=username,
            role=role
        )
        
        self._users[user_id] = user
        logger.info(f"User created: {username} ({role.value})")
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Ottieni utente"""
        return self._users.get(user_id)
    
    def check_permission(
        self,
        user_id: str,
        action: str,
        context: Dict[str, Any] = None
    ) -> Dict:
        """
        Verifica se utente ha permesso per azione
        
        Returns:
            {
                "allowed": bool,
                "reason": str,
                "requires_confirmation": bool,
                "requires_2fa": bool,
                "risk_level": str
            }
        """
        context = context or {}
        
        # Azione in blacklist globale
        if action in self._global_blacklist:
            return self._deny("Action globally blacklisted", action, user_id)
        
        # Utente esiste?
        user = self._users.get(user_id)
        if not user:
            # Utente sistema sempre permesso
            if user_id == "system":
                return self._allow(action, "system")
            return self._deny("User not found", action, user_id)
        
        # Utente attivo?
        if not user.is_active:
            return self._deny("User is inactive", action, user_id)
        
        # Utente bloccato?
        if user.is_locked():
            return self._deny("User is locked", action, user_id)
        
        # Override utente
        if action in user.permissions_override:
            if user.permissions_override[action]:
                return self._allow(action, user_id, "User override")
            else:
                return self._deny("User override denied", action, user_id)
        
        # Permesso esiste?
        permission = self._permissions.get(action)
        if not permission:
            # Permesso non definito: deny per sicurezza
            return self._deny("Permission not defined", action, user_id)
        
        # Ruolo ha permesso?
        if user.role not in permission.allowed_roles:
            return self._deny(
                f"Role {user.role.value} not allowed",
                action, user_id
            )
        
        # Rate limiting
        if permission.max_per_hour > 0:
            count = self._count_recent_actions(user, action, hours=1)
            if count >= permission.max_per_hour:
                return self._deny(
                    f"Rate limit exceeded ({count}/{permission.max_per_hour})",
                    action, user_id
                )
        
        # Cooldown
        if permission.cooldown_seconds > 0:
            if not self._check_cooldown(user, action, permission.cooldown_seconds):
                return self._deny("Cooldown active", action, user_id)
        
        # Aggiorna timestamp
        user.last_active = datetime.now()
        if action not in user.action_timestamps:
            user.action_timestamps[action] = []
        user.action_timestamps[action].append(datetime.now())
        
        # Permesso concesso
        return {
            "allowed": True,
            "reason": "Permission granted",
            "action": action,
            "user_id": user_id,
            "role": user.role.value,
            "requires_confirmation": permission.requires_confirmation,
            "requires_2fa": permission.requires_2fa,
            "risk_level": permission.risk_level.name,
            "category": permission.category.value
        }
    
    def _allow(self, action: str, user_id: str, reason: str = "Allowed") -> Dict:
        """Risposta permesso concesso"""
        return {
            "allowed": True,
            "reason": reason,
            "action": action,
            "user_id": user_id,
            "requires_confirmation": False,
            "requires_2fa": False,
            "risk_level": "NONE"
        }
    
    def _deny(self, reason: str, action: str, user_id: str) -> Dict:
        """Risposta permesso negato"""
        # Log evento sicurezza
        event = SecurityEvent(
            id=f"sec_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(),
            event_type="permission_denied",
            severity="warning",
            user_id=user_id,
            action=action,
            details={"reason": reason},
            risk_score=0.5,
            blocked=True,
            reason=reason
        )
        self._security_events.append(event)
        
        return {
            "allowed": False,
            "reason": reason,
            "action": action,
            "user_id": user_id
        }
    
    def _count_recent_actions(self, user: User, action: str, hours: int) -> int:
        """Conta azioni recenti"""
        if action not in user.action_timestamps:
            return 0
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [t for t in user.action_timestamps[action] if t > cutoff]
        
        # Cleanup vecchi timestamp
        user.action_timestamps[action] = recent
        
        return len(recent)
    
    def _check_cooldown(self, user: User, action: str, seconds: int) -> bool:
        """Verifica cooldown"""
        if action not in user.action_timestamps:
            return True
        
        if not user.action_timestamps[action]:
            return True
        
        last_action = user.action_timestamps[action][-1]
        elapsed = (datetime.now() - last_action).total_seconds()
        
        return elapsed >= seconds
    
    def grant_permission(self, user_id: str, action: str) -> bool:
        """Concedi permesso specifico a utente"""
        user = self._users.get(user_id)
        if user:
            user.permissions_override[action] = True
            return True
        return False
    
    def revoke_permission(self, user_id: str, action: str) -> bool:
        """Revoca permesso specifico a utente"""
        user = self._users.get(user_id)
        if user:
            user.permissions_override[action] = False
            return True
        return False
    
    def lock_user(self, user_id: str, minutes: int = 30) -> bool:
        """Blocca utente"""
        user = self._users.get(user_id)
        if user:
            user.locked_until = datetime.now() + timedelta(minutes=minutes)
            logger.warning(f"User locked: {user_id} for {minutes} minutes")
            return True
        return False
    
    def unlock_user(self, user_id: str) -> bool:
        """Sblocca utente"""
        user = self._users.get(user_id)
        if user:
            user.locked_until = None
            user.failed_attempts = 0
            return True
        return False
    
    def add_to_blacklist(self, action: str):
        """Aggiungi azione a blacklist globale"""
        self._global_blacklist.add(action)
    
    def remove_from_blacklist(self, action: str):
        """Rimuovi azione da blacklist"""
        self._global_blacklist.discard(action)
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Lista permessi utente"""
        user = self._users.get(user_id)
        if not user:
            return []
        
        return list(self._role_permissions.get(user.role, set()))
    
    def get_security_events(self, limit: int = 100) -> List[Dict]:
        """Eventi sicurezza recenti"""
        return [e.to_dict() for e in self._security_events[-limit:]]
    
    def get_stats(self) -> Dict:
        """Statistiche permessi"""
        return {
            "total_users": len(self._users),
            "active_users": sum(1 for u in self._users.values() if u.is_active),
            "locked_users": sum(1 for u in self._users.values() if u.is_locked()),
            "total_permissions": len(self._permissions),
            "blacklisted_actions": len(self._global_blacklist),
            "security_events": len(self._security_events)
        }


# ============ SECURITY SYSTEM (MAIN) ============

class SecuritySystem:
    """
    🔐 Security System
    
    Sistema di sicurezza integrato che combina:
    - Kill Switch Manager
    - Rollback Manager
    - Permission Manager
    """
    
    def __init__(self):
        self.kill_switch = KillSwitchManager()
        self.rollback = RollbackManager()
        self.permissions = PermissionManager()
        
        self._security_level = SecurityLevel.STANDARD
        self._audit_log: List[Dict] = []
        
        # Crea utente sistema
        self.permissions.create_user("system", UserRole.SYSTEM, "system")
        
        # Registra callback cross-component
        self.kill_switch.on_event(self._on_kill_switch_event)
        self.rollback.on_rollback(self._on_rollback_event)
    
    async def _on_kill_switch_event(self, event_type: str, data: Dict):
        """Handler eventi kill switch"""
        self._audit_log.append({
            "component": "kill_switch",
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Se kill switch attivato, rollback automatico
        if event_type == "kill_switch" and data.get("state") == "hard_kill":
            await self.rollback.rollback_last(3)
    
    async def _on_rollback_event(self, event_type: str, data: Dict):
        """Handler eventi rollback"""
        self._audit_log.append({
            "component": "rollback",
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    async def authorize_action(
        self,
        user_id: str,
        action: str,
        context: Dict[str, Any] = None
    ) -> Dict:
        """
        Autorizza azione verificando:
        1. Kill switch non attivo
        2. Permessi utente
        3. Livello sicurezza
        """
        # Kill switch check
        if not self.kill_switch.can_execute(action):
            return {
                "authorized": False,
                "reason": f"Kill switch active: {self.kill_switch.state.value}",
                "action": action
            }
        
        # Permission check
        perm_result = self.permissions.check_permission(user_id, action, context)
        
        if not perm_result["allowed"]:
            return {
                "authorized": False,
                "reason": perm_result["reason"],
                "action": action
            }
        
        # Security level check
        if self._security_level == SecurityLevel.LOCKDOWN:
            return {
                "authorized": False,
                "reason": "System in lockdown",
                "action": action
            }
        
        return {
            "authorized": True,
            "action": action,
            "user_id": user_id,
            "requires_confirmation": perm_result.get("requires_confirmation", False),
            "requires_2fa": perm_result.get("requires_2fa", False),
            "risk_level": perm_result.get("risk_level", "NONE")
        }
    
    async def execute_with_rollback(
        self,
        action_id: str,
        action_type: str,
        executor: Callable,
        rollback_command: str,
        rollback_params: Dict = None,
        original_state: Dict = None
    ) -> Dict:
        """
        Esegue azione con rollback automatico su errore
        """
        # Crea punto rollback
        point = self.rollback.create_rollback_point(
            action_id=action_id,
            action_type=action_type,
            original_state=original_state or {},
            rollback_command=rollback_command,
            rollback_params=rollback_params
        )
        
        try:
            # Esegui azione
            if inspect.iscoroutinefunction(executor):
                result = await executor()
            else:
                result = executor()
            
            return {
                "success": True,
                "result": result,
                "rollback_point": point.id
            }
        
        except Exception as e:
            # Rollback automatico
            rollback_result = await self.rollback.auto_rollback_on_error(action_id, e)
            
            return {
                "success": False,
                "error": str(e),
                "rollback_executed": rollback_result is not None,
                "rollback_result": rollback_result
            }
    
    async def emergency_stop(self, reason: str = "Emergency stop") -> Dict:
        """Stop di emergenza completo"""
        # Attiva kill switch
        kill_result = await self.kill_switch.emergency_kill(reason)
        
        # Rollback ultime azioni
        rollback_result = await self.rollback.rollback_last(5)
        
        # Imposta lockdown
        self._security_level = SecurityLevel.LOCKDOWN
        
        return {
            "success": True,
            "kill_switch": kill_result,
            "rollback": rollback_result,
            "security_level": self._security_level.name
        }
    
    def set_security_level(self, level: SecurityLevel) -> Dict:
        """Imposta livello sicurezza"""
        old_level = self._security_level
        self._security_level = level
        
        self._audit_log.append({
            "event": "security_level_change",
            "from": old_level.name,
            "to": level.name,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "previous": old_level.name,
            "current": level.name
        }
    
    def get_status(self) -> Dict:
        """Stato completo sicurezza"""
        return {
            "security_level": self._security_level.name,
            "kill_switch": self.kill_switch.get_status(),
            "rollback": self.rollback.get_stats(),
            "permissions": self.permissions.get_stats(),
            "audit_log_size": len(self._audit_log)
        }
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Audit log"""
        return self._audit_log[-limit:]


# ============ FACTORY ============

def create_security_system() -> SecuritySystem:
    """Factory per creare sistema sicurezza"""
    return SecuritySystem()
