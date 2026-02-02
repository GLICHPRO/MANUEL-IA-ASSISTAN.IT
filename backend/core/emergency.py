"""
🚨 JARVIS CORE - Emergency System
Kill-switch, rollback automatico e modalità emergenza
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, List, Dict, Any
from enum import Enum


class EmergencyLevel(Enum):
    """Livelli di emergenza"""
    NORMAL = 0          # Operatività normale
    CAUTION = 1         # Attenzione, monitoraggio aumentato
    WARNING = 2         # Allerta, alcune funzioni limitate  
    CRITICAL = 3        # Critico, solo azioni essenziali
    LOCKDOWN = 4        # Blocco totale, solo admin


class EmergencySystem:
    """
    Sistema di emergenza per Jarvis Core
    
    Funzionalità:
    - KILL_SWITCH: Ferma tutto immediatamente
    - ROLLBACK: Annulla ultime N azioni
    - SAFE_MODE: Solo azioni sicure
    - LOCKDOWN: Blocco totale
    """
    
    def __init__(self, mode_manager=None, rollback_manager=None, executor=None):
        self.mode_manager = mode_manager
        self.rollback_manager = rollback_manager
        self.executor = executor
        
        self._emergency_level = EmergencyLevel.NORMAL
        self._is_killed = False
        self._kill_timestamp = None
        self._kill_reason = None
        
        # Azioni permesse per livello
        self._level_permissions = {
            EmergencyLevel.NORMAL: {"all": True},
            EmergencyLevel.CAUTION: {"all": True, "log_all": True},
            EmergencyLevel.WARNING: {
                "allow": ["notify", "search_web", "open_url", "set_volume"],
                "deny_categories": ["system", "critical"]
            },
            EmergencyLevel.CRITICAL: {
                "allow": ["notify"],
                "deny_categories": ["all_except_notify"]
            },
            EmergencyLevel.LOCKDOWN: {
                "allow": [],
                "admin_only": True
            }
        }
        
        # Callbacks per eventi emergenza
        self._emergency_callbacks: List[Callable] = []
        
        # History emergenze
        self._emergency_history = []
        
        # Auto-recovery timer
        self._recovery_task = None
        
        # Running tasks da fermare con kill-switch
        self._active_tasks: List[asyncio.Task] = []
    
    @property
    def is_emergency(self) -> bool:
        """Verifica se siamo in stato di emergenza"""
        return self._emergency_level.value >= EmergencyLevel.WARNING.value
    
    @property
    def is_killed(self) -> bool:
        """Verifica se kill-switch è attivo"""
        return self._is_killed
    
    async def kill_switch(self, reason: str = "Manual kill-switch") -> dict:
        """
        🛑 KILL SWITCH - Ferma TUTTO immediatamente
        
        - Ferma tutte le automazioni
        - Cancella task in corso
        - Passa a modalità PASSIVE
        - Blocca nuove azioni
        """
        self._is_killed = True
        self._kill_timestamp = datetime.now()
        self._kill_reason = reason
        self._emergency_level = EmergencyLevel.LOCKDOWN
        
        # Log
        self._emergency_history.append({
            "event": "KILL_SWITCH",
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        # Ferma tutti i task attivi
        cancelled = 0
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
                cancelled += 1
        
        # Passa a modalità PASSIVE
        if self.mode_manager:
            await self.mode_manager.go_passive("KILL SWITCH ATTIVATO")
        
        # Ferma executor se disponibile
        if self.executor and hasattr(self.executor, 'stop'):
            await self.executor.stop()
        
        # Notifica callbacks
        await self._notify_emergency("KILL_SWITCH", reason)
        
        return {
            "success": True,
            "status": "KILLED",
            "reason": reason,
            "timestamp": self._kill_timestamp.isoformat(),
            "tasks_cancelled": cancelled,
            "message": "⛔ SISTEMA FERMATO - Kill switch attivato"
        }
    
    async def revive(self, admin_code: str = None) -> dict:
        """
        Riattiva il sistema dopo kill-switch
        
        Args:
            admin_code: Codice admin per conferma (opzionale)
        """
        if not self._is_killed:
            return {"success": False, "error": "Sistema non in stato killed"}
        
        self._is_killed = False
        self._emergency_level = EmergencyLevel.CAUTION  # Riparte con cautela
        
        self._emergency_history.append({
            "event": "REVIVE",
            "timestamp": datetime.now().isoformat()
        })
        
        # Passa a COPILOT (non PILOT per sicurezza)
        if self.mode_manager:
            await self.mode_manager.go_copilot("Sistema riattivato dopo kill-switch")
        
        # Notifica
        await self._notify_emergency("REVIVE", "Sistema riattivato")
        
        # Auto-recovery a NORMAL dopo 5 minuti senza problemi
        await self._schedule_auto_recovery(minutes=5)
        
        return {
            "success": True,
            "status": "REVIVED",
            "emergency_level": self._emergency_level.name,
            "message": "✅ Sistema riattivato in modalità CAUTION"
        }
    
    async def emergency_rollback(self, count: int = 5) -> dict:
        """
        Rollback di emergenza delle ultime N azioni
        """
        if not self.rollback_manager:
            return {"success": False, "error": "RollbackManager non disponibile"}
        
        self._emergency_history.append({
            "event": "EMERGENCY_ROLLBACK",
            "count": count,
            "timestamp": datetime.now().isoformat()
        })
        
        results = await self.rollback_manager.rollback_last(count)
        
        successful = sum(1 for r in results if r.get("success"))
        
        return {
            "success": True,
            "rolled_back": successful,
            "total_attempted": count,
            "details": results,
            "message": f"🔄 Rollback completato: {successful}/{count} azioni annullate"
        }
    
    async def set_emergency_level(self, level: EmergencyLevel, reason: str = None) -> dict:
        """Imposta livello di emergenza"""
        old_level = self._emergency_level
        self._emergency_level = level
        
        self._emergency_history.append({
            "event": "LEVEL_CHANGE",
            "from": old_level.name,
            "to": level.name,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        # Aggiusta modalità in base al livello
        if self.mode_manager:
            if level == EmergencyLevel.LOCKDOWN:
                await self.mode_manager.go_passive("Lockdown attivo")
            elif level.value >= EmergencyLevel.WARNING.value:
                await self.mode_manager.go_copilot("Livello emergenza elevato")
        
        await self._notify_emergency("LEVEL_CHANGE", f"{old_level.name} -> {level.name}")
        
        return {
            "success": True,
            "previous_level": old_level.name,
            "current_level": level.name,
            "reason": reason
        }
    
    async def safe_mode(self) -> dict:
        """Attiva modalità sicura (solo azioni safe)"""
        return await self.set_emergency_level(
            EmergencyLevel.WARNING, 
            "Safe mode attivato"
        )
    
    async def lockdown(self, reason: str = "Lockdown manuale") -> dict:
        """Attiva lockdown totale"""
        return await self.set_emergency_level(
            EmergencyLevel.LOCKDOWN,
            reason
        )
    
    async def normalize(self) -> dict:
        """Torna a operatività normale"""
        if self._is_killed:
            return {"success": False, "error": "Eseguire revive() prima"}
        
        return await self.set_emergency_level(
            EmergencyLevel.NORMAL,
            "Ritorno a operatività normale"
        )
    
    def can_execute(self, action: str) -> dict:
        """
        Verifica se un'azione può essere eseguita al livello corrente
        """
        if self._is_killed:
            return {
                "allowed": False,
                "reason": "Kill-switch attivo",
                "level": "KILLED"
            }
        
        level = self._emergency_level
        permissions = self._level_permissions.get(level, {})
        
        # NORMAL: tutto permesso
        if permissions.get("all"):
            return {
                "allowed": True,
                "reason": "Operatività normale",
                "level": level.name
            }
        
        # Check whitelist
        allowed_actions = permissions.get("allow", [])
        if action in allowed_actions:
            return {
                "allowed": True,
                "reason": f"Azione permessa in {level.name}",
                "level": level.name
            }
        
        # LOCKDOWN: niente permesso
        if permissions.get("admin_only"):
            return {
                "allowed": False,
                "reason": "Lockdown attivo - richiede admin",
                "level": level.name
            }
        
        return {
            "allowed": False,
            "reason": f"Azione non permessa in {level.name}",
            "level": level.name,
            "allowed_actions": allowed_actions
        }
    
    async def _notify_emergency(self, event: str, details: str):
        """Notifica callbacks di emergenza"""
        for callback in self._emergency_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, details, self._emergency_level)
                else:
                    callback(event, details, self._emergency_level)
            except Exception:
                pass
    
    async def _schedule_auto_recovery(self, minutes: int):
        """Schedula auto-recovery a NORMAL"""
        if self._recovery_task:
            self._recovery_task.cancel()
        
        async def recovery():
            await asyncio.sleep(minutes * 60)
            if self._emergency_level == EmergencyLevel.CAUTION:
                await self.normalize()
        
        self._recovery_task = asyncio.create_task(recovery())
    
    def register_task(self, task: asyncio.Task):
        """Registra task per kill-switch"""
        self._active_tasks.append(task)
        # Cleanup task completati
        self._active_tasks = [t for t in self._active_tasks if not t.done()]
    
    def on_emergency(self, callback: Callable):
        """Registra callback per eventi emergenza"""
        self._emergency_callbacks.append(callback)
    
    def get_status(self) -> dict:
        """Stato completo del sistema di emergenza"""
        return {
            "emergency_level": self._emergency_level.name,
            "is_killed": self._is_killed,
            "is_emergency": self.is_emergency,
            "kill_timestamp": self._kill_timestamp.isoformat() if self._kill_timestamp else None,
            "kill_reason": self._kill_reason,
            "active_tasks": len([t for t in self._active_tasks if not t.done()]),
            "history_count": len(self._emergency_history)
        }
    
    def get_history(self, limit: int = 20) -> List[dict]:
        """Storico eventi emergenza"""
        return self._emergency_history[-limit:]
    
    # === Shortcuts ===
    
    async def panic(self, reason: str = "PANIC"):
        """Alias per kill_switch"""
        return await self.kill_switch(reason)
    
    async def stop(self):
        """Alias per kill_switch"""
        return await self.kill_switch("Stop richiesto")
    
    async def resume(self):
        """Alias per revive"""
        return await self.revive()
