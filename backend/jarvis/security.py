"""
🔒 JARVIS CORE - Security Manager
Gestione sicurezza, permessi e conferme
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum
import hashlib
import asyncio


class PermissionLevel(Enum):
    """Livelli di permesso per le azioni"""
    PUBLIC = 0          # Chiunque può eseguire
    USER = 1            # Richiede utente autenticato
    ELEVATED = 2        # Richiede conferma vocale
    ADMIN = 3           # Richiede conferma esplicita + PIN
    CRITICAL = 4        # Richiede doppia conferma + timeout


class ActionCategory(Enum):
    """Categorie di azioni per classificazione sicurezza"""
    INFO = "info"               # Solo lettura
    FILE_READ = "file_read"     # Lettura file
    FILE_WRITE = "file_write"   # Scrittura file
    FILE_DELETE = "file_delete" # Eliminazione file
    APP_OPEN = "app_open"       # Apertura applicazioni
    APP_CLOSE = "app_close"     # Chiusura applicazioni
    SYSTEM = "system"           # Comandi sistema
    NETWORK = "network"         # Operazioni rete
    CRITICAL = "critical"       # Shutdown, restart, etc.


class SecurityManager:
    """
    Gestisce sicurezza e permessi per Jarvis Core
    """
    
    def __init__(self):
        self.is_authenticated = False
        self.current_user = None
        self.session_start = None
        self.session_timeout = timedelta(hours=8)
        
        # Mapping azioni -> livello permesso richiesto
        self.action_permissions = {
            # Azioni sicure (PUBLIC)
            "notify": PermissionLevel.PUBLIC,
            "search_web": PermissionLevel.PUBLIC,
            "open_url": PermissionLevel.PUBLIC,
            
            # Azioni utente (USER)
            "open_app": PermissionLevel.USER,
            "open_file": PermissionLevel.USER,
            "set_volume": PermissionLevel.USER,
            "mute": PermissionLevel.USER,
            "copy_to_clipboard": PermissionLevel.USER,
            
            # Azioni elevate (ELEVATED)
            "create_file": PermissionLevel.ELEVATED,
            "copy_file": PermissionLevel.ELEVATED,
            "move_file": PermissionLevel.ELEVATED,
            "run_command": PermissionLevel.ELEVATED,
            "close_app": PermissionLevel.ELEVATED,
            
            # Azioni admin (ADMIN)
            "delete_file": PermissionLevel.ADMIN,
            "lock": PermissionLevel.ADMIN,
            
            # Azioni critiche (CRITICAL)
            "shutdown": PermissionLevel.CRITICAL,
            "restart": PermissionLevel.CRITICAL,
            "sleep": PermissionLevel.CRITICAL,
        }
        
        # Log delle azioni
        self.action_log = []
        
        # Azioni in attesa di conferma
        self.pending_confirmations = {}
        
        # PIN per azioni admin (default, da cambiare)
        self._admin_pin_hash = self._hash_pin("1234")
        
        # Callback per richieste di conferma
        self.confirmation_callback: Optional[Callable] = None
        
    def _hash_pin(self, pin: str) -> str:
        """Hash del PIN per sicurezza"""
        return hashlib.sha256(pin.encode()).hexdigest()
    
    def set_admin_pin(self, old_pin: str, new_pin: str) -> bool:
        """Cambia il PIN admin"""
        if self._hash_pin(old_pin) != self._admin_pin_hash:
            return False
        self._admin_pin_hash = self._hash_pin(new_pin)
        return True
    
    def verify_pin(self, pin: str) -> bool:
        """Verifica il PIN admin"""
        return self._hash_pin(pin) == self._admin_pin_hash
    
    def authenticate(self, user: str = "default") -> bool:
        """Autentica l'utente"""
        self.is_authenticated = True
        self.current_user = user
        self.session_start = datetime.now()
        self._log_action("authenticate", {"user": user}, True)
        return True
    
    def logout(self):
        """Logout utente"""
        self._log_action("logout", {"user": self.current_user}, True)
        self.is_authenticated = False
        self.current_user = None
        self.session_start = None
    
    def is_session_valid(self) -> bool:
        """Verifica se la sessione è ancora valida"""
        if not self.is_authenticated or not self.session_start:
            return False
        return datetime.now() - self.session_start < self.session_timeout
    
    def get_required_permission(self, action: str) -> PermissionLevel:
        """Ottiene il livello di permesso richiesto per un'azione"""
        return self.action_permissions.get(action, PermissionLevel.ELEVATED)
    
    async def check_permission(self, action: str, params: dict = None) -> dict:
        """
        Verifica se l'azione può essere eseguita
        
        Returns:
            {
                "allowed": bool,
                "reason": str,
                "requires_confirmation": bool,
                "confirmation_id": str (se richiede conferma)
            }
        """
        required_level = self.get_required_permission(action)
        
        result = {
            "allowed": False,
            "reason": "",
            "requires_confirmation": False,
            "confirmation_id": None,
            "permission_level": required_level.name
        }
        
        # PUBLIC: sempre permesso
        if required_level == PermissionLevel.PUBLIC:
            result["allowed"] = True
            result["reason"] = "Azione pubblica"
            return result
        
        # Verifica sessione per livelli superiori
        if not self.is_session_valid():
            result["reason"] = "Sessione non valida o scaduta"
            return result
        
        # USER: basta essere autenticati
        if required_level == PermissionLevel.USER:
            result["allowed"] = True
            result["reason"] = "Utente autenticato"
            return result
        
        # ELEVATED: richiede conferma vocale
        if required_level == PermissionLevel.ELEVATED:
            conf_id = await self._request_confirmation(action, params, "vocale")
            result["requires_confirmation"] = True
            result["confirmation_id"] = conf_id
            result["reason"] = "Richiede conferma vocale"
            return result
        
        # ADMIN: richiede conferma + PIN
        if required_level == PermissionLevel.ADMIN:
            conf_id = await self._request_confirmation(action, params, "pin")
            result["requires_confirmation"] = True
            result["confirmation_id"] = conf_id
            result["reason"] = "Richiede conferma con PIN"
            return result
        
        # CRITICAL: doppia conferma + timeout
        if required_level == PermissionLevel.CRITICAL:
            conf_id = await self._request_confirmation(action, params, "critical")
            result["requires_confirmation"] = True
            result["confirmation_id"] = conf_id
            result["reason"] = "Azione critica - richiede doppia conferma"
            return result
        
        return result
    
    async def _request_confirmation(self, action: str, params: dict, conf_type: str) -> str:
        """Crea una richiesta di conferma"""
        conf_id = f"conf_{datetime.now().timestamp()}_{action}"
        
        self.pending_confirmations[conf_id] = {
            "action": action,
            "params": params,
            "type": conf_type,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=2),
            "confirmed": False
        }
        
        # Notifica callback se presente
        if self.confirmation_callback:
            await self.confirmation_callback(conf_id, action, conf_type)
        
        return conf_id
    
    async def confirm_action(self, confirmation_id: str, pin: str = None) -> dict:
        """Conferma un'azione in attesa"""
        if confirmation_id not in self.pending_confirmations:
            return {"success": False, "error": "Conferma non trovata"}
        
        conf = self.pending_confirmations[confirmation_id]
        
        # Verifica scadenza
        if datetime.now() > conf["expires_at"]:
            del self.pending_confirmations[confirmation_id]
            return {"success": False, "error": "Conferma scaduta"}
        
        # Verifica PIN se richiesto
        if conf["type"] in ["pin", "critical"] and pin:
            if not self.verify_pin(pin):
                return {"success": False, "error": "PIN non valido"}
        
        conf["confirmed"] = True
        
        return {
            "success": True,
            "action": conf["action"],
            "params": conf["params"]
        }
    
    def cancel_confirmation(self, confirmation_id: str) -> bool:
        """Cancella una richiesta di conferma"""
        if confirmation_id in self.pending_confirmations:
            del self.pending_confirmations[confirmation_id]
            return True
        return False
    
    def _log_action(self, action: str, params: dict, success: bool):
        """Logga un'azione"""
        self.action_log.append({
            "timestamp": datetime.now().isoformat(),
            "user": self.current_user,
            "action": action,
            "params": params,
            "success": success
        })
        
        # Mantieni solo gli ultimi 1000 log
        if len(self.action_log) > 1000:
            self.action_log = self.action_log[-1000:]
    
    def get_action_log(self, limit: int = 50) -> List[dict]:
        """Ottiene gli ultimi log"""
        return self.action_log[-limit:]
    
    def get_pending_confirmations(self) -> List[dict]:
        """Ottiene le conferme in attesa"""
        now = datetime.now()
        # Rimuovi scadute
        expired = [k for k, v in self.pending_confirmations.items() 
                   if now > v["expires_at"]]
        for k in expired:
            del self.pending_confirmations[k]
        
        return [
            {"id": k, **v} 
            for k, v in self.pending_confirmations.items()
        ]
    
    def set_action_permission(self, action: str, level: PermissionLevel):
        """Imposta il livello di permesso per un'azione"""
        self.action_permissions[action] = level
    
    def get_security_status(self) -> dict:
        """Stato completo della sicurezza"""
        return {
            "authenticated": self.is_authenticated,
            "user": self.current_user,
            "session_valid": self.is_session_valid(),
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "pending_confirmations": len(self.pending_confirmations),
            "recent_actions": len(self.action_log)
        }
