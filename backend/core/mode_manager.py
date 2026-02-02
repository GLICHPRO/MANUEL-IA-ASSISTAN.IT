"""
🎮 JARVIS CORE - Mode Manager
Gestione modalità operative: PASSIVE, COPILOT, PILOT
Gestione livelli risposta: NORMAL, ADVANCED
"""

from enum import Enum
from datetime import datetime
from typing import Optional, Callable, Dict, Any
import asyncio


class OperatingMode(Enum):
    """Modalità operative del sistema (AUTONOMIA)"""
    PASSIVE = "passive"       # Solo risponde, zero azioni
    COPILOT = "copilot"       # Suggerisce, richiede conferma
    PILOT = "pilot"           # Controllo totale, routine, automazioni, hands-free
    EXECUTIVE = "executive"   # Jarvis: supervisione globale, orchestrazione, fallback


class ResponseLevel(Enum):
    """Livello di dettaglio delle risposte (COMPLESSITÀ)"""
    NORMAL = "normal"       # Risposte rapide, amichevoli, base
    ADVANCED = "advanced"   # Analisi complesse, tecnico, dettagliato


class ModeManager:
    """
    Gestisce le modalità operative di Jarvis Core
    
    PASSIVE (0% autonomia):
        - Risponde solo a domande
        - Nessuna azione eseguita
        - Solo analisi e suggerimenti
    
    COPILOT (50% autonomia):
        - Suggerisce azioni
        - Richiede conferma prima di eseguire
        - Azioni reversibili auto-approvate
    
    PILOT (100% autonomia):
        - Controllo completo
        - Esecuzione routine e automazioni
        - Hands-free totale
        - Esegue senza conferma
    
    EXECUTIVE (Jarvis Mode - 100%+ autonomia):
        - Supervisione globale del sistema
        - Decisione finale su tutto
        - Orchestrazione di tutti i moduli
        - Gestione scenari complessi
        - Sistema di fallback automatico
        - Può sovrascrivere decisioni PILOT
    """
    
    def __init__(self):
        self._current_mode = OperatingMode.COPILOT  # Default sicuro
        self._response_level = ResponseLevel.NORMAL  # Default normale
        self._mode_history = []
        self._mode_changed_at = datetime.now()
        self._pilot_session_start = None
        self._pilot_timeout_minutes = 30  # Auto-torna a COPILOT dopo 30 min
        self._pilot_task = None
        
        # Callbacks per notifiche cambio modalità
        self._mode_change_callbacks: list[Callable] = []
        
        # Configurazione Response Level
        self._response_config = {
            ResponseLevel.NORMAL: {
                "style": "friendly",           # Stile amichevole
                "detail_level": "basic",       # Dettaglio base
                "technical_terms": False,      # No termini tecnici
                "show_reasoning": False,       # Non mostra ragionamento
                "show_calculations": False,    # Non mostra calcoli
                "max_response_length": 200,    # Risposta breve
                "include_suggestions": True,   # Include suggerimenti semplici
                "include_alternatives": False, # No alternative
                "emoji_usage": "high"          # Uso alto di emoji
            },
            ResponseLevel.ADVANCED: {
                "style": "technical",          # Stile tecnico
                "detail_level": "comprehensive", # Dettaglio completo
                "technical_terms": True,       # Usa termini tecnici
                "show_reasoning": True,        # Mostra ragionamento
                "show_calculations": True,     # Mostra calcoli
                "max_response_length": 1000,   # Risposta dettagliata
                "include_suggestions": True,   # Include suggerimenti
                "include_alternatives": True,  # Mostra alternative
                "emoji_usage": "low"           # Uso basso di emoji
            }
        }
        
        # Permessi per modalità
        self._mode_permissions = {
            OperatingMode.PASSIVE: {
                "can_execute": False,
                "can_suggest": True,
                "can_analyze": True,
                "requires_confirmation": True,
                "auto_approve_safe": False
            },
            OperatingMode.COPILOT: {
                "can_execute": True,
                "can_suggest": True,
                "can_analyze": True,
                "requires_confirmation": True,
                "auto_approve_safe": True  # Azioni sicure auto-approvate
            },
            OperatingMode.PILOT: {
                "can_execute": True,
                "can_suggest": True,
                "can_analyze": True,
                "requires_confirmation": False,
                "auto_approve_safe": True,
                "can_run_routines": True,
                "can_run_automations": True,
                "hands_free": True
            },
            OperatingMode.EXECUTIVE: {
                "can_execute": True,
                "can_suggest": True,
                "can_analyze": True,
                "requires_confirmation": False,
                "auto_approve_safe": True,
                "can_run_routines": True,
                "can_run_automations": True,
                "hands_free": True,
                "can_orchestrate": True,         # Orchestrazione moduli
                "can_override_pilot": True,      # Può sovrascrivere PILOT
                "global_supervision": True,      # Supervisione globale
                "fallback_management": True,     # Gestione fallback
                "complex_scenarios": True        # Scenari complessi
            }
        }
        
        # Azioni considerate "sicure" (auto-approvate in COPILOT)
        self.safe_actions = {
            "notify", "search_web", "open_url", 
            "set_volume", "mute", "open_app"
        }
        
        # Azioni che richiedono SEMPRE conferma (anche in PILOT, ma non in EXECUTIVE)
        self.always_confirm_actions = {
            "shutdown", "restart", "delete_file", 
            "format_disk", "uninstall"
        }
        
        # Azioni che EXECUTIVE può fare senza conferma
        self.executive_override_actions = {
            "shutdown", "restart", "emergency_stop",
            "module_restart", "fallback_activate"
        }
    
    @property
    def mode(self) -> OperatingMode:
        """Modalità corrente"""
        return self._current_mode
    
    @property
    def mode_name(self) -> str:
        """Nome modalità corrente"""
        return self._current_mode.value.upper()
    
    async def set_mode(self, mode: OperatingMode, reason: str = None) -> dict:
        """
        Cambia modalità operativa
        
        Args:
            mode: Nuova modalità
            reason: Motivo del cambio
            
        Returns:
            Risultato del cambio
        """
        old_mode = self._current_mode
        
        # Log cambio
        self._mode_history.append({
            "from": old_mode.value,
            "to": mode.value,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        self._current_mode = mode
        self._mode_changed_at = datetime.now()
        
        # Se entra in PILOT, avvia timer auto-uscita
        if mode == OperatingMode.PILOT:
            self._pilot_session_start = datetime.now()
            await self._start_pilot_timeout()
        else:
            self._pilot_session_start = None
            if self._pilot_task:
                self._pilot_task.cancel()
        
        # Notifica callbacks
        for callback in self._mode_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_mode, mode)
                else:
                    callback(old_mode, mode)
            except Exception:
                pass
        
        return {
            "success": True,
            "previous_mode": old_mode.value,
            "current_mode": mode.value,
            "changed_at": self._mode_changed_at.isoformat(),
            "reason": reason
        }
    
    async def _start_pilot_timeout(self):
        """Avvia timer per auto-uscita da PILOT mode"""
        if self._pilot_task:
            self._pilot_task.cancel()
        
        async def timeout_handler():
            await asyncio.sleep(self._pilot_timeout_minutes * 60)
            if self._current_mode == OperatingMode.PILOT:
                await self.set_mode(
                    OperatingMode.COPILOT, 
                    f"Auto-timeout dopo {self._pilot_timeout_minutes} minuti"
                )
        
        self._pilot_task = asyncio.create_task(timeout_handler())
    
    def can_execute(self, action: str) -> dict:
        """
        Verifica se un'azione può essere eseguita nella modalità corrente
        
        Returns:
            {
                "allowed": bool,
                "requires_confirmation": bool,
                "reason": str
            }
        """
        permissions = self._mode_permissions[self._current_mode]
        
        result = {
            "allowed": False,
            "requires_confirmation": True,
            "reason": "",
            "mode": self._current_mode.value
        }
        
        # PASSIVE: mai eseguire
        if self._current_mode == OperatingMode.PASSIVE:
            result["reason"] = "Modalità PASSIVE - azioni disabilitate"
            return result
        
        # EXECUTIVE: può fare tutto, incluso override
        if self._current_mode == OperatingMode.EXECUTIVE:
            result["allowed"] = True
            result["requires_confirmation"] = False
            result["reason"] = "Modalità EXECUTIVE - Jarvis ha il controllo totale"
            result["can_override"] = True
            return result
        
        # Azioni critiche: conferma in PILOT, ma non in EXECUTIVE
        if action in self.always_confirm_actions:
            result["allowed"] = True
            result["requires_confirmation"] = True
            result["reason"] = "Azione critica - conferma obbligatoria"
            return result
        
        # PILOT: esegui tutto tranne azioni critiche
        if self._current_mode == OperatingMode.PILOT:
            result["allowed"] = True
            result["requires_confirmation"] = False
            result["reason"] = "Modalità PILOT - autonomia hands-free"
            return result
        
        # COPILOT: dipende se azione sicura
        if self._current_mode == OperatingMode.COPILOT:
            if action in self.safe_actions:
                result["allowed"] = True
                result["requires_confirmation"] = False
                result["reason"] = "Azione sicura - auto-approvata"
            else:
                result["allowed"] = True
                result["requires_confirmation"] = True
                result["reason"] = "Richiede conferma utente"
            return result
        
        return result
    
    def get_mode_info(self) -> dict:
        """Informazioni complete sulla modalità corrente"""
        pilot_remaining = None
        if self._current_mode == OperatingMode.PILOT and self._pilot_session_start:
            elapsed = (datetime.now() - self._pilot_session_start).total_seconds() / 60
            pilot_remaining = max(0, self._pilot_timeout_minutes - elapsed)
        
        return {
            "current_mode": self._current_mode.value,
            "mode_description": self._get_mode_description(),
            "permissions": self._mode_permissions[self._current_mode],
            "changed_at": self._mode_changed_at.isoformat(),
            "pilot_remaining_minutes": pilot_remaining,
            "safe_actions": list(self.safe_actions),
            "always_confirm": list(self.always_confirm_actions)
        }
    
    def _get_mode_description(self) -> str:
        """Descrizione testuale della modalità"""
        descriptions = {
            OperatingMode.PASSIVE: "🔇 Modalità Passiva - Solo analisi, nessuna azione",
            OperatingMode.COPILOT: "🤝 Modalità Copilot - Suggerisce e chiede conferma",
            OperatingMode.PILOT: "🚀 Modalità Pilot - Controllo totale, routine, hands-free",
            OperatingMode.EXECUTIVE: "👔 Modalità Executive - Jarvis: supervisione globale e orchestrazione"
        }
        return descriptions.get(self._current_mode, "Sconosciuta")
    
    def get_history(self, limit: int = 20) -> list:
        """Storico cambi modalità"""
        return self._mode_history[-limit:]
    
    def on_mode_change(self, callback: Callable):
        """Registra callback per cambio modalità"""
        self._mode_change_callbacks.append(callback)
    
    def set_pilot_timeout(self, minutes: int):
        """Imposta timeout per auto-uscita da PILOT"""
        self._pilot_timeout_minutes = max(5, min(minutes, 480))  # 5 min - 8 ore
    
    def extend_pilot_session(self, additional_minutes: int = 30):
        """Estende sessione PILOT"""
        if self._current_mode == OperatingMode.PILOT:
            self._pilot_timeout_minutes += additional_minutes
            return True
        return False
    
    # === Shortcut per cambio rapido ===
    
    async def go_passive(self, reason: str = "Richiesta utente"):
        """Passa a modalità PASSIVE"""
        return await self.set_mode(OperatingMode.PASSIVE, reason)
    
    async def go_copilot(self, reason: str = "Richiesta utente"):
        """Passa a modalità COPILOT"""
        return await self.set_mode(OperatingMode.COPILOT, reason)
    
    async def go_pilot(self, reason: str = "Richiesta utente"):
        """Passa a modalità PILOT"""
        return await self.set_mode(OperatingMode.PILOT, reason)
    
    async def go_executive(self, reason: str = "Richiesta utente"):
        """Passa a modalità EXECUTIVE (Jarvis Mode)"""
        return await self.set_mode(OperatingMode.EXECUTIVE, reason)
    
    def is_passive(self) -> bool:
        return self._current_mode == OperatingMode.PASSIVE
    
    def is_copilot(self) -> bool:
        return self._current_mode == OperatingMode.COPILOT
    
    def is_pilot(self) -> bool:
        return self._current_mode == OperatingMode.PILOT
    
    def is_executive(self) -> bool:
        return self._current_mode == OperatingMode.EXECUTIVE
    
    def is_hands_free(self) -> bool:
        """Verifica se siamo in modalità hands-free (PILOT o EXECUTIVE)"""
        return self._current_mode in [OperatingMode.PILOT, OperatingMode.EXECUTIVE]
    
    def can_orchestrate(self) -> bool:
        """Verifica se può orchestrare moduli (solo EXECUTIVE)"""
        return self._current_mode == OperatingMode.EXECUTIVE
    
    # === Response Level Management ===
    
    @property
    def response_level(self) -> ResponseLevel:
        """Livello risposta corrente"""
        return self._response_level
    
    @property
    def response_level_name(self) -> str:
        """Nome livello risposta"""
        return self._response_level.value.upper()
    
    def set_response_level(self, level: ResponseLevel, reason: str = None) -> dict:
        """Cambia livello di risposta"""
        old_level = self._response_level
        self._response_level = level
        
        self._mode_history.append({
            "type": "response_level",
            "from": old_level.value,
            "to": level.value,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "previous_level": old_level.value,
            "current_level": level.value,
            "reason": reason
        }
    
    def get_response_config(self) -> dict:
        """Ottiene configurazione risposta per livello corrente"""
        return self._response_config[self._response_level].copy()
    
    def go_normal(self, reason: str = "Richiesta utente") -> dict:
        """Passa a livello NORMAL"""
        return self.set_response_level(ResponseLevel.NORMAL, reason)
    
    def go_advanced(self, reason: str = "Richiesta utente") -> dict:
        """Passa a livello ADVANCED"""
        return self.set_response_level(ResponseLevel.ADVANCED, reason)
    
    def is_normal(self) -> bool:
        return self._response_level == ResponseLevel.NORMAL
    
    def is_advanced(self) -> bool:
        return self._response_level == ResponseLevel.ADVANCED
    
    def get_full_status(self) -> dict:
        """Stato completo: modalità + livello risposta"""
        return {
            "operating_mode": {
                "current": self._current_mode.value,
                "description": self._get_mode_description()
            },
            "response_level": {
                "current": self._response_level.value,
                "config": self.get_response_config()
            },
            "combined_description": self._get_combined_description()
        }
    
    def _get_combined_description(self) -> str:
        """Descrizione combinata modalità + livello"""
        mode_emoji = {
            OperatingMode.PASSIVE: "🔇",
            OperatingMode.COPILOT: "🤝",
            OperatingMode.PILOT: "🚀"
        }
        level_emoji = {
            ResponseLevel.NORMAL: "💬",
            ResponseLevel.ADVANCED: "🔬"
        }
        return (
            f"{mode_emoji.get(self._current_mode, '')} {self._current_mode.value.upper()} + "
            f"{level_emoji.get(self._response_level, '')} {self._response_level.value.upper()}"
        )
