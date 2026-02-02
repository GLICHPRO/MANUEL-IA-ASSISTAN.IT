"""
🤖 JARVIS CORE - Automator
Gestione automazioni, task schedulati e routine
"""

import asyncio
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import json


class TriggerType(Enum):
    """Tipi di trigger per automazioni"""
    TIME = "time"               # A orario specifico
    INTERVAL = "interval"       # Ogni X minuti/ore
    EVENT = "event"             # Su evento specifico
    CONDITION = "condition"     # Quando condizione è vera
    VOICE = "voice"             # Su comando vocale
    STARTUP = "startup"         # All'avvio del sistema


class AutomationStatus(Enum):
    """Stati di un'automazione"""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    RUNNING = "running"
    ERROR = "error"


class Automation:
    """Singola automazione"""
    
    def __init__(self, automation_id: str, name: str, trigger: dict, actions: list):
        self.id = automation_id
        self.name = name
        self.trigger = trigger
        self.actions = actions
        self.status = AutomationStatus.ACTIVE
        self.created_at = datetime.now()
        self.last_run = None
        self.run_count = 0
        self.error_count = 0
        self.last_error = None
        self.enabled = True
        
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "trigger": self.trigger,
            "actions": self.actions,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "enabled": self.enabled
        }


class Automator:
    """
    Gestisce automazioni e task schedulati
    """
    
    def __init__(self, executor=None):
        self.executor = executor
        self.automations: Dict[str, Automation] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.event_listeners: Dict[str, List[str]] = {}  # evento -> lista automation_id
        self.is_running = False
        self._scheduler_task = None
        
        # Routine predefinite
        self._init_default_routines()
    
    def _init_default_routines(self):
        """Inizializza routine predefinite"""
        # Routine mattutina
        self.create_automation(
            "morning_routine",
            "Routine Mattutina",
            {"type": TriggerType.TIME.value, "time": "08:00"},
            [
                {"action": "notify", "params": {"message": "Buongiorno! Ecco il riepilogo della giornata."}},
                {"action": "search_web", "params": {"query": "notizie oggi italia"}}
            ]
        )
        
        # Promemoria pausa
        self.create_automation(
            "break_reminder",
            "Promemoria Pausa",
            {"type": TriggerType.INTERVAL.value, "minutes": 60},
            [
                {"action": "notify", "params": {"message": "È ora di fare una pausa! 🧘"}}
            ]
        )
        
        # Disabilita di default
        self.automations["morning_routine"].enabled = False
        self.automations["break_reminder"].enabled = False
    
    def create_automation(self, automation_id: str, name: str, 
                          trigger: dict, actions: list) -> Automation:
        """Crea una nuova automazione"""
        automation = Automation(automation_id, name, trigger, actions)
        self.automations[automation_id] = automation
        
        # Registra listener per eventi
        if trigger.get("type") == TriggerType.EVENT.value:
            event_name = trigger.get("event")
            if event_name not in self.event_listeners:
                self.event_listeners[event_name] = []
            self.event_listeners[event_name].append(automation_id)
        
        return automation
    
    def update_automation(self, automation_id: str, **kwargs) -> Optional[Automation]:
        """Aggiorna un'automazione esistente"""
        if automation_id not in self.automations:
            return None
        
        automation = self.automations[automation_id]
        
        if "name" in kwargs:
            automation.name = kwargs["name"]
        if "trigger" in kwargs:
            automation.trigger = kwargs["trigger"]
        if "actions" in kwargs:
            automation.actions = kwargs["actions"]
        if "enabled" in kwargs:
            automation.enabled = kwargs["enabled"]
        
        return automation
    
    def delete_automation(self, automation_id: str) -> bool:
        """Elimina un'automazione"""
        if automation_id in self.automations:
            # Rimuovi dai listener eventi
            automation = self.automations[automation_id]
            if automation.trigger.get("type") == TriggerType.EVENT.value:
                event_name = automation.trigger.get("event")
                if event_name in self.event_listeners:
                    self.event_listeners[event_name].remove(automation_id)
            
            del self.automations[automation_id]
            return True
        return False
    
    def enable_automation(self, automation_id: str) -> bool:
        """Abilita un'automazione"""
        if automation_id in self.automations:
            self.automations[automation_id].enabled = True
            self.automations[automation_id].status = AutomationStatus.ACTIVE
            return True
        return False
    
    def disable_automation(self, automation_id: str) -> bool:
        """Disabilita un'automazione"""
        if automation_id in self.automations:
            self.automations[automation_id].enabled = False
            self.automations[automation_id].status = AutomationStatus.DISABLED
            return True
        return False
    
    async def run_automation(self, automation_id: str) -> dict:
        """Esegue manualmente un'automazione"""
        if automation_id not in self.automations:
            return {"success": False, "error": "Automazione non trovata"}
        
        automation = self.automations[automation_id]
        
        if not self.executor:
            return {"success": False, "error": "Executor non configurato"}
        
        automation.status = AutomationStatus.RUNNING
        results = []
        
        try:
            for action in automation.actions:
                result = await self.executor.execute(action)
                results.append(result)
                
                # Ferma se un'azione fallisce
                if not result.get("success"):
                    automation.error_count += 1
                    automation.last_error = result.get("error")
                    break
            
            automation.last_run = datetime.now()
            automation.run_count += 1
            automation.status = AutomationStatus.ACTIVE
            
            return {
                "success": True,
                "automation_id": automation_id,
                "results": results
            }
            
        except Exception as e:
            automation.status = AutomationStatus.ERROR
            automation.error_count += 1
            automation.last_error = str(e)
            return {"success": False, "error": str(e)}
    
    async def trigger_event(self, event_name: str, data: dict = None):
        """Attiva le automazioni legate a un evento"""
        if event_name not in self.event_listeners:
            return
        
        for automation_id in self.event_listeners[event_name]:
            if automation_id in self.automations:
                automation = self.automations[automation_id]
                if automation.enabled:
                    await self.run_automation(automation_id)
    
    async def start_scheduler(self):
        """Avvia lo scheduler per automazioni temporizzate"""
        self.is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop_scheduler(self):
        """Ferma lo scheduler"""
        self.is_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
    
    async def _scheduler_loop(self):
        """Loop principale dello scheduler"""
        last_check = {}
        
        while self.is_running:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            
            for automation_id, automation in self.automations.items():
                if not automation.enabled:
                    continue
                
                trigger = automation.trigger
                trigger_type = trigger.get("type")
                
                # Trigger a orario specifico
                if trigger_type == TriggerType.TIME.value:
                    target_time = trigger.get("time")
                    if current_time == target_time:
                        # Evita esecuzioni multiple nello stesso minuto
                        if last_check.get(automation_id) != current_time:
                            last_check[automation_id] = current_time
                            asyncio.create_task(self.run_automation(automation_id))
                
                # Trigger a intervallo
                elif trigger_type == TriggerType.INTERVAL.value:
                    minutes = trigger.get("minutes", 60)
                    if automation.last_run:
                        elapsed = (now - automation.last_run).total_seconds() / 60
                        if elapsed >= minutes:
                            asyncio.create_task(self.run_automation(automation_id))
                    else:
                        # Prima esecuzione
                        asyncio.create_task(self.run_automation(automation_id))
            
            # Check ogni 30 secondi
            await asyncio.sleep(30)
    
    def get_automation(self, automation_id: str) -> Optional[dict]:
        """Ottiene i dettagli di un'automazione"""
        if automation_id in self.automations:
            return self.automations[automation_id].to_dict()
        return None
    
    def list_automations(self, enabled_only: bool = False) -> List[dict]:
        """Lista tutte le automazioni"""
        automations = []
        for automation in self.automations.values():
            if enabled_only and not automation.enabled:
                continue
            automations.append(automation.to_dict())
        return automations
    
    def get_status(self) -> dict:
        """Stato del sistema di automazione"""
        return {
            "scheduler_running": self.is_running,
            "total_automations": len(self.automations),
            "active_automations": sum(1 for a in self.automations.values() if a.enabled),
            "event_listeners": {k: len(v) for k, v in self.event_listeners.items()}
        }
    
    # === Metodi helper per creazione rapida ===
    
    def schedule_at(self, automation_id: str, name: str, 
                    time_str: str, actions: list) -> Automation:
        """Crea automazione a orario specifico"""
        return self.create_automation(
            automation_id, name,
            {"type": TriggerType.TIME.value, "time": time_str},
            actions
        )
    
    def schedule_every(self, automation_id: str, name: str,
                       minutes: int, actions: list) -> Automation:
        """Crea automazione a intervallo"""
        return self.create_automation(
            automation_id, name,
            {"type": TriggerType.INTERVAL.value, "minutes": minutes},
            actions
        )
    
    def on_event(self, automation_id: str, name: str,
                 event: str, actions: list) -> Automation:
        """Crea automazione su evento"""
        return self.create_automation(
            automation_id, name,
            {"type": TriggerType.EVENT.value, "event": event},
            actions
        )
