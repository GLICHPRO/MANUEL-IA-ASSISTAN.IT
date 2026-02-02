"""
🎬 JARVIS CORE - Macro Recorder
Registra e riproduce sequenze di azioni
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid


class RecordingState(Enum):
    """Stati della registrazione"""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


class PlaybackState(Enum):
    """Stati della riproduzione"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class MacroAction:
    """Singola azione registrata"""
    id: str
    action_type: str
    params: dict
    timestamp: datetime
    delay_ms: int = 0  # Ritardo dall'azione precedente
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_type": self.action_type,
            "params": self.params,
            "timestamp": self.timestamp.isoformat(),
            "delay_ms": self.delay_ms
        }


@dataclass
class Macro:
    """Macro registrata"""
    id: str
    name: str
    description: str
    actions: List[MacroAction]
    created_at: datetime
    last_played: Optional[datetime] = None
    play_count: int = 0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "actions": [a.to_dict() for a in self.actions],
            "action_count": len(self.actions),
            "created_at": self.created_at.isoformat(),
            "last_played": self.last_played.isoformat() if self.last_played else None,
            "play_count": self.play_count,
            "tags": self.tags,
            "duration_ms": self._calculate_duration()
        }
    
    def _calculate_duration(self) -> int:
        """Calcola durata totale della macro in ms"""
        return sum(a.delay_ms for a in self.actions)


class MacroRecorder:
    """
    Registratore e riproduttore di macro
    
    Features:
    - Registrazione azioni in tempo reale
    - Riproduzione con timing originale o velocizzata
    - Modifica e composizione macro
    - Loop e ripetizioni
    - Trigger per avvio automatico
    """
    
    def __init__(self, executor=None):
        self.executor = executor
        self.macros: Dict[str, Macro] = {}
        self.recording_state = RecordingState.IDLE
        self.playback_state = PlaybackState.STOPPED
        
        # Stato registrazione corrente
        self._recording_buffer: List[MacroAction] = []
        self._recording_start: Optional[datetime] = None
        self._last_action_time: Optional[datetime] = None
        
        # Stato riproduzione corrente
        self._current_playback: Optional[str] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._playback_speed: float = 1.0
        
        # Trigger configurati
        self.triggers: Dict[str, dict] = {}  # macro_id -> trigger_config
    
    # ============================================
    # RECORDING
    # ============================================
    
    def start_recording(self, name: str = None) -> str:
        """Inizia a registrare una nuova macro"""
        if self.recording_state == RecordingState.RECORDING:
            return "already_recording"
        
        self.recording_state = RecordingState.RECORDING
        self._recording_buffer = []
        self._recording_start = datetime.now()
        self._last_action_time = self._recording_start
        
        return "recording_started"
    
    def pause_recording(self):
        """Mette in pausa la registrazione"""
        if self.recording_state == RecordingState.RECORDING:
            self.recording_state = RecordingState.PAUSED
            return True
        return False
    
    def resume_recording(self):
        """Riprende la registrazione"""
        if self.recording_state == RecordingState.PAUSED:
            self.recording_state = RecordingState.RECORDING
            self._last_action_time = datetime.now()
            return True
        return False
    
    def record_action(self, action_type: str, params: dict = None):
        """Registra un'azione durante la registrazione"""
        if self.recording_state != RecordingState.RECORDING:
            return False
        
        now = datetime.now()
        delay_ms = int((now - self._last_action_time).total_seconds() * 1000)
        
        action = MacroAction(
            id=f"action_{len(self._recording_buffer)}",
            action_type=action_type,
            params=params or {},
            timestamp=now,
            delay_ms=delay_ms
        )
        
        self._recording_buffer.append(action)
        self._last_action_time = now
        
        return True
    
    def stop_recording(self, name: str, description: str = "") -> Optional[Macro]:
        """Ferma la registrazione e salva la macro"""
        if self.recording_state == RecordingState.IDLE:
            return None
        
        if not self._recording_buffer:
            self.recording_state = RecordingState.IDLE
            return None
        
        macro_id = f"macro_{uuid.uuid4().hex[:8]}"
        
        macro = Macro(
            id=macro_id,
            name=name,
            description=description,
            actions=self._recording_buffer.copy(),
            created_at=datetime.now()
        )
        
        self.macros[macro_id] = macro
        
        # Reset stato
        self.recording_state = RecordingState.IDLE
        self._recording_buffer = []
        self._recording_start = None
        self._last_action_time = None
        
        return macro
    
    def cancel_recording(self):
        """Cancella la registrazione in corso"""
        self.recording_state = RecordingState.IDLE
        self._recording_buffer = []
        self._recording_start = None
        self._last_action_time = None
    
    def get_recording_status(self) -> dict:
        """Ottiene lo stato della registrazione"""
        return {
            "state": self.recording_state.value,
            "actions_recorded": len(self._recording_buffer),
            "duration_ms": int((datetime.now() - self._recording_start).total_seconds() * 1000) if self._recording_start else 0
        }
    
    # ============================================
    # PLAYBACK
    # ============================================
    
    async def play(self, macro_id: str, speed: float = 1.0, 
                   loop: bool = False, loop_count: int = 1) -> dict:
        """Riproduce una macro"""
        if macro_id not in self.macros:
            return {"success": False, "error": "Macro non trovata"}
        
        if self.playback_state == PlaybackState.PLAYING:
            return {"success": False, "error": "Riproduzione già in corso"}
        
        macro = self.macros[macro_id]
        self._current_playback = macro_id
        self._playback_speed = speed
        self.playback_state = PlaybackState.PLAYING
        
        # Avvia riproduzione in background
        self._playback_task = asyncio.create_task(
            self._playback_loop(macro, loop, loop_count)
        )
        
        return {
            "success": True,
            "macro_id": macro_id,
            "actions": len(macro.actions),
            "estimated_duration_ms": int(macro._calculate_duration() / speed)
        }
    
    async def _playback_loop(self, macro: Macro, loop: bool, loop_count: int):
        """Loop di riproduzione"""
        iterations = 0
        
        try:
            while True:
                for action in macro.actions:
                    if self.playback_state == PlaybackState.STOPPED:
                        return
                    
                    while self.playback_state == PlaybackState.PAUSED:
                        await asyncio.sleep(0.1)
                    
                    # Applica delay
                    delay = action.delay_ms / 1000 / self._playback_speed
                    if delay > 0:
                        await asyncio.sleep(delay)
                    
                    # Esegui azione
                    await self._execute_action(action)
                
                iterations += 1
                macro.play_count += 1
                macro.last_played = datetime.now()
                
                if not loop or (loop_count > 0 and iterations >= loop_count):
                    break
        
        finally:
            self.playback_state = PlaybackState.STOPPED
            self._current_playback = None
    
    async def _execute_action(self, action: MacroAction):
        """Esegue una singola azione"""
        if self.executor:
            await self.executor.execute({
                "type": action.action_type,
                **action.params
            })
        else:
            # Simula esecuzione
            print(f"[MACRO] Eseguo: {action.action_type} - {action.params}")
    
    def pause_playback(self) -> bool:
        """Mette in pausa la riproduzione"""
        if self.playback_state == PlaybackState.PLAYING:
            self.playback_state = PlaybackState.PAUSED
            return True
        return False
    
    def resume_playback(self) -> bool:
        """Riprende la riproduzione"""
        if self.playback_state == PlaybackState.PAUSED:
            self.playback_state = PlaybackState.PLAYING
            return True
        return False
    
    def stop_playback(self) -> bool:
        """Ferma la riproduzione"""
        if self.playback_state != PlaybackState.STOPPED:
            self.playback_state = PlaybackState.STOPPED
            if self._playback_task:
                self._playback_task.cancel()
            return True
        return False
    
    def get_playback_status(self) -> dict:
        """Ottiene lo stato della riproduzione"""
        return {
            "state": self.playback_state.value,
            "current_macro": self._current_playback,
            "speed": self._playback_speed
        }
    
    # ============================================
    # MACRO MANAGEMENT
    # ============================================
    
    def create_macro(self, name: str, actions: List[dict], 
                     description: str = "") -> Macro:
        """Crea una macro da una lista di azioni"""
        macro_id = f"macro_{uuid.uuid4().hex[:8]}"
        
        macro_actions = []
        for i, action in enumerate(actions):
            macro_actions.append(MacroAction(
                id=f"action_{i}",
                action_type=action.get("type", "unknown"),
                params=action.get("params", {}),
                timestamp=datetime.now(),
                delay_ms=action.get("delay_ms", 500)
            ))
        
        macro = Macro(
            id=macro_id,
            name=name,
            description=description,
            actions=macro_actions,
            created_at=datetime.now()
        )
        
        self.macros[macro_id] = macro
        return macro
    
    def update_macro(self, macro_id: str, **kwargs) -> Optional[Macro]:
        """Aggiorna una macro esistente"""
        if macro_id not in self.macros:
            return None
        
        macro = self.macros[macro_id]
        
        if "name" in kwargs:
            macro.name = kwargs["name"]
        if "description" in kwargs:
            macro.description = kwargs["description"]
        if "tags" in kwargs:
            macro.tags = kwargs["tags"]
        
        return macro
    
    def delete_macro(self, macro_id: str) -> bool:
        """Elimina una macro"""
        if macro_id in self.macros:
            # Rimuovi anche trigger associati
            if macro_id in self.triggers:
                del self.triggers[macro_id]
            del self.macros[macro_id]
            return True
        return False
    
    def duplicate_macro(self, macro_id: str, new_name: str) -> Optional[Macro]:
        """Duplica una macro"""
        if macro_id not in self.macros:
            return None
        
        original = self.macros[macro_id]
        new_id = f"macro_{uuid.uuid4().hex[:8]}"
        
        new_macro = Macro(
            id=new_id,
            name=new_name,
            description=f"Copia di {original.name}",
            actions=original.actions.copy(),
            created_at=datetime.now(),
            tags=original.tags.copy()
        )
        
        self.macros[new_id] = new_macro
        return new_macro
    
    def merge_macros(self, macro_ids: List[str], new_name: str) -> Optional[Macro]:
        """Unisce più macro in una sola"""
        all_actions = []
        
        for macro_id in macro_ids:
            if macro_id not in self.macros:
                return None
            all_actions.extend(self.macros[macro_id].actions)
        
        new_id = f"macro_{uuid.uuid4().hex[:8]}"
        
        merged = Macro(
            id=new_id,
            name=new_name,
            description=f"Macro unita da {len(macro_ids)} macro",
            actions=all_actions,
            created_at=datetime.now()
        )
        
        self.macros[new_id] = merged
        return merged
    
    # ============================================
    # MACRO EDITING
    # ============================================
    
    def add_action_to_macro(self, macro_id: str, action_type: str,
                            params: dict, delay_ms: int = 500,
                            position: int = -1) -> bool:
        """Aggiunge un'azione a una macro esistente"""
        if macro_id not in self.macros:
            return False
        
        macro = self.macros[macro_id]
        
        action = MacroAction(
            id=f"action_{len(macro.actions)}",
            action_type=action_type,
            params=params,
            timestamp=datetime.now(),
            delay_ms=delay_ms
        )
        
        if position < 0 or position >= len(macro.actions):
            macro.actions.append(action)
        else:
            macro.actions.insert(position, action)
        
        return True
    
    def remove_action_from_macro(self, macro_id: str, action_index: int) -> bool:
        """Rimuove un'azione da una macro"""
        if macro_id not in self.macros:
            return False
        
        macro = self.macros[macro_id]
        
        if 0 <= action_index < len(macro.actions):
            macro.actions.pop(action_index)
            return True
        return False
    
    def update_action_delay(self, macro_id: str, action_index: int,
                            new_delay_ms: int) -> bool:
        """Aggiorna il delay di un'azione"""
        if macro_id not in self.macros:
            return False
        
        macro = self.macros[macro_id]
        
        if 0 <= action_index < len(macro.actions):
            macro.actions[action_index].delay_ms = new_delay_ms
            return True
        return False
    
    # ============================================
    # TRIGGERS
    # ============================================
    
    def set_trigger(self, macro_id: str, trigger_type: str, 
                    trigger_config: dict):
        """Imposta un trigger per avvio automatico della macro"""
        if macro_id not in self.macros:
            return False
        
        self.triggers[macro_id] = {
            "type": trigger_type,  # "hotkey", "time", "event"
            "config": trigger_config,
            "enabled": True
        }
        return True
    
    def remove_trigger(self, macro_id: str) -> bool:
        """Rimuove il trigger di una macro"""
        if macro_id in self.triggers:
            del self.triggers[macro_id]
            return True
        return False
    
    async def check_triggers(self, event: dict):
        """Controlla se un evento attiva qualche trigger"""
        event_type = event.get("type")
        
        for macro_id, trigger in self.triggers.items():
            if not trigger.get("enabled"):
                continue
            
            if trigger["type"] == "event" and trigger["config"].get("event") == event_type:
                await self.play(macro_id)
    
    # ============================================
    # QUERY METHODS
    # ============================================
    
    def get_macro(self, macro_id: str) -> Optional[dict]:
        """Ottiene una macro per ID"""
        if macro_id in self.macros:
            return self.macros[macro_id].to_dict()
        return None
    
    def list_macros(self, tag: str = None) -> List[dict]:
        """Lista tutte le macro"""
        macros = []
        for macro in self.macros.values():
            if tag is None or tag in macro.tags:
                macros.append(macro.to_dict())
        return macros
    
    def search_macros(self, query: str) -> List[dict]:
        """Cerca macro per nome o descrizione"""
        results = []
        query_lower = query.lower()
        
        for macro in self.macros.values():
            if (query_lower in macro.name.lower() or 
                query_lower in macro.description.lower()):
                results.append(macro.to_dict())
        
        return results
    
    def get_statistics(self) -> dict:
        """Statistiche sulle macro"""
        total_macros = len(self.macros)
        total_plays = sum(m.play_count for m in self.macros.values())
        total_actions = sum(len(m.actions) for m in self.macros.values())
        
        most_played = max(self.macros.values(), key=lambda m: m.play_count) if self.macros else None
        
        return {
            "total_macros": total_macros,
            "total_actions": total_actions,
            "total_plays": total_plays,
            "recording_state": self.recording_state.value,
            "playback_state": self.playback_state.value,
            "triggers_configured": len(self.triggers),
            "most_played": most_played.name if most_played else None
        }
    
    # ============================================
    # IMPORT/EXPORT
    # ============================================
    
    def export_macro(self, macro_id: str) -> Optional[str]:
        """Esporta una macro come JSON"""
        if macro_id not in self.macros:
            return None
        return json.dumps(self.macros[macro_id].to_dict(), indent=2)
    
    def import_macro(self, json_data: str) -> Optional[Macro]:
        """Importa una macro da JSON"""
        try:
            data = json.loads(json_data)
            
            actions = []
            for action_data in data.get("actions", []):
                actions.append(MacroAction(
                    id=action_data["id"],
                    action_type=action_data["action_type"],
                    params=action_data["params"],
                    timestamp=datetime.fromisoformat(action_data["timestamp"]),
                    delay_ms=action_data["delay_ms"]
                ))
            
            macro_id = f"macro_{uuid.uuid4().hex[:8]}"
            
            macro = Macro(
                id=macro_id,
                name=data.get("name", "Imported Macro"),
                description=data.get("description", ""),
                actions=actions,
                created_at=datetime.now(),
                tags=data.get("tags", [])
            )
            
            self.macros[macro_id] = macro
            return macro
            
        except Exception:
            return None
