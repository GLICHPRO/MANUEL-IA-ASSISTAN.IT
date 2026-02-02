"""
📝 JARVIS CORE - Action Logger & Rollback Manager
Logging completo, reversibilità e spiegabilità delle azioni
"""

import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import asyncio
import hashlib


class ActionRecord:
    """Record singola azione eseguita"""
    
    def __init__(self, action_id: str, action_type: str, params: dict,
                 result: dict, reasoning: str = None):
        self.id = action_id
        self.action_type = action_type
        self.params = params
        self.result = result
        self.reasoning = reasoning
        self.timestamp = datetime.now()
        self.user = None
        self.mode = None
        self.reversible = False
        self.rollback_data = None
        self.rolled_back = False
        self.rollback_timestamp = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_type": self.action_type,
            "params": self.params,
            "result": self.result,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "user": self.user,
            "mode": self.mode,
            "reversible": self.reversible,
            "rollback_data": self.rollback_data,
            "rolled_back": self.rolled_back,
            "rollback_timestamp": self.rollback_timestamp.isoformat() if self.rollback_timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ActionRecord':
        record = cls(
            data["id"], data["action_type"], 
            data["params"], data["result"], data.get("reasoning")
        )
        record.timestamp = datetime.fromisoformat(data["timestamp"])
        record.user = data.get("user")
        record.mode = data.get("mode")
        record.reversible = data.get("reversible", False)
        record.rollback_data = data.get("rollback_data")
        record.rolled_back = data.get("rolled_back", False)
        if data.get("rollback_timestamp"):
            record.rollback_timestamp = datetime.fromisoformat(data["rollback_timestamp"])
        return record


class ActionLogger:
    """
    Logger completo per tutte le azioni di Jarvis
    
    Ogni azione registra:
    - timestamp, user, action_type, params, result
    - reasoning (perché è stata fatta)
    - reversible (se può essere annullata)
    - rollback_data (come annullarla)
    """
    
    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir or "logs/actions")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._records: List[ActionRecord] = []
        self._max_memory_records = 1000
        self._action_counter = 0
        
        # File log giornaliero
        self._current_log_file = None
        self._current_log_date = None
        
        # Callbacks per eventi
        self._log_callbacks: List[Callable] = []
    
    def _generate_action_id(self) -> str:
        """Genera ID univoco per azione"""
        self._action_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"act_{timestamp}_{self._action_counter}"
    
    def _get_log_file(self) -> Path:
        """Ottiene file log per oggi"""
        today = datetime.now().date()
        if self._current_log_date != today:
            self._current_log_date = today
            self._current_log_file = self.log_dir / f"actions_{today.isoformat()}.jsonl"
        return self._current_log_file
    
    async def log_action(self, action_type: str, params: dict, result: dict,
                         reasoning: str = None, user: str = None, 
                         mode: str = None, reversible: bool = False,
                         rollback_data: dict = None) -> ActionRecord:
        """
        Registra un'azione eseguita
        
        Args:
            action_type: Tipo di azione
            params: Parametri dell'azione
            result: Risultato dell'esecuzione
            reasoning: Spiegazione del perché
            user: Utente che ha richiesto
            mode: Modalità operativa (PASSIVE/COPILOT/PILOT)
            reversible: Se l'azione è reversibile
            rollback_data: Dati per annullare l'azione
        """
        action_id = self._generate_action_id()
        
        record = ActionRecord(action_id, action_type, params, result, reasoning)
        record.user = user
        record.mode = mode
        record.reversible = reversible
        record.rollback_data = rollback_data
        
        # Salva in memoria
        self._records.append(record)
        if len(self._records) > self._max_memory_records:
            self._records = self._records[-self._max_memory_records:]
        
        # Salva su file
        await self._write_to_file(record)
        
        # Notifica callbacks
        for callback in self._log_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(record)
                else:
                    callback(record)
            except Exception:
                pass
        
        return record
    
    async def _write_to_file(self, record: ActionRecord):
        """Scrive record su file"""
        log_file = self._get_log_file()
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Errore scrittura log: {e}")
    
    def get_recent(self, limit: int = 50) -> List[dict]:
        """Ottiene le ultime azioni"""
        return [r.to_dict() for r in self._records[-limit:]]
    
    def get_by_type(self, action_type: str, limit: int = 50) -> List[dict]:
        """Filtra azioni per tipo"""
        filtered = [r for r in self._records if r.action_type == action_type]
        return [r.to_dict() for r in filtered[-limit:]]
    
    def get_reversible(self, limit: int = 50) -> List[dict]:
        """Ottiene azioni reversibili non ancora annullate"""
        filtered = [r for r in self._records if r.reversible and not r.rolled_back]
        return [r.to_dict() for r in filtered[-limit:]]
    
    def get_by_id(self, action_id: str) -> Optional[ActionRecord]:
        """Trova azione per ID"""
        for record in self._records:
            if record.id == action_id:
                return record
        return None
    
    def search(self, query: str, limit: int = 50) -> List[dict]:
        """Cerca nelle azioni"""
        query_lower = query.lower()
        results = []
        for record in reversed(self._records):
            if (query_lower in record.action_type.lower() or
                query_lower in str(record.params).lower() or
                query_lower in str(record.reasoning or "").lower()):
                results.append(record.to_dict())
                if len(results) >= limit:
                    break
        return results
    
    def on_log(self, callback: Callable):
        """Registra callback per nuovi log"""
        self._log_callbacks.append(callback)
    
    def get_stats(self) -> dict:
        """Statistiche sui log"""
        by_type = {}
        reversible_count = 0
        rolled_back_count = 0
        
        for record in self._records:
            by_type[record.action_type] = by_type.get(record.action_type, 0) + 1
            if record.reversible:
                reversible_count += 1
            if record.rolled_back:
                rolled_back_count += 1
        
        return {
            "total_records": len(self._records),
            "by_type": by_type,
            "reversible": reversible_count,
            "rolled_back": rolled_back_count,
            "log_dir": str(self.log_dir)
        }


class RollbackManager:
    """
    Gestisce il rollback (annullamento) delle azioni
    """
    
    def __init__(self, logger: ActionLogger, executor=None):
        self.logger = logger
        self.executor = executor
        
        # Strategie di rollback per tipo di azione
        self._rollback_strategies: Dict[str, Callable] = {
            "create_file": self._rollback_create_file,
            "delete_file": self._rollback_delete_file,
            "move_file": self._rollback_move_file,
            "copy_file": self._rollback_copy_file,
            "rename_file": self._rollback_rename_file,
            "open_app": self._rollback_open_app,
            "set_volume": self._rollback_set_volume,
        }
        
        # Backup directory per file eliminati
        self.backup_dir = Path("logs/rollback_backup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def register_rollback_strategy(self, action_type: str, strategy: Callable):
        """Registra strategia di rollback per un tipo di azione"""
        self._rollback_strategies[action_type] = strategy
    
    async def rollback(self, action_id: str) -> dict:
        """
        Annulla un'azione specifica
        
        Args:
            action_id: ID dell'azione da annullare
            
        Returns:
            Risultato del rollback
        """
        record = self.logger.get_by_id(action_id)
        
        if not record:
            return {"success": False, "error": "Azione non trovata"}
        
        if not record.reversible:
            return {"success": False, "error": "Azione non reversibile"}
        
        if record.rolled_back:
            return {"success": False, "error": "Azione già annullata"}
        
        # Trova strategia di rollback
        strategy = self._rollback_strategies.get(record.action_type)
        
        if not strategy:
            return {"success": False, "error": f"Nessuna strategia per {record.action_type}"}
        
        try:
            result = await strategy(record)
            
            if result.get("success"):
                record.rolled_back = True
                record.rollback_timestamp = datetime.now()
                
                # Log del rollback
                await self.logger.log_action(
                    f"rollback_{record.action_type}",
                    {"original_action_id": action_id},
                    result,
                    reasoning=f"Rollback di {record.action_type}",
                    reversible=False
                )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def rollback_last(self, count: int = 1) -> List[dict]:
        """Annulla le ultime N azioni reversibili"""
        reversible = [r for r in reversed(self.logger._records) 
                      if r.reversible and not r.rolled_back]
        
        results = []
        for record in reversible[:count]:
            result = await self.rollback(record.id)
            results.append({"action_id": record.id, **result})
        
        return results
    
    async def rollback_since(self, timestamp: datetime) -> List[dict]:
        """Annulla tutte le azioni dopo un certo timestamp"""
        to_rollback = [r for r in self.logger._records 
                       if r.timestamp > timestamp and r.reversible and not r.rolled_back]
        
        results = []
        for record in reversed(to_rollback):  # Annulla in ordine inverso
            result = await self.rollback(record.id)
            results.append({"action_id": record.id, **result})
        
        return results
    
    # === Strategie di Rollback ===
    
    async def _rollback_create_file(self, record: ActionRecord) -> dict:
        """Annulla creazione file (elimina)"""
        file_path = record.params.get("path") or record.rollback_data.get("path")
        if file_path and os.path.exists(file_path):
            # Backup prima di eliminare
            backup_path = self.backup_dir / f"{record.id}_{os.path.basename(file_path)}"
            shutil.copy2(file_path, backup_path)
            os.remove(file_path)
            return {"success": True, "deleted": file_path, "backup": str(backup_path)}
        return {"success": False, "error": "File non trovato"}
    
    async def _rollback_delete_file(self, record: ActionRecord) -> dict:
        """Annulla eliminazione file (ripristina da backup)"""
        backup_path = record.rollback_data.get("backup_path")
        original_path = record.rollback_data.get("original_path")
        
        if backup_path and os.path.exists(backup_path) and original_path:
            shutil.copy2(backup_path, original_path)
            return {"success": True, "restored": original_path}
        return {"success": False, "error": "Backup non trovato"}
    
    async def _rollback_move_file(self, record: ActionRecord) -> dict:
        """Annulla spostamento file"""
        source = record.rollback_data.get("original_path")
        destination = record.rollback_data.get("new_path")
        
        if destination and os.path.exists(destination) and source:
            shutil.move(destination, source)
            return {"success": True, "moved_back": source}
        return {"success": False, "error": "File non trovato"}
    
    async def _rollback_copy_file(self, record: ActionRecord) -> dict:
        """Annulla copia file (elimina copia)"""
        copy_path = record.rollback_data.get("copy_path")
        if copy_path and os.path.exists(copy_path):
            os.remove(copy_path)
            return {"success": True, "deleted_copy": copy_path}
        return {"success": False, "error": "Copia non trovata"}
    
    async def _rollback_rename_file(self, record: ActionRecord) -> dict:
        """Annulla rinomina file"""
        old_name = record.rollback_data.get("old_name")
        new_name = record.rollback_data.get("new_name")
        
        if new_name and os.path.exists(new_name) and old_name:
            os.rename(new_name, old_name)
            return {"success": True, "renamed_back": old_name}
        return {"success": False, "error": "File non trovato"}
    
    async def _rollback_open_app(self, record: ActionRecord) -> dict:
        """Annulla apertura app (chiudi)"""
        if self.executor:
            app_name = record.params.get("app") or record.params.get("name")
            return await self.executor.execute({
                "action": "close_app",
                "params": {"name": app_name}
            })
        return {"success": False, "error": "Executor non disponibile"}
    
    async def _rollback_set_volume(self, record: ActionRecord) -> dict:
        """Annulla cambio volume (ripristina precedente)"""
        previous_volume = record.rollback_data.get("previous_volume")
        if previous_volume is not None and self.executor:
            return await self.executor.execute({
                "action": "set_volume",
                "params": {"level": previous_volume}
            })
        return {"success": False, "error": "Volume precedente non salvato"}
    
    def prepare_rollback_data(self, action_type: str, params: dict) -> dict:
        """
        Prepara i dati necessari per un futuro rollback
        Da chiamare PRIMA di eseguire l'azione
        """
        rollback_data = {}
        
        if action_type == "delete_file":
            file_path = params.get("path")
            if file_path and os.path.exists(file_path):
                # Crea backup
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.path.basename(file_path)}"
                backup_path = self.backup_dir / backup_name
                shutil.copy2(file_path, backup_path)
                rollback_data = {
                    "original_path": file_path,
                    "backup_path": str(backup_path)
                }
        
        elif action_type == "move_file":
            rollback_data = {
                "original_path": params.get("source"),
                "new_path": params.get("destination")
            }
        
        elif action_type == "copy_file":
            rollback_data = {
                "copy_path": params.get("destination")
            }
        
        elif action_type == "rename_file":
            rollback_data = {
                "old_name": params.get("old_path"),
                "new_name": params.get("new_path")
            }
        
        elif action_type == "create_file":
            rollback_data = {
                "path": params.get("path")
            }
        
        return rollback_data
    
    def get_rollback_status(self) -> dict:
        """Stato del sistema di rollback"""
        return {
            "available_strategies": list(self._rollback_strategies.keys()),
            "backup_dir": str(self.backup_dir),
            "backup_count": len(list(self.backup_dir.glob("*")))
        }
