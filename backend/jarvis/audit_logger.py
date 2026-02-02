# /backend/jarvis/audit_logger.py
"""
JARVIS Audit Logger - Registrazione Azioni con Rollback
Registra tutte le azioni del sistema con possibilità di rollback.
"""

import json
import hashlib
import shutil
import os
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import asyncio
import logging

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Tipi di eventi audit"""
    DECISION = "decision"           # Decisione presa
    EXECUTION = "execution"         # Azione eseguita
    SYSTEM = "system"               # Evento di sistema
    USER = "user"                   # Azione utente
    ERROR = "error"                 # Errore
    SECURITY = "security"           # Evento sicurezza
    ROLLBACK = "rollback"           # Rollback eseguito
    CONFIG = "config"               # Modifica configurazione
    FILE = "file"                   # Operazione su file
    PROCESS = "process"             # Operazione su processo
    NETWORK = "network"             # Operazione di rete


class AuditSeverity(Enum):
    """Severità eventi"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RollbackStatus(Enum):
    """Stato rollback"""
    PENDING = "pending"             # In attesa
    IN_PROGRESS = "in_progress"     # In corso
    COMPLETED = "completed"         # Completato
    FAILED = "failed"               # Fallito
    PARTIAL = "partial"             # Parziale
    NOT_APPLICABLE = "not_applicable"  # Non applicabile


@dataclass
class RollbackAction:
    """Azione di rollback"""
    action_type: str
    params: dict
    description: str
    
    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "params": self.params,
            "description": self.description
        }


@dataclass
class AuditEntry:
    """Entry del log audit"""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    actor: str                      # Chi ha eseguito (user, system, jarvis)
    action: str                     # Descrizione azione
    target: Optional[str]           # Oggetto dell'azione
    details: dict                   # Dettagli aggiuntivi
    result: Optional[str]           # Risultato
    success: bool
    session_id: str
    rollback_action: Optional[RollbackAction] = None
    rollback_status: RollbackStatus = RollbackStatus.NOT_APPLICABLE
    parent_id: Optional[str] = None  # Per azioni correlate
    tags: List[str] = field(default_factory=list)
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calcola checksum per integrità"""
        content = f"{self.id}|{self.timestamp.isoformat()}|{self.action}|{self.actor}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "details": self.details,
            "result": self.result,
            "success": self.success,
            "session_id": self.session_id,
            "rollback_action": self.rollback_action.to_dict() if self.rollback_action else None,
            "rollback_status": self.rollback_status.value,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AuditEntry':
        """Crea entry da dizionario"""
        rollback = None
        if data.get("rollback_action"):
            ra = data["rollback_action"]
            rollback = RollbackAction(
                action_type=ra["action_type"],
                params=ra["params"],
                description=ra["description"]
            )
        
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            severity=AuditSeverity(data["severity"]),
            actor=data["actor"],
            action=data["action"],
            target=data.get("target"),
            details=data.get("details", {}),
            result=data.get("result"),
            success=data.get("success", True),
            session_id=data.get("session_id", ""),
            rollback_action=rollback,
            rollback_status=RollbackStatus(data.get("rollback_status", "not_applicable")),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            checksum=data.get("checksum", "")
        )


@dataclass 
class FileBackup:
    """Backup di un file per rollback"""
    original_path: str
    backup_path: str
    timestamp: datetime
    audit_entry_id: str
    checksum: str
    
    def to_dict(self) -> dict:
        return {
            "original_path": self.original_path,
            "backup_path": self.backup_path,
            "timestamp": self.timestamp.isoformat(),
            "audit_entry_id": self.audit_entry_id,
            "checksum": self.checksum
        }


class AuditLogger:
    """
    Logger Audit per JARVIS.
    Registra tutte le azioni con possibilità di rollback.
    """
    
    def __init__(self, storage_path: str = None, session_id: str = None):
        # Storage
        self.storage_path = Path(storage_path) if storage_path else Path("./audit_logs")
        self.backup_path = self.storage_path / "backups"
        
        # Crea directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Session
        self.session_id = session_id or self._generate_session_id()
        
        # In-memory log
        self.entries: List[AuditEntry] = []
        self.max_entries = 10000
        
        # File backups per rollback
        self.file_backups: Dict[str, FileBackup] = {}
        
        # Entry counter
        self._counter = 0
        
        # Rollback handlers
        self.rollback_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        
        # Stats
        self.stats = {
            "total_logged": 0,
            "by_type": {},
            "by_severity": {},
            "rollbacks_executed": 0,
            "rollbacks_failed": 0
        }
        
        # Current file
        self._current_log_file = self._get_log_file_path()
        
        # Load existing
        self._load_session_log()
    
    def _generate_session_id(self) -> str:
        """Genera ID sessione"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _get_log_file_path(self) -> Path:
        """Path del file log corrente"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.storage_path / f"audit_{date_str}.jsonl"
    
    def _generate_entry_id(self) -> str:
        """Genera ID entry"""
        self._counter += 1
        return f"{self.session_id}_{self._counter:06d}"
    
    def _register_default_handlers(self):
        """Registra handler rollback default"""
        self.rollback_handlers["file_create"] = self._rollback_file_create
        self.rollback_handlers["file_delete"] = self._rollback_file_delete
        self.rollback_handlers["file_modify"] = self._rollback_file_modify
        self.rollback_handlers["file_move"] = self._rollback_file_move
        self.rollback_handlers["file_copy"] = self._rollback_file_copy
        self.rollback_handlers["process_start"] = self._rollback_process_start
        self.rollback_handlers["config_change"] = self._rollback_config_change
    
    # === Logging ===
    
    def log(self, event_type: AuditEventType, action: str,
            actor: str = "jarvis", target: str = None,
            details: dict = None, result: str = None,
            success: bool = True, severity: AuditSeverity = AuditSeverity.INFO,
            rollback_action: RollbackAction = None,
            parent_id: str = None, tags: List[str] = None) -> AuditEntry:
        """
        Registra un evento nel log audit.
        
        Args:
            event_type: Tipo di evento
            action: Descrizione dell'azione
            actor: Chi ha eseguito l'azione
            target: Oggetto dell'azione
            details: Dettagli aggiuntivi
            result: Risultato dell'azione
            success: Se l'azione ha avuto successo
            severity: Severità dell'evento
            rollback_action: Azione per rollback
            parent_id: ID evento correlato
            tags: Tag per categorizzazione
        
        Returns:
            AuditEntry creata
        """
        entry = AuditEntry(
            id=self._generate_entry_id(),
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            actor=actor,
            action=action,
            target=target,
            details=details or {},
            result=result,
            success=success,
            session_id=self.session_id,
            rollback_action=rollback_action,
            rollback_status=RollbackStatus.PENDING if rollback_action else RollbackStatus.NOT_APPLICABLE,
            parent_id=parent_id,
            tags=tags or []
        )
        
        # Store in memory
        self.entries.append(entry)
        
        # Cleanup se oltre limite
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Persist to file
        self._persist_entry(entry)
        
        # Update stats
        self._update_stats(entry)
        
        # Log anche con logger standard
        log_level = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.INFO)
        
        logger.log(log_level, f"[AUDIT] {event_type.value}: {action} -> {result}")
        
        return entry
    
    def _persist_entry(self, entry: AuditEntry):
        """Persiste entry su file"""
        # Check se cambio giorno
        current_file = self._get_log_file_path()
        if current_file != self._current_log_file:
            self._current_log_file = current_file
        
        try:
            with open(self._current_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Errore persistenza audit: {e}")
    
    def _update_stats(self, entry: AuditEntry):
        """Aggiorna statistiche"""
        self.stats["total_logged"] += 1
        
        type_key = entry.event_type.value
        self.stats["by_type"][type_key] = self.stats["by_type"].get(type_key, 0) + 1
        
        sev_key = entry.severity.value
        self.stats["by_severity"][sev_key] = self.stats["by_severity"].get(sev_key, 0) + 1
    
    def _load_session_log(self):
        """Carica log della sessione corrente"""
        if not self._current_log_file.exists():
            return
        
        try:
            with open(self._current_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        if data.get("session_id") == self.session_id:
                            entry = AuditEntry.from_dict(data)
                            self.entries.append(entry)
                            self._counter = max(self._counter, 
                                              int(entry.id.split('_')[-1]))
        except Exception as e:
            logger.error(f"Errore caricamento audit log: {e}")
    
    # === Shortcut Methods ===
    
    def log_decision(self, action: str, details: dict, 
                     success: bool = True) -> AuditEntry:
        """Log decisione"""
        return self.log(
            AuditEventType.DECISION,
            action,
            details=details,
            success=success,
            severity=AuditSeverity.INFO
        )
    
    def log_execution(self, action: str, target: str,
                      result: str, success: bool,
                      rollback: RollbackAction = None) -> AuditEntry:
        """Log esecuzione"""
        return self.log(
            AuditEventType.EXECUTION,
            action,
            target=target,
            result=result,
            success=success,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            rollback_action=rollback
        )
    
    def log_error(self, action: str, error: str,
                  details: dict = None) -> AuditEntry:
        """Log errore"""
        return self.log(
            AuditEventType.ERROR,
            action,
            details=details or {},
            result=error,
            success=False,
            severity=AuditSeverity.ERROR
        )
    
    def log_security(self, action: str, details: dict,
                     severity: AuditSeverity = AuditSeverity.WARNING) -> AuditEntry:
        """Log evento sicurezza"""
        return self.log(
            AuditEventType.SECURITY,
            action,
            details=details,
            severity=severity,
            tags=["security"]
        )
    
    def log_file_operation(self, operation: str, path: str,
                           success: bool, create_backup: bool = True,
                           details: dict = None) -> AuditEntry:
        """Log operazione file con backup opzionale"""
        rollback = None
        
        if create_backup and success:
            # Crea backup per rollback
            if operation in ["modify", "delete"]:
                backup = self._create_file_backup(path)
                if backup:
                    rollback = RollbackAction(
                        action_type=f"file_{operation}",
                        params={"path": path, "backup": backup.backup_path},
                        description=f"Ripristina {path} dal backup"
                    )
            elif operation == "create":
                rollback = RollbackAction(
                    action_type="file_create",
                    params={"path": path},
                    description=f"Elimina file creato: {path}"
                )
        
        return self.log(
            AuditEventType.FILE,
            f"File {operation}: {path}",
            target=path,
            details=details or {},
            success=success,
            rollback_action=rollback,
            tags=["file", operation]
        )
    
    # === File Backup ===
    
    def _create_file_backup(self, file_path: str) -> Optional[FileBackup]:
        """Crea backup di un file"""
        try:
            if not os.path.exists(file_path):
                return None
            
            # Calcola checksum
            with open(file_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()[:16]
            
            # Path backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(file_path)
            backup_filename = f"{timestamp}_{checksum}_{filename}"
            backup_path = self.backup_path / backup_filename
            
            # Copia file
            shutil.copy2(file_path, backup_path)
            
            backup = FileBackup(
                original_path=file_path,
                backup_path=str(backup_path),
                timestamp=datetime.now(),
                audit_entry_id="",  # Sarà settato dopo
                checksum=checksum
            )
            
            self.file_backups[file_path] = backup
            return backup
            
        except Exception as e:
            logger.error(f"Errore creazione backup {file_path}: {e}")
            return None
    
    # === Rollback ===
    
    async def rollback(self, entry_id: str) -> dict:
        """
        Esegue rollback di un'azione.
        
        Args:
            entry_id: ID dell'entry da rollbackare
        
        Returns:
            Risultato del rollback
        """
        # Trova entry
        entry = self.get_entry(entry_id)
        if not entry:
            return {"success": False, "error": "Entry non trovata"}
        
        if not entry.rollback_action:
            return {"success": False, "error": "Rollback non disponibile per questa azione"}
        
        if entry.rollback_status == RollbackStatus.COMPLETED:
            return {"success": False, "error": "Rollback già eseguito"}
        
        # Aggiorna stato
        entry.rollback_status = RollbackStatus.IN_PROGRESS
        
        try:
            # Trova handler
            action_type = entry.rollback_action.action_type
            handler = self.rollback_handlers.get(action_type)
            
            if not handler:
                entry.rollback_status = RollbackStatus.FAILED
                return {"success": False, "error": f"Handler rollback non trovato: {action_type}"}
            
            # Esegui rollback
            result = await handler(entry.rollback_action.params)
            
            if result.get("success"):
                entry.rollback_status = RollbackStatus.COMPLETED
                self.stats["rollbacks_executed"] += 1
                
                # Log rollback
                self.log(
                    AuditEventType.ROLLBACK,
                    f"Rollback eseguito: {entry.action}",
                    target=entry.target,
                    details={"original_entry_id": entry_id},
                    success=True,
                    parent_id=entry_id,
                    tags=["rollback"]
                )
            else:
                entry.rollback_status = RollbackStatus.FAILED
                self.stats["rollbacks_failed"] += 1
            
            return result
            
        except Exception as e:
            entry.rollback_status = RollbackStatus.FAILED
            self.stats["rollbacks_failed"] += 1
            return {"success": False, "error": str(e)}
    
    async def rollback_to_point(self, entry_id: str) -> dict:
        """
        Esegue rollback di tutte le azioni dopo un certo punto.
        
        Args:
            entry_id: ID dell'entry fino a cui tornare
        
        Returns:
            Risultato aggregato
        """
        # Trova indice entry
        target_idx = None
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                target_idx = i
                break
        
        if target_idx is None:
            return {"success": False, "error": "Entry non trovata"}
        
        # Rollback in ordine inverso
        entries_to_rollback = self.entries[target_idx + 1:]
        entries_to_rollback.reverse()
        
        results = []
        for entry in entries_to_rollback:
            if entry.rollback_action and entry.rollback_status == RollbackStatus.PENDING:
                result = await self.rollback(entry.id)
                results.append({
                    "entry_id": entry.id,
                    "action": entry.action,
                    "result": result
                })
        
        success_count = sum(1 for r in results if r["result"].get("success"))
        
        return {
            "success": success_count == len(results),
            "total": len(results),
            "successful": success_count,
            "failed": len(results) - success_count,
            "details": results
        }
    
    # === Rollback Handlers ===
    
    async def _rollback_file_create(self, params: dict) -> dict:
        """Rollback creazione file (elimina)"""
        path = params.get("path")
        try:
            if os.path.exists(path):
                os.remove(path)
            return {"success": True, "action": f"File eliminato: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _rollback_file_delete(self, params: dict) -> dict:
        """Rollback eliminazione file (ripristina da backup)"""
        path = params.get("path")
        backup_path = params.get("backup")
        try:
            if backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, path)
                return {"success": True, "action": f"File ripristinato: {path}"}
            return {"success": False, "error": "Backup non disponibile"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _rollback_file_modify(self, params: dict) -> dict:
        """Rollback modifica file (ripristina da backup)"""
        return await self._rollback_file_delete(params)
    
    async def _rollback_file_move(self, params: dict) -> dict:
        """Rollback spostamento file"""
        source = params.get("source")
        destination = params.get("destination")
        try:
            if os.path.exists(destination):
                shutil.move(destination, source)
            return {"success": True, "action": f"File ripristinato: {source}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _rollback_file_copy(self, params: dict) -> dict:
        """Rollback copia file (elimina copia)"""
        destination = params.get("destination")
        try:
            if os.path.exists(destination):
                os.remove(destination)
            return {"success": True, "action": f"Copia eliminata: {destination}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _rollback_process_start(self, params: dict) -> dict:
        """Rollback avvio processo (termina)"""
        pid = params.get("pid")
        try:
            import psutil
            if pid and psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                proc.terminate()
                return {"success": True, "action": f"Processo {pid} terminato"}
            return {"success": True, "action": "Processo non più attivo"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _rollback_config_change(self, params: dict) -> dict:
        """Rollback modifica configurazione"""
        config_key = params.get("key")
        original_value = params.get("original_value")
        config_store = params.get("store")
        
        # Questo dipende dall'implementazione del config store
        return {"success": False, "error": "Config rollback richiede implementazione specifica"}
    
    def register_rollback_handler(self, action_type: str, handler: Callable):
        """Registra handler rollback personalizzato"""
        self.rollback_handlers[action_type] = handler
    
    # === Query ===
    
    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Ottiene entry per ID"""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None
    
    def get_entries(self, limit: int = 100,
                    event_type: AuditEventType = None,
                    severity: AuditSeverity = None,
                    actor: str = None,
                    success: bool = None,
                    start_time: datetime = None,
                    end_time: datetime = None,
                    tags: List[str] = None) -> List[dict]:
        """
        Query entries con filtri.
        """
        results = []
        
        for entry in reversed(self.entries):
            # Applica filtri
            if event_type and entry.event_type != event_type:
                continue
            if severity and entry.severity != severity:
                continue
            if actor and entry.actor != actor:
                continue
            if success is not None and entry.success != success:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if tags and not any(t in entry.tags for t in tags):
                continue
            
            results.append(entry.to_dict())
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_rollbackable_entries(self, limit: int = 50) -> List[dict]:
        """Ottiene entries che possono essere rollbackate"""
        results = []
        
        for entry in reversed(self.entries):
            if entry.rollback_action and entry.rollback_status == RollbackStatus.PENDING:
                results.append(entry.to_dict())
                if len(results) >= limit:
                    break
        
        return results
    
    def search(self, query: str, limit: int = 100) -> List[dict]:
        """Cerca nel log"""
        query_lower = query.lower()
        results = []
        
        for entry in reversed(self.entries):
            if (query_lower in entry.action.lower() or
                (entry.target and query_lower in entry.target.lower()) or
                query_lower in str(entry.details).lower()):
                results.append(entry.to_dict())
                if len(results) >= limit:
                    break
        
        return results
    
    # === Export ===
    
    def export_log(self, start_date: datetime = None,
                   end_date: datetime = None,
                   format: str = "json") -> str:
        """Esporta log in formato specificato"""
        entries = []
        
        for entry in self.entries:
            if start_date and entry.timestamp < start_date:
                continue
            if end_date and entry.timestamp > end_date:
                continue
            entries.append(entry.to_dict())
        
        if format == "json":
            return json.dumps(entries, indent=2, ensure_ascii=False)
        elif format == "csv":
            # Semplice CSV
            lines = ["timestamp,event_type,severity,actor,action,success"]
            for e in entries:
                lines.append(
                    f"{e['timestamp']},{e['event_type']},{e['severity']},"
                    f"{e['actor']},{e['action']},{e['success']}"
                )
            return "\n".join(lines)
        
        return json.dumps(entries)
    
    # === Cleanup ===
    
    def cleanup_old_logs(self, days: int = 30):
        """Pulisce log più vecchi di X giorni"""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Pulisci file
        for log_file in self.storage_path.glob("audit_*.jsonl"):
            try:
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    log_file.unlink()
                    logger.info(f"Eliminato log vecchio: {log_file}")
            except:
                pass
        
        # Pulisci backup
        for backup in self.backup_path.glob("*"):
            try:
                if backup.stat().st_mtime < cutoff.timestamp():
                    backup.unlink()
            except:
                pass
    
    def get_status(self) -> dict:
        """Stato del logger"""
        return {
            "session_id": self.session_id,
            "entries_in_memory": len(self.entries),
            "file_backups": len(self.file_backups),
            "current_log_file": str(self._current_log_file),
            "stats": self.stats,
            "rollback_handlers": list(self.rollback_handlers.keys())
        }
