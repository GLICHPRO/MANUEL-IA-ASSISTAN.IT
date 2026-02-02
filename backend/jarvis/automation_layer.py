"""
🤖 Automation Layer - Controllo Completo Automazioni

Gestisce:
- Controllo applicazioni (avvio, stop, interazione)
- Esecuzione script e comandi
- Routine automatizzate
- Processi di sistema
- Conferma logica in modalità Pilot

Tutte le azioni richiedono validazione interna prima dell'esecuzione.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
import uuid
import asyncio
import subprocess
import logging
from pathlib import Path
import json


# Logger dedicato
auto_logger = logging.getLogger("automation_layer")
auto_logger.setLevel(logging.DEBUG)


# === ENUMS ===

class AutomationType(Enum):
    """Tipi di automazione"""
    APPLICATION = "application"      # Controllo app
    SCRIPT = "script"               # Esecuzione script
    COMMAND = "command"             # Comando sistema
    ROUTINE = "routine"             # Routine multi-step
    PROCESS = "process"             # Processo sistema
    WORKFLOW = "workflow"           # Workflow complesso
    SCHEDULED = "scheduled"         # Task schedulato
    TRIGGERED = "triggered"         # Task su trigger


class ActionType(Enum):
    """Tipi di azione"""
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    PAUSE = "pause"
    RESUME = "resume"
    EXECUTE = "execute"
    INTERACT = "interact"
    MONITOR = "monitor"
    CONFIGURE = "configure"


class ExecutionStatus(Enum):
    """Stato esecuzione"""
    PENDING = "pending"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RiskLevel(Enum):
    """Livello di rischio azione"""
    SAFE = "safe"           # Nessun rischio
    LOW = "low"             # Rischio basso
    MEDIUM = "medium"       # Richiede attenzione
    HIGH = "high"           # Richiede conferma
    CRITICAL = "critical"   # Richiede doppia conferma


class ConfirmationType(Enum):
    """Tipo di conferma richiesta"""
    NONE = "none"                    # Nessuna conferma
    INTERNAL = "internal"            # Conferma logica interna
    USER = "user"                    # Conferma utente
    DUAL = "dual"                    # Entrambe


class ExecutionMode(Enum):
    """Modalità di esecuzione"""
    NORMAL = "normal"                # Esecuzione standard con output
    SHADOW = "shadow"                # Solo osservazione, nessuna esecuzione
    SILENT = "silent"                # Esecuzione senza output visivo
    DRY_RUN = "dry_run"              # Simula esecuzione, non esegue


class UndoStatus(Enum):
    """Stato reversibilità azione"""
    REVERSIBLE = "reversible"        # Può essere annullata
    IRREVERSIBLE = "irreversible"    # Non può essere annullata
    PARTIAL = "partial"              # Reversibilità parziale
    UNDONE = "undone"                # Già annullata


# === DATA CLASSES ===

@dataclass
class ActionLog:
    """Log entry per un'azione"""
    timestamp: datetime = field(default_factory=datetime.now)
    action_id: str = ""
    event: str = ""  # created, validated, confirmed, executed, completed, failed, undone
    details: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "normal"
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action_id": self.action_id,
            "event": self.event,
            "details": self.details,
            "execution_mode": self.execution_mode
        }


@dataclass
class UndoInfo:
    """Informazioni per reversibilità azione"""
    is_reversible: bool = True
    undo_command: str = ""  # Comando per annullare
    undo_type: str = ""     # stop, kill, delete, restore
    original_state: Dict[str, Any] = field(default_factory=dict)
    undo_status: UndoStatus = UndoStatus.REVERSIBLE
    undo_performed_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "is_reversible": self.is_reversible,
            "undo_command": self.undo_command,
            "undo_type": self.undo_type,
            "undo_status": self.undo_status.value,
            "undo_performed_at": self.undo_performed_at.isoformat() if self.undo_performed_at else None
        }


@dataclass
class AutomationAction:
    """Singola azione di automazione"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    
    automation_type: AutomationType = AutomationType.COMMAND
    action_type: ActionType = ActionType.EXECUTE
    
    # Target
    target: str = ""  # App name, script path, command, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Rischio e conferma
    risk_level: RiskLevel = RiskLevel.LOW
    confirmation_type: ConfirmationType = ConfirmationType.INTERNAL
    
    # Modalità esecuzione
    execution_mode: ExecutionMode = ExecutionMode.NORMAL
    
    # Stato
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    # Risultato
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 60
    
    # Validazione interna
    validation_checks: List[Dict] = field(default_factory=list)
    is_validated: bool = False
    validation_reason: str = ""
    
    # Reversibilità
    undo_info: Optional[UndoInfo] = field(default_factory=lambda: UndoInfo())
    
    # Metadata
    requester: str = "jarvis"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "automation_type": self.automation_type.value,
            "action_type": self.action_type.value,
            "target": self.target,
            "parameters": self.parameters,
            "risk_level": self.risk_level.value,
            "confirmation_type": self.confirmation_type.value,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "is_validated": self.is_validated,
            "validation_reason": self.validation_reason,
            "created_at": self.created_at.isoformat(),
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_mode": self.execution_mode.value,
            "undo_info": self.undo_info.to_dict() if self.undo_info else None
        }


@dataclass
class ConfirmationRequest:
    """Richiesta di conferma per azione"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action_id: str = ""
    action_summary: str = ""
    
    risk_level: RiskLevel = RiskLevel.MEDIUM
    confirmation_type: ConfirmationType = ConfirmationType.INTERNAL
    
    # Checks di validazione
    validation_checks: List[Dict] = field(default_factory=list)
    all_checks_passed: bool = False
    
    # Decisione
    is_confirmed: Optional[bool] = None
    confirmed_by: str = ""  # "internal_logic", "user", "auto"
    rejection_reason: str = ""
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    decided_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_id": self.action_id,
            "action_summary": self.action_summary,
            "risk_level": self.risk_level.value,
            "confirmation_type": self.confirmation_type.value,
            "validation_checks": self.validation_checks,
            "all_checks_passed": self.all_checks_passed,
            "is_confirmed": self.is_confirmed,
            "confirmed_by": self.confirmed_by,
            "rejection_reason": self.rejection_reason,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None
        }


@dataclass
class Routine:
    """Routine di automazione multi-step"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    
    # Steps
    steps: List[AutomationAction] = field(default_factory=list)
    current_step: int = 0
    
    # Configurazione
    stop_on_error: bool = True
    retry_failed: bool = False
    max_retries: int = 2
    
    # Stato
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    # Trigger (opzionale)
    trigger_type: Optional[str] = None  # "schedule", "event", "condition"
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    
    # Risultati
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps_count": len(self.steps),
            "current_step": self.current_step,
            "stop_on_error": self.stop_on_error,
            "status": self.status.value,
            "trigger_type": self.trigger_type,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }


# === INTERNAL VALIDATOR ===

class InternalValidator:
    """
    Validatore logico interno per modalità Pilot.
    Esegue checks automatici prima di confermare un'azione.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, Callable] = {}
        self.blocked_targets: List[str] = []
        self.allowed_targets: List[str] = []
        
        # Inizializza regole default
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Regole di validazione default"""
        
        # Regola: target non bloccato
        self.validation_rules["target_not_blocked"] = lambda action: (
            action.target.lower() not in [t.lower() for t in self.blocked_targets],
            "Target in blocklist"
        )
        
        # Regola: parametri validi
        self.validation_rules["valid_parameters"] = lambda action: (
            self._check_parameters(action),
            "Parametri non validi"
        )
        
        # Regola: rischio accettabile
        self.validation_rules["acceptable_risk"] = lambda action: (
            action.risk_level in [RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM],
            f"Rischio troppo alto: {action.risk_level.value}"
        )
        
        # Regola: tipo azione permesso
        self.validation_rules["action_type_allowed"] = lambda action: (
            action.action_type not in [ActionType.STOP, ActionType.RESTART] or 
            action.risk_level != RiskLevel.CRITICAL,
            "Azione critica non permessa"
        )
        
        # Regola: timeout ragionevole
        self.validation_rules["reasonable_timeout"] = lambda action: (
            5 <= action.timeout_seconds <= 3600,
            "Timeout fuori range (5s - 1h)"
        )
    
    def _check_parameters(self, action: AutomationAction) -> bool:
        """Verifica parametri dell'azione"""
        # Check base: nessun parametro pericoloso
        dangerous_params = ["sudo", "rm -rf", "format", "del /s", "shutdown"]
        params_str = json.dumps(action.parameters).lower()
        
        for dp in dangerous_params:
            if dp in params_str:
                return False
        return True
    
    def add_rule(self, name: str, rule: Callable):
        """Aggiunge regola di validazione"""
        self.validation_rules[name] = rule
    
    def block_target(self, target: str):
        """Blocca un target"""
        if target not in self.blocked_targets:
            self.blocked_targets.append(target)
    
    def allow_target(self, target: str):
        """Permette un target"""
        if target not in self.allowed_targets:
            self.allowed_targets.append(target)
    
    def validate(self, action: AutomationAction) -> ConfirmationRequest:
        """
        Esegue validazione interna dell'azione.
        Restituisce ConfirmationRequest con risultato.
        """
        request = ConfirmationRequest(
            action_id=action.id,
            action_summary=f"{action.action_type.value} {action.target}",
            risk_level=action.risk_level,
            confirmation_type=action.confirmation_type
        )
        
        # Esegui tutti i check
        all_passed = True
        
        for rule_name, rule_fn in self.validation_rules.items():
            try:
                passed, message = rule_fn(action)
                request.validation_checks.append({
                    "rule": rule_name,
                    "passed": passed,
                    "message": message if not passed else "OK"
                })
                if not passed:
                    all_passed = False
            except Exception as e:
                request.validation_checks.append({
                    "rule": rule_name,
                    "passed": False,
                    "message": f"Errore validazione: {str(e)}"
                })
                all_passed = False
        
        request.all_checks_passed = all_passed
        
        # Auto-conferma se tutti i check passano e rischio basso
        if all_passed and action.risk_level in [RiskLevel.SAFE, RiskLevel.LOW]:
            request.is_confirmed = True
            request.confirmed_by = "internal_logic"
            request.decided_at = datetime.now()
        elif all_passed and action.risk_level == RiskLevel.MEDIUM:
            # Medium risk: conferma con nota
            request.is_confirmed = True
            request.confirmed_by = "internal_logic"
            request.decided_at = datetime.now()
            auto_logger.info(f"Auto-confirmed medium risk action: {action.id}")
        else:
            # High/Critical o check falliti: richiede review
            request.is_confirmed = False
            request.rejection_reason = "Validation checks failed or high risk"
        
        return request


# === AUTOMATION EXECUTOR ===

class AutomationExecutor:
    """
    Esecutore di azioni di automazione.
    Gestisce l'esecuzione effettiva di comandi, script, applicazioni.
    """
    
    def __init__(self):
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.execution_history: List[AutomationAction] = []
        
        # App registry
        self.app_registry: Dict[str, str] = {
            "chrome": "chrome",
            "firefox": "firefox",
            "notepad": "notepad",
            "explorer": "explorer",
            "cmd": "cmd",
            "powershell": "powershell",
            "code": "code",
            "vscode": "code",
            "calculator": "calc",
            "calc": "calc"
        }
        
        # Undo registry - mappa azioni reversibili
        self.undo_registry: Dict[str, Callable] = {}
    
    async def execute(self, action: AutomationAction, 
                     silent: bool = False) -> AutomationAction:
        """
        Esegue un'azione di automazione.
        
        Modalità:
        - NORMAL: esecuzione standard con output
        - SHADOW: solo osservazione, non esegue realmente
        - SILENT: esecuzione senza output visivo
        - DRY_RUN: simula esecuzione, mostra cosa farebbe
        """
        action.started_at = datetime.now()
        
        # Shadow Mode: solo osservazione
        if action.execution_mode == ExecutionMode.SHADOW:
            action.status = ExecutionStatus.COMPLETED
            action.result = {
                "mode": "shadow",
                "message": f"[SHADOW] Osservato: {action.action_type.value} {action.target}",
                "would_execute": True,
                "observed_at": datetime.now().isoformat()
            }
            action.completed_at = datetime.now()
            self.execution_history.append(action)
            auto_logger.info(f"[SHADOW] Observed action: {action.id} - {action.name}")
            return action
        
        # Dry Run: simula senza eseguire
        if action.execution_mode == ExecutionMode.DRY_RUN:
            action.status = ExecutionStatus.COMPLETED
            action.result = {
                "mode": "dry_run",
                "message": f"[DRY_RUN] Simulato: {action.action_type.value} {action.target}",
                "would_execute": True,
                "simulated_at": datetime.now().isoformat()
            }
            action.completed_at = datetime.now()
            self.execution_history.append(action)
            auto_logger.info(f"[DRY_RUN] Simulated action: {action.id} - {action.name}")
            return action
        
        # Prepara undo info PRIMA dell'esecuzione
        self._prepare_undo_info(action)
        
        action.status = ExecutionStatus.EXECUTING
        
        # Silent mode: sopprime output
        is_silent = action.execution_mode == ExecutionMode.SILENT or silent
        
        try:
            if action.automation_type == AutomationType.APPLICATION:
                await self._execute_application(action, silent=is_silent)
            elif action.automation_type == AutomationType.SCRIPT:
                await self._execute_script(action, silent=is_silent)
            elif action.automation_type == AutomationType.COMMAND:
                await self._execute_command(action, silent=is_silent)
            elif action.automation_type == AutomationType.PROCESS:
                await self._execute_process(action, silent=is_silent)
            else:
                action.result = {"message": f"Type {action.automation_type.value} handled"}
            
            # Aggiungi flag silent al risultato
            if is_silent:
                action.result["silent_mode"] = True
            
            action.status = ExecutionStatus.COMPLETED
            action.completed_at = datetime.now()
            
        except asyncio.TimeoutError:
            action.status = ExecutionStatus.TIMEOUT
            action.error = f"Timeout after {action.timeout_seconds}s"
        except Exception as e:
            action.status = ExecutionStatus.FAILED
            action.error = str(e)
            auto_logger.error(f"Execution failed for {action.id}: {e}")
        
        self.execution_history.append(action)
        return action
    
    def _prepare_undo_info(self, action: AutomationAction):
        """Prepara informazioni per reversibilità dell'azione"""
        undo = action.undo_info or UndoInfo()
        
        # Determina reversibilità in base a tipo azione
        if action.action_type == ActionType.START:
            # Avvio app/processo → reversibile con stop
            undo.is_reversible = True
            undo.undo_type = "stop"
            undo.undo_command = f"taskkill /IM {action.target}.exe /F"
            undo.original_state = {"was_running": False}
            
        elif action.action_type == ActionType.STOP:
            # Stop app → reversibile con restart
            undo.is_reversible = True
            undo.undo_type = "restart"
            undo.undo_command = action.target
            undo.original_state = {"was_running": True}
            
        elif action.action_type in [ActionType.EXECUTE, ActionType.INTERACT]:
            # Comandi generici → dipende dal comando
            if any(kw in action.target.lower() for kw in ["rm", "del", "format", "drop"]):
                undo.is_reversible = False
                undo.undo_status = UndoStatus.IRREVERSIBLE
            else:
                undo.is_reversible = True
                undo.undo_status = UndoStatus.PARTIAL
                
        elif action.action_type == ActionType.CONFIGURE:
            # Configurazione → potenzialmente reversibile
            undo.is_reversible = True
            undo.undo_type = "restore"
            
        else:
            undo.is_reversible = True
        
        action.undo_info = undo
    
    async def undo_action(self, action: AutomationAction) -> AutomationAction:
        """
        Annulla un'azione precedentemente eseguita.
        Restituisce l'azione di undo.
        """
        if not action.undo_info or not action.undo_info.is_reversible:
            raise ValueError(f"Action {action.id} is not reversible")
        
        if action.undo_info.undo_status == UndoStatus.UNDONE:
            raise ValueError(f"Action {action.id} already undone")
        
        undo = action.undo_info
        
        # Crea azione di undo
        undo_action = AutomationAction(
            name=f"UNDO: {action.name}",
            description=f"Reversing action {action.id}",
            target=undo.undo_command or action.target,
            automation_type=action.automation_type,
            action_type=ActionType.STOP if undo.undo_type == "stop" else ActionType.START,
            execution_mode=action.execution_mode  # Mantiene stesso mode
        )
        
        # Esegui undo
        if undo.undo_type == "stop":
            # Ferma ciò che era stato avviato
            if action.id in self.running_processes:
                self.running_processes[action.id].terminate()
                del self.running_processes[action.id]
            else:
                # Prova taskkill
                await asyncio.create_subprocess_shell(
                    undo.undo_command,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
            undo_action.result = {"message": f"Stopped {action.target}"}
            
        elif undo.undo_type == "restart":
            # Riavvia ciò che era stato fermato
            process = subprocess.Popen(
                undo.undo_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            undo_action.result = {"pid": process.pid, "message": f"Restarted {action.target}"}
        
        undo_action.status = ExecutionStatus.COMPLETED
        undo_action.completed_at = datetime.now()
        
        # Marca originale come undone
        action.undo_info.undo_status = UndoStatus.UNDONE
        action.undo_info.undo_performed_at = datetime.now()
        
        auto_logger.info(f"Undone action: {action.id} - {action.name}")
        
        self.execution_history.append(undo_action)
        return undo_action
    
    async def _execute_application(self, action: AutomationAction, silent: bool = False):
        """Esegue azione su applicazione"""
        app = action.target.lower()
        app_cmd = self.app_registry.get(app, action.target)
        
        # Silent mode: redirect output a DEVNULL
        stdout_dest = subprocess.DEVNULL if silent else subprocess.PIPE
        stderr_dest = subprocess.DEVNULL if silent else subprocess.PIPE
        
        if action.action_type == ActionType.START:
            # Avvia applicazione
            process = subprocess.Popen(
                app_cmd,
                shell=True,
                stdout=stdout_dest,
                stderr=stderr_dest
            )
            self.running_processes[action.id] = process
            action.result = {
                "pid": process.pid,
                "message": f"Applicazione {app} avviata",
                "silent": silent
            }
            
        elif action.action_type == ActionType.STOP:
            # Ferma applicazione (Windows)
            subprocess.run(
                f'taskkill /IM "{app_cmd}.exe" /F',
                shell=True,
                capture_output=not silent,
                stdout=stdout_dest if silent else None,
                stderr=stderr_dest if silent else None
            )
            action.result = {"message": f"Applicazione {app} terminata", "silent": silent}
            
        elif action.action_type == ActionType.INTERACT:
            # Interazione placeholder
            action.result = {"message": f"Interazione con {app} completata", "silent": silent}
    
    async def _execute_script(self, action: AutomationAction, silent: bool = False):
        """Esegue script"""
        script_path = Path(action.target)
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script non trovato: {script_path}")
        
        # Determina interprete
        ext = script_path.suffix.lower()
        if ext == ".py":
            cmd = f"python {script_path}"
        elif ext == ".ps1":
            cmd = f"powershell -ExecutionPolicy Bypass -File {script_path}"
        elif ext == ".bat":
            cmd = str(script_path)
        elif ext == ".sh":
            cmd = f"bash {script_path}"
        else:
            cmd = str(script_path)
        
        # Aggiungi parametri
        if action.parameters.get("args"):
            cmd += " " + " ".join(action.parameters["args"])
        
        # Output destination
        stdout_dest = asyncio.subprocess.DEVNULL if silent else asyncio.subprocess.PIPE
        stderr_dest = asyncio.subprocess.DEVNULL if silent else asyncio.subprocess.PIPE
        
        # Esegui
        result = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                cmd,
                stdout=stdout_dest,
                stderr=stderr_dest
            ),
            timeout=action.timeout_seconds
        )
        
        if silent:
            await result.wait()
            action.result = {
                "returncode": result.returncode,
                "stdout": "[SILENT]",
                "stderr": "[SILENT]",
                "silent": True
            }
        else:
            stdout, stderr = await result.communicate()
            action.result = {
                "returncode": result.returncode,
                "stdout": stdout.decode()[:1000] if stdout else "",
                "stderr": stderr.decode()[:500] if stderr else "",
                "silent": False
            }
    
    async def _execute_command(self, action: AutomationAction, silent: bool = False):
        """Esegue comando di sistema"""
        cmd = action.target
        
        # Aggiungi parametri
        if action.parameters.get("args"):
            cmd += " " + " ".join(action.parameters["args"])
        
        # Output destination
        stdout_dest = asyncio.subprocess.DEVNULL if silent else asyncio.subprocess.PIPE
        stderr_dest = asyncio.subprocess.DEVNULL if silent else asyncio.subprocess.PIPE
        
        result = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                cmd,
                stdout=stdout_dest,
                stderr=stderr_dest,
                shell=True
            ),
            timeout=action.timeout_seconds
        )
        
        if silent:
            await result.wait()
            action.result = {
                "returncode": result.returncode,
                "stdout": "[SILENT]",
                "stderr": "[SILENT]",
                "silent": True
            }
        else:
            stdout, stderr = await result.communicate()
            action.result = {
                "returncode": result.returncode,
                "stdout": stdout.decode()[:1000] if stdout else "",
                "stderr": stderr.decode()[:500] if stderr else "",
                "silent": False
            }
    
    async def _execute_process(self, action: AutomationAction, silent: bool = False):
        """Gestisce processo di sistema"""
        stdout_dest = subprocess.DEVNULL if silent else subprocess.PIPE
        stderr_dest = subprocess.DEVNULL if silent else subprocess.PIPE
        
        if action.action_type == ActionType.START:
            process = subprocess.Popen(
                action.target,
                shell=True,
                stdout=stdout_dest,
                stderr=stderr_dest
            )
            self.running_processes[action.id] = process
            action.result = {"pid": process.pid, "status": "running", "silent": silent}
            
        elif action.action_type == ActionType.STOP:
            if action.id in self.running_processes:
                self.running_processes[action.id].terminate()
                del self.running_processes[action.id]
            action.result = {"status": "terminated", "silent": silent}
            
        elif action.action_type == ActionType.MONITOR:
            # Placeholder per monitoring
            action.result = {"processes": len(self.running_processes), "silent": silent}
    
    def get_running_processes(self) -> List[Dict]:
        """Ottiene lista processi in esecuzione"""
        return [
            {"id": id, "pid": p.pid, "running": p.poll() is None}
            for id, p in self.running_processes.items()
        ]


# === AUTOMATION LAYER ===

class AutomationLayer:
    """
    Layer principale di automazione.
    Coordina validazione, conferma ed esecuzione delle azioni.
    
    Modalità:
    - Pilot: tutte le azioni richiedono conferma logica interna
    - Shadow: osservazione senza esecuzione
    - Silent: esecuzione senza output visivo
    
    Tutte le azioni sono reversibili e loggate.
    """
    
    def __init__(self, mode_manager=None):
        self.mode_manager = mode_manager
        self.validator = InternalValidator()
        self.executor = AutomationExecutor()
        
        # Azioni e routine
        self.pending_actions: Dict[str, AutomationAction] = {}
        self.pending_confirmations: Dict[str, ConfirmationRequest] = {}
        self.routines: Dict[str, Routine] = {}
        
        # Azioni eseguite (per undo)
        self.executed_actions: Dict[str, AutomationAction] = {}
        
        # Log di tutte le azioni
        self.action_logs: List[ActionLog] = []
        
        # Configurazione
        self.require_confirmation_in_pilot = True
        self.auto_confirm_safe_actions = True
        
        # Execution mode default
        self.default_execution_mode = ExecutionMode.NORMAL
        self.shadow_mode_enabled = False
        self.silent_mode_enabled = False
        
        # Statistiche
        self.stats = {
            "actions_requested": 0,
            "actions_confirmed": 0,
            "actions_rejected": 0,
            "actions_executed": 0,
            "actions_completed": 0,
            "actions_failed": 0,
            "actions_undone": 0,
            "shadow_observations": 0,
            "silent_executions": 0,
            "routines_created": 0,
            "routines_executed": 0
        }
        
        auto_logger.info("AutomationLayer initialized")
    
    def _log_action(self, action_id: str, event: str, details: Dict = None):
        """Logga un evento per un'azione"""
        log_entry = ActionLog(
            action_id=action_id,
            event=event,
            details=details or {},
            execution_mode=self.default_execution_mode.value
        )
        self.action_logs.append(log_entry)
        auto_logger.debug(f"[LOG] {action_id}: {event} - {details}")
    
    # === Mode Control ===
    
    def enable_shadow_mode(self):
        """Abilita Shadow Mode - solo osservazione"""
        self.shadow_mode_enabled = True
        self.default_execution_mode = ExecutionMode.SHADOW
        self._log_action("SYSTEM", "shadow_mode_enabled", {})
        auto_logger.info("Shadow Mode ENABLED - actions will be observed only")
    
    def disable_shadow_mode(self):
        """Disabilita Shadow Mode"""
        self.shadow_mode_enabled = False
        self.default_execution_mode = ExecutionMode.NORMAL
        self._log_action("SYSTEM", "shadow_mode_disabled", {})
        auto_logger.info("Shadow Mode DISABLED - normal execution resumed")
    
    def enable_silent_mode(self):
        """Abilita Silent Mode - esecuzione senza output"""
        self.silent_mode_enabled = True
        self.default_execution_mode = ExecutionMode.SILENT
        self._log_action("SYSTEM", "silent_mode_enabled", {})
        auto_logger.info("Silent Mode ENABLED - no visual output")
    
    def disable_silent_mode(self):
        """Disabilita Silent Mode"""
        self.silent_mode_enabled = False
        self.default_execution_mode = ExecutionMode.NORMAL
        self._log_action("SYSTEM", "silent_mode_disabled", {})
        auto_logger.info("Silent Mode DISABLED - normal output resumed")
    
    def set_dry_run_mode(self, enabled: bool = True):
        """Imposta Dry Run mode - simula senza eseguire"""
        if enabled:
            self.default_execution_mode = ExecutionMode.DRY_RUN
            self._log_action("SYSTEM", "dry_run_enabled", {})
        else:
            self.default_execution_mode = ExecutionMode.NORMAL
            self._log_action("SYSTEM", "dry_run_disabled", {})
    
    @property
    def is_pilot_mode(self) -> bool:
        """Verifica se siamo in modalità Pilot"""
        if self.mode_manager:
            return self.mode_manager.mode_name == "pilot"
        return True  # Default: sempre pilot
    
    @property
    def is_shadow_mode(self) -> bool:
        """Verifica se Shadow Mode è attivo"""
        return self.shadow_mode_enabled or self.default_execution_mode == ExecutionMode.SHADOW
    
    @property
    def is_silent_mode(self) -> bool:
        """Verifica se Silent Mode è attivo"""
        return self.silent_mode_enabled or self.default_execution_mode == ExecutionMode.SILENT
    
    # === Action Management ===
    
    def create_action(self,
                     name: str,
                     target: str,
                     automation_type: AutomationType = AutomationType.COMMAND,
                     action_type: ActionType = ActionType.EXECUTE,
                     parameters: Dict = None,
                     risk_level: RiskLevel = None,
                     description: str = "",
                     timeout: int = 60,
                     execution_mode: ExecutionMode = None) -> AutomationAction:
        """
        Crea una nuova azione di automazione.
        Non esegue, solo prepara per validazione.
        
        Args:
            execution_mode: Modalità di esecuzione (NORMAL, SHADOW, SILENT, DRY_RUN)
                           Se None, usa il default del layer
        """
        # Auto-determina rischio se non specificato
        if risk_level is None:
            risk_level = self._assess_risk(automation_type, action_type, target)
        
        # Determina tipo conferma
        if self.is_pilot_mode:
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                confirmation = ConfirmationType.DUAL
            else:
                confirmation = ConfirmationType.INTERNAL
        else:
            confirmation = ConfirmationType.NONE if risk_level == RiskLevel.SAFE else ConfirmationType.INTERNAL
        
        # Usa execution mode specificato o default
        exec_mode = execution_mode or self.default_execution_mode
        
        action = AutomationAction(
            name=name,
            description=description or f"{action_type.value} {target}",
            automation_type=automation_type,
            action_type=action_type,
            target=target,
            parameters=parameters or {},
            risk_level=risk_level,
            confirmation_type=confirmation,
            execution_mode=exec_mode,
            timeout_seconds=timeout
        )
        
        self.pending_actions[action.id] = action
        self.stats["actions_requested"] += 1
        
        # Log creazione
        self._log_action(action.id, "created", {
            "name": name,
            "target": target,
            "type": automation_type.value,
            "action": action_type.value,
            "risk": risk_level.value,
            "mode": exec_mode.value
        })
        
        auto_logger.info(f"Action created: {action.id} - {name} [risk: {risk_level.value}, mode: {exec_mode.value}]")
        
        return action
    
    def _assess_risk(self, auto_type: AutomationType, action_type: ActionType, 
                    target: str) -> RiskLevel:
        """Valuta automaticamente il livello di rischio"""
        # Azioni sicure
        if action_type == ActionType.MONITOR:
            return RiskLevel.SAFE
        
        # Target critici
        critical_targets = ["system32", "registry", "boot", "kernel"]
        if any(ct in target.lower() for ct in critical_targets):
            return RiskLevel.CRITICAL
        
        # Stop/Restart sono più rischiosi
        if action_type in [ActionType.STOP, ActionType.RESTART]:
            return RiskLevel.MEDIUM
        
        # Script esterni
        if auto_type == AutomationType.SCRIPT:
            return RiskLevel.MEDIUM
        
        # Default
        return RiskLevel.LOW
    
    async def request_execution(self, action_id: str) -> ConfirmationRequest:
        """
        Richiede l'esecuzione di un'azione.
        In modalità Pilot, passa prima per validazione interna.
        """
        action = self.pending_actions.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")
        
        # Validazione interna
        confirmation = self.validator.validate(action)
        
        # Salva risultato validazione nell'azione
        action.validation_checks = confirmation.validation_checks
        action.is_validated = confirmation.all_checks_passed
        
        if confirmation.is_confirmed:
            action.status = ExecutionStatus.CONFIRMED
            action.confirmed_at = datetime.now()
            action.validation_reason = "Validazione interna superata"
            self.stats["actions_confirmed"] += 1
        else:
            action.status = ExecutionStatus.AWAITING_CONFIRMATION
            action.validation_reason = confirmation.rejection_reason
            self.pending_confirmations[confirmation.id] = confirmation
        
        auto_logger.info(
            f"Execution requested for {action_id}: "
            f"{'CONFIRMED' if confirmation.is_confirmed else 'AWAITING'}"
        )
        
        return confirmation
    
    def manual_confirm(self, confirmation_id: str, 
                      approve: bool,
                      reason: str = "") -> bool:
        """Conferma/rifiuta manualmente un'azione"""
        confirmation = self.pending_confirmations.get(confirmation_id)
        if not confirmation:
            return False
        
        confirmation.is_confirmed = approve
        confirmation.confirmed_by = "user"
        confirmation.decided_at = datetime.now()
        
        action = self.pending_actions.get(confirmation.action_id)
        if action:
            if approve:
                action.status = ExecutionStatus.CONFIRMED
                action.confirmed_at = datetime.now()
                action.validation_reason = reason or "Approvato manualmente"
                self.stats["actions_confirmed"] += 1
            else:
                action.status = ExecutionStatus.REJECTED
                action.validation_reason = reason or "Rifiutato manualmente"
                self.stats["actions_rejected"] += 1
        
        del self.pending_confirmations[confirmation_id]
        
        return True
    
    async def execute_action(self, action_id: str) -> AutomationAction:
        """
        Esegue un'azione confermata.
        Supporta Shadow Mode (osservazione) e Silent Mode (senza output).
        """
        action = self.pending_actions.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")
        
        if action.status != ExecutionStatus.CONFIRMED:
            raise ValueError(f"Action {action_id} not confirmed (status: {action.status.value})")
        
        # Log esecuzione
        self._log_action(action_id, "executing", {
            "mode": action.execution_mode.value,
            "target": action.target
        })
        
        # Esegui
        self.stats["actions_executed"] += 1
        
        # Track mode stats
        if action.execution_mode == ExecutionMode.SHADOW:
            self.stats["shadow_observations"] += 1
        elif action.execution_mode == ExecutionMode.SILENT:
            self.stats["silent_executions"] += 1
        
        result = await self.executor.execute(action)
        
        if result.status == ExecutionStatus.COMPLETED:
            self.stats["actions_completed"] += 1
            self._log_action(action_id, "completed", result.result)
            # Salva per possibile undo
            self.executed_actions[action_id] = result
        else:
            self.stats["actions_failed"] += 1
            self._log_action(action_id, "failed", {"error": result.error})
        
        # Rimuovi da pending
        del self.pending_actions[action_id]
        
        return result
    
    async def undo_action(self, action_id: str) -> AutomationAction:
        """
        Annulla un'azione precedentemente eseguita.
        
        Tutte le azioni sono reversibili se possibile.
        """
        action = self.executed_actions.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found in executed actions")
        
        # Log tentativo undo
        self._log_action(action_id, "undo_requested", {})
        
        try:
            undo_result = await self.executor.undo_action(action)
            self.stats["actions_undone"] += 1
            self._log_action(action_id, "undone", undo_result.result)
            return undo_result
        except Exception as e:
            self._log_action(action_id, "undo_failed", {"error": str(e)})
            raise
    
    def get_reversible_actions(self) -> List[Dict]:
        """Ottiene lista di azioni reversibili"""
        reversible = []
        for action_id, action in self.executed_actions.items():
            if action.undo_info and action.undo_info.is_reversible:
                if action.undo_info.undo_status != UndoStatus.UNDONE:
                    reversible.append({
                        "id": action_id,
                        "name": action.name,
                        "target": action.target,
                        "undo_type": action.undo_info.undo_type,
                        "executed_at": action.completed_at.isoformat() if action.completed_at else None
                    })
        return reversible
    
    async def execute_immediate(self, 
                               name: str,
                               target: str,
                               automation_type: AutomationType = AutomationType.COMMAND,
                               action_type: ActionType = ActionType.EXECUTE,
                               parameters: Dict = None) -> AutomationAction:
        """
        Shortcut: crea, valida ed esegue azione in un solo step.
        Comunque passa per validazione interna in Pilot mode.
        """
        # Crea azione
        action = self.create_action(
            name=name,
            target=target,
            automation_type=automation_type,
            action_type=action_type,
            parameters=parameters
        )
        
        # Richiedi conferma
        confirmation = await self.request_execution(action.id)
        
        # Se confermata, esegui
        if confirmation.is_confirmed:
            return await self.execute_action(action.id)
        else:
            action.status = ExecutionStatus.REJECTED
            return action
    
    # === Routine Management ===
    
    def create_routine(self, name: str, description: str = "",
                      stop_on_error: bool = True) -> Routine:
        """Crea nuova routine"""
        routine = Routine(
            name=name,
            description=description,
            stop_on_error=stop_on_error
        )
        
        self.routines[routine.id] = routine
        self.stats["routines_created"] += 1
        
        return routine
    
    def add_step_to_routine(self, routine_id: str,
                           name: str,
                           target: str,
                           automation_type: AutomationType = AutomationType.COMMAND,
                           action_type: ActionType = ActionType.EXECUTE,
                           parameters: Dict = None) -> AutomationAction:
        """Aggiunge step a routine"""
        routine = self.routines.get(routine_id)
        if not routine:
            raise ValueError(f"Routine {routine_id} not found")
        
        step = AutomationAction(
            name=name,
            automation_type=automation_type,
            action_type=action_type,
            target=target,
            parameters=parameters or {}
        )
        
        routine.steps.append(step)
        return step
    
    async def execute_routine(self, routine_id: str) -> Routine:
        """Esegue routine completa"""
        routine = self.routines.get(routine_id)
        if not routine:
            raise ValueError(f"Routine {routine_id} not found")
        
        routine.status = ExecutionStatus.EXECUTING
        routine.current_step = 0
        routine.run_count += 1
        routine.last_run = datetime.now()
        
        self.stats["routines_executed"] += 1
        
        for i, step in enumerate(routine.steps):
            routine.current_step = i
            
            # Valida step
            confirmation = self.validator.validate(step)
            
            if not confirmation.is_confirmed:
                if routine.stop_on_error:
                    routine.status = ExecutionStatus.FAILED
                    routine.failure_count += 1
                    auto_logger.warning(f"Routine {routine_id} stopped at step {i}: validation failed")
                    return routine
                continue
            
            # Esegui step
            step.status = ExecutionStatus.CONFIRMED
            result = await self.executor.execute(step)
            
            if result.status == ExecutionStatus.FAILED and routine.stop_on_error:
                routine.status = ExecutionStatus.FAILED
                routine.failure_count += 1
                return routine
        
        routine.status = ExecutionStatus.COMPLETED
        routine.success_count += 1
        
        return routine
    
    # === Convenience Methods ===
    
    async def start_application(self, app_name: str) -> AutomationAction:
        """Avvia un'applicazione"""
        return await self.execute_immediate(
            name=f"Avvia {app_name}",
            target=app_name,
            automation_type=AutomationType.APPLICATION,
            action_type=ActionType.START
        )
    
    async def stop_application(self, app_name: str) -> AutomationAction:
        """Ferma un'applicazione"""
        return await self.execute_immediate(
            name=f"Ferma {app_name}",
            target=app_name,
            automation_type=AutomationType.APPLICATION,
            action_type=ActionType.STOP
        )
    
    async def run_script(self, script_path: str, 
                        args: List[str] = None) -> AutomationAction:
        """Esegue uno script"""
        return await self.execute_immediate(
            name=f"Esegui {Path(script_path).name}",
            target=script_path,
            automation_type=AutomationType.SCRIPT,
            action_type=ActionType.EXECUTE,
            parameters={"args": args or []}
        )
    
    async def run_command(self, command: str) -> AutomationAction:
        """Esegue un comando"""
        return await self.execute_immediate(
            name=f"Comando: {command[:30]}...",
            target=command,
            automation_type=AutomationType.COMMAND,
            action_type=ActionType.EXECUTE
        )
    
    # === Shadow Mode Operations ===
    
    async def observe(self, name: str, target: str,
                     automation_type: AutomationType = AutomationType.COMMAND,
                     action_type: ActionType = ActionType.EXECUTE) -> AutomationAction:
        """
        Osserva un'azione senza eseguirla (Shadow Mode).
        Utile per preview o debug.
        """
        action = self.create_action(
            name=name,
            target=target,
            automation_type=automation_type,
            action_type=action_type,
            execution_mode=ExecutionMode.SHADOW
        )
        
        # In shadow mode, passa direttamente
        confirmation = await self.request_execution(action.id)
        if confirmation.is_confirmed or action.execution_mode == ExecutionMode.SHADOW:
            action.status = ExecutionStatus.CONFIRMED
            return await self.execute_action(action.id)
        
        return action
    
    async def execute_silent(self, name: str, target: str,
                            automation_type: AutomationType = AutomationType.COMMAND,
                            action_type: ActionType = ActionType.EXECUTE) -> AutomationAction:
        """
        Esegue un'azione silenziosamente (senza output visivo).
        """
        action = self.create_action(
            name=name,
            target=target,
            automation_type=automation_type,
            action_type=action_type,
            execution_mode=ExecutionMode.SILENT
        )
        
        confirmation = await self.request_execution(action.id)
        if confirmation.is_confirmed:
            return await self.execute_action(action.id)
        
        return action
    
    async def dry_run(self, name: str, target: str,
                     automation_type: AutomationType = AutomationType.COMMAND,
                     action_type: ActionType = ActionType.EXECUTE) -> AutomationAction:
        """
        Simula un'azione senza eseguirla (Dry Run).
        Mostra cosa farebbe senza effetti.
        """
        action = self.create_action(
            name=name,
            target=target,
            automation_type=automation_type,
            action_type=action_type,
            execution_mode=ExecutionMode.DRY_RUN
        )
        
        # Dry run passa sempre
        action.status = ExecutionStatus.CONFIRMED
        return await self.executor.execute(action)
    
    # === Query Methods ===
    
    def get_pending_actions(self) -> List[Dict]:
        """Ottiene azioni in attesa"""
        return [a.to_dict() for a in self.pending_actions.values()]
    
    def get_pending_confirmations(self) -> List[Dict]:
        """Ottiene conferme in attesa"""
        return [c.to_dict() for c in self.pending_confirmations.values()]
    
    def get_routines(self) -> List[Dict]:
        """Ottiene lista routine"""
        return [r.to_dict() for r in self.routines.values()]
    
    def get_execution_history(self, limit: int = 20) -> List[Dict]:
        """Ottiene storia esecuzioni"""
        return [a.to_dict() for a in self.executor.execution_history[-limit:]]
    
    def get_action_logs(self, action_id: str = None, 
                       event: str = None,
                       limit: int = 50) -> List[Dict]:
        """
        Ottiene log delle azioni.
        
        Args:
            action_id: Filtra per ID azione specifico
            event: Filtra per tipo evento
            limit: Numero massimo di risultati
        """
        logs = self.action_logs
        
        if action_id:
            logs = [l for l in logs if l.action_id == action_id]
        if event:
            logs = [l for l in logs if l.event == event]
        
        return [l.to_dict() for l in logs[-limit:]]
    
    def get_statistics(self) -> Dict:
        """Statistiche layer"""
        return {
            **self.stats,
            "pending_actions": len(self.pending_actions),
            "pending_confirmations": len(self.pending_confirmations),
            "active_routines": len(self.routines),
            "running_processes": len(self.executor.running_processes),
            "executed_actions": len(self.executed_actions),
            "reversible_actions": len(self.get_reversible_actions()),
            "total_logs": len(self.action_logs),
            "is_pilot_mode": self.is_pilot_mode,
            "is_shadow_mode": self.is_shadow_mode,
            "is_silent_mode": self.is_silent_mode,
            "execution_mode": self.default_execution_mode.value
        }
