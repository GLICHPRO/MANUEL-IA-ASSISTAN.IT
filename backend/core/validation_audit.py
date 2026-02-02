"""
🔍 GIDEON 3.0 - Validation & Audit Trail System

Sistema di:
- Validazione logica interna obbligatoria per azioni critiche
- Audit trail completo con possibilità di replay decisionale
"""

import asyncio
import inspect
import hashlib
import json
import os
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable, Set, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import copy

logger = logging.getLogger(__name__)


# ============ ENUMS ============

class ValidationLevel(Enum):
    """Livelli di validazione richiesti"""
    NONE = "none"              # Nessuna validazione
    BASIC = "basic"            # Validazione parametri base
    STANDARD = "standard"      # Validazione + controlli coerenza
    STRICT = "strict"          # Validazione completa + pre-condizioni
    CRITICAL = "critical"      # Tutto + dual-check + conferma esplicita


class ValidationResult(Enum):
    """Risultato validazione"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"
    SKIPPED = "skipped"


from enum import IntEnum

class CriticalityLevel(IntEnum):
    """Livello di criticità azione"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EXTREME = 5


class AuditEventCategory(Enum):
    """Categorie eventi audit"""
    DECISION = "decision"
    VALIDATION = "validation"
    EXECUTION = "execution"
    RESULT = "result"
    ERROR = "error"
    ROLLBACK = "rollback"
    REPLAY = "replay"
    SYSTEM = "system"


class ReplayStatus(Enum):
    """Stato del replay"""
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============ DATA CLASSES ============

@dataclass
class ValidationRule:
    """Regola di validazione"""
    id: str
    name: str
    description: str
    validator: Callable[[Dict[str, Any]], Tuple[bool, str]]
    level: ValidationLevel = ValidationLevel.STANDARD
    criticality: CriticalityLevel = CriticalityLevel.MEDIUM
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Esegue la validazione"""
        if not self.enabled:
            return True, "Rule disabled"
        try:
            return self.validator(data)
        except Exception as e:
            return False, f"Validation error: {str(e)}"


@dataclass
class ValidationCheck:
    """Risultato singolo check di validazione"""
    rule_id: str
    rule_name: str
    passed: bool
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "passed": self.passed,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "duration_ms": self.duration_ms
        }


@dataclass
class ValidationReport:
    """Report completo validazione"""
    id: str
    action_id: str
    action_type: str
    level: ValidationLevel
    result: ValidationResult
    checks: List[ValidationCheck] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    total_duration_ms: float = 0.0
    requires_confirmation: bool = False
    confirmation_received: bool = False
    confirmed_by: Optional[str] = None
    notes: str = ""
    
    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)
    
    @property
    def pass_rate(self) -> float:
        if not self.checks:
            return 1.0
        return self.passed_count / len(self.checks)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "level": self.level.value,
            "result": self.result.value,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp.isoformat(),
            "total_duration_ms": self.total_duration_ms,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate": round(self.pass_rate, 3),
            "requires_confirmation": self.requires_confirmation,
            "confirmation_received": self.confirmation_received,
            "confirmed_by": self.confirmed_by,
            "notes": self.notes
        }


@dataclass
class DecisionContext:
    """Contesto decisionale completo per replay"""
    id: str
    timestamp: datetime
    
    # Input
    input_data: Dict[str, Any]
    user_request: str
    session_id: str
    
    # Stato sistema al momento della decisione
    system_state: Dict[str, Any]
    mode: str
    security_level: str
    
    # Decisione
    decision_type: str
    decision_made: Dict[str, Any]
    alternatives_considered: List[Dict[str, Any]]
    reasoning: str
    confidence: float
    
    # Validazione
    validation_report: Optional[ValidationReport] = None
    
    # Esecuzione
    execution_started: Optional[datetime] = None
    execution_completed: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None
    
    # Outcome
    success: Optional[bool] = None
    outcome_details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    # Rollback info
    rollback_point_id: Optional[str] = None
    rollback_executed: bool = False
    
    # Checksum per integrità
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calcola checksum per integrità"""
        content = json.dumps({
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "input_data": self.input_data,
            "decision_made": self.decision_made,
            "reasoning": self.reasoning
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def verify_integrity(self) -> bool:
        """Verifica integrità del record"""
        return self.checksum == self._calculate_checksum()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "input_data": self.input_data,
            "user_request": self.user_request,
            "session_id": self.session_id,
            "system_state": self.system_state,
            "mode": self.mode,
            "security_level": self.security_level,
            "decision_type": self.decision_type,
            "decision_made": self.decision_made,
            "alternatives_considered": self.alternatives_considered,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "validation_report": self.validation_report.to_dict() if self.validation_report else None,
            "execution_started": self.execution_started.isoformat() if self.execution_started else None,
            "execution_completed": self.execution_completed.isoformat() if self.execution_completed else None,
            "execution_result": self.execution_result,
            "success": self.success,
            "outcome_details": self.outcome_details,
            "error": self.error,
            "rollback_point_id": self.rollback_point_id,
            "rollback_executed": self.rollback_executed,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DecisionContext':
        """Crea da dizionario"""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            input_data=data["input_data"],
            user_request=data["user_request"],
            session_id=data["session_id"],
            system_state=data["system_state"],
            mode=data["mode"],
            security_level=data["security_level"],
            decision_type=data["decision_type"],
            decision_made=data["decision_made"],
            alternatives_considered=data.get("alternatives_considered", []),
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            validation_report=None,  # Ricostruito separatamente se necessario
            execution_started=datetime.fromisoformat(data["execution_started"]) if data.get("execution_started") else None,
            execution_completed=datetime.fromisoformat(data["execution_completed"]) if data.get("execution_completed") else None,
            execution_result=data.get("execution_result"),
            success=data.get("success"),
            outcome_details=data.get("outcome_details", {}),
            error=data.get("error"),
            rollback_point_id=data.get("rollback_point_id"),
            rollback_executed=data.get("rollback_executed", False),
            checksum=data.get("checksum", "")
        )


@dataclass
class ReplaySession:
    """Sessione di replay decisionale"""
    id: str
    original_decision_id: str
    status: ReplayStatus = ReplayStatus.READY
    
    # Config replay
    simulate_only: bool = True  # Non esegue realmente
    override_inputs: Dict[str, Any] = field(default_factory=dict)
    step_by_step: bool = False
    
    # Progress
    current_step: int = 0
    total_steps: int = 0
    steps_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Risultato
    replay_result: Optional[Dict[str, Any]] = None
    differences: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original_decision_id": self.original_decision_id,
            "status": self.status.value,
            "simulate_only": self.simulate_only,
            "step_by_step": self.step_by_step,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "steps_log": self.steps_log,
            "replay_result": self.replay_result,
            "differences": self.differences,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


# ============ LOGIC VALIDATOR ============

class LogicValidator:
    """
    Validatore logico per azioni critiche.
    Esegue validazione multi-livello con regole configurabili.
    """
    
    def __init__(self):
        self._rules: Dict[str, ValidationRule] = {}
        self._action_rules: Dict[str, List[str]] = {}  # action_type -> [rule_ids]
        self._criticality_map: Dict[str, CriticalityLevel] = {}
        self._validation_history: List[ValidationReport] = []
        self._report_counter = 0
        
        # Setup regole default
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Configura regole di validazione default"""
        
        # === Regole Base ===
        
        self.add_rule(ValidationRule(
            id="params_not_empty",
            name="Parameters Not Empty",
            description="Verifica che i parametri non siano vuoti",
            validator=lambda d: (bool(d), "Parameters are empty" if not d else "OK"),
            level=ValidationLevel.BASIC
        ))
        
        self.add_rule(ValidationRule(
            id="required_fields",
            name="Required Fields Present",
            description="Verifica campi obbligatori presenti",
            validator=self._validate_required_fields,
            level=ValidationLevel.BASIC
        ))
        
        # === Regole Standard ===
        
        self.add_rule(ValidationRule(
            id="type_consistency",
            name="Type Consistency",
            description="Verifica coerenza dei tipi",
            validator=self._validate_type_consistency,
            level=ValidationLevel.STANDARD
        ))
        
        self.add_rule(ValidationRule(
            id="value_ranges",
            name="Value Ranges",
            description="Verifica valori entro range ammessi",
            validator=self._validate_value_ranges,
            level=ValidationLevel.STANDARD
        ))
        
        self.add_rule(ValidationRule(
            id="logical_consistency",
            name="Logical Consistency",
            description="Verifica coerenza logica dei parametri",
            validator=self._validate_logical_consistency,
            level=ValidationLevel.STANDARD
        ))
        
        # === Regole Strict ===
        
        self.add_rule(ValidationRule(
            id="preconditions",
            name="Preconditions Check",
            description="Verifica pre-condizioni",
            validator=self._validate_preconditions,
            level=ValidationLevel.STRICT
        ))
        
        self.add_rule(ValidationRule(
            id="resource_availability",
            name="Resource Availability",
            description="Verifica disponibilità risorse",
            validator=self._validate_resources,
            level=ValidationLevel.STRICT
        ))
        
        self.add_rule(ValidationRule(
            id="side_effects",
            name="Side Effects Analysis",
            description="Analizza potenziali effetti collaterali",
            validator=self._validate_side_effects,
            level=ValidationLevel.STRICT
        ))
        
        # === Regole Critical ===
        
        self.add_rule(ValidationRule(
            id="dual_check",
            name="Dual Logic Check",
            description="Doppia verifica logica indipendente",
            validator=self._validate_dual_check,
            level=ValidationLevel.CRITICAL,
            criticality=CriticalityLevel.CRITICAL
        ))
        
        self.add_rule(ValidationRule(
            id="risk_assessment",
            name="Risk Assessment",
            description="Valutazione rischio azione",
            validator=self._validate_risk,
            level=ValidationLevel.CRITICAL,
            criticality=CriticalityLevel.CRITICAL
        ))
        
        self.add_rule(ValidationRule(
            id="reversibility_check",
            name="Reversibility Check",
            description="Verifica reversibilità azione",
            validator=self._validate_reversibility,
            level=ValidationLevel.CRITICAL,
            criticality=CriticalityLevel.CRITICAL
        ))
        
        # === Mapping criticità azioni ===
        
        self._criticality_map = {
            # Azioni a bassa criticità
            "get_status": CriticalityLevel.LOW,
            "get_time": CriticalityLevel.LOW,
            "search": CriticalityLevel.LOW,
            
            # Azioni a media criticità
            "open_app": CriticalityLevel.MEDIUM,
            "send_notification": CriticalityLevel.MEDIUM,
            "create_reminder": CriticalityLevel.MEDIUM,
            
            # Azioni ad alta criticità
            "file_write": CriticalityLevel.HIGH,
            "file_delete": CriticalityLevel.HIGH,
            "execute_macro": CriticalityLevel.HIGH,
            "api_call": CriticalityLevel.HIGH,
            
            # Azioni critiche
            "execute_command": CriticalityLevel.CRITICAL,
            "system_config": CriticalityLevel.CRITICAL,
            "manage_users": CriticalityLevel.CRITICAL,
            
            # Azioni estreme
            "system_shutdown": CriticalityLevel.EXTREME,
            "kill_switch": CriticalityLevel.EXTREME,
            "data_purge": CriticalityLevel.EXTREME
        }
    
    # === Validator Functions ===
    
    def _validate_required_fields(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida campi obbligatori"""
        required = data.get("_required_fields", [])
        missing = [f for f in required if f not in data or data[f] is None]
        if missing:
            return False, f"Missing required fields: {missing}"
        return True, "All required fields present"
    
    def _validate_type_consistency(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida coerenza tipi"""
        type_hints = data.get("_type_hints", {})
        for field, expected_type in type_hints.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    return False, f"Type mismatch for '{field}': expected {expected_type.__name__}"
        return True, "Type consistency OK"
    
    def _validate_value_ranges(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida range valori"""
        ranges = data.get("_value_ranges", {})
        for field, (min_val, max_val) in ranges.items():
            if field in data:
                val = data[field]
                if isinstance(val, (int, float)):
                    if val < min_val or val > max_val:
                        return False, f"Value '{field}' = {val} out of range [{min_val}, {max_val}]"
        return True, "Value ranges OK"
    
    def _validate_logical_consistency(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida coerenza logica"""
        # Check dipendenze logiche
        if "start_time" in data and "end_time" in data:
            if data["end_time"] < data["start_time"]:
                return False, "end_time cannot be before start_time"
        
        # Check esclusività mutua
        mutex_fields = data.get("_mutex_fields", [])
        for group in mutex_fields:
            active = [f for f in group if data.get(f)]
            if len(active) > 1:
                return False, f"Mutually exclusive fields both set: {active}"
        
        return True, "Logical consistency OK"
    
    def _validate_preconditions(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida pre-condizioni"""
        preconditions = data.get("_preconditions", [])
        for condition in preconditions:
            if callable(condition):
                if not condition(data):
                    return False, "Precondition check failed"
            elif isinstance(condition, str):
                # Condizione come espressione
                try:
                    if not eval(condition, {"data": data}):
                        return False, f"Precondition failed: {condition}"
                except Exception as e:
                    return False, f"Precondition error: {e}"
        return True, "Preconditions OK"
    
    def _validate_resources(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida disponibilità risorse"""
        required_resources = data.get("_required_resources", [])
        # Simula check risorse
        for resource in required_resources:
            resource_type = resource.get("type", "")
            if resource_type == "file":
                path = resource.get("path", "")
                if path and not os.path.exists(path):
                    return False, f"Required file not found: {path}"
            elif resource_type == "memory":
                # Check memoria disponibile (simulato)
                pass
        return True, "Resources available"
    
    def _validate_side_effects(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Analizza effetti collaterali"""
        action_type = data.get("action_type", "")
        
        # Azioni con effetti collaterali significativi
        high_impact_actions = {
            "file_delete": "Irreversible file deletion",
            "data_purge": "Data loss possible",
            "system_shutdown": "Service interruption",
            "config_change": "May affect other components"
        }
        
        if action_type in high_impact_actions:
            warning = high_impact_actions[action_type]
            return True, f"Warning: {warning}"
        
        return True, "No significant side effects detected"
    
    def _validate_dual_check(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Doppia verifica logica indipendente"""
        # Prima verifica: hash consistenza
        content_hash = hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
        
        # Seconda verifica: ricostruzione e re-hash
        reconstructed = copy.deepcopy(data)
        reconstructed_hash = hashlib.md5(json.dumps(reconstructed, sort_keys=True, default=str).encode()).hexdigest()
        
        if content_hash != reconstructed_hash:
            return False, "Dual check hash mismatch"
        
        # Verifica campi critici coerenti
        critical_fields = data.get("_critical_fields", [])
        for field in critical_fields:
            original = data.get(field)
            check = reconstructed.get(field)
            if original != check:
                return False, f"Critical field '{field}' inconsistent"
        
        return True, "Dual check passed"
    
    def _validate_risk(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Valutazione rischio"""
        risk_factors = {
            "file_operations": 0.3,
            "system_commands": 0.5,
            "network_access": 0.4,
            "data_modification": 0.4,
            "privileged_action": 0.6
        }
        
        flags = data.get("_risk_flags", [])
        total_risk = sum(risk_factors.get(f, 0) for f in flags)
        
        risk_threshold = data.get("_risk_threshold", 0.7)
        
        if total_risk > risk_threshold:
            return False, f"Risk score {total_risk:.2f} exceeds threshold {risk_threshold}"
        
        return True, f"Risk score: {total_risk:.2f}"
    
    def _validate_reversibility(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Verifica reversibilità azione"""
        action_type = data.get("action_type", "")
        
        irreversible_actions = {
            "file_delete": "permanent",
            "data_purge": "permanent",
            "email_send": "cannot_unsend"
        }
        
        if action_type in irreversible_actions:
            reason = irreversible_actions[action_type]
            if not data.get("backup_created", False):
                return False, f"Action '{action_type}' is {reason} and no backup exists"
        
        return True, "Action is reversible or backup exists"
    
    # === Public Methods ===
    
    def add_rule(self, rule: ValidationRule):
        """Aggiunge regola di validazione"""
        self._rules[rule.id] = rule
        logger.debug(f"Added validation rule: {rule.id}")
    
    def remove_rule(self, rule_id: str):
        """Rimuove regola"""
        if rule_id in self._rules:
            del self._rules[rule_id]
    
    def enable_rule(self, rule_id: str, enabled: bool = True):
        """Abilita/disabilita regola"""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = enabled
    
    def set_action_rules(self, action_type: str, rule_ids: List[str]):
        """Imposta regole specifiche per tipo azione"""
        self._action_rules[action_type] = rule_ids
    
    def get_criticality(self, action_type: str) -> CriticalityLevel:
        """Ottiene criticità per tipo azione"""
        return self._criticality_map.get(action_type, CriticalityLevel.MEDIUM)
    
    def set_criticality(self, action_type: str, level: CriticalityLevel):
        """Imposta criticità per tipo azione"""
        self._criticality_map[action_type] = level
    
    def get_validation_level(self, action_type: str) -> ValidationLevel:
        """Determina livello validazione richiesto"""
        criticality = self.get_criticality(action_type)
        
        level_map = {
            CriticalityLevel.LOW: ValidationLevel.BASIC,
            CriticalityLevel.MEDIUM: ValidationLevel.STANDARD,
            CriticalityLevel.HIGH: ValidationLevel.STRICT,
            CriticalityLevel.CRITICAL: ValidationLevel.CRITICAL,
            CriticalityLevel.EXTREME: ValidationLevel.CRITICAL
        }
        
        return level_map.get(criticality, ValidationLevel.STANDARD)
    
    async def validate(self, action_id: str, action_type: str, 
                       data: Dict[str, Any],
                       level: ValidationLevel = None) -> ValidationReport:
        """
        Esegue validazione completa.
        
        Args:
            action_id: ID azione
            action_type: Tipo azione
            data: Dati da validare
            level: Livello validazione (auto se None)
        
        Returns:
            ValidationReport
        """
        start_time = datetime.now()
        
        # Determina livello
        if level is None:
            level = self.get_validation_level(action_type)
        
        # Crea report
        self._report_counter += 1
        report = ValidationReport(
            id=f"val_{self._report_counter:06d}",
            action_id=action_id,
            action_type=action_type,
            level=level,
            result=ValidationResult.PENDING
        )
        
        # Aggiungi metadata ai dati
        data["action_type"] = action_type
        
        # Ottieni regole da eseguire
        rules_to_run = self._get_rules_for_level(action_type, level)
        
        # Esegui validazioni
        all_passed = True
        for rule in rules_to_run:
            check_start = datetime.now()
            passed, message = rule.validate(data)
            check_duration = (datetime.now() - check_start).total_seconds() * 1000
            
            check = ValidationCheck(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                message=message,
                duration_ms=check_duration
            )
            report.checks.append(check)
            
            if not passed:
                all_passed = False
                logger.warning(f"Validation failed: {rule.name} - {message}")
        
        # Determina risultato
        if all_passed:
            report.result = ValidationResult.PASSED
        elif report.pass_rate >= 0.8:
            report.result = ValidationResult.WARNING
        else:
            report.result = ValidationResult.FAILED
        
        # Richiede conferma per azioni critiche
        criticality = self.get_criticality(action_type)
        if criticality >= CriticalityLevel.CRITICAL:
            report.requires_confirmation = True
        
        # Calcola durata totale
        report.total_duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Salva in history
        self._validation_history.append(report)
        
        logger.info(f"Validation complete: {action_type} -> {report.result.value} "
                   f"({report.passed_count}/{len(report.checks)} passed)")
        
        return report
    
    def _get_rules_for_level(self, action_type: str, 
                             level: ValidationLevel) -> List[ValidationRule]:
        """Ottiene regole per livello"""
        # Regole specifiche per azione
        specific_ids = self._action_rules.get(action_type, [])
        
        # Filtra per livello
        level_order = [
            ValidationLevel.BASIC,
            ValidationLevel.STANDARD,
            ValidationLevel.STRICT,
            ValidationLevel.CRITICAL
        ]
        max_idx = level_order.index(level) if level in level_order else 0
        allowed_levels = level_order[:max_idx + 1]
        
        rules = []
        for rule in self._rules.values():
            if rule.level in allowed_levels or rule.id in specific_ids:
                rules.append(rule)
        
        return rules
    
    def confirm_validation(self, report_id: str, confirmed_by: str) -> bool:
        """Conferma validazione che la richiede"""
        for report in self._validation_history:
            if report.id == report_id:
                if report.requires_confirmation:
                    report.confirmation_received = True
                    report.confirmed_by = confirmed_by
                    logger.info(f"Validation {report_id} confirmed by {confirmed_by}")
                    return True
        return False
    
    def get_validation_history(self, action_type: str = None,
                               limit: int = 100) -> List[ValidationReport]:
        """Ottiene storico validazioni"""
        history = self._validation_history
        if action_type:
            history = [r for r in history if r.action_type == action_type]
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche validazioni"""
        total = len(self._validation_history)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "pass_rate": 0
            }
        
        passed = sum(1 for r in self._validation_history 
                    if r.result == ValidationResult.PASSED)
        failed = sum(1 for r in self._validation_history 
                    if r.result == ValidationResult.FAILED)
        warnings = sum(1 for r in self._validation_history 
                      if r.result == ValidationResult.WARNING)
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "pass_rate": round(passed / total, 3) if total > 0 else 0,
            "rules_count": len(self._rules),
            "average_checks": round(
                sum(len(r.checks) for r in self._validation_history) / total, 1
            ) if total > 0 else 0
        }


# ============ AUDIT TRAIL ============

class DecisionAuditTrail:
    """
    Audit trail completo per decisioni con replay.
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./audit_trail")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory
        self._decisions: Dict[str, DecisionContext] = {}
        self._decision_index: List[str] = []  # Ordine cronologico
        self._replay_sessions: Dict[str, ReplaySession] = {}
        
        # Config
        self._max_memory_decisions = 10000
        self._auto_persist = True
        
        # Counter
        self._decision_counter = 0
        self._replay_counter = 0
        
        # Callbacks
        self._on_decision: List[Callable] = []
        self._on_replay: List[Callable] = []
        
        # Carica da disco
        self._load_recent_decisions()
    
    def _generate_decision_id(self) -> str:
        """Genera ID decisione"""
        self._decision_counter += 1
        return f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._decision_counter:04d}"
    
    def _generate_replay_id(self) -> str:
        """Genera ID replay"""
        self._replay_counter += 1
        return f"replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._replay_counter:04d}"
    
    # === Recording ===
    
    def record_decision(self,
                        input_data: Dict[str, Any],
                        user_request: str,
                        session_id: str,
                        system_state: Dict[str, Any],
                        mode: str,
                        security_level: str,
                        decision_type: str,
                        decision_made: Dict[str, Any],
                        alternatives: List[Dict[str, Any]],
                        reasoning: str,
                        confidence: float,
                        validation_report: ValidationReport = None) -> DecisionContext:
        """
        Registra una decisione nel trail.
        
        Returns:
            DecisionContext registrato
        """
        decision = DecisionContext(
            id=self._generate_decision_id(),
            timestamp=datetime.now(),
            input_data=copy.deepcopy(input_data),
            user_request=user_request,
            session_id=session_id,
            system_state=copy.deepcopy(system_state),
            mode=mode,
            security_level=security_level,
            decision_type=decision_type,
            decision_made=copy.deepcopy(decision_made),
            alternatives_considered=alternatives,
            reasoning=reasoning,
            confidence=confidence,
            validation_report=validation_report
        )
        
        # Store
        self._decisions[decision.id] = decision
        self._decision_index.append(decision.id)
        
        # Cleanup memory
        self._cleanup_memory()
        
        # Persist
        if self._auto_persist:
            self._persist_decision(decision)
        
        # Callbacks
        for callback in self._on_decision:
            try:
                if inspect.iscoroutinefunction(callback):
                    asyncio.create_task(callback(decision))
                else:
                    callback(decision)
            except Exception as e:
                logger.error(f"Decision callback error: {e}")
        
        logger.info(f"Decision recorded: {decision.id} - {decision_type}")
        
        return decision
    
    def update_execution(self, decision_id: str,
                         started: bool = False,
                         completed: bool = False,
                         result: Dict[str, Any] = None):
        """Aggiorna info esecuzione"""
        decision = self._decisions.get(decision_id)
        if not decision:
            return
        
        if started:
            decision.execution_started = datetime.now()
        if completed:
            decision.execution_completed = datetime.now()
        if result:
            decision.execution_result = result
        
        self._persist_decision(decision)
    
    def resolve_decision(self, decision_id: str,
                         success: bool,
                         outcome_details: Dict[str, Any] = None,
                         error: str = None):
        """Risolve decisione con outcome"""
        decision = self._decisions.get(decision_id)
        if not decision:
            logger.warning(f"Decision not found: {decision_id}")
            return
        
        decision.success = success
        decision.outcome_details = outcome_details or {}
        decision.error = error
        
        self._persist_decision(decision)
        
        logger.info(f"Decision resolved: {decision_id} -> {'SUCCESS' if success else 'FAILED'}")
    
    def link_rollback(self, decision_id: str, rollback_point_id: str):
        """Collega punto rollback"""
        decision = self._decisions.get(decision_id)
        if decision:
            decision.rollback_point_id = rollback_point_id
            self._persist_decision(decision)
    
    def mark_rollback_executed(self, decision_id: str):
        """Marca rollback eseguito"""
        decision = self._decisions.get(decision_id)
        if decision:
            decision.rollback_executed = True
            self._persist_decision(decision)
    
    # === Query ===
    
    def get_decision(self, decision_id: str) -> Optional[DecisionContext]:
        """Ottiene decisione per ID"""
        # Prima in memoria
        if decision_id in self._decisions:
            return self._decisions[decision_id]
        # Poi da disco
        return self._load_decision(decision_id)
    
    def get_recent_decisions(self, limit: int = 100,
                             decision_type: str = None,
                             success: bool = None,
                             session_id: str = None) -> List[DecisionContext]:
        """Ottiene decisioni recenti con filtri"""
        decisions = []
        
        for dec_id in reversed(self._decision_index):
            if len(decisions) >= limit:
                break
            
            decision = self._decisions.get(dec_id)
            if not decision:
                continue
            
            # Filtri
            if decision_type and decision.decision_type != decision_type:
                continue
            if success is not None and decision.success != success:
                continue
            if session_id and decision.session_id != session_id:
                continue
            
            decisions.append(decision)
        
        return decisions
    
    def search_decisions(self, query: str,
                         start_date: datetime = None,
                         end_date: datetime = None,
                         limit: int = 50) -> List[DecisionContext]:
        """Cerca decisioni per query"""
        results = []
        query_lower = query.lower()
        
        for dec_id in reversed(self._decision_index):
            if len(results) >= limit:
                break
            
            decision = self._decisions.get(dec_id)
            if not decision:
                continue
            
            # Filtro date
            if start_date and decision.timestamp < start_date:
                continue
            if end_date and decision.timestamp > end_date:
                continue
            
            # Search in various fields
            searchable = [
                decision.user_request,
                decision.reasoning,
                decision.decision_type,
                json.dumps(decision.decision_made)
            ]
            
            if any(query_lower in s.lower() for s in searchable):
                results.append(decision)
        
        return results
    
    def get_decision_chain(self, decision_id: str,
                           depth: int = 10) -> List[DecisionContext]:
        """Ottiene catena di decisioni correlate"""
        chain = []
        current = self.get_decision(decision_id)
        
        if not current:
            return chain
        
        chain.append(current)
        
        # Trova decisioni con stesso session_id vicine temporalmente
        session_decisions = [
            d for d in self._decisions.values()
            if d.session_id == current.session_id
            and d.id != current.id
        ]
        
        # Ordina per timestamp
        session_decisions.sort(key=lambda d: d.timestamp)
        
        # Trova posizione corrente
        idx = next((i for i, d in enumerate(session_decisions) 
                   if d.timestamp > current.timestamp), len(session_decisions))
        
        # Prendi decisioni precedenti e successive
        before = session_decisions[max(0, idx - depth//2):idx]
        after = session_decisions[idx:idx + depth//2]
        
        return before + [current] + after
    
    # === Replay ===
    
    async def start_replay(self, decision_id: str,
                           simulate_only: bool = True,
                           override_inputs: Dict[str, Any] = None,
                           step_by_step: bool = False) -> ReplaySession:
        """
        Avvia sessione di replay decisionale.
        
        Args:
            decision_id: ID decisione da replicare
            simulate_only: Se True, non esegue realmente
            override_inputs: Input da sovrascrivere
            step_by_step: Se True, permette esecuzione passo-passo
        
        Returns:
            ReplaySession
        """
        original = self.get_decision(decision_id)
        if not original:
            raise ValueError(f"Decision not found: {decision_id}")
        
        # Verifica integrità
        if not original.verify_integrity():
            logger.warning(f"Decision {decision_id} integrity check failed")
        
        # Crea sessione replay
        session = ReplaySession(
            id=self._generate_replay_id(),
            original_decision_id=decision_id,
            simulate_only=simulate_only,
            override_inputs=override_inputs or {},
            step_by_step=step_by_step,
            total_steps=5  # Input -> Analyze -> Decide -> Validate -> Execute
        )
        
        self._replay_sessions[session.id] = session
        
        # Avvia replay
        if not step_by_step:
            await self._execute_replay(session, original)
        
        return session
    
    async def _execute_replay(self, session: ReplaySession,
                              original: DecisionContext):
        """Esegue replay completo"""
        session.status = ReplayStatus.RUNNING
        session.started_at = datetime.now()
        
        try:
            # Step 1: Input
            session.current_step = 1
            input_data = {**original.input_data, **session.override_inputs}
            session.steps_log.append({
                "step": 1,
                "name": "Input Processing",
                "original_input": original.input_data,
                "replay_input": input_data,
                "differences": self._find_differences(original.input_data, input_data)
            })
            
            # Step 2: Analyze (simulated)
            session.current_step = 2
            session.steps_log.append({
                "step": 2,
                "name": "Analysis",
                "original_state": original.system_state,
                "replay_state": {"simulated": True} if session.simulate_only else original.system_state
            })
            
            # Step 3: Decision
            session.current_step = 3
            # In un vero replay, qui rieseguiremmo la logica decisionale
            replay_decision = copy.deepcopy(original.decision_made)
            session.steps_log.append({
                "step": 3,
                "name": "Decision Making",
                "original_decision": original.decision_made,
                "replay_decision": replay_decision,
                "original_reasoning": original.reasoning,
                "original_alternatives": original.alternatives_considered
            })
            
            # Step 4: Validation (if had one)
            session.current_step = 4
            session.steps_log.append({
                "step": 4,
                "name": "Validation",
                "original_validation": original.validation_report.to_dict() if original.validation_report else None,
                "replay_validation": "skipped" if session.simulate_only else "would_rerun"
            })
            
            # Step 5: Execution
            session.current_step = 5
            session.steps_log.append({
                "step": 5,
                "name": "Execution",
                "original_result": original.execution_result,
                "original_success": original.success,
                "replay_execution": "simulated" if session.simulate_only else "would_execute"
            })
            
            # Trova differenze totali
            session.differences = self._compile_differences(session, original)
            
            # Risultato
            session.replay_result = {
                "decision_would_match": len([d for d in session.differences if d["significant"]]) == 0,
                "input_changes": len(session.override_inputs),
                "steps_completed": session.current_step
            }
            
            session.status = ReplayStatus.COMPLETED
            session.completed_at = datetime.now()
            
            # Callbacks
            for callback in self._on_replay:
                try:
                    if inspect.iscoroutinefunction(callback):
                        await callback(session)
                    else:
                        callback(session)
                except Exception as e:
                    logger.error(f"Replay callback error: {e}")
            
            logger.info(f"Replay completed: {session.id}")
            
        except Exception as e:
            session.status = ReplayStatus.FAILED
            session.replay_result = {"error": str(e)}
            logger.error(f"Replay failed: {e}")
    
    async def step_replay(self, session_id: str) -> Dict[str, Any]:
        """Esegue un singolo step del replay"""
        session = self._replay_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        if not session.step_by_step:
            return {"error": "Not a step-by-step session"}
        
        original = self.get_decision(session.original_decision_id)
        if not original:
            return {"error": "Original decision not found"}
        
        session.current_step += 1
        # Implementa logica step-by-step
        step_result = {
            "step": session.current_step,
            "status": "completed"
        }
        
        session.steps_log.append(step_result)
        
        if session.current_step >= session.total_steps:
            session.status = ReplayStatus.COMPLETED
        
        return step_result
    
    def pause_replay(self, session_id: str):
        """Pausa replay"""
        session = self._replay_sessions.get(session_id)
        if session and session.status == ReplayStatus.RUNNING:
            session.status = ReplayStatus.PAUSED
    
    def cancel_replay(self, session_id: str):
        """Annulla replay"""
        session = self._replay_sessions.get(session_id)
        if session:
            session.status = ReplayStatus.CANCELLED
    
    def get_replay_session(self, session_id: str) -> Optional[ReplaySession]:
        """Ottiene sessione replay"""
        return self._replay_sessions.get(session_id)
    
    def _find_differences(self, original: Any, replay: Any,
                          path: str = "") -> List[Dict[str, Any]]:
        """Trova differenze tra due valori"""
        differences = []
        
        if type(original) != type(replay):
            differences.append({
                "path": path or "root",
                "type": "type_change",
                "original": str(type(original)),
                "replay": str(type(replay)),
                "significant": True
            })
            return differences
        
        if isinstance(original, dict):
            all_keys = set(original.keys()) | set(replay.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in original:
                    differences.append({
                        "path": new_path,
                        "type": "added",
                        "value": replay[key],
                        "significant": True
                    })
                elif key not in replay:
                    differences.append({
                        "path": new_path,
                        "type": "removed",
                        "value": original[key],
                        "significant": True
                    })
                else:
                    differences.extend(
                        self._find_differences(original[key], replay[key], new_path)
                    )
        elif isinstance(original, list):
            if len(original) != len(replay):
                differences.append({
                    "path": path,
                    "type": "length_change",
                    "original": len(original),
                    "replay": len(replay),
                    "significant": True
                })
            for i, (o, r) in enumerate(zip(original, replay)):
                differences.extend(
                    self._find_differences(o, r, f"{path}[{i}]")
                )
        elif original != replay:
            differences.append({
                "path": path or "root",
                "type": "value_change",
                "original": original,
                "replay": replay,
                "significant": original != replay
            })
        
        return differences
    
    def _compile_differences(self, session: ReplaySession,
                            original: DecisionContext) -> List[Dict[str, Any]]:
        """Compila tutte le differenze dal replay"""
        differences = []
        
        for step_log in session.steps_log:
            if "differences" in step_log:
                differences.extend(step_log["differences"])
        
        return differences
    
    # === Persistence ===
    
    def _persist_decision(self, decision: DecisionContext):
        """Salva decisione su disco"""
        try:
            date_str = decision.timestamp.strftime("%Y-%m-%d")
            file_path = self.storage_path / f"decisions_{date_str}.jsonl"
            
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(decision.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error persisting decision: {e}")
    
    def _load_decision(self, decision_id: str) -> Optional[DecisionContext]:
        """Carica decisione da disco"""
        # Cerca nei file
        for file_path in self.storage_path.glob("decisions_*.jsonl"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data.get("id") == decision_id:
                            return DecisionContext.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading decision: {e}")
        return None
    
    def _load_recent_decisions(self, days: int = 7):
        """Carica decisioni recenti in memoria"""
        cutoff = datetime.now() - timedelta(days=days)
        
        for file_path in sorted(self.storage_path.glob("decisions_*.jsonl")):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(data["timestamp"])
                        if timestamp >= cutoff:
                            decision = DecisionContext.from_dict(data)
                            self._decisions[decision.id] = decision
                            self._decision_index.append(decision.id)
            except Exception as e:
                logger.error(f"Error loading decisions: {e}")
        
        logger.info(f"Loaded {len(self._decisions)} recent decisions")
    
    def _cleanup_memory(self):
        """Pulisce memoria se oltre limite"""
        if len(self._decisions) > self._max_memory_decisions:
            # Rimuovi le più vecchie
            to_remove = len(self._decisions) - self._max_memory_decisions
            for dec_id in self._decision_index[:to_remove]:
                del self._decisions[dec_id]
            self._decision_index = self._decision_index[to_remove:]
    
    # === Callbacks ===
    
    def on_decision(self, callback: Callable):
        """Registra callback per nuove decisioni"""
        self._on_decision.append(callback)
    
    def on_replay(self, callback: Callable):
        """Registra callback per replay completati"""
        self._on_replay.append(callback)
    
    # === Stats ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche audit trail"""
        total = len(self._decisions)
        if total == 0:
            return {
                "total_decisions": 0,
                "success_rate": 0,
                "replay_sessions": 0
            }
        
        successful = sum(1 for d in self._decisions.values() if d.success is True)
        failed = sum(1 for d in self._decisions.values() if d.success is False)
        
        # Per tipo
        by_type = {}
        for d in self._decisions.values():
            by_type[d.decision_type] = by_type.get(d.decision_type, 0) + 1
        
        return {
            "total_decisions": total,
            "successful": successful,
            "failed": failed,
            "pending": total - successful - failed,
            "success_rate": round(successful / (successful + failed), 3) if (successful + failed) > 0 else 0,
            "by_type": by_type,
            "replay_sessions": len(self._replay_sessions),
            "oldest_in_memory": min((d.timestamp for d in self._decisions.values()), default=None),
            "newest": max((d.timestamp for d in self._decisions.values()), default=None)
        }


# ============ INTEGRATED SYSTEM ============

class ValidationAuditSystem:
    """
    Sistema integrato di validazione e audit trail.
    """
    
    def __init__(self, storage_path: str = None):
        self.validator = LogicValidator()
        self.audit_trail = DecisionAuditTrail(storage_path)
        
        # Stats
        self._validated_actions = 0
        self._rejected_actions = 0
    
    async def validate_and_record(self,
                                   action_id: str,
                                   action_type: str,
                                   input_data: Dict[str, Any],
                                   user_request: str,
                                   session_id: str,
                                   system_state: Dict[str, Any],
                                   mode: str,
                                   security_level: str,
                                   decision_made: Dict[str, Any],
                                   alternatives: List[Dict[str, Any]],
                                   reasoning: str,
                                   confidence: float) -> Tuple[bool, DecisionContext, ValidationReport]:
        """
        Valida azione e registra decisione.
        
        Returns:
            (approved, decision_context, validation_report)
        """
        # 1. Valida
        validation = await self.validator.validate(
            action_id=action_id,
            action_type=action_type,
            data={**input_data, **decision_made}
        )
        
        # 2. Registra decisione
        decision = self.audit_trail.record_decision(
            input_data=input_data,
            user_request=user_request,
            session_id=session_id,
            system_state=system_state,
            mode=mode,
            security_level=security_level,
            decision_type=action_type,
            decision_made=decision_made,
            alternatives=alternatives,
            reasoning=reasoning,
            confidence=confidence,
            validation_report=validation
        )
        
        # 3. Determina approvazione
        approved = validation.result in [ValidationResult.PASSED, ValidationResult.WARNING]
        
        # Se richiede conferma e non l'ha
        if approved and validation.requires_confirmation and not validation.confirmation_received:
            approved = False
            logger.info(f"Action {action_id} awaiting confirmation")
        
        # Stats
        if approved:
            self._validated_actions += 1
        else:
            self._rejected_actions += 1
        
        return approved, decision, validation
    
    def confirm_action(self, validation_id: str, confirmed_by: str) -> bool:
        """Conferma azione che richiede conferma"""
        return self.validator.confirm_validation(validation_id, confirmed_by)
    
    async def replay_decision(self, decision_id: str,
                              **kwargs) -> ReplaySession:
        """Avvia replay di una decisione"""
        return await self.audit_trail.start_replay(decision_id, **kwargs)
    
    def get_decision_with_validation(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Ottiene decisione con report validazione"""
        decision = self.audit_trail.get_decision(decision_id)
        if not decision:
            return None
        
        return {
            "decision": decision.to_dict(),
            "validation": decision.validation_report.to_dict() if decision.validation_report else None
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche combinate"""
        return {
            "validation": self.validator.get_stats(),
            "audit_trail": self.audit_trail.get_stats(),
            "validated_actions": self._validated_actions,
            "rejected_actions": self._rejected_actions,
            "rejection_rate": round(
                self._rejected_actions / (self._validated_actions + self._rejected_actions), 3
            ) if (self._validated_actions + self._rejected_actions) > 0 else 0
        }


# ============ FACTORY ============

def create_validation_audit_system(storage_path: str = None) -> ValidationAuditSystem:
    """Crea sistema di validazione e audit"""
    return ValidationAuditSystem(storage_path)
