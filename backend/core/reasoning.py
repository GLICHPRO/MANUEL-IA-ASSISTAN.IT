"""
🧠 Multi-Step Reasoning Engine

Supporta ragionamento in più passaggi con verifica costante dei risultati.
Integrato nel sistema di comunicazione Jarvis ↔ Gideon.

Features:
- Catene di ragionamento strutturate
- Checkpoint e validazione progressiva
- Feedback loop continuo
- Verifica risultati intermedi
- Rollback se step fallisce
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from datetime import datetime
import uuid
import logging


# Logger dedicato
reasoning_logger = logging.getLogger("multi_step_reasoning")
reasoning_logger.setLevel(logging.DEBUG)


class StepStatus(Enum):
    """Stato di uno step di ragionamento"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_VERIFICATION = "awaiting_verification"
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class StepType(Enum):
    """Tipo di step nel ragionamento"""
    ANALYSIS = "analysis"           # Analisi dati/contesto
    PREDICTION = "prediction"       # Previsione
    SIMULATION = "simulation"       # Simulazione scenario
    EVALUATION = "evaluation"       # Valutazione risultati
    DECISION = "decision"          # Decisione
    VERIFICATION = "verification"  # Verifica
    CHECKPOINT = "checkpoint"      # Checkpoint validazione
    SYNTHESIS = "synthesis"        # Sintesi risultati


class VerificationResult(Enum):
    """Risultato verifica"""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    NEEDS_REVIEW = "needs_review"


@dataclass
class ReasoningStep:
    """Singolo step di ragionamento"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_number: int = 0
    step_type: StepType = StepType.ANALYSIS
    description: str = ""
    
    # Input/Output
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # Stato
    status: StepStatus = StepStatus.PENDING
    
    # Verifica
    verification_criteria: List[str] = field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    verification_notes: str = ""
    
    # Metriche
    confidence: float = 0.0
    execution_time_ms: float = 0.0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Dipendenze
    depends_on: List[str] = field(default_factory=list)  # IDs degli step precedenti
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status.value,
            "verification_criteria": self.verification_criteria,
            "verification_result": self.verification_result.value if self.verification_result else None,
            "verification_notes": self.verification_notes,
            "confidence": round(self.confidence, 3),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "depends_on": self.depends_on
        }


@dataclass
class ReasoningChain:
    """Catena completa di ragionamento multi-step"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    
    # Steps
    steps: List[ReasoningStep] = field(default_factory=list)
    current_step_index: int = 0
    
    # Stato globale
    status: StepStatus = StepStatus.PENDING
    
    # Contesto condiviso tra step
    shared_context: Dict[str, Any] = field(default_factory=dict)
    
    # Risultato finale
    final_result: Dict[str, Any] = field(default_factory=dict)
    final_confidence: float = 0.0
    
    # Checkpoints per rollback
    checkpoints: List[Dict] = field(default_factory=list)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metriche
    total_execution_time_ms: float = 0.0
    verification_pass_rate: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "status": self.status.value,
            "shared_context": self.shared_context,
            "final_result": self.final_result,
            "final_confidence": round(self.final_confidence, 3),
            "checkpoints": len(self.checkpoints),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_execution_time_ms": round(self.total_execution_time_ms, 2),
            "verification_pass_rate": round(self.verification_pass_rate, 3)
        }
    
    @property
    def current_step(self) -> Optional[ReasoningStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status in [StepStatus.VERIFIED, StepStatus.SKIPPED])
        return completed / len(self.steps)


@dataclass
class VerificationCheckpoint:
    """Checkpoint per verifica e potenziale rollback"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    chain_id: str = ""
    step_index: int = 0
    
    # Snapshot stato
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    steps_snapshot: List[Dict] = field(default_factory=list)
    
    # Criteri verifica
    verification_criteria: List[str] = field(default_factory=list)
    must_pass_all: bool = False  # Se True, tutti i criteri devono passare
    
    # Risultato
    is_verified: bool = False
    failed_criteria: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "chain_id": self.chain_id,
            "step_index": self.step_index,
            "verification_criteria": self.verification_criteria,
            "must_pass_all": self.must_pass_all,
            "is_verified": self.is_verified,
            "failed_criteria": self.failed_criteria,
            "timestamp": self.timestamp.isoformat()
        }


class MultiStepReasoning:
    """
    Engine per ragionamento multi-step con verifica continua.
    
    Workflow:
    1. Crea catena di ragionamento con step definiti
    2. Esegue ogni step in sequenza
    3. Verifica risultati di ogni step
    4. Crea checkpoint per rollback
    5. Sintetizza risultato finale
    """
    
    def __init__(self):
        self.active_chains: Dict[str, ReasoningChain] = {}
        self.completed_chains: List[ReasoningChain] = []
        self.verification_handlers: Dict[StepType, Callable] = {}
        
        # Configurazione
        self.auto_verify = True
        self.create_checkpoints = True
        self.max_retries_per_step = 2
        
        # Statistiche
        self.stats = {
            "chains_created": 0,
            "chains_completed": 0,
            "chains_failed": 0,
            "steps_executed": 0,
            "verifications_passed": 0,
            "verifications_failed": 0,
            "rollbacks_performed": 0
        }
        
        reasoning_logger.info("MultiStepReasoning engine initialized")
    
    # === Chain Management ===
    
    def create_chain(self, name: str, description: str = "",
                    initial_context: Dict = None) -> ReasoningChain:
        """Crea nuova catena di ragionamento"""
        chain = ReasoningChain(
            name=name,
            description=description,
            shared_context=initial_context or {}
        )
        
        self.active_chains[chain.id] = chain
        self.stats["chains_created"] += 1
        
        reasoning_logger.info(f"Chain created: {chain.id} - {name}")
        
        return chain
    
    def add_step(self, chain_id: str,
                step_type: StepType,
                description: str,
                input_data: Dict = None,
                verification_criteria: List[str] = None,
                depends_on: List[str] = None) -> ReasoningStep:
        """Aggiunge step alla catena"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            raise ValueError(f"Chain {chain_id} not found")
        
        step = ReasoningStep(
            step_number=len(chain.steps) + 1,
            step_type=step_type,
            description=description,
            input_data=input_data or {},
            verification_criteria=verification_criteria or [],
            depends_on=depends_on or []
        )
        
        chain.steps.append(step)
        
        reasoning_logger.debug(f"Step added to chain {chain_id}: {step.step_number}. {description}")
        
        return step
    
    def add_analysis_step(self, chain_id: str, description: str,
                         data: Dict, criteria: List[str] = None) -> ReasoningStep:
        """Shortcut per aggiungere step di analisi"""
        return self.add_step(
            chain_id, StepType.ANALYSIS, description,
            input_data={"data": data},
            verification_criteria=criteria or ["output_present", "confidence_above_threshold"]
        )
    
    def add_prediction_step(self, chain_id: str, description: str,
                           context: Dict, criteria: List[str] = None) -> ReasoningStep:
        """Shortcut per step di previsione"""
        return self.add_step(
            chain_id, StepType.PREDICTION, description,
            input_data={"context": context},
            verification_criteria=criteria or ["prediction_valid", "confidence_sufficient"]
        )
    
    def add_verification_step(self, chain_id: str, description: str,
                             target_steps: List[str] = None) -> ReasoningStep:
        """Aggiunge step di verifica esplicito"""
        return self.add_step(
            chain_id, StepType.VERIFICATION, description,
            input_data={"verify_steps": target_steps or []},
            verification_criteria=["all_targets_verified"]
        )
    
    def add_checkpoint(self, chain_id: str,
                      criteria: List[str],
                      must_pass_all: bool = True) -> VerificationCheckpoint:
        """Crea checkpoint per verifica e potenziale rollback"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            raise ValueError(f"Chain {chain_id} not found")
        
        checkpoint = VerificationCheckpoint(
            chain_id=chain_id,
            step_index=len(chain.steps),
            context_snapshot=dict(chain.shared_context),
            steps_snapshot=[s.to_dict() for s in chain.steps],
            verification_criteria=criteria,
            must_pass_all=must_pass_all
        )
        
        chain.checkpoints.append(checkpoint.to_dict())
        
        reasoning_logger.info(f"Checkpoint created at step {checkpoint.step_index} for chain {chain_id}")
        
        return checkpoint
    
    # === Execution ===
    
    def start_chain(self, chain_id: str) -> bool:
        """Avvia esecuzione della catena"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            return False
        
        chain.status = StepStatus.IN_PROGRESS
        chain.started_at = datetime.now()
        chain.current_step_index = 0
        
        reasoning_logger.info(f"Chain {chain_id} started with {len(chain.steps)} steps")
        
        return True
    
    def execute_step(self, chain_id: str, step_result: Dict,
                    confidence: float = 0.8) -> ReasoningStep:
        """
        Esegue lo step corrente con il risultato fornito.
        Chiamato da Gideon dopo aver elaborato lo step.
        """
        chain = self.active_chains.get(chain_id)
        if not chain or not chain.current_step:
            raise ValueError(f"No current step for chain {chain_id}")
        
        step = chain.current_step
        start_time = datetime.now()
        
        # Aggiorna step
        step.status = StepStatus.IN_PROGRESS
        step.started_at = start_time
        step.output_data = step_result
        step.confidence = confidence
        
        # Aggiungi output al contesto condiviso
        chain.shared_context[f"step_{step.step_number}_output"] = step_result
        chain.shared_context[f"step_{step.step_number}_confidence"] = confidence
        
        # Calcola tempo esecuzione
        end_time = datetime.now()
        step.execution_time_ms = (end_time - start_time).total_seconds() * 1000
        step.completed_at = end_time
        
        # Auto-verifica se abilitata
        if self.auto_verify and step.verification_criteria:
            self._verify_step(chain, step)
        else:
            step.status = StepStatus.AWAITING_VERIFICATION
        
        self.stats["steps_executed"] += 1
        
        reasoning_logger.info(
            f"Step {step.step_number} executed: {step.description[:30]}... "
            f"[confidence: {confidence:.2f}, status: {step.status.value}]"
        )
        
        return step
    
    def _verify_step(self, chain: ReasoningChain, step: ReasoningStep) -> VerificationResult:
        """Verifica automatica dello step"""
        failed_criteria = []
        
        for criterion in step.verification_criteria:
            passed = self._check_criterion(criterion, step, chain)
            if not passed:
                failed_criteria.append(criterion)
        
        if not failed_criteria:
            step.verification_result = VerificationResult.PASSED
            step.status = StepStatus.VERIFIED
            self.stats["verifications_passed"] += 1
        elif len(failed_criteria) < len(step.verification_criteria):
            step.verification_result = VerificationResult.PARTIAL
            step.status = StepStatus.AWAITING_VERIFICATION
            step.verification_notes = f"Failed: {', '.join(failed_criteria)}"
        else:
            step.verification_result = VerificationResult.FAILED
            step.status = StepStatus.FAILED
            step.verification_notes = f"All criteria failed"
            self.stats["verifications_failed"] += 1
        
        return step.verification_result
    
    def _check_criterion(self, criterion: str, step: ReasoningStep,
                        chain: ReasoningChain) -> bool:
        """Verifica singolo criterio"""
        
        # Criteri standard
        if criterion == "output_present":
            return bool(step.output_data)
        
        elif criterion == "confidence_above_threshold":
            return step.confidence >= 0.5
        
        elif criterion == "confidence_sufficient":
            return step.confidence >= 0.7
        
        elif criterion == "prediction_valid":
            output = step.output_data
            return (
                output.get("prediction") is not None or
                output.get("value") is not None or
                output.get("result") is not None
            )
        
        elif criterion == "no_errors":
            return "error" not in step.output_data and "errors" not in step.output_data
        
        elif criterion == "risk_acceptable":
            risk = step.output_data.get("risk_score", step.output_data.get("risk", 0))
            return risk < 0.7
        
        elif criterion == "all_targets_verified":
            target_ids = step.input_data.get("verify_steps", [])
            for target_id in target_ids:
                target_step = next((s for s in chain.steps if s.id == target_id), None)
                if target_step and target_step.verification_result != VerificationResult.PASSED:
                    return False
            return True
        
        elif criterion == "consistency_check":
            # Verifica coerenza con step precedenti
            prev_outputs = [
                chain.shared_context.get(f"step_{i}_output", {})
                for i in range(1, step.step_number)
            ]
            # Placeholder - logica di coerenza specifica
            return True
        
        # Handler personalizzato
        if step.step_type in self.verification_handlers:
            return self.verification_handlers[step.step_type](criterion, step, chain)
        
        # Default: passa se non riconosciuto
        return True
    
    def verify_step_manually(self, chain_id: str, step_id: str,
                            result: VerificationResult,
                            notes: str = "") -> bool:
        """Verifica manuale di uno step"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            return False
        
        step = next((s for s in chain.steps if s.id == step_id), None)
        if not step:
            return False
        
        step.verification_result = result
        step.verification_notes = notes
        
        if result == VerificationResult.PASSED:
            step.status = StepStatus.VERIFIED
            self.stats["verifications_passed"] += 1
        elif result == VerificationResult.FAILED:
            step.status = StepStatus.FAILED
            self.stats["verifications_failed"] += 1
        
        reasoning_logger.info(f"Manual verification for step {step_id}: {result.value}")
        
        return True
    
    def advance_to_next_step(self, chain_id: str) -> Optional[ReasoningStep]:
        """Avanza allo step successivo"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            return None
        
        # Verifica che step corrente sia completato
        current = chain.current_step
        if current and current.status not in [StepStatus.VERIFIED, StepStatus.SKIPPED]:
            reasoning_logger.warning(f"Cannot advance: current step not verified")
            return None
        
        chain.current_step_index += 1
        
        if chain.current_step_index >= len(chain.steps):
            # Catena completata
            self._complete_chain(chain)
            return None
        
        next_step = chain.current_step
        reasoning_logger.info(f"Advanced to step {next_step.step_number}: {next_step.description[:30]}...")
        
        return next_step
    
    def _complete_chain(self, chain: ReasoningChain):
        """Completa la catena e calcola risultato finale"""
        chain.status = StepStatus.VERIFIED
        chain.completed_at = datetime.now()
        
        # Calcola tempo totale
        if chain.started_at:
            chain.total_execution_time_ms = (
                chain.completed_at - chain.started_at
            ).total_seconds() * 1000
        
        # Calcola pass rate
        verified = sum(1 for s in chain.steps if s.verification_result == VerificationResult.PASSED)
        chain.verification_pass_rate = verified / len(chain.steps) if chain.steps else 0
        
        # Calcola confidenza finale (media ponderata)
        confidences = [s.confidence for s in chain.steps if s.confidence > 0]
        chain.final_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Sintetizza risultato finale
        chain.final_result = self._synthesize_result(chain)
        
        # Sposta in completed
        del self.active_chains[chain.id]
        self.completed_chains.append(chain)
        self.stats["chains_completed"] += 1
        
        reasoning_logger.info(
            f"Chain {chain.id} completed. "
            f"Pass rate: {chain.verification_pass_rate:.0%}, "
            f"Final confidence: {chain.final_confidence:.2f}"
        )
    
    def _synthesize_result(self, chain: ReasoningChain) -> Dict:
        """Sintetizza risultato finale dalla catena"""
        result = {
            "chain_id": chain.id,
            "chain_name": chain.name,
            "steps_completed": len([s for s in chain.steps if s.status == StepStatus.VERIFIED]),
            "total_steps": len(chain.steps),
            "verification_pass_rate": chain.verification_pass_rate,
            "final_confidence": chain.final_confidence,
            "execution_time_ms": chain.total_execution_time_ms,
            "step_outputs": {}
        }
        
        # Raccogli output significativi
        for step in chain.steps:
            if step.output_data:
                result["step_outputs"][f"step_{step.step_number}"] = {
                    "type": step.step_type.value,
                    "description": step.description,
                    "output": step.output_data,
                    "confidence": step.confidence,
                    "verified": step.verification_result == VerificationResult.PASSED
                }
        
        # Estrai conclusioni chiave
        last_step = chain.steps[-1] if chain.steps else None
        if last_step and last_step.step_type == StepType.SYNTHESIS:
            result["conclusion"] = last_step.output_data
        
        return result
    
    # === Rollback ===
    
    def rollback_to_checkpoint(self, chain_id: str, 
                              checkpoint_index: int = -1) -> bool:
        """Rollback della catena a un checkpoint precedente"""
        chain = self.active_chains.get(chain_id)
        if not chain or not chain.checkpoints:
            return False
        
        # Usa ultimo checkpoint se non specificato
        checkpoint_data = chain.checkpoints[checkpoint_index]
        
        # Ripristina contesto
        chain.shared_context = checkpoint_data.get("context_snapshot", {})
        
        # Ripristina step index
        chain.current_step_index = checkpoint_data.get("step_index", 0)
        
        # Marca step successivi come rolled back
        for i in range(checkpoint_data.get("step_index", 0), len(chain.steps)):
            chain.steps[i].status = StepStatus.ROLLED_BACK
            chain.steps[i].output_data = {}
        
        self.stats["rollbacks_performed"] += 1
        
        reasoning_logger.warning(
            f"Chain {chain_id} rolled back to checkpoint at step {checkpoint_data.get('step_index')}"
        )
        
        return True
    
    def retry_step(self, chain_id: str, step_id: str = None) -> bool:
        """Riprova uno step fallito"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            return False
        
        if step_id:
            step = next((s for s in chain.steps if s.id == step_id), None)
        else:
            step = chain.current_step
        
        if not step:
            return False
        
        # Reset step
        step.status = StepStatus.PENDING
        step.output_data = {}
        step.verification_result = None
        step.verification_notes = ""
        step.confidence = 0.0
        
        reasoning_logger.info(f"Step {step.step_number} reset for retry")
        
        return True
    
    # === Query ===
    
    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Ottiene catena per ID"""
        return self.active_chains.get(chain_id)
    
    def get_chain_progress(self, chain_id: str) -> Dict:
        """Ottiene progresso catena"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            return {}
        
        return {
            "chain_id": chain.id,
            "name": chain.name,
            "status": chain.status.value,
            "progress": chain.progress,
            "current_step": chain.current_step_index + 1,
            "total_steps": len(chain.steps),
            "current_step_description": chain.current_step.description if chain.current_step else None,
            "verified_steps": sum(1 for s in chain.steps if s.verification_result == VerificationResult.PASSED),
            "failed_steps": sum(1 for s in chain.steps if s.status == StepStatus.FAILED)
        }
    
    def get_step_details(self, chain_id: str, step_number: int) -> Optional[Dict]:
        """Ottiene dettagli di uno step specifico"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            return None
        
        step = next((s for s in chain.steps if s.step_number == step_number), None)
        return step.to_dict() if step else None
    
    def get_verification_summary(self, chain_id: str) -> Dict:
        """Sommario verifiche della catena"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            return {}
        
        return {
            "total_steps": len(chain.steps),
            "passed": sum(1 for s in chain.steps if s.verification_result == VerificationResult.PASSED),
            "failed": sum(1 for s in chain.steps if s.verification_result == VerificationResult.FAILED),
            "partial": sum(1 for s in chain.steps if s.verification_result == VerificationResult.PARTIAL),
            "pending": sum(1 for s in chain.steps if s.verification_result is None),
            "checkpoints": len(chain.checkpoints),
            "details": [
                {
                    "step": s.step_number,
                    "type": s.step_type.value,
                    "status": s.status.value,
                    "verification": s.verification_result.value if s.verification_result else "pending",
                    "confidence": s.confidence
                }
                for s in chain.steps
            ]
        }
    
    def get_statistics(self) -> Dict:
        """Statistiche globali"""
        return {
            **self.stats,
            "active_chains": len(self.active_chains),
            "completed_chains": len(self.completed_chains),
            "success_rate": (
                self.stats["chains_completed"] / self.stats["chains_created"]
                if self.stats["chains_created"] > 0 else 0
            ),
            "verification_pass_rate": (
                self.stats["verifications_passed"] / 
                (self.stats["verifications_passed"] + self.stats["verifications_failed"])
                if (self.stats["verifications_passed"] + self.stats["verifications_failed"]) > 0 else 0
            )
        }


# === Integration with Communication ===

class ReasoningCommunicator:
    """
    Integra multi-step reasoning con il sistema di comunicazione Jarvis ↔ Gideon.
    Gestisce il flusso di messaggi per ogni step della catena.
    """
    
    def __init__(self, reasoning_engine: MultiStepReasoning, communication_bridge=None):
        self.reasoning = reasoning_engine
        self.comm = communication_bridge
        self.active_conversations: Dict[str, str] = {}  # chain_id -> correlation_id
    
    def start_reasoning_conversation(self, chain_id: str,
                                    objective: str,
                                    context: Dict = None) -> str:
        """Avvia conversazione di ragionamento"""
        if self.comm:
            from .communication import Objective, Sender, MessageType
            
            obj = Objective(
                description=f"Multi-step reasoning: {objective}",
                context=context or {},
                metadata={"chain_id": chain_id, "reasoning_type": "multi_step"}
            )
            
            correlation_id = self.comm.jarvis.send_objective(obj)
            self.active_conversations[chain_id] = correlation_id
            
            return correlation_id
        
        return chain_id
    
    def send_step_request(self, chain_id: str, step: ReasoningStep) -> str:
        """Invia richiesta per elaborazione step a Gideon"""
        if self.comm:
            from .communication import Sender, MessageType, MessagePriority
            
            correlation_id = self.active_conversations.get(chain_id, chain_id)
            
            payload = {
                "chain_id": chain_id,
                "step_id": step.id,
                "step_number": step.step_number,
                "step_type": step.step_type.value,
                "description": step.description,
                "input_data": step.input_data,
                "verification_criteria": step.verification_criteria,
                "shared_context": self.reasoning.get_chain(chain_id).shared_context if self.reasoning.get_chain(chain_id) else {}
            }
            
            # Determina tipo messaggio basato su step type
            msg_type_map = {
                StepType.ANALYSIS: MessageType.ANALYSIS_REQUEST,
                StepType.PREDICTION: MessageType.ANALYSIS_REQUEST,
                StepType.SIMULATION: MessageType.SIMULATION_REQUEST,
                StepType.EVALUATION: MessageType.ANALYSIS_REQUEST,
                StepType.VERIFICATION: MessageType.VALIDATION_REQUEST
            }
            
            msg_type = msg_type_map.get(step.step_type, MessageType.ANALYSIS_REQUEST)
            
            msg = self.comm.channel.send(
                sender=Sender.JARVIS,
                recipient=Sender.GIDEON,
                message_type=msg_type,
                payload=payload,
                correlation_id=correlation_id
            )
            
            return msg.id
        
        return step.id
    
    def send_step_result(self, chain_id: str, step: ReasoningStep,
                        result: Dict, confidence: float) -> str:
        """Invia risultato step da Gideon a Jarvis"""
        if self.comm:
            from .communication import Sender, MessageType
            
            correlation_id = self.active_conversations.get(chain_id, chain_id)
            
            payload = {
                "chain_id": chain_id,
                "step_id": step.id,
                "step_number": step.step_number,
                "result": result,
                "confidence": confidence,
                "verification_status": step.verification_result.value if step.verification_result else "pending"
            }
            
            msg = self.comm.channel.send(
                sender=Sender.GIDEON,
                recipient=Sender.JARVIS,
                message_type=MessageType.ANALYSIS_RESULT,
                payload=payload,
                correlation_id=correlation_id
            )
            
            return msg.id
        
        return step.id
    
    def send_verification_result(self, chain_id: str,
                                verification_passed: bool,
                                details: Dict = None) -> str:
        """Invia risultato verifica"""
        if self.comm:
            from .communication import Sender, MessageType, MessagePriority
            
            correlation_id = self.active_conversations.get(chain_id, chain_id)
            chain = self.reasoning.get_chain(chain_id)
            
            payload = {
                "chain_id": chain_id,
                "verification_passed": verification_passed,
                "details": details or {},
                "progress": chain.progress if chain else 0,
                "summary": self.reasoning.get_verification_summary(chain_id)
            }
            
            msg = self.comm.channel.send(
                sender=Sender.GIDEON,
                recipient=Sender.JARVIS,
                message_type=MessageType.VALIDATION_RESULT,
                payload=payload,
                correlation_id=correlation_id,
                priority=MessagePriority.HIGH if not verification_passed else MessagePriority.NORMAL
            )
            
            return msg.id
        
        return chain_id
