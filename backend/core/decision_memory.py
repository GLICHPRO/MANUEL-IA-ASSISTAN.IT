"""
🎯 Memoria Decisionale - Apprendimento dai Risultati delle Azioni

Sistema che traccia e apprende da:
- Decisioni prese e loro outcome
- Azioni eseguite e risultati
- Contesto in cui sono state prese
- Feedback positivo/negativo

Permette di:
- Migliorare le decisioni future
- Evitare errori ripetuti
- Ottimizzare strategie nel tempo
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import json
import math
import logging
from pathlib import Path


# Logger
decision_logger = logging.getLogger("decision_memory")
decision_logger.setLevel(logging.DEBUG)


# === ENUMS ===

class DecisionOutcome(Enum):
    """Esito della decisione"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    PENDING = "pending"


class FeedbackType(Enum):
    """Tipo di feedback"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"


class AdaptationType(Enum):
    """Tipo di adattamento"""
    PARAMETER_TUNING = "parameter_tuning"   # Ottimizzazione parametri
    STRATEGY_SWITCH = "strategy_switch"     # Cambio strategia
    THRESHOLD_ADJUST = "threshold_adjust"   # Aggiustamento soglie
    RULE_UPDATE = "rule_update"             # Aggiornamento regole
    WEIGHT_SHIFT = "weight_shift"           # Modifica pesi


# === DATA CLASSES ===

@dataclass
class DecisionRecord:
    """Record di una decisione presa"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # La decisione
    decision_type: str = ""  # "action", "strategy", "response", etc.
    decision_made: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)  # Alternative considerate
    
    # Contesto in cui è stata presa
    context: Dict[str, Any] = field(default_factory=dict)
    input_data: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence al momento della decisione
    initial_confidence: float = 0.5
    
    # Outcome
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    outcome_details: Dict[str, Any] = field(default_factory=dict)
    
    # Metriche risultato
    execution_time_ms: Optional[int] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    # Feedback ricevuto
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    decided_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    # Metadata
    source: str = "system"
    tags: List[str] = field(default_factory=list)
    
    def resolve(self, outcome: DecisionOutcome, details: Dict = None):
        """Risolve la decisione con un outcome"""
        self.outcome = outcome
        self.outcome_details = details or {}
        self.resolved_at = datetime.now()
    
    def add_feedback(self, feedback_type: FeedbackType, 
                    message: str = "",
                    details: Dict = None):
        """Aggiunge feedback alla decisione"""
        self.feedback.append({
            "type": feedback_type.value,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def success_score(self) -> float:
        """Calcola score di successo (0-1)"""
        scores = {
            DecisionOutcome.SUCCESS: 1.0,
            DecisionOutcome.PARTIAL: 0.6,
            DecisionOutcome.FAILURE: 0.0,
            DecisionOutcome.UNKNOWN: 0.5,
            DecisionOutcome.PENDING: 0.5
        }
        
        base_score = scores.get(self.outcome, 0.5)
        
        # Aggiusta con feedback
        positive = sum(1 for f in self.feedback if f["type"] == "positive")
        negative = sum(1 for f in self.feedback if f["type"] == "negative")
        
        if positive + negative > 0:
            feedback_adjustment = (positive - negative) / (positive + negative) * 0.2
            base_score = max(0, min(1, base_score + feedback_adjustment))
        
        return base_score
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "decision_type": self.decision_type,
            "decision_made": self.decision_made,
            "alternatives_count": len(self.alternatives),
            "context": self.context,
            "initial_confidence": self.initial_confidence,
            "outcome": self.outcome.value,
            "outcome_details": self.outcome_details,
            "feedback_count": len(self.feedback),
            "success_score": round(self.success_score(), 3),
            "decided_at": self.decided_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "tags": self.tags
        }


@dataclass
class ActionResult:
    """Risultato di un'azione eseguita"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # L'azione
    action_type: str = ""
    action_params: Dict[str, Any] = field(default_factory=dict)
    
    # Risultato
    success: bool = False
    result_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    # Contesto
    context: Dict[str, Any] = field(default_factory=dict)
    decision_id: Optional[str] = None  # Collegamento alla decisione
    
    # Metriche
    execution_time_ms: int = 0
    retry_count: int = 0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def complete(self, success: bool, result_data: Dict = None, error: str = None):
        """Completa l'azione"""
        self.success = success
        self.result_data = result_data or {}
        self.error = error
        self.completed_at = datetime.now()
        self.execution_time_ms = int(
            (self.completed_at - self.started_at).total_seconds() * 1000
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_type": self.action_type,
            "action_params": self.action_params,
            "success": self.success,
            "result_data": self.result_data,
            "error": self.error,
            "decision_id": self.decision_id,
            "execution_time_ms": self.execution_time_ms,
            "retry_count": self.retry_count,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class LearningInsight:
    """Insight appreso dall'esperienza"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # L'insight
    insight_type: str = ""  # "pattern", "rule", "correlation", "optimization"
    description: str = ""
    
    # Condizioni che attivano l'insight
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Raccomandazione/azione suggerita
    recommendation: Dict[str, Any] = field(default_factory=dict)
    
    # Evidenza
    supporting_decisions: List[str] = field(default_factory=list)
    confidence: float = 0.5
    sample_size: int = 0
    
    # Stato
    is_active: bool = True
    times_applied: int = 0
    times_correct: int = 0
    
    # Timing
    discovered_at: datetime = field(default_factory=datetime.now)
    last_validated: datetime = field(default_factory=datetime.now)
    
    def accuracy(self) -> float:
        """Calcola accuratezza dell'insight"""
        if self.times_applied == 0:
            return 0.5
        return self.times_correct / self.times_applied
    
    def apply(self, correct: bool):
        """Registra applicazione dell'insight"""
        self.times_applied += 1
        if correct:
            self.times_correct += 1
        self.last_validated = datetime.now()
        
        # Aggiorna confidence
        self.confidence = (self.confidence * 0.9 + self.accuracy() * 0.1)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "insight_type": self.insight_type,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 3),
            "accuracy": round(self.accuracy(), 3),
            "sample_size": self.sample_size,
            "times_applied": self.times_applied,
            "is_active": self.is_active,
            "discovered_at": self.discovered_at.isoformat()
        }


@dataclass
class Adaptation:
    """Adattamento appreso e applicato"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    adaptation_type: AdaptationType = AdaptationType.PARAMETER_TUNING
    description: str = ""
    
    # Cosa è stato cambiato
    target: str = ""  # Nome del parametro/strategia/regola
    old_value: Any = None
    new_value: Any = None
    
    # Motivazione
    trigger_reason: str = ""
    evidence: List[str] = field(default_factory=list)  # Decision IDs
    
    # Risultato
    improvement_expected: float = 0.0
    improvement_actual: Optional[float] = None
    
    # Timing
    applied_at: datetime = field(default_factory=datetime.now)
    evaluated_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "adaptation_type": self.adaptation_type.value,
            "description": self.description,
            "target": self.target,
            "old_value": str(self.old_value),
            "new_value": str(self.new_value),
            "trigger_reason": self.trigger_reason,
            "improvement_expected": self.improvement_expected,
            "improvement_actual": self.improvement_actual,
            "applied_at": self.applied_at.isoformat()
        }


# === DECISION MEMORY ===

class DecisionMemory:
    """
    Sistema di memoria decisionale.
    Traccia decisioni, apprende dai risultati, adatta comportamento.
    """
    
    def __init__(self, storage_path: Path = None, max_records: int = 5000):
        self.storage_path = storage_path or Path("data/decision_memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_records = max_records
        
        # Storage
        self.decisions: Dict[str, DecisionRecord] = {}
        self.actions: Dict[str, ActionResult] = {}
        self.insights: Dict[str, LearningInsight] = {}
        self.adaptations: List[Adaptation] = []
        
        # Indici
        self.decisions_by_type: Dict[str, List[str]] = defaultdict(list)
        self.decisions_by_outcome: Dict[DecisionOutcome, List[str]] = defaultdict(list)
        self.actions_by_type: Dict[str, List[str]] = defaultdict(list)
        
        # Metriche aggregate
        self.metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "total_actions": 0,
            "successful_actions": 0,
            "adaptations_made": 0,
            "insights_discovered": 0
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.min_sample_size = 5
        
        # Parametri adattabili (che possono essere tuned)
        self.adaptive_params: Dict[str, Any] = {
            "default_timeout": 60,
            "retry_count": 3,
            "confidence_threshold": 0.5,
            "risk_tolerance": 0.3
        }
        
        # Carica dati
        self._load()
        
        decision_logger.info(f"DecisionMemory initialized: {len(self.decisions)} decisions, "
                           f"{len(self.insights)} insights")
    
    # === Decision Recording ===
    
    def record_decision(self,
                       decision_type: str,
                       decision_made: Dict[str, Any],
                       context: Dict[str, Any],
                       input_data: Dict[str, Any] = None,
                       confidence: float = 0.5,
                       alternatives: List[Dict] = None,
                       tags: List[str] = None) -> DecisionRecord:
        """Registra una nuova decisione"""
        record = DecisionRecord(
            decision_type=decision_type,
            decision_made=decision_made,
            context=context,
            input_data=input_data or {},
            initial_confidence=confidence,
            alternatives=alternatives or [],
            tags=tags or []
        )
        
        self.decisions[record.id] = record
        self.decisions_by_type[decision_type].append(record.id)
        
        self.metrics["total_decisions"] += 1
        
        # Cleanup se necessario
        if len(self.decisions) > self.max_records:
            self._cleanup_old_records()
        
        decision_logger.debug(f"Decision recorded: {record.id} ({decision_type})")
        
        return record
    
    def resolve_decision(self,
                        decision_id: str,
                        outcome: DecisionOutcome,
                        details: Dict = None,
                        execution_time_ms: int = None) -> Optional[DecisionRecord]:
        """Risolve una decisione con il suo outcome"""
        record = self.decisions.get(decision_id)
        if not record:
            return None
        
        record.resolve(outcome, details)
        
        if execution_time_ms is not None:
            record.execution_time_ms = execution_time_ms
        
        # Aggiorna indici
        self.decisions_by_outcome[outcome].append(decision_id)
        
        if outcome == DecisionOutcome.SUCCESS:
            self.metrics["successful_decisions"] += 1
        
        # Trigger learning
        self._learn_from_decision(record)
        
        self._save_periodically()
        
        decision_logger.debug(f"Decision resolved: {decision_id} -> {outcome.value}")
        
        return record
    
    def add_decision_feedback(self,
                             decision_id: str,
                             feedback_type: FeedbackType,
                             message: str = "",
                             details: Dict = None):
        """Aggiunge feedback a una decisione"""
        record = self.decisions.get(decision_id)
        if record:
            record.add_feedback(feedback_type, message, details)
            self._learn_from_feedback(record, feedback_type)
    
    # === Action Recording ===
    
    def record_action(self,
                     action_type: str,
                     action_params: Dict[str, Any],
                     context: Dict[str, Any] = None,
                     decision_id: str = None) -> ActionResult:
        """Registra un'azione in esecuzione"""
        action = ActionResult(
            action_type=action_type,
            action_params=action_params,
            context=context or {},
            decision_id=decision_id
        )
        
        self.actions[action.id] = action
        self.actions_by_type[action_type].append(action.id)
        
        self.metrics["total_actions"] += 1
        
        return action
    
    def complete_action(self,
                       action_id: str,
                       success: bool,
                       result_data: Dict = None,
                       error: str = None) -> Optional[ActionResult]:
        """Completa un'azione con il risultato"""
        action = self.actions.get(action_id)
        if not action:
            return None
        
        action.complete(success, result_data, error)
        
        if success:
            self.metrics["successful_actions"] += 1
        
        # Se collegata a una decisione, propagate result
        if action.decision_id:
            outcome = DecisionOutcome.SUCCESS if success else DecisionOutcome.FAILURE
            self.resolve_decision(action.decision_id, outcome, result_data)
        
        # Learn from action
        self._learn_from_action(action)
        
        return action
    
    # === Learning ===
    
    def _learn_from_decision(self, decision: DecisionRecord):
        """Apprende dalla decisione completata"""
        # Cerca pattern simili
        similar = self._find_similar_decisions(decision)
        
        if len(similar) >= self.min_sample_size:
            # Calcola success rate per questo tipo di decisione nel contesto
            success_rates = [d.success_score() for d, _ in similar]
            avg_success = sum(success_rates) / len(success_rates)
            
            # Genera insight se c'è un pattern chiaro
            if avg_success > 0.8 or avg_success < 0.3:
                self._generate_insight(decision, similar, avg_success)
    
    def _learn_from_action(self, action: ActionResult):
        """Apprende dall'azione completata"""
        # Trova azioni simili
        similar_actions = [
            self.actions[aid]
            for aid in self.actions_by_type.get(action.action_type, [])[-100:]
            if aid != action.id and aid in self.actions
        ]
        
        if len(similar_actions) >= self.min_sample_size:
            # Calcola metriche
            success_rate = sum(1 for a in similar_actions if a.success) / len(similar_actions)
            avg_time = sum(a.execution_time_ms for a in similar_actions) / len(similar_actions)
            
            # Se performance sotto soglia, suggerisci adattamento
            if success_rate < 0.5:
                self._suggest_adaptation(
                    AdaptationType.STRATEGY_SWITCH,
                    f"Action type '{action.action_type}' has low success rate",
                    action.action_type,
                    evidence=[action.id]
                )
    
    def _learn_from_feedback(self, decision: DecisionRecord, feedback_type: FeedbackType):
        """Apprende dal feedback ricevuto"""
        if feedback_type == FeedbackType.CORRECTION:
            # Feedback di correzione è particolarmente importante
            self._generate_correction_insight(decision)
    
    def _find_similar_decisions(self, 
                               decision: DecisionRecord,
                               limit: int = 20) -> List[Tuple[DecisionRecord, float]]:
        """Trova decisioni simili"""
        results = []
        
        for did in self.decisions_by_type.get(decision.decision_type, []):
            if did == decision.id:
                continue
            
            other = self.decisions.get(did)
            if not other or other.outcome == DecisionOutcome.PENDING:
                continue
            
            # Calcola similarità
            score = self._decision_similarity(decision, other)
            
            if score > 0.5:
                results.append((other, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _decision_similarity(self, d1: DecisionRecord, d2: DecisionRecord) -> float:
        """Calcola similarità tra due decisioni"""
        score = 0.0
        
        # Stesso tipo
        if d1.decision_type == d2.decision_type:
            score += 0.4
        
        # Contesto simile
        common_context_keys = set(d1.context.keys()) & set(d2.context.keys())
        if common_context_keys:
            matching_values = sum(
                1 for k in common_context_keys
                if d1.context.get(k) == d2.context.get(k)
            )
            score += (matching_values / len(common_context_keys)) * 0.3
        
        # Tags comuni
        if d1.tags and d2.tags:
            common_tags = set(d1.tags) & set(d2.tags)
            score += len(common_tags) / max(len(d1.tags), len(d2.tags)) * 0.2
        
        # Confidence simile
        conf_diff = abs(d1.initial_confidence - d2.initial_confidence)
        score += (1 - conf_diff) * 0.1
        
        return score
    
    def _generate_insight(self, 
                         decision: DecisionRecord,
                         similar: List[Tuple[DecisionRecord, float]],
                         avg_success: float):
        """Genera nuovo insight"""
        # Estrai condizioni comuni
        common_conditions = self._extract_common_conditions(
            [d.context for d, _ in similar]
        )
        
        if not common_conditions:
            return
        
        insight_type = "success_pattern" if avg_success > 0.7 else "failure_pattern"
        
        description = (
            f"Quando {common_conditions}, decisioni di tipo '{decision.decision_type}' "
            f"hanno success rate del {avg_success:.0%}"
        )
        
        recommendation = {
            "action": "apply" if avg_success > 0.7 else "avoid",
            "decision_type": decision.decision_type,
            "conditions": common_conditions
        }
        
        insight = LearningInsight(
            insight_type=insight_type,
            description=description,
            trigger_conditions=common_conditions,
            recommendation=recommendation,
            supporting_decisions=[d.id for d, _ in similar[:10]],
            confidence=min(0.9, len(similar) / 20),
            sample_size=len(similar)
        )
        
        # Verifica se già esiste insight simile
        for existing in self.insights.values():
            if self._insights_similar(insight, existing):
                # Aggiorna esistente invece di creare nuovo
                existing.sample_size = len(similar)
                existing.confidence = insight.confidence
                existing.last_validated = datetime.now()
                return
        
        self.insights[insight.id] = insight
        self.metrics["insights_discovered"] += 1
        
        decision_logger.info(f"New insight discovered: {insight.description[:50]}...")
    
    def _extract_common_conditions(self, contexts: List[Dict]) -> Dict[str, Any]:
        """Estrae condizioni comuni da una lista di contesti"""
        if not contexts:
            return {}
        
        # Conta occorrenze per ogni chiave-valore
        value_counts = defaultdict(lambda: defaultdict(int))
        
        for ctx in contexts:
            for key, value in ctx.items():
                value_counts[key][str(value)] += 1
        
        # Mantieni solo chiavi-valori presenti in >60% dei casi
        threshold = len(contexts) * 0.6
        common = {}
        
        for key, values in value_counts.items():
            for value, count in values.items():
                if count >= threshold:
                    common[key] = value
                    break
        
        return common
    
    def _insights_similar(self, i1: LearningInsight, i2: LearningInsight) -> bool:
        """Verifica se due insight sono simili"""
        if i1.insight_type != i2.insight_type:
            return False
        
        if i1.recommendation.get("decision_type") != i2.recommendation.get("decision_type"):
            return False
        
        # Condizioni simili
        common_keys = set(i1.trigger_conditions.keys()) & set(i2.trigger_conditions.keys())
        if not common_keys:
            return False
        
        matching = sum(
            1 for k in common_keys
            if i1.trigger_conditions.get(k) == i2.trigger_conditions.get(k)
        )
        
        return matching / len(common_keys) > 0.7
    
    def _generate_correction_insight(self, decision: DecisionRecord):
        """Genera insight da correzione"""
        # Estrai la correzione dal feedback
        corrections = [
            f for f in decision.feedback
            if f["type"] == "correction"
        ]
        
        if not corrections:
            return
        
        insight = LearningInsight(
            insight_type="correction",
            description=f"Correzione per decisione tipo '{decision.decision_type}'",
            trigger_conditions=decision.context,
            recommendation={
                "action": "correct",
                "original": decision.decision_made,
                "corrections": [c.get("details", {}) for c in corrections]
            },
            supporting_decisions=[decision.id],
            confidence=0.8,
            sample_size=1
        )
        
        self.insights[insight.id] = insight
        self.metrics["insights_discovered"] += 1
    
    def _suggest_adaptation(self,
                           adaptation_type: AdaptationType,
                           reason: str,
                           target: str,
                           evidence: List[str] = None):
        """Suggerisce un adattamento"""
        # Calcola nuovo valore basato su metriche
        current_value = self.adaptive_params.get(target)
        new_value = self._calculate_adapted_value(adaptation_type, target)
        
        if new_value is None or new_value == current_value:
            return
        
        adaptation = Adaptation(
            adaptation_type=adaptation_type,
            description=reason,
            target=target,
            old_value=current_value,
            new_value=new_value,
            trigger_reason=reason,
            evidence=evidence or [],
            improvement_expected=0.1  # Stima conservativa
        )
        
        self.adaptations.append(adaptation)
        
        decision_logger.info(f"Adaptation suggested: {target} -> {new_value}")
    
    def _calculate_adapted_value(self, 
                                adaptation_type: AdaptationType,
                                target: str) -> Any:
        """Calcola nuovo valore adattato"""
        current = self.adaptive_params.get(target)
        
        if current is None:
            return None
        
        if adaptation_type == AdaptationType.PARAMETER_TUNING:
            # Piccolo aggiustamento
            if isinstance(current, (int, float)):
                return current * (1 + self.learning_rate)
        
        elif adaptation_type == AdaptationType.THRESHOLD_ADJUST:
            # Aggiusta soglie basandosi su success rate
            success_rate = self.get_success_rate()
            
            if success_rate < 0.5:
                # Abbassa soglie
                return current * 0.9 if isinstance(current, (int, float)) else current
            elif success_rate > 0.8:
                # Alza soglie
                return current * 1.1 if isinstance(current, (int, float)) else current
        
        return None
    
    # === Query Methods ===
    
    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Recupera decisione"""
        return self.decisions.get(decision_id)
    
    def get_recent_decisions(self, 
                            n: int = 20,
                            decision_type: str = None,
                            outcome: DecisionOutcome = None) -> List[DecisionRecord]:
        """Recupera decisioni recenti"""
        decisions = list(self.decisions.values())
        
        if decision_type:
            decisions = [d for d in decisions if d.decision_type == decision_type]
        if outcome:
            decisions = [d for d in decisions if d.outcome == outcome]
        
        decisions.sort(key=lambda d: d.decided_at, reverse=True)
        return decisions[:n]
    
    def get_insights(self,
                    insight_type: str = None,
                    min_confidence: float = 0.0,
                    active_only: bool = True) -> List[LearningInsight]:
        """Recupera insights"""
        insights = list(self.insights.values())
        
        if insight_type:
            insights = [i for i in insights if i.insight_type == insight_type]
        if min_confidence > 0:
            insights = [i for i in insights if i.confidence >= min_confidence]
        if active_only:
            insights = [i for i in insights if i.is_active]
        
        insights.sort(key=lambda i: i.confidence, reverse=True)
        return insights
    
    def get_applicable_insights(self, context: Dict[str, Any]) -> List[LearningInsight]:
        """Trova insights applicabili al contesto"""
        applicable = []
        
        for insight in self.insights.values():
            if not insight.is_active:
                continue
            
            # Verifica se condizioni matchano
            conditions_met = True
            for key, expected in insight.trigger_conditions.items():
                if context.get(key) != expected:
                    conditions_met = False
                    break
            
            if conditions_met:
                applicable.append(insight)
        
        applicable.sort(key=lambda i: i.confidence, reverse=True)
        return applicable
    
    def apply_insight(self, insight_id: str, correct: bool):
        """Registra applicazione di un insight"""
        insight = self.insights.get(insight_id)
        if insight:
            insight.apply(correct)
    
    def get_pending_adaptations(self) -> List[Adaptation]:
        """Recupera adattamenti in attesa di applicazione"""
        return [a for a in self.adaptations if a.improvement_actual is None]
    
    def apply_adaptation(self, adaptation_id: str):
        """Applica un adattamento"""
        for adaptation in self.adaptations:
            if adaptation.id == adaptation_id:
                self.adaptive_params[adaptation.target] = adaptation.new_value
                self.metrics["adaptations_made"] += 1
                
                decision_logger.info(
                    f"Adaptation applied: {adaptation.target} = {adaptation.new_value}"
                )
                return True
        return False
    
    def evaluate_adaptation(self, adaptation_id: str, improvement: float):
        """Valuta risultato di un adattamento"""
        for adaptation in self.adaptations:
            if adaptation.id == adaptation_id:
                adaptation.improvement_actual = improvement
                adaptation.evaluated_at = datetime.now()
                
                # Se peggiore del previsto, considera rollback
                if improvement < 0:
                    self.adaptive_params[adaptation.target] = adaptation.old_value
                    decision_logger.warning(
                        f"Adaptation rolled back: {adaptation.target}"
                    )
                
                return True
        return False
    
    # === Metrics ===
    
    def get_success_rate(self, decision_type: str = None) -> float:
        """Calcola success rate"""
        if decision_type:
            decision_ids = self.decisions_by_type.get(decision_type, [])
        else:
            decision_ids = list(self.decisions.keys())
        
        if not decision_ids:
            return 0.5
        
        resolved = [
            self.decisions[did]
            for did in decision_ids
            if did in self.decisions and self.decisions[did].outcome != DecisionOutcome.PENDING
        ]
        
        if not resolved:
            return 0.5
        
        return sum(d.success_score() for d in resolved) / len(resolved)
    
    def get_stats(self) -> Dict:
        """Statistiche memoria decisionale"""
        return {
            **self.metrics,
            "total_decisions_stored": len(self.decisions),
            "total_actions_stored": len(self.actions),
            "total_insights": len(self.insights),
            "active_insights": sum(1 for i in self.insights.values() if i.is_active),
            "total_adaptations": len(self.adaptations),
            "overall_success_rate": round(self.get_success_rate(), 3),
            "adaptive_params": self.adaptive_params
        }
    
    # === Persistence ===
    
    def _save_periodically(self):
        """Salva periodicamente"""
        if len(self.decisions) % 50 == 0:
            self._save()
    
    def _save(self):
        """Salva su disco"""
        try:
            # Salva decisioni
            decisions_data = {
                did: d.to_dict()
                for did, d in list(self.decisions.items())[-2000:]
            }
            
            with open(self.storage_path / "decisions.json", "w") as f:
                json.dump(decisions_data, f)
            
            # Salva insights
            insights_data = {iid: i.to_dict() for iid, i in self.insights.items()}
            
            with open(self.storage_path / "insights.json", "w") as f:
                json.dump(insights_data, f)
            
            # Salva adaptations
            adaptations_data = [a.to_dict() for a in self.adaptations[-100:]]
            
            with open(self.storage_path / "adaptations.json", "w") as f:
                json.dump(adaptations_data, f)
            
            # Salva metrics
            with open(self.storage_path / "metrics.json", "w") as f:
                json.dump(self.metrics, f)
                
        except Exception as e:
            decision_logger.error(f"Failed to save decision memory: {e}")
    
    def _load(self):
        """Carica da disco"""
        # Load decisions
        decisions_file = self.storage_path / "decisions.json"
        if decisions_file.exists():
            try:
                with open(decisions_file) as f:
                    data = json.load(f)
                
                for did, ddata in data.items():
                    record = DecisionRecord(
                        id=did,
                        decision_type=ddata.get("decision_type", ""),
                        decision_made=ddata.get("decision_made", {}),
                        context=ddata.get("context", {}),
                        initial_confidence=ddata.get("initial_confidence", 0.5),
                        outcome=DecisionOutcome(ddata.get("outcome", "unknown")),
                        outcome_details=ddata.get("outcome_details", {}),
                        tags=ddata.get("tags", [])
                    )
                    record.decided_at = datetime.fromisoformat(ddata["decided_at"])
                    if ddata.get("resolved_at"):
                        record.resolved_at = datetime.fromisoformat(ddata["resolved_at"])
                    
                    self.decisions[did] = record
                    self.decisions_by_type[record.decision_type].append(did)
                    self.decisions_by_outcome[record.outcome].append(did)
                    
            except Exception as e:
                decision_logger.error(f"Failed to load decisions: {e}")
        
        # Load insights
        insights_file = self.storage_path / "insights.json"
        if insights_file.exists():
            try:
                with open(insights_file) as f:
                    data = json.load(f)
                
                for iid, idata in data.items():
                    insight = LearningInsight(
                        id=iid,
                        insight_type=idata.get("insight_type", ""),
                        description=idata.get("description", ""),
                        trigger_conditions=idata.get("trigger_conditions", {}),
                        recommendation=idata.get("recommendation", {}),
                        confidence=idata.get("confidence", 0.5),
                        sample_size=idata.get("sample_size", 0),
                        is_active=idata.get("is_active", True),
                        times_applied=idata.get("times_applied", 0)
                    )
                    self.insights[iid] = insight
                    
            except Exception as e:
                decision_logger.error(f"Failed to load insights: {e}")
        
        # Load metrics
        metrics_file = self.storage_path / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    self.metrics.update(json.load(f))
            except Exception as e:
                decision_logger.error(f"Failed to load metrics: {e}")
    
    def _cleanup_old_records(self):
        """Rimuove record vecchi"""
        # Mantieni solo i più recenti
        keep_count = int(self.max_records * 0.8)
        
        decisions_sorted = sorted(
            self.decisions.values(),
            key=lambda d: d.decided_at,
            reverse=True
        )
        
        keep_ids = {d.id for d in decisions_sorted[:keep_count]}
        remove_ids = set(self.decisions.keys()) - keep_ids
        
        for did in remove_ids:
            del self.decisions[did]
        
        # Ricostruisci indici
        self.decisions_by_type = defaultdict(list)
        self.decisions_by_outcome = defaultdict(list)
        
        for d in self.decisions.values():
            self.decisions_by_type[d.decision_type].append(d.id)
            self.decisions_by_outcome[d.outcome].append(d.id)
        
        decision_logger.info(f"Cleaned up {len(remove_ids)} old decisions")
