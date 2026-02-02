# /backend/gideon/intent_outcome_resolver.py
"""
🔮 GIDEON 3.0 - Intent vs Outcome Resolver
Confronta obiettivi richiesti con risultati attesi/ottenuti.
NON esegue azioni - fornisce solo analisi e riconciliazione.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
import logging

logger = logging.getLogger(__name__)


class IntentStatus(Enum):
    """Stato di un intent"""
    PENDING = "pending"           # In attesa
    IN_PROGRESS = "in_progress"   # In elaborazione
    RESOLVED = "resolved"         # Risolto
    PARTIAL = "partial"           # Parzialmente risolto
    FAILED = "failed"            # Fallito
    CANCELLED = "cancelled"       # Cancellato


class MatchQuality(Enum):
    """Qualità del match intent-outcome"""
    EXACT = "exact"               # Match perfetto
    CLOSE = "close"               # Molto vicino
    PARTIAL = "partial"           # Parziale
    DIVERGENT = "divergent"       # Divergente
    OPPOSITE = "opposite"         # Opposto
    UNRELATED = "unrelated"       # Non correlato


class GapType(Enum):
    """Tipi di gap tra intent e outcome"""
    NONE = "none"                 # Nessun gap
    SCOPE = "scope"               # Gap di scope
    QUALITY = "quality"           # Gap di qualità
    TIMING = "timing"             # Gap temporale
    COMPLETENESS = "completeness" # Gap di completezza
    ACCURACY = "accuracy"         # Gap di accuratezza
    SIDE_EFFECT = "side_effect"   # Effetti collaterali


class ResolutionAction(Enum):
    """Azioni di risoluzione"""
    ACCEPT = "accept"             # Accetta outcome
    RETRY = "retry"               # Riprova
    MODIFY = "modify"             # Modifica approccio
    ESCALATE = "escalate"         # Escalation
    ABORT = "abort"               # Interrompi
    COMPENSATE = "compensate"     # Compensa


@dataclass
class Intent:
    """Rappresentazione di un intent"""
    id: str
    description: str
    
    # Requirements
    expected_outcomes: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    # Priority and timing
    priority: int = 5
    deadline: Optional[datetime] = None
    
    # Status
    status: IntentStatus = IntentStatus.PENDING
    
    # Context
    context: Dict = field(default_factory=dict)
    parameters: Dict = field(default_factory=dict)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "expected_outcomes": self.expected_outcomes,
            "success_criteria": self.success_criteria,
            "constraints": self.constraints,
            "priority": self.priority,
            "status": self.status.value,
            "deadline": self.deadline.isoformat() if self.deadline else None
        }


@dataclass
class Outcome:
    """Rappresentazione di un outcome"""
    id: str
    intent_id: str
    description: str
    
    # Results
    achieved_results: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Quality
    success_rate: float = 0.0
    completeness: float = 0.0
    
    # Side effects
    side_effects: List[str] = field(default_factory=list)
    unexpected_results: List[str] = field(default_factory=list)
    
    # Timing
    execution_time_ms: float = 0.0
    occurred_at: datetime = field(default_factory=datetime.now)
    
    # Error info
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "intent_id": self.intent_id,
            "description": self.description,
            "achieved_results": self.achieved_results,
            "metrics": self.metrics,
            "success_rate": round(self.success_rate, 2),
            "completeness": round(self.completeness, 2),
            "side_effects": self.side_effects,
            "errors": self.errors
        }


@dataclass
class Gap:
    """Gap identificato tra intent e outcome"""
    gap_type: GapType
    description: str
    severity: float  # 0-1
    
    # Details
    expected: str = ""
    actual: str = ""
    difference: float = 0.0
    
    # Resolution
    resolvable: bool = True
    suggested_action: ResolutionAction = ResolutionAction.MODIFY
    
    def to_dict(self) -> dict:
        return {
            "type": self.gap_type.value,
            "description": self.description,
            "severity": round(self.severity, 2),
            "expected": self.expected,
            "actual": self.actual,
            "resolvable": self.resolvable,
            "suggested_action": self.suggested_action.value
        }


@dataclass
class Resolution:
    """Risoluzione proposta per gap"""
    id: str
    intent_id: str
    outcome_id: str
    
    # Match analysis
    match_quality: MatchQuality
    match_score: float  # 0-1
    
    # Gaps
    gaps: List[Gap] = field(default_factory=list)
    total_gap_severity: float = 0.0
    
    # Recommendation
    recommended_action: ResolutionAction = ResolutionAction.ACCEPT
    action_details: str = ""
    confidence: float = 0.5
    
    # Alternative approaches
    alternatives: List[Dict] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "intent_id": self.intent_id,
            "outcome_id": self.outcome_id,
            "match_quality": self.match_quality.value,
            "match_score": round(self.match_score, 3),
            "gaps": [g.to_dict() for g in self.gaps],
            "total_gap_severity": round(self.total_gap_severity, 2),
            "recommended_action": self.recommended_action.value,
            "action_details": self.action_details,
            "confidence": round(self.confidence, 2),
            "alternatives": self.alternatives
        }


@dataclass
class ReconciliationResult:
    """Risultato della riconciliazione"""
    intent: Intent
    outcome: Outcome
    resolution: Resolution
    
    # Final status
    is_satisfied: bool = False
    satisfaction_level: float = 0.0
    
    # Learnings
    learnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "intent": self.intent.to_dict(),
            "outcome": self.outcome.to_dict(),
            "resolution": self.resolution.to_dict(),
            "is_satisfied": self.is_satisfied,
            "satisfaction_level": round(self.satisfaction_level, 2),
            "learnings": self.learnings
        }


class IntentOutcomeResolver:
    """
    Risolve discrepanze tra intent e outcome.
    Analizza, confronta e propone azioni correttive.
    """
    
    def __init__(self):
        # Intent tracking
        self.intents: Dict[str, Intent] = {}
        self.outcomes: Dict[str, Outcome] = {}
        self.resolutions: List[Resolution] = []
        
        # History
        self.reconciliation_history: List[ReconciliationResult] = []
        
        # Thresholds
        self.satisfaction_threshold = 0.7
        self.exact_match_threshold = 0.95
        self.close_match_threshold = 0.8
        self.partial_match_threshold = 0.5
        
        # Learning
        self.gap_patterns: Dict[str, List[Gap]] = {}
        self.successful_resolutions: Dict[str, int] = {}
        
        # Counters
        self._intent_counter = 0
        self._outcome_counter = 0
        self._resolution_counter = 0
    
    # === Intent Management ===
    
    def register_intent(self, description: str,
                        expected_outcomes: List[str] = None,
                        success_criteria: List[str] = None,
                        constraints: List[str] = None,
                        priority: int = 5,
                        deadline: datetime = None,
                        context: Dict = None) -> Intent:
        """
        Registra un nuovo intent.
        """
        self._intent_counter += 1
        intent_id = f"intent_{self._intent_counter}"
        
        intent = Intent(
            id=intent_id,
            description=description,
            expected_outcomes=expected_outcomes or [],
            success_criteria=success_criteria or [],
            constraints=constraints or [],
            priority=priority,
            deadline=deadline,
            context=context or {}
        )
        
        self.intents[intent_id] = intent
        logger.info(f"Intent registrato: {intent_id}")
        
        return intent
    
    def update_intent_status(self, intent_id: str, status: IntentStatus) -> bool:
        """Aggiorna stato intent"""
        if intent_id not in self.intents:
            return False
        
        self.intents[intent_id].status = status
        if status == IntentStatus.RESOLVED:
            self.intents[intent_id].resolved_at = datetime.now()
        
        return True
    
    # === Outcome Recording ===
    
    def record_outcome(self, intent_id: str, description: str,
                       achieved_results: List[str] = None,
                       metrics: Dict[str, float] = None,
                       side_effects: List[str] = None,
                       errors: List[str] = None,
                       execution_time_ms: float = 0.0) -> Outcome:
        """
        Registra outcome per un intent.
        """
        self._outcome_counter += 1
        outcome_id = f"outcome_{self._outcome_counter}"
        
        outcome = Outcome(
            id=outcome_id,
            intent_id=intent_id,
            description=description,
            achieved_results=achieved_results or [],
            metrics=metrics or {},
            side_effects=side_effects or [],
            errors=errors or [],
            execution_time_ms=execution_time_ms
        )
        
        # Calculate completeness
        if intent_id in self.intents:
            intent = self.intents[intent_id]
            if intent.expected_outcomes:
                matched = sum(1 for exp in intent.expected_outcomes 
                            if any(exp.lower() in res.lower() 
                                  for res in achieved_results or []))
                outcome.completeness = matched / len(intent.expected_outcomes)
        
        # Calculate success rate
        outcome.success_rate = self._calculate_success_rate(outcome, 
                                                           self.intents.get(intent_id))
        
        self.outcomes[outcome_id] = outcome
        return outcome
    
    def _calculate_success_rate(self, outcome: Outcome, intent: Optional[Intent]) -> float:
        """Calcola success rate dell'outcome"""
        if not intent:
            return 0.5
        
        rate = 0.0
        factors = 0
        
        # Completeness factor
        rate += outcome.completeness * 0.4
        factors += 0.4
        
        # Error factor
        if not outcome.errors:
            rate += 0.3
        else:
            rate += 0.3 * max(0, 1 - len(outcome.errors) * 0.2)
        factors += 0.3
        
        # Metrics factor (if defined)
        if outcome.metrics:
            metric_scores = list(outcome.metrics.values())
            avg_metric = statistics.mean(metric_scores) if metric_scores else 0.5
            rate += avg_metric * 0.3
            factors += 0.3
        
        return rate / factors if factors else 0.5
    
    # === Resolution ===
    
    def resolve(self, intent_id: str, outcome_id: str) -> Resolution:
        """
        Risolve confronto tra intent e outcome.
        """
        if intent_id not in self.intents:
            raise ValueError(f"Intent {intent_id} not found")
        if outcome_id not in self.outcomes:
            raise ValueError(f"Outcome {outcome_id} not found")
        
        intent = self.intents[intent_id]
        outcome = self.outcomes[outcome_id]
        
        self._resolution_counter += 1
        resolution_id = f"res_{self._resolution_counter}"
        
        # Analyze match
        match_quality, match_score = self._analyze_match(intent, outcome)
        
        # Identify gaps
        gaps = self._identify_gaps(intent, outcome)
        
        # Calculate total severity
        total_severity = sum(g.severity for g in gaps) / len(gaps) if gaps else 0
        
        # Determine recommended action
        action, details = self._recommend_action(match_quality, gaps, outcome)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(intent, outcome, gaps)
        
        # Calculate confidence
        confidence = self._calculate_resolution_confidence(match_score, gaps)
        
        resolution = Resolution(
            id=resolution_id,
            intent_id=intent_id,
            outcome_id=outcome_id,
            match_quality=match_quality,
            match_score=match_score,
            gaps=gaps,
            total_gap_severity=total_severity,
            recommended_action=action,
            action_details=details,
            confidence=confidence,
            alternatives=alternatives
        )
        
        self.resolutions.append(resolution)
        return resolution
    
    def _analyze_match(self, intent: Intent, outcome: Outcome) -> Tuple[MatchQuality, float]:
        """Analizza qualità del match"""
        score = 0.0
        
        # Completeness contributes to score
        score += outcome.completeness * 0.4
        
        # Success rate contributes
        score += outcome.success_rate * 0.3
        
        # Criteria matching
        if intent.success_criteria:
            criteria_met = 0
            for criterion in intent.success_criteria:
                # Simple keyword matching (could be enhanced)
                if any(criterion.lower() in res.lower() 
                      for res in outcome.achieved_results):
                    criteria_met += 1
            criteria_score = criteria_met / len(intent.success_criteria)
            score += criteria_score * 0.2
        else:
            score += 0.2
        
        # Penalty for errors
        if outcome.errors:
            score -= min(0.3, len(outcome.errors) * 0.1)
        
        # Penalty for side effects
        if outcome.side_effects:
            score -= min(0.2, len(outcome.side_effects) * 0.05)
        
        score = max(0, min(1, score))
        
        # Determine quality category
        if score >= self.exact_match_threshold:
            quality = MatchQuality.EXACT
        elif score >= self.close_match_threshold:
            quality = MatchQuality.CLOSE
        elif score >= self.partial_match_threshold:
            quality = MatchQuality.PARTIAL
        elif score > 0.2:
            quality = MatchQuality.DIVERGENT
        else:
            quality = MatchQuality.OPPOSITE if outcome.errors else MatchQuality.UNRELATED
        
        return quality, score
    
    def _identify_gaps(self, intent: Intent, outcome: Outcome) -> List[Gap]:
        """Identifica gap tra intent e outcome"""
        gaps = []
        
        # Completeness gap
        if outcome.completeness < 1.0:
            missing = [exp for exp in intent.expected_outcomes 
                      if not any(exp.lower() in res.lower() 
                                for res in outcome.achieved_results)]
            
            gaps.append(Gap(
                gap_type=GapType.COMPLETENESS,
                description=f"Outcome incompleto: mancano {len(missing)} risultati attesi",
                severity=1 - outcome.completeness,
                expected=str(intent.expected_outcomes),
                actual=str(outcome.achieved_results),
                suggested_action=ResolutionAction.RETRY
            ))
        
        # Quality gap
        if outcome.success_rate < self.satisfaction_threshold:
            gaps.append(Gap(
                gap_type=GapType.QUALITY,
                description=f"Qualità insufficiente: {outcome.success_rate:.0%} vs {self.satisfaction_threshold:.0%} richiesto",
                severity=self.satisfaction_threshold - outcome.success_rate,
                expected=f"{self.satisfaction_threshold:.0%}",
                actual=f"{outcome.success_rate:.0%}",
                suggested_action=ResolutionAction.MODIFY
            ))
        
        # Timing gap
        if intent.deadline and outcome.occurred_at > intent.deadline:
            delay = (outcome.occurred_at - intent.deadline).total_seconds()
            gaps.append(Gap(
                gap_type=GapType.TIMING,
                description=f"Deadline superata di {delay/3600:.1f} ore",
                severity=min(1.0, delay / (24 * 3600)),
                expected=intent.deadline.isoformat(),
                actual=outcome.occurred_at.isoformat(),
                resolvable=False,
                suggested_action=ResolutionAction.COMPENSATE
            ))
        
        # Side effects gap
        if outcome.side_effects:
            for effect in outcome.side_effects:
                gaps.append(Gap(
                    gap_type=GapType.SIDE_EFFECT,
                    description=f"Effetto collaterale: {effect}",
                    severity=0.3,
                    actual=effect,
                    suggested_action=ResolutionAction.COMPENSATE
                ))
        
        # Error gap
        if outcome.errors:
            gaps.append(Gap(
                gap_type=GapType.ACCURACY,
                description=f"Errori riscontrati: {len(outcome.errors)}",
                severity=min(1.0, len(outcome.errors) * 0.3),
                actual=str(outcome.errors),
                suggested_action=ResolutionAction.RETRY
            ))
        
        return gaps
    
    def _recommend_action(self, match_quality: MatchQuality,
                          gaps: List[Gap],
                          outcome: Outcome) -> Tuple[ResolutionAction, str]:
        """Raccomanda azione di risoluzione"""
        
        if match_quality == MatchQuality.EXACT:
            return ResolutionAction.ACCEPT, "Outcome soddisfa completamente l'intent"
        
        if match_quality == MatchQuality.CLOSE:
            if any(g.gap_type == GapType.SIDE_EFFECT for g in gaps):
                return ResolutionAction.COMPENSATE, "Accettare con compensazione effetti collaterali"
            return ResolutionAction.ACCEPT, "Outcome sufficientemente vicino all'intent"
        
        if match_quality == MatchQuality.PARTIAL:
            retry_worthy = any(g.gap_type in [GapType.COMPLETENESS, GapType.ACCURACY] 
                             and g.resolvable for g in gaps)
            if retry_worthy:
                return ResolutionAction.RETRY, "Riprovare per completare risultati mancanti"
            return ResolutionAction.MODIFY, "Modificare approccio per migliorare match"
        
        if match_quality == MatchQuality.DIVERGENT:
            if outcome.errors:
                return ResolutionAction.RETRY, "Errori critici, necessario riprovare"
            return ResolutionAction.MODIFY, "Significativa divergenza, rivedere approccio"
        
        if match_quality == MatchQuality.OPPOSITE:
            return ResolutionAction.ABORT, "Outcome opposto all'intent, abortire"
        
        return ResolutionAction.ESCALATE, "Situazione non chiara, escalation necessaria"
    
    def _generate_alternatives(self, intent: Intent, outcome: Outcome,
                                gaps: List[Gap]) -> List[Dict]:
        """Genera approcci alternativi"""
        alternatives = []
        
        for gap in gaps:
            if gap.gap_type == GapType.COMPLETENESS:
                alternatives.append({
                    "approach": "Esecuzione incrementale",
                    "description": "Completare i risultati mancanti separatamente",
                    "estimated_effort": "medio"
                })
            
            elif gap.gap_type == GapType.QUALITY:
                alternatives.append({
                    "approach": "Ottimizzazione parametri",
                    "description": "Regolare parametri per migliorare qualità",
                    "estimated_effort": "basso"
                })
            
            elif gap.gap_type == GapType.TIMING:
                alternatives.append({
                    "approach": "Parallelizzazione",
                    "description": "Eseguire in parallelo per ridurre tempo",
                    "estimated_effort": "alto"
                })
        
        # Remove duplicates
        seen = set()
        unique = []
        for alt in alternatives:
            key = alt["approach"]
            if key not in seen:
                seen.add(key)
                unique.append(alt)
        
        return unique[:3]  # Max 3 alternatives
    
    def _calculate_resolution_confidence(self, match_score: float,
                                          gaps: List[Gap]) -> float:
        """Calcola confidenza nella risoluzione"""
        base_confidence = match_score
        
        # Reduce for unresolvable gaps
        unresolvable = sum(1 for g in gaps if not g.resolvable)
        base_confidence -= unresolvable * 0.1
        
        # Historical success rate for similar resolutions
        # (simplified - would use pattern matching in production)
        
        return max(0.3, min(0.95, base_confidence))
    
    # === Reconciliation ===
    
    def reconcile(self, intent_id: str, outcome_id: str) -> ReconciliationResult:
        """
        Esegue riconciliazione completa tra intent e outcome.
        """
        resolution = self.resolve(intent_id, outcome_id)
        
        intent = self.intents[intent_id]
        outcome = self.outcomes[outcome_id]
        
        # Determine satisfaction
        is_satisfied = resolution.match_score >= self.satisfaction_threshold
        
        # Extract learnings
        learnings = self._extract_learnings(intent, outcome, resolution)
        
        # Update intent status
        if is_satisfied:
            self.update_intent_status(intent_id, IntentStatus.RESOLVED)
        elif resolution.recommended_action == ResolutionAction.ABORT:
            self.update_intent_status(intent_id, IntentStatus.FAILED)
        else:
            self.update_intent_status(intent_id, IntentStatus.PARTIAL)
        
        result = ReconciliationResult(
            intent=intent,
            outcome=outcome,
            resolution=resolution,
            is_satisfied=is_satisfied,
            satisfaction_level=resolution.match_score,
            learnings=learnings
        )
        
        self.reconciliation_history.append(result)
        
        # Track patterns
        self._track_patterns(resolution)
        
        return result
    
    def _extract_learnings(self, intent: Intent, outcome: Outcome,
                           resolution: Resolution) -> List[str]:
        """Estrae learnings dalla riconciliazione"""
        learnings = []
        
        if resolution.match_quality == MatchQuality.EXACT:
            learnings.append("Approccio efficace, mantenere per casi simili")
        
        for gap in resolution.gaps:
            if gap.gap_type == GapType.COMPLETENESS:
                learnings.append(
                    f"Definire expected outcomes più specifici per garantire completezza"
                )
            elif gap.gap_type == GapType.TIMING:
                learnings.append(
                    f"Allocare più tempo o parallelizzare per rispettare deadline"
                )
            elif gap.gap_type == GapType.SIDE_EFFECT:
                learnings.append(
                    f"Considerare effetti collaterali nella pianificazione"
                )
        
        if outcome.errors:
            learnings.append(
                f"Implementare validazione più robusta per prevenire errori"
            )
        
        return learnings
    
    def _track_patterns(self, resolution: Resolution):
        """Traccia pattern per apprendimento"""
        for gap in resolution.gaps:
            gap_key = gap.gap_type.value
            if gap_key not in self.gap_patterns:
                self.gap_patterns[gap_key] = []
            self.gap_patterns[gap_key].append(gap)
        
        # Track successful resolutions
        action_key = resolution.recommended_action.value
        if action_key not in self.successful_resolutions:
            self.successful_resolutions[action_key] = 0
        if resolution.match_score >= self.satisfaction_threshold:
            self.successful_resolutions[action_key] += 1
    
    # === Queries ===
    
    def get_intent(self, intent_id: str) -> Optional[Dict]:
        """Ottiene intent per ID"""
        if intent_id in self.intents:
            return self.intents[intent_id].to_dict()
        return None
    
    def get_pending_intents(self) -> List[Dict]:
        """Ottiene intents pendenti"""
        return [i.to_dict() for i in self.intents.values() 
                if i.status == IntentStatus.PENDING]
    
    def get_resolution_history(self, intent_id: str = None) -> List[Dict]:
        """Ottiene storia risoluzioni"""
        results = self.reconciliation_history
        if intent_id:
            results = [r for r in results if r.intent.id == intent_id]
        return [r.to_dict() for r in results]
    
    def get_gap_statistics(self) -> Dict:
        """Statistiche sui gap"""
        stats = {}
        
        for gap_type, gaps in self.gap_patterns.items():
            stats[gap_type] = {
                "count": len(gaps),
                "avg_severity": statistics.mean(g.severity for g in gaps) if gaps else 0,
                "resolvable_ratio": sum(1 for g in gaps if g.resolvable) / len(gaps) if gaps else 0
            }
        
        return stats
    
    def get_status(self) -> Dict:
        """Stato del resolver"""
        return {
            "total_intents": len(self.intents),
            "pending_intents": sum(1 for i in self.intents.values() 
                                  if i.status == IntentStatus.PENDING),
            "resolved_intents": sum(1 for i in self.intents.values() 
                                   if i.status == IntentStatus.RESOLVED),
            "total_outcomes": len(self.outcomes),
            "total_resolutions": len(self.resolutions),
            "reconciliations": len(self.reconciliation_history),
            "satisfaction_rate": self._calculate_satisfaction_rate(),
            "gap_patterns": len(self.gap_patterns)
        }
    
    def _calculate_satisfaction_rate(self) -> float:
        """Calcola tasso di soddisfazione"""
        if not self.reconciliation_history:
            return 0.0
        
        satisfied = sum(1 for r in self.reconciliation_history if r.is_satisfied)
        return satisfied / len(self.reconciliation_history)
