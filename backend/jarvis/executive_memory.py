# /backend/jarvis/executive_memory.py
"""
JARVIS Executive Memory - Memoria Decisionale e Apprendimento
Registra decisioni, apprende dai risultati e aggiorna strategie.
"""

import json
import hashlib
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
import logging
import asyncio

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Tipi di outcome"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ERROR = "error"


class LearningSignal(Enum):
    """Segnali di apprendimento"""
    POSITIVE = "positive"      # Utente soddisfatto
    NEGATIVE = "negative"      # Utente insoddisfatto
    NEUTRAL = "neutral"        # Nessun feedback
    CORRECTION = "correction"  # Utente ha corretto
    REPEAT = "repeat"          # Utente ha ripetuto richiesta


class StrategyUpdate(Enum):
    """Tipi di aggiornamento strategia"""
    REINFORCE = "reinforce"    # Rafforza strategia
    WEAKEN = "weaken"          # Indebolisci strategia
    ADAPT = "adapt"            # Adatta parametri
    REPLACE = "replace"        # Sostituisci strategia
    COMBINE = "combine"        # Combina strategie


@dataclass
class Decision:
    """Registrazione di una decisione"""
    id: str
    timestamp: datetime
    intent: str
    context: dict
    options_considered: List[dict]
    chosen_option: dict
    reasoning: str
    confidence: float
    strategy_used: str
    priority_score: float
    risk_level: str
    user_id: str = "default"
    session_id: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
            "context": self.context,
            "options_considered": self.options_considered,
            "chosen_option": self.chosen_option,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used,
            "priority_score": self.priority_score,
            "risk_level": self.risk_level,
            "user_id": self.user_id,
            "session_id": self.session_id
        }


@dataclass
class ExecutionOutcome:
    """Outcome di un'esecuzione"""
    decision_id: str
    timestamp: datetime
    outcome_type: OutcomeType
    execution_time: float
    result: dict
    error: Optional[str] = None
    user_feedback: Optional[str] = None
    learning_signal: LearningSignal = LearningSignal.NEUTRAL
    metrics: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "outcome_type": self.outcome_type.value,
            "execution_time": self.execution_time,
            "result": self.result,
            "error": self.error,
            "user_feedback": self.user_feedback,
            "learning_signal": self.learning_signal.value,
            "metrics": self.metrics
        }


@dataclass
class StrategyProfile:
    """Profilo di una strategia"""
    name: str
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    avg_confidence: float = 0.5
    avg_execution_time: float = 0.0
    effectiveness_score: float = 0.5
    contexts_effective: List[str] = field(default_factory=list)
    contexts_ineffective: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None
    adaptations: List[dict] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.5
        return self.success_count / self.total_uses
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_uses": self.total_uses,
            "success_rate": self.success_rate,
            "avg_confidence": self.avg_confidence,
            "avg_execution_time": self.avg_execution_time,
            "effectiveness_score": self.effectiveness_score,
            "contexts_effective": self.contexts_effective,
            "contexts_ineffective": self.contexts_ineffective,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }


@dataclass
class UserPreference:
    """Preferenza utente appresa"""
    key: str
    value: Any
    confidence: float
    times_observed: int
    last_observed: datetime
    context_tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "times_observed": self.times_observed,
            "last_observed": self.last_observed.isoformat(),
            "context_tags": self.context_tags
        }


@dataclass
class Pattern:
    """Pattern riconosciuto"""
    id: str
    name: str
    pattern_type: str  # temporal, behavioral, contextual
    description: str
    conditions: List[dict]
    actions: List[dict]
    confidence: float
    occurrences: int
    last_seen: datetime
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "conditions": self.conditions,
            "actions": self.actions,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "last_seen": self.last_seen.isoformat()
        }


class ExecutiveMemory:
    """
    Memoria esecutiva di JARVIS.
    Registra decisioni, apprende dai risultati e aggiorna strategie.
    """
    
    def __init__(self, storage_path: str = None):
        # Storage
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Decision Memory
        self.decisions: Dict[str, Decision] = {}
        self.decision_history: List[str] = []  # IDs in order
        self.max_decisions = 10000
        
        # Outcome Memory
        self.outcomes: Dict[str, ExecutionOutcome] = {}
        
        # Strategy Profiles
        self.strategies: Dict[str, StrategyProfile] = {}
        self._init_default_strategies()
        
        # User Preferences
        self.user_preferences: Dict[str, Dict[str, UserPreference]] = defaultdict(dict)
        
        # Pattern Recognition
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_candidates: List[dict] = []
        
        # Intent-Action Associations
        self.intent_action_map: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Context Memory
        self.context_memory: List[dict] = []
        self.max_context_memory = 1000
        
        # Learning Parameters
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        self.min_confidence = 0.1
        self.reinforcement_threshold = 0.7
        
        # Analytics
        self.analytics = {
            "total_decisions": 0,
            "total_outcomes": 0,
            "positive_outcomes": 0,
            "negative_outcomes": 0,
            "patterns_discovered": 0,
            "strategy_adaptations": 0
        }
        
        # Load persisted data
        if self.storage_path:
            self._load_from_storage()
    
    def _init_default_strategies(self):
        """Inizializza profili strategia di default"""
        default_strategies = [
            "rule_based",
            "learning_based", 
            "hybrid",
            "conservative",
            "aggressive",
            "adaptive"
        ]
        
        for strategy in default_strategies:
            self.strategies[strategy] = StrategyProfile(name=strategy)
    
    # === Decision Recording ===
    
    def record_decision(self, intent: str, context: dict, 
                        options: List[dict], chosen: dict,
                        reasoning: str, confidence: float,
                        strategy: str, priority: float,
                        risk: str, user_id: str = "default",
                        session_id: str = "") -> Decision:
        """Registra una decisione"""
        decision_id = self._generate_decision_id(intent, context)
        
        decision = Decision(
            id=decision_id,
            timestamp=datetime.now(),
            intent=intent,
            context=context,
            options_considered=options,
            chosen_option=chosen,
            reasoning=reasoning,
            confidence=confidence,
            strategy_used=strategy,
            priority_score=priority,
            risk_level=risk,
            user_id=user_id,
            session_id=session_id
        )
        
        # Store decision
        self.decisions[decision_id] = decision
        self.decision_history.append(decision_id)
        
        # Update strategy profile
        if strategy in self.strategies:
            profile = self.strategies[strategy]
            profile.total_uses += 1
            profile.last_used = datetime.now()
            profile.avg_confidence = (
                (profile.avg_confidence * (profile.total_uses - 1) + confidence)
                / profile.total_uses
            )
        
        # Update intent-action association
        action_type = chosen.get("type", "unknown")
        self.intent_action_map[intent][action_type] += 1
        
        # Store context
        self._store_context(context, intent, decision_id)
        
        # Analytics
        self.analytics["total_decisions"] += 1
        
        # Cleanup old decisions
        self._cleanup_old_decisions()
        
        logger.info(f"Decisione registrata: {decision_id}")
        return decision
    
    def _generate_decision_id(self, intent: str, context: dict) -> str:
        """Genera ID univoco per decisione"""
        timestamp = datetime.now().isoformat()
        content = f"{intent}_{timestamp}_{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _cleanup_old_decisions(self):
        """Rimuove decisioni vecchie se oltre il limite"""
        while len(self.decision_history) > self.max_decisions:
            old_id = self.decision_history.pop(0)
            if old_id in self.decisions:
                del self.decisions[old_id]
            if old_id in self.outcomes:
                del self.outcomes[old_id]
    
    # === Outcome Recording ===
    
    def record_outcome(self, decision_id: str, outcome_type: OutcomeType,
                       execution_time: float, result: dict,
                       error: str = None, user_feedback: str = None,
                       learning_signal: LearningSignal = LearningSignal.NEUTRAL,
                       metrics: dict = None) -> ExecutionOutcome:
        """Registra l'outcome di una decisione"""
        outcome = ExecutionOutcome(
            decision_id=decision_id,
            timestamp=datetime.now(),
            outcome_type=outcome_type,
            execution_time=execution_time,
            result=result,
            error=error,
            user_feedback=user_feedback,
            learning_signal=learning_signal,
            metrics=metrics or {}
        )
        
        self.outcomes[decision_id] = outcome
        self.analytics["total_outcomes"] += 1
        
        # Process learning
        if decision_id in self.decisions:
            decision = self.decisions[decision_id]
            self._process_learning(decision, outcome)
        
        # Update analytics
        if outcome_type == OutcomeType.SUCCESS:
            self.analytics["positive_outcomes"] += 1
        elif outcome_type in [OutcomeType.FAILURE, OutcomeType.ERROR]:
            self.analytics["negative_outcomes"] += 1
        
        logger.info(f"Outcome registrato per decisione {decision_id}: {outcome_type.value}")
        return outcome
    
    def _process_learning(self, decision: Decision, outcome: ExecutionOutcome):
        """Processa apprendimento da outcome"""
        strategy = decision.strategy_used
        
        if strategy not in self.strategies:
            self.strategies[strategy] = StrategyProfile(name=strategy)
        
        profile = self.strategies[strategy]
        
        # Update success/failure counts
        if outcome.outcome_type == OutcomeType.SUCCESS:
            profile.success_count += 1
            self._update_strategy_effectiveness(profile, 1.0)
            
            # Track effective context
            context_key = self._get_context_key(decision.context)
            if context_key not in profile.contexts_effective:
                profile.contexts_effective.append(context_key)
                
        elif outcome.outcome_type in [OutcomeType.FAILURE, OutcomeType.ERROR]:
            profile.failure_count += 1
            self._update_strategy_effectiveness(profile, 0.0)
            
            # Track ineffective context
            context_key = self._get_context_key(decision.context)
            if context_key not in profile.contexts_ineffective:
                profile.contexts_ineffective.append(context_key)
        
        # Update execution time average
        profile.avg_execution_time = (
            (profile.avg_execution_time * (profile.total_uses - 1) + outcome.execution_time)
            / profile.total_uses
        )
        
        # Process learning signal
        self._apply_learning_signal(decision, outcome)
        
        # Check for pattern candidates
        self._check_pattern_candidate(decision, outcome)
    
    def _update_strategy_effectiveness(self, profile: StrategyProfile, reward: float):
        """Aggiorna effectiveness score con reinforcement learning"""
        # Q-learning style update
        profile.effectiveness_score = (
            profile.effectiveness_score + 
            self.learning_rate * (reward - profile.effectiveness_score)
        )
        
        # Clamp to [0, 1]
        profile.effectiveness_score = max(0.0, min(1.0, profile.effectiveness_score))
    
    def _apply_learning_signal(self, decision: Decision, outcome: ExecutionOutcome):
        """Applica segnale di apprendimento"""
        signal = outcome.learning_signal
        
        if signal == LearningSignal.POSITIVE:
            # Rafforza associazione intent-action
            action_type = decision.chosen_option.get("type", "unknown")
            self.intent_action_map[decision.intent][action_type] *= (1 + self.learning_rate)
            
            # Aggiorna preferenze utente
            self._learn_preference(decision.user_id, decision.intent, 
                                   decision.chosen_option, positive=True)
            
        elif signal == LearningSignal.NEGATIVE:
            # Indebolisci associazione
            action_type = decision.chosen_option.get("type", "unknown")
            self.intent_action_map[decision.intent][action_type] *= (1 - self.learning_rate)
            
        elif signal == LearningSignal.CORRECTION:
            # L'utente ha corretto - forte segnale negativo per scelta attuale
            action_type = decision.chosen_option.get("type", "unknown")
            self.intent_action_map[decision.intent][action_type] *= 0.5
            
            # Registra correzione per analisi
            self._record_correction(decision, outcome)
    
    def _get_context_key(self, context: dict) -> str:
        """Genera chiave contestuale"""
        relevant_keys = ["time_of_day", "day_type", "app_context", "urgency"]
        key_parts = []
        for k in relevant_keys:
            if k in context:
                key_parts.append(f"{k}:{context[k]}")
        return "|".join(key_parts) if key_parts else "default"
    
    # === Preference Learning ===
    
    def _learn_preference(self, user_id: str, intent: str, 
                          action: dict, positive: bool):
        """Apprende preferenza utente"""
        pref_key = f"{intent}_{action.get('type', 'unknown')}"
        
        if pref_key in self.user_preferences[user_id]:
            pref = self.user_preferences[user_id][pref_key]
            pref.times_observed += 1
            pref.last_observed = datetime.now()
            
            # Aggiorna confidence
            if positive:
                pref.confidence = min(1.0, pref.confidence + self.learning_rate)
            else:
                pref.confidence = max(self.min_confidence, pref.confidence - self.learning_rate)
        else:
            self.user_preferences[user_id][pref_key] = UserPreference(
                key=pref_key,
                value=action,
                confidence=0.6 if positive else 0.4,
                times_observed=1,
                last_observed=datetime.now()
            )
    
    def get_user_preferences(self, user_id: str, intent: str = None) -> List[dict]:
        """Ottiene preferenze utente"""
        prefs = []
        for key, pref in self.user_preferences.get(user_id, {}).items():
            if intent is None or pref.key.startswith(intent):
                prefs.append(pref.to_dict())
        return sorted(prefs, key=lambda x: x["confidence"], reverse=True)
    
    def get_preferred_action(self, user_id: str, intent: str) -> Optional[dict]:
        """Ottiene azione preferita per intent"""
        best_pref = None
        best_confidence = 0
        
        for key, pref in self.user_preferences.get(user_id, {}).items():
            if key.startswith(intent) and pref.confidence > best_confidence:
                best_confidence = pref.confidence
                best_pref = pref
        
        if best_pref and best_confidence > self.reinforcement_threshold:
            return {
                "action": best_pref.value,
                "confidence": best_confidence,
                "times_observed": best_pref.times_observed
            }
        return None
    
    # === Pattern Recognition ===
    
    def _check_pattern_candidate(self, decision: Decision, outcome: ExecutionOutcome):
        """Verifica se c'è un pattern emergente"""
        # Crea candidato pattern
        candidate = {
            "intent": decision.intent,
            "context_key": self._get_context_key(decision.context),
            "action_type": decision.chosen_option.get("type"),
            "outcome": outcome.outcome_type.value,
            "timestamp": datetime.now(),
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday()
        }
        
        self.pattern_candidates.append(candidate)
        
        # Mantieni solo ultimi 500 candidati
        if len(self.pattern_candidates) > 500:
            self.pattern_candidates = self.pattern_candidates[-500:]
        
        # Cerca pattern ogni 50 candidati
        if len(self.pattern_candidates) % 50 == 0:
            self._discover_patterns()
    
    def _discover_patterns(self):
        """Scopre pattern nei dati"""
        # Pattern temporale - stessa azione alla stessa ora
        temporal_patterns = defaultdict(list)
        for c in self.pattern_candidates:
            key = (c["intent"], c["action_type"], c["hour"])
            temporal_patterns[key].append(c)
        
        for key, candidates in temporal_patterns.items():
            if len(candidates) >= 5:  # Minimo 5 occorrenze
                success_rate = sum(
                    1 for c in candidates if c["outcome"] == "success"
                ) / len(candidates)
                
                if success_rate >= 0.7:  # 70% success rate
                    intent, action_type, hour = key
                    pattern_id = f"temporal_{intent}_{action_type}_{hour}"
                    
                    if pattern_id not in self.patterns:
                        self.patterns[pattern_id] = Pattern(
                            id=pattern_id,
                            name=f"Pattern Temporale: {intent}",
                            pattern_type="temporal",
                            description=f"Alle ore {hour}, {intent} -> {action_type}",
                            conditions=[{"type": "hour", "value": hour}],
                            actions=[{"type": action_type}],
                            confidence=success_rate,
                            occurrences=len(candidates),
                            last_seen=datetime.now()
                        )
                        self.analytics["patterns_discovered"] += 1
                        logger.info(f"Nuovo pattern scoperto: {pattern_id}")
                    else:
                        # Aggiorna pattern esistente
                        pattern = self.patterns[pattern_id]
                        pattern.occurrences = len(candidates)
                        pattern.confidence = success_rate
                        pattern.last_seen = datetime.now()
        
        # Pattern comportamentale - sequenze di azioni
        self._discover_behavioral_patterns()
    
    def _discover_behavioral_patterns(self):
        """Scopre pattern comportamentali (sequenze)"""
        # Analizza sequenze di 3 azioni
        if len(self.decision_history) < 3:
            return
        
        sequences = defaultdict(int)
        for i in range(len(self.decision_history) - 2):
            seq = []
            for j in range(3):
                dec_id = self.decision_history[i + j]
                if dec_id in self.decisions:
                    seq.append(self.decisions[dec_id].intent)
            
            if len(seq) == 3:
                sequences[tuple(seq)] += 1
        
        # Pattern con almeno 3 occorrenze
        for seq, count in sequences.items():
            if count >= 3:
                pattern_id = f"behavioral_{'_'.join(seq)}"
                if pattern_id not in self.patterns:
                    self.patterns[pattern_id] = Pattern(
                        id=pattern_id,
                        name=f"Sequenza: {' -> '.join(seq)}",
                        pattern_type="behavioral",
                        description=f"L'utente spesso fa: {seq[0]} poi {seq[1]} poi {seq[2]}",
                        conditions=[],
                        actions=[],
                        confidence=min(1.0, count / 10),
                        occurrences=count,
                        last_seen=datetime.now()
                    )
                    self.analytics["patterns_discovered"] += 1
    
    def get_relevant_patterns(self, intent: str, context: dict) -> List[dict]:
        """Ottiene pattern rilevanti per intent e contesto"""
        relevant = []
        current_hour = datetime.now().hour
        
        for pattern in self.patterns.values():
            relevance = 0.0
            
            # Check temporal match
            if pattern.pattern_type == "temporal":
                for cond in pattern.conditions:
                    if cond.get("type") == "hour" and cond.get("value") == current_hour:
                        relevance += 0.5
            
            # Check intent match
            if intent in pattern.description.lower():
                relevance += 0.3
            
            if relevance > 0:
                relevant.append({
                    **pattern.to_dict(),
                    "relevance": relevance
                })
        
        return sorted(relevant, key=lambda x: x["relevance"], reverse=True)
    
    # === Strategy Recommendation ===
    
    def recommend_strategy(self, intent: str, context: dict) -> Tuple[str, float]:
        """Raccomanda strategia basata su storia"""
        context_key = self._get_context_key(context)
        best_strategy = "hybrid"
        best_score = 0.0
        
        for name, profile in self.strategies.items():
            score = profile.effectiveness_score
            
            # Bonus se contesto è in lista efficace
            if context_key in profile.contexts_effective:
                score *= 1.2
            
            # Penalità se contesto è in lista inefficace
            if context_key in profile.contexts_ineffective:
                score *= 0.8
            
            # Considera recency
            if profile.last_used:
                days_since = (datetime.now() - profile.last_used).days
                recency_factor = self.decay_factor ** days_since
                score *= recency_factor
            
            if score > best_score:
                best_score = score
                best_strategy = name
        
        return best_strategy, min(1.0, best_score)
    
    def update_strategy(self, strategy_name: str, update_type: StrategyUpdate,
                        params: dict = None):
        """Aggiorna manualmente una strategia"""
        if strategy_name not in self.strategies:
            return
        
        profile = self.strategies[strategy_name]
        
        if update_type == StrategyUpdate.REINFORCE:
            profile.effectiveness_score = min(1.0, profile.effectiveness_score + 0.1)
        elif update_type == StrategyUpdate.WEAKEN:
            profile.effectiveness_score = max(0.0, profile.effectiveness_score - 0.1)
        elif update_type == StrategyUpdate.ADAPT:
            if params:
                profile.adaptations.append({
                    "timestamp": datetime.now().isoformat(),
                    "params": params
                })
        
        self.analytics["strategy_adaptations"] += 1
        logger.info(f"Strategia {strategy_name} aggiornata: {update_type.value}")
    
    # === Context Memory ===
    
    def _store_context(self, context: dict, intent: str, decision_id: str):
        """Memorizza contesto"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "intent": intent,
            "decision_id": decision_id
        }
        
        self.context_memory.append(entry)
        
        if len(self.context_memory) > self.max_context_memory:
            self.context_memory = self.context_memory[-self.max_context_memory:]
    
    def get_similar_contexts(self, context: dict, limit: int = 5) -> List[dict]:
        """Trova contesti simili"""
        scored = []
        
        for entry in self.context_memory:
            similarity = self._calculate_context_similarity(context, entry["context"])
            if similarity > 0.5:  # Solo se abbastanza simile
                scored.append({
                    **entry,
                    "similarity": similarity
                })
        
        return sorted(scored, key=lambda x: x["similarity"], reverse=True)[:limit]
    
    def _calculate_context_similarity(self, ctx1: dict, ctx2: dict) -> float:
        """Calcola similarità tra due contesti"""
        if not ctx1 or not ctx2:
            return 0.0
        
        keys1 = set(ctx1.keys())
        keys2 = set(ctx2.keys())
        common_keys = keys1 & keys2
        
        if not common_keys:
            return 0.0
        
        matches = sum(1 for k in common_keys if ctx1[k] == ctx2[k])
        return matches / len(keys1 | keys2)
    
    # === Correction Recording ===
    
    def _record_correction(self, decision: Decision, outcome: ExecutionOutcome):
        """Registra una correzione utente"""
        correction = {
            "decision_id": decision.id,
            "original_action": decision.chosen_option,
            "timestamp": datetime.now().isoformat(),
            "context": decision.context,
            "intent": decision.intent,
            "feedback": outcome.user_feedback
        }
        
        # Store for analysis
        if not hasattr(self, 'corrections'):
            self.corrections = []
        self.corrections.append(correction)
    
    # === Analytics & Insights ===
    
    def get_insights(self) -> dict:
        """Genera insights dall'analisi dei dati"""
        insights = {
            "summary": self.analytics.copy(),
            "strategy_rankings": [],
            "top_patterns": [],
            "improvement_suggestions": []
        }
        
        # Ranking strategie
        rankings = []
        for name, profile in self.strategies.items():
            rankings.append({
                "name": name,
                "effectiveness": profile.effectiveness_score,
                "success_rate": profile.success_rate,
                "total_uses": profile.total_uses
            })
        insights["strategy_rankings"] = sorted(
            rankings, key=lambda x: x["effectiveness"], reverse=True
        )
        
        # Top patterns
        patterns_list = [p.to_dict() for p in self.patterns.values()]
        insights["top_patterns"] = sorted(
            patterns_list, key=lambda x: x["confidence"], reverse=True
        )[:5]
        
        # Suggerimenti
        if self.analytics["total_outcomes"] > 0:
            success_rate = self.analytics["positive_outcomes"] / self.analytics["total_outcomes"]
            
            if success_rate < 0.7:
                insights["improvement_suggestions"].append({
                    "type": "low_success_rate",
                    "message": f"Success rate {success_rate:.1%} è sotto il 70%. Considera strategie più conservative.",
                    "priority": "high"
                })
            
            # Check underperforming strategies
            for name, profile in self.strategies.items():
                if profile.total_uses > 10 and profile.success_rate < 0.5:
                    insights["improvement_suggestions"].append({
                        "type": "weak_strategy",
                        "message": f"Strategia '{name}' ha success rate basso ({profile.success_rate:.1%}). Considera di ridurne l'uso.",
                        "priority": "medium"
                    })
        
        return insights
    
    def get_decision_history(self, limit: int = 50, 
                             user_id: str = None,
                             intent: str = None) -> List[dict]:
        """Ottiene history delle decisioni"""
        history = []
        
        for dec_id in reversed(self.decision_history[-limit:]):
            if dec_id in self.decisions:
                decision = self.decisions[dec_id]
                
                # Filter by user_id
                if user_id and decision.user_id != user_id:
                    continue
                
                # Filter by intent
                if intent and decision.intent != intent:
                    continue
                
                entry = decision.to_dict()
                
                # Add outcome if available
                if dec_id in self.outcomes:
                    entry["outcome"] = self.outcomes[dec_id].to_dict()
                
                history.append(entry)
        
        return history
    
    # === Persistence ===
    
    def _load_from_storage(self):
        """Carica dati da storage"""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            memory_file = self.storage_path / "executive_memory.json"
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore strategies
                for name, profile_data in data.get("strategies", {}).items():
                    self.strategies[name] = StrategyProfile(
                        name=name,
                        success_count=profile_data.get("success_count", 0),
                        failure_count=profile_data.get("failure_count", 0),
                        total_uses=profile_data.get("total_uses", 0),
                        avg_confidence=profile_data.get("avg_confidence", 0.5),
                        effectiveness_score=profile_data.get("effectiveness_score", 0.5)
                    )
                
                # Restore analytics
                self.analytics = data.get("analytics", self.analytics)
                
                logger.info("Executive memory caricata da storage")
                
        except Exception as e:
            logger.error(f"Errore caricamento memory: {e}")
    
    async def save_to_storage(self):
        """Salva dati su storage"""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                "strategies": {
                    name: profile.to_dict() 
                    for name, profile in self.strategies.items()
                },
                "patterns": {
                    pid: p.to_dict() for pid, p in self.patterns.items()
                },
                "analytics": self.analytics,
                "saved_at": datetime.now().isoformat()
            }
            
            memory_file = self.storage_path / "executive_memory.json"
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info("Executive memory salvata")
            
        except Exception as e:
            logger.error(f"Errore salvataggio memory: {e}")
    
    # === Status ===
    
    def get_status(self) -> dict:
        """Stato della memoria esecutiva"""
        return {
            "total_decisions": len(self.decisions),
            "total_outcomes": len(self.outcomes),
            "strategies_tracked": len(self.strategies),
            "patterns_discovered": len(self.patterns),
            "users_with_preferences": len(self.user_preferences),
            "context_memory_size": len(self.context_memory),
            "analytics": self.analytics
        }
