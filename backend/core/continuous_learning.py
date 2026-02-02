"""
🔄 Continuous Learning Adapter - Sistema di Apprendimento Continuo

Coordina l'apprendimento tra tutti i sistemi di memoria:
- Episodic Memory (eventi e scenari)
- Decision Memory (decisioni e risultati)
- Memory System (STM/LTM, pattern, strategie)

Funzionalità:
- Consolidamento memoria automatico
- Pattern recognition cross-system
- Adattamento comportamentale dinamico
- Feedback loop continuo
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import json
import asyncio
import logging
from pathlib import Path


# Logger
learning_logger = logging.getLogger("continuous_learning")
learning_logger.setLevel(logging.DEBUG)


# === ENUMS ===

class LearningMode(Enum):
    """Modalità di apprendimento"""
    PASSIVE = "passive"         # Solo osservazione
    ACTIVE = "active"           # Apprendimento normale
    AGGRESSIVE = "aggressive"   # Apprendimento accelerato
    CONSERVATIVE = "conservative"  # Apprendimento cauto


class LearningEvent(Enum):
    """Tipo di evento di apprendimento"""
    PATTERN_DETECTED = "pattern_detected"
    INSIGHT_GENERATED = "insight_generated"
    ADAPTATION_SUGGESTED = "adaptation_suggested"
    CONSOLIDATION_DONE = "consolidation_done"
    FEEDBACK_PROCESSED = "feedback_processed"
    STRATEGY_UPDATED = "strategy_updated"


class TriggerType(Enum):
    """Tipo di trigger per apprendimento"""
    TIME_BASED = "time_based"       # Periodico
    EVENT_BASED = "event_based"     # Su evento
    THRESHOLD_BASED = "threshold"   # Su soglia
    MANUAL = "manual"               # Manuale


# === DATA CLASSES ===

@dataclass
class LearningMetrics:
    """Metriche di apprendimento"""
    patterns_discovered: int = 0
    insights_generated: int = 0
    adaptations_applied: int = 0
    consolidations_performed: int = 0
    feedback_processed: int = 0
    strategies_updated: int = 0
    
    # Performance
    accuracy_improvement: float = 0.0
    efficiency_improvement: float = 0.0
    
    # Timing
    last_learning_cycle: Optional[datetime] = None
    total_learning_time_ms: int = 0
    
    def to_dict(self) -> dict:
        return {
            "patterns_discovered": self.patterns_discovered,
            "insights_generated": self.insights_generated,
            "adaptations_applied": self.adaptations_applied,
            "consolidations_performed": self.consolidations_performed,
            "feedback_processed": self.feedback_processed,
            "strategies_updated": self.strategies_updated,
            "accuracy_improvement": round(self.accuracy_improvement, 4),
            "efficiency_improvement": round(self.efficiency_improvement, 4),
            "last_learning_cycle": self.last_learning_cycle.isoformat() if self.last_learning_cycle else None,
            "total_learning_time_ms": self.total_learning_time_ms
        }


@dataclass
class LearningCycle:
    """Ciclo di apprendimento"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Trigger
    trigger: TriggerType = TriggerType.TIME_BASED
    trigger_reason: str = ""
    
    # Input
    events_processed: int = 0
    decisions_analyzed: int = 0
    feedback_items: int = 0
    
    # Output
    patterns_found: int = 0
    insights_created: int = 0
    adaptations_suggested: int = 0
    
    # Results
    success: bool = False
    error: Optional[str] = None
    
    def complete(self, success: bool = True, error: str = None):
        self.completed_at = datetime.now()
        self.success = success
        self.error = error
    
    def duration_ms(self) -> int:
        if not self.completed_at:
            return 0
        return int((self.completed_at - self.started_at).total_seconds() * 1000)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms(),
            "trigger": self.trigger.value,
            "trigger_reason": self.trigger_reason,
            "events_processed": self.events_processed,
            "decisions_analyzed": self.decisions_analyzed,
            "feedback_items": self.feedback_items,
            "patterns_found": self.patterns_found,
            "insights_created": self.insights_created,
            "adaptations_suggested": self.adaptations_suggested,
            "success": self.success,
            "error": self.error
        }


@dataclass
class CrossSystemPattern:
    """Pattern rilevato cross-system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Pattern
    pattern_type: str = ""  # "correlation", "sequence", "causal"
    description: str = ""
    
    # Elementi coinvolti
    episodic_events: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    memory_keys: List[str] = field(default_factory=list)
    
    # Statistiche
    occurrences: int = 0
    confidence: float = 0.5
    
    # Timing
    discovered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "occurrences": self.occurrences,
            "confidence": round(self.confidence, 3),
            "discovered_at": self.discovered_at.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }


# === CONTINUOUS LEARNING ADAPTER ===

class ContinuousLearningAdapter:
    """
    Adattatore per apprendimento continuo.
    Coordina e sincronizza tutti i sistemi di memoria.
    """
    
    def __init__(self,
                 memory_system = None,
                 episodic_memory = None,
                 decision_memory = None,
                 storage_path: Path = None):
        """
        Args:
            memory_system: MemorySystem instance (STM/LTM)
            episodic_memory: EpisodicMemory instance
            decision_memory: DecisionMemory instance
        """
        self.memory_system = memory_system
        self.episodic_memory = episodic_memory
        self.decision_memory = decision_memory
        
        self.storage_path = storage_path or Path("data/learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # State
        self.mode = LearningMode.ACTIVE
        self.is_running = False
        
        # Metrics
        self.metrics = LearningMetrics()
        
        # History
        self.learning_cycles: List[LearningCycle] = []
        self.cross_patterns: Dict[str, CrossSystemPattern] = {}
        
        # Configuration
        self.config = {
            "consolidation_interval": timedelta(minutes=30),
            "pattern_detection_threshold": 3,
            "min_confidence_for_adaptation": 0.7,
            "max_cycles_history": 100,
            "learning_rate": 0.1
        }
        
        # Callbacks
        self.event_callbacks: Dict[LearningEvent, List[Callable]] = defaultdict(list)
        
        # Timing
        self.last_consolidation = datetime.now()
        
        self._load()
        
        learning_logger.info("ContinuousLearningAdapter initialized")
    
    # === Memory Integration ===
    
    def connect_memory_system(self, memory_system):
        """Connette Memory System"""
        self.memory_system = memory_system
        learning_logger.info("Memory System connected")
    
    def connect_episodic_memory(self, episodic_memory):
        """Connette Episodic Memory"""
        self.episodic_memory = episodic_memory
        learning_logger.info("Episodic Memory connected")
    
    def connect_decision_memory(self, decision_memory):
        """Connette Decision Memory"""
        self.decision_memory = decision_memory
        learning_logger.info("Decision Memory connected")
    
    # === Learning Cycles ===
    
    def start_learning_cycle(self,
                            trigger: TriggerType = TriggerType.MANUAL,
                            reason: str = "") -> LearningCycle:
        """Avvia un ciclo di apprendimento"""
        cycle = LearningCycle(
            trigger=trigger,
            trigger_reason=reason
        )
        
        try:
            # 1. Raccolta dati
            cycle.events_processed = self._gather_episodic_data()
            cycle.decisions_analyzed = self._gather_decision_data()
            cycle.feedback_items = self._gather_feedback()
            
            # 2. Analisi pattern
            patterns = self._detect_patterns()
            cycle.patterns_found = len(patterns)
            self.metrics.patterns_discovered += len(patterns)
            
            # 3. Generazione insights
            insights = self._generate_insights(patterns)
            cycle.insights_created = len(insights)
            self.metrics.insights_generated += len(insights)
            
            # 4. Suggerisci adattamenti
            adaptations = self._suggest_adaptations(patterns, insights)
            cycle.adaptations_suggested = len(adaptations)
            
            # 5. Consolidamento memoria
            if self._should_consolidate():
                self._consolidate_memory()
                self.metrics.consolidations_performed += 1
            
            cycle.complete(success=True)
            
        except Exception as e:
            cycle.complete(success=False, error=str(e))
            learning_logger.error(f"Learning cycle failed: {e}")
        
        # Update metrics
        self.metrics.last_learning_cycle = cycle.started_at
        self.metrics.total_learning_time_ms += cycle.duration_ms()
        
        # Save cycle
        self.learning_cycles.append(cycle)
        if len(self.learning_cycles) > self.config["max_cycles_history"]:
            self.learning_cycles = self.learning_cycles[-self.config["max_cycles_history"]:]
        
        self._emit_event(LearningEvent.CONSOLIDATION_DONE, {"cycle": cycle.to_dict()})
        
        self._save()
        
        return cycle
    
    def _gather_episodic_data(self) -> int:
        """Raccoglie dati episodici"""
        if not self.episodic_memory:
            return 0
        
        # Prendi eventi recenti
        recent = self.episodic_memory.get_recent_events(
            since=datetime.now() - timedelta(hours=1)
        )
        
        return len(recent)
    
    def _gather_decision_data(self) -> int:
        """Raccoglie dati decisioni"""
        if not self.decision_memory:
            return 0
        
        decisions = self.decision_memory.get_recent_decisions(n=50)
        return len(decisions)
    
    def _gather_feedback(self) -> int:
        """Raccoglie feedback"""
        count = 0
        
        if self.decision_memory:
            for d in self.decision_memory.get_recent_decisions(n=50):
                count += len(d.feedback)
        
        return count
    
    def _detect_patterns(self) -> List[CrossSystemPattern]:
        """Rileva pattern cross-system"""
        patterns = []
        
        if self.mode == LearningMode.PASSIVE:
            return patterns
        
        # Pattern da episodic + decision
        if self.episodic_memory and self.decision_memory:
            patterns.extend(self._detect_episodic_decision_patterns())
        
        # Pattern da memory system
        if self.memory_system:
            patterns.extend(self._detect_memory_patterns())
        
        return patterns
    
    def _detect_episodic_decision_patterns(self) -> List[CrossSystemPattern]:
        """Rileva pattern tra eventi episodici e decisioni"""
        patterns = []
        
        # Cerca correlazioni tra eventi e decisioni
        recent_events = self.episodic_memory.get_recent_events(
            since=datetime.now() - timedelta(hours=6)
        )
        
        recent_decisions = self.decision_memory.get_recent_decisions(n=100)
        
        # Raggruppa per timestamp vicini
        event_decisions = defaultdict(list)
        
        for event in recent_events:
            # Trova decisioni vicine temporalmente (entro 1 minuto)
            for decision in recent_decisions:
                time_diff = abs((event.timestamp - decision.decided_at).total_seconds())
                if time_diff < 60:
                    event_decisions[event.id].append(decision)
        
        # Cerca sequenze ripetute
        sequences = defaultdict(list)
        
        for event_id, decisions in event_decisions.items():
            event = self.episodic_memory.events.get(event_id)
            if not event:
                continue
            
            for decision in decisions:
                key = f"{event.event_type.value}:{decision.decision_type}"
                sequences[key].append({
                    "event_id": event_id,
                    "decision_id": decision.id,
                    "outcome": decision.outcome.value
                })
        
        # Pattern se sequenza ripetuta 3+ volte
        threshold = self.config["pattern_detection_threshold"]
        
        for key, occurrences in sequences.items():
            if len(occurrences) >= threshold:
                # Calcola success rate
                success_count = sum(
                    1 for o in occurrences 
                    if o["outcome"] == "success"
                )
                confidence = success_count / len(occurrences)
                
                pattern = CrossSystemPattern(
                    pattern_type="correlation",
                    description=f"Pattern: {key} (seen {len(occurrences)} times)",
                    episodic_events=[o["event_id"] for o in occurrences[:10]],
                    decisions=[o["decision_id"] for o in occurrences[:10]],
                    occurrences=len(occurrences),
                    confidence=confidence
                )
                
                # Verifica se pattern già esistente
                existing = self._find_similar_pattern(pattern)
                if existing:
                    existing.occurrences += len(occurrences)
                    existing.confidence = (existing.confidence * 0.7 + confidence * 0.3)
                    existing.last_seen = datetime.now()
                else:
                    self.cross_patterns[pattern.id] = pattern
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_memory_patterns(self) -> List[CrossSystemPattern]:
        """Rileva pattern dal memory system"""
        patterns = []
        
        if not self.memory_system:
            return patterns
        
        # Usa i pattern del LTM
        if hasattr(self.memory_system, 'ltm'):
            ltm_patterns = self.memory_system.ltm.get_patterns(min_confidence=0.6)
            
            for ltm_pattern in ltm_patterns:
                pattern = CrossSystemPattern(
                    pattern_type="memory",
                    description=ltm_pattern.description,
                    memory_keys=list(ltm_pattern.context.keys())[:5],
                    occurrences=ltm_pattern.occurrences,
                    confidence=ltm_pattern.confidence
                )
                
                existing = self._find_similar_pattern(pattern)
                if not existing:
                    self.cross_patterns[pattern.id] = pattern
                    patterns.append(pattern)
        
        return patterns
    
    def _find_similar_pattern(self, pattern: CrossSystemPattern) -> Optional[CrossSystemPattern]:
        """Trova pattern simile esistente"""
        for existing in self.cross_patterns.values():
            if existing.pattern_type != pattern.pattern_type:
                continue
            
            # Confronta elementi
            if (set(existing.episodic_events) & set(pattern.episodic_events) and
                set(existing.decisions) & set(pattern.decisions)):
                return existing
            
            if (set(existing.memory_keys) & set(pattern.memory_keys) and
                len(set(existing.memory_keys) & set(pattern.memory_keys)) > 2):
                return existing
        
        return None
    
    def _generate_insights(self, patterns: List[CrossSystemPattern]) -> List[Dict]:
        """Genera insights dai pattern"""
        insights = []
        
        for pattern in patterns:
            if pattern.confidence < self.config["min_confidence_for_adaptation"]:
                continue
            
            insight = {
                "pattern_id": pattern.id,
                "type": pattern.pattern_type,
                "description": pattern.description,
                "confidence": pattern.confidence,
                "recommendation": self._insight_recommendation(pattern)
            }
            
            insights.append(insight)
            
            self._emit_event(LearningEvent.INSIGHT_GENERATED, insight)
        
        return insights
    
    def _insight_recommendation(self, pattern: CrossSystemPattern) -> str:
        """Genera raccomandazione per pattern"""
        if pattern.confidence > 0.8:
            return f"High confidence pattern detected. Consider automating: {pattern.description}"
        elif pattern.confidence > 0.6:
            return f"Moderate pattern detected. Monitor for confirmation: {pattern.description}"
        else:
            return f"Weak pattern. Continue gathering data: {pattern.description}"
    
    def _suggest_adaptations(self,
                            patterns: List[CrossSystemPattern],
                            insights: List[Dict]) -> List[Dict]:
        """Suggerisce adattamenti basati su pattern e insights"""
        adaptations = []
        
        if self.mode in [LearningMode.PASSIVE, LearningMode.CONSERVATIVE]:
            return adaptations
        
        for pattern in patterns:
            if pattern.confidence < self.config["min_confidence_for_adaptation"]:
                continue
            
            # Suggerisci adattamento
            if pattern.confidence > 0.8 and pattern.occurrences > 10:
                adaptation = {
                    "type": "parameter_optimization",
                    "target": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "reason": f"Pattern '{pattern.description}' highly reliable",
                    "action": "Consider caching/automating this pattern"
                }
                
                adaptations.append(adaptation)
                self._emit_event(LearningEvent.ADAPTATION_SUGGESTED, adaptation)
        
        return adaptations
    
    def _should_consolidate(self) -> bool:
        """Verifica se fare consolidamento"""
        time_since = datetime.now() - self.last_consolidation
        return time_since >= self.config["consolidation_interval"]
    
    def _consolidate_memory(self):
        """Consolida memoria tra sistemi"""
        # Consolida STM -> LTM nel memory system
        if self.memory_system:
            if hasattr(self.memory_system, 'consolidate'):
                self.memory_system.consolidate()
        
        # Salva episodic memory
        if self.episodic_memory:
            if hasattr(self.episodic_memory, '_save'):
                self.episodic_memory._save()
        
        # Salva decision memory
        if self.decision_memory:
            if hasattr(self.decision_memory, '_save'):
                self.decision_memory._save()
        
        self.last_consolidation = datetime.now()
        
        learning_logger.info("Memory consolidation completed")
    
    # === Feedback Processing ===
    
    def process_feedback(self,
                        feedback_type: str,
                        target_id: str,
                        positive: bool,
                        details: Dict = None):
        """Processa feedback da utente/sistema"""
        self.metrics.feedback_processed += 1
        
        # Propaga a decision memory
        if self.decision_memory:
            from .decision_memory import FeedbackType
            fb_type = FeedbackType.POSITIVE if positive else FeedbackType.NEGATIVE
            self.decision_memory.add_decision_feedback(target_id, fb_type, details=details)
        
        # Aggiorna pattern correlati
        for pattern in self.cross_patterns.values():
            if target_id in pattern.decisions or target_id in pattern.episodic_events:
                # Aggiusta confidence
                adjustment = self.config["learning_rate"] * (1 if positive else -1)
                pattern.confidence = max(0, min(1, pattern.confidence + adjustment))
        
        self._emit_event(LearningEvent.FEEDBACK_PROCESSED, {
            "target_id": target_id,
            "positive": positive
        })
    
    # === Mode Control ===
    
    def set_mode(self, mode: LearningMode):
        """Imposta modalità apprendimento"""
        old_mode = self.mode
        self.mode = mode
        
        learning_logger.info(f"Learning mode changed: {old_mode.value} -> {mode.value}")
    
    def pause_learning(self):
        """Pausa apprendimento"""
        self.set_mode(LearningMode.PASSIVE)
    
    def resume_learning(self):
        """Riprende apprendimento"""
        self.set_mode(LearningMode.ACTIVE)
    
    # === Callbacks ===
    
    def on_event(self, event_type: LearningEvent, callback: Callable):
        """Registra callback per evento"""
        self.event_callbacks[event_type].append(callback)
    
    def _emit_event(self, event_type: LearningEvent, data: Dict):
        """Emetti evento"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                learning_logger.error(f"Callback error: {e}")
    
    # === Query Methods ===
    
    def get_metrics(self) -> Dict:
        """Ritorna metriche apprendimento"""
        return self.metrics.to_dict()
    
    def get_recent_cycles(self, n: int = 10) -> List[Dict]:
        """Ritorna cicli recenti"""
        return [c.to_dict() for c in self.learning_cycles[-n:]]
    
    def get_patterns(self, min_confidence: float = 0.0) -> List[Dict]:
        """Ritorna pattern scoperti"""
        patterns = [
            p.to_dict() for p in self.cross_patterns.values()
            if p.confidence >= min_confidence
        ]
        patterns.sort(key=lambda p: p["confidence"], reverse=True)
        return patterns
    
    def get_learning_summary(self) -> Dict:
        """Ritorna summary apprendimento"""
        return {
            "mode": self.mode.value,
            "is_running": self.is_running,
            "metrics": self.metrics.to_dict(),
            "total_patterns": len(self.cross_patterns),
            "high_confidence_patterns": sum(
                1 for p in self.cross_patterns.values()
                if p.confidence > 0.7
            ),
            "cycles_completed": len(self.learning_cycles),
            "last_consolidation": self.last_consolidation.isoformat(),
            "connected_systems": {
                "memory_system": self.memory_system is not None,
                "episodic_memory": self.episodic_memory is not None,
                "decision_memory": self.decision_memory is not None
            }
        }
    
    # === Persistence ===
    
    def _save(self):
        """Salva stato su disco"""
        try:
            state = {
                "metrics": self.metrics.to_dict(),
                "patterns": {pid: p.to_dict() for pid, p in self.cross_patterns.items()},
                "recent_cycles": [c.to_dict() for c in self.learning_cycles[-50:]],
                "config": {k: str(v) for k, v in self.config.items()},
                "last_consolidation": self.last_consolidation.isoformat()
            }
            
            with open(self.storage_path / "learning_state.json", "w") as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            learning_logger.error(f"Failed to save learning state: {e}")
    
    def _load(self):
        """Carica stato da disco"""
        state_file = self.storage_path / "learning_state.json"
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file) as f:
                state = json.load(f)
            
            # Load metrics
            if "metrics" in state:
                m = state["metrics"]
                self.metrics.patterns_discovered = m.get("patterns_discovered", 0)
                self.metrics.insights_generated = m.get("insights_generated", 0)
                self.metrics.adaptations_applied = m.get("adaptations_applied", 0)
                self.metrics.consolidations_performed = m.get("consolidations_performed", 0)
                self.metrics.feedback_processed = m.get("feedback_processed", 0)
            
            # Load patterns
            if "patterns" in state:
                for pid, pdata in state["patterns"].items():
                    pattern = CrossSystemPattern(
                        id=pid,
                        pattern_type=pdata.get("pattern_type", ""),
                        description=pdata.get("description", ""),
                        occurrences=pdata.get("occurrences", 0),
                        confidence=pdata.get("confidence", 0.5)
                    )
                    self.cross_patterns[pid] = pattern
            
            # Load consolidation time
            if "last_consolidation" in state:
                self.last_consolidation = datetime.fromisoformat(state["last_consolidation"])
                
        except Exception as e:
            learning_logger.error(f"Failed to load learning state: {e}")


# === UNIFIED MEMORY COORDINATOR ===

class UnifiedMemoryCoordinator:
    """
    Coordinatore unificato per tutti i sistemi di memoria.
    Fornisce interfaccia semplificata per operazioni cross-system.
    """
    
    def __init__(self):
        self.memory_system = None
        self.episodic_memory = None
        self.decision_memory = None
        self.learning_adapter = None
        
        self._initialized = False
    
    def initialize(self,
                  memory_system = None,
                  episodic_memory = None,
                  decision_memory = None):
        """Inizializza con i sistemi di memoria"""
        self.memory_system = memory_system
        self.episodic_memory = episodic_memory
        self.decision_memory = decision_memory
        
        # Crea learning adapter
        self.learning_adapter = ContinuousLearningAdapter(
            memory_system=memory_system,
            episodic_memory=episodic_memory,
            decision_memory=decision_memory
        )
        
        self._initialized = True
        
        learning_logger.info("UnifiedMemoryCoordinator initialized")
    
    def record(self, event_type: str, content: Any, context: Dict = None) -> str:
        """Record unificato - registra in tutti i sistemi appropriati"""
        ids = []
        
        # Episodic memory
        if self.episodic_memory:
            from .episodic_memory import EventType
            
            et = EventType.USER_INPUT
            if "action" in event_type.lower():
                et = EventType.SYSTEM_ACTION
            elif "decision" in event_type.lower():
                et = EventType.DECISION
            elif "result" in event_type.lower():
                et = EventType.RESULT
            elif "error" in event_type.lower():
                et = EventType.ERROR
            
            event = self.episodic_memory.record_event(
                event_type=et,
                content=content,
                context=context
            )
            ids.append(f"event:{event.id}")
        
        # Short-term memory
        if self.memory_system:
            key = f"{event_type}_{datetime.now().timestamp()}"
            self.memory_system.remember_short(key, content, context)
            ids.append(f"stm:{key}")
        
        return ",".join(ids)
    
    def remember_decision(self,
                         decision_type: str,
                         decision: Dict,
                         context: Dict,
                         confidence: float = 0.5) -> str:
        """Registra una decisione"""
        if not self.decision_memory:
            return ""
        
        record = self.decision_memory.record_decision(
            decision_type=decision_type,
            decision_made=decision,
            context=context,
            confidence=confidence
        )
        
        return record.id
    
    def resolve_decision(self, decision_id: str, success: bool, details: Dict = None):
        """Risolve una decisione"""
        if not self.decision_memory:
            return
        
        from .decision_memory import DecisionOutcome
        
        outcome = DecisionOutcome.SUCCESS if success else DecisionOutcome.FAILURE
        self.decision_memory.resolve_decision(decision_id, outcome, details)
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Ricerca unificata in tutti i sistemi"""
        results = []
        
        # Search episodic
        if self.episodic_memory:
            events = self.episodic_memory.search_events(query)[:limit//2]
            for event in events:
                results.append({
                    "source": "episodic",
                    "type": event.event_type.value,
                    "content": event.content,
                    "timestamp": event.timestamp.isoformat()
                })
        
        # Search memory system
        if self.memory_system:
            memories = self.memory_system.search(query, limit=limit//2)
            for mem in memories:
                results.append({
                    "source": "memory",
                    "type": "memory",
                    "content": mem,
                    "timestamp": None
                })
        
        return results
    
    def get_applicable_knowledge(self, context: Dict) -> Dict:
        """Recupera conoscenza applicabile al contesto"""
        knowledge = {
            "patterns": [],
            "insights": [],
            "strategies": [],
            "relevant_decisions": []
        }
        
        # Decision insights
        if self.decision_memory:
            insights = self.decision_memory.get_applicable_insights(context)
            knowledge["insights"] = [i.to_dict() for i in insights[:5]]
        
        # Memory patterns
        if self.memory_system and hasattr(self.memory_system, 'ltm'):
            patterns = self.memory_system.ltm.get_patterns(min_confidence=0.6)
            knowledge["patterns"] = [p.to_dict() for p in patterns[:5]]
            
            strategies = self.memory_system.ltm.get_strategies(context)
            knowledge["strategies"] = [s.to_dict() for s in strategies[:3]]
        
        # Learning patterns
        if self.learning_adapter:
            cross_patterns = self.learning_adapter.get_patterns(min_confidence=0.6)
            knowledge["cross_patterns"] = cross_patterns[:5]
        
        return knowledge
    
    def trigger_learning_cycle(self) -> Dict:
        """Forza un ciclo di apprendimento"""
        if not self.learning_adapter:
            return {"error": "Learning adapter not initialized"}
        
        from .continuous_learning import TriggerType
        
        cycle = self.learning_adapter.start_learning_cycle(
            trigger=TriggerType.MANUAL,
            reason="Manual trigger"
        )
        
        return cycle.to_dict()
    
    def get_system_status(self) -> Dict:
        """Status di tutti i sistemi"""
        status = {
            "initialized": self._initialized,
            "systems": {}
        }
        
        if self.memory_system:
            status["systems"]["memory_system"] = self.memory_system.get_stats()
        
        if self.episodic_memory:
            status["systems"]["episodic_memory"] = self.episodic_memory.get_stats()
        
        if self.decision_memory:
            status["systems"]["decision_memory"] = self.decision_memory.get_stats()
        
        if self.learning_adapter:
            status["systems"]["learning"] = self.learning_adapter.get_learning_summary()
        
        return status
