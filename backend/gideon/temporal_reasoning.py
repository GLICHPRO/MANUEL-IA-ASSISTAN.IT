# /backend/gideon/temporal_reasoning.py
"""
🔮 GIDEON 3.0 - Temporal Reasoning
Considera sequenze temporali e impatti futuri nelle decisioni.
NON esegue azioni - fornisce solo analisi temporale e previsioni.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
import math
import logging

logger = logging.getLogger(__name__)


class TimeHorizon(Enum):
    """Orizzonti temporali"""
    IMMEDIATE = "immediate"       # Secondi
    SHORT_TERM = "short_term"     # Minuti-ore
    MEDIUM_TERM = "medium_term"   # Ore-giorni
    LONG_TERM = "long_term"       # Giorni-settimane
    STRATEGIC = "strategic"       # Settimane-mesi


class TemporalRelation(Enum):
    """Relazioni temporali tra eventi"""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    MEETS = "meets"              # Fine di A = inizio di B
    STARTS = "starts"            # A inizia quando B inizia
    FINISHES = "finishes"        # A finisce quando B finisce
    EQUALS = "equals"
    CONTAINS = "contains"


class ImpactType(Enum):
    """Tipi di impatto temporale"""
    TRANSIENT = "transient"       # Temporaneo
    PERSISTENT = "persistent"     # Persistente
    CUMULATIVE = "cumulative"     # Si accumula
    DECAYING = "decaying"         # Decade nel tempo
    DELAYED = "delayed"           # Ritardato
    CASCADING = "cascading"       # A cascata


@dataclass
class TemporalEvent:
    """Evento con dimensione temporale"""
    id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    # Temporal properties
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None  # daily, weekly, etc.
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    # Impact
    impact_type: ImpactType = ImpactType.TRANSIENT
    impact_magnitude: float = 0.5
    impact_decay_rate: float = 0.1  # Per hour
    
    # Metadata
    priority: int = 5
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.end_time and not self.duration:
            self.duration = self.end_time - self.start_time
        elif self.duration and not self.end_time:
            self.end_time = self.start_time + self.duration
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "is_recurring": self.is_recurring,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "impact_type": self.impact_type.value,
            "impact_magnitude": self.impact_magnitude,
            "priority": self.priority
        }


@dataclass
class TemporalImpact:
    """Impatto di un'azione nel tempo"""
    action_id: str
    horizon: TimeHorizon
    
    # Impact values over time
    immediate_impact: float = 0.0
    short_term_impact: float = 0.0
    medium_term_impact: float = 0.0
    long_term_impact: float = 0.0
    
    # Characteristics
    impact_type: ImpactType = ImpactType.TRANSIENT
    peak_time: Optional[timedelta] = None
    decay_half_life: Optional[timedelta] = None
    
    # Risks
    reversal_possibility: float = 0.8
    cascading_risk: float = 0.2
    
    # Side effects
    side_effects: List[Dict] = field(default_factory=list)
    
    def total_impact(self) -> float:
        """Calcola impatto totale pesato"""
        weights = {
            "immediate": 0.4,
            "short": 0.3,
            "medium": 0.2,
            "long": 0.1
        }
        return (
            weights["immediate"] * self.immediate_impact +
            weights["short"] * self.short_term_impact +
            weights["medium"] * self.medium_term_impact +
            weights["long"] * self.long_term_impact
        )
    
    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "horizon": self.horizon.value,
            "immediate_impact": round(self.immediate_impact, 3),
            "short_term_impact": round(self.short_term_impact, 3),
            "medium_term_impact": round(self.medium_term_impact, 3),
            "long_term_impact": round(self.long_term_impact, 3),
            "total_impact": round(self.total_impact(), 3),
            "impact_type": self.impact_type.value,
            "reversal_possibility": round(self.reversal_possibility, 3),
            "cascading_risk": round(self.cascading_risk, 3),
            "side_effects": self.side_effects
        }


@dataclass
class TemporalSequence:
    """Sequenza di eventi temporali"""
    id: str
    name: str
    events: List[TemporalEvent] = field(default_factory=list)
    
    # Sequence properties
    total_duration: Optional[timedelta] = None
    critical_path: List[str] = field(default_factory=list)
    
    # Analysis
    bottlenecks: List[str] = field(default_factory=list)
    parallel_opportunities: List[Tuple[str, str]] = field(default_factory=list)
    
    def calculate_duration(self) -> timedelta:
        """Calcola durata totale considerando dipendenze"""
        if not self.events:
            return timedelta(0)
        
        start = min(e.start_time for e in self.events)
        end = max(e.end_time for e in self.events if e.end_time)
        return end - start if end else timedelta(0)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "events": [e.to_dict() for e in self.events],
            "total_duration_seconds": self.total_duration.total_seconds() if self.total_duration else None,
            "critical_path": self.critical_path,
            "bottlenecks": self.bottlenecks,
            "parallel_opportunities": self.parallel_opportunities
        }


@dataclass
class FutureProjection:
    """Proiezione futura basata su azioni attuali"""
    scenario_id: str
    projection_horizon: TimeHorizon
    
    # Projected states
    projected_states: Dict[str, Any] = field(default_factory=dict)
    confidence_by_time: Dict[str, float] = field(default_factory=dict)
    
    # Key milestones
    milestones: List[Dict] = field(default_factory=list)
    
    # Risk windows
    risk_windows: List[Dict] = field(default_factory=list)
    
    # Opportunities
    opportunity_windows: List[Dict] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "horizon": self.projection_horizon.value,
            "projected_states": self.projected_states,
            "confidence_by_time": self.confidence_by_time,
            "milestones": self.milestones,
            "risk_windows": self.risk_windows,
            "opportunity_windows": self.opportunity_windows
        }


class TemporalReasoning:
    """
    Motore di ragionamento temporale per Gideon.
    Analizza sequenze, dipendenze e impatti futuri.
    """
    
    def __init__(self):
        # Event storage
        self.events: Dict[str, TemporalEvent] = {}
        self.sequences: Dict[str, TemporalSequence] = {}
        
        # Impact models
        self.impact_models: Dict[str, Callable] = {}
        self._register_default_models()
        
        # Time horizons configuration
        self.horizon_durations = {
            TimeHorizon.IMMEDIATE: timedelta(seconds=30),
            TimeHorizon.SHORT_TERM: timedelta(hours=1),
            TimeHorizon.MEDIUM_TERM: timedelta(days=1),
            TimeHorizon.LONG_TERM: timedelta(weeks=1),
            TimeHorizon.STRATEGIC: timedelta(days=30)
        }
        
        # Counters
        self._event_counter = 0
        self._sequence_counter = 0
    
    def _register_default_models(self):
        """Registra modelli di impatto predefiniti"""
        # Exponential decay
        self.impact_models["exponential_decay"] = lambda t, mag, rate: mag * math.exp(-rate * t)
        
        # Linear decay
        self.impact_models["linear_decay"] = lambda t, mag, rate: max(0, mag - rate * t)
        
        # Delayed impact
        self.impact_models["delayed"] = lambda t, mag, delay: mag if t >= delay else 0
        
        # Cumulative
        self.impact_models["cumulative"] = lambda t, mag, rate: mag * (1 + rate * t)
        
        # Bell curve (peak then decay)
        self.impact_models["bell_curve"] = lambda t, mag, peak: mag * math.exp(-((t - peak) ** 2) / (2 * peak ** 2))
    
    # === Event Management ===
    
    def create_event(self, name: str, start_time: datetime,
                     duration: timedelta = None,
                     end_time: datetime = None,
                     **kwargs) -> TemporalEvent:
        """Crea un nuovo evento temporale"""
        self._event_counter += 1
        event_id = f"evt_{self._event_counter}"
        
        event = TemporalEvent(
            id=event_id,
            name=name,
            start_time=start_time,
            end_time=end_time,
            duration=duration or timedelta(minutes=5),
            impact_type=ImpactType(kwargs.get("impact_type", "transient")),
            impact_magnitude=kwargs.get("impact_magnitude", 0.5),
            depends_on=kwargs.get("depends_on", []),
            blocks=kwargs.get("blocks", []),
            priority=kwargs.get("priority", 5),
            is_recurring=kwargs.get("is_recurring", False),
            recurrence_pattern=kwargs.get("recurrence_pattern")
        )
        
        self.events[event_id] = event
        return event
    
    def add_dependency(self, event_id: str, depends_on_id: str) -> bool:
        """Aggiunge dipendenza tra eventi"""
        if event_id not in self.events or depends_on_id not in self.events:
            return False
        
        self.events[event_id].depends_on.append(depends_on_id)
        self.events[depends_on_id].blocks.append(event_id)
        return True
    
    # === Temporal Relations ===
    
    def get_relation(self, event_a: TemporalEvent,
                     event_b: TemporalEvent) -> TemporalRelation:
        """Determina relazione temporale tra due eventi"""
        a_start, a_end = event_a.start_time, event_a.end_time
        b_start, b_end = event_b.start_time, event_b.end_time
        
        if not a_end:
            a_end = a_start + (event_a.duration or timedelta(0))
        if not b_end:
            b_end = b_start + (event_b.duration or timedelta(0))
        
        # Check relations (Allen's interval algebra)
        if a_end < b_start:
            return TemporalRelation.BEFORE
        elif a_start > b_end:
            return TemporalRelation.AFTER
        elif a_end == b_start:
            return TemporalRelation.MEETS
        elif a_start == b_start and a_end == b_end:
            return TemporalRelation.EQUALS
        elif a_start == b_start:
            return TemporalRelation.STARTS
        elif a_end == b_end:
            return TemporalRelation.FINISHES
        elif a_start <= b_start and a_end >= b_end:
            return TemporalRelation.CONTAINS
        elif a_start < b_end and a_end > b_start:
            return TemporalRelation.OVERLAPS
        else:
            return TemporalRelation.DURING
    
    def find_conflicts(self, events: List[TemporalEvent] = None) -> List[Tuple[str, str, str]]:
        """Trova conflitti temporali tra eventi"""
        if events is None:
            events = list(self.events.values())
        
        conflicts = []
        
        for i, event_a in enumerate(events):
            for event_b in events[i+1:]:
                relation = self.get_relation(event_a, event_b)
                
                # Check for conflicts
                if relation in [TemporalRelation.OVERLAPS, TemporalRelation.DURING,
                               TemporalRelation.CONTAINS, TemporalRelation.EQUALS]:
                    # Check if they're incompatible
                    if event_a.id in event_b.depends_on or event_b.id in event_a.depends_on:
                        conflicts.append((event_a.id, event_b.id, f"dependency_overlap"))
                    else:
                        conflicts.append((event_a.id, event_b.id, relation.value))
        
        return conflicts
    
    # === Sequence Analysis ===
    
    def create_sequence(self, name: str, event_ids: List[str]) -> TemporalSequence:
        """Crea sequenza da eventi esistenti"""
        self._sequence_counter += 1
        seq_id = f"seq_{self._sequence_counter}"
        
        events = [self.events[eid] for eid in event_ids if eid in self.events]
        
        sequence = TemporalSequence(
            id=seq_id,
            name=name,
            events=sorted(events, key=lambda e: e.start_time)
        )
        
        # Calculate properties
        sequence.total_duration = sequence.calculate_duration()
        sequence.critical_path = self._find_critical_path(events)
        sequence.bottlenecks = self._find_bottlenecks(events)
        sequence.parallel_opportunities = self._find_parallel_opportunities(events)
        
        self.sequences[seq_id] = sequence
        return sequence
    
    def _find_critical_path(self, events: List[TemporalEvent]) -> List[str]:
        """Trova il percorso critico nella sequenza"""
        if not events:
            return []
        
        # Build dependency graph
        graph = {e.id: e.depends_on for e in events}
        durations = {e.id: e.duration.total_seconds() if e.duration else 0 for e in events}
        
        # Find longest path (simplified)
        critical_path = []
        current_time = 0
        
        for event in sorted(events, key=lambda e: e.start_time):
            if not event.depends_on or all(d in critical_path for d in event.depends_on):
                critical_path.append(event.id)
        
        return critical_path
    
    def _find_bottlenecks(self, events: List[TemporalEvent]) -> List[str]:
        """Identifica bottleneck nella sequenza"""
        bottlenecks = []
        
        for event in events:
            # High number of dependents = bottleneck
            blocked_count = len(event.blocks)
            if blocked_count >= 2:
                bottlenecks.append(event.id)
        
        return bottlenecks
    
    def _find_parallel_opportunities(self, events: List[TemporalEvent]) -> List[Tuple[str, str]]:
        """Trova opportunità di parallelizzazione"""
        opportunities = []
        
        for i, event_a in enumerate(events):
            for event_b in events[i+1:]:
                # Can run in parallel if no dependencies
                if (event_a.id not in event_b.depends_on and
                    event_b.id not in event_a.depends_on and
                    not set(event_a.depends_on) & set(event_b.depends_on)):
                    opportunities.append((event_a.id, event_b.id))
        
        return opportunities
    
    # === Impact Analysis ===
    
    def analyze_impact(self, action: Dict, context: Dict = None) -> TemporalImpact:
        """
        Analizza l'impatto temporale di un'azione.
        
        Args:
            action: Descrizione dell'azione
            context: Contesto attuale
        
        Returns:
            TemporalImpact con impatti su diversi orizzonti
        """
        action_id = action.get("id", "unknown")
        action_type = action.get("type", "generic")
        
        # Determine impact type
        impact_type = self._determine_impact_type(action)
        
        # Calculate impacts per horizon
        immediate = self._calculate_immediate_impact(action, context)
        short_term = self._calculate_short_term_impact(action, context, immediate)
        medium_term = self._calculate_medium_term_impact(action, context, short_term)
        long_term = self._calculate_long_term_impact(action, context, medium_term)
        
        # Determine reversal possibility
        reversal = self._calculate_reversal_possibility(action)
        
        # Calculate cascading risk
        cascading = self._calculate_cascading_risk(action, context)
        
        # Identify side effects
        side_effects = self._identify_side_effects(action, context)
        
        return TemporalImpact(
            action_id=action_id,
            horizon=self._determine_primary_horizon(action),
            immediate_impact=immediate,
            short_term_impact=short_term,
            medium_term_impact=medium_term,
            long_term_impact=long_term,
            impact_type=impact_type,
            reversal_possibility=reversal,
            cascading_risk=cascading,
            side_effects=side_effects
        )
    
    def _determine_impact_type(self, action: Dict) -> ImpactType:
        """Determina tipo di impatto"""
        action_type = action.get("type", "").lower()
        
        if "delete" in action_type or "remove" in action_type:
            return ImpactType.PERSISTENT
        elif "config" in action_type or "setting" in action_type:
            return ImpactType.PERSISTENT
        elif "cache" in action_type or "temp" in action_type:
            return ImpactType.TRANSIENT
        elif "schedule" in action_type or "delay" in action_type:
            return ImpactType.DELAYED
        elif "accumulate" in action_type or "aggregate" in action_type:
            return ImpactType.CUMULATIVE
        elif "workflow" in action_type or "chain" in action_type:
            return ImpactType.CASCADING
        else:
            return ImpactType.DECAYING
    
    def _calculate_immediate_impact(self, action: Dict, context: Dict) -> float:
        """Calcola impatto immediato"""
        base_impact = action.get("impact", 0.5)
        urgency = action.get("urgency", 0.5)
        
        # System state modifier
        system_health = (context or {}).get("system_health", 0.8)
        
        return base_impact * (0.5 + urgency * 0.5) * (2 - system_health)
    
    def _calculate_short_term_impact(self, action: Dict, context: Dict,
                                      immediate: float) -> float:
        """Calcola impatto a breve termine"""
        decay_rate = action.get("decay_rate", 0.3)
        impact_type = self._determine_impact_type(action)
        
        if impact_type == ImpactType.TRANSIENT:
            return immediate * (1 - decay_rate)
        elif impact_type == ImpactType.CUMULATIVE:
            return immediate * 1.2
        elif impact_type == ImpactType.PERSISTENT:
            return immediate * 0.95
        else:
            return immediate * 0.8
    
    def _calculate_medium_term_impact(self, action: Dict, context: Dict,
                                       short_term: float) -> float:
        """Calcola impatto a medio termine"""
        impact_type = self._determine_impact_type(action)
        
        if impact_type == ImpactType.TRANSIENT:
            return short_term * 0.3
        elif impact_type == ImpactType.CUMULATIVE:
            return short_term * 1.3
        elif impact_type == ImpactType.PERSISTENT:
            return short_term * 0.9
        elif impact_type == ImpactType.CASCADING:
            return short_term * 1.5
        else:
            return short_term * 0.5
    
    def _calculate_long_term_impact(self, action: Dict, context: Dict,
                                     medium_term: float) -> float:
        """Calcola impatto a lungo termine"""
        impact_type = self._determine_impact_type(action)
        
        if impact_type in [ImpactType.TRANSIENT, ImpactType.DECAYING]:
            return 0.0
        elif impact_type == ImpactType.PERSISTENT:
            return medium_term * 0.8
        elif impact_type == ImpactType.CUMULATIVE:
            return medium_term * 1.2
        elif impact_type == ImpactType.CASCADING:
            return medium_term * 2.0
        else:
            return medium_term * 0.3
    
    def _determine_primary_horizon(self, action: Dict) -> TimeHorizon:
        """Determina orizzonte temporale principale"""
        action_type = action.get("type", "").lower()
        
        if any(k in action_type for k in ["instant", "immediate", "now"]):
            return TimeHorizon.IMMEDIATE
        elif any(k in action_type for k in ["quick", "fast", "temp"]):
            return TimeHorizon.SHORT_TERM
        elif any(k in action_type for k in ["schedule", "plan"]):
            return TimeHorizon.MEDIUM_TERM
        elif any(k in action_type for k in ["strategy", "long"]):
            return TimeHorizon.STRATEGIC
        else:
            return TimeHorizon.SHORT_TERM
    
    def _calculate_reversal_possibility(self, action: Dict) -> float:
        """Calcola possibilità di reversione"""
        action_type = action.get("type", "").lower()
        
        if any(k in action_type for k in ["delete", "remove", "destroy"]):
            return 0.2  # Difficile da revertire
        elif any(k in action_type for k in ["create", "add", "new"]):
            return 0.9  # Facile da revertire
        elif any(k in action_type for k in ["modify", "update", "change"]):
            return 0.7
        elif any(k in action_type for k in ["config", "setting"]):
            return 0.8
        else:
            return 0.6
    
    def _calculate_cascading_risk(self, action: Dict, context: Dict) -> float:
        """Calcola rischio di effetti a cascata"""
        dependencies = action.get("dependencies", [])
        dependent_count = len(action.get("dependents", []))
        
        base_risk = 0.1
        
        # More dependents = higher cascading risk
        base_risk += min(0.4, dependent_count * 0.1)
        
        # Critical system = higher risk
        if action.get("critical", False):
            base_risk += 0.2
        
        return min(1.0, base_risk)
    
    def _identify_side_effects(self, action: Dict, context: Dict) -> List[Dict]:
        """Identifica possibili effetti collaterali"""
        side_effects = []
        action_type = action.get("type", "").lower()
        
        if "system" in action_type:
            side_effects.append({
                "type": "performance",
                "description": "Possibile impatto su performance sistema",
                "probability": 0.3
            })
        
        if "data" in action_type or "database" in action_type:
            side_effects.append({
                "type": "data_integrity",
                "description": "Possibile impatto su integrità dati",
                "probability": 0.2
            })
        
        if "network" in action_type or "api" in action_type:
            side_effects.append({
                "type": "connectivity",
                "description": "Possibile impatto su connettività",
                "probability": 0.25
            })
        
        return side_effects
    
    # === Future Projection ===
    
    def project_future(self, current_state: Dict,
                       planned_actions: List[Dict],
                       horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM) -> FutureProjection:
        """
        Proietta stato futuro basato su azioni pianificate.
        """
        projection = FutureProjection(
            scenario_id=f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            projection_horizon=horizon
        )
        
        # Initialize projected state
        projected = current_state.copy()
        horizon_duration = self.horizon_durations[horizon]
        
        # Simulate actions over time
        time_steps = 10
        step_duration = horizon_duration / time_steps
        
        milestones = []
        risk_windows = []
        opportunity_windows = []
        
        cumulative_impact = 0.0
        
        for step in range(time_steps):
            current_time = datetime.now() + step_duration * step
            time_key = current_time.strftime("%Y-%m-%d %H:%M")
            
            # Apply relevant actions
            step_impact = 0.0
            for action in planned_actions:
                action_time = action.get("scheduled_time")
                if action_time and action_time <= current_time:
                    impact = self.analyze_impact(action, projected)
                    step_impact += impact.total_impact()
            
            cumulative_impact += step_impact
            
            # Update projected state
            projected["cumulative_impact"] = cumulative_impact
            projected["step"] = step
            
            # Confidence decreases over time
            confidence = max(0.3, 0.95 - step * 0.05)
            projection.confidence_by_time[time_key] = confidence
            
            # Identify milestones
            if step_impact > 0.3:
                milestones.append({
                    "time": time_key,
                    "type": "high_impact_action",
                    "impact": step_impact
                })
            
            # Identify risk windows
            if cumulative_impact > 0.7:
                risk_windows.append({
                    "start": time_key,
                    "type": "high_cumulative_impact",
                    "severity": cumulative_impact
                })
            
            # Identify opportunities
            if step == 0 or (step > 0 and cumulative_impact < 0.3):
                opportunity_windows.append({
                    "time": time_key,
                    "type": "low_impact_window",
                    "flexibility": 1 - cumulative_impact
                })
        
        projection.projected_states = projected
        projection.milestones = milestones
        projection.risk_windows = risk_windows
        projection.opportunity_windows = opportunity_windows[:3]
        
        return projection
    
    # === Temporal Optimization ===
    
    def optimize_schedule(self, events: List[TemporalEvent],
                          constraints: Dict = None) -> List[TemporalEvent]:
        """
        Ottimizza scheduling degli eventi.
        """
        if not events:
            return []
        
        # Sort by priority and dependencies
        sorted_events = sorted(events, key=lambda e: (-e.priority, len(e.depends_on)))
        
        # Resolve dependencies
        scheduled = []
        current_time = datetime.now()
        
        for event in sorted_events:
            # Find earliest possible start
            earliest = current_time
            
            for dep_id in event.depends_on:
                dep_event = next((e for e in scheduled if e.id == dep_id), None)
                if dep_event and dep_event.end_time:
                    earliest = max(earliest, dep_event.end_time)
            
            # Schedule event
            event.start_time = earliest
            if event.duration:
                event.end_time = earliest + event.duration
            
            scheduled.append(event)
            
            if event.end_time:
                current_time = event.end_time
        
        return scheduled
    
    def suggest_optimal_timing(self, action: Dict,
                                context: Dict = None) -> Dict:
        """Suggerisce timing ottimale per un'azione"""
        impact = self.analyze_impact(action, context)
        
        # Best timing based on impact type
        if impact.impact_type == ImpactType.TRANSIENT:
            best_timing = "immediate"
            reason = "L'impatto è temporaneo, meglio procedere subito"
        elif impact.impact_type == ImpactType.CASCADING:
            best_timing = "off_peak"
            reason = "Rischio cascata, meglio durante periodi di basso utilizzo"
        elif impact.reversal_possibility < 0.5:
            best_timing = "planned_maintenance"
            reason = "Difficile da revertire, richiede pianificazione"
        else:
            best_timing = "standard"
            reason = "Nessuna controindicazione temporale particolare"
        
        return {
            "suggested_timing": best_timing,
            "reason": reason,
            "impact_analysis": impact.to_dict(),
            "recommended_window": self._get_recommended_window(best_timing)
        }
    
    def _get_recommended_window(self, timing_type: str) -> Dict:
        """Restituisce finestra temporale raccomandata"""
        now = datetime.now()
        
        windows = {
            "immediate": {
                "start": now,
                "end": now + timedelta(minutes=5)
            },
            "off_peak": {
                "start": now.replace(hour=2, minute=0),
                "end": now.replace(hour=5, minute=0)
            },
            "planned_maintenance": {
                "start": now + timedelta(days=1),
                "end": now + timedelta(days=2)
            },
            "standard": {
                "start": now,
                "end": now + timedelta(hours=4)
            }
        }
        
        window = windows.get(timing_type, windows["standard"])
        return {
            "start": window["start"].isoformat(),
            "end": window["end"].isoformat()
        }
    
    # === Status ===
    
    def get_status(self) -> Dict:
        """Stato del reasoning temporale"""
        return {
            "total_events": len(self.events),
            "total_sequences": len(self.sequences),
            "impact_models": list(self.impact_models.keys()),
            "horizons": {h.value: str(d) for h, d in self.horizon_durations.items()}
        }
