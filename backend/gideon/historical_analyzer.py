# /backend/gideon/historical_analyzer.py
"""
🔮 GIDEON 3.0 - Historical Analyzer
Analizza dati storici e apprende dai risultati passati.
NON esegue azioni - fornisce solo analisi e pattern.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import math
import logging
import json

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Direzione del trend"""
    STRONG_UP = "strong_up"
    UP = "up"
    STABLE = "stable"
    DOWN = "down"
    STRONG_DOWN = "strong_down"
    VOLATILE = "volatile"


class PatternType(Enum):
    """Tipi di pattern"""
    RECURRING = "recurring"         # Si ripete periodicamente
    SEASONAL = "seasonal"           # Legato a stagione/periodo
    CYCLICAL = "cyclical"           # Cicli più lunghi
    ANOMALY = "anomaly"             # Outlier/anomalia
    CORRELATION = "correlation"     # Correlazione tra eventi
    SEQUENCE = "sequence"           # Sequenza di eventi
    CLUSTER = "cluster"             # Raggruppamento


class LearningType(Enum):
    """Tipi di apprendimento"""
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    OPTIMIZATION = "optimization"
    PREFERENCE = "preference"
    CONTEXT_RULE = "context_rule"


@dataclass
class HistoricalEvent:
    """Evento storico"""
    id: str
    event_type: str
    timestamp: datetime
    
    # Outcome
    success: bool
    outcome_value: float = 0.0
    
    # Context
    context: Dict = field(default_factory=dict)
    parameters: Dict = field(default_factory=dict)
    
    # Metadata
    duration_ms: float = 0.0
    user_id: str = ""
    session_id: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "outcome_value": self.outcome_value,
            "context": self.context,
            "parameters": self.parameters,
            "duration_ms": self.duration_ms,
            "tags": self.tags
        }


@dataclass
class Pattern:
    """Pattern identificato"""
    id: str
    pattern_type: PatternType
    name: str
    description: str
    
    # Pattern data
    conditions: List[Dict] = field(default_factory=list)
    frequency: float = 0.0          # Quanto spesso appare
    confidence: float = 0.0         # Confidenza nel pattern
    support: int = 0                # Numero di occorrenze
    
    # Temporal
    time_span: Optional[timedelta] = None
    periodicity: Optional[timedelta] = None
    
    # Correlation
    correlated_events: List[str] = field(default_factory=list)
    correlation_strength: float = 0.0
    
    # Insight
    insight: str = ""
    actionable: bool = False
    
    discovered_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "frequency": round(self.frequency, 3),
            "confidence": round(self.confidence, 3),
            "support": self.support,
            "periodicity_hours": self.periodicity.total_seconds() / 3600 if self.periodicity else None,
            "correlated_events": self.correlated_events,
            "correlation_strength": round(self.correlation_strength, 3),
            "insight": self.insight,
            "actionable": self.actionable
        }


@dataclass
class Learning:
    """Apprendimento dal passato"""
    id: str
    learning_type: LearningType
    description: str
    
    # What was learned
    rule: str                       # Regola/condizione appresa
    confidence: float = 0.0
    evidence_count: int = 0
    
    # Application
    applies_to: List[str] = field(default_factory=list)
    parameters: Dict = field(default_factory=dict)
    
    # Performance
    improvement: float = 0.0        # Miglioramento osservato
    
    learned_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    applications: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.learning_type.value,
            "description": self.description,
            "rule": self.rule,
            "confidence": round(self.confidence, 3),
            "evidence_count": self.evidence_count,
            "applies_to": self.applies_to,
            "improvement": round(self.improvement, 3),
            "applications": self.applications
        }


@dataclass
class HistoricalAnalysis:
    """Risultato analisi storica"""
    query: str
    time_range: Tuple[datetime, datetime]
    
    # Statistics
    total_events: int = 0
    success_rate: float = 0.0
    average_duration: float = 0.0
    
    # Trends
    trend: TrendDirection = TrendDirection.STABLE
    trend_strength: float = 0.0
    
    # Patterns found
    patterns: List[Pattern] = field(default_factory=list)
    
    # Learnings
    learnings: List[Learning] = field(default_factory=list)
    
    # Predictions
    predicted_success_rate: float = 0.0
    prediction_confidence: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "time_range": [t.isoformat() for t in self.time_range],
            "statistics": {
                "total_events": self.total_events,
                "success_rate": round(self.success_rate, 3),
                "average_duration": round(self.average_duration, 2)
            },
            "trend": {
                "direction": self.trend.value,
                "strength": round(self.trend_strength, 3)
            },
            "patterns": [p.to_dict() for p in self.patterns],
            "learnings": [l.to_dict() for l in self.learnings],
            "prediction": {
                "success_rate": round(self.predicted_success_rate, 3),
                "confidence": round(self.prediction_confidence, 3)
            },
            "recommendations": self.recommendations
        }


class HistoricalAnalyzer:
    """
    Analizzatore storico per Gideon.
    Analizza dati passati, identifica pattern e apprende.
    """
    
    def __init__(self, max_history: int = 10000):
        # Storage
        self.events: List[HistoricalEvent] = []
        self.max_history = max_history
        
        # Indices
        self.events_by_type: Dict[str, List[HistoricalEvent]] = defaultdict(list)
        self.events_by_date: Dict[str, List[HistoricalEvent]] = defaultdict(list)
        
        # Learned patterns and rules
        self.patterns: Dict[str, Pattern] = {}
        self.learnings: Dict[str, Learning] = {}
        
        # Statistics cache
        self.stats_cache: Dict[str, Dict] = {}
        self.cache_ttl = timedelta(minutes=5)
        self.last_cache_update: datetime = datetime.now()
        
        # Counters
        self._event_counter = 0
        self._pattern_counter = 0
        self._learning_counter = 0
    
    # === Event Recording ===
    
    def record_event(self, event_type: str, success: bool,
                     context: Dict = None, parameters: Dict = None,
                     duration_ms: float = 0.0, outcome_value: float = 0.0,
                     tags: List[str] = None) -> HistoricalEvent:
        """
        Registra un evento storico.
        
        Args:
            event_type: Tipo di evento
            success: Se l'evento ha avuto successo
            context: Contesto dell'evento
            parameters: Parametri usati
            duration_ms: Durata in ms
            outcome_value: Valore outcome (0-1)
            tags: Tag per categorizzazione
        
        Returns:
            Evento registrato
        """
        self._event_counter += 1
        
        event = HistoricalEvent(
            id=f"evt_{self._event_counter}",
            event_type=event_type,
            timestamp=datetime.now(),
            success=success,
            outcome_value=outcome_value,
            context=context or {},
            parameters=parameters or {},
            duration_ms=duration_ms,
            tags=tags or []
        )
        
        # Store
        self.events.append(event)
        self.events_by_type[event_type].append(event)
        date_key = event.timestamp.strftime("%Y-%m-%d")
        self.events_by_date[date_key].append(event)
        
        # Trim if needed
        if len(self.events) > self.max_history:
            self._trim_history()
        
        # Invalidate cache
        self.stats_cache.clear()
        
        # Trigger learning
        self._incremental_learn(event)
        
        return event
    
    def _trim_history(self):
        """Rimuove eventi più vecchi"""
        excess = len(self.events) - self.max_history
        if excess > 0:
            removed = self.events[:excess]
            self.events = self.events[excess:]
            
            # Update indices
            for event in removed:
                if event in self.events_by_type[event.event_type]:
                    self.events_by_type[event.event_type].remove(event)
                date_key = event.timestamp.strftime("%Y-%m-%d")
                if event in self.events_by_date[date_key]:
                    self.events_by_date[date_key].remove(event)
    
    # === Analysis ===
    
    def analyze(self, event_type: str = None,
                time_range: Tuple[datetime, datetime] = None,
                tags: List[str] = None) -> HistoricalAnalysis:
        """
        Esegue analisi storica completa.
        
        Args:
            event_type: Filtra per tipo (opzionale)
            time_range: Range temporale (opzionale)
            tags: Filtra per tag (opzionale)
        
        Returns:
            HistoricalAnalysis completa
        """
        # Filter events
        filtered = self._filter_events(event_type, time_range, tags)
        
        if not filtered:
            return HistoricalAnalysis(
                query=event_type or "all",
                time_range=time_range or (datetime.now() - timedelta(days=30), datetime.now())
            )
        
        # Calculate statistics
        total = len(filtered)
        successes = sum(1 for e in filtered if e.success)
        success_rate = successes / total if total > 0 else 0.0
        
        durations = [e.duration_ms for e in filtered if e.duration_ms > 0]
        avg_duration = statistics.mean(durations) if durations else 0.0
        
        # Detect trend
        trend, strength = self._detect_trend(filtered)
        
        # Find patterns
        patterns = self._find_patterns(filtered)
        
        # Get relevant learnings
        learnings = self._get_relevant_learnings(event_type)
        
        # Predict future
        pred_success, pred_conf = self._predict_future(filtered, patterns)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(filtered, patterns, learnings)
        
        actual_range = (
            min(e.timestamp for e in filtered),
            max(e.timestamp for e in filtered)
        )
        
        return HistoricalAnalysis(
            query=event_type or "all",
            time_range=actual_range,
            total_events=total,
            success_rate=success_rate,
            average_duration=avg_duration,
            trend=trend,
            trend_strength=strength,
            patterns=patterns,
            learnings=learnings,
            predicted_success_rate=pred_success,
            prediction_confidence=pred_conf,
            recommendations=recommendations
        )
    
    def _filter_events(self, event_type: str = None,
                       time_range: Tuple[datetime, datetime] = None,
                       tags: List[str] = None) -> List[HistoricalEvent]:
        """Filtra eventi"""
        if event_type:
            filtered = self.events_by_type.get(event_type, [])
        else:
            filtered = self.events
        
        if time_range:
            start, end = time_range
            filtered = [e for e in filtered if start <= e.timestamp <= end]
        
        if tags:
            filtered = [e for e in filtered if any(t in e.tags for t in tags)]
        
        return filtered
    
    # === Trend Detection ===
    
    def _detect_trend(self, events: List[HistoricalEvent]) -> Tuple[TrendDirection, float]:
        """Rileva trend nei dati"""
        if len(events) < 5:
            return TrendDirection.STABLE, 0.0
        
        # Sort by time
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Calculate success rate in windows
        window_size = max(1, len(sorted_events) // 5)
        windows = []
        
        for i in range(0, len(sorted_events), window_size):
            window = sorted_events[i:i+window_size]
            if window:
                rate = sum(1 for e in window if e.success) / len(window)
                windows.append(rate)
        
        if len(windows) < 2:
            return TrendDirection.STABLE, 0.0
        
        # Calculate trend
        diffs = [windows[i+1] - windows[i] for i in range(len(windows)-1)]
        avg_diff = statistics.mean(diffs)
        
        # Calculate volatility
        if len(windows) > 2:
            volatility = statistics.stdev(windows)
        else:
            volatility = 0
        
        # Determine direction
        strength = abs(avg_diff)
        
        if volatility > 0.2:
            return TrendDirection.VOLATILE, volatility
        elif avg_diff > 0.1:
            return TrendDirection.STRONG_UP, strength
        elif avg_diff > 0.03:
            return TrendDirection.UP, strength
        elif avg_diff < -0.1:
            return TrendDirection.STRONG_DOWN, strength
        elif avg_diff < -0.03:
            return TrendDirection.DOWN, strength
        else:
            return TrendDirection.STABLE, strength
    
    # === Pattern Finding ===
    
    def _find_patterns(self, events: List[HistoricalEvent]) -> List[Pattern]:
        """Trova pattern nei dati"""
        patterns = []
        
        # Find recurring patterns
        recurring = self._find_recurring_patterns(events)
        patterns.extend(recurring)
        
        # Find temporal patterns
        temporal = self._find_temporal_patterns(events)
        patterns.extend(temporal)
        
        # Find correlation patterns
        correlations = self._find_correlations(events)
        patterns.extend(correlations)
        
        # Find sequence patterns
        sequences = self._find_sequences(events)
        patterns.extend(sequences)
        
        # Store significant patterns
        for p in patterns:
            if p.confidence > 0.6:
                self.patterns[p.id] = p
        
        return patterns
    
    def _find_recurring_patterns(self, events: List[HistoricalEvent]) -> List[Pattern]:
        """Trova pattern che si ripetono"""
        patterns = []
        
        # Group by context patterns
        context_groups = defaultdict(list)
        for event in events:
            # Create context key
            key = json.dumps(sorted(event.context.items()), default=str)
            context_groups[key].append(event)
        
        for key, group in context_groups.items():
            if len(group) >= 3:  # Min support
                success_rate = sum(1 for e in group if e.success) / len(group)
                
                self._pattern_counter += 1
                pattern = Pattern(
                    id=f"pat_{self._pattern_counter}",
                    pattern_type=PatternType.RECURRING,
                    name=f"Recurring Context Pattern",
                    description=f"Context pattern che appare {len(group)} volte",
                    conditions=[{"context": key}],
                    frequency=len(group) / len(events),
                    confidence=0.5 + success_rate * 0.5,
                    support=len(group),
                    insight=f"Success rate: {success_rate:.1%} in questo contesto",
                    actionable=success_rate > 0.7 or success_rate < 0.3
                )
                patterns.append(pattern)
        
        return patterns[:5]  # Top 5
    
    def _find_temporal_patterns(self, events: List[HistoricalEvent]) -> List[Pattern]:
        """Trova pattern temporali"""
        patterns = []
        
        # Group by hour of day
        hourly_success = defaultdict(list)
        for event in events:
            hour = event.timestamp.hour
            hourly_success[hour].append(event.success)
        
        # Find best/worst hours
        hourly_rates = {
            h: sum(s)/len(s) for h, s in hourly_success.items() if len(s) >= 3
        }
        
        if hourly_rates:
            best_hour = max(hourly_rates, key=hourly_rates.get)
            worst_hour = min(hourly_rates, key=hourly_rates.get)
            
            if hourly_rates[best_hour] - hourly_rates[worst_hour] > 0.15:
                self._pattern_counter += 1
                patterns.append(Pattern(
                    id=f"pat_{self._pattern_counter}",
                    pattern_type=PatternType.SEASONAL,
                    name="Hourly Performance Pattern",
                    description=f"Performance varia per ora del giorno",
                    conditions=[{"best_hour": best_hour, "worst_hour": worst_hour}],
                    frequency=1.0,
                    confidence=0.7,
                    support=len(events),
                    insight=f"Best: {best_hour}:00 ({hourly_rates[best_hour]:.1%}), Worst: {worst_hour}:00 ({hourly_rates[worst_hour]:.1%})",
                    actionable=True
                ))
        
        # Group by day of week
        daily_success = defaultdict(list)
        for event in events:
            day = event.timestamp.weekday()
            daily_success[day].append(event.success)
        
        daily_rates = {
            d: sum(s)/len(s) for d, s in daily_success.items() if len(s) >= 3
        }
        
        if daily_rates:
            best_day = max(daily_rates, key=daily_rates.get)
            days = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
            
            self._pattern_counter += 1
            patterns.append(Pattern(
                id=f"pat_{self._pattern_counter}",
                pattern_type=PatternType.SEASONAL,
                name="Daily Performance Pattern",
                description=f"Performance varia per giorno",
                frequency=1.0,
                confidence=0.6,
                support=len(events),
                insight=f"Best day: {days[best_day]} ({daily_rates[best_day]:.1%})"
            ))
        
        return patterns
    
    def _find_correlations(self, events: List[HistoricalEvent]) -> List[Pattern]:
        """Trova correlazioni tra eventi"""
        patterns = []
        
        # Find parameter correlations with success
        param_success = defaultdict(lambda: {"success": [], "fail": []})
        
        for event in events:
            for param, value in event.parameters.items():
                if isinstance(value, (int, float)):
                    key = param
                    if event.success:
                        param_success[key]["success"].append(value)
                    else:
                        param_success[key]["fail"].append(value)
        
        for param, data in param_success.items():
            if len(data["success"]) >= 3 and len(data["fail"]) >= 3:
                avg_success = statistics.mean(data["success"])
                avg_fail = statistics.mean(data["fail"])
                
                if abs(avg_success - avg_fail) > 0:
                    # Calculate correlation
                    diff_ratio = abs(avg_success - avg_fail) / max(abs(avg_success), abs(avg_fail), 0.01)
                    
                    if diff_ratio > 0.2:
                        self._pattern_counter += 1
                        direction = "higher" if avg_success > avg_fail else "lower"
                        patterns.append(Pattern(
                            id=f"pat_{self._pattern_counter}",
                            pattern_type=PatternType.CORRELATION,
                            name=f"Parameter Correlation: {param}",
                            description=f"{param} correlato con successo",
                            correlation_strength=diff_ratio,
                            confidence=min(0.9, 0.5 + diff_ratio),
                            support=len(data["success"]) + len(data["fail"]),
                            insight=f"Success quando {param} è {direction} (avg success: {avg_success:.2f}, fail: {avg_fail:.2f})",
                            actionable=True
                        ))
        
        return patterns[:3]
    
    def _find_sequences(self, events: List[HistoricalEvent]) -> List[Pattern]:
        """Trova sequenze di eventi"""
        patterns = []
        
        if len(events) < 5:
            return patterns
        
        # Sort by time
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Find failure sequences
        failure_sequences = []
        current_failures = 0
        
        for event in sorted_events:
            if not event.success:
                current_failures += 1
            else:
                if current_failures >= 3:
                    failure_sequences.append(current_failures)
                current_failures = 0
        
        if failure_sequences:
            self._pattern_counter += 1
            patterns.append(Pattern(
                id=f"pat_{self._pattern_counter}",
                pattern_type=PatternType.SEQUENCE,
                name="Failure Sequence Pattern",
                description="Sequenze di fallimenti consecutivi",
                frequency=len(failure_sequences) / len(events),
                confidence=0.7,
                support=len(failure_sequences),
                insight=f"Trovate {len(failure_sequences)} sequenze di fallimenti (max: {max(failure_sequences)})",
                actionable=True
            ))
        
        return patterns
    
    # === Learning ===
    
    def _incremental_learn(self, event: HistoricalEvent):
        """Apprendimento incrementale da nuovo evento"""
        event_type = event.event_type
        
        # Get recent events of same type
        recent = self.events_by_type.get(event_type, [])[-100:]
        
        if len(recent) < 10:
            return
        
        # Learn success patterns
        if event.success:
            self._learn_success_pattern(event, recent)
        else:
            self._learn_failure_pattern(event, recent)
    
    def _learn_success_pattern(self, event: HistoricalEvent, recent: List[HistoricalEvent]):
        """Apprende da successo"""
        # Find common context in successes
        successes = [e for e in recent if e.success]
        
        if len(successes) < 5:
            return
        
        # Find common parameters
        common_params = self._find_common_params(successes)
        
        if common_params:
            self._learning_counter += 1
            learning = Learning(
                id=f"learn_{self._learning_counter}",
                learning_type=LearningType.SUCCESS_PATTERN,
                description=f"Pattern di successo per {event.event_type}",
                rule=f"IF params match {common_params} THEN likely success",
                confidence=len(successes) / len(recent),
                evidence_count=len(successes),
                applies_to=[event.event_type],
                parameters=common_params,
                improvement=0.1
            )
            self.learnings[learning.id] = learning
    
    def _learn_failure_pattern(self, event: HistoricalEvent, recent: List[HistoricalEvent]):
        """Apprende da fallimento"""
        failures = [e for e in recent if not e.success]
        
        if len(failures) < 3:
            return
        
        # Find common context in failures
        common_context = self._find_common_context(failures)
        
        if common_context:
            self._learning_counter += 1
            learning = Learning(
                id=f"learn_{self._learning_counter}",
                learning_type=LearningType.FAILURE_PATTERN,
                description=f"Pattern di fallimento per {event.event_type}",
                rule=f"IF context match {common_context} THEN likely failure",
                confidence=len(failures) / len(recent),
                evidence_count=len(failures),
                applies_to=[event.event_type],
                parameters={"avoid_context": common_context}
            )
            self.learnings[learning.id] = learning
    
    def _find_common_params(self, events: List[HistoricalEvent]) -> Dict:
        """Trova parametri comuni"""
        if not events:
            return {}
        
        common = {}
        first_params = events[0].parameters
        
        for key, value in first_params.items():
            if isinstance(value, (int, float)):
                values = [e.parameters.get(key) for e in events if key in e.parameters]
                if len(values) >= len(events) * 0.8:
                    avg = statistics.mean(v for v in values if v is not None)
                    std = statistics.stdev(values) if len(values) > 1 else 0
                    if std < avg * 0.3:  # Low variance
                        common[key] = {"avg": avg, "std": std}
        
        return common
    
    def _find_common_context(self, events: List[HistoricalEvent]) -> Dict:
        """Trova contesto comune"""
        if not events:
            return {}
        
        # Count context values
        context_counts = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            for key, value in event.context.items():
                context_counts[key][str(value)] += 1
        
        common = {}
        threshold = len(events) * 0.6
        
        for key, values in context_counts.items():
            for value, count in values.items():
                if count >= threshold:
                    common[key] = value
        
        return common
    
    def _get_relevant_learnings(self, event_type: str = None) -> List[Learning]:
        """Ottiene apprendimenti rilevanti"""
        relevant = []
        
        for learning in self.learnings.values():
            if event_type is None or event_type in learning.applies_to:
                relevant.append(learning)
        
        return sorted(relevant, key=lambda l: l.confidence, reverse=True)[:5]
    
    # === Prediction ===
    
    def _predict_future(self, events: List[HistoricalEvent],
                        patterns: List[Pattern]) -> Tuple[float, float]:
        """Predice success rate futuro"""
        if not events:
            return 0.5, 0.3
        
        # Base: historical rate
        historical_rate = sum(1 for e in events if e.success) / len(events)
        
        # Adjust for trend
        trend, strength = self._detect_trend(events)
        
        trend_adjustment = 0.0
        if trend in [TrendDirection.UP, TrendDirection.STRONG_UP]:
            trend_adjustment = strength * 0.5
        elif trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
            trend_adjustment = -strength * 0.5
        
        predicted = max(0.0, min(1.0, historical_rate + trend_adjustment))
        
        # Confidence based on data and patterns
        data_confidence = min(0.9, 0.3 + len(events) * 0.01)
        pattern_confidence = max(p.confidence for p in patterns) if patterns else 0.5
        
        confidence = (data_confidence + pattern_confidence) / 2
        
        return predicted, confidence
    
    def _generate_recommendations(self, events: List[HistoricalEvent],
                                   patterns: List[Pattern],
                                   learnings: List[Learning]) -> List[str]:
        """Genera raccomandazioni"""
        recommendations = []
        
        # From patterns
        for pattern in patterns:
            if pattern.actionable:
                recommendations.append(f"Pattern: {pattern.insight}")
        
        # From learnings
        for learning in learnings:
            if learning.learning_type == LearningType.SUCCESS_PATTERN:
                recommendations.append(f"Usa parametri: {learning.parameters}")
            elif learning.learning_type == LearningType.FAILURE_PATTERN:
                recommendations.append(f"Evita contesto: {learning.parameters.get('avoid_context', {})}")
        
        # From trend
        if events:
            trend, _ = self._detect_trend(events)
            if trend == TrendDirection.STRONG_DOWN:
                recommendations.append("⚠️ Trend negativo: investigare cause recenti")
            elif trend == TrendDirection.VOLATILE:
                recommendations.append("⚠️ Alta volatilità: stabilizzare operazioni")
        
        return recommendations[:5]
    
    # === Query Methods ===
    
    def get_success_rate(self, event_type: str,
                         days: int = 30) -> Dict:
        """Ottiene success rate per tipo evento"""
        start = datetime.now() - timedelta(days=days)
        events = [e for e in self.events_by_type.get(event_type, [])
                  if e.timestamp >= start]
        
        if not events:
            return {"event_type": event_type, "success_rate": 0.0, "count": 0}
        
        return {
            "event_type": event_type,
            "success_rate": sum(1 for e in events if e.success) / len(events),
            "count": len(events),
            "period_days": days
        }
    
    def get_performance_over_time(self, event_type: str = None,
                                   granularity: str = "day") -> List[Dict]:
        """Performance nel tempo"""
        if event_type:
            events = self.events_by_type.get(event_type, [])
        else:
            events = self.events
        
        # Group by granularity
        groups = defaultdict(list)
        
        for event in events:
            if granularity == "hour":
                key = event.timestamp.strftime("%Y-%m-%d %H:00")
            elif granularity == "day":
                key = event.timestamp.strftime("%Y-%m-%d")
            elif granularity == "week":
                key = event.timestamp.strftime("%Y-W%W")
            else:
                key = event.timestamp.strftime("%Y-%m")
            
            groups[key].append(event)
        
        result = []
        for period, group in sorted(groups.items()):
            result.append({
                "period": period,
                "count": len(group),
                "success_rate": sum(1 for e in group if e.success) / len(group),
                "avg_duration": statistics.mean(e.duration_ms for e in group if e.duration_ms > 0) if any(e.duration_ms > 0 for e in group) else 0
            })
        
        return result
    
    def apply_learning(self, event_type: str, current_context: Dict) -> Dict:
        """Applica apprendimenti a contesto corrente"""
        applicable = [l for l in self.learnings.values()
                     if event_type in l.applies_to]
        
        suggestions = {
            "recommended_params": {},
            "avoid_contexts": [],
            "confidence_boost": 0.0
        }
        
        for learning in applicable:
            learning.last_applied = datetime.now()
            learning.applications += 1
            
            if learning.learning_type == LearningType.SUCCESS_PATTERN:
                suggestions["recommended_params"].update(learning.parameters)
                suggestions["confidence_boost"] += learning.confidence * 0.1
            
            elif learning.learning_type == LearningType.FAILURE_PATTERN:
                avoid = learning.parameters.get("avoid_context", {})
                if avoid:
                    suggestions["avoid_contexts"].append(avoid)
        
        return suggestions
    
    # === Status ===
    
    def get_status(self) -> Dict:
        """Stato dell'analyzer"""
        return {
            "total_events": len(self.events),
            "event_types": len(self.events_by_type),
            "patterns_found": len(self.patterns),
            "learnings_acquired": len(self.learnings),
            "max_history": self.max_history,
            "oldest_event": self.events[0].timestamp.isoformat() if self.events else None,
            "newest_event": self.events[-1].timestamp.isoformat() if self.events else None
        }
    
    def export_learnings(self) -> Dict:
        """Esporta tutti gli apprendimenti"""
        return {
            "patterns": [p.to_dict() for p in self.patterns.values()],
            "learnings": [l.to_dict() for l in self.learnings.values()],
            "exported_at": datetime.now().isoformat()
        }
