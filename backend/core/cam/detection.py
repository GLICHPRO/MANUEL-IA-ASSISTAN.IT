"""
🔍 CAM DETECTION LAYER
======================
"Siamo in crisi?"

Unisce segnali da:
- Tool interni
- Dati esterni
- Comportamento umano
- Qualità dei dati

Trigger CAM se:
- Conflitti tra tool
- Rischio ↑ rapido
- Confidenza ↓
- Input emotivamente carichi
- Overload operativo

📌 CAM può attivarsi anche in modo "soft" (parziale)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import logging
from collections import defaultdict
import statistics
import re

logger = logging.getLogger(__name__)


class CrisisLevel(Enum):
    """Livelli di crisi"""
    NONE = 0  # Operazioni normali
    WATCH = 1  # Monitoraggio aumentato
    SOFT = 2  # CAM parziale attivo
    ACTIVE = 3  # CAM completamente attivo
    CRITICAL = 4  # Massima allerta


class SignalSource(Enum):
    """Sorgenti dei segnali"""
    TOOL_INTERNAL = "tool_internal"  # Tool GIDEON
    DATA_EXTERNAL = "data_external"  # Dati esterni
    HUMAN_BEHAVIOR = "human_behavior"  # Comportamento utente
    DATA_QUALITY = "data_quality"  # Qualità dati
    SYSTEM_STATUS = "system_status"  # Stato sistema
    TIME_PRESSURE = "time_pressure"  # Pressione temporale
    CONFLICT_DETECTION = "conflict_detection"  # Conflitti rilevati


class SignalType(Enum):
    """Tipi di segnale"""
    # Tool conflicts
    TOOL_DISAGREEMENT = "tool_disagreement"
    TOOL_FAILURE = "tool_failure"
    TOOL_TIMEOUT = "tool_timeout"
    
    # Risk signals
    RISK_SPIKE = "risk_spike"
    RISK_SUSTAINED_HIGH = "risk_sustained_high"
    
    # Confidence signals
    CONFIDENCE_DROP = "confidence_drop"
    CONFIDENCE_INCONSISTENT = "confidence_inconsistent"
    
    # Human signals
    EMOTIONAL_INPUT = "emotional_input"
    RAPID_QUERIES = "rapid_queries"
    ERROR_RATE_HIGH = "error_rate_high"
    DECISION_REVERSAL = "decision_reversal"
    
    # Operational signals
    OVERLOAD = "overload"
    DEADLINE_PRESSURE = "deadline_pressure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    
    # Data signals
    DATA_INCONSISTENCY = "data_inconsistency"
    DATA_MISSING = "data_missing"
    DATA_STALE = "data_stale"


@dataclass
class CrisisSignal:
    """Segnale di crisi"""
    signal_id: str
    signal_type: SignalType
    source: SignalSource
    severity: float  # 0-1
    confidence: float  # 0-1
    description: str
    evidence: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weighted_severity(self) -> float:
        """Severity pesata per confidence"""
        return self.severity * self.confidence


@dataclass
class CrisisAssessment:
    """Valutazione complessiva crisi"""
    level: CrisisLevel
    score: float  # 0-1
    active_signals: List[CrisisSignal]
    dominant_source: SignalSource
    trigger_reasons: List[str]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class EmotionalInputDetector:
    """Rileva input emotivamente carichi"""
    
    # Pattern indicatori stress/panico
    STRESS_PATTERNS = [
        r'\b(urgent|urgente|asap|immediately|subito|ora|adesso|help|aiuto)\b',
        r'\b(critical|critico|emergency|emergenza|disaster|disastro)\b',
        r'\b(panic|panico|worried|preoccupato|scared|spaventato)\b',
        r'[!?]{2,}',  # Multiple punctuation
        r'[A-Z]{4,}',  # All caps words
    ]
    
    # Pattern frustrazione
    FRUSTRATION_PATTERNS = [
        r'\b(doesn\'t work|non funziona|broken|rotto|again|ancora)\b',
        r'\b(why|perché|come mai)\b.*[?!]{2,}',
        r'\b(stupid|stupido|useless|inutile)\b',
    ]
    
    def __init__(self):
        self.stress_regex = [re.compile(p, re.IGNORECASE) for p in self.STRESS_PATTERNS]
        self.frustration_regex = [re.compile(p, re.IGNORECASE) for p in self.FRUSTRATION_PATTERNS]
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analizza testo per contenuto emotivo"""
        
        stress_score = 0
        frustration_score = 0
        
        for pattern in self.stress_regex:
            if pattern.search(text):
                stress_score += 0.2
        
        for pattern in self.frustration_regex:
            if pattern.search(text):
                frustration_score += 0.25
        
        return {
            'stress': min(stress_score, 1.0),
            'frustration': min(frustration_score, 1.0),
            'combined': min((stress_score + frustration_score) / 2, 1.0)
        }


class CrisisSignalAggregator:
    """
    Aggregatore principale segnali di crisi.
    
    Monitora continuamente per:
    - Conflitti tra tool
    - Spike di rischio
    - Drop di confidenza
    - Input emotivi
    - Overload operativo
    """
    
    # Soglie per livelli crisi
    THRESHOLDS = {
        CrisisLevel.WATCH: 0.3,
        CrisisLevel.SOFT: 0.5,
        CrisisLevel.ACTIVE: 0.7,
        CrisisLevel.CRITICAL: 0.85
    }
    
    # Pesi per fonte segnale
    SOURCE_WEIGHTS = {
        SignalSource.TOOL_INTERNAL: 1.0,
        SignalSource.DATA_EXTERNAL: 0.8,
        SignalSource.HUMAN_BEHAVIOR: 0.9,
        SignalSource.DATA_QUALITY: 0.7,
        SignalSource.SYSTEM_STATUS: 1.0,
        SignalSource.TIME_PRESSURE: 0.6,
        SignalSource.CONFLICT_DETECTION: 1.2
    }
    
    # Decay time per segnali (minuti)
    SIGNAL_DECAY = {
        SignalType.TOOL_FAILURE: 30,
        SignalType.RISK_SPIKE: 15,
        SignalType.EMOTIONAL_INPUT: 10,
        SignalType.OVERLOAD: 20,
    }
    DEFAULT_DECAY = 15
    
    def __init__(self):
        self.signals: List[CrisisSignal] = []
        self.signal_history: List[CrisisSignal] = []
        self.current_level = CrisisLevel.NONE
        self.emotional_detector = EmotionalInputDetector()
        
        # Tracking per pattern
        self.risk_history: List[tuple] = []  # (timestamp, score)
        self.confidence_history: List[tuple] = []
        self.query_timestamps: List[datetime] = []
        
        # Tool status
        self.tool_status: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks per notifica
        self.on_level_change: Optional[Callable[[CrisisLevel, CrisisLevel], None]] = None
    
    def register_signal(
        self,
        signal_type: SignalType,
        source: SignalSource,
        severity: float,
        description: str,
        evidence: Optional[List[str]] = None,
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CrisisSignal:
        """Registra un nuovo segnale"""
        
        signal = CrisisSignal(
            signal_id=f"sig_{datetime.now().timestamp()}",
            signal_type=signal_type,
            source=source,
            severity=min(max(severity, 0), 1),
            confidence=min(max(confidence, 0), 1),
            description=description,
            evidence=evidence or [],
            metadata=metadata or {}
        )
        
        self.signals.append(signal)
        self.signal_history.append(signal)
        
        # Limita history
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        logger.info(f"🔍 Segnale crisi: [{signal_type.value}] {description} (sev: {severity:.2f})")
        
        # Rivaluta livello
        self._evaluate_level()
        
        return signal
    
    def analyze_user_input(self, text: str) -> Optional[CrisisSignal]:
        """Analizza input utente per segnali emotivi"""
        
        emotions = self.emotional_detector.analyze(text)
        
        if emotions['combined'] > 0.3:
            return self.register_signal(
                signal_type=SignalType.EMOTIONAL_INPUT,
                source=SignalSource.HUMAN_BEHAVIOR,
                severity=emotions['combined'],
                description="Input emotivamente carico rilevato",
                evidence=[
                    f"Stress score: {emotions['stress']:.2f}",
                    f"Frustration score: {emotions['frustration']:.2f}"
                ],
                confidence=0.7,
                metadata={'emotions': emotions, 'text_length': len(text)}
            )
        
        return None
    
    def track_query(self):
        """Traccia query per rilevare rapid-fire"""
        
        now = datetime.now()
        self.query_timestamps.append(now)
        
        # Mantieni solo ultimi 5 minuti
        cutoff = now - timedelta(minutes=5)
        self.query_timestamps = [t for t in self.query_timestamps if t > cutoff]
        
        # Check rapid queries (>10 in 2 minuti)
        recent = [t for t in self.query_timestamps if t > now - timedelta(minutes=2)]
        
        if len(recent) > 10:
            self.register_signal(
                signal_type=SignalType.RAPID_QUERIES,
                source=SignalSource.HUMAN_BEHAVIOR,
                severity=min(len(recent) / 20, 1.0),
                description=f"Query rapide: {len(recent)} in 2 minuti",
                evidence=[f"Query count: {len(recent)}"],
                confidence=0.9
            )
    
    def track_risk(self, risk_score: float):
        """Traccia evoluzione rischio"""
        
        now = datetime.now()
        self.risk_history.append((now, risk_score))
        
        # Mantieni solo ultima ora
        cutoff = now - timedelta(hours=1)
        self.risk_history = [(t, s) for t, s in self.risk_history if t > cutoff]
        
        if len(self.risk_history) >= 5:
            # Check spike (aumento >30% in 5 minuti)
            recent = [s for t, s in self.risk_history if t > now - timedelta(minutes=5)]
            older = [s for t, s in self.risk_history if t <= now - timedelta(minutes=5)]
            
            if recent and older:
                recent_avg = statistics.mean(recent)
                older_avg = statistics.mean(older)
                
                if recent_avg - older_avg > 0.3:
                    self.register_signal(
                        signal_type=SignalType.RISK_SPIKE,
                        source=SignalSource.DATA_EXTERNAL,
                        severity=recent_avg,
                        description=f"Spike rischio: {older_avg:.1%} → {recent_avg:.1%}",
                        evidence=[
                            f"Delta: +{(recent_avg - older_avg):.1%}",
                            f"Timeframe: 5 minuti"
                        ],
                        confidence=0.85
                    )
            
            # Check sustained high (>70% per >10 minuti)
            sustained = [s for t, s in self.risk_history if t > now - timedelta(minutes=10)]
            if sustained and all(s > 0.7 for s in sustained) and len(sustained) >= 10:
                self.register_signal(
                    signal_type=SignalType.RISK_SUSTAINED_HIGH,
                    source=SignalSource.DATA_EXTERNAL,
                    severity=statistics.mean(sustained),
                    description="Rischio sostenuto alto",
                    evidence=[f"Media: {statistics.mean(sustained):.1%} per 10+ minuti"],
                    confidence=0.9
                )
    
    def track_confidence(self, confidence: float):
        """Traccia evoluzione confidenza"""
        
        now = datetime.now()
        self.confidence_history.append((now, confidence))
        
        # Mantieni solo ultimi 30 minuti
        cutoff = now - timedelta(minutes=30)
        self.confidence_history = [(t, c) for t, c in self.confidence_history if t > cutoff]
        
        if len(self.confidence_history) >= 5:
            # Check drop (calo >20% in 5 minuti)
            recent = [c for t, c in self.confidence_history if t > now - timedelta(minutes=5)]
            older = [c for t, c in self.confidence_history if t <= now - timedelta(minutes=5)]
            
            if recent and older:
                recent_avg = statistics.mean(recent)
                older_avg = statistics.mean(older)
                
                if older_avg - recent_avg > 0.2:
                    self.register_signal(
                        signal_type=SignalType.CONFIDENCE_DROP,
                        source=SignalSource.TOOL_INTERNAL,
                        severity=1 - recent_avg,
                        description=f"Drop confidenza: {older_avg:.1%} → {recent_avg:.1%}",
                        evidence=[f"Delta: -{(older_avg - recent_avg):.1%}"],
                        confidence=0.8
                    )
    
    def report_tool_conflict(
        self,
        tool1: str,
        tool2: str,
        conflict_description: str,
        severity: float = 0.6
    ):
        """Riporta conflitto tra tool"""
        
        self.register_signal(
            signal_type=SignalType.TOOL_DISAGREEMENT,
            source=SignalSource.CONFLICT_DETECTION,
            severity=severity,
            description=f"Conflitto tra {tool1} e {tool2}",
            evidence=[conflict_description],
            confidence=0.9,
            metadata={'tool1': tool1, 'tool2': tool2}
        )
    
    def report_tool_failure(self, tool_name: str, error: str):
        """Riporta fallimento tool"""
        
        self.tool_status[tool_name] = {
            'status': 'failed',
            'error': error,
            'timestamp': datetime.now()
        }
        
        self.register_signal(
            signal_type=SignalType.TOOL_FAILURE,
            source=SignalSource.TOOL_INTERNAL,
            severity=0.7,
            description=f"Tool '{tool_name}' fallito",
            evidence=[error],
            confidence=1.0,
            metadata={'tool': tool_name}
        )
    
    def report_data_issue(
        self,
        issue_type: str,
        description: str,
        severity: float = 0.5
    ):
        """Riporta problema dati"""
        
        signal_type = {
            'inconsistency': SignalType.DATA_INCONSISTENCY,
            'missing': SignalType.DATA_MISSING,
            'stale': SignalType.DATA_STALE
        }.get(issue_type, SignalType.DATA_INCONSISTENCY)
        
        self.register_signal(
            signal_type=signal_type,
            source=SignalSource.DATA_QUALITY,
            severity=severity,
            description=description,
            evidence=[f"Type: {issue_type}"],
            confidence=0.8
        )
    
    def _evaluate_level(self):
        """Valuta livello crisi complessivo"""
        
        # Pulisci segnali scaduti
        self._cleanup_expired_signals()
        
        if not self.signals:
            new_level = CrisisLevel.NONE
        else:
            # Calcola score aggregato
            score = self._calculate_aggregate_score()
            
            # Determina livello
            new_level = CrisisLevel.NONE
            for level in [CrisisLevel.CRITICAL, CrisisLevel.ACTIVE, 
                         CrisisLevel.SOFT, CrisisLevel.WATCH]:
                if score >= self.THRESHOLDS[level]:
                    new_level = level
                    break
        
        # Notifica se cambiato
        if new_level != self.current_level:
            old_level = self.current_level
            self.current_level = new_level
            
            logger.warning(f"🚨 Livello crisi: {old_level.name} → {new_level.name}")
            
            if self.on_level_change:
                self.on_level_change(old_level, new_level)
    
    def _calculate_aggregate_score(self) -> float:
        """Calcola score aggregato da tutti i segnali"""
        
        if not self.signals:
            return 0.0
        
        # Raggruppa per fonte
        by_source: Dict[SignalSource, List[CrisisSignal]] = defaultdict(list)
        for signal in self.signals:
            by_source[signal.source].append(signal)
        
        # Calcola score per fonte (media pesata)
        source_scores = {}
        for source, signals in by_source.items():
            avg_severity = statistics.mean(s.weighted_severity for s in signals)
            weight = self.SOURCE_WEIGHTS.get(source, 1.0)
            source_scores[source] = avg_severity * weight
        
        # Score finale: media pesata + bonus per multiple fonti
        if source_scores:
            base_score = statistics.mean(source_scores.values())
            
            # Bonus per correlazione tra fonti (più fonti = più serio)
            source_count_bonus = min((len(source_scores) - 1) * 0.1, 0.3)
            
            return min(base_score + source_count_bonus, 1.0)
        
        return 0.0
    
    def _cleanup_expired_signals(self):
        """Rimuovi segnali scaduti"""
        
        now = datetime.now()
        valid_signals = []
        
        for signal in self.signals:
            decay_minutes = self.SIGNAL_DECAY.get(
                signal.signal_type, 
                self.DEFAULT_DECAY
            )
            expiry = signal.timestamp + timedelta(minutes=decay_minutes)
            
            if now < expiry:
                valid_signals.append(signal)
        
        self.signals = valid_signals
    
    def get_assessment(self) -> CrisisAssessment:
        """Ottiene valutazione crisi completa"""
        
        self._cleanup_expired_signals()
        score = self._calculate_aggregate_score()
        
        # Trova fonte dominante
        if self.signals:
            source_counts = defaultdict(int)
            for s in self.signals:
                source_counts[s.source] += s.weighted_severity
            dominant = max(source_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant = SignalSource.SYSTEM_STATUS
        
        # Genera trigger reasons
        trigger_reasons = []
        for signal in self.signals[:5]:  # Top 5
            trigger_reasons.append(f"[{signal.signal_type.value}] {signal.description}")
        
        # Genera recommended actions
        actions = self._generate_recommendations()
        
        return CrisisAssessment(
            level=self.current_level,
            score=score,
            active_signals=self.signals.copy(),
            dominant_source=dominant,
            trigger_reasons=trigger_reasons,
            recommended_actions=actions
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Genera raccomandazioni basate su segnali"""
        
        recommendations = []
        
        signal_types = {s.signal_type for s in self.signals}
        
        if SignalType.TOOL_FAILURE in signal_types:
            recommendations.append("Verificare status tool e considerare alternative")
        
        if SignalType.TOOL_DISAGREEMENT in signal_types:
            recommendations.append("Analizzare manualmente i risultati contrastanti")
        
        if SignalType.RISK_SPIKE in signal_types:
            recommendations.append("Attivare monitoraggio intensivo")
        
        if SignalType.EMOTIONAL_INPUT in signal_types:
            recommendations.append("Rallentare risposte e semplificare output")
        
        if SignalType.RAPID_QUERIES in signal_types:
            recommendations.append("Verificare se l'utente ha bisogno di supporto")
        
        if SignalType.DATA_INCONSISTENCY in signal_types:
            recommendations.append("Validare fonti dati prima di procedere")
        
        if self.current_level == CrisisLevel.CRITICAL:
            recommendations.insert(0, "⚠️ CONSIDERARE INTERVENTO UMANO DIRETTO")
        
        return recommendations or ["Continuare monitoraggio standard"]
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        assessment = self.get_assessment()
        
        level_emoji = {
            CrisisLevel.NONE: "🟢",
            CrisisLevel.WATCH: "🟡",
            CrisisLevel.SOFT: "🟠",
            CrisisLevel.ACTIVE: "🔴",
            CrisisLevel.CRITICAL: "🚨"
        }
        
        signals_str = ""
        for signal in assessment.active_signals[:10]:
            signals_str += f"- [{signal.signal_type.value}] {signal.description} (sev: {signal.severity:.1%})\n"
        
        recommendations_str = "\n".join(f"- {r}" for r in assessment.recommended_actions)
        
        return f"""
# 🔍 Crisis Detection Layer

## Status
{level_emoji[assessment.level]} **{assessment.level.name}** (score: {assessment.score:.1%})

## Fonte Dominante
**{assessment.dominant_source.value}**

## Segnali Attivi ({len(assessment.active_signals)})
{signals_str or '- Nessun segnale attivo'}

## Raccomandazioni
{recommendations_str}

## Tracking
| Metrica | Valore |
|---------|--------|
| Risk samples | {len(self.risk_history)} |
| Confidence samples | {len(self.confidence_history)} |
| Query count (5min) | {len(self.query_timestamps)} |
| Signals in history | {len(self.signal_history)} |
"""


# Singleton
_signal_aggregator: Optional[CrisisSignalAggregator] = None


def get_signal_aggregator() -> CrisisSignalAggregator:
    """Ottiene istanza singleton"""
    global _signal_aggregator
    if _signal_aggregator is None:
        _signal_aggregator = CrisisSignalAggregator()
    return _signal_aggregator
