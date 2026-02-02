"""
🔄 POST-CRISIS RECONSTRUCTOR
==============================
Dopo ogni crisi, GIDEON:
1. Ricostruisce la timeline
2. Identifica trigger → cascate
3. Propone "lessons learned"
4. Aggiorna regole automatiche
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class CrisisPhase(Enum):
    """Fasi di una crisi"""
    PRE_CRISIS = "pre_crisis"
    TRIGGER = "trigger"
    ESCALATION = "escalation"
    PEAK = "peak"
    MITIGATION = "mitigation"
    RESOLUTION = "resolution"
    POST_CRISIS = "post_crisis"


class EventType(Enum):
    """Tipi di eventi nella timeline"""
    ANOMALY = "anomaly"
    ALERT = "alert"
    DECISION = "decision"
    ACTION = "action"
    FAILURE = "failure"
    RECOVERY = "recovery"
    HUMAN_INTERVENTION = "human_intervention"
    SYSTEM_RESPONSE = "system_response"
    CASCADE = "cascade"
    MITIGATION = "mitigation"


class LessonCategory(Enum):
    """Categorie di lezioni apprese"""
    DETECTION = "detection"          # Migliorare rilevamento
    RESPONSE = "response"            # Migliorare risposta
    PREVENTION = "prevention"        # Prevenire ricorrenza
    COMMUNICATION = "communication"  # Migliorare comunicazione
    PROCESS = "process"              # Migliorare processi
    TECHNOLOGY = "technology"        # Miglioramenti tecnici
    TRAINING = "training"            # Formazione necessaria


@dataclass
class TimelineEvent:
    """Evento nella timeline della crisi"""
    timestamp: datetime
    event_type: EventType
    phase: CrisisPhase
    description: str
    severity: int  # 1-10
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    caused_by: Optional[str] = None  # ID evento precedente
    leads_to: List[str] = field(default_factory=list)  # ID eventi successivi
    event_id: str = field(default_factory=lambda: f"evt_{datetime.now().timestamp()}")


@dataclass
class CascadeChain:
    """Catena di eventi a cascata"""
    chain_id: str
    trigger_event: TimelineEvent
    cascade_events: List[TimelineEvent]
    total_impact: float
    duration: timedelta
    broken_at: Optional[TimelineEvent]
    description: str


@dataclass
class LessonLearned:
    """Lezione appresa dalla crisi"""
    lesson_id: str
    category: LessonCategory
    title: str
    description: str
    evidence: List[str]
    recommended_actions: List[str]
    priority: int  # 1-5
    estimated_effort: str  # "low", "medium", "high"
    potential_impact: str
    assigned_to: Optional[str]
    deadline: Optional[datetime]


@dataclass
class RuleProposal:
    """Proposta di nuova regola automatica"""
    rule_id: str
    trigger_condition: str
    action: str
    rationale: str
    source_events: List[str]
    confidence: float
    requires_approval: bool
    auto_activate: bool


@dataclass
class CrisisReport:
    """Report completo della crisi"""
    crisis_id: str
    title: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[timedelta]
    timeline: List[TimelineEvent]
    cascades: List[CascadeChain]
    lessons: List[LessonLearned]
    rule_proposals: List[RuleProposal]
    metrics: Dict[str, Any]
    summary: str
    severity_peak: int
    resolution_status: str
    generated_at: datetime = field(default_factory=datetime.now)


class PostCrisisReconstructor:
    """
    Ricostruttore post-crisi per GIDEON.
    
    Dopo ogni crisi:
    1. Ricostruisce la timeline
    2. Identifica trigger → cascate
    3. Propone "lessons learned"
    4. Aggiorna regole automatiche
    """
    
    def __init__(self):
        self.active_events: List[TimelineEvent] = []
        self.crises_archive: List[CrisisReport] = []
        self.proposed_rules: List[RuleProposal] = []
        self.lessons_database: List[LessonLearned] = []
        
    def record_event(
        self,
        event_type: EventType,
        description: str,
        severity: int = 5,
        source: str = "system",
        phase: Optional[CrisisPhase] = None,
        data: Optional[Dict[str, Any]] = None,
        caused_by: Optional[str] = None
    ) -> TimelineEvent:
        """Registra un evento durante la crisi"""
        
        # Determina fase automaticamente se non specificata
        if phase is None:
            phase = self._infer_phase(event_type, severity)
        
        event = TimelineEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            phase=phase,
            description=description,
            severity=severity,
            source=source,
            data=data or {},
            caused_by=caused_by
        )
        
        # Link causale
        if caused_by:
            for e in self.active_events:
                if e.event_id == caused_by:
                    e.leads_to.append(event.event_id)
                    break
        
        self.active_events.append(event)
        logger.info(f"🔄 Evento registrato: [{event_type.value}] {description[:50]}")
        
        return event
    
    def _infer_phase(self, event_type: EventType, severity: int) -> CrisisPhase:
        """Inferisce la fase dalla tipologia evento"""
        
        if event_type == EventType.ANOMALY:
            return CrisisPhase.PRE_CRISIS if severity < 5 else CrisisPhase.TRIGGER
        elif event_type == EventType.CASCADE:
            return CrisisPhase.ESCALATION
        elif event_type == EventType.FAILURE:
            return CrisisPhase.PEAK if severity >= 7 else CrisisPhase.ESCALATION
        elif event_type == EventType.MITIGATION:
            return CrisisPhase.MITIGATION
        elif event_type == EventType.RECOVERY:
            return CrisisPhase.RESOLUTION
        
        return CrisisPhase.ESCALATION
    
    async def reconstruct_crisis(
        self,
        crisis_title: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> CrisisReport:
        """
        Ricostruisce la crisi dagli eventi registrati.
        """
        
        logger.info("🔄 Avvio ricostruzione crisi...")
        
        # Filtra eventi per periodo
        events = self._filter_events(start_time, end_time)
        
        if not events:
            logger.warning("Nessun evento trovato per la ricostruzione")
            events = self.active_events[-50:]  # Ultimi 50
        
        # Ordina timeline
        timeline = sorted(events, key=lambda e: e.timestamp)
        
        # Identifica cascate
        cascades = self._identify_cascades(timeline)
        
        # Genera lessons learned
        lessons = self._generate_lessons(timeline, cascades)
        
        # Proponi nuove regole
        rules = self._propose_rules(timeline, cascades, lessons)
        
        # Calcola metriche
        metrics = self._calculate_metrics(timeline, cascades)
        
        # Genera summary
        summary = self._generate_summary(timeline, cascades, lessons, metrics)
        
        # Determina tempi
        actual_start = timeline[0].timestamp if timeline else datetime.now()
        actual_end = timeline[-1].timestamp if timeline else datetime.now()
        
        report = CrisisReport(
            crisis_id=f"crisis_{actual_start.strftime('%Y%m%d_%H%M%S')}",
            title=crisis_title or self._generate_title(timeline),
            start_time=actual_start,
            end_time=actual_end,
            duration=actual_end - actual_start,
            timeline=timeline,
            cascades=cascades,
            lessons=lessons,
            rule_proposals=rules,
            metrics=metrics,
            summary=summary,
            severity_peak=max((e.severity for e in timeline), default=0),
            resolution_status=self._determine_resolution_status(timeline)
        )
        
        # Archivia
        self.crises_archive.append(report)
        self.lessons_database.extend(lessons)
        self.proposed_rules.extend(rules)
        
        # Clear eventi attivi
        self.active_events = []
        
        logger.info(f"🔄 Crisi ricostruita: {report.crisis_id}")
        
        return report
    
    def _filter_events(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[TimelineEvent]:
        """Filtra eventi per periodo"""
        
        events = self.active_events
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events
    
    def _identify_cascades(self, timeline: List[TimelineEvent]) -> List[CascadeChain]:
        """Identifica catene di eventi a cascata"""
        
        cascades = []
        visited = set()
        
        # Trova trigger events
        triggers = [e for e in timeline if e.event_type == EventType.ANOMALY and e.severity >= 5]
        
        for trigger in triggers:
            if trigger.event_id in visited:
                continue
            
            # Segui la catena
            chain_events = [trigger]
            current_id = trigger.event_id
            
            while True:
                next_events = [
                    e for e in timeline 
                    if e.caused_by == current_id and e.event_id not in visited
                ]
                
                if not next_events:
                    break
                
                # Prendi il più severo
                next_event = max(next_events, key=lambda e: e.severity)
                chain_events.append(next_event)
                visited.add(next_event.event_id)
                current_id = next_event.event_id
            
            if len(chain_events) > 1:
                # Trova dove si è rotta la catena (se presente)
                broken_at = None
                for e in chain_events:
                    if e.event_type in [EventType.MITIGATION, EventType.HUMAN_INTERVENTION]:
                        broken_at = e
                        break
                
                cascade = CascadeChain(
                    chain_id=f"cascade_{trigger.event_id}",
                    trigger_event=trigger,
                    cascade_events=chain_events,
                    total_impact=sum(e.severity for e in chain_events),
                    duration=chain_events[-1].timestamp - trigger.timestamp,
                    broken_at=broken_at,
                    description=f"Cascata iniziata da: {trigger.description[:50]}"
                )
                cascades.append(cascade)
        
        return cascades
    
    def _generate_lessons(
        self,
        timeline: List[TimelineEvent],
        cascades: List[CascadeChain]
    ) -> List[LessonLearned]:
        """Genera lessons learned dalla crisi"""
        
        lessons = []
        lesson_counter = 0
        
        # Lesson da detection tardiva
        high_severity_events = [e for e in timeline if e.severity >= 7]
        first_alert = next((e for e in timeline if e.event_type == EventType.ALERT), None)
        
        if high_severity_events and first_alert:
            first_critical = high_severity_events[0]
            if first_alert.timestamp > first_critical.timestamp:
                delay = first_alert.timestamp - first_critical.timestamp
                lesson_counter += 1
                lessons.append(LessonLearned(
                    lesson_id=f"lesson_{lesson_counter}",
                    category=LessonCategory.DETECTION,
                    title="Ritardo nel rilevamento anomalie critiche",
                    description=f"L'alert è stato generato {delay} dopo l'anomalia critica iniziale",
                    evidence=[first_critical.description, first_alert.description],
                    recommended_actions=[
                        "Abbassare soglie di alert per questa tipologia",
                        "Implementare detection proattiva",
                        "Aggiungere monitoring real-time"
                    ],
                    priority=1,
                    estimated_effort="medium",
                    potential_impact="Rilevamento 50% più veloce",
                    assigned_to=None,
                    deadline=None
                ))
        
        # Lesson da cascate lunghe
        for cascade in cascades:
            if len(cascade.cascade_events) >= 3:
                lesson_counter += 1
                lessons.append(LessonLearned(
                    lesson_id=f"lesson_{lesson_counter}",
                    category=LessonCategory.PREVENTION,
                    title=f"Effetto cascata con {len(cascade.cascade_events)} eventi",
                    description=f"Una cascata si è propagata per {cascade.duration}, causando impatto totale {cascade.total_impact}",
                    evidence=[e.description for e in cascade.cascade_events[:3]],
                    recommended_actions=[
                        "Implementare circuit breaker tra componenti",
                        f"Monitorare specificamente: {cascade.trigger_event.source}",
                        "Aggiungere isolation tra sistemi collegati"
                    ],
                    priority=2,
                    estimated_effort="high",
                    potential_impact="Riduzione propagazione 70%",
                    assigned_to=None,
                    deadline=None
                ))
        
        # Lesson da interventi umani efficaci
        human_interventions = [e for e in timeline if e.event_type == EventType.HUMAN_INTERVENTION]
        for intervention in human_interventions:
            lesson_counter += 1
            lessons.append(LessonLearned(
                lesson_id=f"lesson_{lesson_counter}",
                category=LessonCategory.PROCESS,
                title="Intervento umano efficace",
                description=f"L'intervento '{intervention.description[:50]}' ha contribuito alla risoluzione",
                evidence=[intervention.description],
                recommended_actions=[
                    "Documentare procedura per automatizzazione futura",
                    "Creare playbook per scenario simile",
                    "Valutare se automatizzabile"
                ],
                priority=3,
                estimated_effort="low",
                potential_impact="Risposta automatica in futuro",
                assigned_to=None,
                deadline=None
            ))
        
        # Lesson da recovery
        recovery_events = [e for e in timeline if e.event_type == EventType.RECOVERY]
        if recovery_events:
            recovery_time = recovery_events[-1].timestamp - timeline[0].timestamp
            lesson_counter += 1
            lessons.append(LessonLearned(
                lesson_id=f"lesson_{lesson_counter}",
                category=LessonCategory.RESPONSE,
                title=f"Tempo di recovery: {recovery_time}",
                description="Analisi del tempo necessario per il ripristino completo",
                evidence=[e.description for e in recovery_events],
                recommended_actions=[
                    "Definire RTO (Recovery Time Objective) se non presente",
                    "Verificare se recovery è in linea con SLA",
                    "Ottimizzare procedure di recovery"
                ],
                priority=2,
                estimated_effort="medium",
                potential_impact=f"Potenziale riduzione recovery del 30%",
                assigned_to=None,
                deadline=None
            ))
        
        return lessons
    
    def _propose_rules(
        self,
        timeline: List[TimelineEvent],
        cascades: List[CascadeChain],
        lessons: List[LessonLearned]
    ) -> List[RuleProposal]:
        """Propone nuove regole automatiche basate sulla crisi"""
        
        rules = []
        rule_counter = 0
        
        # Regola per trigger ricorrenti
        trigger_sources = defaultdict(int)
        for cascade in cascades:
            trigger_sources[cascade.trigger_event.source] += 1
        
        for source, count in trigger_sources.items():
            if count >= 1:  # Almeno 1 trigger da questa source
                rule_counter += 1
                rules.append(RuleProposal(
                    rule_id=f"rule_{rule_counter}",
                    trigger_condition=f"anomaly.source == '{source}' AND anomaly.severity >= 5",
                    action="alert_immediately AND prepare_isolation",
                    rationale=f"Il source '{source}' ha causato {count} cascate in questa crisi",
                    source_events=[c.trigger_event.event_id for c in cascades if c.trigger_event.source == source],
                    confidence=0.7 + (count * 0.1),
                    requires_approval=True,
                    auto_activate=False
                ))
        
        # Regola per pattern temporali
        events_by_hour = defaultdict(int)
        for event in timeline:
            events_by_hour[event.timestamp.hour] += 1
        
        peak_hour = max(events_by_hour, key=events_by_hour.get) if events_by_hour else None
        if peak_hour is not None:
            rule_counter += 1
            rules.append(RuleProposal(
                rule_id=f"rule_{rule_counter}",
                trigger_condition=f"current_hour == {peak_hour}",
                action="increase_monitoring_level",
                rationale=f"Picco di eventi osservato alle ore {peak_hour}",
                source_events=[e.event_id for e in timeline if e.timestamp.hour == peak_hour][:5],
                confidence=0.5,
                requires_approval=True,
                auto_activate=False
            ))
        
        # Regola da lesson di detection
        detection_lessons = [l for l in lessons if l.category == LessonCategory.DETECTION]
        for lesson in detection_lessons:
            rule_counter += 1
            rules.append(RuleProposal(
                rule_id=f"rule_{rule_counter}",
                trigger_condition="anomaly_pattern_match == true",
                action="lower_alert_threshold_temporarily",
                rationale=f"Basato su: {lesson.title}",
                source_events=lesson.evidence[:3],
                confidence=0.6,
                requires_approval=True,
                auto_activate=False
            ))
        
        return rules
    
    def _calculate_metrics(
        self,
        timeline: List[TimelineEvent],
        cascades: List[CascadeChain]
    ) -> Dict[str, Any]:
        """Calcola metriche della crisi"""
        
        if not timeline:
            return {}
        
        # Durata totale
        duration = timeline[-1].timestamp - timeline[0].timestamp
        
        # Severità media e picco
        severities = [e.severity for e in timeline]
        
        # Time to detect (primo alert dopo primo anomaly)
        first_anomaly = next((e for e in timeline if e.event_type == EventType.ANOMALY), None)
        first_alert = next((e for e in timeline if e.event_type == EventType.ALERT), None)
        
        ttd = None
        if first_anomaly and first_alert:
            ttd = first_alert.timestamp - first_anomaly.timestamp
        
        # Time to mitigate
        first_mitigation = next((e for e in timeline if e.event_type == EventType.MITIGATION), None)
        ttm = None
        if first_anomaly and first_mitigation:
            ttm = first_mitigation.timestamp - first_anomaly.timestamp
        
        # Time to resolve
        last_recovery = next((e for e in reversed(timeline) if e.event_type == EventType.RECOVERY), None)
        ttr = None
        if first_anomaly and last_recovery:
            ttr = last_recovery.timestamp - first_anomaly.timestamp
        
        return {
            'total_duration': str(duration),
            'total_events': len(timeline),
            'severity_avg': sum(severities) / len(severities),
            'severity_max': max(severities),
            'severity_min': min(severities),
            'cascade_count': len(cascades),
            'cascade_avg_length': sum(len(c.cascade_events) for c in cascades) / len(cascades) if cascades else 0,
            'time_to_detect': str(ttd) if ttd else 'N/A',
            'time_to_mitigate': str(ttm) if ttm else 'N/A',
            'time_to_resolve': str(ttr) if ttr else 'N/A',
            'human_interventions': sum(1 for e in timeline if e.event_type == EventType.HUMAN_INTERVENTION),
            'failures': sum(1 for e in timeline if e.event_type == EventType.FAILURE),
            'events_by_type': dict(defaultdict(int, {e.event_type.value: 1 for e in timeline})),
            'events_by_phase': dict(defaultdict(int, {e.phase.value: 1 for e in timeline}))
        }
    
    def _generate_summary(
        self,
        timeline: List[TimelineEvent],
        cascades: List[CascadeChain],
        lessons: List[LessonLearned],
        metrics: Dict[str, Any]
    ) -> str:
        """Genera summary testuale della crisi"""
        
        if not timeline:
            return "Nessun evento registrato."
        
        summary_parts = []
        
        # Overview
        summary_parts.append(f"Crisi durata {metrics.get('total_duration', 'N/A')} con {len(timeline)} eventi totali.")
        
        # Severità
        summary_parts.append(f"Severità massima raggiunta: {metrics.get('severity_max', 'N/A')}/10.")
        
        # Cascate
        if cascades:
            summary_parts.append(f"Rilevate {len(cascades)} catene di eventi a cascata.")
        
        # Tempi chiave
        if metrics.get('time_to_detect') != 'N/A':
            summary_parts.append(f"Tempo di detection: {metrics['time_to_detect']}.")
        if metrics.get('time_to_resolve') != 'N/A':
            summary_parts.append(f"Tempo di risoluzione: {metrics['time_to_resolve']}.")
        
        # Lessons
        if lessons:
            high_priority = [l for l in lessons if l.priority <= 2]
            summary_parts.append(f"Identificate {len(lessons)} lessons learned ({len(high_priority)} ad alta priorità).")
        
        return " ".join(summary_parts)
    
    def _generate_title(self, timeline: List[TimelineEvent]) -> str:
        """Genera titolo automatico"""
        
        if not timeline:
            return "Crisi senza eventi"
        
        # Trova evento principale
        main_event = max(timeline, key=lambda e: e.severity)
        
        return f"Crisi - {main_event.event_type.value}: {main_event.description[:40]}"
    
    def _determine_resolution_status(self, timeline: List[TimelineEvent]) -> str:
        """Determina stato risoluzione"""
        
        if not timeline:
            return "unknown"
        
        last_event = timeline[-1]
        
        if last_event.event_type == EventType.RECOVERY:
            return "resolved"
        elif last_event.event_type == EventType.MITIGATION:
            return "mitigated"
        elif last_event.phase == CrisisPhase.POST_CRISIS:
            return "closed"
        else:
            return "ongoing"
    
    def format_report(self, report: CrisisReport) -> str:
        """Formatta report per visualizzazione"""
        
        return f"""
# 🔄 Post-Crisis Report

## 📋 Overview
- **ID**: {report.crisis_id}
- **Titolo**: {report.title}
- **Inizio**: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Fine**: {report.end_time.strftime('%Y-%m-%d %H:%M:%S') if report.end_time else 'In corso'}
- **Durata**: {report.duration}
- **Stato**: {report.resolution_status}
- **Severità Picco**: {report.severity_peak}/10

---

## 📊 Summary
{report.summary}

---

## 📈 Metriche Chiave
| Metrica | Valore |
|---------|--------|
| Eventi Totali | {report.metrics.get('total_events', 'N/A')} |
| Severità Media | {report.metrics.get('severity_avg', 0):.1f} |
| Cascate Rilevate | {report.metrics.get('cascade_count', 0)} |
| Time to Detect | {report.metrics.get('time_to_detect', 'N/A')} |
| Time to Resolve | {report.metrics.get('time_to_resolve', 'N/A')} |
| Interventi Umani | {report.metrics.get('human_interventions', 0)} |

---

## 📜 Timeline ({len(report.timeline)} eventi)
{chr(10).join(f"- [{e.timestamp.strftime('%H:%M:%S')}] [{e.phase.value}] {e.event_type.value}: {e.description[:60]}" for e in report.timeline[:10])}
{f'... e altri {len(report.timeline) - 10} eventi' if len(report.timeline) > 10 else ''}

---

## ⛓️ Cascate ({len(report.cascades)})
{chr(10).join(f"- **{c.chain_id}**: {len(c.cascade_events)} eventi, impatto totale {c.total_impact}" for c in report.cascades) or 'Nessuna cascata rilevata'}

---

## 📚 Lessons Learned ({len(report.lessons)})
{chr(10).join(f"### [{l.priority}] {l.title}{chr(10)}{l.description}{chr(10)}**Azioni**: {', '.join(l.recommended_actions[:2])}" for l in sorted(report.lessons, key=lambda x: x.priority)[:5]) or 'Nessuna lesson identificata'}

---

## 🔧 Regole Proposte ({len(report.rule_proposals)})
{chr(10).join(f"- **{r.rule_id}**: IF {r.trigger_condition[:40]}... THEN {r.action} (conf: {r.confidence:.0%})" for r in report.rule_proposals[:5]) or 'Nessuna regola proposta'}

---

*Report generato: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    def get_archived_crises(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Ritorna crisi archiviate"""
        return [
            {
                'crisis_id': c.crisis_id,
                'title': c.title,
                'start_time': c.start_time.isoformat(),
                'duration': str(c.duration),
                'severity_peak': c.severity_peak,
                'resolution_status': c.resolution_status,
                'lessons_count': len(c.lessons)
            }
            for c in self.crises_archive[-limit:]
        ]
    
    def get_all_lessons(self, category: Optional[LessonCategory] = None) -> List[LessonLearned]:
        """Ritorna tutte le lessons learned"""
        if category:
            return [l for l in self.lessons_database if l.category == category]
        return self.lessons_database
    
    def get_pending_rules(self) -> List[RuleProposal]:
        """Ritorna regole in attesa di approvazione"""
        return [r for r in self.proposed_rules if r.requires_approval]
