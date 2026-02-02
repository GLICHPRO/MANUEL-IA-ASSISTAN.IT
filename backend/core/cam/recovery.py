"""
🔄 CAM RECOVERY LAYER
======================
Recupero e apprendimento post-crisi.

Componenti:
- CrisisTimelineReconstructor: Ricostruisce timeline crisi
- LessonsExtractor: Estrae lezioni apprese
- GradualPowerRestore: Ripristino graduale autonomia

"Ogni crisi è un'opportunità di apprendimento"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipi di evento nella timeline"""
    CRISIS_START = "crisis_start"
    LEVEL_CHANGE = "level_change"
    DECISION = "decision"
    ACTION = "action"
    ALERT = "alert"
    HUMAN_INPUT = "human_input"
    SYSTEM_RESPONSE = "system_response"
    ERROR = "error"
    RECOVERY_START = "recovery_start"
    CRISIS_END = "crisis_end"


class LessonCategory(Enum):
    """Categorie lezioni apprese"""
    DETECTION = "detection"  # Come rilevare meglio
    RESPONSE = "response"  # Come rispondere meglio
    COMMUNICATION = "communication"  # Come comunicare meglio
    DECISION = "decision"  # Come decidere meglio
    PREVENTION = "prevention"  # Come prevenire
    PROCESS = "process"  # Miglioramenti processo


class RestorePhase(Enum):
    """Fasi di ripristino"""
    STABILIZATION = "stabilization"  # Stabilizzazione
    MONITORING = "monitoring"  # Monitoraggio intensivo
    GRADUAL_RESTORE = "gradual_restore"  # Ripristino graduale
    NORMAL_OPS = "normal_ops"  # Operazioni normali


@dataclass
class TimelineEvent:
    """Evento nella timeline"""
    id: str
    event_type: EventType
    timestamp: datetime
    description: str
    actor: str  # Chi ha causato l'evento
    impact: str  # Impatto
    data: Dict[str, Any] = field(default_factory=dict)
    crisis_level_at_time: Optional[str] = None


@dataclass
class Lesson:
    """Lezione appresa"""
    id: str
    category: LessonCategory
    title: str
    description: str
    evidence: List[str]
    recommendations: List[str]
    priority: int  # 1-5
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RestoreCheckpoint:
    """Checkpoint di ripristino"""
    phase: RestorePhase
    autonomy_level: float  # 0-1
    features_enabled: List[str]
    conditions_to_advance: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================
# CRISIS TIMELINE RECONSTRUCTOR
# ============================================================

class CrisisTimelineReconstructor:
    """
    Ricostruisce timeline completa della crisi.
    
    Per ogni crisi:
    - Cosa è successo
    - Quando
    - Chi ha fatto cosa
    - Conseguenze
    """
    
    def __init__(self):
        self.timelines: Dict[str, List[TimelineEvent]] = {}
        self.current_crisis_id: Optional[str] = None
    
    def start_crisis(
        self,
        crisis_id: str,
        trigger_description: str,
        initial_level: str
    ) -> str:
        """Avvia tracking nuova crisi"""
        
        self.current_crisis_id = crisis_id
        self.timelines[crisis_id] = []
        
        self.add_event(TimelineEvent(
            id=f"{crisis_id}_start",
            event_type=EventType.CRISIS_START,
            timestamp=datetime.now(),
            description=trigger_description,
            actor="system",
            impact="Crisi iniziata",
            crisis_level_at_time=initial_level
        ))
        
        logger.info(f"📊 Timeline tracking avviato per crisi {crisis_id}")
        return crisis_id
    
    def add_event(self, event: TimelineEvent):
        """Aggiunge evento alla timeline"""
        
        crisis_id = self.current_crisis_id
        if not crisis_id:
            logger.warning("Nessuna crisi attiva per aggiungere evento")
            return
        
        if crisis_id not in self.timelines:
            self.timelines[crisis_id] = []
        
        self.timelines[crisis_id].append(event)
    
    def record_level_change(
        self,
        from_level: str,
        to_level: str,
        reason: str
    ):
        """Registra cambio livello crisi"""
        
        self.add_event(TimelineEvent(
            id=f"{datetime.now().timestamp()}_level",
            event_type=EventType.LEVEL_CHANGE,
            timestamp=datetime.now(),
            description=f"Livello crisi: {from_level} → {to_level}",
            actor="system",
            impact=reason,
            data={'from': from_level, 'to': to_level},
            crisis_level_at_time=to_level
        ))
    
    def record_decision(
        self,
        decision_id: str,
        description: str,
        decision_maker: str,
        options_considered: List[str]
    ):
        """Registra decisione presa"""
        
        self.add_event(TimelineEvent(
            id=f"{decision_id}",
            event_type=EventType.DECISION,
            timestamp=datetime.now(),
            description=description,
            actor=decision_maker,
            impact="Decisione registrata",
            data={'options': options_considered}
        ))
    
    def record_action(
        self,
        action_id: str,
        description: str,
        executor: str,
        result: str
    ):
        """Registra azione eseguita"""
        
        self.add_event(TimelineEvent(
            id=f"{action_id}",
            event_type=EventType.ACTION,
            timestamp=datetime.now(),
            description=description,
            actor=executor,
            impact=result
        ))
    
    def end_crisis(
        self,
        resolution: str,
        final_level: str
    ):
        """Chiude timeline crisi"""
        
        self.add_event(TimelineEvent(
            id=f"{self.current_crisis_id}_end",
            event_type=EventType.CRISIS_END,
            timestamp=datetime.now(),
            description=resolution,
            actor="system",
            impact="Crisi conclusa",
            crisis_level_at_time=final_level
        ))
        
        logger.info(f"📊 Timeline crisi {self.current_crisis_id} completata")
        self.current_crisis_id = None
    
    def get_timeline(
        self,
        crisis_id: str
    ) -> List[TimelineEvent]:
        """Ottiene timeline crisi"""
        return self.timelines.get(crisis_id, [])
    
    def get_duration(self, crisis_id: str) -> Optional[timedelta]:
        """Calcola durata crisi"""
        
        events = self.get_timeline(crisis_id)
        
        start = next((e for e in events if e.event_type == EventType.CRISIS_START), None)
        end = next((e for e in events if e.event_type == EventType.CRISIS_END), None)
        
        if start and end:
            return end.timestamp - start.timestamp
        elif start:
            return datetime.now() - start.timestamp
        
        return None
    
    def format_timeline(self, crisis_id: str) -> str:
        """Formatta timeline per visualizzazione"""
        
        events = self.get_timeline(crisis_id)
        duration = self.get_duration(crisis_id)
        
        if not events:
            return "Timeline non disponibile"
        
        output = f"""
# 📊 Timeline Crisi: {crisis_id}

**Durata:** {duration if duration else 'In corso'}

## Eventi

| Tempo | Tipo | Descrizione | Attore | Impatto |
|-------|------|-------------|--------|---------|
"""
        
        start_time = events[0].timestamp
        
        for event in events:
            elapsed = event.timestamp - start_time
            minutes = int(elapsed.total_seconds() / 60)
            
            type_emoji = {
                EventType.CRISIS_START: "🔴",
                EventType.LEVEL_CHANGE: "📈",
                EventType.DECISION: "🎯",
                EventType.ACTION: "⚡",
                EventType.ALERT: "⚠️",
                EventType.HUMAN_INPUT: "👤",
                EventType.SYSTEM_RESPONSE: "🤖",
                EventType.ERROR: "❌",
                EventType.RECOVERY_START: "🔄",
                EventType.CRISIS_END: "✅"
            }
            
            output += f"| +{minutes}m | {type_emoji.get(event.event_type, '•')} {event.event_type.value} | {event.description[:40]} | {event.actor} | {event.impact[:30]} |\n"
        
        return output


# ============================================================
# LESSONS EXTRACTOR
# ============================================================

class LessonsExtractor:
    """
    Estrae lezioni apprese dalla crisi.
    
    Analizza:
    - Cosa ha funzionato
    - Cosa non ha funzionato
    - Cosa potevamo fare meglio
    """
    
    def __init__(self):
        self.lessons: Dict[str, List[Lesson]] = {}
    
    def extract_from_timeline(
        self,
        crisis_id: str,
        timeline: List[TimelineEvent]
    ) -> List[Lesson]:
        """Estrae lezioni da timeline"""
        
        lessons = []
        
        # Analizza pattern
        lessons.extend(self._analyze_detection(timeline))
        lessons.extend(self._analyze_response_time(timeline))
        lessons.extend(self._analyze_decisions(timeline))
        lessons.extend(self._analyze_errors(timeline))
        
        self.lessons[crisis_id] = lessons
        
        return lessons
    
    def _analyze_detection(
        self,
        timeline: List[TimelineEvent]
    ) -> List[Lesson]:
        """Analizza detection"""
        
        lessons = []
        
        # Trova eventi prima di crisis_start
        start_idx = next(
            (i for i, e in enumerate(timeline) 
             if e.event_type == EventType.CRISIS_START),
            0
        )
        
        # Ci sono stati segnali prima?
        pre_crisis_alerts = [
            e for e in timeline[:start_idx]
            if e.event_type == EventType.ALERT
        ]
        
        if not pre_crisis_alerts:
            lessons.append(Lesson(
                id=f"detection_{datetime.now().timestamp()}",
                category=LessonCategory.DETECTION,
                title="Nessun alert pre-crisi",
                description="La crisi non è stata anticipata da alert",
                evidence=["Nessun alert registrato prima di CRISIS_START"],
                recommendations=[
                    "Rivedere soglie di alerting",
                    "Considerare nuovi segnali deboli",
                    "Implementare early warning system"
                ],
                priority=4
            ))
        
        return lessons
    
    def _analyze_response_time(
        self,
        timeline: List[TimelineEvent]
    ) -> List[Lesson]:
        """Analizza tempi di risposta"""
        
        lessons = []
        
        # Trova tempo tra start e prima azione
        start = next(
            (e for e in timeline if e.event_type == EventType.CRISIS_START),
            None
        )
        first_action = next(
            (e for e in timeline if e.event_type == EventType.ACTION),
            None
        )
        
        if start and first_action:
            response_time = (first_action.timestamp - start.timestamp).total_seconds()
            
            if response_time > 300:  # > 5 minuti
                lessons.append(Lesson(
                    id=f"response_{datetime.now().timestamp()}",
                    category=LessonCategory.RESPONSE,
                    title="Tempo risposta elevato",
                    description=f"Prima azione dopo {response_time/60:.1f} minuti",
                    evidence=[
                        f"CRISIS_START: {start.timestamp}",
                        f"Prima ACTION: {first_action.timestamp}"
                    ],
                    recommendations=[
                        "Preparare playbook per risposte rapide",
                        "Automatizzare prime azioni difensive",
                        "Migliorare sistema di notifica"
                    ],
                    priority=3
                ))
        
        return lessons
    
    def _analyze_decisions(
        self,
        timeline: List[TimelineEvent]
    ) -> List[Lesson]:
        """Analizza decisioni"""
        
        lessons = []
        
        decisions = [e for e in timeline if e.event_type == EventType.DECISION]
        
        # Troppe decisioni in poco tempo?
        if len(decisions) > 5:
            time_span = decisions[-1].timestamp - decisions[0].timestamp
            if time_span.total_seconds() < 1800:  # 30 minuti
                lessons.append(Lesson(
                    id=f"decision_{datetime.now().timestamp()}",
                    category=LessonCategory.DECISION,
                    title="Molte decisioni in poco tempo",
                    description=f"{len(decisions)} decisioni in {time_span.total_seconds()/60:.0f} minuti",
                    evidence=[f"Decisione: {d.description[:30]}" for d in decisions[:5]],
                    recommendations=[
                        "Raggruppare decisioni correlate",
                        "Definire decision matrix pre-crisi",
                        "Delegare decisioni minori"
                    ],
                    priority=2
                ))
        
        return lessons
    
    def _analyze_errors(
        self,
        timeline: List[TimelineEvent]
    ) -> List[Lesson]:
        """Analizza errori"""
        
        lessons = []
        
        errors = [e for e in timeline if e.event_type == EventType.ERROR]
        
        if errors:
            lessons.append(Lesson(
                id=f"errors_{datetime.now().timestamp()}",
                category=LessonCategory.PROCESS,
                title=f"{len(errors)} errori durante crisi",
                description="Errori rilevati durante gestione crisi",
                evidence=[f"Errore: {e.description[:30]}" for e in errors],
                recommendations=[
                    "Analizzare root cause errori",
                    "Implementare controlli aggiuntivi",
                    "Migliorare error handling"
                ],
                priority=4
            ))
        
        return lessons
    
    def add_manual_lesson(
        self,
        crisis_id: str,
        lesson: Lesson
    ):
        """Aggiunge lezione manuale"""
        
        if crisis_id not in self.lessons:
            self.lessons[crisis_id] = []
        
        self.lessons[crisis_id].append(lesson)
    
    def get_lessons(
        self,
        crisis_id: str,
        category: Optional[LessonCategory] = None
    ) -> List[Lesson]:
        """Ottiene lezioni, opzionalmente filtrate"""
        
        all_lessons = self.lessons.get(crisis_id, [])
        
        if category:
            return [l for l in all_lessons if l.category == category]
        
        return all_lessons
    
    def format_lessons(self, crisis_id: str) -> str:
        """Formatta lezioni per visualizzazione"""
        
        lessons = self.get_lessons(crisis_id)
        
        if not lessons:
            return "Nessuna lezione estratta"
        
        output = f"""
# 📚 Lezioni Apprese - Crisi {crisis_id}

"""
        
        # Raggruppa per categoria
        by_category: Dict[LessonCategory, List[Lesson]] = defaultdict(list)
        for lesson in lessons:
            by_category[lesson.category].append(lesson)
        
        category_emoji = {
            LessonCategory.DETECTION: "🔍",
            LessonCategory.RESPONSE: "⚡",
            LessonCategory.COMMUNICATION: "💬",
            LessonCategory.DECISION: "🎯",
            LessonCategory.PREVENTION: "🛡️",
            LessonCategory.PROCESS: "⚙️"
        }
        
        for category, cat_lessons in by_category.items():
            output += f"## {category_emoji.get(category, '•')} {category.value.title()}\n\n"
            
            for lesson in sorted(cat_lessons, key=lambda l: l.priority, reverse=True):
                priority_stars = "⭐" * lesson.priority
                
                output += f"""
### {lesson.title} {priority_stars}

{lesson.description}

**Evidenze:**
{chr(10).join(f'- {e}' for e in lesson.evidence)}

**Raccomandazioni:**
{chr(10).join(f'- {r}' for r in lesson.recommendations)}

---
"""
        
        return output


# ============================================================
# GRADUAL POWER RESTORE
# ============================================================

class GradualPowerRestore:
    """
    Ripristina gradualmente autonomia dopo crisi.
    
    Fasi:
    1. Stabilizzazione
    2. Monitoraggio intensivo
    3. Ripristino graduale
    4. Operazioni normali
    """
    
    # Configurazione fasi
    PHASE_CONFIG = {
        RestorePhase.STABILIZATION: {
            'autonomy': 0.2,
            'features': ['read_only', 'monitoring', 'alerts'],
            'min_duration_minutes': 15,
            'conditions': ['no_new_crisis_signals', 'systems_stable']
        },
        RestorePhase.MONITORING: {
            'autonomy': 0.4,
            'features': ['read_only', 'monitoring', 'alerts', 'suggestions'],
            'min_duration_minutes': 30,
            'conditions': ['no_new_crisis_signals', 'systems_stable', 'human_approved']
        },
        RestorePhase.GRADUAL_RESTORE: {
            'autonomy': 0.7,
            'features': ['read_only', 'monitoring', 'alerts', 'suggestions', 'reversible_actions'],
            'min_duration_minutes': 60,
            'conditions': ['no_new_crisis_signals', 'systems_stable', 'performance_normal']
        },
        RestorePhase.NORMAL_OPS: {
            'autonomy': 1.0,
            'features': ['all'],
            'min_duration_minutes': 0,
            'conditions': []
        }
    }
    
    def __init__(self):
        self.current_phase = RestorePhase.STABILIZATION
        self.phase_history: List[RestoreCheckpoint] = []
        self.phase_start_time: Optional[datetime] = None
        self.conditions_met: Dict[str, bool] = {}
    
    def start_recovery(self):
        """Avvia processo di recovery"""
        
        self.current_phase = RestorePhase.STABILIZATION
        self.phase_start_time = datetime.now()
        self.conditions_met = {}
        
        self._record_checkpoint()
        
        logger.info(f"🔄 Recovery avviato - Fase: {self.current_phase.value}")
    
    def _record_checkpoint(self):
        """Registra checkpoint"""
        
        config = self.PHASE_CONFIG[self.current_phase]
        
        self.phase_history.append(RestoreCheckpoint(
            phase=self.current_phase,
            autonomy_level=config['autonomy'],
            features_enabled=config['features'],
            conditions_to_advance=config['conditions']
        ))
    
    def update_condition(self, condition: str, met: bool):
        """Aggiorna stato condizione"""
        
        self.conditions_met[condition] = met
        
        # Check se possiamo avanzare
        if self._can_advance():
            self.advance_phase()
    
    def _can_advance(self) -> bool:
        """Verifica se condizioni per avanzare sono soddisfatte"""
        
        if self.current_phase == RestorePhase.NORMAL_OPS:
            return False
        
        config = self.PHASE_CONFIG[self.current_phase]
        
        # Check durata minima
        if self.phase_start_time:
            elapsed = (datetime.now() - self.phase_start_time).total_seconds() / 60
            if elapsed < config['min_duration_minutes']:
                return False
        
        # Check condizioni
        for condition in config['conditions']:
            if not self.conditions_met.get(condition, False):
                return False
        
        return True
    
    def advance_phase(self) -> bool:
        """Avanza alla fase successiva"""
        
        phases = list(RestorePhase)
        current_idx = phases.index(self.current_phase)
        
        if current_idx >= len(phases) - 1:
            return False
        
        old_phase = self.current_phase
        self.current_phase = phases[current_idx + 1]
        self.phase_start_time = datetime.now()
        self.conditions_met = {}
        
        self._record_checkpoint()
        
        logger.info(f"🔄 Recovery avanzato: {old_phase.value} → {self.current_phase.value}")
        
        return True
    
    def get_current_config(self) -> Dict[str, Any]:
        """Ottiene configurazione fase corrente"""
        
        config = self.PHASE_CONFIG[self.current_phase]
        
        # Calcola tempo in fase
        time_in_phase = None
        if self.phase_start_time:
            time_in_phase = datetime.now() - self.phase_start_time
        
        return {
            'phase': self.current_phase.value,
            'autonomy_level': config['autonomy'],
            'features_enabled': config['features'],
            'time_in_phase': str(time_in_phase) if time_in_phase else None,
            'min_duration_minutes': config['min_duration_minutes'],
            'conditions_required': config['conditions'],
            'conditions_met': self.conditions_met
        }
    
    def get_autonomy_level(self) -> float:
        """Ottiene livello autonomia corrente"""
        return self.PHASE_CONFIG[self.current_phase]['autonomy']
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Verifica se feature è abilitata"""
        
        features = self.PHASE_CONFIG[self.current_phase]['features']
        return 'all' in features or feature in features
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        config = self.get_current_config()
        
        phase_emoji = {
            RestorePhase.STABILIZATION: "🔴",
            RestorePhase.MONITORING: "🟡",
            RestorePhase.GRADUAL_RESTORE: "🟢",
            RestorePhase.NORMAL_OPS: "✅"
        }
        
        # Progress bar
        phases = list(RestorePhase)
        current_idx = phases.index(self.current_phase)
        progress = "█" * (current_idx + 1) + "░" * (len(phases) - current_idx - 1)
        
        conditions_status = "\n".join(
            f"- {'✅' if self.conditions_met.get(c, False) else '⬜'} {c}"
            for c in config['conditions_required']
        )
        
        return f"""
# 🔄 Recovery Status

## Fase Corrente
{phase_emoji[self.current_phase]} **{self.current_phase.value.replace('_', ' ').title()}**

## Progresso
[{progress}] {current_idx + 1}/{len(phases)}

## Autonomia
**{config['autonomy_level']:.0%}**

## Features Abilitate
{', '.join(config['features_enabled'])}

## Tempo in Fase
{config['time_in_phase'] or 'Appena iniziato'}
(Minimo richiesto: {config['min_duration_minutes']} minuti)

## Condizioni per Avanzare
{conditions_status if conditions_status else 'Nessuna condizione richiesta'}
"""


# ============================================================
# UNIFIED RECOVERY LAYER
# ============================================================

class RecoveryLayer:
    """
    Recovery Layer unificato per CAM.
    
    "Ogni crisi è un'opportunità di apprendimento"
    """
    
    def __init__(self):
        self.timeline = CrisisTimelineReconstructor()
        self.lessons = LessonsExtractor()
        self.restore = GradualPowerRestore()
    
    def start_crisis_tracking(
        self,
        crisis_id: str,
        trigger: str,
        initial_level: str
    ):
        """Avvia tracking crisi"""
        self.timeline.start_crisis(crisis_id, trigger, initial_level)
    
    def end_crisis_and_recover(
        self,
        resolution: str,
        final_level: str
    ):
        """Termina crisi e avvia recovery"""
        
        # Chiudi timeline
        crisis_id = self.timeline.current_crisis_id
        self.timeline.end_crisis(resolution, final_level)
        
        # Estrai lezioni
        if crisis_id:
            timeline_events = self.timeline.get_timeline(crisis_id)
            self.lessons.extract_from_timeline(crisis_id, timeline_events)
        
        # Avvia recovery
        self.restore.start_recovery()
        
        logger.info("✅ Crisi terminata - Recovery avviato")
    
    def get_full_report(self, crisis_id: str) -> str:
        """Ottiene report completo post-crisi"""
        
        return f"""
{self.timeline.format_timeline(crisis_id)}

---

{self.lessons.format_lessons(crisis_id)}

---

{self.restore.format_status()}
"""


# Singleton
_recovery_layer: Optional[RecoveryLayer] = None


def get_recovery_layer() -> RecoveryLayer:
    """Ottiene istanza singleton"""
    global _recovery_layer
    if _recovery_layer is None:
        _recovery_layer = RecoveryLayer()
    return _recovery_layer
