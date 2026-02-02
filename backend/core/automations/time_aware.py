"""
⏰ TIME-AWARE AUTOMATION
=========================
GIDEON è consapevole del tempo:
- Adatta risposte in base all'orario
- Considera urgenza e deadline
- Rispetta ritmi lavorativi
- Gestisce task time-sensitive

"So che sono le 23:00, vuoi davvero deployare adesso?"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import calendar

logger = logging.getLogger(__name__)


class TimeContext(Enum):
    """Contesto temporale"""
    WORK_HOURS = "work_hours"
    AFTER_HOURS = "after_hours"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    NIGHT = "night"
    EARLY_MORNING = "early_morning"


class UrgencyLevel(Enum):
    """Livelli di urgenza"""
    CRITICAL = 5    # Immediato
    HIGH = 4        # Ore
    MEDIUM = 3      # Oggi
    LOW = 2         # Questa settimana
    MINIMAL = 1     # Quando possibile


class TimeSensitivity(Enum):
    """Sensibilità temporale dell'azione"""
    TIME_CRITICAL = "critical"    # Deve essere fatto ORA
    TIME_SENSITIVE = "sensitive"  # Meglio presto
    TIME_NEUTRAL = "neutral"      # Indifferente
    TIME_FLEXIBLE = "flexible"    # Può aspettare


@dataclass
class TimeConstraint:
    """Vincolo temporale"""
    deadline: Optional[datetime]
    not_before: Optional[datetime]
    not_after: Optional[datetime]
    preferred_time: Optional[dt_time]
    avoid_times: List[str]  # ["night", "weekend"]
    timezone: str = "Europe/Rome"


@dataclass
class TimeAwareDecision:
    """Decisione time-aware"""
    action: str
    original_timing: str
    recommended_timing: str
    warnings: List[str]
    urgency: UrgencyLevel
    sensitivity: TimeSensitivity
    can_proceed: bool
    reason: str


@dataclass
class WorkSchedule:
    """Orario di lavoro"""
    work_start: dt_time = field(default_factory=lambda: dt_time(9, 0))
    work_end: dt_time = field(default_factory=lambda: dt_time(18, 0))
    lunch_start: dt_time = field(default_factory=lambda: dt_time(12, 30))
    lunch_end: dt_time = field(default_factory=lambda: dt_time(14, 0))
    work_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Lun-Ven
    holidays: List[datetime] = field(default_factory=list)


class TimeAwareAutomation:
    """
    Sistema di automazione time-aware.
    
    Considera il tempo in ogni decisione:
    - Orario lavorativo vs personale
    - Urgenza e deadline
    - Ritmi circadiani
    - Calendario e festività
    """
    
    # Azioni rischiose fuori orario
    RISKY_AFTER_HOURS = [
        'deploy', 'release', 'publish',
        'delete', 'migrate', 'update_prod',
        'send_broadcast', 'commit_main'
    ]
    
    # Azioni da evitare di notte
    AVOID_AT_NIGHT = [
        'email_client', 'notification',
        'call', 'meeting', 'presentation'
    ]
    
    def __init__(self, schedule: Optional[WorkSchedule] = None):
        self.schedule = schedule or WorkSchedule()
        self.timezone = "Europe/Rome"
        self.action_history: List[Dict[str, Any]] = []
        self.deadline_tracker: Dict[str, datetime] = {}
        
    def get_time_context(
        self,
        timestamp: Optional[datetime] = None
    ) -> TimeContext:
        """Determina contesto temporale"""
        
        dt = timestamp or datetime.now()
        
        # Check holiday
        if dt.date() in [h.date() for h in self.schedule.holidays]:
            return TimeContext.HOLIDAY
        
        # Check weekend
        if dt.weekday() not in self.schedule.work_days:
            return TimeContext.WEEKEND
        
        # Check time of day
        current_time = dt.time()
        
        if current_time < dt_time(6, 0):
            return TimeContext.NIGHT
        elif current_time < dt_time(8, 0):
            return TimeContext.EARLY_MORNING
        elif self.schedule.work_start <= current_time <= self.schedule.work_end:
            return TimeContext.WORK_HOURS
        else:
            return TimeContext.AFTER_HOURS
    
    async def evaluate_action(
        self,
        action: str,
        constraint: Optional[TimeConstraint] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TimeAwareDecision:
        """
        Valuta un'azione considerando il tempo.
        """
        
        now = datetime.now()
        time_ctx = self.get_time_context(now)
        
        warnings = []
        can_proceed = True
        reason = "OK"
        
        # Valuta urgenza
        urgency = self._determine_urgency(constraint, context)
        sensitivity = self._determine_sensitivity(action, constraint)
        
        # Check azioni rischiose fuori orario
        if any(r in action.lower() for r in self.RISKY_AFTER_HOURS):
            if time_ctx in [TimeContext.AFTER_HOURS, TimeContext.WEEKEND, TimeContext.NIGHT]:
                warnings.append(f"⚠️ Azione rischiosa fuori orario lavorativo ({time_ctx.value})")
                if urgency.value < UrgencyLevel.HIGH.value:
                    can_proceed = False
                    reason = "Azione rischiosa consigliata solo in orario lavorativo"
        
        # Check azioni notturne
        if any(a in action.lower() for a in self.AVOID_AT_NIGHT):
            if time_ctx == TimeContext.NIGHT:
                warnings.append("🌙 Azione sconsigliata di notte")
                can_proceed = urgency == UrgencyLevel.CRITICAL
                reason = "Azione non appropriata per l'orario notturno"
        
        # Check deadline
        if constraint and constraint.deadline:
            time_to_deadline = constraint.deadline - now
            if time_to_deadline < timedelta(0):
                warnings.append("❌ Deadline superata!")
                urgency = UrgencyLevel.CRITICAL
            elif time_to_deadline < timedelta(hours=1):
                warnings.append(f"⏰ Deadline tra {time_to_deadline}!")
                urgency = UrgencyLevel.CRITICAL
            elif time_to_deadline < timedelta(hours=4):
                warnings.append(f"⏰ Deadline vicina: {time_to_deadline}")
                if urgency.value < UrgencyLevel.HIGH.value:
                    urgency = UrgencyLevel.HIGH
        
        # Check not_before
        if constraint and constraint.not_before and now < constraint.not_before:
            can_proceed = False
            reason = f"Non eseguibile prima di {constraint.not_before.strftime('%H:%M')}"
            warnings.append(f"⏳ Attendere fino a {constraint.not_before.strftime('%H:%M')}")
        
        # Check not_after
        if constraint and constraint.not_after and now > constraint.not_after:
            can_proceed = False
            reason = f"Finestra temporale chiusa dopo {constraint.not_after.strftime('%H:%M')}"
            warnings.append("❌ Finestra temporale scaduta")
        
        # Check avoid_times
        if constraint and constraint.avoid_times:
            if time_ctx.value in constraint.avoid_times:
                warnings.append(f"⚠️ Momento sconsigliato: {time_ctx.value}")
                if urgency.value < UrgencyLevel.HIGH.value:
                    can_proceed = False
                    reason = f"Evitare esecuzione durante: {time_ctx.value}"
        
        # Determina timing raccomandato
        recommended = self._recommend_timing(action, time_ctx, constraint, urgency)
        
        return TimeAwareDecision(
            action=action,
            original_timing="now",
            recommended_timing=recommended,
            warnings=warnings,
            urgency=urgency,
            sensitivity=sensitivity,
            can_proceed=can_proceed,
            reason=reason if not can_proceed or warnings else "Procedi normalmente"
        )
    
    def _determine_urgency(
        self,
        constraint: Optional[TimeConstraint],
        context: Optional[Dict[str, Any]]
    ) -> UrgencyLevel:
        """Determina urgenza dell'azione"""
        
        # Urgenza esplicita nel contesto
        if context and 'urgency' in context:
            urgency_map = {
                'critical': UrgencyLevel.CRITICAL,
                'high': UrgencyLevel.HIGH,
                'medium': UrgencyLevel.MEDIUM,
                'low': UrgencyLevel.LOW
            }
            return urgency_map.get(context['urgency'], UrgencyLevel.MEDIUM)
        
        # Basata su deadline
        if constraint and constraint.deadline:
            time_left = constraint.deadline - datetime.now()
            if time_left < timedelta(hours=1):
                return UrgencyLevel.CRITICAL
            elif time_left < timedelta(hours=4):
                return UrgencyLevel.HIGH
            elif time_left < timedelta(days=1):
                return UrgencyLevel.MEDIUM
        
        return UrgencyLevel.MEDIUM
    
    def _determine_sensitivity(
        self,
        action: str,
        constraint: Optional[TimeConstraint]
    ) -> TimeSensitivity:
        """Determina sensibilità temporale"""
        
        action_lower = action.lower()
        
        # Azioni time-critical
        if any(t in action_lower for t in ['emergency', 'critical', 'urgent']):
            return TimeSensitivity.TIME_CRITICAL
        
        # Azioni time-sensitive
        if any(t in action_lower for t in ['deploy', 'release', 'meeting', 'call']):
            return TimeSensitivity.TIME_SENSITIVE
        
        # Azioni flessibili
        if any(t in action_lower for t in ['report', 'document', 'review']):
            return TimeSensitivity.TIME_FLEXIBLE
        
        return TimeSensitivity.TIME_NEUTRAL
    
    def _recommend_timing(
        self,
        action: str,
        current_context: TimeContext,
        constraint: Optional[TimeConstraint],
        urgency: UrgencyLevel
    ) -> str:
        """Raccomanda timing ottimale"""
        
        now = datetime.now()
        
        # Urgenza critica = ora
        if urgency == UrgencyLevel.CRITICAL:
            return "immediately"
        
        # Azione rischiosa e fuori orario
        if any(r in action.lower() for r in self.RISKY_AFTER_HOURS):
            if current_context != TimeContext.WORK_HOURS:
                # Prossimo orario lavorativo
                next_work = self._next_work_time(now)
                return f"schedule for {next_work.strftime('%Y-%m-%d %H:%M')}"
        
        # Usa tempo preferito se specificato
        if constraint and constraint.preferred_time:
            preferred_dt = datetime.combine(now.date(), constraint.preferred_time)
            if preferred_dt > now:
                return f"schedule for {preferred_dt.strftime('%H:%M')}"
        
        # Default basato su urgenza
        if urgency == UrgencyLevel.HIGH:
            return "within 2 hours"
        elif urgency == UrgencyLevel.MEDIUM:
            return "today"
        else:
            return "this week"
    
    def _next_work_time(self, from_dt: datetime) -> datetime:
        """Calcola prossimo orario lavorativo"""
        
        dt = from_dt
        
        # Se oggi è giorno lavorativo e prima dell'inizio
        if dt.weekday() in self.schedule.work_days:
            work_start_today = datetime.combine(dt.date(), self.schedule.work_start)
            if dt < work_start_today:
                return work_start_today
        
        # Altrimenti, prossimo giorno lavorativo
        dt = dt + timedelta(days=1)
        while dt.weekday() not in self.schedule.work_days:
            dt = dt + timedelta(days=1)
        
        return datetime.combine(dt.date(), self.schedule.work_start)
    
    def set_deadline(
        self,
        task_id: str,
        deadline: datetime
    ):
        """Imposta deadline per un task"""
        self.deadline_tracker[task_id] = deadline
        logger.info(f"⏰ Deadline impostata: {task_id} -> {deadline}")
    
    def check_deadlines(self) -> List[Dict[str, Any]]:
        """Check deadline imminenti"""
        
        now = datetime.now()
        alerts = []
        
        for task_id, deadline in self.deadline_tracker.items():
            time_left = deadline - now
            
            if time_left < timedelta(0):
                alerts.append({
                    'task_id': task_id,
                    'status': 'overdue',
                    'deadline': deadline.isoformat(),
                    'overdue_by': str(-time_left)
                })
            elif time_left < timedelta(hours=1):
                alerts.append({
                    'task_id': task_id,
                    'status': 'critical',
                    'deadline': deadline.isoformat(),
                    'time_left': str(time_left)
                })
            elif time_left < timedelta(hours=4):
                alerts.append({
                    'task_id': task_id,
                    'status': 'urgent',
                    'deadline': deadline.isoformat(),
                    'time_left': str(time_left)
                })
        
        return alerts
    
    def add_holiday(self, date: datetime):
        """Aggiunge festività"""
        self.schedule.holidays.append(date)
    
    def get_status(self) -> Dict[str, Any]:
        """Ritorna status time-aware"""
        
        now = datetime.now()
        context = self.get_time_context(now)
        
        return {
            'current_time': now.isoformat(),
            'time_context': context.value,
            'is_work_hours': context == TimeContext.WORK_HOURS,
            'work_schedule': {
                'start': self.schedule.work_start.isoformat(),
                'end': self.schedule.work_end.isoformat(),
                'work_days': self.schedule.work_days
            },
            'active_deadlines': len(self.deadline_tracker),
            'urgent_deadlines': len([
                d for d in self.deadline_tracker.values()
                if d - now < timedelta(hours=4)
            ])
        }
    
    def format_decision(self, decision: TimeAwareDecision) -> str:
        """Formatta decisione per visualizzazione"""
        
        urgency_emoji = {
            UrgencyLevel.CRITICAL: '🚨',
            UrgencyLevel.HIGH: '⚡',
            UrgencyLevel.MEDIUM: '📌',
            UrgencyLevel.LOW: '📝',
            UrgencyLevel.MINIMAL: '💤'
        }
        
        emoji = urgency_emoji.get(decision.urgency, '📋')
        proceed_emoji = '✅' if decision.can_proceed else '🛑'
        
        return f"""
## ⏰ Time-Aware Decision

{proceed_emoji} **{decision.action}**

- **Urgenza**: {emoji} {decision.urgency.name}
- **Sensibilità**: {decision.sensitivity.value}
- **Timing consigliato**: {decision.recommended_timing}

### Status
**Può procedere**: {'Sì' if decision.can_proceed else 'No'}
**Motivo**: {decision.reason}

### Warnings
{chr(10).join(f"- {w}" for w in decision.warnings) or '✅ Nessun warning'}
"""
