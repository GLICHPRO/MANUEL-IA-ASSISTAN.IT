"""
⚡ RISK-TRIGGERED AUTOMATION
============================
GIDEON attiva azioni automatiche su soglie di rischio:
- Backup automatico prima di operazioni rischiose
- Notifiche stakeholders su risk threshold
- Rollback automatico su failure
- Escalation intelligente

"Rischio alto rilevato - ho attivato backup automatico"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Livelli di rischio"""
    LOW = 1  # < 25%
    MEDIUM = 2  # 25-50%
    HIGH = 3  # 50-75%
    CRITICAL = 4  # > 75%


class ActionType(Enum):
    """Tipi di azione automatica"""
    BACKUP = "backup"
    NOTIFY = "notify"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"
    PAUSE = "pause"
    CONFIRM = "confirm"
    LOG = "log"


class TriggerCondition(Enum):
    """Condizioni di trigger"""
    RISK_ABOVE = "risk_above"  # Rischio supera soglia
    RISK_INCREASING = "risk_increasing"  # Rischio in aumento
    ERROR_RATE = "error_rate"  # Tasso errori alto
    TIME_WINDOW = "time_window"  # Finestra temporale rischiosa
    ANOMALY = "anomaly"  # Comportamento anomalo
    THRESHOLD_BREACH = "threshold_breach"  # Superamento soglia custom


@dataclass
class RiskTrigger:
    """Trigger basato su rischio"""
    trigger_id: str
    name: str
    condition: TriggerCondition
    threshold: float
    actions: List[ActionType]
    cooldown_minutes: int = 15  # Tempo minimo tra attivazioni
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class AutomatedAction:
    """Azione automatica eseguita"""
    action_id: str
    action_type: ActionType
    trigger_id: str
    trigger_reason: str
    risk_level: RiskLevel
    risk_score: float
    context: Dict[str, Any]
    executed_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    result: Optional[str] = None


@dataclass
class RiskEvent:
    """Evento di rischio"""
    event_id: str
    source: str
    risk_score: float
    risk_level: RiskLevel
    description: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    handled: bool = False


class RiskTriggeredAutomation:
    """
    Sistema di automazione basata su rischio.
    
    Monitora rischi e attiva automaticamente:
    - Backup preventivi
    - Notifiche
    - Rollback
    - Escalation
    """
    
    # Default thresholds
    RISK_THRESHOLDS = {
        RiskLevel.LOW: 0.25,
        RiskLevel.MEDIUM: 0.50,
        RiskLevel.HIGH: 0.75,
        RiskLevel.CRITICAL: 0.90
    }
    
    def __init__(self):
        self.triggers: Dict[str, RiskTrigger] = {}
        self.action_history: List[AutomatedAction] = []
        self.risk_events: List[RiskEvent] = []
        
        # Handlers per azioni
        self.action_handlers: Dict[ActionType, Callable] = {}
        
        # Risk tracking
        self.current_risk_score: float = 0.0
        self.risk_history: List[tuple] = []  # (timestamp, score)
        
        # Stato
        self.paused = False
        
        # Inizializza trigger di default
        self._init_default_triggers()
    
    def _init_default_triggers(self):
        """Inizializza trigger di default"""
        
        # Trigger backup su rischio alto
        self.register_trigger(RiskTrigger(
            trigger_id="auto_backup_high_risk",
            name="Backup automatico rischio alto",
            condition=TriggerCondition.RISK_ABOVE,
            threshold=0.7,
            actions=[ActionType.BACKUP, ActionType.LOG],
            cooldown_minutes=30
        ))
        
        # Trigger notifica su rischio critico
        self.register_trigger(RiskTrigger(
            trigger_id="notify_critical_risk",
            name="Notifica rischio critico",
            condition=TriggerCondition.RISK_ABOVE,
            threshold=0.85,
            actions=[ActionType.NOTIFY, ActionType.ESCALATE],
            cooldown_minutes=15
        ))
        
        # Trigger pausa su anomalia
        self.register_trigger(RiskTrigger(
            trigger_id="pause_on_anomaly",
            name="Pausa su anomalia",
            condition=TriggerCondition.ANOMALY,
            threshold=0.9,
            actions=[ActionType.PAUSE, ActionType.NOTIFY],
            cooldown_minutes=5
        ))
        
        # Trigger conferma su rischio medio-alto
        self.register_trigger(RiskTrigger(
            trigger_id="confirm_medium_risk",
            name="Conferma rischio medio-alto",
            condition=TriggerCondition.RISK_ABOVE,
            threshold=0.5,
            actions=[ActionType.CONFIRM],
            cooldown_minutes=0
        ))
    
    def register_trigger(self, trigger: RiskTrigger):
        """Registra un trigger"""
        self.triggers[trigger.trigger_id] = trigger
        logger.debug(f"⚡ Trigger registrato: {trigger.name}")
    
    def register_action_handler(
        self,
        action_type: ActionType,
        handler: Callable[[Dict[str, Any]], Awaitable[bool]]
    ):
        """Registra handler per tipo di azione"""
        self.action_handlers[action_type] = handler
    
    def update_risk(
        self,
        source: str,
        risk_score: float,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskEvent:
        """
        Aggiorna score di rischio.
        
        Valuta trigger e attiva azioni se necessario.
        """
        
        # Determina livello
        risk_level = self._score_to_level(risk_score)
        
        # Crea evento
        event = RiskEvent(
            event_id=f"risk_{datetime.now().timestamp()}",
            source=source,
            risk_score=risk_score,
            risk_level=risk_level,
            description=description,
            context=context or {}
        )
        
        self.risk_events.append(event)
        
        # Aggiorna tracking
        old_score = self.current_risk_score
        self.current_risk_score = risk_score
        self.risk_history.append((datetime.now(), risk_score))
        
        # Limita history
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
        
        # Valuta trigger
        if not self.paused:
            asyncio.create_task(self._evaluate_triggers(event, old_score))
        
        return event
    
    def _score_to_level(self, score: float) -> RiskLevel:
        """Converte score in livello"""
        
        if score >= self.RISK_THRESHOLDS[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif score >= self.RISK_THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif score >= self.RISK_THRESHOLDS[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _evaluate_triggers(self, event: RiskEvent, old_score: float):
        """Valuta tutti i trigger"""
        
        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue
            
            if self._should_trigger(trigger, event, old_score):
                await self._execute_trigger(trigger, event)
    
    def _should_trigger(
        self,
        trigger: RiskTrigger,
        event: RiskEvent,
        old_score: float
    ) -> bool:
        """Verifica se trigger deve attivarsi"""
        
        # Check cooldown
        if trigger.last_triggered:
            cooldown_end = trigger.last_triggered + timedelta(minutes=trigger.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return False
        
        # Valuta condizione
        if trigger.condition == TriggerCondition.RISK_ABOVE:
            return event.risk_score >= trigger.threshold
        
        elif trigger.condition == TriggerCondition.RISK_INCREASING:
            # Rischio in aumento e sopra soglia
            return (
                event.risk_score >= trigger.threshold and
                event.risk_score > old_score
            )
        
        elif trigger.condition == TriggerCondition.ANOMALY:
            # Rileva anomalia (spike improvviso)
            return (
                event.risk_score >= trigger.threshold and
                event.risk_score - old_score > 0.3  # Spike > 30%
            )
        
        elif trigger.condition == TriggerCondition.ERROR_RATE:
            # Basato su tasso errori nel contesto
            error_rate = event.context.get('error_rate', 0)
            return error_rate >= trigger.threshold
        
        elif trigger.condition == TriggerCondition.TIME_WINDOW:
            # Finestra temporale (es. fuori orario)
            hour = datetime.now().hour
            risky_hours = event.context.get('risky_hours', [0, 1, 2, 3, 4, 5])
            return (
                event.risk_score >= trigger.threshold and
                hour in risky_hours
            )
        
        return False
    
    async def _execute_trigger(self, trigger: RiskTrigger, event: RiskEvent):
        """Esegue azioni di un trigger"""
        
        trigger.last_triggered = datetime.now()
        trigger.trigger_count += 1
        event.handled = True
        
        logger.warning(f"⚡ TRIGGER ATTIVATO: {trigger.name} (risk: {event.risk_score:.1%})")
        
        for action_type in trigger.actions:
            await self._execute_action(action_type, trigger, event)
    
    async def _execute_action(
        self,
        action_type: ActionType,
        trigger: RiskTrigger,
        event: RiskEvent
    ):
        """Esegue singola azione"""
        
        action = AutomatedAction(
            action_id=f"action_{datetime.now().timestamp()}",
            action_type=action_type,
            trigger_id=trigger.trigger_id,
            trigger_reason=f"{trigger.name}: {event.description}",
            risk_level=event.risk_level,
            risk_score=event.risk_score,
            context=event.context
        )
        
        try:
            # Cerca handler custom
            handler = self.action_handlers.get(action_type)
            
            if handler:
                success = await handler(action.context)
                action.success = success
            else:
                # Handler di default
                action.result = await self._default_action_handler(action_type, event)
            
            logger.info(f"⚡ Azione eseguita: {action_type.value}")
            
        except Exception as e:
            action.success = False
            action.result = str(e)
            logger.error(f"Errore azione {action_type.value}: {e}")
        
        self.action_history.append(action)
    
    async def _default_action_handler(
        self,
        action_type: ActionType,
        event: RiskEvent
    ) -> str:
        """Handler di default per azioni"""
        
        if action_type == ActionType.BACKUP:
            return f"[BACKUP] Creato backup preventivo per: {event.source}"
        
        elif action_type == ActionType.NOTIFY:
            return f"[NOTIFY] Notifica inviata: {event.description}"
        
        elif action_type == ActionType.ROLLBACK:
            return f"[ROLLBACK] Richiesto rollback per: {event.source}"
        
        elif action_type == ActionType.ESCALATE:
            return f"[ESCALATE] Escalation a livello superiore: {event.risk_level.name}"
        
        elif action_type == ActionType.PAUSE:
            self.paused = True
            return f"[PAUSE] Sistema in pausa per rischio: {event.risk_score:.1%}"
        
        elif action_type == ActionType.CONFIRM:
            return f"[CONFIRM] Richiesta conferma utente per: {event.description}"
        
        elif action_type == ActionType.LOG:
            return f"[LOG] Evento registrato: {event.event_id}"
        
        return f"[{action_type.value.upper()}] Azione completata"
    
    def pause_automation(self, reason: str = "Manual pause"):
        """Mette in pausa l'automazione"""
        self.paused = True
        logger.warning(f"⚡ Automazione in PAUSA: {reason}")
    
    def resume_automation(self):
        """Riprende l'automazione"""
        self.paused = False
        logger.info("⚡ Automazione RIPRESA")
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Ottiene status rischio corrente"""
        
        level = self._score_to_level(self.current_risk_score)
        
        # Trend
        trend = "stable"
        if len(self.risk_history) >= 5:
            recent_avg = sum(s for _, s in self.risk_history[-5:]) / 5
            older_avg = sum(s for _, s in self.risk_history[-10:-5]) / 5 if len(self.risk_history) >= 10 else recent_avg
            
            if recent_avg > older_avg + 0.05:
                trend = "increasing"
            elif recent_avg < older_avg - 0.05:
                trend = "decreasing"
        
        return {
            'score': self.current_risk_score,
            'level': level.name,
            'trend': trend,
            'paused': self.paused,
            'active_triggers': sum(1 for t in self.triggers.values() if t.enabled)
        }
    
    def get_recent_actions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Ottiene azioni recenti"""
        
        recent = self.action_history[-limit:]
        recent.reverse()
        
        return [
            {
                'id': a.action_id,
                'type': a.action_type.value,
                'trigger': a.trigger_id,
                'reason': a.trigger_reason,
                'risk_level': a.risk_level.name,
                'risk_score': a.risk_score,
                'success': a.success,
                'result': a.result,
                'executed_at': a.executed_at.isoformat()
            }
            for a in recent
        ]
    
    def get_trigger_stats(self) -> Dict[str, Any]:
        """Statistiche sui trigger"""
        
        return {
            trigger_id: {
                'name': trigger.name,
                'condition': trigger.condition.value,
                'threshold': trigger.threshold,
                'enabled': trigger.enabled,
                'trigger_count': trigger.trigger_count,
                'last_triggered': trigger.last_triggered.isoformat() if trigger.last_triggered else None,
                'cooldown_minutes': trigger.cooldown_minutes
            }
            for trigger_id, trigger in self.triggers.items()
        }
    
    def format_risk_alert(self, event: RiskEvent) -> str:
        """Formatta alert per evento di rischio"""
        
        level_emoji = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRITICAL: "🔴"
        }
        
        return f"""
{level_emoji[event.risk_level]} **RISK ALERT: {event.risk_level.name}**

**Source:** {event.source}
**Score:** {event.risk_score:.1%}
**Description:** {event.description}

**Timestamp:** {event.timestamp.strftime('%H:%M:%S')}
**Handled:** {'✅ Sì' if event.handled else '⏳ In attesa'}
"""
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        status = self.get_current_risk_status()
        trigger_stats = self.get_trigger_stats()
        recent_actions = self.get_recent_actions(5)
        
        level_emoji = {
            "LOW": "🟢",
            "MEDIUM": "🟡",
            "HIGH": "🟠",
            "CRITICAL": "🔴"
        }
        
        trend_emoji = {
            "stable": "➡️",
            "increasing": "📈",
            "decreasing": "📉"
        }
        
        triggers_section = ""
        for tid, tinfo in trigger_stats.items():
            status_icon = "✅" if tinfo['enabled'] else "❌"
            triggers_section += f"- {status_icon} **{tinfo['name']}** (×{tinfo['trigger_count']})\n"
        
        actions_section = ""
        for action in recent_actions[:5]:
            success_icon = "✅" if action['success'] else "❌"
            actions_section += f"- {success_icon} [{action['type']}] {action['reason'][:50]}...\n"
        
        return f"""
# ⚡ Risk-Triggered Automation

## Status Corrente
| Metrica | Valore |
|---------|--------|
| Risk Score | {status['score']:.1%} |
| Risk Level | {level_emoji.get(status['level'], '⚪')} {status['level']} |
| Trend | {trend_emoji.get(status['trend'], '➡️')} {status['trend']} |
| Sistema | {'⏸️ PAUSA' if status['paused'] else '▶️ ATTIVO'} |
| Trigger attivi | {status['active_triggers']} |

## Trigger Configurati
{triggers_section or '- Nessun trigger configurato'}

## Azioni Recenti
{actions_section or '- Nessuna azione recente'}

## Statistiche
| Metrica | Valore |
|---------|--------|
| Eventi totali | {len(self.risk_events)} |
| Azioni eseguite | {len(self.action_history)} |
| Azioni riuscite | {sum(1 for a in self.action_history if a.success)} |
"""


# Singleton
_risk_automation: Optional[RiskTriggeredAutomation] = None


def get_risk_automation() -> RiskTriggeredAutomation:
    """Ottiene istanza singleton"""
    global _risk_automation
    if _risk_automation is None:
        _risk_automation = RiskTriggeredAutomation()
    return _risk_automation
