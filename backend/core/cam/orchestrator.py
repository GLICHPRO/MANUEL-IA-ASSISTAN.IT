"""
🎯 CAM ORCHESTRATOR
====================
Classe principale Crisis Automation Mode.

Unisce tutti i 6 layer:
1. Detection - "Siamo in crisi?"
2. Control - "Governa la potenza"
3. Reasoning - "Massima potenza cognitiva"
4. Automation - "Automazione intelligente"
5. Human Interface - "Human-in-the-Loop"
6. Recovery - "Apprendimento post-crisi"

REGOLE ASSOLUTE:
❌ Nessuna decisione irreversibile automatica
❌ Nessuna azione senza spiegazione
❌ Nessuna escalation senza consenso umano
✅ Sempre audit trail
✅ Sempre possibilità di STOP
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
import asyncio
import uuid

# Import layers
from .detection import (
    CrisisSignalAggregator,
    CrisisLevel,
    CrisisSignal,
    SignalType,
    SignalSource,
    get_signal_aggregator
)
from .control import (
    ControlLayer,
    AutonomyClamp,
    TemporalGovernor,
    SafeStateEnforcer,
    ControlState,
    get_control_layer
)
from .reasoning import (
    ReasoningLayer,
    MultiPathReasoningEngine,
    UncertaintyMapper,
    NoActionIntelligence,
    get_reasoning_layer
)
from .automation import (
    AutomationLayer,
    CAMAutomationManager,
    PreDecisionProcessor,
    RiskTriggeredActions,
    get_automation_layer
)
from .human_interface import (
    HumanInterfaceLayer,
    CrisisUIMode,
    RoleAwareViews,
    CognitiveLoadMonitor,
    UserRole,
    get_human_interface_layer
)
from .recovery import (
    RecoveryLayer,
    CrisisTimelineReconstructor,
    LessonsExtractor,
    GradualPowerRestore,
    get_recovery_layer
)

logger = logging.getLogger(__name__)


class CAMStatus(Enum):
    """Status del Crisis Automation Mode"""
    INACTIVE = "inactive"  # Non attivo
    MONITORING = "monitoring"  # Monitoraggio attivo
    ENGAGED = "engaged"  # Crisi rilevata, in gestione
    RECOVERY = "recovery"  # Post-crisi, ripristino
    EMERGENCY_STOP = "emergency_stop"  # Stop emergenza


@dataclass
class CAMState:
    """Stato completo CAM"""
    status: CAMStatus
    crisis_level: CrisisLevel
    control_state: ControlState
    autonomy_level: float
    active_crisis_id: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CAMEvent:
    """Evento CAM per audit"""
    id: str
    event_type: str
    description: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CAMDecisionRequest:
    """Richiesta di decisione al CAM"""
    id: str
    context: str
    options: List[str]
    urgency: str  # low, medium, high, critical
    requires_human: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================
# CRISIS AUTOMATION MODE - MAIN CLASS
# ============================================================

class CrisisAutomationMode:
    """
    🎯 GIDEON Crisis Automation Mode (CAM)
    
    Sistema integrato per gestione crisi con:
    - Detection automatica
    - Control adattivo
    - Reasoning multi-path
    - Automation intelligente
    - Human interface adattiva
    - Recovery strutturato
    
    FILOSOFIA:
    - "Explain > Act" - Prima spiega, poi agisci
    - "Slow is Smooth, Smooth is Fast"
    - "Uncertainty is a Signal"
    - "Human-in-the-Loop sempre"
    - "Safe-State > Wrong-State"
    """
    
    VERSION = "1.0.0"
    
    def __init__(self):
        # Status
        self.status = CAMStatus.INACTIVE
        self.state_history: List[CAMState] = []
        
        # Event log (audit trail)
        self.event_log: List[CAMEvent] = []
        
        # Active crisis
        self.active_crisis_id: Optional[str] = None
        
        # Layers
        self.detection = get_signal_aggregator()
        self.control = get_control_layer()
        self.reasoning = get_reasoning_layer()
        self.automation = get_automation_layer()
        self.human_interface = get_human_interface_layer()
        self.recovery = get_recovery_layer()
        
        # Callbacks
        self._on_crisis_detected: List[Callable] = []
        self._on_status_change: List[Callable] = []
        self._on_human_needed: List[Callable] = []
        
        # Configuration
        self.config = {
            'auto_engage_threshold': CrisisLevel.SOFT,
            'require_human_for_actions': True,
            'max_autonomous_level': CrisisLevel.WATCH,
            'audit_all_events': True
        }
    
    # ==================== LIFECYCLE ====================
    
    async def initialize(self):
        """Inizializza CAM"""
        
        await self.automation.initialize()
        
        self.status = CAMStatus.MONITORING
        self._log_event("cam_initialized", "CAM inizializzato e in monitoraggio")
        
        logger.info("🎯 CAM inizializzato - Status: MONITORING")
    
    async def shutdown(self):
        """Shutdown CAM"""
        
        if self.active_crisis_id:
            self._log_event("shutdown_with_crisis", 
                          f"Shutdown con crisi attiva: {self.active_crisis_id}")
        
        self.status = CAMStatus.INACTIVE
        self._log_event("cam_shutdown", "CAM arrestato")
        
        logger.info("🎯 CAM arrestato")
    
    # ==================== DETECTION ====================
    
    async def process_signal(
        self,
        signal_type: SignalType,
        source: SignalSource,
        severity: float,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processa segnale in ingresso.
        
        Returns:
            Stato crisi aggiornato
        """
        
        # Crea segnale
        signal = CrisisSignal(
            signal_type=signal_type,
            source=source,
            severity=severity,
            description=description,
            metadata=metadata or {}
        )
        
        # Aggiungi a detection
        self.detection.add_signal(signal)
        
        # Valuta stato
        assessment = self.detection.evaluate()
        
        self._log_event("signal_processed", f"Segnale {signal_type.value}: {description[:50]}", {
            'severity': severity,
            'current_level': assessment.crisis_level.value
        })
        
        # Check se engaggiare
        if (assessment.crisis_level.value >= self.config['auto_engage_threshold'].value 
            and self.status == CAMStatus.MONITORING):
            await self._engage_crisis(assessment)
        
        # Update control se engaged
        if self.status == CAMStatus.ENGAGED:
            self.control.update_for_crisis(assessment.crisis_level.value)
        
        return {
            'crisis_level': assessment.crisis_level.value,
            'severity_score': assessment.severity_score,
            'recommendations': assessment.recommendations,
            'cam_status': self.status.value
        }
    
    # ==================== CRISIS MANAGEMENT ====================
    
    async def _engage_crisis(self, assessment):
        """Entra in modalità crisi"""
        
        self.active_crisis_id = f"crisis_{uuid.uuid4().hex[:8]}"
        self.status = CAMStatus.ENGAGED
        
        # Avvia tracking recovery
        self.recovery.start_crisis_tracking(
            self.active_crisis_id,
            assessment.recommendations[0] if assessment.recommendations else "Crisi rilevata",
            assessment.crisis_level.value
        )
        
        # Aggiorna control
        self.control.update_for_crisis(assessment.crisis_level.value)
        
        self._log_event("crisis_engaged", f"Crisi attivata: {self.active_crisis_id}", {
            'level': assessment.crisis_level.value,
            'score': assessment.severity_score
        })
        
        # Notify callbacks
        for callback in self._on_crisis_detected:
            try:
                await callback(self.active_crisis_id, assessment)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        logger.warning(f"🚨 CAM ENGAGED - Crisis ID: {self.active_crisis_id}")
    
    async def resolve_crisis(self, resolution: str):
        """Risolve crisi e avvia recovery"""
        
        if not self.active_crisis_id:
            return
        
        crisis_id = self.active_crisis_id
        
        # Termina e avvia recovery
        self.recovery.end_crisis_and_recover(
            resolution,
            self.detection.current_level.value
        )
        
        self.status = CAMStatus.RECOVERY
        
        self._log_event("crisis_resolved", f"Crisi {crisis_id} risolta: {resolution}")
        
        logger.info(f"✅ Crisi {crisis_id} risolta - Avvio recovery")
        
        self.active_crisis_id = None
    
    async def complete_recovery(self):
        """Completa recovery e torna a monitoring"""
        
        if self.status != CAMStatus.RECOVERY:
            return
        
        self.status = CAMStatus.MONITORING
        self.control.restore_defaults()
        
        self._log_event("recovery_complete", "Recovery completato, tornato a monitoring")
        
        logger.info("✅ Recovery completato - Tornato a MONITORING")
    
    # ==================== EMERGENCY STOP ====================
    
    def emergency_stop(self, reason: str = "Richiesta utente"):
        """
        🛑 EMERGENCY STOP
        
        Ferma immediatamente tutte le operazioni automatiche.
        """
        
        self.status = CAMStatus.EMERGENCY_STOP
        
        # Stop automation
        self.automation.emergency_stop(reason)
        
        # Force safe state
        self.control.safe_state.enter_safe_state("Emergency stop: " + reason)
        
        self._log_event("emergency_stop", f"EMERGENCY STOP: {reason}")
        
        logger.critical(f"🛑 EMERGENCY STOP ATTIVATO: {reason}")
    
    def resume_operations(self):
        """Riprende operazioni dopo stop"""
        
        if self.status != CAMStatus.EMERGENCY_STOP:
            return
        
        self.automation.resume()
        
        # Torna allo stato appropriato
        if self.active_crisis_id:
            self.status = CAMStatus.ENGAGED
        else:
            self.status = CAMStatus.MONITORING
        
        self._log_event("operations_resumed", "Operazioni riprese")
        
        logger.info("✅ Operazioni riprese")
    
    # ==================== REASONING ====================
    
    async def analyze_situation(
        self,
        problem: str,
        context: Dict[str, Any],
        known_facts: List[str],
        unknown_factors: List[str]
    ) -> Dict[str, Any]:
        """
        Analisi completa situazione con multi-path reasoning.
        
        Returns:
            Analisi con consensus, uncertainty map, no-action analysis
        """
        
        analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
        
        result = await self.reasoning.full_analysis(
            analysis_id,
            problem,
            context,
            known_facts,
            unknown_factors
        )
        
        self._log_event("analysis_performed", f"Analisi {analysis_id}", {
            'confidence': result['consensus']['confidence']
        })
        
        return result
    
    # ==================== DECISIONS ====================
    
    async def request_decision(
        self,
        context: str,
        options: List[str],
        urgency: str = "medium"
    ) -> CAMDecisionRequest:
        """
        Richiede decisione (sempre human-in-the-loop per decisioni importanti).
        """
        
        request = CAMDecisionRequest(
            id=f"decision_{uuid.uuid4().hex[:8]}",
            context=context,
            options=options,
            urgency=urgency,
            requires_human=True  # SEMPRE true per decisioni
        )
        
        self._log_event("decision_requested", f"Decisione richiesta: {context[:50]}", {
            'options': options,
            'urgency': urgency
        })
        
        # Notify human needed
        for callback in self._on_human_needed:
            try:
                await callback(request)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return request
    
    async def record_decision(
        self,
        decision_id: str,
        chosen_option: str,
        decision_maker: str,
        rationale: str
    ):
        """Registra decisione presa"""
        
        if self.active_crisis_id:
            self.recovery.timeline.record_decision(
                decision_id,
                f"Decisione: {chosen_option} - {rationale}",
                decision_maker,
                []
            )
        
        self._log_event("decision_recorded", f"Decisione {decision_id}: {chosen_option}", {
            'decision_maker': decision_maker,
            'rationale': rationale
        })
    
    # ==================== ACTIONS ====================
    
    async def request_action(
        self,
        action_id: str,
        description: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Richiede esecuzione azione con tutti i controlli CAM.
        """
        
        # Check permission dal control layer
        allowed, reason = await self.control.request_permission(action_id, context)
        
        if not allowed:
            self._log_event("action_blocked", f"Azione {action_id} bloccata: {reason}")
            return {
                'allowed': False,
                'reason': reason,
                'action_id': action_id
            }
        
        # Check safe state
        if not self.control.safe_state.can_proceed(action_id, context):
            self._log_event("action_blocked_safe", f"Azione {action_id} bloccata da safe state")
            return {
                'allowed': False,
                'reason': "Sistema in safe state",
                'action_id': action_id
            }
        
        # Log e procedi
        self._log_event("action_approved", f"Azione {action_id} approvata", context)
        
        return {
            'allowed': True,
            'action_id': action_id,
            'explanation': f"Azione approvata: {description}"
        }
    
    # ==================== UI ====================
    
    def get_ui_state(self, user_id: str) -> Dict[str, Any]:
        """Ottiene stato UI per utente"""
        
        return self.human_interface.update_for_crisis(
            self.detection.current_level.value,
            user_id
        )
    
    def register_user(self, user_id: str, role: UserRole):
        """Registra utente"""
        self.human_interface.role_views.register_user(user_id, role)
    
    # ==================== STATUS ====================
    
    def get_state(self) -> CAMState:
        """Ottiene stato corrente"""
        
        return CAMState(
            status=self.status,
            crisis_level=self.detection.current_level,
            control_state=self.control.state,
            autonomy_level=self.control.clamp.get_autonomy_level("ACTIONS"),
            active_crisis_id=self.active_crisis_id
        )
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Ottiene dashboard completa"""
        
        state = self.get_state()
        
        return {
            'cam': {
                'status': state.status.value,
                'version': self.VERSION,
                'active_crisis': state.active_crisis_id
            },
            'detection': {
                'level': state.crisis_level.value,
                'signals_active': len(self.detection.signals)
            },
            'control': {
                'state': state.control_state.value,
                'autonomy': state.autonomy_level
            },
            'recovery': {
                'phase': self.recovery.restore.current_phase.value,
                'autonomy': self.recovery.restore.get_autonomy_level()
            },
            'rules': {
                'no_irreversible_auto': True,
                'always_explain': True,
                'human_in_loop': True,
                'audit_trail': True,
                'stop_available': True
            }
        }
    
    # ==================== AUDIT ====================
    
    def _log_event(
        self,
        event_type: str,
        description: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Log evento per audit trail"""
        
        event = CAMEvent(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            description=description,
            timestamp=datetime.now(),
            data=data or {}
        )
        
        self.event_log.append(event)
        
        # Keep only last 1000 events in memory
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]
    
    def get_audit_log(
        self,
        limit: int = 100,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Ottiene audit log"""
        
        events = self.event_log
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return [
            {
                'id': e.id,
                'type': e.event_type,
                'description': e.description,
                'timestamp': e.timestamp.isoformat(),
                'data': e.data
            }
            for e in events[-limit:]
        ]
    
    # ==================== CALLBACKS ====================
    
    def on_crisis_detected(self, callback: Callable):
        """Registra callback per crisi rilevata"""
        self._on_crisis_detected.append(callback)
    
    def on_status_change(self, callback: Callable):
        """Registra callback per cambio status"""
        self._on_status_change.append(callback)
    
    def on_human_needed(self, callback: Callable):
        """Registra callback per richiesta intervento umano"""
        self._on_human_needed.append(callback)
    
    # ==================== FORMATTED OUTPUT ====================
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        state = self.get_state()
        dashboard = self.get_dashboard()
        
        status_emoji = {
            CAMStatus.INACTIVE: "⚪",
            CAMStatus.MONITORING: "🟢",
            CAMStatus.ENGAGED: "🔴",
            CAMStatus.RECOVERY: "🟡",
            CAMStatus.EMERGENCY_STOP: "🛑"
        }
        
        level_emoji = {
            "NONE": "🟢",
            "WATCH": "🟡",
            "SOFT": "🟠",
            "ACTIVE": "🔴",
            "CRITICAL": "⚫"
        }
        
        return f"""
# 🎯 GIDEON Crisis Automation Mode (CAM)
**Version:** {self.VERSION}

## Status
{status_emoji[state.status]} **{state.status.value.upper()}**

## Crisis Level
{level_emoji.get(state.crisis_level.value, '⚪')} **{state.crisis_level.value}**

## Control State
**Stato:** {state.control_state.value}
**Autonomia:** {state.autonomy_level:.0%}

## Active Crisis
{'🚨 ' + state.active_crisis_id if state.active_crisis_id else '✅ Nessuna crisi attiva'}

## Recovery Phase
**{dashboard['recovery']['phase'].replace('_', ' ').title()}**

---

## Regole Attive
- {'✅' if dashboard['rules']['no_irreversible_auto'] else '❌'} No decisioni irreversibili automatiche
- {'✅' if dashboard['rules']['always_explain'] else '❌'} Sempre spiegazione
- {'✅' if dashboard['rules']['human_in_loop'] else '❌'} Human-in-the-Loop
- {'✅' if dashboard['rules']['audit_trail'] else '❌'} Audit trail completo
- {'✅' if dashboard['rules']['stop_available'] else '❌'} STOP sempre disponibile

---

> *"Explain > Act | Slow is Smooth, Smooth is Fast | Human-in-the-Loop sempre"*
"""


# ============================================================
# SINGLETON & FACTORY
# ============================================================

_cam_instance: Optional[CrisisAutomationMode] = None


def get_cam() -> CrisisAutomationMode:
    """Ottiene istanza singleton CAM"""
    global _cam_instance
    if _cam_instance is None:
        _cam_instance = CrisisAutomationMode()
    return _cam_instance


async def initialize_cam() -> CrisisAutomationMode:
    """Inizializza e ottiene CAM"""
    cam = get_cam()
    await cam.initialize()
    return cam


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'CrisisAutomationMode',
    'CAMStatus',
    'CAMState',
    'CAMEvent',
    'CAMDecisionRequest',
    'get_cam',
    'initialize_cam'
]
