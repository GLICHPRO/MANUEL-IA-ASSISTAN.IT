"""
🤖 CAM AUTOMATION LAYER
========================
Automazione intelligente per crisi.

Componenti:
- CAMAutomationManager: Gestore centrale automazioni
- PreDecisionProcessor: Elabora PRIMA che umano chieda
- RiskTriggeredActions: Azioni automatiche su soglie rischio
- SelfCorrectionLoop: Auto-correzione con monitoraggio

❌ Nessuna decisione irreversibile automatica
❌ Nessuna azione senza spiegazione
✅ Sempre possibilità di STOP
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
import logging
import asyncio
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class AutomationType(Enum):
    """Tipi di automazione"""
    PREPARATION = "preparation"  # Pre-elaborazione dati
    MONITORING = "monitoring"  # Monitoraggio continuo
    ALERT = "alert"  # Notifiche
    DEFENSIVE = "defensive"  # Azioni difensive
    INFORMATIVE = "informative"  # Raccolta info
    REVERSIBLE = "reversible"  # Azioni reversibili
    # MAI automatici
    IRREVERSIBLE = "irreversible"  # ❌ Solo umano


class ActionReversibility(Enum):
    """Reversibilità azione"""
    FULLY_REVERSIBLE = "fully_reversible"  # 100% annullabile
    PARTIALLY_REVERSIBLE = "partially_reversible"  # Alcuni effetti permanenti
    TIME_REVERSIBLE = "time_reversible"  # Reversibile entro N tempo
    IRREVERSIBLE = "irreversible"  # ❌ MAI automatico


class TriggerType(Enum):
    """Tipi di trigger"""
    RISK_THRESHOLD = "risk_threshold"
    TIME_BASED = "time_based"
    PATTERN_DETECTED = "pattern_detected"
    HUMAN_REQUESTED = "human_requested"
    CRISIS_LEVEL = "crisis_level"


@dataclass
class AutomationAction:
    """Azione automatica"""
    id: str
    name: str
    automation_type: AutomationType
    reversibility: ActionReversibility
    description: str
    handler: Optional[Callable] = None
    enabled: bool = True
    requires_explanation: bool = True
    cooldown_seconds: int = 60
    last_executed: Optional[datetime] = None
    execution_count: int = 0


@dataclass
class TriggerCondition:
    """Condizione di trigger"""
    id: str
    trigger_type: TriggerType
    threshold: float
    action_ids: List[str]
    description: str
    enabled: bool = True


@dataclass
class ExecutionRecord:
    """Record esecuzione"""
    action_id: str
    timestamp: datetime
    trigger: str
    success: bool
    explanation: str
    result: Optional[str] = None
    reversed: bool = False
    reverse_timestamp: Optional[datetime] = None


@dataclass
class PreProcessedData:
    """Dati pre-elaborati"""
    key: str
    data: Any
    timestamp: datetime
    ttl_seconds: int = 300
    used: bool = False


# ============================================================
# CAM AUTOMATION MANAGER
# ============================================================

class CAMAutomationManager:
    """
    Gestore centrale delle automazioni CAM.
    
    Principi:
    - ❌ MAI azioni irreversibili automatiche
    - ✅ Sempre spiegazione
    - ✅ Sempre possibilità STOP
    - ✅ Audit trail completo
    """
    
    def __init__(self):
        self.actions: Dict[str, AutomationAction] = {}
        self.triggers: Dict[str, TriggerCondition] = {}
        self.execution_history: List[ExecutionRecord] = []
        
        # Emergency stop
        self._emergency_stop = False
        self._stop_reason: Optional[str] = None
        
        # Rate limiting
        self._action_timestamps: Dict[str, List[datetime]] = defaultdict(list)
        
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Registra azioni di default"""
        
        # Azioni informative (sicure)
        self.register_action(AutomationAction(
            id="gather_context",
            name="Raccolta Contesto",
            automation_type=AutomationType.INFORMATIVE,
            reversibility=ActionReversibility.FULLY_REVERSIBLE,
            description="Raccoglie informazioni aggiuntive sul contesto"
        ))
        
        self.register_action(AutomationAction(
            id="prepare_analysis",
            name="Preparazione Analisi",
            automation_type=AutomationType.PREPARATION,
            reversibility=ActionReversibility.FULLY_REVERSIBLE,
            description="Pre-elabora dati per analisi"
        ))
        
        self.register_action(AutomationAction(
            id="risk_alert",
            name="Alert Rischio",
            automation_type=AutomationType.ALERT,
            reversibility=ActionReversibility.FULLY_REVERSIBLE,
            description="Notifica aumento rischio"
        ))
        
        # Azioni difensive (con cautela)
        self.register_action(AutomationAction(
            id="defensive_snapshot",
            name="Snapshot Difensivo",
            automation_type=AutomationType.DEFENSIVE,
            reversibility=ActionReversibility.FULLY_REVERSIBLE,
            description="Crea snapshot stato corrente"
        ))
        
        self.register_action(AutomationAction(
            id="safe_mode_prep",
            name="Preparazione Safe Mode",
            automation_type=AutomationType.DEFENSIVE,
            reversibility=ActionReversibility.FULLY_REVERSIBLE,
            description="Prepara passaggio a safe mode",
            cooldown_seconds=120
        ))
    
    def register_action(self, action: AutomationAction) -> bool:
        """Registra nuova azione"""
        
        # ❌ Rifiuta azioni irreversibili automatiche
        if action.reversibility == ActionReversibility.IRREVERSIBLE:
            logger.error(f"❌ RIFIUTATO: Azione irreversibile '{action.name}' non può essere automatica")
            return False
        
        # ❌ Tutte le azioni DEVONO avere spiegazione
        if not action.requires_explanation:
            logger.warning(f"⚠️ Forzata spiegazione per '{action.name}'")
            action.requires_explanation = True
        
        self.actions[action.id] = action
        logger.info(f"✅ Azione registrata: {action.name}")
        return True
    
    def register_trigger(self, trigger: TriggerCondition) -> bool:
        """Registra trigger"""
        
        # Verifica che tutte le azioni esistano
        for action_id in trigger.action_ids:
            if action_id not in self.actions:
                logger.error(f"❌ Azione '{action_id}' non trovata per trigger")
                return False
            
            # Verifica reversibilità
            action = self.actions[action_id]
            if action.reversibility == ActionReversibility.IRREVERSIBLE:
                logger.error(f"❌ Trigger non può attivare azione irreversibile '{action_id}'")
                return False
        
        self.triggers[trigger.id] = trigger
        return True
    
    async def execute_action(
        self,
        action_id: str,
        trigger: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Esegue azione con tutti i controlli.
        
        Returns:
            (success, explanation, result)
        """
        
        # STOP check
        if self._emergency_stop:
            return False, f"🛑 EMERGENCY STOP: {self._stop_reason}", None
        
        # Azione esiste?
        action = self.actions.get(action_id)
        if not action:
            return False, f"Azione '{action_id}' non trovata", None
        
        # Abilitata?
        if not action.enabled:
            return False, f"Azione '{action.name}' disabilitata", None
        
        # Cooldown
        if action.last_executed:
            elapsed = (datetime.now() - action.last_executed).total_seconds()
            if elapsed < action.cooldown_seconds:
                remaining = action.cooldown_seconds - elapsed
                return False, f"Cooldown attivo ({remaining:.0f}s rimanenti)", None
        
        # ❌ Blocca irreversibili (doppio check)
        if action.reversibility == ActionReversibility.IRREVERSIBLE:
            return False, "❌ Azioni irreversibili richiedono approvazione umana", None
        
        # Genera spiegazione
        explanation = self._generate_explanation(action, trigger, context)
        
        # Esegui
        result = None
        success = True
        
        try:
            if action.handler:
                result = await action.handler(context or {})
            else:
                result = f"Azione simulata: {action.name}"
        except Exception as e:
            success = False
            result = str(e)
            logger.error(f"Errore esecuzione {action_id}: {e}")
        
        # Record
        action.last_executed = datetime.now()
        action.execution_count += 1
        
        self.execution_history.append(ExecutionRecord(
            action_id=action_id,
            timestamp=datetime.now(),
            trigger=trigger,
            success=success,
            explanation=explanation,
            result=result
        ))
        
        logger.info(f"✅ Azione eseguita: {action.name} | Trigger: {trigger}")
        
        return success, explanation, result
    
    def _generate_explanation(
        self,
        action: AutomationAction,
        trigger: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Genera spiegazione azione"""
        
        return f"""
## 🤖 Azione Automatica Eseguita

**Azione:** {action.name}
**Tipo:** {action.automation_type.value}
**Reversibilità:** {action.reversibility.value}

### Motivo
{action.description}

### Trigger
{trigger}

### Contesto
{context if context else 'Nessun contesto aggiuntivo'}

### Sicurezza
- ✅ Azione reversibile
- ✅ Spiegazione fornita
- ✅ STOP disponibile
"""
    
    def emergency_stop(self, reason: str = "Richiesta utente"):
        """Attiva emergency stop"""
        self._emergency_stop = True
        self._stop_reason = reason
        logger.warning(f"🛑 EMERGENCY STOP ATTIVATO: {reason}")
    
    def resume(self):
        """Riprende operazioni"""
        self._emergency_stop = False
        self._stop_reason = None
        logger.info("✅ Operazioni riprese")
    
    def is_stopped(self) -> bool:
        """Check stop status"""
        return self._emergency_stop
    
    async def check_triggers(
        self,
        metrics: Dict[str, float]
    ) -> List[str]:
        """Verifica trigger e esegue azioni"""
        
        executed = []
        
        for trigger_id, trigger in self.triggers.items():
            if not trigger.enabled:
                continue
            
            should_fire = False
            
            if trigger.trigger_type == TriggerType.RISK_THRESHOLD:
                risk = metrics.get('risk_level', 0)
                should_fire = risk >= trigger.threshold
            
            elif trigger.trigger_type == TriggerType.CRISIS_LEVEL:
                crisis = metrics.get('crisis_level', 0)
                should_fire = crisis >= trigger.threshold
            
            if should_fire:
                for action_id in trigger.action_ids:
                    success, _, _ = await self.execute_action(
                        action_id,
                        f"Trigger: {trigger.description}"
                    )
                    if success:
                        executed.append(action_id)
        
        return executed


# ============================================================
# PRE-DECISION PROCESSOR
# ============================================================

class PreDecisionProcessor:
    """
    Elabora PRIMA che l'umano chieda.
    
    Anticipa bisogni informativi in situazioni di crisi.
    """
    
    def __init__(self):
        self.pre_processed: Dict[str, PreProcessedData] = {}
        self.processing_queue: List[str] = []
        self._processors: Dict[str, Callable] = {}
    
    def register_processor(
        self,
        key: str,
        processor: Callable[[Dict[str, Any]], Any],
        description: str = ""
    ):
        """Registra pre-processor"""
        self._processors[key] = processor
        logger.info(f"Pre-processor registrato: {key}")
    
    async def pre_process(
        self,
        key: str,
        context: Dict[str, Any],
        ttl_seconds: int = 300
    ) -> bool:
        """Esegue pre-elaborazione"""
        
        if key not in self._processors:
            return False
        
        try:
            data = await self._processors[key](context)
            
            self.pre_processed[key] = PreProcessedData(
                key=key,
                data=data,
                timestamp=datetime.now(),
                ttl_seconds=ttl_seconds
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Pre-processing fallito per {key}: {e}")
            return False
    
    def get_pre_processed(
        self,
        key: str
    ) -> Optional[Any]:
        """Recupera dati pre-elaborati"""
        
        data = self.pre_processed.get(key)
        
        if not data:
            return None
        
        # TTL check
        age = (datetime.now() - data.timestamp).total_seconds()
        if age > data.ttl_seconds:
            del self.pre_processed[key]
            return None
        
        data.used = True
        return data.data
    
    def cleanup_expired(self):
        """Rimuove dati scaduti"""
        
        now = datetime.now()
        expired = []
        
        for key, data in self.pre_processed.items():
            age = (now - data.timestamp).total_seconds()
            if age > data.ttl_seconds:
                expired.append(key)
        
        for key in expired:
            del self.pre_processed[key]
        
        return len(expired)
    
    async def anticipate_needs(
        self,
        crisis_level: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Anticipa bisogni basandosi su livello crisi"""
        
        pre_processed = []
        
        # Definisci cosa pre-elaborare per ogni livello
        needs_by_level = {
            "WATCH": ["context_summary", "recent_history"],
            "SOFT": ["context_summary", "recent_history", "risk_factors", "stakeholders"],
            "ACTIVE": ["context_summary", "recent_history", "risk_factors", "stakeholders", 
                      "options_analysis", "timeline"],
            "CRITICAL": ["context_summary", "recent_history", "risk_factors", "stakeholders",
                        "options_analysis", "timeline", "emergency_contacts", "safe_state_plan"]
        }
        
        needs = needs_by_level.get(crisis_level, [])
        
        for need in needs:
            if need in self._processors:
                success = await self.pre_process(need, context)
                if success:
                    pre_processed.append(need)
        
        return pre_processed


# ============================================================
# RISK-TRIGGERED ACTIONS
# ============================================================

class RiskTriggeredActions:
    """
    Azioni automatiche basate su soglie di rischio.
    
    SOLO azioni reversibili e difensive.
    """
    
    # Soglie predefinite
    DEFAULT_THRESHOLDS = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "critical": 0.9
    }
    
    def __init__(self, automation_manager: CAMAutomationManager):
        self.manager = automation_manager
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        self.triggered_at: Dict[str, datetime] = {}
        
        # Cooldown tra trigger stesso livello
        self.level_cooldown = timedelta(minutes=5)
    
    async def evaluate_risk(
        self,
        risk_score: float,
        risk_context: Dict[str, Any]
    ) -> List[str]:
        """Valuta rischio e trigger azioni appropriate"""
        
        triggered_actions = []
        now = datetime.now()
        
        # Determina livello
        level = "none"
        for level_name, threshold in sorted(self.thresholds.items(), 
                                            key=lambda x: x[1], reverse=True):
            if risk_score >= threshold:
                level = level_name
                break
        
        # Check cooldown
        last_trigger = self.triggered_at.get(level)
        if last_trigger and (now - last_trigger) < self.level_cooldown:
            return []
        
        # Azioni per livello
        actions_by_level = {
            "low": ["gather_context"],
            "medium": ["gather_context", "prepare_analysis"],
            "high": ["gather_context", "prepare_analysis", "risk_alert"],
            "critical": ["gather_context", "prepare_analysis", "risk_alert", 
                        "defensive_snapshot", "safe_mode_prep"]
        }
        
        for action_id in actions_by_level.get(level, []):
            success, _, _ = await self.manager.execute_action(
                action_id,
                f"Risk level {level} ({risk_score:.0%})",
                risk_context
            )
            if success:
                triggered_actions.append(action_id)
        
        if triggered_actions:
            self.triggered_at[level] = now
        
        return triggered_actions


# ============================================================
# SELF-CORRECTION LOOP
# ============================================================

class SelfCorrectionLoop:
    """
    Loop di auto-correzione con monitoraggio.
    
    Monitora azioni e corregge se necessario.
    """
    
    def __init__(self, automation_manager: CAMAutomationManager):
        self.manager = automation_manager
        self.monitoring_interval = 30  # secondi
        self.corrections_made: List[Dict[str, Any]] = []
        self._running = False
    
    async def start_monitoring(self):
        """Avvia loop di monitoraggio"""
        
        self._running = True
        
        while self._running:
            corrections = await self._check_and_correct()
            
            if corrections:
                self.corrections_made.extend(corrections)
                logger.info(f"🔄 Auto-correzioni applicate: {len(corrections)}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    def stop_monitoring(self):
        """Ferma loop"""
        self._running = False
    
    async def _check_and_correct(self) -> List[Dict[str, Any]]:
        """Verifica e corregge"""
        
        corrections = []
        
        # Controlla azioni recenti
        recent = [r for r in self.manager.execution_history 
                  if (datetime.now() - r.timestamp).total_seconds() < 300]
        
        for record in recent:
            if not record.success and not record.reversed:
                # Prova a correggere
                correction = await self._attempt_correction(record)
                if correction:
                    corrections.append(correction)
        
        # Controlla pattern anomali
        anomalies = self._detect_anomalies()
        for anomaly in anomalies:
            correction = await self._handle_anomaly(anomaly)
            if correction:
                corrections.append(correction)
        
        return corrections
    
    async def _attempt_correction(
        self,
        record: ExecutionRecord
    ) -> Optional[Dict[str, Any]]:
        """Tenta correzione per azione fallita"""
        
        action = self.manager.actions.get(record.action_id)
        
        if not action:
            return None
        
        # Se reversibile, possiamo tentare rollback
        if action.reversibility in [ActionReversibility.FULLY_REVERSIBLE,
                                     ActionReversibility.PARTIALLY_REVERSIBLE]:
            record.reversed = True
            record.reverse_timestamp = datetime.now()
            
            return {
                'type': 'rollback',
                'action_id': record.action_id,
                'timestamp': datetime.now(),
                'reason': 'Azione fallita, rollback applicato'
            }
        
        return None
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Rileva anomalie nei pattern"""
        
        anomalies = []
        
        # Troppe esecuzioni stessa azione
        recent = self.manager.execution_history[-50:]
        action_counts = defaultdict(int)
        
        for record in recent:
            action_counts[record.action_id] += 1
        
        for action_id, count in action_counts.items():
            if count > 10:  # Soglia
                anomalies.append({
                    'type': 'excessive_execution',
                    'action_id': action_id,
                    'count': count
                })
        
        # Troppi fallimenti
        failures = sum(1 for r in recent if not r.success)
        if failures > len(recent) * 0.3:  # >30% fallimenti
            anomalies.append({
                'type': 'high_failure_rate',
                'rate': failures / len(recent) if recent else 0
            })
        
        return anomalies
    
    async def _handle_anomaly(
        self,
        anomaly: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Gestisce anomalia rilevata"""
        
        if anomaly['type'] == 'excessive_execution':
            action = self.manager.actions.get(anomaly['action_id'])
            if action:
                # Aumenta cooldown temporaneamente
                original_cooldown = action.cooldown_seconds
                action.cooldown_seconds = original_cooldown * 3
                
                return {
                    'type': 'cooldown_increased',
                    'action_id': anomaly['action_id'],
                    'original': original_cooldown,
                    'new': action.cooldown_seconds,
                    'reason': f"Esecuzioni eccessive: {anomaly['count']}"
                }
        
        elif anomaly['type'] == 'high_failure_rate':
            # Considera safe mode
            logger.warning(f"⚠️ Alto tasso fallimenti: {anomaly['rate']:.0%}")
            
            return {
                'type': 'warning_issued',
                'issue': 'high_failure_rate',
                'rate': anomaly['rate'],
                'recommendation': 'Considerare safe mode'
            }
        
        return None


# ============================================================
# UNIFIED AUTOMATION LAYER
# ============================================================

class AutomationLayer:
    """
    Automation Layer unificato per CAM.
    
    Integra tutti i componenti di automazione.
    """
    
    def __init__(self):
        self.manager = CAMAutomationManager()
        self.pre_processor = PreDecisionProcessor()
        self.risk_triggered = RiskTriggeredActions(self.manager)
        self.self_correction = SelfCorrectionLoop(self.manager)
    
    async def initialize(self):
        """Inizializza layer"""
        
        # Registra pre-processors di default
        self._register_default_preprocessors()
        
        logger.info("✅ Automation Layer inizializzato")
    
    def _register_default_preprocessors(self):
        """Registra pre-processors di default"""
        
        async def context_summary(ctx):
            return f"Context summary at {datetime.now()}: {len(ctx)} keys"
        
        async def recent_history(ctx):
            return f"Recent history: last {ctx.get('history_depth', 10)} events"
        
        async def risk_factors(ctx):
            return ["Factor 1", "Factor 2", "Factor 3"]
        
        self.pre_processor.register_processor("context_summary", context_summary)
        self.pre_processor.register_processor("recent_history", recent_history)
        self.pre_processor.register_processor("risk_factors", risk_factors)
    
    def emergency_stop(self, reason: str = "Richiesta utente"):
        """Emergency stop globale"""
        self.manager.emergency_stop(reason)
        self.self_correction.stop_monitoring()
    
    def resume(self):
        """Riprende operazioni"""
        self.manager.resume()


# Singleton
_automation_layer: Optional[AutomationLayer] = None


def get_automation_layer() -> AutomationLayer:
    """Ottiene istanza singleton"""
    global _automation_layer
    if _automation_layer is None:
        _automation_layer = AutomationLayer()
    return _automation_layer
