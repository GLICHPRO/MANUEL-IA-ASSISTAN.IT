"""
🛡️ SAFE STATE CONTROLLER
==========================
Definisce uno stato "safe" a cui GIDEON torna se:
- Perde connessione
- Rileva anomalie nel proprio comportamento
- Supera soglie operative

Come un "return-to-home" per droni.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
import asyncio
from collections import deque

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Stati del sistema"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    SAFE_MODE = "safe_mode"
    RECOVERY = "recovery"
    SHUTDOWN = "shutdown"


class SafeTrigger(Enum):
    """Trigger per safe mode"""
    CONNECTION_LOST = "connection_lost"
    ANOMALY_DETECTED = "anomaly_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    MANUAL_TRIGGER = "manual_trigger"
    HEALTH_CHECK_FAILED = "health_failed"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_BREACH = "security_breach"
    TIMEOUT = "timeout"
    ERROR_RATE_HIGH = "error_rate"
    DEPENDENCY_FAILURE = "dependency_failure"


class RecoveryAction(Enum):
    """Azioni di recovery"""
    WAIT = "wait"
    RETRY = "retry"
    RESTART_COMPONENT = "restart"
    NOTIFY_HUMAN = "notify"
    FULL_RESTART = "full_restart"
    ESCALATE = "escalate"


@dataclass
class SafeStateConfig:
    """Configurazione safe state"""
    # Soglie
    max_error_rate: float = 0.1           # 10% error rate
    max_response_time: float = 5.0        # 5 secondi
    max_memory_usage: float = 0.85        # 85% memoria
    max_cpu_usage: float = 0.90           # 90% CPU
    connection_timeout: float = 30.0      # 30 secondi
    health_check_interval: float = 60.0   # 1 minuto
    
    # Comportamenti safe mode
    disable_external_calls: bool = True
    disable_write_operations: bool = True
    disable_automation: bool = True
    enable_readonly_mode: bool = True
    rate_limit_factor: float = 0.1        # 10% della capacità normale
    
    # Recovery
    auto_recovery: bool = True
    recovery_wait_time: float = 60.0      # 1 minuto
    max_recovery_attempts: int = 3
    escalation_after_attempts: int = 2


@dataclass
class SafeStateSnapshot:
    """Snapshot dello stato al momento del trigger"""
    timestamp: datetime
    trigger: SafeTrigger
    trigger_details: Dict[str, Any]
    system_metrics: Dict[str, float]
    active_operations: List[str]
    pending_operations: List[str]
    state_before: SystemState


@dataclass
class RecoveryAttempt:
    """Tentativo di recovery"""
    timestamp: datetime
    attempt_number: int
    action: RecoveryAction
    success: bool
    details: str
    duration: timedelta


@dataclass
class SafeStateStatus:
    """Status completo del safe state"""
    current_state: SystemState
    in_safe_mode: bool
    time_in_current_state: timedelta
    last_trigger: Optional[SafeTrigger]
    last_trigger_time: Optional[datetime]
    recovery_attempts: int
    pending_recovery: bool
    disabled_features: List[str]
    health_score: float  # 0-1
    next_health_check: datetime


class SafeStateController:
    """
    Controller per Safe State di GIDEON.
    
    Definisce uno stato "safe" a cui tornare se:
    - Perde connessione
    - Rileva anomalie nel proprio comportamento
    - Supera soglie operative
    
    Come un "return-to-home" per droni.
    """
    
    def __init__(self, config: Optional[SafeStateConfig] = None):
        self.config = config or SafeStateConfig()
        self.state = SystemState.NORMAL
        self.state_history: deque = deque(maxlen=100)
        self.last_state_change: datetime = datetime.now()
        self.snapshots: List[SafeStateSnapshot] = []
        self.recovery_log: List[RecoveryAttempt] = []
        self.recovery_attempts: int = 0
        self.pending_recovery: bool = False
        
        # Metriche
        self.error_count: int = 0
        self.request_count: int = 0
        self.last_health_check: datetime = datetime.now()
        self.last_successful_operation: datetime = datetime.now()
        
        # Callbacks
        self.on_safe_mode_enter: List[Callable] = []
        self.on_safe_mode_exit: List[Callable] = []
        self.on_recovery_complete: List[Callable] = []
        
        # Features disabilitabili
        self.feature_status: Dict[str, bool] = {
            'external_api_calls': True,
            'database_writes': True,
            'automation': True,
            'file_operations': True,
            'notifications': True,
            'ai_inference': True
        }
        
        logger.info("🛡️ SafeStateController inizializzato")
    
    async def trigger_safe_mode(
        self,
        trigger: SafeTrigger,
        details: Optional[Dict[str, Any]] = None
    ) -> SafeStateStatus:
        """
        Attiva la modalità safe.
        """
        
        if self.state == SystemState.SAFE_MODE:
            logger.info(f"🛡️ Già in safe mode, trigger aggiuntivo: {trigger.value}")
            return self.get_status()
        
        logger.warning(f"🛡️ ATTIVAZIONE SAFE MODE - Trigger: {trigger.value}")
        
        # Crea snapshot
        snapshot = SafeStateSnapshot(
            timestamp=datetime.now(),
            trigger=trigger,
            trigger_details=details or {},
            system_metrics=self._get_current_metrics(),
            active_operations=self._get_active_operations(),
            pending_operations=self._get_pending_operations(),
            state_before=self.state
        )
        self.snapshots.append(snapshot)
        
        # Cambia stato
        previous_state = self.state
        self.state = SystemState.SAFE_MODE
        self.last_state_change = datetime.now()
        
        # Registra nella history
        self.state_history.append({
            'timestamp': datetime.now().isoformat(),
            'from': previous_state.value,
            'to': SystemState.SAFE_MODE.value,
            'trigger': trigger.value
        })
        
        # Disabilita features
        await self._apply_safe_restrictions()
        
        # Notifica callbacks
        for callback in self.on_safe_mode_enter:
            try:
                await callback(snapshot) if asyncio.iscoroutinefunction(callback) else callback(snapshot)
            except Exception as e:
                logger.error(f"Safe mode callback error: {e}")
        
        # Avvia recovery automatico se configurato
        if self.config.auto_recovery:
            asyncio.create_task(self._auto_recovery())
        
        return self.get_status()
    
    async def _apply_safe_restrictions(self):
        """Applica restrizioni safe mode"""
        
        if self.config.disable_external_calls:
            self.feature_status['external_api_calls'] = False
            logger.info("🛡️ External API calls DISABILITATI")
        
        if self.config.disable_write_operations:
            self.feature_status['database_writes'] = False
            self.feature_status['file_operations'] = False
            logger.info("🛡️ Write operations DISABILITATI")
        
        if self.config.disable_automation:
            self.feature_status['automation'] = False
            logger.info("🛡️ Automation DISABILITATA")
    
    async def _auto_recovery(self):
        """Tenta recovery automatico"""
        
        self.pending_recovery = True
        
        # Attendi prima del primo tentativo
        await asyncio.sleep(self.config.recovery_wait_time)
        
        while self.state == SystemState.SAFE_MODE and self.recovery_attempts < self.config.max_recovery_attempts:
            self.recovery_attempts += 1
            
            logger.info(f"🛡️ Recovery attempt {self.recovery_attempts}/{self.config.max_recovery_attempts}")
            
            success = await self._attempt_recovery()
            
            if success:
                await self.exit_safe_mode("auto_recovery")
                self.pending_recovery = False
                return
            
            # Escalation dopo tentativi falliti
            if self.recovery_attempts >= self.config.escalation_after_attempts:
                await self._escalate()
            
            # Attendi prima del prossimo tentativo
            await asyncio.sleep(self.config.recovery_wait_time)
        
        self.pending_recovery = False
        logger.warning("🛡️ Auto-recovery fallito dopo max tentativi")
    
    async def _attempt_recovery(self) -> bool:
        """Singolo tentativo di recovery"""
        
        start = datetime.now()
        
        try:
            # Check health
            health_ok = await self._check_health()
            
            # Check che il trigger sia risolto
            trigger_resolved = await self._check_trigger_resolved()
            
            success = health_ok and trigger_resolved
            
            # Log attempt
            self.recovery_log.append(RecoveryAttempt(
                timestamp=start,
                attempt_number=self.recovery_attempts,
                action=RecoveryAction.RETRY,
                success=success,
                details="Health check e trigger check",
                duration=datetime.now() - start
            ))
            
            return success
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            
            self.recovery_log.append(RecoveryAttempt(
                timestamp=start,
                attempt_number=self.recovery_attempts,
                action=RecoveryAction.RETRY,
                success=False,
                details=f"Errore: {str(e)}",
                duration=datetime.now() - start
            ))
            
            return False
    
    async def _check_health(self) -> bool:
        """Check health del sistema"""
        
        metrics = self._get_current_metrics()
        
        # Check error rate
        if self.request_count > 0:
            error_rate = self.error_count / self.request_count
            if error_rate > self.config.max_error_rate:
                return False
        
        # Check risorse
        if metrics.get('memory_usage', 0) > self.config.max_memory_usage:
            return False
        
        if metrics.get('cpu_usage', 0) > self.config.max_cpu_usage:
            return False
        
        return True
    
    async def _check_trigger_resolved(self) -> bool:
        """Verifica se il trigger originale è risolto"""
        
        if not self.snapshots:
            return True
        
        last_snapshot = self.snapshots[-1]
        trigger = last_snapshot.trigger
        
        # Check specifici per trigger
        if trigger == SafeTrigger.CONNECTION_LOST:
            # Verifica connettività (simulato)
            return True  # In produzione: actual connectivity check
        
        elif trigger == SafeTrigger.ERROR_RATE_HIGH:
            if self.request_count > 10:
                return (self.error_count / self.request_count) < self.config.max_error_rate
        
        elif trigger == SafeTrigger.RESOURCE_EXHAUSTION:
            metrics = self._get_current_metrics()
            return (
                metrics.get('memory_usage', 1) < self.config.max_memory_usage and
                metrics.get('cpu_usage', 1) < self.config.max_cpu_usage
            )
        
        # Default: considera risolto dopo il wait time
        return True
    
    async def _escalate(self):
        """Escala a intervento umano"""
        
        logger.warning("🛡️ ESCALATION - Recovery automatico fallito, richiesto intervento umano")
        
        # In produzione: invia notifica
        self.recovery_log.append(RecoveryAttempt(
            timestamp=datetime.now(),
            attempt_number=self.recovery_attempts,
            action=RecoveryAction.ESCALATE,
            success=False,
            details="Escalation a intervento umano",
            duration=timedelta(0)
        ))
    
    async def exit_safe_mode(self, reason: str = "manual"):
        """Esce dalla modalità safe"""
        
        if self.state != SystemState.SAFE_MODE:
            logger.info("🛡️ Non in safe mode, exit ignorato")
            return
        
        logger.info(f"🛡️ USCITA SAFE MODE - Motivo: {reason}")
        
        # Transizione a recovery
        self.state = SystemState.RECOVERY
        self.last_state_change = datetime.now()
        
        # Riabilita features gradualmente
        await self._restore_features()
        
        # Transizione a normal
        self.state = SystemState.NORMAL
        self.last_state_change = datetime.now()
        self.recovery_attempts = 0
        
        # Reset contatori
        self.error_count = 0
        self.request_count = 0
        
        # Registra nella history
        self.state_history.append({
            'timestamp': datetime.now().isoformat(),
            'from': SystemState.SAFE_MODE.value,
            'to': SystemState.NORMAL.value,
            'reason': reason
        })
        
        # Notifica callbacks
        for callback in self.on_safe_mode_exit:
            try:
                await callback(reason) if asyncio.iscoroutinefunction(callback) else callback(reason)
            except Exception as e:
                logger.error(f"Safe mode exit callback error: {e}")
    
    async def _restore_features(self):
        """Ripristina features gradualmente"""
        
        # Ripristina in ordine di priorità
        priority_order = [
            'ai_inference',
            'notifications',
            'database_writes',
            'external_api_calls',
            'file_operations',
            'automation'
        ]
        
        for feature in priority_order:
            if feature in self.feature_status:
                self.feature_status[feature] = True
                logger.info(f"🛡️ Feature ripristinata: {feature}")
                await asyncio.sleep(0.5)  # Graduale
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Verifica se una feature è abilitata"""
        return self.feature_status.get(feature, True)
    
    def record_operation(self, success: bool):
        """Registra esito operazione per tracking error rate"""
        
        self.request_count += 1
        if not success:
            self.error_count += 1
        else:
            self.last_successful_operation = datetime.now()
        
        # Check se superata soglia
        if self.request_count > 10:
            error_rate = self.error_count / self.request_count
            if error_rate > self.config.max_error_rate:
                asyncio.create_task(
                    self.trigger_safe_mode(
                        SafeTrigger.ERROR_RATE_HIGH,
                        {'error_rate': error_rate}
                    )
                )
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Ottiene metriche correnti"""
        
        try:
            import psutil
            return {
                'cpu_usage': psutil.cpu_percent() / 100,
                'memory_usage': psutil.virtual_memory().percent / 100,
                'disk_usage': psutil.disk_usage('/').percent / 100 if hasattr(psutil, 'disk_usage') else 0,
            }
        except:
            return {
                'cpu_usage': 0.5,
                'memory_usage': 0.5,
                'disk_usage': 0.5
            }
    
    def _get_active_operations(self) -> List[str]:
        """Ottiene operazioni attive (placeholder)"""
        return []
    
    def _get_pending_operations(self) -> List[str]:
        """Ottiene operazioni in attesa (placeholder)"""
        return []
    
    def get_status(self) -> SafeStateStatus:
        """Ritorna status completo"""
        
        now = datetime.now()
        time_in_state = now - self.last_state_change
        
        disabled_features = [
            f for f, enabled in self.feature_status.items() if not enabled
        ]
        
        last_trigger = self.snapshots[-1].trigger if self.snapshots else None
        last_trigger_time = self.snapshots[-1].timestamp if self.snapshots else None
        
        # Calcola health score
        metrics = self._get_current_metrics()
        health_score = 1.0 - (
            (metrics.get('cpu_usage', 0) + metrics.get('memory_usage', 0)) / 2
        )
        
        return SafeStateStatus(
            current_state=self.state,
            in_safe_mode=self.state == SystemState.SAFE_MODE,
            time_in_current_state=time_in_state,
            last_trigger=last_trigger,
            last_trigger_time=last_trigger_time,
            recovery_attempts=self.recovery_attempts,
            pending_recovery=self.pending_recovery,
            disabled_features=disabled_features,
            health_score=health_score,
            next_health_check=self.last_health_check + timedelta(seconds=self.config.health_check_interval)
        )
    
    def register_callback(
        self,
        event: str,
        callback: Callable
    ):
        """Registra callback per eventi"""
        
        if event == 'enter':
            self.on_safe_mode_enter.append(callback)
        elif event == 'exit':
            self.on_safe_mode_exit.append(callback)
        elif event == 'recovery':
            self.on_recovery_complete.append(callback)
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        status = self.get_status()
        
        state_emoji = {
            SystemState.NORMAL: '✅',
            SystemState.DEGRADED: '⚠️',
            SystemState.SAFE_MODE: '🛡️',
            SystemState.RECOVERY: '🔄',
            SystemState.SHUTDOWN: '🛑'
        }
        
        emoji = state_emoji.get(status.current_state, '❓')
        
        return f"""
# 🛡️ Safe State Controller

{emoji} **Stato Corrente**: {status.current_state.value}
⏱️ **Tempo in stato**: {status.time_in_current_state}
💚 **Health Score**: {status.health_score:.1%}

---

## Status
- **In Safe Mode**: {'Sì' if status.in_safe_mode else 'No'}
- **Recovery in corso**: {'Sì' if status.pending_recovery else 'No'}
- **Tentativi recovery**: {status.recovery_attempts}

---

## Ultimo Trigger
- **Tipo**: {status.last_trigger.value if status.last_trigger else 'Nessuno'}
- **Timestamp**: {status.last_trigger_time.strftime('%Y-%m-%d %H:%M:%S') if status.last_trigger_time else 'N/A'}

---

## Features Disabilitate
{chr(10).join(f"- ❌ {f}" for f in status.disabled_features) or '✅ Tutte le features attive'}

---

## Features Attive
{chr(10).join(f"- ✅ {f}" for f, enabled in self.feature_status.items() if enabled)}

---

## Configurazione
| Parametro | Valore |
|-----------|--------|
| Max Error Rate | {self.config.max_error_rate:.0%} |
| Max Response Time | {self.config.max_response_time}s |
| Max Memory | {self.config.max_memory_usage:.0%} |
| Auto Recovery | {'Sì' if self.config.auto_recovery else 'No'} |
| Max Recovery Attempts | {self.config.max_recovery_attempts} |

---

*Prossimo health check: {status.next_health_check.strftime('%H:%M:%S')}*
"""
    
    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Ritorna history transizioni stato"""
        return list(self.state_history)[-limit:]
    
    def get_recovery_log(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Ritorna log recovery attempts"""
        return [
            {
                'timestamp': r.timestamp.isoformat(),
                'attempt': r.attempt_number,
                'action': r.action.value,
                'success': r.success,
                'details': r.details,
                'duration': str(r.duration)
            }
            for r in self.recovery_log[-limit:]
        ]
