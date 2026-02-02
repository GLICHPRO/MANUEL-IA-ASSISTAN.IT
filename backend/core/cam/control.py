"""
🎮 CAM CONTROL LAYER
====================
Governa la potenza del sistema.

Componenti:
- AutonomyClamp: Riduce/rimodula automazioni
- TemporalGovernor: Gestisce il tempo come risorsa
- SafeStateEnforcer: Garantisce stato sicuro

Autonomia ≠ 0
Autonomia = controllata
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class ControlState(Enum):
    """Stati del Control Layer"""
    NORMAL = "normal"  # Operazioni normali
    MODERATED = "moderated"  # Autonomia ridotta
    RESTRICTED = "restricted"  # Autonomia minima
    SAFE_STATE = "safe_state"  # Solo analisi, zero azioni


class AutonomyDimension(Enum):
    """Dimensioni di autonomia controllabili"""
    ACTIONS = "actions"  # Azioni automatiche
    SUGGESTIONS = "suggestions"  # Suggerimenti proattivi
    SPEED = "speed"  # Velocità decisionale
    SCOPE = "scope"  # Ampiezza decisioni
    IRREVERSIBILITY = "irreversibility"  # Azioni irreversibili


@dataclass
class AutonomyLevel:
    """Livello autonomia per dimensione"""
    dimension: AutonomyDimension
    current: float  # 0-1
    baseline: float  # Livello normale
    minimum: float  # Minimo consentito
    reason: Optional[str] = None


@dataclass
class TemporalConstraint:
    """Vincolo temporale"""
    constraint_id: str
    description: str
    delay_seconds: float
    applies_to: List[str]  # Tipi di azione
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SafeStateConfig:
    """Configurazione Safe State"""
    allow_analysis: bool = True
    allow_explanations: bool = True
    allow_read_only: bool = True
    allow_suggestions: bool = False
    allow_actions: bool = False
    allow_external_calls: bool = False
    reason: str = ""
    activated_at: Optional[datetime] = None


# ============================================================
# AUTONOMY CLAMP
# ============================================================

class AutonomyClamp:
    """
    Riduce / rimodula autonomia del sistema.
    
    Controlla:
    - Automazioni
    - Suggerimenti
    - Velocità decisionale
    
    Autonomia ≠ 0, Autonomia = controllata
    """
    
    # Baseline levels per dimensione
    BASELINE_LEVELS = {
        AutonomyDimension.ACTIONS: 0.7,
        AutonomyDimension.SUGGESTIONS: 0.8,
        AutonomyDimension.SPEED: 0.6,
        AutonomyDimension.SCOPE: 0.7,
        AutonomyDimension.IRREVERSIBILITY: 0.3  # Sempre basso per default
    }
    
    # Minimum levels (mai sotto)
    MINIMUM_LEVELS = {
        AutonomyDimension.ACTIONS: 0.0,
        AutonomyDimension.SUGGESTIONS: 0.2,
        AutonomyDimension.SPEED: 0.1,
        AutonomyDimension.SCOPE: 0.1,
        AutonomyDimension.IRREVERSIBILITY: 0.0  # Zero per azioni irreversibili
    }
    
    # Reduction factors per crisis level
    CRISIS_REDUCTION = {
        'NONE': 1.0,
        'WATCH': 0.9,
        'SOFT': 0.7,
        'ACTIVE': 0.5,
        'CRITICAL': 0.3
    }
    
    def __init__(self):
        self.levels: Dict[AutonomyDimension, AutonomyLevel] = {}
        self.manual_overrides: Dict[AutonomyDimension, float] = {}
        self.crisis_level: str = 'NONE'
        
        # Inizializza livelli
        self._init_levels()
    
    def _init_levels(self):
        """Inizializza livelli autonomia"""
        for dim in AutonomyDimension:
            self.levels[dim] = AutonomyLevel(
                dimension=dim,
                current=self.BASELINE_LEVELS[dim],
                baseline=self.BASELINE_LEVELS[dim],
                minimum=self.MINIMUM_LEVELS[dim]
            )
    
    def set_crisis_level(self, level: str):
        """Aggiorna livello crisi e ricalcola autonomia"""
        
        self.crisis_level = level
        reduction = self.CRISIS_REDUCTION.get(level, 1.0)
        
        for dim, lvl in self.levels.items():
            # Applica riduzione mantenendo minimum
            if dim not in self.manual_overrides:
                new_level = max(lvl.baseline * reduction, lvl.minimum)
                lvl.current = new_level
                lvl.reason = f"Crisis level: {level}"
        
        logger.info(f"🎮 Autonomia modulata per crisi: {level} (reduction: {reduction})")
    
    def override_dimension(
        self,
        dimension: AutonomyDimension,
        level: float,
        reason: str
    ):
        """Override manuale per dimensione"""
        
        if dimension in self.levels:
            self.levels[dimension].current = max(level, self.levels[dimension].minimum)
            self.levels[dimension].reason = reason
            self.manual_overrides[dimension] = level
            
            logger.info(f"🎮 Override autonomia {dimension.value}: {level} ({reason})")
    
    def clear_override(self, dimension: AutonomyDimension):
        """Rimuove override manuale"""
        
        if dimension in self.manual_overrides:
            del self.manual_overrides[dimension]
            # Ricalcola con crisis level
            self.set_crisis_level(self.crisis_level)
    
    def can_perform(
        self,
        action_type: str,
        required_autonomy: float = 0.5,
        is_irreversible: bool = False
    ) -> tuple[bool, str]:
        """
        Verifica se un'azione può essere eseguita.
        
        Returns:
            (può_eseguire, motivo)
        """
        
        # Check irreversibilità
        if is_irreversible:
            irr_level = self.levels[AutonomyDimension.IRREVERSIBILITY].current
            if irr_level < required_autonomy:
                return False, f"Azione irreversibile bloccata (autonomia: {irr_level:.1%})"
        
        # Check azioni
        action_level = self.levels[AutonomyDimension.ACTIONS].current
        if action_level < required_autonomy:
            return False, f"Autonomia azioni insufficiente ({action_level:.1%} < {required_autonomy:.1%})"
        
        return True, "OK"
    
    def get_suggestion_throttle(self) -> float:
        """Ritorna fattore di throttle per suggerimenti"""
        return self.levels[AutonomyDimension.SUGGESTIONS].current
    
    def get_speed_factor(self) -> float:
        """Ritorna fattore di velocità"""
        return self.levels[AutonomyDimension.SPEED].current
    
    def get_all_levels(self) -> Dict[str, Dict[str, Any]]:
        """Ritorna tutti i livelli"""
        return {
            dim.value: {
                'current': lvl.current,
                'baseline': lvl.baseline,
                'minimum': lvl.minimum,
                'reason': lvl.reason
            }
            for dim, lvl in self.levels.items()
        }
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        levels_str = ""
        for dim, lvl in self.levels.items():
            bar = "█" * int(lvl.current * 10) + "░" * (10 - int(lvl.current * 10))
            override = " 🔧" if dim in self.manual_overrides else ""
            levels_str += f"| {dim.value:15} | {bar} {lvl.current:.0%} |{override}\n"
        
        return f"""
## 🎮 Autonomy Clamp

**Crisis Level:** {self.crisis_level}
**Reduction Factor:** {self.CRISIS_REDUCTION.get(self.crisis_level, 1.0):.0%}

### Livelli Autonomia
| Dimensione | Livello |
|------------|---------|
{levels_str}

### Override Attivi
{chr(10).join(f"- **{d.value}**: {v:.0%}" for d, v in self.manual_overrides.items()) or "Nessuno"}
"""


# ============================================================
# TEMPORAL GOVERNOR
# ============================================================

class TemporalGovernor:
    """
    Gestisce il tempo come risorsa.
    
    - Introduce micro-delay strategici
    - Impedisce escalation impulsive
    - Segnala finestre critiche
    
    "Slow is Smooth, Smooth is Fast"
    """
    
    # Delay di default per tipo azione (secondi)
    DEFAULT_DELAYS = {
        'decision': 2.0,
        'action': 1.0,
        'suggestion': 0.5,
        'response': 0.3,
        'irreversible': 5.0,
        'external_call': 1.5
    }
    
    # Moltiplicatori per crisis level
    CRISIS_MULTIPLIERS = {
        'NONE': 1.0,
        'WATCH': 1.2,
        'SOFT': 1.5,
        'ACTIVE': 2.0,
        'CRITICAL': 3.0
    }
    
    def __init__(self):
        self.constraints: Dict[str, TemporalConstraint] = {}
        self.crisis_level: str = 'NONE'
        self.last_actions: Dict[str, datetime] = {}
        
        # Cooldown tracking
        self.cooldowns: Dict[str, timedelta] = {}
        
        # Critical windows
        self.critical_windows: List[tuple] = []  # (start, end, reason)
    
    def set_crisis_level(self, level: str):
        """Aggiorna livello crisi"""
        self.crisis_level = level
    
    def add_constraint(
        self,
        constraint_id: str,
        description: str,
        delay_seconds: float,
        applies_to: List[str]
    ):
        """Aggiunge vincolo temporale"""
        
        self.constraints[constraint_id] = TemporalConstraint(
            constraint_id=constraint_id,
            description=description,
            delay_seconds=delay_seconds,
            applies_to=applies_to
        )
    
    def remove_constraint(self, constraint_id: str):
        """Rimuove vincolo temporale"""
        if constraint_id in self.constraints:
            del self.constraints[constraint_id]
    
    def get_required_delay(self, action_type: str) -> float:
        """
        Calcola delay richiesto per tipo azione.
        
        Considera:
        - Delay base
        - Vincoli custom
        - Crisis multiplier
        """
        
        # Delay base
        base_delay = self.DEFAULT_DELAYS.get(action_type, 0.5)
        
        # Aggiungi vincoli custom
        for constraint in self.constraints.values():
            if constraint.active and action_type in constraint.applies_to:
                base_delay = max(base_delay, constraint.delay_seconds)
        
        # Applica multiplier crisi
        multiplier = self.CRISIS_MULTIPLIERS.get(self.crisis_level, 1.0)
        
        return base_delay * multiplier
    
    async def enforce_delay(self, action_type: str) -> float:
        """
        Applica delay richiesto.
        
        Returns:
            Delay effettivo applicato (secondi)
        """
        
        delay = self.get_required_delay(action_type)
        
        if delay > 0:
            logger.debug(f"⏱️ Delay {delay:.1f}s per {action_type}")
            await asyncio.sleep(delay)
        
        # Registra azione
        self.last_actions[action_type] = datetime.now()
        
        return delay
    
    def set_cooldown(self, action_type: str, duration: timedelta):
        """Imposta cooldown per tipo azione"""
        self.cooldowns[action_type] = duration
    
    def is_in_cooldown(self, action_type: str) -> tuple[bool, float]:
        """
        Verifica se azione è in cooldown.
        
        Returns:
            (in_cooldown, secondi_rimanenti)
        """
        
        if action_type not in self.cooldowns:
            return False, 0
        
        last = self.last_actions.get(action_type)
        if not last:
            return False, 0
        
        cooldown = self.cooldowns[action_type]
        elapsed = datetime.now() - last
        
        if elapsed < cooldown:
            remaining = (cooldown - elapsed).total_seconds()
            return True, remaining
        
        return False, 0
    
    def add_critical_window(
        self,
        start: datetime,
        end: datetime,
        reason: str
    ):
        """Definisce finestra temporale critica"""
        self.critical_windows.append((start, end, reason))
    
    def is_in_critical_window(self) -> tuple[bool, Optional[str]]:
        """Verifica se siamo in finestra critica"""
        
        now = datetime.now()
        
        for start, end, reason in self.critical_windows:
            if start <= now <= end:
                return True, reason
        
        return False, None
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        in_critical, critical_reason = self.is_in_critical_window()
        
        constraints_str = ""
        for cid, c in self.constraints.items():
            status = "✅" if c.active else "❌"
            constraints_str += f"- {status} **{cid}**: {c.delay_seconds}s ({c.description})\n"
        
        return f"""
## ⏱️ Temporal Governor

**Crisis Level:** {self.crisis_level}
**Multiplier:** {self.CRISIS_MULTIPLIERS.get(self.crisis_level, 1.0)}x
**In Critical Window:** {'⚠️ Sì - ' + critical_reason if in_critical else '✅ No'}

### Delay Attuali
| Tipo Azione | Delay Base | Delay Effettivo |
|-------------|------------|-----------------|
{chr(10).join(f"| {t} | {d}s | {d * self.CRISIS_MULTIPLIERS.get(self.crisis_level, 1.0):.1f}s |" for t, d in self.DEFAULT_DELAYS.items())}

### Vincoli Custom
{constraints_str or "Nessuno"}

### Cooldown Attivi
{chr(10).join(f"- **{t}**: {cd.total_seconds()}s" for t, cd in self.cooldowns.items()) or "Nessuno"}
"""


# ============================================================
# SAFE STATE ENFORCER
# ============================================================

class SafeStateEnforcer:
    """
    Garantisce Safe State quando necessario.
    
    Se:
    - Dati corrotti
    - Tool offline
    - Conflitti gravi
    
    👉 Entra in Safe State:
    - Solo analisi
    - Solo spiegazioni
    - Zero azioni attive
    
    Safe-State > Wrong-State
    """
    
    # Trigger automatici per safe state
    AUTO_TRIGGERS = [
        'data_corruption',
        'multiple_tool_failure',
        'critical_conflict',
        'human_override',
        'emergency_stop'
    ]
    
    def __init__(self):
        self.config = SafeStateConfig()
        self.is_active = False
        self.trigger_reason: Optional[str] = None
        self.trigger_history: List[Dict[str, Any]] = []
        
        # Blocked actions tracking
        self.blocked_actions: List[Dict[str, Any]] = []
    
    def activate(self, reason: str, config: Optional[SafeStateConfig] = None):
        """Attiva Safe State"""
        
        if config:
            self.config = config
        else:
            # Config di default per safe state
            self.config = SafeStateConfig(
                allow_analysis=True,
                allow_explanations=True,
                allow_read_only=True,
                allow_suggestions=False,
                allow_actions=False,
                allow_external_calls=False,
                reason=reason
            )
        
        self.config.activated_at = datetime.now()
        self.is_active = True
        self.trigger_reason = reason
        
        self.trigger_history.append({
            'reason': reason,
            'activated_at': datetime.now().isoformat(),
            'config': {
                'analysis': self.config.allow_analysis,
                'explanations': self.config.allow_explanations,
                'actions': self.config.allow_actions
            }
        })
        
        logger.warning(f"🛡️ SAFE STATE ATTIVATO: {reason}")
    
    def deactivate(self, confirmation_code: Optional[str] = None):
        """Disattiva Safe State"""
        
        # In produzione richiedere confirmation_code
        
        self.is_active = False
        deactivation_time = datetime.now()
        
        duration = None
        if self.config.activated_at:
            duration = (deactivation_time - self.config.activated_at).total_seconds()
        
        self.trigger_history.append({
            'event': 'deactivation',
            'deactivated_at': deactivation_time.isoformat(),
            'duration_seconds': duration,
            'previous_reason': self.trigger_reason
        })
        
        self.trigger_reason = None
        
        logger.info(f"🛡️ Safe State disattivato (durata: {duration:.0f}s)" if duration else "🛡️ Safe State disattivato")
    
    def check_permission(self, action_type: str) -> tuple[bool, str]:
        """
        Verifica se azione è permessa.
        
        Returns:
            (permesso, motivo)
        """
        
        if not self.is_active:
            return True, "Safe State non attivo"
        
        # Mapping azioni a permessi
        permission_map = {
            'analysis': self.config.allow_analysis,
            'explanation': self.config.allow_explanations,
            'read': self.config.allow_read_only,
            'suggestion': self.config.allow_suggestions,
            'action': self.config.allow_actions,
            'write': self.config.allow_actions,
            'execute': self.config.allow_actions,
            'external': self.config.allow_external_calls,
            'api_call': self.config.allow_external_calls
        }
        
        # Trova permesso applicabile
        allowed = True
        for key, perm in permission_map.items():
            if key in action_type.lower():
                allowed = perm
                break
        
        if not allowed:
            self._log_blocked(action_type)
            return False, f"Bloccato da Safe State: {self.config.reason}"
        
        return True, "Permesso"
    
    def _log_blocked(self, action_type: str):
        """Log azione bloccata"""
        
        self.blocked_actions.append({
            'action_type': action_type,
            'blocked_at': datetime.now().isoformat(),
            'reason': self.config.reason
        })
        
        # Limita history
        if len(self.blocked_actions) > 100:
            self.blocked_actions = self.blocked_actions[-100:]
    
    def get_allowed_operations(self) -> List[str]:
        """Lista operazioni permesse in safe state"""
        
        if not self.is_active:
            return ["ALL"]
        
        allowed = []
        if self.config.allow_analysis:
            allowed.append("Analysis")
        if self.config.allow_explanations:
            allowed.append("Explanations")
        if self.config.allow_read_only:
            allowed.append("Read-only operations")
        if self.config.allow_suggestions:
            allowed.append("Suggestions")
        if self.config.allow_actions:
            allowed.append("Actions")
        if self.config.allow_external_calls:
            allowed.append("External API calls")
        
        return allowed or ["NONE"]
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        if not self.is_active:
            return """
## 🛡️ Safe State Enforcer

**Status:** ✅ Non attivo
**Operazioni:** Tutte permesse
"""
        
        duration = ""
        if self.config.activated_at:
            elapsed = (datetime.now() - self.config.activated_at).total_seconds()
            duration = f"\n**Durata:** {elapsed:.0f} secondi"
        
        allowed_ops = self.get_allowed_operations()
        
        return f"""
## 🛡️ Safe State Enforcer

**Status:** 🔴 ATTIVO
**Motivo:** {self.config.reason}{duration}

### Operazioni Permesse
{chr(10).join(f"- ✅ {op}" for op in allowed_ops)}

### Operazioni Bloccate
- ❌ Actions: {'Bloccate' if not self.config.allow_actions else 'Permesse'}
- ❌ External calls: {'Bloccate' if not self.config.allow_external_calls else 'Permesse'}
- ❌ Suggestions: {'Bloccate' if not self.config.allow_suggestions else 'Permesse'}

### Azioni Bloccate Recenti ({len(self.blocked_actions)})
{chr(10).join(f"- {a['action_type']} ({a['blocked_at'][:19]})" for a in self.blocked_actions[-5:]) or "Nessuna"}
"""


# ============================================================
# UNIFIED CONTROL LAYER
# ============================================================

class ControlLayer:
    """
    Control Layer unificato.
    
    Integra:
    - AutonomyClamp
    - TemporalGovernor
    - SafeStateEnforcer
    """
    
    def __init__(self):
        self.autonomy = AutonomyClamp()
        self.temporal = TemporalGovernor()
        self.safe_state = SafeStateEnforcer()
        self.state = ControlState.NORMAL
    
    def set_crisis_level(self, level: str):
        """Propaga crisis level a tutti i componenti"""
        
        self.autonomy.set_crisis_level(level)
        self.temporal.set_crisis_level(level)
        
        # Determina stato
        if self.safe_state.is_active:
            self.state = ControlState.SAFE_STATE
        elif level == 'CRITICAL':
            self.state = ControlState.RESTRICTED
        elif level in ['ACTIVE', 'SOFT']:
            self.state = ControlState.MODERATED
        else:
            self.state = ControlState.NORMAL
    
    def activate_safe_state(self, reason: str):
        """Attiva safe state e aggiorna stato"""
        self.safe_state.activate(reason)
        self.state = ControlState.SAFE_STATE
    
    def deactivate_safe_state(self):
        """Disattiva safe state"""
        self.safe_state.deactivate()
        # Ricalcola stato basato su crisis level
        self.set_crisis_level(self.autonomy.crisis_level)
    
    async def request_permission(
        self,
        action_type: str,
        required_autonomy: float = 0.5,
        is_irreversible: bool = False
    ) -> tuple[bool, str, float]:
        """
        Richiede permesso per azione.
        
        Returns:
            (permesso, motivo, delay_applicato)
        """
        
        # 1. Check safe state
        safe_ok, safe_reason = self.safe_state.check_permission(action_type)
        if not safe_ok:
            return False, safe_reason, 0
        
        # 2. Check autonomy
        auto_ok, auto_reason = self.autonomy.can_perform(
            action_type, required_autonomy, is_irreversible
        )
        if not auto_ok:
            return False, auto_reason, 0
        
        # 3. Check cooldown
        in_cooldown, remaining = self.temporal.is_in_cooldown(action_type)
        if in_cooldown:
            return False, f"In cooldown ({remaining:.1f}s rimanenti)", 0
        
        # 4. Applica delay
        delay = await self.temporal.enforce_delay(action_type)
        
        return True, "Permesso", delay
    
    def format_status(self) -> str:
        """Formatta status completo"""
        
        state_emoji = {
            ControlState.NORMAL: "🟢",
            ControlState.MODERATED: "🟡",
            ControlState.RESTRICTED: "🟠",
            ControlState.SAFE_STATE: "🔴"
        }
        
        return f"""
# 🎮 CAM Control Layer

## Stato Globale
{state_emoji[self.state]} **{self.state.value.upper()}**

---

{self.autonomy.format_status()}

---

{self.temporal.format_status()}

---

{self.safe_state.format_status()}
"""


# Singleton
_control_layer: Optional[ControlLayer] = None


def get_control_layer() -> ControlLayer:
    """Ottiene istanza singleton"""
    global _control_layer
    if _control_layer is None:
        _control_layer = ControlLayer()
    return _control_layer
