"""
🚧 AUTONOMY BOUNDARY CONTROLLER
===============================
GIDEON gestisce confini di autonomia dinamici:
- Negozia livelli di autonomia
- Adatta autorizzazioni a contesto
- Escalation graduale
- Trasparenza sui limiti

"Per questa azione mi serve autorizzazione - ecco cosa farei se confermato"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AutonomyLevel(Enum):
    """Livelli di autonomia"""
    NONE = 0  # Solo suggerimenti, nessuna azione
    MINIMAL = 1  # Azioni reversibili minori
    STANDARD = 2  # Azioni normali con notifica
    ELEVATED = 3  # Azioni significative autonome
    FULL = 4  # Autonomia completa


class ActionCategory(Enum):
    """Categorie di azioni"""
    READ = "read"  # Lettura dati
    WRITE = "write"  # Scrittura file
    EXECUTE = "execute"  # Esecuzione comandi
    NETWORK = "network"  # Operazioni rete
    SYSTEM = "system"  # Operazioni sistema
    DATA_MODIFY = "data_modify"  # Modifica dati
    DATA_DELETE = "data_delete"  # Cancellazione dati
    CONFIG_CHANGE = "config_change"  # Cambio configurazione
    EXTERNAL_API = "external_api"  # API esterne
    SECURITY = "security"  # Operazioni sicurezza


class EscalationReason(Enum):
    """Motivi di escalation"""
    RISK_TOO_HIGH = "risk_too_high"
    OUTSIDE_BOUNDARY = "outside_boundary"
    USER_OVERRIDE_REQUIRED = "user_override"
    POLICY_VIOLATION = "policy_violation"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    FIRST_TIME_ACTION = "first_time"


@dataclass
class AutonomyBoundary:
    """Confine di autonomia per categoria"""
    category: ActionCategory
    max_level: AutonomyLevel
    conditions: List[str]  # Condizioni per il livello
    cooldown_minutes: int = 0  # Tempo tra azioni autonome
    requires_logging: bool = True
    last_action: Optional[datetime] = None


@dataclass
class AutonomyRequest:
    """Richiesta di autorizzazione"""
    request_id: str
    category: ActionCategory
    action_description: str
    current_level: AutonomyLevel
    required_level: AutonomyLevel
    risk_score: float
    context: Dict[str, Any]
    what_would_happen: str
    alternatives: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    responded: bool = False
    approved: Optional[bool] = None


@dataclass
class AutonomyPolicy:
    """Policy di autonomia"""
    policy_id: str
    name: str
    description: str
    boundaries: Dict[ActionCategory, AutonomyBoundary]
    effective_from: datetime = field(default_factory=datetime.now)
    effective_until: Optional[datetime] = None
    created_by: str = "system"


@dataclass
class ActionProposal:
    """Proposta di azione"""
    proposal_id: str
    action: str
    category: ActionCategory
    risk_score: float
    reversible: bool
    what_it_does: str
    why_proposed: str
    alternatives: List[str]
    requires_approval: bool
    created_at: datetime = field(default_factory=datetime.now)


class AutonomyBoundaryController:
    """
    Controller per confini di autonomia dinamici.
    
    Gestisce:
    - Livelli di autonomia per categoria
    - Escalation e richieste approvazione
    - Negoziazione dinamica
    - Audit delle azioni
    """
    
    # Rischio base per categoria
    BASE_RISK = {
        ActionCategory.READ: 0.1,
        ActionCategory.WRITE: 0.4,
        ActionCategory.EXECUTE: 0.6,
        ActionCategory.NETWORK: 0.5,
        ActionCategory.SYSTEM: 0.7,
        ActionCategory.DATA_MODIFY: 0.5,
        ActionCategory.DATA_DELETE: 0.8,
        ActionCategory.CONFIG_CHANGE: 0.6,
        ActionCategory.EXTERNAL_API: 0.5,
        ActionCategory.SECURITY: 0.9
    }
    
    def __init__(self):
        self.current_policy: Optional[AutonomyPolicy] = None
        self.pending_requests: Dict[str, AutonomyRequest] = {}
        self.action_history: List[Dict[str, Any]] = []
        self.temporary_elevations: Dict[ActionCategory, Tuple[AutonomyLevel, datetime]] = {}
        
        # Inizializza policy di default
        self._init_default_policy()
    
    def _init_default_policy(self):
        """Inizializza policy di default"""
        
        default_boundaries = {
            ActionCategory.READ: AutonomyBoundary(
                category=ActionCategory.READ,
                max_level=AutonomyLevel.ELEVATED,
                conditions=["File nel workspace"],
                requires_logging=False
            ),
            ActionCategory.WRITE: AutonomyBoundary(
                category=ActionCategory.WRITE,
                max_level=AutonomyLevel.STANDARD,
                conditions=["File backup disponibile", "In workspace"],
                cooldown_minutes=1,
                requires_logging=True
            ),
            ActionCategory.EXECUTE: AutonomyBoundary(
                category=ActionCategory.EXECUTE,
                max_level=AutonomyLevel.MINIMAL,
                conditions=["Comandi safe-list", "Non distruttivi"],
                cooldown_minutes=5,
                requires_logging=True
            ),
            ActionCategory.NETWORK: AutonomyBoundary(
                category=ActionCategory.NETWORK,
                max_level=AutonomyLevel.STANDARD,
                conditions=["URL trusted", "Solo GET"],
                requires_logging=True
            ),
            ActionCategory.SYSTEM: AutonomyBoundary(
                category=ActionCategory.SYSTEM,
                max_level=AutonomyLevel.NONE,
                conditions=["Richiede sempre approvazione"],
                requires_logging=True
            ),
            ActionCategory.DATA_MODIFY: AutonomyBoundary(
                category=ActionCategory.DATA_MODIFY,
                max_level=AutonomyLevel.STANDARD,
                conditions=["Backup automatico", "Modifiche incrementali"],
                cooldown_minutes=2,
                requires_logging=True
            ),
            ActionCategory.DATA_DELETE: AutonomyBoundary(
                category=ActionCategory.DATA_DELETE,
                max_level=AutonomyLevel.NONE,
                conditions=["Richiede sempre approvazione"],
                requires_logging=True
            ),
            ActionCategory.CONFIG_CHANGE: AutonomyBoundary(
                category=ActionCategory.CONFIG_CHANGE,
                max_level=AutonomyLevel.MINIMAL,
                conditions=["Solo impostazioni non critiche"],
                cooldown_minutes=10,
                requires_logging=True
            ),
            ActionCategory.EXTERNAL_API: AutonomyBoundary(
                category=ActionCategory.EXTERNAL_API,
                max_level=AutonomyLevel.STANDARD,
                conditions=["API whitelisted", "Rate limit rispettato"],
                cooldown_minutes=1,
                requires_logging=True
            ),
            ActionCategory.SECURITY: AutonomyBoundary(
                category=ActionCategory.SECURITY,
                max_level=AutonomyLevel.NONE,
                conditions=["Richiede sempre approvazione"],
                requires_logging=True
            )
        }
        
        self.current_policy = AutonomyPolicy(
            policy_id="default_policy",
            name="Policy Standard GIDEON",
            description="Policy di default con autonomia moderata",
            boundaries=default_boundaries
        )
    
    def can_act(
        self,
        category: ActionCategory,
        required_level: AutonomyLevel = AutonomyLevel.STANDARD,
        risk_score: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifica se può agire autonomamente.
        
        Returns:
            (può_agire, motivo_se_no)
        """
        
        if not self.current_policy:
            return False, "Nessuna policy attiva"
        
        boundary = self.current_policy.boundaries.get(category)
        if not boundary:
            return False, f"Categoria {category.value} non configurata"
        
        # Check elevazione temporanea
        if category in self.temporary_elevations:
            elevated_level, expires = self.temporary_elevations[category]
            if datetime.now() < expires:
                if elevated_level.value >= required_level.value:
                    return True, None
        
        # Check livello
        if boundary.max_level.value < required_level.value:
            return False, f"Livello richiesto ({required_level.value}) supera massimo ({boundary.max_level.value})"
        
        # Check cooldown
        if boundary.cooldown_minutes > 0 and boundary.last_action:
            cooldown_end = boundary.last_action + timedelta(minutes=boundary.cooldown_minutes)
            if datetime.now() < cooldown_end:
                remaining = (cooldown_end - datetime.now()).seconds
                return False, f"Cooldown attivo ({remaining}s rimanenti)"
        
        # Check rischio
        if risk_score is None:
            risk_score = self.BASE_RISK.get(category, 0.5)
        
        # Rischio alto richiede livello più alto
        if risk_score > 0.7 and required_level.value < AutonomyLevel.ELEVATED.value:
            return False, f"Rischio alto ({risk_score:.1%}) richiede approvazione"
        
        return True, None
    
    def request_approval(
        self,
        category: ActionCategory,
        action_description: str,
        what_would_happen: str,
        risk_score: Optional[float] = None,
        alternatives: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AutonomyRequest:
        """Richiede approvazione per azione"""
        
        if risk_score is None:
            risk_score = self.BASE_RISK.get(category, 0.5)
        
        boundary = self.current_policy.boundaries.get(category) if self.current_policy else None
        current_level = boundary.max_level if boundary else AutonomyLevel.NONE
        
        # Determina livello richiesto basato su rischio
        if risk_score > 0.8:
            required_level = AutonomyLevel.FULL
        elif risk_score > 0.6:
            required_level = AutonomyLevel.ELEVATED
        elif risk_score > 0.4:
            required_level = AutonomyLevel.STANDARD
        else:
            required_level = AutonomyLevel.MINIMAL
        
        request = AutonomyRequest(
            request_id=f"req_{datetime.now().timestamp()}",
            category=category,
            action_description=action_description,
            current_level=current_level,
            required_level=required_level,
            risk_score=risk_score,
            context=context or {},
            what_would_happen=what_would_happen,
            alternatives=alternatives or []
        )
        
        self.pending_requests[request.request_id] = request
        
        logger.info(f"🚧 Richiesta approvazione: {action_description}")
        
        return request
    
    def approve_request(self, request_id: str, grant_temporary_elevation: bool = False) -> bool:
        """Approva richiesta"""
        
        request = self.pending_requests.get(request_id)
        if not request:
            return False
        
        request.responded = True
        request.approved = True
        
        # Log
        self._log_action(request, approved=True)
        
        # Elevazione temporanea
        if grant_temporary_elevation:
            self._grant_temporary_elevation(
                request.category,
                request.required_level,
                duration_minutes=30
            )
        
        logger.info(f"✅ Richiesta approvata: {request.action_description}")
        
        return True
    
    def deny_request(self, request_id: str, reason: Optional[str] = None) -> bool:
        """Nega richiesta"""
        
        request = self.pending_requests.get(request_id)
        if not request:
            return False
        
        request.responded = True
        request.approved = False
        
        # Log
        self._log_action(request, approved=False, reason=reason)
        
        logger.info(f"❌ Richiesta negata: {request.action_description}")
        
        return True
    
    def _grant_temporary_elevation(
        self,
        category: ActionCategory,
        level: AutonomyLevel,
        duration_minutes: int
    ):
        """Concede elevazione temporanea"""
        
        expires = datetime.now() + timedelta(minutes=duration_minutes)
        self.temporary_elevations[category] = (level, expires)
        
        logger.info(f"🔓 Elevazione temporanea: {category.value} -> {level.value} per {duration_minutes}min")
    
    def _log_action(
        self,
        request: AutonomyRequest,
        approved: bool,
        reason: Optional[str] = None
    ):
        """Log azione per audit"""
        
        self.action_history.append({
            'request_id': request.request_id,
            'category': request.category.value,
            'action': request.action_description,
            'risk_score': request.risk_score,
            'approved': approved,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limita storia
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]
    
    def mark_action_completed(self, category: ActionCategory):
        """Marca azione come completata (per cooldown)"""
        
        if self.current_policy and category in self.current_policy.boundaries:
            self.current_policy.boundaries[category].last_action = datetime.now()
    
    def create_proposal(
        self,
        action: str,
        category: ActionCategory,
        what_it_does: str,
        why_proposed: str,
        alternatives: Optional[List[str]] = None,
        risk_score: Optional[float] = None,
        reversible: bool = True
    ) -> ActionProposal:
        """Crea proposta di azione"""
        
        if risk_score is None:
            risk_score = self.BASE_RISK.get(category, 0.5)
        
        can_do, _ = self.can_act(category)
        
        proposal = ActionProposal(
            proposal_id=f"prop_{datetime.now().timestamp()}",
            action=action,
            category=category,
            risk_score=risk_score,
            reversible=reversible,
            what_it_does=what_it_does,
            why_proposed=why_proposed,
            alternatives=alternatives or [],
            requires_approval=not can_do
        )
        
        return proposal
    
    def format_proposal(self, proposal: ActionProposal) -> str:
        """Formatta proposta per visualizzazione"""
        
        risk_emoji = "🟢" if proposal.risk_score < 0.3 else "🟡" if proposal.risk_score < 0.6 else "🟠" if proposal.risk_score < 0.8 else "🔴"
        
        alternatives_str = ""
        if proposal.alternatives:
            alternatives_str = "\n**Alternative:**\n" + "\n".join(f"- {a}" for a in proposal.alternatives)
        
        return f"""
## {'🚧 Richiesta Approvazione' if proposal.requires_approval else '⚡ Proposta Azione'}

**Azione:** {proposal.action}

**Categoria:** {proposal.category.value}
**Rischio:** {risk_emoji} {proposal.risk_score:.1%}
**Reversibile:** {'✅ Sì' if proposal.reversible else '⚠️ No'}

### Cosa farebbe
{proposal.what_it_does}

### Perché proposta
{proposal.why_proposed}

{alternatives_str}

{f'> ⚠️ **Richiede approvazione esplicita**' if proposal.requires_approval else '> ✅ Può procedere autonomamente'}
"""
    
    def format_request(self, request: AutonomyRequest) -> str:
        """Formatta richiesta per visualizzazione"""
        
        risk_emoji = "🟢" if request.risk_score < 0.3 else "🟡" if request.risk_score < 0.6 else "🟠" if request.risk_score < 0.8 else "🔴"
        
        alternatives_str = ""
        if request.alternatives:
            alternatives_str = "\n**Alternative disponibili:**\n" + "\n".join(f"- {a}" for a in request.alternatives)
        
        return f"""
## 🚧 Richiesta Autorizzazione

**ID:** `{request.request_id}`
**Azione:** {request.action_description}

**Categoria:** {request.category.value}
**Livello corrente:** {request.current_level.name}
**Livello richiesto:** {request.required_level.name}
**Rischio:** {risk_emoji} {request.risk_score:.1%}

### Cosa accadrebbe
{request.what_would_happen}

{alternatives_str}

### Comandi
- **Approva:** conferma questa azione
- **Nega:** rifiuta e suggerisci alternativa
- **Eleva:** approva e aumenta autonomia per questa categoria
"""
    
    def get_current_boundaries(self) -> Dict[str, Dict[str, Any]]:
        """Ottiene confini correnti"""
        
        if not self.current_policy:
            return {}
        
        result = {}
        for cat, boundary in self.current_policy.boundaries.items():
            # Check elevazione temporanea
            effective_level = boundary.max_level
            temp_expires = None
            
            if cat in self.temporary_elevations:
                temp_level, expires = self.temporary_elevations[cat]
                if datetime.now() < expires:
                    effective_level = temp_level
                    temp_expires = expires.isoformat()
            
            result[cat.value] = {
                'max_level': boundary.max_level.name,
                'effective_level': effective_level.name,
                'conditions': boundary.conditions,
                'cooldown_minutes': boundary.cooldown_minutes,
                'requires_logging': boundary.requires_logging,
                'temporary_elevation_expires': temp_expires
            }
        
        return result
    
    def get_pending_requests_count(self) -> int:
        """Conta richieste pendenti"""
        return sum(1 for r in self.pending_requests.values() if not r.responded)
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        boundaries = self.get_current_boundaries()
        pending = self.get_pending_requests_count()
        
        boundaries_str = ""
        for cat, info in boundaries.items():
            level_emoji = "🟢" if info['effective_level'] in ['ELEVATED', 'FULL'] else "🟡" if info['effective_level'] == 'STANDARD' else "🟠" if info['effective_level'] == 'MINIMAL' else "🔴"
            temp_indicator = " ⏱️" if info.get('temporary_elevation_expires') else ""
            boundaries_str += f"- {level_emoji} **{cat}**: {info['effective_level']}{temp_indicator}\n"
        
        return f"""
# 🚧 Autonomy Boundary Controller

## Policy Attiva
**{self.current_policy.name if self.current_policy else 'Nessuna'}**

## Confini Correnti
{boundaries_str}

## Statistiche
| Metrica | Valore |
|---------|--------|
| Richieste pendenti | {pending} |
| Azioni loggate | {len(self.action_history)} |
| Elevazioni temporanee | {len(self.temporary_elevations)} |

## Elevazioni Temporanee Attive
{chr(10).join(f"- **{cat.value}**: {level.name} (scade: {exp.strftime('%H:%M')})" for cat, (level, exp) in self.temporary_elevations.items() if datetime.now() < exp) or '- Nessuna'}
"""


# Import necessario
from typing import Tuple


# Singleton
_autonomy_controller: Optional[AutonomyBoundaryController] = None


def get_autonomy_controller() -> AutonomyBoundaryController:
    """Ottiene istanza singleton"""
    global _autonomy_controller
    if _autonomy_controller is None:
        _autonomy_controller = AutonomyBoundaryController()
    return _autonomy_controller
