"""
⏸️ DECISION FREEZE CONTROLLER
==============================
Sospende o rallenta decisioni automatiche quando:
- Conflitto tra tool
- Confidenza sotto soglia
- Escalation emotiva negli input
- Dati insufficienti

"Fermare una decisione sbagliata salva più di mille decisioni veloci"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class FreezeReason(Enum):
    """Motivi per il freeze delle decisioni"""
    TOOL_CONFLICT = "conflitto_tra_tool"
    LOW_CONFIDENCE = "confidenza_insufficiente"
    EMOTIONAL_ESCALATION = "escalation_emotiva"
    INSUFFICIENT_DATA = "dati_insufficienti"
    ETHICAL_CONCERN = "preoccupazione_etica"
    TIME_PRESSURE = "pressione_temporale_sospetta"
    CONTRADICTION = "contraddizione_rilevata"
    HUMAN_OVERRIDE = "override_umano"
    SYSTEM_STRESS = "stress_sistema"


class DecisionMode(Enum):
    """Modalità di decisione"""
    NORMAL = "normale"
    SLOW = "rallentato"
    FROZEN = "congelato"
    HUMAN_REQUIRED = "richiede_umano"


@dataclass
class FreezeEvent:
    """Evento di freeze"""
    reason: FreezeReason
    confidence: float
    details: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None


@dataclass
class PendingDecision:
    """Decisione in attesa"""
    decision_id: str
    description: str
    options: List[Dict[str, Any]]
    confidence: float
    freeze_reason: FreezeReason
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    human_input_required: bool = True
    auto_resolve_to: Optional[str] = None  # Opzione default se scade


class DecisionFreezeController:
    """
    Controller per freeze e slow-mode delle decisioni.
    
    Filosofia:
    - Il dubbio è più sicuro della certezza falsa
    - Meglio chiedere che sbagliare
    - La velocità non è sempre virtù
    """
    
    # Soglie configurabili
    CONFIDENCE_THRESHOLD = 0.65  # Sotto questa soglia → slow mode
    CRITICAL_CONFIDENCE = 0.40   # Sotto questa soglia → freeze
    EMOTIONAL_KEYWORDS = [
        'urgente', 'subito', 'emergenza', 'adesso', 'immediatamente',
        'critico', 'disastro', 'panico', 'aiuto', 'pericolo',
        'urgent', 'now', 'emergency', 'critical', 'help', 'danger'
    ]
    
    def __init__(self):
        self.mode = DecisionMode.NORMAL
        self.freeze_events: List[FreezeEvent] = []
        self.pending_decisions: Dict[str, PendingDecision] = {}
        self.decision_history: List[Dict[str, Any]] = []
        self.slow_mode_delay = 2.0  # secondi di delay in slow mode
        self._decision_counter = 0
        
    async def evaluate_decision(
        self,
        action: str,
        confidence: float,
        tool_outputs: List[Dict[str, Any]],
        user_input: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Valuta se una decisione può procedere.
        
        Returns:
            {
                'can_proceed': bool,
                'mode': DecisionMode,
                'reason': Optional[str],
                'pending_id': Optional[str],
                'recommendation': str
            }
        """
        reasons = []
        
        # Check 1: Confidenza
        if confidence < self.CRITICAL_CONFIDENCE:
            reasons.append((FreezeReason.LOW_CONFIDENCE, 
                           f"Confidenza {confidence:.1%} sotto soglia critica"))
        elif confidence < self.CONFIDENCE_THRESHOLD:
            reasons.append((FreezeReason.LOW_CONFIDENCE,
                           f"Confidenza {confidence:.1%} sotto soglia"))
        
        # Check 2: Conflitto tra tool
        if len(tool_outputs) > 1:
            conflict = self._detect_tool_conflict(tool_outputs)
            if conflict:
                reasons.append((FreezeReason.TOOL_CONFLICT, conflict))
        
        # Check 3: Escalation emotiva
        if user_input:
            emotional = self._detect_emotional_escalation(user_input)
            if emotional:
                reasons.append((FreezeReason.EMOTIONAL_ESCALATION, emotional))
        
        # Check 4: Contraddizioni nel contesto
        if context:
            contradiction = self._detect_contradictions(context)
            if contradiction:
                reasons.append((FreezeReason.CONTRADICTION, contradiction))
        
        # Check 5: Pressione temporale sospetta
        if user_input:
            time_pressure = self._detect_time_pressure(user_input)
            if time_pressure:
                reasons.append((FreezeReason.TIME_PRESSURE, time_pressure))
        
        # Determina modalità
        if not reasons:
            return {
                'can_proceed': True,
                'mode': DecisionMode.NORMAL,
                'reason': None,
                'pending_id': None,
                'recommendation': "Procedi con l'azione"
            }
        
        # Valuta severità
        has_critical = any(
            r[0] in [FreezeReason.LOW_CONFIDENCE, FreezeReason.TOOL_CONFLICT, 
                     FreezeReason.ETHICAL_CONCERN]
            and confidence < self.CRITICAL_CONFIDENCE
            for r in reasons
        )
        
        if has_critical or len(reasons) >= 2:
            # FREEZE - richiede intervento umano
            mode = DecisionMode.FROZEN
            pending_id = await self._create_pending_decision(
                action, confidence, reasons, tool_outputs
            )
            
            return {
                'can_proceed': False,
                'mode': mode,
                'reason': "; ".join([r[1] for r in reasons]),
                'pending_id': pending_id,
                'recommendation': self._generate_freeze_message(reasons, confidence)
            }
        else:
            # SLOW MODE - procede con delay e avviso
            mode = DecisionMode.SLOW
            await asyncio.sleep(self.slow_mode_delay)
            
            # Log evento
            for reason, detail in reasons:
                self.freeze_events.append(FreezeEvent(
                    reason=reason,
                    confidence=confidence,
                    details=detail
                ))
            
            return {
                'can_proceed': True,
                'mode': mode,
                'reason': "; ".join([r[1] for r in reasons]),
                'pending_id': None,
                'recommendation': f"⚠️ Procedo con cautela: {reasons[0][1]}"
            }
    
    def _detect_tool_conflict(self, outputs: List[Dict[str, Any]]) -> Optional[str]:
        """Rileva conflitti tra output di tool diversi"""
        if len(outputs) < 2:
            return None
            
        # Confronta raccomandazioni
        recommendations = []
        for out in outputs:
            rec = out.get('recommendation') or out.get('action') or out.get('result')
            if rec:
                recommendations.append(rec)
        
        # Cerca contraddizioni semplici
        for i, rec1 in enumerate(recommendations):
            for rec2 in recommendations[i+1:]:
                if isinstance(rec1, str) and isinstance(rec2, str):
                    # Contraddizioni ovvie
                    if ('no' in rec1.lower() and 'sì' in rec2.lower()) or \
                       ('sì' in rec1.lower() and 'no' in rec2.lower()) or \
                       ('stop' in rec1.lower() and 'proceed' in rec2.lower()):
                        return f"Conflitto: '{rec1[:50]}' vs '{rec2[:50]}'"
        
        # Confronta confidence se presenti
        confidences = [out.get('confidence', 0.5) for out in outputs]
        if max(confidences) - min(confidences) > 0.3:
            return f"Discrepanza confidenza significativa: {min(confidences):.1%} - {max(confidences):.1%}"
        
        return None
    
    def _detect_emotional_escalation(self, user_input: str) -> Optional[str]:
        """Rileva segni di escalation emotiva nell'input"""
        input_lower = user_input.lower()
        
        # Conta keyword emotive
        emotional_count = sum(1 for kw in self.EMOTIONAL_KEYWORDS if kw in input_lower)
        
        # Conta punti esclamativi e maiuscole
        exclamation_ratio = user_input.count('!') / max(len(user_input), 1)
        caps_ratio = sum(1 for c in user_input if c.isupper()) / max(len(user_input), 1)
        
        if emotional_count >= 2 or exclamation_ratio > 0.05 or caps_ratio > 0.5:
            return f"Rilevata possibile escalation emotiva (keywords: {emotional_count}, urgenza nel tono)"
        
        return None
    
    def _detect_contradictions(self, context: Dict[str, Any]) -> Optional[str]:
        """Rileva contraddizioni nel contesto"""
        # Controlla se ci sono stati recenti che contraddicono quello attuale
        history = context.get('recent_decisions', [])
        current = context.get('current_action')
        
        if history and current:
            for past in history[-3:]:  # Ultime 3 decisioni
                past_action = past.get('action', '')
                if self._actions_contradict(past_action, current):
                    return f"Possibile contraddizione con decisione recente: {past_action}"
        
        return None
    
    def _actions_contradict(self, action1: str, action2: str) -> bool:
        """Verifica se due azioni si contraddicono"""
        opposites = [
            ('enable', 'disable'),
            ('start', 'stop'),
            ('allow', 'deny'),
            ('approve', 'reject'),
            ('increase', 'decrease'),
            ('open', 'close'),
        ]
        
        a1, a2 = action1.lower(), action2.lower()
        for pos, neg in opposites:
            if (pos in a1 and neg in a2) or (neg in a1 and pos in a2):
                return True
        return False
    
    def _detect_time_pressure(self, user_input: str) -> Optional[str]:
        """Rileva pressione temporale sospetta"""
        pressure_patterns = [
            r'entro \d+ (minuti|ore|secondi)',
            r'hai solo \d+',
            r'devi farlo (subito|adesso|ora)',
            r'non c\'è tempo',
            r'deadline',
            r'scade tra',
        ]
        
        for pattern in pressure_patterns:
            if re.search(pattern, user_input.lower()):
                return "Rilevata pressione temporale - verificare se reale o artificiale"
        
        return None
    
    async def _create_pending_decision(
        self,
        action: str,
        confidence: float,
        reasons: List[tuple],
        tool_outputs: List[Dict[str, Any]]
    ) -> str:
        """Crea una decisione pending che richiede input umano"""
        self._decision_counter += 1
        decision_id = f"DEC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._decision_counter}"
        
        pending = PendingDecision(
            decision_id=decision_id,
            description=action,
            options=[
                {'id': 'proceed', 'label': 'Procedi comunque', 'risk': 'alto'},
                {'id': 'modify', 'label': 'Modifica parametri', 'risk': 'medio'},
                {'id': 'cancel', 'label': 'Annulla', 'risk': 'basso'},
                {'id': 'defer', 'label': 'Rimanda', 'risk': 'basso'},
            ],
            confidence=confidence,
            freeze_reason=reasons[0][0],
            expires_at=datetime.now() + timedelta(hours=24),
            human_input_required=True,
            auto_resolve_to='defer'  # Se scade, rimanda
        )
        
        self.pending_decisions[decision_id] = pending
        
        logger.warning(
            f"⏸️ DECISION FROZEN [{decision_id}]: {action[:100]} | "
            f"Confidence: {confidence:.1%} | Reason: {reasons[0][0].value}"
        )
        
        return decision_id
    
    def _generate_freeze_message(
        self,
        reasons: List[tuple],
        confidence: float
    ) -> str:
        """Genera messaggio esplicativo per il freeze"""
        message = f"""
🔒 **DECISIONE SOSPESA**

La decisione automatica è stata sospesa per i seguenti motivi:

"""
        for reason, detail in reasons:
            message += f"• **{reason.value}**: {detail}\n"
        
        message += f"""
📊 **Livello di confidenza**: {confidence:.1%}

⚠️ **Azione raccomandata**: Revisione umana richiesta prima di procedere.

💡 Questa sospensione protegge da decisioni affrettate o basate su dati insufficienti.
"""
        return message
    
    async def resolve_pending(
        self,
        decision_id: str,
        resolution: str,
        human_id: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Risolve una decisione pending"""
        if decision_id not in self.pending_decisions:
            return {'success': False, 'error': 'Decisione non trovata'}
        
        pending = self.pending_decisions[decision_id]
        
        # Registra risoluzione
        resolution_record = {
            'decision_id': decision_id,
            'original_action': pending.description,
            'resolution': resolution,
            'human_id': human_id,
            'notes': notes,
            'confidence_at_freeze': pending.confidence,
            'freeze_reason': pending.freeze_reason.value,
            'frozen_at': pending.created_at.isoformat(),
            'resolved_at': datetime.now().isoformat(),
            'time_frozen_seconds': (datetime.now() - pending.created_at).total_seconds()
        }
        
        self.decision_history.append(resolution_record)
        del self.pending_decisions[decision_id]
        
        logger.info(
            f"✅ DECISION RESOLVED [{decision_id}]: {resolution} by {human_id}"
        )
        
        return {
            'success': True,
            'resolution': resolution,
            'can_proceed': resolution == 'proceed',
            'record': resolution_record
        }
    
    def get_pending_decisions(self) -> List[Dict[str, Any]]:
        """Ritorna tutte le decisioni pending"""
        return [
            {
                'id': p.decision_id,
                'description': p.description,
                'options': p.options,
                'confidence': p.confidence,
                'reason': p.freeze_reason.value,
                'created_at': p.created_at.isoformat(),
                'expires_at': p.expires_at.isoformat() if p.expires_at else None,
                'time_pending_minutes': (datetime.now() - p.created_at).total_seconds() / 60
            }
            for p in self.pending_decisions.values()
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Ritorna lo stato del controller"""
        return {
            'mode': self.mode.value,
            'pending_count': len(self.pending_decisions),
            'recent_freezes': len([e for e in self.freeze_events 
                                   if (datetime.now() - e.timestamp).total_seconds() < 3600]),
            'confidence_threshold': self.CONFIDENCE_THRESHOLD,
            'critical_threshold': self.CRITICAL_CONFIDENCE,
            'slow_mode_delay': self.slow_mode_delay,
            'pending_decisions': self.get_pending_decisions(),
            'resolution_history': self.decision_history[-10:]
        }
