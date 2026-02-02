"""
⏰ PRE-DECISION AUTOMATION
===========================
GIDEON anticipa decisioni future:
- Prepara opzioni in background
- Pre-analizza scenari probabili
- Presenta raccomandazioni proattive

"Prima che tu lo chieda, ho già preparato le opzioni"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class PredictionConfidence(Enum):
    """Livelli di confidenza nella predizione"""
    VERY_HIGH = 5    # >90% probabilità
    HIGH = 4         # 70-90%
    MODERATE = 3     # 50-70%
    LOW = 2          # 30-50%
    SPECULATIVE = 1  # <30%


class DecisionCategory(Enum):
    """Categorie di decisioni anticipabili"""
    ROUTINE = "routine"           # Decisioni ricorrenti
    CONTEXTUAL = "contextual"     # Basate su contesto
    SCHEDULED = "scheduled"       # Programmate
    REACTIVE = "reactive"         # Reazione a eventi
    STRATEGIC = "strategic"       # Strategiche/importanti


@dataclass
class AnticipatedDecision:
    """Decisione anticipata"""
    decision_id: str
    category: DecisionCategory
    description: str
    trigger_conditions: List[str]
    predicted_time: Optional[datetime]
    confidence: PredictionConfidence
    prepared_options: List[Dict[str, Any]]
    background_analysis: Dict[str, Any]
    data_gathered: List[str]
    expiration: datetime
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PreDecisionContext:
    """Contesto per pre-decisione"""
    user_patterns: Dict[str, Any]
    recent_decisions: List[str]
    active_projects: List[str]
    current_focus: Optional[str]
    time_of_day: str
    day_of_week: str
    pending_items: List[str]


class PreDecisionAutomation:
    """
    Sistema di automazione pre-decisionale.
    
    Anticipa le decisioni che l'utente dovrà prendere:
    - Analizza pattern comportamentali
    - Prepara opzioni in background
    - Pre-carica dati rilevanti
    - Presenta raccomandazioni proattive
    """
    
    # Tempo massimo in cache
    MAX_CACHE_AGE = timedelta(hours=2)
    
    # Pattern temporali comuni
    TIME_PATTERNS = {
        'morning': ['check_emails', 'review_tasks', 'standup_prep'],
        'midday': ['review_progress', 'prioritize_afternoon'],
        'evening': ['daily_summary', 'plan_tomorrow'],
        'weekly_start': ['weekly_planning', 'resource_allocation'],
        'weekly_end': ['weekly_review', 'report_generation']
    }
    
    def __init__(self):
        self.anticipated_decisions: Dict[str, AnticipatedDecision] = {}
        self.decision_history: List[Dict[str, Any]] = []
        self.user_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.active_preparations: Dict[str, asyncio.Task] = {}
        self.background_data: Dict[str, Any] = {}
        
    async def analyze_context(
        self,
        context: Dict[str, Any]
    ) -> List[AnticipatedDecision]:
        """
        Analizza contesto e anticipa decisioni.
        """
        
        anticipated = []
        
        # 1. Check pattern temporali
        time_based = await self._check_time_patterns(context)
        anticipated.extend(time_based)
        
        # 2. Check pattern comportamentali utente
        behavioral = await self._check_behavioral_patterns(context)
        anticipated.extend(behavioral)
        
        # 3. Check decisioni reattive
        reactive = await self._check_reactive_triggers(context)
        anticipated.extend(reactive)
        
        # 4. Check decisioni programmate
        scheduled = await self._check_scheduled_decisions(context)
        anticipated.extend(scheduled)
        
        # Memorizza
        for decision in anticipated:
            self.anticipated_decisions[decision.decision_id] = decision
            
            # Avvia preparazione in background
            if decision.decision_id not in self.active_preparations:
                task = asyncio.create_task(
                    self._prepare_decision_background(decision)
                )
                self.active_preparations[decision.decision_id] = task
        
        # Cleanup vecchie
        await self._cleanup_expired()
        
        return anticipated
    
    async def _check_time_patterns(
        self,
        context: Dict[str, Any]
    ) -> List[AnticipatedDecision]:
        """Check pattern basati sul tempo"""
        
        anticipated = []
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        
        # Determina periodo del giorno
        if 6 <= hour < 10:
            period = 'morning'
        elif 11 <= hour < 14:
            period = 'midday'
        elif 17 <= hour < 20:
            period = 'evening'
        else:
            period = None
        
        # Determina momento settimana
        if weekday == 0:  # Lunedì
            week_period = 'weekly_start'
        elif weekday == 4:  # Venerdì
            week_period = 'weekly_end'
        else:
            week_period = None
        
        # Genera anticipazioni
        patterns_to_check = []
        if period:
            patterns_to_check.extend(self.TIME_PATTERNS.get(period, []))
        if week_period:
            patterns_to_check.extend(self.TIME_PATTERNS.get(week_period, []))
        
        for pattern in patterns_to_check:
            decision = AnticipatedDecision(
                decision_id=f"time_{pattern}_{now.strftime('%Y%m%d_%H')}",
                category=DecisionCategory.ROUTINE,
                description=f"Attività tipica: {pattern.replace('_', ' ')}",
                trigger_conditions=[f"time={period}", f"weekday={weekday}"],
                predicted_time=now + timedelta(minutes=30),
                confidence=PredictionConfidence.MODERATE,
                prepared_options=[],
                background_analysis={},
                data_gathered=[],
                expiration=now + self.MAX_CACHE_AGE
            )
            anticipated.append(decision)
        
        return anticipated
    
    async def _check_behavioral_patterns(
        self,
        context: Dict[str, Any]
    ) -> List[AnticipatedDecision]:
        """Check pattern comportamentali"""
        
        anticipated = []
        recent_actions = context.get('recent_actions', [])
        
        # Pattern: dopo login, check notifiche
        if 'login' in recent_actions:
            anticipated.append(AnticipatedDecision(
                decision_id=f"behavioral_post_login_{datetime.now().timestamp()}",
                category=DecisionCategory.CONTEXTUAL,
                description="Review notifiche e messaggi in attesa",
                trigger_conditions=['post_login'],
                predicted_time=datetime.now() + timedelta(minutes=5),
                confidence=PredictionConfidence.HIGH,
                prepared_options=[
                    {'action': 'show_notifications', 'label': 'Mostra notifiche'},
                    {'action': 'show_pending', 'label': 'Elementi in attesa'}
                ],
                background_analysis={'notification_count': context.get('notifications', 0)},
                data_gathered=['notifications', 'pending_tasks'],
                expiration=datetime.now() + timedelta(hours=1)
            ))
        
        # Pattern: dopo modifica file, suggerisci commit
        if 'file_edit' in recent_actions:
            anticipated.append(AnticipatedDecision(
                decision_id=f"behavioral_commit_{datetime.now().timestamp()}",
                category=DecisionCategory.CONTEXTUAL,
                description="Salvare modifiche nel repository?",
                trigger_conditions=['files_modified'],
                predicted_time=datetime.now() + timedelta(minutes=15),
                confidence=PredictionConfidence.MODERATE,
                prepared_options=[
                    {'action': 'git_commit', 'label': 'Commit modifiche'},
                    {'action': 'git_diff', 'label': 'Vedi diff'},
                    {'action': 'postpone', 'label': 'Più tardi'}
                ],
                background_analysis={},
                data_gathered=['modified_files'],
                expiration=datetime.now() + timedelta(hours=1)
            ))
        
        return anticipated
    
    async def _check_reactive_triggers(
        self,
        context: Dict[str, Any]
    ) -> List[AnticipatedDecision]:
        """Check trigger reattivi"""
        
        anticipated = []
        
        # Alert attivi
        if context.get('active_alerts'):
            for alert in context['active_alerts'][:3]:
                anticipated.append(AnticipatedDecision(
                    decision_id=f"reactive_alert_{alert.get('id', 'unknown')}",
                    category=DecisionCategory.REACTIVE,
                    description=f"Rispondere ad alert: {alert.get('title', 'Unknown')}",
                    trigger_conditions=[f"alert_active={alert.get('id')}"],
                    predicted_time=datetime.now(),
                    confidence=PredictionConfidence.VERY_HIGH,
                    prepared_options=[
                        {'action': 'acknowledge', 'label': 'Riconosci'},
                        {'action': 'investigate', 'label': 'Investiga'},
                        {'action': 'escalate', 'label': 'Escala'}
                    ],
                    background_analysis={'alert_details': alert},
                    data_gathered=['alert_history', 'related_metrics'],
                    expiration=datetime.now() + timedelta(hours=4)
                ))
        
        return anticipated
    
    async def _check_scheduled_decisions(
        self,
        context: Dict[str, Any]
    ) -> List[AnticipatedDecision]:
        """Check decisioni programmate"""
        
        anticipated = []
        scheduled = context.get('scheduled_items', [])
        
        now = datetime.now()
        
        for item in scheduled:
            scheduled_time = item.get('time')
            if scheduled_time and isinstance(scheduled_time, datetime):
                # Anticipa di 15 minuti
                if now < scheduled_time < now + timedelta(minutes=30):
                    anticipated.append(AnticipatedDecision(
                        decision_id=f"scheduled_{item.get('id', 'unknown')}",
                        category=DecisionCategory.SCHEDULED,
                        description=f"Preparazione per: {item.get('title', 'Evento')}",
                        trigger_conditions=[f"scheduled_time={scheduled_time}"],
                        predicted_time=scheduled_time - timedelta(minutes=15),
                        confidence=PredictionConfidence.VERY_HIGH,
                        prepared_options=[
                            {'action': 'prepare', 'label': 'Prepara materiali'},
                            {'action': 'reschedule', 'label': 'Riprogramma'},
                            {'action': 'delegate', 'label': 'Delega'}
                        ],
                        background_analysis={'scheduled_item': item},
                        data_gathered=['related_docs', 'participants'],
                        expiration=scheduled_time + timedelta(hours=1)
                    ))
        
        return anticipated
    
    async def _prepare_decision_background(
        self,
        decision: AnticipatedDecision
    ):
        """Prepara dati in background per una decisione"""
        
        logger.info(f"⏰ Preparazione background: {decision.decision_id}")
        
        try:
            # Simula raccolta dati
            await asyncio.sleep(0.5)  # In produzione: actual data gathering
            
            # Aggiorna con dati raccolti
            decision.background_analysis['prepared_at'] = datetime.now().isoformat()
            decision.background_analysis['data_freshness'] = 'fresh'
            
            # Pre-genera opzioni se possibile
            if not decision.prepared_options:
                decision.prepared_options = await self._generate_options(decision)
            
            logger.info(f"⏰ Preparazione completata: {decision.decision_id}")
            
        except Exception as e:
            logger.error(f"Errore preparazione {decision.decision_id}: {e}")
    
    async def _generate_options(
        self,
        decision: AnticipatedDecision
    ) -> List[Dict[str, Any]]:
        """Genera opzioni per una decisione"""
        
        # Default options
        return [
            {'action': 'proceed', 'label': 'Procedi', 'confidence': 0.7},
            {'action': 'postpone', 'label': 'Rimanda', 'confidence': 0.2},
            {'action': 'delegate', 'label': 'Delega', 'confidence': 0.1}
        ]
    
    async def _cleanup_expired(self):
        """Rimuove decisioni scadute"""
        
        now = datetime.now()
        expired = [
            k for k, v in self.anticipated_decisions.items()
            if v.expiration < now
        ]
        
        for key in expired:
            del self.anticipated_decisions[key]
            if key in self.active_preparations:
                self.active_preparations[key].cancel()
                del self.active_preparations[key]
    
    def record_user_decision(
        self,
        decision_type: str,
        decision_value: Any,
        context: Optional[Dict[str, Any]] = None
    ):
        """Registra decisione utente per apprendimento pattern"""
        
        self.user_patterns[decision_type].append(datetime.now())
        
        self.decision_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': decision_type,
            'value': decision_value,
            'context': context
        })
        
        # Limita history
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
    
    def get_ready_decisions(self) -> List[AnticipatedDecision]:
        """Ritorna decisioni pronte da presentare"""
        
        now = datetime.now()
        ready = []
        
        for decision in self.anticipated_decisions.values():
            # Decisione è pronta se:
            # 1. Tempo previsto raggiunto o passato
            # 2. Confidenza sufficiente
            # 3. Dati preparati
            
            if decision.predicted_time and decision.predicted_time <= now:
                if decision.confidence.value >= 3:  # Almeno MODERATE
                    if decision.background_analysis.get('prepared_at'):
                        ready.append(decision)
        
        return sorted(ready, key=lambda d: d.confidence.value, reverse=True)
    
    def get_all_anticipated(self) -> List[Dict[str, Any]]:
        """Ritorna tutte le decisioni anticipate"""
        
        return [
            {
                'id': d.decision_id,
                'category': d.category.value,
                'description': d.description,
                'predicted_time': d.predicted_time.isoformat() if d.predicted_time else None,
                'confidence': d.confidence.name,
                'options_count': len(d.prepared_options),
                'expires': d.expiration.isoformat()
            }
            for d in self.anticipated_decisions.values()
        ]
    
    def format_decision(self, decision: AnticipatedDecision) -> str:
        """Formatta decisione per visualizzazione"""
        
        confidence_emoji = {
            PredictionConfidence.VERY_HIGH: '🎯',
            PredictionConfidence.HIGH: '✅',
            PredictionConfidence.MODERATE: '📊',
            PredictionConfidence.LOW: '❓',
            PredictionConfidence.SPECULATIVE: '🔮'
        }
        
        emoji = confidence_emoji.get(decision.confidence, '❓')
        
        return f"""
## {emoji} Decisione Anticipata

**{decision.description}**

- **Categoria**: {decision.category.value}
- **Confidenza**: {decision.confidence.name}
- **Tempo previsto**: {decision.predicted_time.strftime('%H:%M') if decision.predicted_time else 'N/A'}

### Opzioni Preparate
{chr(10).join(f"- {opt.get('label', opt.get('action'))}" for opt in decision.prepared_options) or 'Nessuna opzione preparata'}

*Scade: {decision.expiration.strftime('%H:%M')}*
"""
