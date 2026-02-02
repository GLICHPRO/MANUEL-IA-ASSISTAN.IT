"""
👤 CAM HUMAN INTERFACE LAYER
=============================
Interfaccia adattiva per crisi.

Componenti:
- CrisisUIMode: UI che si adatta al livello crisi
- RoleAwareViews: Vista diversa per ruolo (Operatore/Decisore/Supervisore)
- CognitiveLoadMonitor: Monitora e adatta carico cognitivo

"Human-in-the-Loop sempre"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """Ruoli utente"""
    OPERATOR = "operator"  # Chi esegue azioni
    DECISION_MAKER = "decision_maker"  # Chi decide strategie
    SUPERVISOR = "supervisor"  # Chi supervisiona tutto
    OBSERVER = "observer"  # Solo lettura


class UIMode(Enum):
    """Modalità UI"""
    NORMAL = "normal"  # Interfaccia standard
    FOCUSED = "focused"  # Ridotta, elementi chiave
    CRISIS = "crisis"  # Massima chiarezza
    MINIMAL = "minimal"  # Solo essenziale


class CognitiveState(Enum):
    """Stato cognitivo stimato"""
    FRESH = "fresh"  # Lucido
    NORMAL = "normal"  # Normale
    FATIGUED = "fatigued"  # Affaticato
    STRESSED = "stressed"  # Sotto stress
    OVERLOADED = "overloaded"  # Sovraccarico


class InformationPriority(Enum):
    """Priorità informazione"""
    CRITICAL = "critical"  # Sempre visibile
    HIGH = "high"  # Visibile in crisi
    MEDIUM = "medium"  # Visibile se spazio
    LOW = "low"  # Nascosto in crisi


@dataclass
class UIElement:
    """Elemento UI"""
    id: str
    name: str
    priority: InformationPriority
    content: Any
    visible_in_modes: List[UIMode]
    roles: List[UserRole] = field(default_factory=lambda: list(UserRole))


@dataclass
class ViewConfig:
    """Configurazione vista per ruolo"""
    role: UserRole
    default_mode: UIMode
    visible_sections: List[str]
    action_permissions: List[str]
    max_info_density: float  # 0-1


@dataclass
class CognitiveMetrics:
    """Metriche cognitive"""
    session_duration: timedelta
    actions_count: int
    decisions_count: int
    error_rate: float
    response_time_avg: float  # secondi
    stress_indicators: int


# ============================================================
# CRISIS UI MODE
# ============================================================

class CrisisUIMode:
    """
    UI che si adatta automaticamente al livello di crisi.
    
    Principi:
    - Meno è meglio in crisi
    - Focus su azioni possibili
    - Chiarezza massima
    """
    
    # Mapping crisi -> UI mode
    CRISIS_MODE_MAP = {
        "NONE": UIMode.NORMAL,
        "WATCH": UIMode.NORMAL,
        "SOFT": UIMode.FOCUSED,
        "ACTIVE": UIMode.CRISIS,
        "CRITICAL": UIMode.MINIMAL
    }
    
    def __init__(self):
        self.current_mode = UIMode.NORMAL
        self.elements: Dict[str, UIElement] = {}
        self.mode_history: List[Dict[str, Any]] = []
    
    def register_element(self, element: UIElement):
        """Registra elemento UI"""
        self.elements[element.id] = element
    
    def set_crisis_level(self, crisis_level: str) -> UIMode:
        """Imposta modo UI basato su livello crisi"""
        
        new_mode = self.CRISIS_MODE_MAP.get(crisis_level, UIMode.NORMAL)
        
        if new_mode != self.current_mode:
            self.mode_history.append({
                'from': self.current_mode,
                'to': new_mode,
                'crisis_level': crisis_level,
                'timestamp': datetime.now()
            })
            self.current_mode = new_mode
            logger.info(f"🎨 UI Mode: {self.current_mode.value}")
        
        return self.current_mode
    
    def get_visible_elements(
        self,
        role: Optional[UserRole] = None
    ) -> List[UIElement]:
        """Ottiene elementi visibili per modo e ruolo corrente"""
        
        visible = []
        
        for element in self.elements.values():
            # Check visibilità in modo corrente
            if self.current_mode not in element.visible_in_modes:
                continue
            
            # Check ruolo
            if role and role not in element.roles:
                continue
            
            visible.append(element)
        
        # Ordina per priorità
        priority_order = {
            InformationPriority.CRITICAL: 0,
            InformationPriority.HIGH: 1,
            InformationPriority.MEDIUM: 2,
            InformationPriority.LOW: 3
        }
        
        return sorted(visible, key=lambda e: priority_order[e.priority])
    
    def get_mode_config(self) -> Dict[str, Any]:
        """Ottiene configurazione modo corrente"""
        
        configs = {
            UIMode.NORMAL: {
                'layout': 'full',
                'animations': True,
                'detail_level': 'high',
                'colors': 'standard',
                'font_size': 'normal',
                'max_elements': 50
            },
            UIMode.FOCUSED: {
                'layout': 'focused',
                'animations': False,
                'detail_level': 'medium',
                'colors': 'high_contrast',
                'font_size': 'larger',
                'max_elements': 30
            },
            UIMode.CRISIS: {
                'layout': 'crisis',
                'animations': False,
                'detail_level': 'low',
                'colors': 'crisis_palette',
                'font_size': 'large',
                'max_elements': 15
            },
            UIMode.MINIMAL: {
                'layout': 'minimal',
                'animations': False,
                'detail_level': 'minimal',
                'colors': 'high_contrast_critical',
                'font_size': 'extra_large',
                'max_elements': 5
            }
        }
        
        return configs.get(self.current_mode, configs[UIMode.NORMAL])
    
    def generate_crisis_ui(self) -> Dict[str, Any]:
        """Genera struttura UI per crisi"""
        
        config = self.get_mode_config()
        elements = self.get_visible_elements()
        
        return {
            'mode': self.current_mode.value,
            'config': config,
            'elements': [
                {
                    'id': e.id,
                    'name': e.name,
                    'priority': e.priority.value,
                    'content': e.content
                }
                for e in elements[:config['max_elements']]
            ],
            'actions': self._get_available_actions(),
            'status_bar': self._get_status_bar()
        }
    
    def _get_available_actions(self) -> List[Dict[str, Any]]:
        """Ottiene azioni disponibili per modo"""
        
        actions_by_mode = {
            UIMode.NORMAL: ['view', 'edit', 'create', 'delete', 'analyze', 'export'],
            UIMode.FOCUSED: ['view', 'edit', 'analyze', 'alert'],
            UIMode.CRISIS: ['view', 'stop', 'alert', 'escalate'],
            UIMode.MINIMAL: ['stop', 'escalate', 'help']
        }
        
        action_list = actions_by_mode.get(self.current_mode, [])
        
        return [
            {'id': a, 'label': a.title(), 'enabled': True}
            for a in action_list
        ]
    
    def _get_status_bar(self) -> Dict[str, Any]:
        """Genera status bar"""
        
        return {
            'mode': self.current_mode.value,
            'timestamp': datetime.now().isoformat(),
            'emergency_stop_visible': self.current_mode in [UIMode.CRISIS, UIMode.MINIMAL],
            'help_visible': True
        }


# ============================================================
# ROLE AWARE VIEWS
# ============================================================

class RoleAwareViews:
    """
    Viste diverse per ruolo:
    - Operatore: Focus su azioni immediate
    - Decisore: Focus su opzioni e trade-off
    - Supervisore: Overview e audit
    """
    
    # Configurazioni default per ruolo
    DEFAULT_CONFIGS = {
        UserRole.OPERATOR: ViewConfig(
            role=UserRole.OPERATOR,
            default_mode=UIMode.FOCUSED,
            visible_sections=['actions', 'status', 'alerts', 'quick_info'],
            action_permissions=['execute', 'report', 'escalate'],
            max_info_density=0.6
        ),
        UserRole.DECISION_MAKER: ViewConfig(
            role=UserRole.DECISION_MAKER,
            default_mode=UIMode.FOCUSED,
            visible_sections=['options', 'analysis', 'risks', 'timeline', 'recommendations'],
            action_permissions=['decide', 'approve', 'delegate', 'escalate'],
            max_info_density=0.8
        ),
        UserRole.SUPERVISOR: ViewConfig(
            role=UserRole.SUPERVISOR,
            default_mode=UIMode.NORMAL,
            visible_sections=['overview', 'audit', 'metrics', 'team', 'history'],
            action_permissions=['view_all', 'override', 'audit', 'configure'],
            max_info_density=1.0
        ),
        UserRole.OBSERVER: ViewConfig(
            role=UserRole.OBSERVER,
            default_mode=UIMode.NORMAL,
            visible_sections=['overview', 'status'],
            action_permissions=['view'],
            max_info_density=0.4
        )
    }
    
    def __init__(self):
        self.configs = self.DEFAULT_CONFIGS.copy()
        self.active_users: Dict[str, UserRole] = {}
    
    def register_user(self, user_id: str, role: UserRole):
        """Registra utente con ruolo"""
        self.active_users[user_id] = role
        logger.info(f"👤 User {user_id} registrato come {role.value}")
    
    def get_view(
        self,
        user_id: str,
        crisis_level: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ottiene vista personalizzata per utente"""
        
        role = self.active_users.get(user_id, UserRole.OBSERVER)
        config = self.configs[role]
        
        # Filtra dati per sezioni visibili
        filtered_data = {
            section: data.get(section, {})
            for section in config.visible_sections
            if section in data
        }
        
        # Adatta densità
        if config.max_info_density < 1.0:
            filtered_data = self._reduce_density(
                filtered_data,
                config.max_info_density
            )
        
        return {
            'user_id': user_id,
            'role': role.value,
            'mode': config.default_mode.value,
            'sections': filtered_data,
            'actions': config.action_permissions,
            'crisis_level': crisis_level
        }
    
    def _reduce_density(
        self,
        data: Dict[str, Any],
        max_density: float
    ) -> Dict[str, Any]:
        """Riduce densità informazioni"""
        
        if max_density >= 1.0:
            return data
        
        reduced = {}
        
        for key, value in data.items():
            if isinstance(value, list):
                # Taglia liste
                keep = max(1, int(len(value) * max_density))
                reduced[key] = value[:keep]
            elif isinstance(value, dict):
                # Ricorsivo
                reduced[key] = self._reduce_density(value, max_density)
            else:
                reduced[key] = value
        
        return reduced
    
    def get_role_specific_message(
        self,
        message: str,
        role: UserRole
    ) -> str:
        """Adatta messaggio per ruolo"""
        
        prefixes = {
            UserRole.OPERATOR: "📋 AZIONE RICHIESTA",
            UserRole.DECISION_MAKER: "📊 DECISIONE NECESSARIA",
            UserRole.SUPERVISOR: "📈 REPORT",
            UserRole.OBSERVER: "ℹ️ INFO"
        }
        
        return f"{prefixes[role]}: {message}"
    
    def can_perform_action(
        self,
        user_id: str,
        action: str
    ) -> bool:
        """Verifica se utente può eseguire azione"""
        
        role = self.active_users.get(user_id, UserRole.OBSERVER)
        config = self.configs[role]
        
        return action in config.action_permissions


# ============================================================
# COGNITIVE LOAD MONITOR
# ============================================================

class CognitiveLoadMonitor:
    """
    Monitora carico cognitivo e adatta interfaccia.
    
    Indicatori:
    - Durata sessione
    - Numero azioni
    - Tasso errori
    - Tempo risposta
    """
    
    # Soglie per stati cognitivi
    THRESHOLDS = {
        'session_hours_fatigued': 4,
        'session_hours_stressed': 6,
        'error_rate_stressed': 0.15,
        'response_time_stressed': 10,  # secondi
        'actions_per_hour_overload': 100
    }
    
    def __init__(self):
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_states: Dict[str, CognitiveState] = {}
    
    def start_session(self, user_id: str):
        """Avvia tracciamento sessione"""
        
        self.user_sessions[user_id] = {
            'start': datetime.now(),
            'actions': [],
            'decisions': [],
            'errors': [],
            'response_times': []
        }
        self.user_states[user_id] = CognitiveState.FRESH
    
    def record_action(
        self,
        user_id: str,
        action_type: str,
        success: bool,
        response_time: float
    ):
        """Registra azione utente"""
        
        session = self.user_sessions.get(user_id)
        if not session:
            self.start_session(user_id)
            session = self.user_sessions[user_id]
        
        session['actions'].append({
            'type': action_type,
            'timestamp': datetime.now(),
            'success': success
        })
        
        session['response_times'].append(response_time)
        
        if not success:
            session['errors'].append({
                'type': action_type,
                'timestamp': datetime.now()
            })
        
        # Aggiorna stato
        self._update_state(user_id)
    
    def record_decision(self, user_id: str, decision_complexity: float):
        """Registra decisione"""
        
        session = self.user_sessions.get(user_id)
        if session:
            session['decisions'].append({
                'complexity': decision_complexity,
                'timestamp': datetime.now()
            })
    
    def _update_state(self, user_id: str):
        """Aggiorna stato cognitivo"""
        
        metrics = self.get_metrics(user_id)
        if not metrics:
            return
        
        # Determina stato
        state = CognitiveState.FRESH
        
        # Check durata
        hours = metrics.session_duration.total_seconds() / 3600
        if hours > self.THRESHOLDS['session_hours_stressed']:
            state = CognitiveState.STRESSED
        elif hours > self.THRESHOLDS['session_hours_fatigued']:
            state = CognitiveState.FATIGUED
        
        # Check error rate
        if metrics.error_rate > self.THRESHOLDS['error_rate_stressed']:
            state = CognitiveState.STRESSED
        
        # Check response time
        if metrics.response_time_avg > self.THRESHOLDS['response_time_stressed']:
            if state == CognitiveState.STRESSED:
                state = CognitiveState.OVERLOADED
            else:
                state = CognitiveState.STRESSED
        
        # Check actions/hour
        if hours > 0:
            actions_per_hour = metrics.actions_count / hours
            if actions_per_hour > self.THRESHOLDS['actions_per_hour_overload']:
                state = CognitiveState.OVERLOADED
        
        old_state = self.user_states.get(user_id)
        self.user_states[user_id] = state
        
        if old_state != state:
            logger.info(f"🧠 User {user_id}: {old_state} → {state}")
    
    def get_metrics(self, user_id: str) -> Optional[CognitiveMetrics]:
        """Ottiene metriche per utente"""
        
        session = self.user_sessions.get(user_id)
        if not session:
            return None
        
        duration = datetime.now() - session['start']
        actions = session['actions']
        errors = session['errors']
        response_times = session['response_times']
        
        error_rate = len(errors) / len(actions) if actions else 0
        avg_response = sum(response_times) / len(response_times) if response_times else 0
        
        return CognitiveMetrics(
            session_duration=duration,
            actions_count=len(actions),
            decisions_count=len(session['decisions']),
            error_rate=error_rate,
            response_time_avg=avg_response,
            stress_indicators=self._count_stress_indicators(session)
        )
    
    def _count_stress_indicators(self, session: Dict) -> int:
        """Conta indicatori di stress"""
        
        indicators = 0
        
        # Errori recenti
        recent_errors = [
            e for e in session['errors']
            if (datetime.now() - e['timestamp']).total_seconds() < 300
        ]
        if len(recent_errors) > 2:
            indicators += 1
        
        # Response time in aumento
        times = session['response_times'][-10:]
        if len(times) >= 5:
            first_half = sum(times[:len(times)//2]) / (len(times)//2)
            second_half = sum(times[len(times)//2:]) / (len(times) - len(times)//2)
            if second_half > first_half * 1.5:
                indicators += 1
        
        return indicators
    
    def get_state(self, user_id: str) -> CognitiveState:
        """Ottiene stato cognitivo"""
        return self.user_states.get(user_id, CognitiveState.FRESH)
    
    def get_recommendations(self, user_id: str) -> List[str]:
        """Ottiene raccomandazioni basate su stato"""
        
        state = self.get_state(user_id)
        
        recommendations = {
            CognitiveState.FRESH: [],
            CognitiveState.NORMAL: [],
            CognitiveState.FATIGUED: [
                "💡 Considera una pausa breve",
                "☕ Potresti aver bisogno di un caffè",
                "⚡ La complessità delle decisioni potrebbe essere ridotta"
            ],
            CognitiveState.STRESSED: [
                "⚠️ Livello stress elevato rilevato",
                "🧘 Raccomandato: pausa di 10-15 minuti",
                "👥 Considera delegare decisioni non urgenti",
                "📉 Riduciamo la complessità dell'interfaccia"
            ],
            CognitiveState.OVERLOADED: [
                "🔴 SOVRACCARICO COGNITIVO RILEVATO",
                "🛑 Fortemente raccomandato: stop attività complesse",
                "👥 Passaggio consegne a collega raccomandato",
                "🚨 Le decisioni critiche dovrebbero essere rimandate"
            ]
        }
        
        return recommendations.get(state, [])
    
    def get_ui_adjustments(self, user_id: str) -> Dict[str, Any]:
        """Ottiene regolazioni UI basate su stato"""
        
        state = self.get_state(user_id)
        
        adjustments = {
            CognitiveState.FRESH: {
                'reduce_info': False,
                'increase_contrast': False,
                'slow_animations': False,
                'suggest_breaks': False
            },
            CognitiveState.NORMAL: {
                'reduce_info': False,
                'increase_contrast': False,
                'slow_animations': False,
                'suggest_breaks': False
            },
            CognitiveState.FATIGUED: {
                'reduce_info': True,
                'increase_contrast': True,
                'slow_animations': True,
                'suggest_breaks': True,
                'break_interval_minutes': 60
            },
            CognitiveState.STRESSED: {
                'reduce_info': True,
                'increase_contrast': True,
                'slow_animations': True,
                'suggest_breaks': True,
                'break_interval_minutes': 30,
                'simplify_decisions': True
            },
            CognitiveState.OVERLOADED: {
                'reduce_info': True,
                'increase_contrast': True,
                'slow_animations': True,
                'suggest_breaks': True,
                'break_interval_minutes': 15,
                'simplify_decisions': True,
                'block_complex_actions': True,
                'recommend_handoff': True
            }
        }
        
        return adjustments.get(state, adjustments[CognitiveState.NORMAL])


# ============================================================
# UNIFIED HUMAN INTERFACE LAYER
# ============================================================

class HumanInterfaceLayer:
    """
    Human Interface Layer unificato per CAM.
    
    "Human-in-the-Loop sempre"
    """
    
    def __init__(self):
        self.ui_mode = CrisisUIMode()
        self.role_views = RoleAwareViews()
        self.cognitive_monitor = CognitiveLoadMonitor()
    
    def update_for_crisis(
        self,
        crisis_level: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Aggiorna interfaccia per livello crisi"""
        
        # Aggiorna UI mode
        mode = self.ui_mode.set_crisis_level(crisis_level)
        
        # Ottieni stato cognitivo
        cognitive_state = self.cognitive_monitor.get_state(user_id)
        
        # Ottieni aggiustamenti
        adjustments = self.cognitive_monitor.get_ui_adjustments(user_id)
        
        # Ottieni raccomandazioni
        recommendations = self.cognitive_monitor.get_recommendations(user_id)
        
        return {
            'ui_mode': mode.value,
            'crisis_level': crisis_level,
            'cognitive_state': cognitive_state.value,
            'adjustments': adjustments,
            'recommendations': recommendations,
            'ui_config': self.ui_mode.get_mode_config()
        }
    
    def get_personalized_view(
        self,
        user_id: str,
        crisis_level: str,
        full_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ottiene vista personalizzata completa"""
        
        # Vista base per ruolo
        view = self.role_views.get_view(user_id, crisis_level, full_data)
        
        # Arricchisci con stato cognitivo
        cognitive_state = self.cognitive_monitor.get_state(user_id)
        adjustments = self.cognitive_monitor.get_ui_adjustments(user_id)
        
        # Applica riduzioni se necessario
        if adjustments.get('reduce_info'):
            view['sections'] = self.role_views._reduce_density(
                view['sections'],
                0.5  # Riduci ulteriormente
            )
        
        view['cognitive'] = {
            'state': cognitive_state.value,
            'adjustments': adjustments,
            'recommendations': self.cognitive_monitor.get_recommendations(user_id)
        }
        
        return view


# Singleton
_human_interface: Optional[HumanInterfaceLayer] = None


def get_human_interface_layer() -> HumanInterfaceLayer:
    """Ottiene istanza singleton"""
    global _human_interface
    if _human_interface is None:
        _human_interface = HumanInterfaceLayer()
    return _human_interface
