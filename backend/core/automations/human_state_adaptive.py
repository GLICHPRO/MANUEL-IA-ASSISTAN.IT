"""
👤 HUMAN-STATE ADAPTIVE SYSTEM
==============================
GIDEON si adatta allo stato dell'utente:
- Rileva stress, fretta, stanchezza
- Adatta verbosità e complessità
- Modula proattività
- Supporto empatico

"Noto che sei sotto pressione - semplifico la risposta"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class UserState(Enum):
    """Stati dell'utente"""
    NORMAL = "normal"  # Stato normale
    FOCUSED = "focused"  # Concentrato, non disturbare
    STRESSED = "stressed"  # Sotto stress
    RUSHED = "rushed"  # Di fretta
    TIRED = "tired"  # Stanco
    FRUSTRATED = "frustrated"  # Frustrato
    LEARNING = "learning"  # Sta imparando
    EXPLORING = "exploring"  # Sta esplorando


class AdaptationDimension(Enum):
    """Dimensioni di adattamento"""
    VERBOSITY = "verbosity"  # Quanto parlare
    COMPLEXITY = "complexity"  # Complessità risposte
    PROACTIVITY = "proactivity"  # Iniziativa GIDEON
    FORMALITY = "formality"  # Formalità
    DETAIL_LEVEL = "detail_level"  # Livello dettaglio
    RESPONSE_SPEED = "response_speed"  # Priorità velocità vs qualità


class IndicatorType(Enum):
    """Tipi di indicatori stato"""
    TYPING_SPEED = "typing_speed"
    MESSAGE_LENGTH = "message_length"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    QUESTION_TYPE = "question_type"
    TIME_OF_DAY = "time_of_day"
    SESSION_DURATION = "session_duration"
    TOPIC_SWITCHING = "topic_switching"


@dataclass
class StateIndicator:
    """Indicatore di stato"""
    indicator_type: IndicatorType
    value: float  # Valore normalizzato 0-1
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UserStateAnalysis:
    """Analisi stato utente"""
    primary_state: UserState
    secondary_states: List[UserState]
    confidence: float
    indicators: List[StateIndicator]
    adaptations_recommended: Dict[AdaptationDimension, float]
    support_message: Optional[str] = None


@dataclass
class AdaptationProfile:
    """Profilo di adattamento per stato"""
    state: UserState
    verbosity: float  # 0-1 (0=minimo, 1=massimo)
    complexity: float  # 0-1
    proactivity: float  # 0-1
    formality: float  # 0-1
    detail_level: float  # 0-1
    response_speed_priority: float  # 0-1 (1=priorità velocità)


@dataclass
class InteractionEvent:
    """Evento di interazione"""
    event_id: str
    message: str
    timestamp: datetime
    typing_duration: Optional[float] = None  # secondi
    contains_errors: bool = False
    question_type: Optional[str] = None


class HumanStateAdaptiveSystem:
    """
    Sistema adattivo allo stato umano.
    
    Rileva e si adatta a:
    - Stress
    - Fretta
    - Stanchezza
    - Frustrazione
    - Modalità di apprendimento
    """
    
    # Profili di adattamento predefiniti
    DEFAULT_PROFILES = {
        UserState.NORMAL: AdaptationProfile(
            state=UserState.NORMAL,
            verbosity=0.6,
            complexity=0.6,
            proactivity=0.5,
            formality=0.5,
            detail_level=0.6,
            response_speed_priority=0.5
        ),
        UserState.STRESSED: AdaptationProfile(
            state=UserState.STRESSED,
            verbosity=0.3,  # Meno verboso
            complexity=0.4,  # Più semplice
            proactivity=0.3,  # Meno proattivo
            formality=0.3,  # Meno formale
            detail_level=0.4,  # Meno dettagli
            response_speed_priority=0.7  # Priorità velocità
        ),
        UserState.RUSHED: AdaptationProfile(
            state=UserState.RUSHED,
            verbosity=0.2,  # Minimo
            complexity=0.3,
            proactivity=0.2,
            formality=0.2,
            detail_level=0.3,
            response_speed_priority=0.9  # Massima velocità
        ),
        UserState.TIRED: AdaptationProfile(
            state=UserState.TIRED,
            verbosity=0.4,
            complexity=0.3,  # Semplice
            proactivity=0.4,
            formality=0.3,
            detail_level=0.4,
            response_speed_priority=0.6
        ),
        UserState.FRUSTRATED: AdaptationProfile(
            state=UserState.FRUSTRATED,
            verbosity=0.3,
            complexity=0.3,
            proactivity=0.2,  # Non aggiungere cose
            formality=0.4,
            detail_level=0.5,
            response_speed_priority=0.7
        ),
        UserState.LEARNING: AdaptationProfile(
            state=UserState.LEARNING,
            verbosity=0.8,  # Più verboso
            complexity=0.5,  # Moderata
            proactivity=0.7,  # Suggerimenti attivi
            formality=0.5,
            detail_level=0.8,  # Molti dettagli
            response_speed_priority=0.3  # Qualità > velocità
        ),
        UserState.EXPLORING: AdaptationProfile(
            state=UserState.EXPLORING,
            verbosity=0.7,
            complexity=0.7,
            proactivity=0.8,  # Molto proattivo
            formality=0.4,
            detail_level=0.7,
            response_speed_priority=0.4
        ),
        UserState.FOCUSED: AdaptationProfile(
            state=UserState.FOCUSED,
            verbosity=0.4,
            complexity=0.7,  # Può gestire complessità
            proactivity=0.2,  # Non disturbare
            formality=0.5,
            detail_level=0.5,
            response_speed_priority=0.5
        )
    }
    
    # Messaggi di supporto per stato
    SUPPORT_MESSAGES = {
        UserState.STRESSED: [
            "Noto che potresti essere sotto pressione. Rispondo in modo conciso.",
            "Semplifico la risposta dato il momento.",
            "Vado dritto al punto."
        ],
        UserState.RUSHED: [
            "Capisco che hai poco tempo. Ecco l'essenziale:",
            "In breve:",
            "Risposta rapida:"
        ],
        UserState.TIRED: [
            "Risposta semplificata:",
            "Ti faccio un riassunto:",
            "Ecco la versione breve:"
        ],
        UserState.FRUSTRATED: [
            "Capisco che potrebbe essere frustrante. Proviamo così:",
            "Approccio alternativo:",
            "Semplifichiamo:"
        ],
        UserState.LEARNING: [
            "Ottima domanda! Ecco una spiegazione dettagliata:",
            "Lasciami spiegare passo per passo:",
            "Ecco come funziona:"
        ]
    }
    
    def __init__(self):
        self.profiles = self.DEFAULT_PROFILES.copy()
        self.interaction_history: List[InteractionEvent] = []
        self.state_history: List[Tuple[datetime, UserState]] = []
        self.current_state = UserState.NORMAL
        
        # Pesi per indicatori
        self.indicator_weights = {
            IndicatorType.TYPING_SPEED: 0.15,
            IndicatorType.MESSAGE_LENGTH: 0.1,
            IndicatorType.ERROR_RATE: 0.2,
            IndicatorType.RESPONSE_TIME: 0.15,
            IndicatorType.QUESTION_TYPE: 0.15,
            IndicatorType.TIME_OF_DAY: 0.1,
            IndicatorType.SESSION_DURATION: 0.1,
            IndicatorType.TOPIC_SWITCHING: 0.05
        }
    
    def record_interaction(
        self,
        message: str,
        typing_duration: Optional[float] = None,
        contains_errors: bool = False,
        question_type: Optional[str] = None
    ) -> str:
        """Registra interazione utente"""
        
        event = InteractionEvent(
            event_id=f"int_{datetime.now().timestamp()}",
            message=message,
            timestamp=datetime.now(),
            typing_duration=typing_duration,
            contains_errors=contains_errors,
            question_type=question_type
        )
        
        self.interaction_history.append(event)
        
        # Limita storia
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
        
        return event.event_id
    
    def analyze_state(self) -> UserStateAnalysis:
        """Analizza stato corrente dell'utente"""
        
        indicators = []
        
        # 1. Analizza velocità typing
        typing_indicator = self._analyze_typing_speed()
        if typing_indicator:
            indicators.append(typing_indicator)
        
        # 2. Analizza lunghezza messaggi
        length_indicator = self._analyze_message_length()
        if length_indicator:
            indicators.append(length_indicator)
        
        # 3. Analizza tasso errori
        error_indicator = self._analyze_error_rate()
        if error_indicator:
            indicators.append(error_indicator)
        
        # 4. Analizza tempo di risposta
        response_indicator = self._analyze_response_time()
        if response_indicator:
            indicators.append(response_indicator)
        
        # 5. Analizza ora del giorno
        time_indicator = self._analyze_time_of_day()
        indicators.append(time_indicator)
        
        # 6. Analizza durata sessione
        session_indicator = self._analyze_session_duration()
        if session_indicator:
            indicators.append(session_indicator)
        
        # Inferisci stato
        state_scores = self._calculate_state_scores(indicators)
        
        # Ordina per score
        sorted_states = sorted(
            state_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        primary_state = sorted_states[0][0] if sorted_states else UserState.NORMAL
        primary_score = sorted_states[0][1] if sorted_states else 0.5
        
        # Secondary states (score significativo)
        secondary = [
            state for state, score in sorted_states[1:4]
            if score > 0.3
        ]
        
        # Aggiorna stato corrente
        self.current_state = primary_state
        self.state_history.append((datetime.now(), primary_state))
        
        # Calcola adattamenti
        adaptations = self._calculate_adaptations(primary_state, secondary)
        
        # Messaggio di supporto
        support = self._get_support_message(primary_state) if primary_state != UserState.NORMAL else None
        
        return UserStateAnalysis(
            primary_state=primary_state,
            secondary_states=secondary,
            confidence=primary_score,
            indicators=indicators,
            adaptations_recommended=adaptations,
            support_message=support
        )
    
    def _analyze_typing_speed(self) -> Optional[StateIndicator]:
        """Analizza velocità di battitura"""
        
        recent = [
            e for e in self.interaction_history[-10:]
            if e.typing_duration is not None
        ]
        
        if not recent:
            return None
        
        # Caratteri al secondo
        speeds = []
        for e in recent:
            if e.typing_duration > 0:
                cps = len(e.message) / e.typing_duration
                speeds.append(cps)
        
        if not speeds:
            return None
        
        avg_speed = sum(speeds) / len(speeds)
        
        # Normalizza (assume 3 cps = normale, 6+ = veloce/fretta)
        normalized = min(avg_speed / 6.0, 1.0)
        
        return StateIndicator(
            indicator_type=IndicatorType.TYPING_SPEED,
            value=normalized,
            confidence=0.7 if len(speeds) >= 5 else 0.4
        )
    
    def _analyze_message_length(self) -> Optional[StateIndicator]:
        """Analizza lunghezza messaggi"""
        
        recent = self.interaction_history[-10:]
        
        if not recent:
            return None
        
        avg_length = sum(len(e.message) for e in recent) / len(recent)
        
        # Normalizza (assume 50 char = normale, <20 = corto/fretta)
        normalized = min(avg_length / 100.0, 1.0)
        
        return StateIndicator(
            indicator_type=IndicatorType.MESSAGE_LENGTH,
            value=normalized,
            confidence=0.6
        )
    
    def _analyze_error_rate(self) -> Optional[StateIndicator]:
        """Analizza tasso di errori/typo"""
        
        recent = self.interaction_history[-10:]
        
        if not recent:
            return None
        
        error_count = sum(1 for e in recent if e.contains_errors)
        error_rate = error_count / len(recent)
        
        return StateIndicator(
            indicator_type=IndicatorType.ERROR_RATE,
            value=error_rate,
            confidence=0.7
        )
    
    def _analyze_response_time(self) -> Optional[StateIndicator]:
        """Analizza tempo tra messaggi"""
        
        if len(self.interaction_history) < 2:
            return None
        
        recent = self.interaction_history[-10:]
        intervals = []
        
        for i in range(1, len(recent)):
            delta = (recent[i].timestamp - recent[i-1].timestamp).total_seconds()
            if delta < 300:  # Max 5 minuti
                intervals.append(delta)
        
        if not intervals:
            return None
        
        avg_interval = sum(intervals) / len(intervals)
        
        # Normalizza (assume 30s = normale, <5s = fretta)
        normalized = min(avg_interval / 60.0, 1.0)
        
        return StateIndicator(
            indicator_type=IndicatorType.RESPONSE_TIME,
            value=1 - normalized,  # Inverti: basso intervallo = alto valore
            confidence=0.6
        )
    
    def _analyze_time_of_day(self) -> StateIndicator:
        """Analizza ora del giorno"""
        
        hour = datetime.now().hour
        
        # Stanchezza probabile in tarda notte/prima mattina
        if hour >= 23 or hour < 6:
            value = 0.8  # Alta probabilità stanchezza
        elif hour >= 21 or hour < 8:
            value = 0.5
        else:
            value = 0.2
        
        return StateIndicator(
            indicator_type=IndicatorType.TIME_OF_DAY,
            value=value,
            confidence=0.5
        )
    
    def _analyze_session_duration(self) -> Optional[StateIndicator]:
        """Analizza durata sessione"""
        
        if not self.interaction_history:
            return None
        
        first = self.interaction_history[0].timestamp
        duration = (datetime.now() - first).total_seconds() / 3600  # ore
        
        # Stanchezza aumenta con durata (>3 ore = alto)
        normalized = min(duration / 4.0, 1.0)
        
        return StateIndicator(
            indicator_type=IndicatorType.SESSION_DURATION,
            value=normalized,
            confidence=0.6
        )
    
    def _calculate_state_scores(
        self,
        indicators: List[StateIndicator]
    ) -> Dict[UserState, float]:
        """Calcola scores per ogni stato"""
        
        scores = {state: 0.0 for state in UserState}
        
        # Mappatura indicatori -> stati
        state_mappings = {
            UserState.STRESSED: {
                IndicatorType.TYPING_SPEED: 0.7,  # Veloce = stress
                IndicatorType.ERROR_RATE: 0.8,  # Errori = stress
                IndicatorType.RESPONSE_TIME: 0.5,
            },
            UserState.RUSHED: {
                IndicatorType.TYPING_SPEED: 0.8,
                IndicatorType.MESSAGE_LENGTH: -0.6,  # Corto = fretta (inverso)
                IndicatorType.RESPONSE_TIME: 0.7,
            },
            UserState.TIRED: {
                IndicatorType.TIME_OF_DAY: 0.8,
                IndicatorType.SESSION_DURATION: 0.7,
                IndicatorType.ERROR_RATE: 0.5,
            },
            UserState.FRUSTRATED: {
                IndicatorType.ERROR_RATE: 0.6,
                IndicatorType.TOPIC_SWITCHING: 0.7,
            },
            UserState.LEARNING: {
                IndicatorType.MESSAGE_LENGTH: 0.5,  # Domande lunghe
            },
            UserState.FOCUSED: {
                IndicatorType.RESPONSE_TIME: -0.5,  # Intervalli regolari
            }
        }
        
        # Calcola score per stato
        for state, mappings in state_mappings.items():
            state_score = 0.0
            weight_sum = 0.0
            
            for indicator in indicators:
                if indicator.indicator_type in mappings:
                    mapping_weight = mappings[indicator.indicator_type]
                    indicator_weight = self.indicator_weights.get(indicator.indicator_type, 0.1)
                    
                    if mapping_weight < 0:
                        # Inverso
                        contribution = (1 - indicator.value) * abs(mapping_weight)
                    else:
                        contribution = indicator.value * mapping_weight
                    
                    state_score += contribution * indicator_weight * indicator.confidence
                    weight_sum += indicator_weight
            
            if weight_sum > 0:
                scores[state] = state_score / weight_sum
        
        # NORMAL è default se altri score bassi
        max_other = max(v for k, v in scores.items() if k != UserState.NORMAL)
        scores[UserState.NORMAL] = max(0.5 - max_other, 0.2)
        
        return scores
    
    def _calculate_adaptations(
        self,
        primary: UserState,
        secondary: List[UserState]
    ) -> Dict[AdaptationDimension, float]:
        """Calcola adattamenti raccomandati"""
        
        # Parti dal profilo primario
        profile = self.profiles.get(primary, self.profiles[UserState.NORMAL])
        
        adaptations = {
            AdaptationDimension.VERBOSITY: profile.verbosity,
            AdaptationDimension.COMPLEXITY: profile.complexity,
            AdaptationDimension.PROACTIVITY: profile.proactivity,
            AdaptationDimension.FORMALITY: profile.formality,
            AdaptationDimension.DETAIL_LEVEL: profile.detail_level,
            AdaptationDimension.RESPONSE_SPEED: profile.response_speed_priority,
        }
        
        # Modera con stati secondari
        for state in secondary[:2]:
            sec_profile = self.profiles.get(state)
            if sec_profile:
                for dim in AdaptationDimension:
                    primary_val = adaptations[dim]
                    sec_val = getattr(sec_profile, dim.value, primary_val)
                    # Media pesata (primario 0.7, secondario 0.3)
                    adaptations[dim] = primary_val * 0.7 + sec_val * 0.3
        
        return adaptations
    
    def _get_support_message(self, state: UserState) -> Optional[str]:
        """Ottiene messaggio di supporto"""
        
        messages = self.SUPPORT_MESSAGES.get(state, [])
        if not messages:
            return None
        
        # Ruota messaggi
        import random
        return random.choice(messages)
    
    def get_current_adaptations(self) -> Dict[AdaptationDimension, float]:
        """Ottiene adattamenti correnti"""
        
        profile = self.profiles.get(self.current_state, self.profiles[UserState.NORMAL])
        
        return {
            AdaptationDimension.VERBOSITY: profile.verbosity,
            AdaptationDimension.COMPLEXITY: profile.complexity,
            AdaptationDimension.PROACTIVITY: profile.proactivity,
            AdaptationDimension.FORMALITY: profile.formality,
            AdaptationDimension.DETAIL_LEVEL: profile.detail_level,
            AdaptationDimension.RESPONSE_SPEED: profile.response_speed_priority,
        }
    
    def should_simplify_response(self) -> bool:
        """Indica se semplificare risposta"""
        
        adaptations = self.get_current_adaptations()
        return adaptations[AdaptationDimension.COMPLEXITY] < 0.4
    
    def should_be_proactive(self) -> bool:
        """Indica se essere proattivo"""
        
        adaptations = self.get_current_adaptations()
        return adaptations[AdaptationDimension.PROACTIVITY] > 0.5
    
    def get_verbosity_level(self) -> str:
        """Ottiene livello verbosità"""
        
        adaptations = self.get_current_adaptations()
        verbosity = adaptations[AdaptationDimension.VERBOSITY]
        
        if verbosity < 0.3:
            return "minimal"
        elif verbosity < 0.5:
            return "concise"
        elif verbosity < 0.7:
            return "normal"
        else:
            return "detailed"
    
    def format_state_analysis(self, analysis: UserStateAnalysis) -> str:
        """Formatta analisi per visualizzazione"""
        
        state_emoji = {
            UserState.NORMAL: "😊",
            UserState.STRESSED: "😰",
            UserState.RUSHED: "⏰",
            UserState.TIRED: "😴",
            UserState.FRUSTRATED: "😤",
            UserState.LEARNING: "📚",
            UserState.EXPLORING: "🔍",
            UserState.FOCUSED: "🎯"
        }
        
        indicators_str = "\n".join(
            f"- {i.indicator_type.value}: {i.value:.2f} (conf: {i.confidence:.1%})"
            for i in analysis.indicators[:5]
        )
        
        adaptations_str = "\n".join(
            f"- {d.value}: {v:.2f}"
            for d, v in analysis.adaptations_recommended.items()
        )
        
        return f"""
## 👤 Analisi Stato Utente

### Stato Rilevato
{state_emoji.get(analysis.primary_state, '❓')} **{analysis.primary_state.value.upper()}** (conf: {analysis.confidence:.1%})

{f'Stati secondari: {", ".join(s.value for s in analysis.secondary_states)}' if analysis.secondary_states else ''}

### Indicatori
{indicators_str}

### Adattamenti Raccomandati
{adaptations_str}

{f'### Supporto' + chr(10) + f'> {analysis.support_message}' if analysis.support_message else ''}
"""
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        state_emoji = {
            UserState.NORMAL: "😊",
            UserState.STRESSED: "😰",
            UserState.RUSHED: "⏰",
            UserState.TIRED: "😴",
            UserState.FRUSTRATED: "😤",
            UserState.LEARNING: "📚",
            UserState.EXPLORING: "🔍",
            UserState.FOCUSED: "🎯"
        }
        
        adaptations = self.get_current_adaptations()
        
        return f"""
# 👤 Human-State Adaptive System

## Stato Corrente
{state_emoji.get(self.current_state, '❓')} **{self.current_state.value.upper()}**

## Adattamenti Attivi
| Dimensione | Valore |
|------------|--------|
| Verbosità | {adaptations[AdaptationDimension.VERBOSITY]:.2f} |
| Complessità | {adaptations[AdaptationDimension.COMPLEXITY]:.2f} |
| Proattività | {adaptations[AdaptationDimension.PROACTIVITY]:.2f} |
| Formalità | {adaptations[AdaptationDimension.FORMALITY]:.2f} |
| Dettagli | {adaptations[AdaptationDimension.DETAIL_LEVEL]:.2f} |
| Priorità velocità | {adaptations[AdaptationDimension.RESPONSE_SPEED]:.2f} |

## Statistiche Sessione
| Metrica | Valore |
|---------|--------|
| Interazioni | {len(self.interaction_history)} |
| Cambi stato | {len(self.state_history)} |
| Verbosità consigliata | {self.get_verbosity_level()} |
| Semplificare? | {'✅ Sì' if self.should_simplify_response() else '❌ No'} |
| Essere proattivo? | {'✅ Sì' if self.should_be_proactive() else '❌ No'} |
"""


# Singleton
_human_state_system: Optional[HumanStateAdaptiveSystem] = None


def get_human_state_system() -> HumanStateAdaptiveSystem:
    """Ottiene istanza singleton"""
    global _human_state_system
    if _human_state_system is None:
        _human_state_system = HumanStateAdaptiveSystem()
    return _human_state_system
