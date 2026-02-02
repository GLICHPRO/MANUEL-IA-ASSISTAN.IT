"""
🧠 COGNITIVE LOAD MONITOR
==========================
Monitora il carico cognitivo sull'utente:
- Troppe info = riduci
- Output troppo tecnici = semplifica
- Decisioni consecutive = proponi pausa

Ispirato a UX cognitiva e human factors
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
from collections import deque
import re

logger = logging.getLogger(__name__)


class LoadLevel(Enum):
    """Livelli di carico cognitivo"""
    MINIMAL = 1    # Carico minimo, utente rilassato
    LOW = 2        # Carico basso, gestibile facilmente
    MODERATE = 3   # Carico moderato, attenzione richiesta
    HIGH = 4       # Carico alto, rischio sovraccarico
    CRITICAL = 5   # Sovraccarico, intervento necessario


class OutputComplexity(Enum):
    """Complessità dell'output"""
    TRIVIAL = 1     # Risposta sì/no, info singola
    SIMPLE = 2      # 2-3 informazioni, concetti base
    MODERATE = 3    # 5-7 informazioni, alcuni tecnicismi
    COMPLEX = 4     # 8-12 informazioni, molto tecnico
    OVERWHELMING = 5 # >12 info, altamente specializzato


class CognitiveEvent(Enum):
    """Eventi che influenzano il carico cognitivo"""
    NEW_INFORMATION = "new_info"
    DECISION_REQUIRED = "decision"
    ERROR_MESSAGE = "error"
    COMPLEX_OUTPUT = "complex"
    RAPID_SEQUENCE = "rapid"
    TOPIC_SWITCH = "topic_switch"
    EMOTIONAL_CONTENT = "emotional"
    ACTION_CONFIRMATION = "action_confirm"


@dataclass
class UserInteraction:
    """Singola interazione utente"""
    timestamp: datetime
    interaction_type: str  # 'input', 'output', 'decision', 'action'
    complexity: OutputComplexity
    content_length: int
    technical_terms: int
    decisions_presented: int
    response_time: Optional[timedelta] = None  # Tempo di risposta utente
    user_hesitation: bool = False  # Se utente ha esitato


@dataclass
class CognitiveState:
    """Stato cognitivo corrente dell'utente"""
    load_level: LoadLevel
    cumulative_score: float  # 0-100
    info_density: float      # Info per minuto
    decision_pressure: float # Pressione decisionale
    fatigue_indicator: float # Indicatore fatica
    focus_quality: float     # Qualità focus stimata
    last_break: Optional[datetime]
    session_duration: timedelta
    topics_covered: int
    warnings: List[str]


@dataclass
class LoadReduction:
    """Suggerimento per riduzione carico"""
    strategy: str
    original_complexity: OutputComplexity
    target_complexity: OutputComplexity
    actions: List[str]
    expected_improvement: float  # Percentuale


class CognitiveLoadMonitor:
    """
    Monitor del carico cognitivo per GIDEON.
    
    Monitora e adatta l'interazione basandosi su:
    - Quantità di informazioni presentate
    - Complessità degli output
    - Frequenza decisioni richieste
    - Tempo di risposta dell'utente
    - Durata della sessione
    """
    
    # Costanti
    MAX_INFO_PER_MINUTE = 15
    MAX_DECISIONS_IN_ROW = 3
    OPTIMAL_SESSION_MINUTES = 45
    BREAK_SUGGESTION_THRESHOLD = 0.7  # 70% carico
    
    # Pesi per calcolo carico
    WEIGHTS = {
        'info_density': 0.25,
        'decision_pressure': 0.30,
        'complexity': 0.20,
        'fatigue': 0.15,
        'rapid_fire': 0.10
    }
    
    def __init__(self):
        self.interactions: deque = deque(maxlen=100)  # Ultime 100 interazioni
        self.session_start: datetime = datetime.now()
        self.last_break: Optional[datetime] = None
        self.decisions_in_row: int = 0
        self.topics: List[str] = []
        self.current_state: CognitiveState = self._initial_state()
        self.adaptations_made: List[Dict[str, Any]] = []
        
    def _initial_state(self) -> CognitiveState:
        """Stato iniziale"""
        return CognitiveState(
            load_level=LoadLevel.MINIMAL,
            cumulative_score=0.0,
            info_density=0.0,
            decision_pressure=0.0,
            fatigue_indicator=0.0,
            focus_quality=1.0,
            last_break=None,
            session_duration=timedelta(0),
            topics_covered=0,
            warnings=[]
        )
    
    def record_interaction(
        self,
        interaction_type: str,
        content: str,
        decisions_presented: int = 0,
        response_time: Optional[float] = None,  # Secondi
        user_hesitation: bool = False
    ) -> CognitiveState:
        """Registra un'interazione e aggiorna lo stato"""
        
        # Analizza contenuto
        complexity = self._analyze_complexity(content)
        technical_terms = self._count_technical_terms(content)
        
        interaction = UserInteraction(
            timestamp=datetime.now(),
            interaction_type=interaction_type,
            complexity=complexity,
            content_length=len(content),
            technical_terms=technical_terms,
            decisions_presented=decisions_presented,
            response_time=timedelta(seconds=response_time) if response_time else None,
            user_hesitation=user_hesitation
        )
        
        self.interactions.append(interaction)
        
        # Aggiorna contatore decisioni
        if decisions_presented > 0:
            self.decisions_in_row += decisions_presented
        elif interaction_type == 'input':
            self.decisions_in_row = 0  # Reset quando utente risponde
        
        # Aggiorna stato
        self.current_state = self._calculate_state()
        
        logger.debug(f"🧠 Interazione: {interaction_type}, Carico: {self.current_state.load_level.name}")
        
        return self.current_state
    
    def _analyze_complexity(self, content: str) -> OutputComplexity:
        """Analizza complessità del contenuto"""
        
        # Conteggi base
        sentences = len(re.findall(r'[.!?]+', content)) or 1
        words = len(content.split())
        technical = self._count_technical_terms(content)
        lists = len(re.findall(r'^\s*[-*•]\s', content, re.MULTILINE))
        code_blocks = len(re.findall(r'```', content)) // 2
        
        # Score complessità
        score = 0
        
        # Lunghezza
        if words > 500:
            score += 3
        elif words > 200:
            score += 2
        elif words > 100:
            score += 1
        
        # Termini tecnici
        technical_ratio = technical / max(words, 1)
        if technical_ratio > 0.15:
            score += 2
        elif technical_ratio > 0.05:
            score += 1
        
        # Struttura
        if lists > 5 or code_blocks > 2:
            score += 1
        
        # Densità info (words per sentence)
        density = words / sentences
        if density > 30:
            score += 1
        
        # Mappa a enum
        if score <= 1:
            return OutputComplexity.TRIVIAL
        elif score <= 2:
            return OutputComplexity.SIMPLE
        elif score <= 4:
            return OutputComplexity.MODERATE
        elif score <= 6:
            return OutputComplexity.COMPLEX
        else:
            return OutputComplexity.OVERWHELMING
    
    def _count_technical_terms(self, content: str) -> int:
        """Conta termini tecnici"""
        
        technical_patterns = [
            r'\b(API|SDK|CLI|GUI|URL|HTTP|SQL|JSON|XML)\b',
            r'\b\w+\(\)',  # Funzioni
            r'\b[A-Z][a-z]+[A-Z]\w*\b',  # CamelCase
            r'\b\d+\.\d+\.\d+\b',  # Versioni
            r'`[^`]+`',  # Codice inline
            r'\b(exception|error|warning|debug|log)\b',
            r'\b(latency|throughput|bandwidth|protocol)\b',
            r'\b(authentication|authorization|token|session)\b',
        ]
        
        count = 0
        for pattern in technical_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))
        
        return count
    
    def _calculate_state(self) -> CognitiveState:
        """Calcola stato cognitivo corrente"""
        
        now = datetime.now()
        session_duration = now - self.session_start
        
        # Calcola metriche
        info_density = self._calculate_info_density()
        decision_pressure = self._calculate_decision_pressure()
        fatigue = self._calculate_fatigue(session_duration)
        focus = self._estimate_focus_quality()
        rapid_fire = self._detect_rapid_fire()
        
        # Score composito (0-100)
        cumulative = (
            info_density * self.WEIGHTS['info_density'] +
            decision_pressure * self.WEIGHTS['decision_pressure'] +
            self._complexity_score() * self.WEIGHTS['complexity'] +
            fatigue * self.WEIGHTS['fatigue'] +
            rapid_fire * self.WEIGHTS['rapid_fire']
        ) * 100
        
        cumulative = min(100, max(0, cumulative))
        
        # Determina livello
        load_level = self._score_to_level(cumulative)
        
        # Genera warnings
        warnings = self._generate_warnings(
            cumulative, info_density, decision_pressure, fatigue, session_duration
        )
        
        return CognitiveState(
            load_level=load_level,
            cumulative_score=cumulative,
            info_density=info_density,
            decision_pressure=decision_pressure,
            fatigue_indicator=fatigue,
            focus_quality=focus,
            last_break=self.last_break,
            session_duration=session_duration,
            topics_covered=len(set(self.topics)),
            warnings=warnings
        )
    
    def _calculate_info_density(self) -> float:
        """Calcola densità informazioni (0-1)"""
        
        # Interazioni ultimo minuto
        one_min_ago = datetime.now() - timedelta(minutes=1)
        recent = [i for i in self.interactions if i.timestamp > one_min_ago]
        
        if not recent:
            return 0.0
        
        # Info = lunghezza contenuto + termini tecnici
        total_info = sum(i.content_length / 100 + i.technical_terms for i in recent)
        
        # Normalizza
        return min(1.0, total_info / self.MAX_INFO_PER_MINUTE)
    
    def _calculate_decision_pressure(self) -> float:
        """Calcola pressione decisionale (0-1)"""
        
        # Decisioni consecutive
        consecutive_factor = min(1.0, self.decisions_in_row / self.MAX_DECISIONS_IN_ROW)
        
        # Decisioni recenti (ultimi 5 minuti)
        five_min_ago = datetime.now() - timedelta(minutes=5)
        recent_decisions = sum(
            i.decisions_presented for i in self.interactions 
            if i.timestamp > five_min_ago
        )
        
        recent_factor = min(1.0, recent_decisions / 10)
        
        return (consecutive_factor * 0.6 + recent_factor * 0.4)
    
    def _calculate_fatigue(self, session_duration: timedelta) -> float:
        """Calcola indicatore fatica (0-1)"""
        
        session_minutes = session_duration.total_seconds() / 60
        
        # Fatica base da durata
        base_fatigue = min(1.0, session_minutes / (self.OPTIMAL_SESSION_MINUTES * 2))
        
        # Bonus se nessuna pausa
        if self.last_break is None and session_minutes > self.OPTIMAL_SESSION_MINUTES:
            base_fatigue *= 1.3
        
        # Fatica da cambio topic frequente
        if len(set(self.topics[-10:])) > 5:  # Molti topic diversi
            base_fatigue *= 1.2
        
        return min(1.0, base_fatigue)
    
    def _estimate_focus_quality(self) -> float:
        """Stima qualità del focus (0-1, 1 = ottimo)"""
        
        # Basato su tempo di risposta
        recent = list(self.interactions)[-10:]
        
        if not recent:
            return 1.0
        
        # Check esitazioni
        hesitations = sum(1 for i in recent if i.user_hesitation)
        
        # Check tempi risposta lunghi
        slow_responses = sum(
            1 for i in recent 
            if i.response_time and i.response_time > timedelta(seconds=30)
        )
        
        focus = 1.0 - (hesitations * 0.1) - (slow_responses * 0.1)
        
        return max(0.0, focus)
    
    def _detect_rapid_fire(self) -> float:
        """Rileva sequenze rapide (0-1)"""
        
        recent = list(self.interactions)[-10:]
        
        if len(recent) < 2:
            return 0.0
        
        # Calcola intervalli
        intervals = []
        for i in range(1, len(recent)):
            delta = (recent[i].timestamp - recent[i-1].timestamp).total_seconds()
            intervals.append(delta)
        
        # Media intervalli
        avg_interval = sum(intervals) / len(intervals) if intervals else float('inf')
        
        # Rapid se media < 5 secondi
        if avg_interval < 5:
            return 1.0
        elif avg_interval < 15:
            return 0.5
        elif avg_interval < 30:
            return 0.2
        
        return 0.0
    
    def _complexity_score(self) -> float:
        """Score di complessità media recente (0-1)"""
        
        recent = list(self.interactions)[-10:]
        
        if not recent:
            return 0.0
        
        avg_complexity = sum(i.complexity.value for i in recent) / len(recent)
        
        return (avg_complexity - 1) / 4  # Normalizza 1-5 a 0-1
    
    def _score_to_level(self, score: float) -> LoadLevel:
        """Converte score a livello"""
        
        if score < 20:
            return LoadLevel.MINIMAL
        elif score < 40:
            return LoadLevel.LOW
        elif score < 60:
            return LoadLevel.MODERATE
        elif score < 80:
            return LoadLevel.HIGH
        else:
            return LoadLevel.CRITICAL
    
    def _generate_warnings(
        self,
        score: float,
        info_density: float,
        decision_pressure: float,
        fatigue: float,
        session_duration: timedelta
    ) -> List[str]:
        """Genera warnings appropriati"""
        
        warnings = []
        
        if score >= 80:
            warnings.append("⚠️ CARICO CRITICO: Considerare una pausa")
        
        if info_density > 0.8:
            warnings.append("📊 Troppe informazioni in poco tempo")
        
        if decision_pressure > 0.7:
            warnings.append("🎯 Troppe decisioni consecutive richieste")
        
        if fatigue > 0.7:
            warnings.append("😴 Sessione lunga - suggerita pausa")
        
        if session_duration > timedelta(minutes=self.OPTIMAL_SESSION_MINUTES):
            if self.last_break is None:
                warnings.append(f"⏰ Sessione attiva da {session_duration} senza pause")
        
        return warnings
    
    async def suggest_adaptation(
        self,
        content: str,
        current_complexity: Optional[OutputComplexity] = None
    ) -> LoadReduction:
        """Suggerisce adattamento basato sullo stato"""
        
        state = self.current_state
        
        if current_complexity is None:
            current_complexity = self._analyze_complexity(content)
        
        # Determina complessità target
        if state.load_level.value >= 4:  # HIGH o CRITICAL
            target = OutputComplexity.SIMPLE
        elif state.load_level.value >= 3:  # MODERATE
            target = OutputComplexity.MODERATE
        else:
            target = current_complexity  # Nessuna riduzione
        
        # Genera azioni
        actions = []
        
        if current_complexity.value > target.value:
            if current_complexity.value - target.value >= 2:
                actions.append("Dividere in più messaggi brevi")
                actions.append("Rimuovere dettagli non essenziali")
            actions.append("Semplificare linguaggio tecnico")
            actions.append("Usare elenchi puntati invece di paragrafi")
        
        if state.decision_pressure > 0.5:
            actions.append("Ridurre opzioni presentate")
            actions.append("Suggerire una scelta predefinita")
        
        if state.info_density > 0.6:
            actions.append("Attendere prima di aggiungere info")
            actions.append("Chiedere se pronto per altri dettagli")
        
        if state.fatigue_indicator > 0.6:
            actions.append("Suggerire una pausa")
            actions.append("Offrire summary invece di dettagli")
        
        if not actions:
            actions.append("Nessun adattamento necessario")
        
        improvement = (current_complexity.value - target.value) * 0.2 if current_complexity.value > target.value else 0
        
        return LoadReduction(
            strategy="simplify" if current_complexity.value > target.value else "maintain",
            original_complexity=current_complexity,
            target_complexity=target,
            actions=actions,
            expected_improvement=improvement
        )
    
    def simplify_output(
        self,
        content: str,
        target_complexity: OutputComplexity = OutputComplexity.SIMPLE
    ) -> str:
        """Semplifica output se necessario"""
        
        current = self._analyze_complexity(content)
        
        if current.value <= target_complexity.value:
            return content
        
        # Strategie di semplificazione
        simplified = content
        
        # 1. Rimuovi blocchi codice non essenziali
        if target_complexity.value <= 2:
            simplified = re.sub(r'```[\s\S]*?```', '[Codice rimosso per chiarezza]', simplified)
        
        # 2. Accorcia liste lunghe
        lines = simplified.split('\n')
        if len(lines) > 10 and target_complexity.value <= 3:
            # Mantieni prime 5 e ultime 2
            simplified = '\n'.join(lines[:5] + ['...'] + lines[-2:])
        
        # 3. Rimuovi parentesi con dettagli
        if target_complexity.value <= 2:
            simplified = re.sub(r'\([^)]{50,}\)', '', simplified)
        
        # 4. Tronca se ancora troppo lungo
        max_length = {1: 200, 2: 400, 3: 800, 4: 1500, 5: 3000}
        limit = max_length.get(target_complexity.value, 1000)
        
        if len(simplified) > limit:
            simplified = simplified[:limit-20] + '...\n[Output semplificato]'
        
        return simplified
    
    def record_break(self):
        """Registra una pausa"""
        self.last_break = datetime.now()
        logger.info("🧠 Pausa registrata")
    
    def record_topic_change(self, topic: str):
        """Registra cambio di argomento"""
        self.topics.append(topic)
    
    def get_status(self) -> Dict[str, Any]:
        """Ritorna stato corrente formattato"""
        
        state = self.current_state
        
        return {
            'load_level': state.load_level.name,
            'score': round(state.cumulative_score, 1),
            'info_density': round(state.info_density * 100, 1),
            'decision_pressure': round(state.decision_pressure * 100, 1),
            'fatigue': round(state.fatigue_indicator * 100, 1),
            'focus_quality': round(state.focus_quality * 100, 1),
            'session_duration': str(state.session_duration),
            'topics_covered': state.topics_covered,
            'last_break': state.last_break.isoformat() if state.last_break else None,
            'warnings': state.warnings,
            'recommendation': self._get_recommendation(state)
        }
    
    def _get_recommendation(self, state: CognitiveState) -> str:
        """Genera raccomandazione basata sullo stato"""
        
        if state.load_level == LoadLevel.CRITICAL:
            return "🛑 STOP: Prendere una pausa immediata"
        elif state.load_level == LoadLevel.HIGH:
            return "⚠️ Semplificare output e ridurre decisioni"
        elif state.load_level == LoadLevel.MODERATE:
            return "📊 Monitorare carico, evitare info aggiuntive non richieste"
        elif state.load_level == LoadLevel.LOW:
            return "✅ Carico gestibile, procedere normalmente"
        else:
            return "🌟 Ottimo: utente pronto per informazioni dettagliate"
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        status = self.get_status()
        
        level_emoji = {
            'MINIMAL': '🌟',
            'LOW': '✅',
            'MODERATE': '📊',
            'HIGH': '⚠️',
            'CRITICAL': '🛑'
        }
        
        emoji = level_emoji.get(status['load_level'], '❓')
        
        return f"""
# 🧠 Cognitive Load Monitor

{emoji} **Livello Carico**: {status['load_level']}
📈 **Score**: {status['score']}/100

---

## Metriche
| Indicatore | Valore |
|------------|--------|
| Densità Info | {status['info_density']}% |
| Pressione Decisioni | {status['decision_pressure']}% |
| Indicatore Fatica | {status['fatigue']}% |
| Qualità Focus | {status['focus_quality']}% |

---

## Sessione
- **Durata**: {status['session_duration']}
- **Topic Trattati**: {status['topics_covered']}
- **Ultima Pausa**: {status['last_break'] or 'Nessuna'}

---

## Warnings
{chr(10).join(f"- {w}" for w in status['warnings']) or '✅ Nessun warning'}

---

## Raccomandazione
{status['recommendation']}
"""
