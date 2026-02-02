"""
🎯 INTENT-AWARE SYSTEM
======================
GIDEON inferisce intenti oltre le richieste esplicite:
- Analisi profonda delle parole
- Riconoscimento obiettivi impliciti
- Anticipazione bisogni reali
- Clarificazione proattiva

"Chiedi X, ma il tuo obiettivo sembra Y - confermi?"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import logging
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Tipi di intent rilevabili"""
    # Intent diretti
    QUERY = "query"  # Domanda informativa
    ACTION = "action"  # Richiesta azione
    CREATION = "creation"  # Creare qualcosa
    MODIFICATION = "modification"  # Modificare
    DELETION = "deletion"  # Eliminare
    
    # Intent impliciti
    LEARNING = "learning"  # Vuole imparare
    DEBUGGING = "debugging"  # Sta debuggando
    PLANNING = "planning"  # Sta pianificando
    COMPARING = "comparing"  # Sta comparando opzioni
    VALIDATING = "validating"  # Vuole conferma
    EXPLORING = "exploring"  # Sta esplorando opzioni


class ConfidenceLevel(Enum):
    """Livello di confidenza nell'inferenza"""
    HIGH = "high"  # > 85%
    MEDIUM = "medium"  # 60-85%
    LOW = "low"  # < 60%


class ClarificationNeed(Enum):
    """Necessità di clarificazione"""
    NONE = "none"  # Chiaro
    RECOMMENDED = "recommended"  # Consigliata
    REQUIRED = "required"  # Necessaria


@dataclass
class InferredIntent:
    """Intent inferito"""
    primary_intent: IntentType
    secondary_intents: List[IntentType]
    confidence: float
    confidence_level: ConfidenceLevel
    raw_request: str
    interpreted_goal: str
    assumptions_made: List[str]
    clarification_need: ClarificationNeed
    suggested_clarification: Optional[str] = None


@dataclass
class IntentContext:
    """Contesto per inferenza intent"""
    recent_topics: List[str]
    current_task: Optional[str]
    user_expertise: str  # "beginner", "intermediate", "expert"
    conversation_mood: str  # "casual", "urgent", "technical"
    time_context: str  # "working", "after_hours"


@dataclass
class IntentPattern:
    """Pattern per riconoscimento intent"""
    keywords: List[str]
    phrases: List[str]
    context_signals: List[str]
    intent_type: IntentType
    weight: float = 1.0


class IntentAwareSystem:
    """
    Sistema di inferenza intenti per GIDEON.
    
    Analizza richieste per comprendere:
    - Obiettivi reali (non solo espliciti)
    - Contesto della richiesta
    - Bisogni sottostanti
    """
    
    def __init__(self):
        self.intent_patterns = self._initialize_patterns()
        self.intent_history: List[InferredIntent] = []
        self.user_patterns: Dict[str, List[IntentType]] = defaultdict(list)
        
        # Mapping obiettivi impliciti
        self.implicit_goals = {
            "how do i": IntentType.LEARNING,
            "come faccio": IntentType.LEARNING,
            "why isn't": IntentType.DEBUGGING,
            "perché non": IntentType.DEBUGGING,
            "which is better": IntentType.COMPARING,
            "quale è meglio": IntentType.COMPARING,
            "is this correct": IntentType.VALIDATING,
            "è corretto": IntentType.VALIDATING,
            "what are my options": IntentType.EXPLORING,
            "quali opzioni": IntentType.EXPLORING,
        }
    
    def _initialize_patterns(self) -> List[IntentPattern]:
        """Inizializza pattern di riconoscimento"""
        
        return [
            # Query patterns
            IntentPattern(
                keywords=["what", "why", "how", "when", "where", "cosa", "perché", "come", "quando", "dove"],
                phrases=["tell me", "explain", "dimmi", "spiega"],
                context_signals=["?"],
                intent_type=IntentType.QUERY,
                weight=1.0
            ),
            
            # Action patterns
            IntentPattern(
                keywords=["run", "execute", "start", "stop", "esegui", "avvia", "ferma"],
                phrases=["please do", "can you", "puoi", "per favore"],
                context_signals=["!"],
                intent_type=IntentType.ACTION,
                weight=1.2
            ),
            
            # Creation patterns
            IntentPattern(
                keywords=["create", "make", "build", "generate", "crea", "costruisci", "genera"],
                phrases=["new file", "add", "implement", "nuovo file", "aggiungi", "implementa"],
                context_signals=[],
                intent_type=IntentType.CREATION,
                weight=1.1
            ),
            
            # Modification patterns
            IntentPattern(
                keywords=["change", "update", "modify", "fix", "cambia", "aggiorna", "modifica", "correggi"],
                phrases=["refactor", "improve", "migliora", "ottimizza"],
                context_signals=[],
                intent_type=IntentType.MODIFICATION,
                weight=1.1
            ),
            
            # Debugging patterns
            IntentPattern(
                keywords=["error", "bug", "issue", "problem", "errore", "problema", "crash"],
                phrases=["doesn't work", "not working", "non funziona", "broken"],
                context_signals=["traceback", "exception", "stacktrace"],
                intent_type=IntentType.DEBUGGING,
                weight=1.5
            ),
            
            # Learning patterns
            IntentPattern(
                keywords=["learn", "understand", "explain", "tutorial", "impara", "capire", "spiega"],
                phrases=["how does", "what is", "come funziona", "cos'è"],
                context_signals=["example", "esempio"],
                intent_type=IntentType.LEARNING,
                weight=1.0
            ),
            
            # Planning patterns
            IntentPattern(
                keywords=["plan", "design", "architect", "structure", "pianifica", "progetta", "architettura"],
                phrases=["best way to", "how should i", "modo migliore per", "come dovrei"],
                context_signals=["steps", "phases", "fasi"],
                intent_type=IntentType.PLANNING,
                weight=1.0
            ),
        ]
    
    def infer_intent(
        self,
        request: str,
        context: Optional[IntentContext] = None
    ) -> InferredIntent:
        """
        Inferisce intent da una richiesta.
        
        Analizza:
        1. Pattern linguistici
        2. Contesto conversazione
        3. Storia utente
        4. Obiettivi impliciti
        """
        
        request_lower = request.lower()
        
        # Score per ogni tipo di intent
        intent_scores: Dict[IntentType, float] = defaultdict(float)
        
        # 1. Analisi pattern
        for pattern in self.intent_patterns:
            score = self._match_pattern(request_lower, pattern)
            intent_scores[pattern.intent_type] += score
        
        # 2. Analisi obiettivi impliciti
        for phrase, intent_type in self.implicit_goals.items():
            if phrase in request_lower:
                intent_scores[intent_type] += 0.5
        
        # 3. Boost da contesto
        if context:
            self._apply_context_boost(intent_scores, context)
        
        # 4. Boost da storia
        self._apply_history_boost(intent_scores)
        
        # Determina intent primario e secondari
        sorted_intents = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        primary_intent = sorted_intents[0][0] if sorted_intents else IntentType.QUERY
        primary_score = sorted_intents[0][1] if sorted_intents else 0
        
        # Normalizza score
        total_score = sum(s for _, s in sorted_intents) or 1
        confidence = min(primary_score / total_score, 1.0) if total_score > 0 else 0.5
        
        # Secondary intents (score > 30% del primario)
        secondary_intents = [
            intent for intent, score in sorted_intents[1:4]
            if score > primary_score * 0.3
        ]
        
        # Determina confidence level
        confidence_level = self._determine_confidence_level(confidence)
        
        # Interpreta obiettivo
        interpreted_goal = self._interpret_goal(request, primary_intent, secondary_intents)
        
        # Assunzioni fatte
        assumptions = self._identify_assumptions(request, primary_intent, context)
        
        # Necessità clarificazione
        clarification_need = self._assess_clarification_need(
            confidence, len(assumptions), secondary_intents
        )
        
        # Suggerimento clarificazione
        suggested_clarification = None
        if clarification_need != ClarificationNeed.NONE:
            suggested_clarification = self._generate_clarification(
                request, primary_intent, secondary_intents, assumptions
            )
        
        inferred = InferredIntent(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            confidence_level=confidence_level,
            raw_request=request,
            interpreted_goal=interpreted_goal,
            assumptions_made=assumptions,
            clarification_need=clarification_need,
            suggested_clarification=suggested_clarification
        )
        
        # Registra in storia
        self.intent_history.append(inferred)
        self.user_patterns["recent"].append(primary_intent)
        if len(self.user_patterns["recent"]) > 20:
            self.user_patterns["recent"] = self.user_patterns["recent"][-20:]
        
        return inferred
    
    def _match_pattern(self, text: str, pattern: IntentPattern) -> float:
        """Calcola score match pattern"""
        
        score = 0.0
        
        # Match keywords
        for keyword in pattern.keywords:
            if keyword in text:
                score += 0.3 * pattern.weight
        
        # Match phrases
        for phrase in pattern.phrases:
            if phrase in text:
                score += 0.5 * pattern.weight
        
        # Match context signals
        for signal in pattern.context_signals:
            if signal in text:
                score += 0.2 * pattern.weight
        
        return score
    
    def _apply_context_boost(
        self,
        scores: Dict[IntentType, float],
        context: IntentContext
    ):
        """Applica boost basato su contesto"""
        
        # Boost per task corrente
        if context.current_task:
            if "debug" in context.current_task.lower():
                scores[IntentType.DEBUGGING] *= 1.3
            elif "implement" in context.current_task.lower():
                scores[IntentType.CREATION] *= 1.3
        
        # Boost per expertise
        if context.user_expertise == "beginner":
            scores[IntentType.LEARNING] *= 1.2
        elif context.user_expertise == "expert":
            scores[IntentType.ACTION] *= 1.1
        
        # Boost per mood
        if context.conversation_mood == "urgent":
            scores[IntentType.DEBUGGING] *= 1.2
            scores[IntentType.ACTION] *= 1.2
        elif context.conversation_mood == "casual":
            scores[IntentType.EXPLORING] *= 1.2
    
    def _apply_history_boost(self, scores: Dict[IntentType, float]):
        """Applica boost basato su storia"""
        
        recent = self.user_patterns.get("recent", [])
        
        if len(recent) >= 3:
            # Ultimo intent ripetuto
            last_3 = recent[-3:]
            for intent_type in set(last_3):
                count = last_3.count(intent_type)
                if count >= 2:
                    scores[intent_type] *= (1 + count * 0.1)
    
    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determina livello confidenza"""
        
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _interpret_goal(
        self,
        request: str,
        primary: IntentType,
        secondary: List[IntentType]
    ) -> str:
        """Genera interpretazione obiettivo"""
        
        goal_templates = {
            IntentType.QUERY: "Ottenere informazioni su: {topic}",
            IntentType.ACTION: "Eseguire: {topic}",
            IntentType.CREATION: "Creare: {topic}",
            IntentType.MODIFICATION: "Modificare: {topic}",
            IntentType.DELETION: "Eliminare: {topic}",
            IntentType.LEARNING: "Capire come funziona: {topic}",
            IntentType.DEBUGGING: "Risolvere problema con: {topic}",
            IntentType.PLANNING: "Pianificare: {topic}",
            IntentType.COMPARING: "Comparare opzioni per: {topic}",
            IntentType.VALIDATING: "Validare approccio per: {topic}",
            IntentType.EXPLORING: "Esplorare possibilità per: {topic}",
        }
        
        # Estrai topic approssimativo
        topic = self._extract_topic(request)
        
        goal = goal_templates.get(primary, "Assistere con: {topic}").format(topic=topic)
        
        # Aggiungi secondary se rilevanti
        if secondary:
            secondary_goals = [
                goal_templates.get(s, "").format(topic=topic).split(":")[0]
                for s in secondary[:2]
            ]
            if secondary_goals:
                goal += f" (anche: {', '.join(secondary_goals)})"
        
        return goal
    
    def _extract_topic(self, request: str) -> str:
        """Estrae topic principale dalla richiesta"""
        
        # Rimuovi parole comuni
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
            "di", "da", "in", "su", "per", "con", "come", "che", "cosa",
            "can", "you", "please", "help", "me", "mi", "puoi", "per", "favore"
        }
        
        words = request.lower().split()
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Prendi le prime parole rilevanti
        return " ".join(content_words[:5]) if content_words else request[:50]
    
    def _identify_assumptions(
        self,
        request: str,
        intent: IntentType,
        context: Optional[IntentContext]
    ) -> List[str]:
        """Identifica assunzioni fatte"""
        
        assumptions = []
        
        # Assunzioni per intent
        if intent == IntentType.CREATION:
            assumptions.append("Vuoi creare un nuovo item (non modificare esistente)")
        
        if intent == IntentType.DEBUGGING:
            assumptions.append("Il problema è riproducibile")
            assumptions.append("Hai accesso ai log/errori")
        
        if intent == IntentType.ACTION:
            assumptions.append("Hai le autorizzazioni necessarie")
        
        # Assunzioni da contesto
        if context:
            if context.time_context == "after_hours":
                assumptions.append("Questo è lavoro urgente fuori orario")
        
        # Assunzioni da request
        if "all" in request.lower() or "tutto" in request.lower():
            assumptions.append("'Tutto' si riferisce allo scope corrente")
        
        if "best" in request.lower() or "migliore" in request.lower():
            assumptions.append("'Migliore' considera trade-off bilanciati")
        
        return assumptions
    
    def _assess_clarification_need(
        self,
        confidence: float,
        num_assumptions: int,
        secondary_intents: List[IntentType]
    ) -> ClarificationNeed:
        """Valuta necessità di clarificazione"""
        
        # Score di ambiguità
        ambiguity_score = 0
        
        # Confidenza bassa aumenta ambiguità
        if confidence < 0.6:
            ambiguity_score += 2
        elif confidence < 0.8:
            ambiguity_score += 1
        
        # Molte assunzioni aumentano ambiguità
        ambiguity_score += min(num_assumptions, 3) * 0.5
        
        # Secondary intents forti aumentano ambiguità
        ambiguity_score += len(secondary_intents) * 0.3
        
        if ambiguity_score >= 3:
            return ClarificationNeed.REQUIRED
        elif ambiguity_score >= 1.5:
            return ClarificationNeed.RECOMMENDED
        else:
            return ClarificationNeed.NONE
    
    def _generate_clarification(
        self,
        request: str,
        primary: IntentType,
        secondary: List[IntentType],
        assumptions: List[str]
    ) -> str:
        """Genera domanda di clarificazione"""
        
        if secondary:
            # Clarifica tra intent multipli
            options = [self._intent_to_phrase(i) for i in [primary] + secondary]
            return f"Per assicurarmi di capire correttamente: vorresti {options[0]}, oppure {', '.join(options[1:])}?"
        
        if assumptions:
            # Clarifica assunzioni
            main_assumption = assumptions[0]
            return f"Assumo che {main_assumption.lower()}. È corretto?"
        
        return f"Puoi specificare meglio cosa intendi con '{request[:50]}...'?"
    
    def _intent_to_phrase(self, intent: IntentType) -> str:
        """Converte intent in frase leggibile"""
        
        phrases = {
            IntentType.QUERY: "ottenere informazioni",
            IntentType.ACTION: "eseguire un'azione",
            IntentType.CREATION: "creare qualcosa di nuovo",
            IntentType.MODIFICATION: "modificare qualcosa di esistente",
            IntentType.DELETION: "eliminare qualcosa",
            IntentType.LEARNING: "imparare come funziona",
            IntentType.DEBUGGING: "risolvere un problema",
            IntentType.PLANNING: "pianificare un'attività",
            IntentType.COMPARING: "comparare opzioni",
            IntentType.VALIDATING: "validare un approccio",
            IntentType.EXPLORING: "esplorare possibilità",
        }
        
        return phrases.get(intent, str(intent.value))
    
    def should_clarify(self, inferred: InferredIntent) -> bool:
        """Determina se chiedere clarificazione"""
        return inferred.clarification_need != ClarificationNeed.NONE
    
    def get_clarification_message(self, inferred: InferredIntent) -> Optional[str]:
        """Ottiene messaggio di clarificazione formattato"""
        
        if not inferred.suggested_clarification:
            return None
        
        if inferred.clarification_need == ClarificationNeed.REQUIRED:
            prefix = "🔍 **Clarificazione necessaria**"
        else:
            prefix = "💡 **Suggerimento**"
        
        return f"{prefix}\n\n{inferred.suggested_clarification}"
    
    def format_intent_analysis(self, inferred: InferredIntent) -> str:
        """Formatta analisi intent per visualizzazione"""
        
        confidence_emoji = {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🔴"
        }
        
        clarification_emoji = {
            ClarificationNeed.NONE: "✅",
            ClarificationNeed.RECOMMENDED: "💡",
            ClarificationNeed.REQUIRED: "⚠️"
        }
        
        secondary_str = ", ".join(
            self._intent_to_phrase(i) for i in inferred.secondary_intents
        ) if inferred.secondary_intents else "Nessuno"
        
        assumptions_str = "\n".join(
            f"  - {a}" for a in inferred.assumptions_made
        ) if inferred.assumptions_made else "  Nessuna"
        
        return f"""
## 🎯 Analisi Intent

**Richiesta originale:** "{inferred.raw_request[:100]}..."

### Intent Rilevato
| Campo | Valore |
|-------|--------|
| Intent primario | **{self._intent_to_phrase(inferred.primary_intent)}** |
| Intent secondari | {secondary_str} |
| Confidenza | {confidence_emoji[inferred.confidence_level]} {inferred.confidence:.1%} ({inferred.confidence_level.value}) |

### Interpretazione
**Obiettivo interpretato:** {inferred.interpreted_goal}

### Assunzioni fatte
{assumptions_str}

### Clarificazione
{clarification_emoji[inferred.clarification_need]} **{inferred.clarification_need.value.title()}**
{f'> {inferred.suggested_clarification}' if inferred.suggested_clarification else ''}
"""


# Singleton
_intent_system: Optional[IntentAwareSystem] = None


def get_intent_system() -> IntentAwareSystem:
    """Ottiene istanza singleton"""
    global _intent_system
    if _intent_system is None:
        _intent_system = IntentAwareSystem()
    return _intent_system
