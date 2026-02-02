"""
🗣️ Natural Voice Engine - GIDEON 3.0

Sistema di sintesi vocale naturale con:
- Pause realistiche basate su punteggiatura e contenuto
- Intonazioni coerenti con tipo di messaggio
- Enfasi su parole chiave
- Velocità adattiva
- Espressività vocale
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum
import re
import math
import asyncio
import logging
from datetime import datetime


# Logger
voice_logger = logging.getLogger("natural_voice")
voice_logger.setLevel(logging.DEBUG)


# === ENUMS ===

class VoiceEmotion(Enum):
    """Emozioni vocali"""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    SERIOUS = "serious"
    EXCITED = "excited"
    CONCERNED = "concerned"
    CONFIDENT = "confident"
    CURIOUS = "curious"
    CALM = "calm"
    URGENT = "urgent"


class PauseType(Enum):
    """Tipi di pausa"""
    NONE = 0
    MICRO = 50          # 50ms - tra parole
    SHORT = 150         # 150ms - virgola
    MEDIUM = 300        # 300ms - punto e virgola
    LONG = 500          # 500ms - punto
    PARAGRAPH = 800     # 800ms - nuovo paragrafo
    DRAMATIC = 1200     # 1.2s - enfasi drammatica


class IntonationPattern(Enum):
    """Pattern di intonazione"""
    STATEMENT = "statement"       # Affermazione - tono discendente
    QUESTION = "question"         # Domanda - tono ascendente
    EXCLAMATION = "exclamation"   # Esclamazione - tono alto
    LIST = "list"                 # Lista - tono costante
    EMPHASIS = "emphasis"         # Enfasi - tono variato
    WHISPER = "whisper"           # Sussurro - tono basso
    ANNOUNCEMENT = "announcement" # Annuncio - tono elevato


# === DATA CLASSES ===

@dataclass
class SpeechSegment:
    """Segmento di testo da pronunciare"""
    text: str
    pause_before: int = 0           # ms
    pause_after: int = 0            # ms
    rate_multiplier: float = 1.0    # Velocità (0.5-2.0)
    pitch_shift: int = 0            # Semitoni (-12 to +12)
    volume: float = 1.0             # Volume (0-1)
    emphasis: bool = False          # Enfasi
    intonation: IntonationPattern = IntonationPattern.STATEMENT
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "pause_before": self.pause_before,
            "pause_after": self.pause_after,
            "rate": self.rate_multiplier,
            "pitch": self.pitch_shift,
            "volume": self.volume,
            "emphasis": self.emphasis,
            "intonation": self.intonation.value
        }


@dataclass
class VoiceProfile:
    """Profilo vocale"""
    name: str = "Gideon"
    base_rate: int = 175            # Parole per minuto base
    base_pitch: int = 0             # Pitch base
    base_volume: float = 0.9
    
    # Variazioni emotive
    emotion_settings: Dict[str, Dict] = field(default_factory=dict)
    
    # Preferenze pause
    pause_multiplier: float = 1.0
    
    def __post_init__(self):
        if not self.emotion_settings:
            self.emotion_settings = {
                VoiceEmotion.NEUTRAL.value: {"rate": 1.0, "pitch": 0, "volume": 0.9},
                VoiceEmotion.FRIENDLY.value: {"rate": 1.05, "pitch": 2, "volume": 0.95},
                VoiceEmotion.SERIOUS.value: {"rate": 0.95, "pitch": -2, "volume": 0.85},
                VoiceEmotion.EXCITED.value: {"rate": 1.15, "pitch": 4, "volume": 1.0},
                VoiceEmotion.CONCERNED.value: {"rate": 0.9, "pitch": -1, "volume": 0.8},
                VoiceEmotion.CONFIDENT.value: {"rate": 1.0, "pitch": 1, "volume": 0.95},
                VoiceEmotion.CURIOUS.value: {"rate": 1.0, "pitch": 3, "volume": 0.9},
                VoiceEmotion.CALM.value: {"rate": 0.85, "pitch": -3, "volume": 0.75},
                VoiceEmotion.URGENT.value: {"rate": 1.2, "pitch": 2, "volume": 1.0}
            }


@dataclass
class SpeechOutput:
    """Output completo per sintesi vocale"""
    segments: List[SpeechSegment] = field(default_factory=list)
    total_duration_estimate: float = 0.0  # secondi
    emotion: VoiceEmotion = VoiceEmotion.NEUTRAL
    profile: VoiceProfile = field(default_factory=VoiceProfile)
    
    # Metadata
    original_text: str = ""
    word_count: int = 0
    sentence_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "duration_estimate": round(self.total_duration_estimate, 2),
            "emotion": self.emotion.value,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count
        }
    
    def get_ssml(self) -> str:
        """Genera SSML per TTS avanzati"""
        ssml = ['<speak>']
        
        for seg in self.segments:
            if seg.pause_before > 0:
                ssml.append(f'<break time="{seg.pause_before}ms"/>')
            
            # Prosody
            rate = f"{int(seg.rate_multiplier * 100)}%"
            pitch = f"{seg.pitch_shift:+d}st" if seg.pitch_shift != 0 else "default"
            
            if seg.emphasis:
                ssml.append(f'<emphasis level="strong">')
            
            ssml.append(f'<prosody rate="{rate}" pitch="{pitch}">')
            ssml.append(seg.text)
            ssml.append('</prosody>')
            
            if seg.emphasis:
                ssml.append('</emphasis>')
            
            if seg.pause_after > 0:
                ssml.append(f'<break time="{seg.pause_after}ms"/>')
        
        ssml.append('</speak>')
        return ''.join(ssml)


# === NATURAL VOICE ENGINE ===

class NaturalVoiceEngine:
    """
    Engine per sintesi vocale naturale.
    
    Caratteristiche:
    - Analisi semantica del testo
    - Pause contestuali
    - Intonazione adattiva
    - Enfasi automatica
    - Supporto emozioni
    """
    
    def __init__(self, profile: VoiceProfile = None):
        self.profile = profile or VoiceProfile()
        
        # Pattern per analisi
        self._init_patterns()
        
        # Parole chiave per enfasi
        self.emphasis_words = {
            "importante", "attenzione", "urgente", "critico", "essenziale",
            "fondamentale", "nota", "ricorda", "mai", "sempre", "assolutamente",
            "importante", "attention", "urgent", "critical", "essential",
            "note", "remember", "never", "always", "absolutely"
        }
        
        # Connettori per pause
        self.connectors = {
            "inoltre": PauseType.MEDIUM,
            "quindi": PauseType.MEDIUM,
            "tuttavia": PauseType.MEDIUM,
            "però": PauseType.SHORT,
            "perché": PauseType.SHORT,
            "dunque": PauseType.MEDIUM,
            "infatti": PauseType.MEDIUM,
            "inoltre": PauseType.MEDIUM,
            "however": PauseType.MEDIUM,
            "therefore": PauseType.MEDIUM,
            "because": PauseType.SHORT,
            "finally": PauseType.MEDIUM
        }
        
        voice_logger.info("NaturalVoiceEngine initialized")
    
    def _init_patterns(self):
        """Inizializza pattern regex"""
        # Punteggiatura
        self.punct_pause = {
            '.': PauseType.LONG,
            '!': PauseType.LONG,
            '?': PauseType.LONG,
            ',': PauseType.SHORT,
            ';': PauseType.MEDIUM,
            ':': PauseType.MEDIUM,
            '-': PauseType.MICRO,
            '—': PauseType.SHORT,
            '...': PauseType.PARAGRAPH,
            '\n': PauseType.PARAGRAPH,
            '\n\n': PauseType.DRAMATIC
        }
        
        # Pattern domande
        self.question_starters = re.compile(
            r'^(chi|cosa|come|quando|dove|perché|quale|quanti|quanto|'
            r'who|what|how|when|where|why|which)\b',
            re.IGNORECASE
        )
        
        # Pattern liste
        self.list_pattern = re.compile(r'^[\d•\-\*]\s*[.)\]]?\s*', re.MULTILINE)
        
        # Pattern numeri/dati
        self.number_pattern = re.compile(r'\b\d+([.,]\d+)?%?\b')
        
        # Pattern enfasi (grassetto, maiuscolo)
        self.emphasis_pattern = re.compile(r'\*\*([^*]+)\*\*|__([^_]+)__|([A-Z]{2,})')
    
    def process_text(self, text: str, emotion: VoiceEmotion = None) -> SpeechOutput:
        """
        Processa testo per sintesi vocale naturale.
        
        Args:
            text: Testo da processare
            emotion: Emozione da applicare
        
        Returns:
            SpeechOutput con segmenti pronti per TTS
        """
        if not text:
            return SpeechOutput()
        
        # Rileva emozione se non specificata
        if emotion is None:
            emotion = self._detect_emotion(text)
        
        # Suddividi in frasi
        sentences = self._split_sentences(text)
        
        # Processa ogni frase
        segments = []
        for i, sentence in enumerate(sentences):
            sentence_segments = self._process_sentence(sentence, i, len(sentences), emotion)
            segments.extend(sentence_segments)
        
        # Calcola durata stimata
        duration = self._estimate_duration(segments)
        
        # Crea output
        output = SpeechOutput(
            segments=segments,
            total_duration_estimate=duration,
            emotion=emotion,
            profile=self.profile,
            original_text=text,
            word_count=len(text.split()),
            sentence_count=len(sentences)
        )
        
        voice_logger.debug(f"Processed text: {output.word_count} words, {output.sentence_count} sentences, ~{duration:.1f}s")
        
        return output
    
    def _detect_emotion(self, text: str) -> VoiceEmotion:
        """Rileva emozione dal testo"""
        text_lower = text.lower()
        
        # Urgenza
        if any(w in text_lower for w in ["urgente", "immediato", "subito", "urgent", "immediately"]):
            return VoiceEmotion.URGENT
        
        # Preoccupazione
        if any(w in text_lower for w in ["attenzione", "problema", "errore", "warning", "error"]):
            return VoiceEmotion.CONCERNED
        
        # Entusiasmo
        if '!' in text and any(w in text_lower for w in ["ottimo", "perfetto", "grande", "great", "perfect"]):
            return VoiceEmotion.EXCITED
        
        # Domanda = curiosità
        if text.strip().endswith('?'):
            return VoiceEmotion.CURIOUS
        
        # Formalità
        if any(w in text_lower for w in ["importante", "nota", "ricorda", "important", "note"]):
            return VoiceEmotion.SERIOUS
        
        # Default amichevole
        return VoiceEmotion.FRIENDLY
    
    def _split_sentences(self, text: str) -> List[str]:
        """Suddivide testo in frasi"""
        # Normalizza newline
        text = text.replace('\r\n', '\n')
        
        # Split su punteggiatura finale
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Rimuovi vuoti
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _process_sentence(self, sentence: str, index: int, total: int, 
                          emotion: VoiceEmotion) -> List[SpeechSegment]:
        """Processa una singola frase"""
        segments = []
        
        # Determina intonazione
        intonation = self._get_intonation(sentence)
        
        # Ottieni modificatori emozione
        emotion_mods = self.profile.emotion_settings.get(
            emotion.value, 
            {"rate": 1.0, "pitch": 0, "volume": 0.9}
        )
        
        # Pausa iniziale (più lunga se non prima frase)
        pause_before = PauseType.PARAGRAPH.value if index > 0 else 0
        
        # Suddividi in parti (per virgole, etc.)
        parts = self._split_on_punctuation(sentence)
        
        for i, (part_text, punct) in enumerate(parts):
            if not part_text.strip():
                continue
            
            # Controlla enfasi
            emphasis = self._should_emphasize(part_text)
            
            # Rate variabile per importanza
            rate = emotion_mods["rate"]
            if emphasis:
                rate *= 0.9  # Più lento per enfasi
            
            # Pitch
            pitch = emotion_mods["pitch"]
            if intonation == IntonationPattern.QUESTION:
                pitch += 2  # Alza per domande
            elif intonation == IntonationPattern.EXCLAMATION:
                pitch += 3
            
            # Crea segmento
            segment = SpeechSegment(
                text=part_text.strip(),
                pause_before=pause_before if i == 0 else 0,
                pause_after=self.punct_pause.get(punct, PauseType.NONE).value if isinstance(
                    self.punct_pause.get(punct, PauseType.NONE), PauseType
                ) else self.punct_pause.get(punct, 0),
                rate_multiplier=rate,
                pitch_shift=pitch,
                volume=emotion_mods["volume"],
                emphasis=emphasis,
                intonation=intonation
            )
            
            segments.append(segment)
            pause_before = 0  # Reset per parti successive
        
        return segments
    
    def _split_on_punctuation(self, text: str) -> List[Tuple[str, str]]:
        """Suddivide testo su punteggiatura, restituendo (testo, punteggiatura)"""
        parts = []
        current = ""
        
        for char in text:
            if char in ',.;:!?-—':
                if current:
                    parts.append((current, char))
                    current = ""
            else:
                current += char
        
        if current:
            parts.append((current, ""))
        
        return parts
    
    def _get_intonation(self, sentence: str) -> IntonationPattern:
        """Determina pattern intonazione"""
        sentence = sentence.strip()
        
        if sentence.endswith('?'):
            return IntonationPattern.QUESTION
        elif sentence.endswith('!'):
            return IntonationPattern.EXCLAMATION
        elif self.list_pattern.match(sentence):
            return IntonationPattern.LIST
        elif self.question_starters.match(sentence):
            return IntonationPattern.QUESTION
        else:
            return IntonationPattern.STATEMENT
    
    def _should_emphasize(self, text: str) -> bool:
        """Determina se il testo deve essere enfatizzato"""
        text_lower = text.lower()
        
        # Parole chiave
        for word in self.emphasis_words:
            if word in text_lower:
                return True
        
        # Pattern enfasi (grassetto, MAIUSCOLO)
        if self.emphasis_pattern.search(text):
            return True
        
        # Numeri importanti (percentuali, grandi numeri)
        numbers = self.number_pattern.findall(text)
        if numbers:
            return True
        
        return False
    
    def _estimate_duration(self, segments: List[SpeechSegment]) -> float:
        """Stima durata totale in secondi"""
        total_ms = 0
        words_per_minute = self.profile.base_rate
        
        for seg in segments:
            # Pausa
            total_ms += seg.pause_before + seg.pause_after
            
            # Durata parlato
            word_count = len(seg.text.split())
            words_per_second = (words_per_minute * seg.rate_multiplier) / 60
            segment_duration = (word_count / words_per_second) * 1000
            total_ms += segment_duration
        
        return total_ms / 1000
    
    def add_thinking_pause(self, output: SpeechOutput) -> SpeechOutput:
        """Aggiunge pausa di pensiero all'inizio"""
        if output.segments:
            output.segments[0].pause_before = PauseType.MEDIUM.value
        return output
    
    def add_dramatic_pause(self, output: SpeechOutput, after_segment: int = 0) -> SpeechOutput:
        """Aggiunge pausa drammatica dopo un segmento"""
        if after_segment < len(output.segments):
            output.segments[after_segment].pause_after = PauseType.DRAMATIC.value
        return output


# === RESPONSE COMPOSER ===

@dataclass
class ReasoningContext:
    """Contesto del ragionamento completato"""
    query: str
    reasoning_steps: List[str] = field(default_factory=list)
    confidence: float = 0.0
    data_sources: List[str] = field(default_factory=list)
    decision: Optional[str] = None
    action_taken: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComposedResponse:
    """Risposta composta finale"""
    text: str                       # Testo risposta
    speech: SpeechOutput           # Output vocale
    emotion: VoiceEmotion
    confidence: float
    
    # Metadata
    reasoning_summary: str = ""
    is_complete: bool = True
    follow_up_suggested: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "speech": self.speech.to_dict(),
            "emotion": self.emotion.value,
            "confidence": round(self.confidence, 2),
            "reasoning_summary": self.reasoning_summary,
            "is_complete": self.is_complete,
            "follow_up": self.follow_up_suggested
        }


class ResponseComposer:
    """
    Compone risposte finali dopo ragionamento completo.
    
    Principi:
    - UNA sola risposta finale (no stream parziali)
    - Sintesi del ragionamento, non dump completo
    - Adattamento tono/stile al contesto
    - Risposta naturale, non robotica
    """
    
    def __init__(self, voice_engine: NaturalVoiceEngine = None):
        self.voice = voice_engine or NaturalVoiceEngine()
        
        # Template risposte
        self.templates = self._init_templates()
        
        # Frasi connettivo
        self.connectors = {
            "intro": [
                "Ecco cosa ho trovato.",
                "Ho analizzato la situazione.",
                "Dopo aver esaminato i dati,",
                "In base alla mia analisi,"
            ],
            "conclusion": [
                "In sintesi,",
                "Per concludere,",
                "Riassumendo,",
                "In definitiva,"
            ],
            "suggestion": [
                "Ti suggerisco di",
                "Potresti considerare",
                "La mia raccomandazione è",
                "Consiglierei di"
            ],
            "uncertainty": [
                "Non sono completamente sicuro, ma",
                "Con una certa riserva,",
                "Potrebbe essere che",
                "È possibile che"
            ]
        }
        
        voice_logger.info("ResponseComposer initialized")
    
    def _init_templates(self) -> Dict[str, str]:
        """Inizializza template risposte"""
        return {
            "simple_answer": "{answer}",
            "with_reasoning": "{intro} {reasoning}. {answer}",
            "with_action": "{intro} {answer}. {action_result}",
            "uncertain": "{uncertainty} {answer}. {suggestion}",
            "error": "Mi dispiace, {error_message}. {suggestion}",
            "confirmation": "{confirmation}. {details}",
            "list": "{intro}\n{list_items}",
            "comparison": "{intro} {comparison}. {conclusion}"
        }
    
    def compose(self, context: ReasoningContext, 
                response_text: str,
                include_reasoning: bool = False) -> ComposedResponse:
        """
        Compone risposta finale dal contesto di ragionamento.
        
        Args:
            context: Contesto con ragionamento completato
            response_text: Testo risposta principale
            include_reasoning: Se includere sintesi ragionamento
        
        Returns:
            ComposedResponse pronta per output
        """
        # Determina emozione
        emotion = self._determine_emotion(context, response_text)
        
        # Costruisci testo finale
        final_text = self._build_final_text(
            context, response_text, include_reasoning
        )
        
        # Processa per voce
        speech = self.voice.process_text(final_text, emotion)
        
        # Aggiungi pausa pensiero se ragionamento complesso
        if len(context.reasoning_steps) > 3:
            speech = self.voice.add_thinking_pause(speech)
        
        # Sintesi ragionamento
        reasoning_summary = self._summarize_reasoning(context)
        
        # Suggerimento follow-up
        follow_up = self._suggest_follow_up(context, response_text)
        
        return ComposedResponse(
            text=final_text,
            speech=speech,
            emotion=emotion,
            confidence=context.confidence,
            reasoning_summary=reasoning_summary,
            is_complete=True,
            follow_up_suggested=follow_up
        )
    
    def compose_simple(self, text: str, emotion: VoiceEmotion = None) -> ComposedResponse:
        """Compone risposta semplice senza contesto ragionamento"""
        if emotion is None:
            emotion = VoiceEmotion.FRIENDLY
        
        speech = self.voice.process_text(text, emotion)
        
        return ComposedResponse(
            text=text,
            speech=speech,
            emotion=emotion,
            confidence=1.0,
            is_complete=True
        )
    
    def _determine_emotion(self, context: ReasoningContext, 
                           response: str) -> VoiceEmotion:
        """Determina emozione appropriata"""
        # Alta confidenza = confident
        if context.confidence > 0.9:
            return VoiceEmotion.CONFIDENT
        
        # Bassa confidenza = concerned
        if context.confidence < 0.5:
            return VoiceEmotion.CONCERNED
        
        # Azione eseguita = excited/friendly
        if context.action_taken:
            return VoiceEmotion.EXCITED
        
        # Molti step = serious (ragionamento complesso)
        if len(context.reasoning_steps) > 5:
            return VoiceEmotion.SERIOUS
        
        # Default
        return VoiceEmotion.FRIENDLY
    
    def _build_final_text(self, context: ReasoningContext,
                          response: str, include_reasoning: bool) -> str:
        """Costruisce testo finale"""
        parts = []
        
        # Intro opzionale per risposte complesse
        if len(context.reasoning_steps) > 2 and include_reasoning:
            import random
            intro = random.choice(self.connectors["intro"])
            parts.append(intro)
        
        # Sintesi ragionamento se richiesto
        if include_reasoning and context.reasoning_steps:
            summary = self._summarize_reasoning(context)
            if summary:
                parts.append(summary)
        
        # Risposta principale
        parts.append(response)
        
        # Risultato azione se presente
        if context.action_taken:
            parts.append(f"Ho {context.action_taken}.")
        
        # Unisci
        text = " ".join(parts)
        
        # Pulisci
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _summarize_reasoning(self, context: ReasoningContext) -> str:
        """Riassume il ragionamento in modo conciso"""
        if not context.reasoning_steps:
            return ""
        
        # Per pochi step, non serve sintesi
        if len(context.reasoning_steps) <= 2:
            return ""
        
        # Sintesi breve
        step_count = len(context.reasoning_steps)
        
        if step_count <= 5:
            return f"Ho considerato {step_count} aspetti della tua richiesta."
        else:
            return f"Ho analizzato la situazione attraverso {step_count} passaggi di ragionamento."
    
    def _suggest_follow_up(self, context: ReasoningContext, 
                           response: str) -> Optional[str]:
        """Suggerisce domanda di follow-up"""
        # Se bassa confidenza
        if context.confidence < 0.7:
            return "Vuoi che approfondisca qualche aspetto?"
        
        # Se molte fonti
        if len(context.data_sources) > 3:
            return "Posso mostrarti i dettagli delle mie fonti?"
        
        # Se azione
        if context.action_taken:
            return "Vuoi che faccia qualcos'altro?"
        
        return None


# === CONVERSATION MANAGER ===

@dataclass
class ConversationTurn:
    """Singolo turno di conversazione"""
    role: str  # "user" o "assistant"
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    emotion: Optional[VoiceEmotion] = None
    confidence: float = 1.0
    was_spoken: bool = False


class ConversationManager:
    """
    Gestisce il flusso conversazionale.
    
    Caratteristiche:
    - Mantiene contesto conversazione
    - Gestisce turni user/assistant
    - Garantisce risposta unica finale
    - Tracking stato conversazione
    """
    
    def __init__(self, composer: ResponseComposer = None):
        self.composer = composer or ResponseComposer()
        self.history: List[ConversationTurn] = []
        self.max_history = 20
        
        # Stato
        self.is_processing = False
        self.current_reasoning: Optional[ReasoningContext] = None
        
        # Callbacks
        self.on_response_ready: Optional[Callable[[ComposedResponse], None]] = None
        self.on_thinking_start: Optional[Callable[[], None]] = None
        self.on_thinking_end: Optional[Callable[[], None]] = None
        
        voice_logger.info("ConversationManager initialized")
    
    def add_user_input(self, text: str) -> str:
        """Aggiunge input utente"""
        turn = ConversationTurn(role="user", text=text)
        self._add_turn(turn)
        return turn.text
    
    def start_reasoning(self, query: str) -> ReasoningContext:
        """Inizia fase di ragionamento"""
        self.is_processing = True
        self.current_reasoning = ReasoningContext(query=query)
        
        if self.on_thinking_start:
            self.on_thinking_start()
        
        return self.current_reasoning
    
    def add_reasoning_step(self, step: str):
        """Aggiunge step al ragionamento corrente"""
        if self.current_reasoning:
            self.current_reasoning.reasoning_steps.append(step)
    
    def complete_reasoning(self, response_text: str, 
                           confidence: float = 0.8,
                           action_taken: str = None,
                           include_reasoning_summary: bool = False) -> ComposedResponse:
        """
        Completa ragionamento e genera risposta finale unica.
        
        Args:
            response_text: Testo risposta
            confidence: Confidenza (0-1)
            action_taken: Descrizione azione eseguita
            include_reasoning_summary: Se includere sintesi ragionamento
        
        Returns:
            Risposta composta finale
        """
        if not self.current_reasoning:
            self.current_reasoning = ReasoningContext(query="")
        
        self.current_reasoning.confidence = confidence
        self.current_reasoning.action_taken = action_taken
        
        # Componi risposta
        response = self.composer.compose(
            self.current_reasoning,
            response_text,
            include_reasoning_summary
        )
        
        # Aggiungi alla storia
        turn = ConversationTurn(
            role="assistant",
            text=response.text,
            emotion=response.emotion,
            confidence=confidence
        )
        self._add_turn(turn)
        
        # Reset stato
        self.is_processing = False
        self.current_reasoning = None
        
        if self.on_thinking_end:
            self.on_thinking_end()
        
        if self.on_response_ready:
            self.on_response_ready(response)
        
        return response
    
    def quick_response(self, text: str, 
                       emotion: VoiceEmotion = None) -> ComposedResponse:
        """Risposta rapida senza ragionamento complesso"""
        response = self.composer.compose_simple(text, emotion)
        
        turn = ConversationTurn(
            role="assistant",
            text=text,
            emotion=emotion or VoiceEmotion.FRIENDLY
        )
        self._add_turn(turn)
        
        return response
    
    def _add_turn(self, turn: ConversationTurn):
        """Aggiunge turno alla storia"""
        self.history.append(turn)
        
        # Limita storia
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_summary(self) -> str:
        """Ottiene riassunto contesto per AI"""
        if not self.history:
            return "Nuova conversazione."
        
        recent = self.history[-5:]
        summary = []
        for turn in recent:
            prefix = "Utente" if turn.role == "user" else "Gideon"
            summary.append(f"{prefix}: {turn.text[:100]}...")
        
        return "\n".join(summary)
    
    def get_last_user_query(self) -> Optional[str]:
        """Ottiene ultima query utente"""
        for turn in reversed(self.history):
            if turn.role == "user":
                return turn.text
        return None
    
    def clear_history(self):
        """Pulisce storia conversazione"""
        self.history.clear()
        self.current_reasoning = None
        self.is_processing = False
    
    def get_stats(self) -> Dict:
        """Statistiche conversazione"""
        user_turns = sum(1 for t in self.history if t.role == "user")
        assistant_turns = sum(1 for t in self.history if t.role == "assistant")
        
        return {
            "total_turns": len(self.history),
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
            "is_processing": self.is_processing,
            "has_active_reasoning": self.current_reasoning is not None
        }
