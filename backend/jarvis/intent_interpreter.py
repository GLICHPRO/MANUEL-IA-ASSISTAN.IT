"""
🧠 JARVIS CORE - Advanced Intent Interpreter

Traduce linguaggio umano in obiettivi strutturati:
- NLU multi-layer (pattern, semantico, contestuale)
- Entity extraction avanzata con slot filling
- Intent disambiguation con confidence scoring
- Context awareness per conversazioni multi-turno
- Support per comandi composti e impliciti
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field


class IntentCategory(Enum):
    """Categorie di intenti"""
    SYSTEM = "system"           # Controllo sistema (shutdown, restart, etc.)
    FILE = "file"               # Operazioni file
    APP = "app"                 # Gestione applicazioni
    WEB = "web"                 # Navigazione web
    INFO = "info"               # Richiesta informazioni
    CALCULATION = "calculation" # Calcoli
    AUTOMATION = "automation"   # Automazioni
    COMMUNICATION = "communication"  # Email, messaggi
    MEDIA = "media"             # Audio, video
    SETTINGS = "settings"       # Impostazioni
    CONVERSATION = "conversation"    # Chit-chat
    COMPOUND = "compound"       # Comandi multipli
    UNKNOWN = "unknown"         # Non riconosciuto


class SlotType(Enum):
    """Tipi di slot per entity extraction"""
    APP_NAME = "app_name"
    FILE_NAME = "file_name"
    FILE_PATH = "file_path"
    URL = "url"
    QUERY = "query"
    NUMBER = "number"
    TIME = "time"
    DATE = "date"
    DURATION = "duration"
    LOCATION = "location"
    PERSON = "person"
    EMAIL = "email"
    PHONE = "phone"


@dataclass
class Slot:
    """Slot per entity extraction"""
    type: SlotType
    value: Any
    confidence: float
    raw_text: str
    start_pos: int = 0
    end_pos: int = 0


class Intent:
    """Rappresenta un intento rilevato con metadati ricchi"""
    
    def __init__(self, name: str, category: IntentCategory, confidence: float,
                 entities: Dict = None, original_text: str = ""):
        self.name = name
        self.category = category
        self.confidence = confidence
        self.entities = entities or {}
        self.original_text = original_text
        self.timestamp = datetime.now()
        self.alternatives: List['Intent'] = []
        
        # Extended attributes
        self.slots: List[Slot] = []
        self.is_compound = False
        self.sub_intents: List['Intent'] = []
        self.requires_context = False
        self.sentiment: Optional[str] = None  # positive, negative, neutral
        self.urgency_markers: List[str] = []
        self.negation = False
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "confidence": self.confidence,
            "entities": self.entities,
            "original_text": self.original_text,
            "timestamp": self.timestamp.isoformat(),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "slots": [{"type": s.type.value, "value": s.value, "confidence": s.confidence} 
                     for s in self.slots],
            "is_compound": self.is_compound,
            "sub_intents": [si.to_dict() for si in self.sub_intents],
            "sentiment": self.sentiment,
            "negation": self.negation,
            "urgency_markers": self.urgency_markers
        }
    
    def has_slot(self, slot_type: SlotType) -> bool:
        """Verifica se ha uno slot di un certo tipo"""
        return any(s.type == slot_type for s in self.slots)
    
    def get_slot(self, slot_type: SlotType) -> Optional[Slot]:
        """Ottiene slot per tipo"""
        for s in self.slots:
            if s.type == slot_type:
                return s
        return None


class IntentInterpreter:
    """
    🧠 Advanced Intent Interpreter
    
    Pipeline NLU:
    1. Preprocessing (normalizzazione, tokenizzazione)
    2. Pattern matching (regole base)
    3. Semantic analysis (significato profondo)
    4. Entity extraction (slot filling)
    5. Context integration (storico conversazione)
    6. Confidence scoring multi-fattore
    7. Disambiguation (gestione ambiguità)
    """
    
    def __init__(self):
        # Context per conversazioni multi-turno
        self._context_history: List[Intent] = []
        self._max_context = 10
        self._current_topic: Optional[str] = None
        
        # Pattern per riconoscimento intenti (estesi)
        self._intent_patterns = self._build_intent_patterns()
        
        # Entity extractors avanzati
        self._entity_extractors = self._build_entity_extractors()
        
        # App name mappings
        self._app_mappings = self._build_app_mappings()
        
        # Urgency markers
        self._urgency_markers = {
            "high": ["subito", "immediatamente", "ora", "adesso", "urgente", 
                     "veloce", "presto", "quick", "now", "asap", "urgent"],
            "medium": ["quando puoi", "appena possibile"],
            "low": ["con calma", "quando hai tempo", "senza fretta"]
        }
        
        # Negation patterns
        self._negation_patterns = [
            r"\b(non|no|niente|mai|senza)\b",
            r"\b(don't|doesn't|not|never|without)\b"
        ]
        
        # Compound command connectors
        self._compound_connectors = [
            r"\b(poi|e poi|quindi|dopo|e)\b",
            r"\b(then|and then|after that|and)\b",
            r"[,;]"
        ]
        
        # Sentiment indicators
        self._sentiment_markers = {
            "positive": ["grazie", "perfetto", "ottimo", "bene", "bravo", "thanks"],
            "negative": ["problema", "errore", "male", "sbagliato", "non funziona"],
            "neutral": []
        }
    
    def _build_intent_patterns(self) -> Dict:
        """Costruisce pattern intenti estesi"""
        return {
            # SYSTEM
            "shutdown": {
                "patterns": [r"spegni(\s+il)?(\s+computer)?", r"shutdown", r"arresta(\s+il\s+sistema)?", 
                           r"chiudi\s+tutto", r"power\s+off"],
                "category": IntentCategory.SYSTEM,
                "entities": [],
                "priority": "high",
                "requires_confirmation": True
            },
            "restart": {
                "patterns": [r"riavvia(\s+il)?(\s+pc)?", r"restart", r"riparti", r"reboot"],
                "category": IntentCategory.SYSTEM,
                "entities": [],
                "priority": "high",
                "requires_confirmation": True
            },
            "lock": {
                "patterns": [r"blocca(\s+lo)?(\s+schermo)?", r"lock(\s+screen)?", r"blocca\s+pc"],
                "category": IntentCategory.SYSTEM,
                "entities": [],
                "priority": "normal"
            },
            "sleep": {
                "patterns": [r"sospendi", r"sleep", r"vai\s+in\s+sospensione", r"standby"],
                "category": IntentCategory.SYSTEM,
                "entities": [],
                "priority": "normal"
            },
            "screenshot": {
                "patterns": [r"screenshot", r"cattura\s+schermo", r"foto\s+schermo", r"screen\s*cap"],
                "category": IntentCategory.SYSTEM,
                "entities": [],
                "priority": "normal"
            },
            
            # FILE
            "open_file": {
                "patterns": [r"apri(\s+il)?(\s+file)?\s+(.+\.[\w]+)", r"open(\s+file)?\s+(.+\.[\w]+)"],
                "category": IntentCategory.FILE,
                "entities": ["file_name"],
                "priority": "normal"
            },
            "create_file": {
                "patterns": [r"crea(\s+un)?(\s+nuovo)?\s+file", r"new\s+file", r"nuovo\s+documento"],
                "category": IntentCategory.FILE,
                "entities": ["file_name"],
                "priority": "normal"
            },
            "delete_file": {
                "patterns": [r"elimina(\s+il)?\s+file", r"cancella(\s+il)?\s+file", r"delete\s+file"],
                "category": IntentCategory.FILE,
                "entities": ["file_name"],
                "priority": "high",
                "requires_confirmation": True
            },
            "find_file": {
                "patterns": [r"trova(\s+il)?\s+file", r"cerca(\s+il)?\s+file", r"dov['\s]?[eè]\s+(.+)"],
                "category": IntentCategory.FILE,
                "entities": ["file_name"],
                "priority": "normal"
            },
            "copy_file": {
                "patterns": [r"copia(\s+il)?\s+file", r"copy\s+file"],
                "category": IntentCategory.FILE,
                "entities": ["file_name", "destination"],
                "priority": "normal"
            },
            "move_file": {
                "patterns": [r"sposta(\s+il)?\s+file", r"move\s+file"],
                "category": IntentCategory.FILE,
                "entities": ["file_name", "destination"],
                "priority": "normal"
            },
            
            # APP
            "open_app": {
                "patterns": [r"apri\s+(.+)", r"avvia\s+(.+)", r"lancia\s+(.+)", 
                           r"esegui\s+(.+)", r"open\s+(.+)", r"start\s+(.+)", r"run\s+(.+)"],
                "category": IntentCategory.APP,
                "entities": ["app_name"],
                "priority": "high"
            },
            "close_app": {
                "patterns": [r"chiudi\s+(.+)", r"termina\s+(.+)", r"esci\s+da\s+(.+)", 
                           r"close\s+(.+)", r"quit\s+(.+)", r"exit\s+(.+)"],
                "category": IntentCategory.APP,
                "entities": ["app_name"],
                "priority": "high"
            },
            "switch_app": {
                "patterns": [r"vai\s+a\s+(.+)", r"passa\s+a\s+(.+)", r"switch\s+to\s+(.+)"],
                "category": IntentCategory.APP,
                "entities": ["app_name"],
                "priority": "normal"
            },
            "minimize_app": {
                "patterns": [r"minimizza(\s+(.+))?", r"minimize(\s+(.+))?"],
                "category": IntentCategory.APP,
                "entities": ["app_name"],
                "priority": "low"
            },
            "maximize_app": {
                "patterns": [r"massimizza(\s+(.+))?", r"maximize(\s+(.+))?", r"fullscreen"],
                "category": IntentCategory.APP,
                "entities": ["app_name"],
                "priority": "low"
            },
            
            # WEB
            "search_web": {
                "patterns": [r"cerca(\s+su)?\s+(google|web|internet)\s+(.+)", r"googla\s+(.+)",
                           r"search(\s+for)?\s+(.+)", r"trova\s+online\s+(.+)"],
                "category": IntentCategory.WEB,
                "entities": ["query"],
                "priority": "normal"
            },
            "open_url": {
                "patterns": [r"apri(\s+il\s+sito)?\s+(.+\.(com|it|org|net|io|dev))", 
                           r"vai\s+(su|a)\s+(.+)", r"go\s+to\s+(.+)"],
                "category": IntentCategory.WEB,
                "entities": ["url"],
                "priority": "normal"
            },
            "open_youtube": {
                "patterns": [r"(apri|vai\s+su)\s+youtube", r"youtube"],
                "category": IntentCategory.WEB,
                "entities": [],
                "priority": "normal",
                "default_url": "https://www.youtube.com"
            },
            "search_youtube": {
                "patterns": [r"cerca\s+su\s+youtube\s+(.+)", r"youtube\s+(.+)"],
                "category": IntentCategory.WEB,
                "entities": ["query"],
                "priority": "normal"
            },
            
            # INFO
            "weather": {
                "patterns": [r"(che\s+)?tempo\s+fa(\s+a\s+(.+))?", r"meteo(\s+(.+))?", 
                           r"previsioni(\s+(.+))?", r"weather(\s+in\s+(.+))?"],
                "category": IntentCategory.INFO,
                "entities": ["location"],
                "priority": "normal"
            },
            "time": {
                "patterns": [r"che\s+ora\s+[eè]", r"ora\s+attuale", r"che\s+ore\s+sono",
                           r"what\s+time", r"current\s+time"],
                "category": IntentCategory.INFO,
                "entities": [],
                "priority": "low"
            },
            "date": {
                "patterns": [r"che\s+giorno\s+[eè]", r"data\s+di\s+oggi", r"oggi\s+[eè]",
                           r"what\s+day", r"today's\s+date"],
                "category": IntentCategory.INFO,
                "entities": [],
                "priority": "low"
            },
            "wikipedia": {
                "patterns": [r"cerca\s+su\s+wikipedia\s+(.+)", r"cos['\s]?[eè]\s+(.+)", 
                           r"chi\s+[eè]\s+(.+)", r"what\s+is\s+(.+)", r"who\s+is\s+(.+)"],
                "category": IntentCategory.INFO,
                "entities": ["topic"],
                "priority": "normal"
            },
            "define": {
                "patterns": [r"definisci\s+(.+)", r"definizione\s+di\s+(.+)", 
                           r"define\s+(.+)", r"meaning\s+of\s+(.+)"],
                "category": IntentCategory.INFO,
                "entities": ["topic"],
                "priority": "normal"
            },
            "translate": {
                "patterns": [r"traduci\s+(.+)", r"traduzione\s+di\s+(.+)",
                           r"translate\s+(.+)", r"come\s+si\s+dice\s+(.+)"],
                "category": IntentCategory.INFO,
                "entities": ["text", "target_language"],
                "priority": "normal"
            },
            
            # CALCULATION
            "calculate": {
                "patterns": [r"calcola\s+(.+)", r"quanto\s+(fa|è)\s+(.+)", 
                           r"(\d+)\s*[\+\-\*\/x]\s*(\d+)", r"calculate\s+(.+)"],
                "category": IntentCategory.CALCULATION,
                "entities": ["expression"],
                "priority": "normal"
            },
            "convert": {
                "patterns": [r"converti\s+(.+)\s+(in|a)\s+(.+)", r"convert\s+(.+)\s+to\s+(.+)",
                           r"quanti\s+(.+)\s+sono\s+(.+)"],
                "category": IntentCategory.CALCULATION,
                "entities": ["value", "from_unit", "to_unit"],
                "priority": "normal"
            },
            
            # AUTOMATION
            "create_routine": {
                "patterns": [r"crea(\s+una)?\s+routine", r"nuova\s+automazione",
                           r"create\s+routine", r"new\s+automation"],
                "category": IntentCategory.AUTOMATION,
                "entities": ["routine_name"],
                "priority": "low"
            },
            "run_routine": {
                "patterns": [r"esegui(\s+la)?\s+routine\s+(.+)", r"avvia\s+automazione\s+(.+)",
                           r"run\s+routine\s+(.+)"],
                "category": IntentCategory.AUTOMATION,
                "entities": ["routine_name"],
                "priority": "normal"
            },
            "run_workflow": {
                "patterns": [r"esegui(\s+il)?\s+workflow\s+(.+)", r"run\s+workflow\s+(.+)"],
                "category": IntentCategory.AUTOMATION,
                "entities": ["workflow_name"],
                "priority": "normal"
            },
            "play_macro": {
                "patterns": [r"esegui(\s+la)?\s+macro\s+(.+)", r"play\s+macro\s+(.+)"],
                "category": IntentCategory.AUTOMATION,
                "entities": ["macro_name"],
                "priority": "normal"
            },
            
            # MEDIA
            "play_music": {
                "patterns": [r"metti(\s+la)?\s+musica", r"riproduci\s+(.+)", 
                           r"play\s+(.+)", r"ascolta\s+(.+)"],
                "category": IntentCategory.MEDIA,
                "entities": ["track"],
                "priority": "normal"
            },
            "pause_music": {
                "patterns": [r"pausa", r"metti\s+in\s+pausa", r"pause", r"stop\s+music"],
                "category": IntentCategory.MEDIA,
                "entities": [],
                "priority": "normal"
            },
            "next_track": {
                "patterns": [r"prossima(\s+canzone)?", r"next(\s+track)?", r"skip"],
                "category": IntentCategory.MEDIA,
                "entities": [],
                "priority": "normal"
            },
            "previous_track": {
                "patterns": [r"precedente", r"previous(\s+track)?", r"back"],
                "category": IntentCategory.MEDIA,
                "entities": [],
                "priority": "normal"
            },
            "volume_up": {
                "patterns": [r"alza(\s+il)?\s+volume", r"volume\s+up", r"più\s+forte"],
                "category": IntentCategory.MEDIA,
                "entities": [],
                "priority": "normal"
            },
            "volume_down": {
                "patterns": [r"abbassa(\s+il)?\s+volume", r"volume\s+down", r"più\s+piano"],
                "category": IntentCategory.MEDIA,
                "entities": [],
                "priority": "normal"
            },
            "set_volume": {
                "patterns": [r"volume(\s+al)?\s+(\d+)", r"set\s+volume\s+(\d+)"],
                "category": IntentCategory.MEDIA,
                "entities": ["level"],
                "priority": "normal"
            },
            "mute": {
                "patterns": [r"muto", r"silenzia", r"togli(\s+l['\s])?audio", r"mute"],
                "category": IntentCategory.MEDIA,
                "entities": [],
                "priority": "normal"
            },
            
            # COMMUNICATION
            "send_email": {
                "patterns": [r"invia(\s+una)?\s+email\s+a\s+(.+)", r"scrivi(\s+una)?\s+mail\s+a\s+(.+)",
                           r"send\s+email\s+to\s+(.+)"],
                "category": IntentCategory.COMMUNICATION,
                "entities": ["recipient", "subject", "body"],
                "priority": "normal"
            },
            "read_email": {
                "patterns": [r"leggi(\s+le)?\s+email", r"controlla(\s+la)?\s+posta",
                           r"check\s+email", r"read\s+mail"],
                "category": IntentCategory.COMMUNICATION,
                "entities": [],
                "priority": "normal"
            },
            
            # SETTINGS
            "change_mode": {
                "patterns": [r"modalit[àa]\s+(pilot|copilot|passive|executive)", 
                           r"cambia\s+modalit[àa](\s+in\s+(.+))?",
                           r"mode\s+(pilot|copilot|passive|executive)"],
                "category": IntentCategory.SETTINGS,
                "entities": ["mode"],
                "priority": "high"
            },
            "set_reminder": {
                "patterns": [r"ricordami(\s+di)?\s+(.+)", r"imposta(\s+un)?\s+promemoria",
                           r"remind\s+me\s+to\s+(.+)", r"set\s+reminder"],
                "category": IntentCategory.SETTINGS,
                "entities": ["reminder_text", "time"],
                "priority": "normal"
            },
            "set_timer": {
                "patterns": [r"timer(\s+di)?\s+(\d+)\s+(minuti|secondi|ore)",
                           r"set\s+timer\s+(\d+)\s+(minutes|seconds|hours)"],
                "category": IntentCategory.SETTINGS,
                "entities": ["duration"],
                "priority": "normal"
            },
            "set_alarm": {
                "patterns": [r"sveglia(\s+alle)?\s+(\d{1,2}[:.]\d{2})",
                           r"alarm(\s+at)?\s+(\d{1,2}[:.]\d{2})"],
                "category": IntentCategory.SETTINGS,
                "entities": ["time"],
                "priority": "normal"
            },
            
            # CONVERSATION
            "greeting": {
                "patterns": [r"^ciao\b", r"^buongiorno\b", r"^buonasera\b", r"^salve\b", 
                           r"^hey\b", r"^hi\b", r"^hello\b"],
                "category": IntentCategory.CONVERSATION,
                "entities": [],
                "priority": "low"
            },
            "farewell": {
                "patterns": [r"^(ciao|arrivederci|a\s+dopo|bye|goodbye)\b"],
                "category": IntentCategory.CONVERSATION,
                "entities": [],
                "priority": "low"
            },
            "thanks": {
                "patterns": [r"grazie", r"ti\s+ringrazio", r"thanks", r"thank\s+you"],
                "category": IntentCategory.CONVERSATION,
                "entities": [],
                "priority": "low"
            },
            "help": {
                "patterns": [r"aiuto", r"help", r"cosa\s+puoi\s+fare", r"come\s+funzioni",
                           r"what\s+can\s+you\s+do"],
                "category": IntentCategory.CONVERSATION,
                "entities": [],
                "priority": "normal"
            },
            "affirmative": {
                "patterns": [r"^(s[ìi]|ok|va\s+bene|certo|confermo|yes|yeah|sure)\b"],
                "category": IntentCategory.CONVERSATION,
                "entities": [],
                "priority": "high"  # High per conferme rapide
            },
            "negative": {
                "patterns": [r"^(no|niente|annulla|cancel|stop)\b"],
                "category": IntentCategory.CONVERSATION,
                "entities": [],
                "priority": "high"  # High per interruzioni rapide
            },
            "repeat": {
                "patterns": [r"ripeti", r"non\s+ho\s+capito", r"come\?", r"repeat", r"what\?"],
                "category": IntentCategory.CONVERSATION,
                "entities": [],
                "priority": "normal"
            }
        }
    
    def _build_entity_extractors(self) -> Dict:
        """Costruisce extractors per entità avanzati"""
        return {
            "file_name": [
                r"(?:file\s+)?([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+)",
                r"(?:chiamato|named)\s+([a-zA-Z0-9_\-\.]+)"
            ],
            "app_name": [
                r"(?:apri|avvia|chiudi|lancia|open|close|start)\s+(?:l['\s])?(.+?)(?:\s+e\s+|\s*$)",
            ],
            "url": [
                r"((?:https?://)?[\w\-\.]+\.[a-z]{2,}(?:/\S*)?)",
                r"(?:sito|site|website)\s+(.+)"
            ],
            "query": [
                r"(?:cerca|search|find|googla)\s+(?:su\s+\w+\s+)?(.+)",
            ],
            "location": [
                r"(?:a|di|in|at|near)\s+([A-Za-z\s]+)",
            ],
            "topic": [
                r"(?:cos['\s]?[eè]|chi\s+[eè]|what\s+is|who\s+is)\s+(?:un[oa]?\s+)?(.+)",
            ],
            "expression": [
                r"([\d\s\+\-\*\/x\^\(\)\.]+)",
                r"(\d+\s*[\+\-\*\/x]\s*\d+)"
            ],
            "level": [
                r"(?:volume\s+)?(?:al\s+)?(\d+)(?:\s*%)?",
            ],
            "mode": [
                r"\b(pilot|copilot|passive|executive)\b",
            ],
            "routine_name": [
                r"routine\s+(?:chiamata\s+)?(.+)",
                r"automazione\s+(.+)"
            ],
            "time": [
                r"(?:alle?\s+)?(\d{1,2}[:.]\d{2})",
                r"(?:tra\s+)?(\d+)\s+(minuti?|ore?|secondi?)",
                r"(?:in\s+)?(\d+)\s+(minutes?|hours?|seconds?)"
            ],
            "duration": [
                r"(?:per\s+)?(\d+)\s+(minuti?|ore?|secondi?|minutes?|hours?|seconds?)",
            ],
            "email": [
                r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            ],
            "number": [
                r"(\d+(?:\.\d+)?)",
            ],
            "person": [
                r"(?:a|di|per|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            ]
        }
    
    def _build_app_mappings(self) -> Dict:
        """Costruisce mappings nome app → eseguibile"""
        return {
            # Browser
            "chrome": "chrome",
            "google chrome": "chrome",
            "browser": "chrome",
            "firefox": "firefox",
            "edge": "msedge",
            "microsoft edge": "msedge",
            
            # Office
            "word": "winword",
            "microsoft word": "winword",
            "excel": "excel",
            "microsoft excel": "excel",
            "powerpoint": "powerpnt",
            "outlook": "outlook",
            "teams": "teams",
            "microsoft teams": "teams",
            
            # Utility
            "notepad": "notepad",
            "blocco note": "notepad",
            "calcolatrice": "calc",
            "calculator": "calc",
            "calc": "calc",
            "esplora file": "explorer",
            "file explorer": "explorer",
            "explorer": "explorer",
            
            # Development
            "terminale": "wt",
            "terminal": "wt",
            "cmd": "cmd",
            "prompt": "cmd",
            "command prompt": "cmd",
            "powershell": "powershell",
            "vscode": "code",
            "visual studio code": "code",
            "vs code": "code",
            "code": "code",
            
            # Media & Social
            "spotify": "spotify",
            "discord": "discord",
            "slack": "slack",
            "zoom": "zoom",
            "skype": "skype",
            
            # System
            "impostazioni": "ms-settings:",
            "settings": "ms-settings:",
            "pannello di controllo": "control",
            "control panel": "control",
            "task manager": "taskmgr",
            "gestione attività": "taskmgr"
        }

    def interpret(self, text: str, context: dict = None) -> Intent:
        """
        Interpreta l'intento principale dal testo
        
        Pipeline:
        1. Preprocessing
        2. Detect compound commands
        3. Pattern matching
        4. Entity extraction
        5. Context integration
        6. Confidence scoring
        7. Alternatives generation
        
        Args:
            text: Input testuale dell'utente
            context: Contesto opzionale (storico, modalità, etc.)
            
        Returns:
            Intent con confidenza e entità estratte
        """
        context = context or {}
        
        # Preprocessing
        text_clean = self._preprocess(text)
        text_original = text
        
        # Check for compound commands
        if self._is_compound_command(text_clean):
            return self._parse_compound_command(text_clean, context)
        
        # Detect negation
        has_negation = self._detect_negation(text_clean)
        
        # Detect urgency
        urgency_markers = self._detect_urgency(text_clean)
        
        # Detect sentiment
        sentiment = self._detect_sentiment(text_clean)
        
        # Trova intento con confidenza più alta
        best_intent = None
        best_confidence = 0.0
        alternatives = []
        
        for intent_name, config in self._intent_patterns.items():
            for pattern in config["patterns"]:
                match = re.search(pattern, text_clean, re.IGNORECASE)
                if match:
                    # Calcola confidenza multi-fattore
                    confidence = self._calculate_confidence(
                        match, text_clean, pattern, intent_name, context
                    )
                    
                    # Estrai entità
                    entities = self._extract_entities(text_clean, config["entities"], match)
                    
                    intent = Intent(
                        name=intent_name,
                        category=config["category"],
                        confidence=confidence,
                        entities=entities,
                        original_text=text_original
                    )
                    
                    # Aggiungi metadata
                    intent.negation = has_negation
                    intent.urgency_markers = urgency_markers
                    intent.sentiment = sentiment
                    
                    # Estrai slots
                    intent.slots = self._extract_slots(text_clean, config["entities"])
                    
                    if confidence > best_confidence:
                        if best_intent:
                            alternatives.append(best_intent)
                        best_intent = intent
                        best_confidence = confidence
                    elif confidence > 0.3:
                        alternatives.append(intent)
        
        # Se nessun intento trovato, usa context o unknown
        if not best_intent:
            best_intent = self._handle_unknown(text_original, context)
        
        # Aggiungi alternative (top 3)
        best_intent.alternatives = sorted(
            alternatives, 
            key=lambda x: x.confidence, 
            reverse=True
        )[:3]
        
        # Update context history
        self._update_context(best_intent)
        
        return best_intent
    
    def _preprocess(self, text: str) -> str:
        """Preprocessa il testo con normalizzazione avanzata"""
        # Lowercase
        text = text.lower().strip()
        
        # Rimuovi punteggiatura finale
        text = re.sub(r'[.!?]+$', '', text)
        
        # Normalizza spazi
        text = re.sub(r'\s+', ' ', text)
        
        # Espandi contrazioni comuni
        contractions = {
            "cos'è": "cosa è",
            "dov'è": "dove è", 
            "com'è": "come è",
            "l'": "la ",
            "un'": "una ",
            "dell'": "della ",
            "nell'": "nella ",
            "all'": "alla "
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _calculate_confidence(self, match, text: str, pattern: str, 
                            intent_name: str, context: Dict) -> float:
        """Calcola confidenza multi-fattore"""
        # Base confidence
        confidence = 0.5
        
        # Bonus per match ratio
        match_ratio = len(match.group()) / len(text)
        confidence += match_ratio * 0.25
        
        # Bonus per pattern specificity
        pattern_specificity = min(len(pattern) / 50, 0.15)
        confidence += pattern_specificity
        
        # Bonus per context match (stesso topic)
        if self._current_topic and self._current_topic == intent_name:
            confidence += 0.1
        
        # Bonus se match dall'inizio
        if match.start() == 0:
            confidence += 0.05
        
        # Bonus per storia recente
        if self._context_history:
            recent_intents = [i.name for i in self._context_history[-3:]]
            if intent_name in recent_intents:
                confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _extract_entities(self, text: str, entity_types: List[str], 
                         match: re.Match = None) -> Dict:
        """Estrae entità dal testo con slot filling"""
        entities = {}
        
        for entity_type in entity_types:
            if entity_type in self._entity_extractors:
                patterns = self._entity_extractors[entity_type]
                
                for pattern in patterns:
                    entity_match = re.search(pattern, text, re.IGNORECASE)
                    if entity_match:
                        value = entity_match.group(1) if entity_match.groups() else entity_match.group()
                        
                        # Normalizzazioni specifiche
                        if entity_type == "app_name":
                            value = self._normalize_app_name(value)
                        elif entity_type == "url":
                            value = self._normalize_url(value)
                        elif entity_type == "time":
                            value = self._normalize_time(value)
                        
                        entities[entity_type] = value.strip()
                        break
        
        return entities
    
    def _extract_slots(self, text: str, entity_types: List[str]) -> List[Slot]:
        """Estrae slots tipizzati"""
        slots = []
        
        for entity_type in entity_types:
            slot_type = SlotType[entity_type.upper()] if entity_type.upper() in SlotType.__members__ else None
            
            if slot_type and entity_type in self._entity_extractors:
                for pattern in self._entity_extractors[entity_type]:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        value = match.group(1) if match.groups() else match.group()
                        slots.append(Slot(
                            type=slot_type,
                            value=value.strip(),
                            confidence=0.8,
                            raw_text=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end()
                        ))
                        break
        
        return slots
    
    def _normalize_app_name(self, app_name: str) -> str:
        """Normalizza nome applicazione"""
        app_name_lower = app_name.lower().strip()
        return self._app_mappings.get(app_name_lower, app_name)
    
    def _normalize_url(self, url: str) -> str:
        """Normalizza URL"""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url
    
    def _normalize_time(self, time_str: str) -> str:
        """Normalizza formato tempo"""
        # Converti formati vari in HH:MM
        time_str = time_str.replace('.', ':')
        return time_str
    
    def _detect_negation(self, text: str) -> bool:
        """Rileva negazione nel testo"""
        for pattern in self._negation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _detect_urgency(self, text: str) -> List[str]:
        """Rileva markers di urgenza"""
        found_markers = []
        for level, markers in self._urgency_markers.items():
            for marker in markers:
                if marker in text.lower():
                    found_markers.append(f"{level}:{marker}")
        return found_markers
    
    def _detect_sentiment(self, text: str) -> str:
        """Rileva sentiment del testo"""
        text_lower = text.lower()
        
        for marker in self._sentiment_markers["positive"]:
            if marker in text_lower:
                return "positive"
        
        for marker in self._sentiment_markers["negative"]:
            if marker in text_lower:
                return "negative"
        
        return "neutral"
    
    def _is_compound_command(self, text: str) -> bool:
        """Verifica se è un comando composto"""
        for pattern in self._compound_connectors:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _parse_compound_command(self, text: str, context: Dict) -> Intent:
        """Parse comando composto in sub-intents"""
        # Split by connectors
        parts = re.split(r'\s+(?:poi|e poi|quindi|dopo|e|then|and then)\s+|[,;]\s*', 
                        text, flags=re.IGNORECASE)
        
        sub_intents = []
        for part in parts:
            part = part.strip()
            if part:
                sub_intent = self.interpret(part, context)
                sub_intents.append(sub_intent)
        
        if not sub_intents:
            return self._handle_unknown(text, context)
        
        # Create compound intent
        compound = Intent(
            name="compound_command",
            category=IntentCategory.COMPOUND,
            confidence=min(si.confidence for si in sub_intents),
            entities={"parts": [si.to_dict() for si in sub_intents]},
            original_text=text
        )
        compound.is_compound = True
        compound.sub_intents = sub_intents
        
        return compound
    
    def _handle_unknown(self, text: str, context: Dict) -> Intent:
        """Gestisce intent non riconosciuto"""
        # Prova a inferire da context
        if self._context_history:
            last_intent = self._context_history[-1]
            
            # Se è una risposta breve, potrebbe essere continuazione
            if len(text.split()) <= 3:
                return Intent(
                    name="continuation",
                    category=last_intent.category,
                    confidence=0.4,
                    entities={"text": text, "context_intent": last_intent.name},
                    original_text=text
                )
        
        return Intent(
            name="unknown",
            category=IntentCategory.UNKNOWN,
            confidence=0.0,
            entities={"raw_text": text},
            original_text=text
        )
    
    def _update_context(self, intent: Intent):
        """Aggiorna context history"""
        self._context_history.append(intent)
        
        # Mantieni solo ultimi N
        if len(self._context_history) > self._max_context:
            self._context_history.pop(0)
        
        # Update current topic
        if intent.category != IntentCategory.CONVERSATION:
            self._current_topic = intent.name
    
    # ========== PUBLIC API ==========
    
    def get_intent_action(self, intent: Intent) -> Dict:
        """Converte un intento in azione eseguibile"""
        action_mapping = {
            # System
            "shutdown": {"action": "shutdown", "params": {}},
            "restart": {"action": "restart", "params": {}},
            "lock": {"action": "lock", "params": {}},
            "sleep": {"action": "sleep", "params": {}},
            "screenshot": {"action": "screenshot", "params": {}},
            
            # File
            "open_file": {"action": "open_file", "params": {"path": intent.entities.get("file_name", "")}},
            "create_file": {"action": "create_file", "params": {"path": intent.entities.get("file_name", "")}},
            "delete_file": {"action": "delete_file", "params": {"path": intent.entities.get("file_name", "")}},
            
            # App
            "open_app": {"action": "open_app", "params": {"name": intent.entities.get("app_name", "")}},
            "close_app": {"action": "close_app", "params": {"name": intent.entities.get("app_name", "")}},
            
            # Web
            "search_web": {"action": "search_web", "params": {"query": intent.entities.get("query", "")}},
            "open_url": {"action": "open_url", "params": {"url": intent.entities.get("url", "")}},
            
            # Media
            "volume_up": {"action": "volume_up", "params": {}},
            "volume_down": {"action": "volume_down", "params": {}},
            "set_volume": {"action": "set_volume", "params": {"level": intent.entities.get("level", 50)}},
            "mute": {"action": "mute", "params": {}},
            
            # Default
            "unknown": {"action": "respond", "params": {"message": "Non ho capito la richiesta"}}
        }
        
        return action_mapping.get(intent.name, {"action": "respond", "params": {}})
    
    def add_intent_pattern(self, intent_name: str, patterns: List[str], 
                          category: IntentCategory, entities: List[str] = None,
                          priority: str = "normal"):
        """Aggiunge nuovo pattern di intento"""
        self._intent_patterns[intent_name] = {
            "patterns": patterns,
            "category": category,
            "entities": entities or [],
            "priority": priority
        }
    
    def add_app_mapping(self, alias: str, executable: str):
        """Aggiunge mapping nome app"""
        self._app_mappings[alias.lower()] = executable
    
    def add_entity_extractor(self, entity_type: str, patterns: List[str]):
        """Aggiunge extractor per entità"""
        self._entity_extractors[entity_type] = patterns
    
    def get_supported_intents(self) -> List[str]:
        """Lista intenti supportati"""
        return list(self._intent_patterns.keys())
    
    def get_context_history(self) -> List[Dict]:
        """Ottiene storico context"""
        return [i.to_dict() for i in self._context_history]
    
    def clear_context(self):
        """Pulisce context history"""
        self._context_history.clear()
        self._current_topic = None
    
    def get_status(self) -> Dict:
        """Stato dell'interpreter"""
        return {
            "supported_intents": len(self._intent_patterns),
            "entity_extractors": len(self._entity_extractors),
            "app_mappings": len(self._app_mappings),
            "categories": [c.value for c in IntentCategory],
            "context_history_size": len(self._context_history),
            "current_topic": self._current_topic
        }
