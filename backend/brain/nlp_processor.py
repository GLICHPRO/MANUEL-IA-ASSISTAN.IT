"""
NLP Processor - Natural Language Processing Engine
Handles intent extraction, entity recognition, and language understanding
Enhanced with smart local responses
"""

from typing import Dict, List, Any, Optional
from loguru import logger
import re
import random
from datetime import datetime
import math


class NLPProcessor:
    """Natural Language Processing for Gideon with Smart Responses"""
    
    # Cache per intent già processati (LRU-like)
    _intent_cache: dict = {}
    _cache_max_size: int = 500
    
    def __init__(self):
        self.intents_patterns = {
            # Time - ampia varietà di modi per chiedere l'ora
            "time": [
                r"\b(che ore|che ora|l'ora|ora è|ore sono|orario)\b",
                r"\b(dimmi l'ora|sai che ore|sai l'ora)\b",
                r"\b(quanto tempo|che giorno|data di oggi|oggi che giorno)\b"
            ],
            # Status - stato del sistema
            "status": [
                r"\b(stato|status|situazione|condizioni|come stai|come va)\b",
                r"\b(sistema|server|servizi|pc|computer)\b",
                r"\b(sei attivo|funziona tutto|tutto ok|tutto bene)\b"
            ],
            # Analysis - analisi e controlli
            "analysis": [
                r"\b(analizza|esamina|controlla|verifica|scansiona|check)\b",
                r"\b(performance|prestazioni|metriche|statistiche)\b",
                r"\b(system|cleanup|pulizia|pulisci)\b"
            ],
            # Optimization - ottimizzazione
            "optimization": [
                r"\b(ottimizza|migliora|potenzia|velocizza)\b",
                r"\b(suggerimenti|consigli|raccomandazioni|tips)\b",
                r"\b(come posso migliorare|cosa posso fare)\b"
            ],
            # Control - comandi di controllo
            "control": [
                r"\b(avvia|ferma|riavvia|deploy|rollback|start|stop|restart)\b",
                r"\b(esegui|lancia|termina|apri|chiudi)\b"
            ],
            # Math/Calculation - calcoli matematici
            "calculation": [
                r"\b(calcola|calcolo|quanto fa|somma|sottrai|moltiplica|dividi)\b",
                r"\b(radice|quadrato|potenza|percentuale)\b",
                r"\d+\s*[\+\-\*\/\^]\s*\d+",
                r"\b(\d+)\s*(più|meno|per|diviso|x)\s*(\d+)\b"
            ],
            # Information - domande generiche
            "information": [
                r"\b(cos[aè]|cosa|come|perch[eé]|quando|dove|chi|qual[eè])\b",
                r"\b(spiegami|dimmi|raccontami|parlami)\b"
            ],
            # Greeting - saluti
            "greeting": [
                r"\b(ciao|buongiorno|buonasera|buonanotte|salve|hey|ehi|yo)\b",
                r"^\s*(ciao|hey|ehi)\s*$",
                r"\bgideon\b"
            ],
            # Thanks - ringraziamenti
            "thanks": [
                r"\b(grazie|thanks|ringrazio|ti ringrazio)\b",
                r"\b(grande|bravo|ottimo lavoro|ben fatto)\b"
            ],
            # Help - richieste di aiuto
            "help": [
                r"\b(aiuto|help|aiutami|assistenza)\b",
                r"\b(cosa puoi fare|che sai fare|funzionalit[aà])\b",
                r"\b(comandi|istruzioni|guida)\b"
            ],
            # Identity - chi sei
            "identity": [
                r"\b(chi sei|come ti chiami|il tuo nome|presentati)\b",
                r"\b(cosa sei|che cosa sei|sei un)\b"
            ],
            # Weather (placeholder)
            "weather": [
                r"\b(meteo|tempo|piove|sole|temperatura|clima)\b",
                r"\b(che tempo fa|previsioni)\b"
            ],
        }
        
        self.entity_patterns = {
            "target": [
                r"\b(database|db|server|sistema|servizio|applicazione|app)\b",
                r"\b(cpu|memoria|disco|rete|network)\b"
            ],
            "action": [
                r"\b(restart|riavvia|stop|ferma|start|avvia|deploy|rollback)\b"
            ],
            "timeframe": [
                r"\b(oggi|ieri|domani|settimana|mese|anno)\b",
                r"\b(\d+)\s*(ore?|giorni?|minuti?)\b"
            ],
            "number": [
                r"\b(\d+(?:\.\d+)?)\b"
            ]
        }
        
        # Smart responses for various intents (no API needed)
        self.smart_responses = {
            "greeting": [
                "Ciao! Sono Gideon, come posso aiutarti oggi? 😊",
                "Ehi! Tutto bene? Dimmi cosa ti serve!",
                "Buongiorno! Sono pronto ad assisterti.",
                "Ciao! Chiedimi qualsiasi cosa, sono qui per te!",
                "Hey! Gideon al tuo servizio. Come posso essere utile?"
            ],
            "thanks": [
                "Figurati! È un piacere aiutarti! 😊",
                "Di niente! Se hai altre domande, sono qui.",
                "Grazie a te! Fammi sapere se serve altro.",
                "Prego! Sempre a disposizione!",
                "Non c'è di che! È il mio lavoro 🚀"
            ],
            "identity": [
                "Sono G.I.D.E.O.N. - Generative Intelligence for Dynamic Executive Operations Network. Un assistente AI avanzato creato per aiutarti!",
                "Mi chiamo Gideon! Sono il tuo assistente personale AI, specializzato in analisi di sistema, automazione e supporto intelligente.",
                "Sono Gideon, un'intelligenza artificiale progettata per essere il tuo copilota digitale. Posso aiutarti con analisi, calcoli, automazioni e molto altro!"
            ],
            "weather": [
                "Non ho accesso diretto ai dati meteo in tempo reale, ma posso suggerirti di controllare un servizio meteo online. Vuoi che apra il browser?",
                "Per il meteo preciso ti consiglio di controllare un'app dedicata. Posso aiutarti con altro nel frattempo?"
            ],
        }
        
    async def load_models(self):
        """Load NLP models"""
        logger.info("📚 Loading NLP models...")
        logger.info("✅ NLP models loaded (enhanced local mode)")
        
    async def extract_intent(self, text: str) -> Dict[str, Any]:
        """
        Extract intent from text with enhanced detection
        Uses caching for repeated queries
        """
        text_lower = text.lower().strip()
        
        # Check cache first (velocizza risposte ripetute)
        cache_key = text_lower[:100]  # Limita lunghezza chiave
        if cache_key in NLPProcessor._intent_cache:
            cached = NLPProcessor._intent_cache[cache_key]
            logger.debug(f"⚡ Intent cache hit: {cached['intent']}")
            return cached
        
        # Quick pattern matching for common phrases (risposta istantanea)
        quick_match = self._quick_intent_match(text_lower)
        if quick_match:
            entities = await self._extract_entities(text_lower)
            result = {
                "intent": quick_match,
                "confidence": 0.95,
                "entities": entities
            }
            self._add_to_cache(cache_key, result)
            return result
        
        # Standard pattern matching
        best_intent = "conversation"
        best_score = 0.0
        
        for intent, patterns in self.intents_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches)
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Calculate confidence
        confidence = min(0.95, 0.5 + (best_score * 0.15))
        
        # Extract entities
        entities = await self._extract_entities(text_lower)
        
        result = {
            "intent": best_intent,
            "confidence": confidence,
            "entities": entities
        }
        
        # Cache result
        self._add_to_cache(cache_key, result)
        
        return result
    
    def _add_to_cache(self, key: str, value: dict):
        """Add to cache with size limit"""
        if len(NLPProcessor._intent_cache) >= NLPProcessor._cache_max_size:
            # Rimuovi primi 100 elementi (FIFO semplificato)
            keys_to_remove = list(NLPProcessor._intent_cache.keys())[:100]
            for k in keys_to_remove:
                del NLPProcessor._intent_cache[k]
        NLPProcessor._intent_cache[key] = value
    
    def _quick_intent_match(self, text: str) -> Optional[str]:
        """Quick match for common phrases"""
        quick_patterns = {
            "time": ["che ore", "che ora", "l'ora", "ore sono", "ora è"],
            "greeting": ["ciao", "buongiorno", "buonasera", "salve", "hey gideon", "ehi gideon"],
            "thanks": ["grazie", "thanks", "ottimo"],
            "help": ["aiuto", "help", "cosa puoi fare", "comandi"],
            "status": ["come stai", "tutto bene", "stato sistema", "sei attivo"],
            "identity": ["chi sei", "come ti chiami", "il tuo nome"],
        }
        
        for intent, phrases in quick_patterns.items():
            for phrase in phrases:
                if phrase in text:
                    return intent
        return None
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text"""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    entities[entity_type] = match.group(0)
                    break
        
        # Extract numbers for calculations
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            entities['numbers'] = [float(n) for n in numbers]
        
        return entities
    
    def get_smart_response(self, intent: str) -> Optional[str]:
        """Get a smart response for an intent without API"""
        if intent in self.smart_responses:
            return random.choice(self.smart_responses[intent])
        return None
    
    async def handle_calculation(self, text: str) -> Optional[Dict[str, Any]]:
        """Handle math calculations locally"""
        text_lower = text.lower()
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if len(numbers) < 2:
            return None
        
        a, b = float(numbers[0]), float(numbers[1])
        result = None
        operation = ""
        
        # Detect operation
        if any(op in text_lower for op in ["più", "+", "somma", "aggiungi"]):
            result = a + b
            operation = "somma"
        elif any(op in text_lower for op in ["meno", "-", "sottrai", "differenza"]):
            result = a - b
            operation = "sottrazione"
        elif any(op in text_lower for op in ["per", "x", "*", "moltiplica", "moltiplicato"]):
            result = a * b
            operation = "moltiplicazione"
        elif any(op in text_lower for op in ["diviso", "/", "dividi", "divisione"]):
            result = a / b if b != 0 else "infinito (divisione per zero!)"
            operation = "divisione"
        elif any(op in text_lower for op in ["potenza", "elevato", "^", "alla"]):
            result = a ** b
            operation = "potenza"
        elif any(op in text_lower for op in ["radice", "sqrt"]):
            result = math.sqrt(a)
            operation = "radice quadrata"
        elif any(op in text_lower for op in ["percentuale", "%"]):
            result = (a / 100) * b
            operation = "percentuale"
        
        if result is not None:
            # Risposta semplice: solo il risultato
            if isinstance(result, float):
                if result == int(result):
                    response = f"{int(result)}"
                else:
                    response = f"{result:.2f}"
            else:
                response = f"{result}"
            
            return {
                "success": True,
                "intent": "calculation",
                "text": response,
                "tts_text": f"Il risultato è {response}",
                "data": {
                    "operation": operation,
                    "operands": [a, b],
                    "result": result
                }
            }
        
        return None
    
    async def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        positive_words = ["bene", "ottimo", "perfetto", "grazie", "eccellente", 
                        "fantastico", "grande", "bravo", "wow", "bellissimo"]
        negative_words = ["male", "problema", "errore", "critico", "fallito", 
                         "pessimo", "sbagliato", "no", "non funziona", "rotto"]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {"positive": 0.5, "negative": 0.0, "neutral": 0.5}
        
        return {
            "positive": pos_count / max(1, total),
            "negative": neg_count / max(1, total),
            "neutral": 1.0 - ((pos_count + neg_count) / max(1, total))
        }
