"""
Gideon Assistant Brain - Core Intelligence System
Handles command processing, NLP, and decision making
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger
import json

from core.config import settings
from brain.nlp_processor import NLPProcessor
from brain.memory_manager import MemoryManager
from brain.optimizer import OptimizerEngine
from brain.reasoning_engine import ReasoningEngine
from brain.ai_search import AISearchService
from brain.action_manager import ActionManager
from brain.personality import GideonPersonality, get_personality
from brain.ai_providers import get_ai_manager, AIProviderManager
from database.models import Interaction, AnalysisResult


class GideonAssistant:
    """Main assistant brain coordinating all AI functions"""
    
    def __init__(self):
        self.nlp = NLPProcessor()
        self.memory = MemoryManager()
        self.optimizer = OptimizerEngine()
        self.reasoning = ReasoningEngine()
        self.ai_manager = get_ai_manager()  # Multi-provider AI system
        self.ai_search = AISearchService()
        self.action_manager = ActionManager()
        self.personality = get_personality()
        self.is_initialized = False
        self.pilot_mode_active = False
        self.pilot_activated_at: Optional[datetime] = None
        self.current_level = "normal"  # normal, advanced, pilot
        
        # Gideon AI personality levels (The Flash style)
        self.level_configs = {
            "normal": {
                "autonomy": 0.2,
                "proactive": False,
                "auto_execute": False,
                "deep_reasoning": False,
                "personality_style": "friendly_assistant"
            },
            "advanced": {
                "autonomy": 0.5,
                "proactive": False,
                "auto_execute": False,
                "deep_reasoning": True,
                "personality_style": "analytical_expert"
            },
            "pilot": {
                "autonomy": 1.0,
                "proactive": True,
                "auto_execute": True,
                "deep_reasoning": True,
                "personality_style": "autonomous_ai"
            }
        }
        
    async def initialize(self):
        """Initialize all brain components"""
        logger.info("🧠 Initializing G.I.D.E.O.N. Brain...")
        logger.info("   Generative Intelligence for Dynamic Executive Operations Network")
        
        await self.nlp.load_models()
        await self.memory.initialize()
        await self.optimizer.initialize()
        await self.reasoning.initialize()
        await self.ai_search.initialize()
        await self.action_manager.initialize()
        await self.ai_manager.initialize()  # Initialize AI providers
        
        self.is_initialized = True
        logger.info("✅ G.I.D.E.O.N. initialized successfully")
        
    async def shutdown(self):
        """Cleanup brain resources"""
        await self.memory.close()
        await self.reasoning.shutdown()
        await self.optimizer.shutdown()
        await self.ai_search.shutdown()
        await self.action_manager.shutdown()
        await self.ai_manager.shutdown()  # Shutdown AI providers
        
    def is_ready(self) -> bool:
        """Check if brain is ready"""
        return self.is_initialized
    
    def get_level_config(self) -> Dict[str, Any]:
        """Get current level configuration"""
        return self.level_configs.get(self.current_level, self.level_configs["normal"])
    
    def _requires_autonomous_reasoning(self, text: str) -> bool:
        """
        Detect if a query requires autonomous reasoning.
        Used to auto-activate reasoning for complex questions.
        In Advanced/Pilot mode, always use reasoning for complex queries.
        """
        text_lower = text.lower()
        
        # In pilot mode, use reasoning more aggressively
        config = self.get_level_config()
        if config["deep_reasoning"]:
            # Lower threshold for reasoning in advanced/pilot mode
            if len(text.split()) > 5 or text.endswith("?"):
                return True
        
        # Complex reasoning indicators
        reasoning_indicators = [
            # Why/How questions
            "perché", "perche", "come mai", "per quale motivo",
            # Explanations
            "spiegami", "spiega", "cosa significa", "come funziona",
            # Analysis
            "analizza", "valuta", "confronta", "differenza tra",
            # Decisions
            "cosa dovrei", "conviene", "meglio se", "consigliami",
            # Complex calculations
            "se", "allora", "quindi", "supponendo",
            # Predictions
            "cosa succederebbe", "cosa accadrebbe", "prevedi",
            # Causality
            "conseguenze", "impatto", "effetto di",
            # Problem solving
            "risolvi", "problema", "soluzione per", "come posso",
            # Deep thinking
            "secondo te", "cosa pensi", "tua opinione"
        ]
        
        # Check for indicators
        has_indicator = any(ind in text_lower for ind in reasoning_indicators)
        
        # Check for question complexity (long questions often need reasoning)
        is_complex_question = text.endswith("?") and len(text.split()) > 8
        
        # Multiple clauses suggest complexity
        has_multiple_clauses = any(conj in text_lower for conj in [" e ", " o ", " ma ", " però ", " quindi "])
        
        return has_indicator or (is_complex_question and has_multiple_clauses)
    
    # ============ FAST RESPONSE CACHE ============
    _response_cache: dict = {}
    _cache_ttl: int = 300  # 5 minuti
    
    def _get_cached_response(self, text: str) -> Optional[Dict]:
        """Get cached response if available and fresh"""
        import time
        cache_key = text.lower().strip()[:100]
        if cache_key in self._response_cache:
            cached, timestamp = self._response_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"⚡ Response cache hit")
                return cached
            else:
                del self._response_cache[cache_key]
        return None
    
    def _cache_response(self, text: str, response: Dict):
        """Cache a response"""
        import time
        if len(self._response_cache) > 200:
            # Cleanup old entries
            current = time.time()
            self._response_cache = {
                k: v for k, v in self._response_cache.items()
                if current - v[1] < self._cache_ttl
            }
        cache_key = text.lower().strip()[:100]
        self._response_cache[cache_key] = (response, time.time())
        
    async def process_command(
        self,
        text: str,
        mode: str = "text",
        context: Optional[Dict[str, Any]] = None,
        pilot_execute: bool = False,
        max_tokens: int = 300,
        temperature: float = 0.5
    ) -> Dict[str, Any]:
        """
        Process a command from user
        
        Args:
            text: Command text
            mode: 'voice', 'text', or 'pilot'
            context: Additional context
            pilot_execute: If True, execute pilot mode commands
            max_tokens: Max tokens for AI response
            temperature: Temperature for AI response
            
        Returns:
            Response dictionary with result
        """
        # Store parameters for AI calls
        self._current_max_tokens = max_tokens
        self._current_temperature = temperature
        logger.info(f"📝 Processing command: {text[:50]}... (mode: {mode}, tokens={max_tokens})")
        
        try:
            # ===== FAST PATH: Check cache first =====
            cached = self._get_cached_response(text)
            if cached and mode != "pilot":
                return cached
            
            # ===== SUPER FAST PATH: AI queries skip all NLP =====
            text_lower = text.lower()
            if any(word in text_lower for word in ["chatgpt", "chiedi a gpt", "chiedi all'ai", "cerca su chatgpt", "domanda a chatgpt"]):
                response = await self._handle_chatgpt_query(text)
                self._cache_response(text, response)
                return response
            
            if any(word in text_lower for word in ["perplexity", "cerca con perplexity", "chiedi a perplexity"]):
                response = await self._handle_perplexity_query(text)
                self._cache_response(text, response)
                return response
            
            # If in pilot mode, use enhanced processing
            if mode == "pilot" or pilot_execute:
                return await self._handle_pilot_command(text, context)
            
            # Get relevant context from memory
            memory_context = await self.memory.get_relevant_context(text)
            
            # Merge with provided context
            full_context = {**(context or {}), "memory": memory_context}
            
            # Extract intent and entities
            intent_result = await self.nlp.extract_intent(text)
            intent = intent_result["intent"]
            confidence = intent_result["confidence"]
            entities = intent_result["entities"]
            
            logger.debug(f"Intent: {intent} ({confidence:.2f}), Entities: {entities}")
            logger.debug(f"Recent topics: {memory_context.get('recent_topics', [])}")
            
            # Check for pilot deactivation (must be before activation check)
            text_lower = text.lower()
            if ("disattiva" in text_lower or "esci" in text_lower or "termina" in text_lower) and ("pilot" in text_lower or "pilota" in text_lower):
                return await self._handle_pilot_deactivation()
            
            # Check for pilot activation
            if "pilot" in text.lower() and "attiva" in text.lower():
                return await self._handle_pilot_activation(text)
            
            # AI queries già gestite nel SUPER FAST PATH sopra
            
            # Route to appropriate handler
            if intent == "time":
                response = await self._handle_time()
            elif intent == "greeting":
                response = await self._handle_greeting(text)
            elif intent == "thanks":
                response = await self._handle_thanks()
            elif intent == "identity":
                response = await self._handle_identity()
            elif intent == "help":
                response = await self._handle_help()
            elif intent == "calculation":
                response = await self._handle_calculation_smart(text, entities)
            elif intent == "weather":
                response = await self._handle_weather(text)
            elif intent == "status":
                response = await self._handle_system_status()
            elif intent == "analysis":
                response = await self._handle_analysis_request(entities)
            elif intent == "optimization":
                response = await self._handle_optimization(entities)
            elif intent == "control":
                response = await self._handle_control_command(entities)
            elif intent == "information":
                response = await self._handle_information_query(text, entities)
            elif any(kw in text.lower() for kw in ["calcola", "percentuale", "metriche"]):
                # Calculation mode - only for explicit calculation requests
                response = await self._handle_calculation(text, entities)
                # If calculation failed, fall through to AI
                if not response.get("success"):
                    response = None
            elif any(kw in text.lower() for kw in ["ottimizza", "migliora", "suggerimenti", "proponi", "consigli"]):
                # Optimization suggestions mode
                response = await self._handle_smart_optimization(text, entities)
            elif any(kw in text.lower() for kw in ["ragiona", "pensa", "analizza autonomamente", "ragionamento"]):
                # Explicit autonomous thinking mode
                response = await self._handle_autonomous_thinking(text, entities)
            elif self._requires_autonomous_reasoning(text):
                # Auto-detect complex queries that need reasoning
                response = await self._handle_autonomous_thinking(text, entities)
            
            # ===== NUOVI COMANDI =====
            # Date
            elif any(kw in text.lower() for kw in ["che giorno", "data di oggi", "che data", "giorno è"]):
                response = await self._handle_date()
            # Open website
            elif any(kw in text.lower() for kw in ["apri sito", "vai su", "apri google", "apri youtube", "apri facebook", 
                                                     "apri instagram", "apri twitter", "apri linkedin", "apri github",
                                                     "apri amazon", "apri netflix", "apri spotify", "apri whatsapp"]):
                response = await self._handle_open_website(text)
            # Open app
            elif any(kw in text.lower() for kw in ["apri app", "apri applicazione", "avvia", "lancia", "apri notepad",
                                                    "apri calcolatrice", "apri chrome", "apri browser", "apri terminale",
                                                    "apri esplora", "apri paint", "apri word", "apri excel", 
                                                    "apri vscode", "apri spotify", "apri discord"]):
                response = await self._handle_open_app(text)
            # Screenshot
            elif any(kw in text.lower() for kw in ["screenshot", "cattura schermo", "foto schermo", "screen"]):
                response = await self._handle_screenshot()
            # Volume
            elif any(kw in text.lower() for kw in ["volume", "alza volume", "abbassa volume", "muto", "silenzia"]):
                response = await self._handle_volume(text)
            # Battery
            elif any(kw in text.lower() for kw in ["batteria", "carica", "autonomia", "percentuale batteria"]):
                response = await self._handle_battery()
            # Notes
            elif any(kw in text.lower() for kw in ["nota", "salva nota", "ricorda", "annota", "prendi nota", 
                                                    "mostra note", "le mie note", "cancella note"]):
                response = await self._handle_note(text)
            # Timer
            elif any(kw in text.lower() for kw in ["timer", "sveglia", "promemoria tra", "avvisami tra"]):
                response = await self._handle_timer(text)
            # Empty trash
            elif any(kw in text.lower() for kw in ["svuota cestino", "cestino", "pulisci cestino"]):
                response = await self._handle_empty_trash()
            # Search files
            elif any(kw in text.lower() for kw in ["cerca file", "trova file", "dove è il file", "cerca documento"]):
                response = await self._handle_search_files(text)
            # Lock PC
            elif any(kw in text.lower() for kw in ["blocca pc", "blocca computer", "blocca schermo", "lock"]):
                response = await self._handle_lock_pc()
            # Shutdown/Restart
            elif any(kw in text.lower() for kw in ["spegni pc", "spegni computer", "riavvia", "restart", 
                                                    "sospendi", "iberna", "arresta"]):
                response = await self._handle_shutdown(text)
            # Wikipedia / Chi è / Cos'è
            elif any(kw in text.lower() for kw in ["chi è", "chi era", "cos'è", "cosa è", "wikipedia", 
                                                    "dimmi di", "parlami di", "spiegami"]):
                response = await self._handle_wikipedia(text)
            # Traduzione
            elif any(kw in text.lower() for kw in ["traduci", "traduzione", "come si dice", "in inglese", 
                                                    "in italiano", "in francese", "in spagnolo", "in tedesco"]):
                response = await self._handle_translate(text)
            # Notizie
            elif any(kw in text.lower() for kw in ["notizie", "news", "ultime notizie", "cosa succede", 
                                                    "novità", "aggiornamenti"]):
                response = await self._handle_news(text)
            # Conversione valuta
            elif any(kw in text.lower() for kw in ["converti", "conversione", "euro in", "dollari in", 
                                                    "cambio", "valuta", "tasso"]):
                response = await self._handle_currency(text)
            # Definizione parola
            elif any(kw in text.lower() for kw in ["definizione", "significa", "significato", "definisci"]):
                response = await self._handle_definition(text)
            # Barzelletta / Joke
            elif any(kw in text.lower() for kw in ["barzelletta", "battuta", "racconta una barzelletta", 
                                                    "fammi ridere", "joke"]):
                response = await self._handle_joke()
            # Citazione / Quote
            elif any(kw in text.lower() for kw in ["citazione", "frase del giorno", "quote", "aforisma", 
                                                    "frase celebre", "ispirazione"]):
                response = await self._handle_quote()
            # Previsioni meteo estese
            elif any(kw in text.lower() for kw in ["previsioni", "meteo domani", "meteo settimana", 
                                                    "che tempo farà"]):
                response = await self._handle_weather_forecast(text)
            # ===== FINE NUOVI COMANDI =====
            
            else:
                # Default: smart conversational response with context
                response = await self._handle_smart_response(text, entities, intent_result)
            
            # Improve confidence based on historical data
            improved_confidence = await self.memory.calculate_improved_confidence(intent, confidence)
            
            # Save interaction to memory with full context
            await self.memory.save_interaction(
                query=text,
                response=response["text"],
                intent=intent,
                confidence=improved_confidence,
                mode=mode,
                context=full_context
            )
            
            # Add metadata to response
            response["confidence"] = improved_confidence
            response["avatar_expression"] = self._get_expression_for_intent(intent)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            error_msg = self.personality.get_error_phrase()
            return {
                "success": False,
                "text": error_msg,
                "tts_text": self.personality.wrap_for_tts(error_msg),
                "error": str(e)
            }
    
    async def _handle_greeting(self, text: str) -> Dict[str, Any]:
        """Handle greeting with personality"""
        greeting = self.personality.get_greeting()
        response = f"{greeting} Sono Gideon, il tuo assistente. Come posso aiutarti?"
        
        return {
            "success": True,
            "intent": "greeting",
            "text": response,
            "tts_text": self.personality.wrap_for_tts(response),
            "avatar_expression": "happy"
        }
    
    async def _handle_help(self) -> Dict[str, Any]:
        """Handle help request"""
        level = self.personality.current_level.value
        
        if level == "pilot":
            response = """Comandi disponibili:
• Ora/Data
• Stato sistema
• Analizza performance
• Ottimizza
• Pilot mode (attivo)
• Azioni: apri, chiudi, riavvia"""
        elif level == "advanced":
            response = """Funzionalità avanzate disponibili:
• Query temporali: "Che ore sono?"
• Stato sistema: "Come sta il sistema?"
• Analisi: "Analizza le prestazioni"
• Ottimizzazione: "Suggerimenti di ottimizzazione"
• Controlli: "Riavvia servizio X"
• Calcoli: "Calcola percentuale..."
• Ragionamento: "Ragiona su..."
Usa "Pilot mode" per controllo completo."""
        else:
            response = """Ciao! Ecco cosa posso fare per te:
• 🕐 Chiedi l'ora: "Che ore sono?"
• 📊 Stato sistema: "Come sta il sistema?"
• 🔍 Analisi: "Analizza le prestazioni"
• 💡 Ottimizzazioni: "Cosa posso migliorare?"
• 🤖 Domande: Chiedimi qualsiasi cosa!
• 🎤 Comandi vocali: Clicca il microfono e parla

Prova a dire "Ciao Gideon" o "Che ore sono?" 😊"""
        
        return {
            "success": True,
            "intent": "help",
            "text": response,
            "tts_text": self.personality.wrap_for_tts("Ecco cosa posso fare per te. Puoi chiedermi l'ora, lo stato del sistema, analisi, ottimizzazioni, o farmi domande generali."),
            "avatar_expression": "happy"
        }
    
    async def _handle_thanks(self) -> Dict[str, Any]:
        """Handle thank you with personality"""
        smart_response = self.nlp.get_smart_response("thanks")
        response = smart_response or "Di niente! Sono qui per aiutarti 😊"
        
        return {
            "success": True,
            "intent": "thanks",
            "text": response,
            "tts_text": self.personality.wrap_for_tts(response),
            "avatar_expression": "happy"
        }
    
    async def _handle_identity(self) -> Dict[str, Any]:
        """Handle identity questions (who are you?)"""
        smart_response = self.nlp.get_smart_response("identity")
        response = smart_response or "Sono G.I.D.E.O.N., il tuo assistente AI personale! Posso aiutarti con analisi, calcoli, ottimizzazioni e molto altro."
        
        return {
            "success": True,
            "intent": "identity",
            "text": response,
            "tts_text": self.personality.wrap_for_tts(response),
            "avatar_expression": "happy"
        }
    
    async def _handle_weather(self, text: str = "") -> Dict[str, Any]:
        """Handle weather questions using Open-Meteo API (free, no API key needed)"""
        import httpx
        
        # Extract city from text or use default
        text_lower = text.lower()
        
        # Common Italian cities with coordinates
        cities = {
            "roma": (41.9028, 12.4964, "Roma"),
            "milano": (45.4642, 9.1900, "Milano"),
            "napoli": (40.8518, 14.2681, "Napoli"),
            "torino": (45.0703, 7.6869, "Torino"),
            "firenze": (43.7696, 11.2558, "Firenze"),
            "bologna": (44.4949, 11.3426, "Bologna"),
            "venezia": (45.4408, 12.3155, "Venezia"),
            "palermo": (38.1157, 13.3615, "Palermo"),
            "genova": (44.4056, 8.9463, "Genova"),
            "bari": (41.1171, 16.8719, "Bari"),
            "catania": (37.5079, 15.0830, "Catania"),
            "verona": (45.4384, 10.9916, "Verona"),
            "padova": (45.4064, 11.8768, "Padova"),
            "trieste": (45.6495, 13.7768, "Trieste"),
            "brescia": (45.5416, 10.2118, "Brescia"),
            "parma": (44.8015, 10.3279, "Parma"),
            "modena": (44.6471, 10.9252, "Modena"),
            "reggio": (44.6989, 10.6297, "Reggio Emilia"),
            "perugia": (43.1107, 12.3908, "Perugia"),
            "cagliari": (39.2238, 9.1217, "Cagliari"),
        }
        
        # Find city in text
        city_name = "Roma"  # Default
        lat, lon = 41.9028, 12.4964
        
        for city_key, (city_lat, city_lon, city_display) in cities.items():
            if city_key in text_lower:
                lat, lon = city_lat, city_lon
                city_name = city_display
                break
        
        try:
            # Use Open-Meteo API (free, no key required)
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=Europe/Rome"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                data = resp.json()
            
            current = data.get("current", {})
            temp = current.get("temperature_2m", "N/A")
            humidity = current.get("relative_humidity_2m", "N/A")
            wind = current.get("wind_speed_10m", "N/A")
            weather_code = current.get("weather_code", 0)
            
            # Weather code to description and emoji
            weather_descriptions = {
                0: ("Sereno", "☀️"),
                1: ("Prevalentemente sereno", "🌤️"),
                2: ("Parzialmente nuvoloso", "⛅"),
                3: ("Nuvoloso", "☁️"),
                45: ("Nebbia", "🌫️"),
                48: ("Nebbia gelata", "🌫️"),
                51: ("Pioggerella leggera", "🌧️"),
                53: ("Pioggerella", "🌧️"),
                55: ("Pioggerella intensa", "🌧️"),
                61: ("Pioggia leggera", "🌧️"),
                63: ("Pioggia", "🌧️"),
                65: ("Pioggia intensa", "🌧️"),
                71: ("Neve leggera", "❄️"),
                73: ("Neve", "❄️"),
                75: ("Neve intensa", "❄️"),
                77: ("Grandine", "🌨️"),
                80: ("Rovesci leggeri", "🌦️"),
                81: ("Rovesci", "🌦️"),
                82: ("Rovesci intensi", "🌦️"),
                85: ("Neve a tratti", "🌨️"),
                86: ("Neve intensa a tratti", "🌨️"),
                95: ("Temporale", "⛈️"),
                96: ("Temporale con grandine", "⛈️"),
                99: ("Temporale forte con grandine", "⛈️"),
            }
            
            desc, emoji = weather_descriptions.get(weather_code, ("Variabile", "🌡️"))
            
            response = f"{emoji} Meteo a {city_name}:\n"
            response += f"🌡️ Temperatura: {temp}°C\n"
            response += f"☁️ Condizioni: {desc}\n"
            response += f"💧 Umidità: {humidity}%\n"
            response += f"💨 Vento: {wind} km/h"
            
            tts_response = f"A {city_name} ci sono {temp} gradi, {desc.lower()}, umidità al {humidity} percento"
            
            return {
                "success": True,
                "intent": "weather",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(tts_response),
                "data": {
                    "city": city_name,
                    "temperature": temp,
                    "humidity": humidity,
                    "wind": wind,
                    "description": desc
                },
                "avatar_expression": "happy"
            }
            
        except Exception as e:
            response = f"⚠️ Non riesco a recuperare il meteo. Errore: {str(e)}"
            return {
                "success": False,
                "intent": "weather",
                "text": response,
                "tts_text": self.personality.wrap_for_tts("Mi dispiace, non riesco a recuperare i dati meteo al momento"),
                "avatar_expression": "sad"
            }
    
    async def _handle_calculation_smart(self, text: str, entities: Dict) -> Dict[str, Any]:
        """Handle calculation with NLP smart handler"""
        # Try smart calculation first
        calc_result = await self.nlp.handle_calculation(text)
        if calc_result:
            return calc_result
        
        # Fallback to original handler
        return await self._handle_calculation(text, entities)
    
    async def _handle_time(self) -> Dict[str, Any]:
        """Handle time query with personality"""
        # Use personality for natural response
        response_text = self.personality.get_time_response()
        
        now = datetime.now()
        time_str = now.strftime("%H:%M")
        date_str = now.strftime("%d %B %Y")
        
        return {
            "success": True,
            "intent": "time",
            "text": response_text,
            "tts_text": self.personality.wrap_for_tts(response_text),
            "data": {
                "time": time_str,
                "date": date_str,
                "timestamp": now.isoformat()
            }
        }
    
    # ============== NUOVI COMANDI ==============
    
    async def _handle_date(self) -> Dict[str, Any]:
        """Handle date query"""
        now = datetime.now()
        giorni = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
        mesi = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", 
                "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"]
        
        giorno_nome = giorni[now.weekday()]
        mese_nome = mesi[now.month - 1]
        
        response = f"📅 Oggi è {giorno_nome} {now.day} {mese_nome} {now.year}"
        
        return {
            "success": True,
            "intent": "date",
            "text": response,
            "tts_text": self.personality.wrap_for_tts(response),
            "data": {"date": now.strftime("%Y-%m-%d"), "day_name": giorno_nome}
        }
    
    async def _handle_open_website(self, text: str) -> Dict[str, Any]:
        """Open a website in browser"""
        import webbrowser
        
        text_lower = text.lower()
        
        # Common websites mapping
        sites = {
            "google": "https://www.google.com",
            "youtube": "https://www.youtube.com",
            "facebook": "https://www.facebook.com",
            "instagram": "https://www.instagram.com",
            "twitter": "https://www.twitter.com",
            "x": "https://www.x.com",
            "linkedin": "https://www.linkedin.com",
            "github": "https://www.github.com",
            "gmail": "https://mail.google.com",
            "whatsapp": "https://web.whatsapp.com",
            "amazon": "https://www.amazon.it",
            "netflix": "https://www.netflix.com",
            "spotify": "https://open.spotify.com",
            "reddit": "https://www.reddit.com",
            "wikipedia": "https://www.wikipedia.org",
            "chatgpt": "https://chat.openai.com",
            "claude": "https://claude.ai",
        }
        
        url = None
        site_name = None
        
        for name, site_url in sites.items():
            if name in text_lower:
                url = site_url
                site_name = name.capitalize()
                break
        
        # Try to extract URL from text
        if not url:
            import re
            url_pattern = r'https?://[^\s]+'
            match = re.search(url_pattern, text)
            if match:
                url = match.group()
                site_name = url
            else:
                # Try to find domain-like text
                words = text.split()
                for word in words:
                    if '.' in word and len(word) > 3:
                        url = f"https://{word}" if not word.startswith('http') else word
                        site_name = word
                        break
        
        if url:
            try:
                webbrowser.open(url)
                response = f"🌐 Sto aprendo {site_name}..."
                return {
                    "success": True,
                    "intent": "open_website",
                    "text": response,
                    "tts_text": self.personality.wrap_for_tts(response),
                    "data": {"url": url, "site": site_name}
                }
            except Exception as e:
                return {
                    "success": False,
                    "intent": "open_website",
                    "text": f"❌ Errore nell'aprire il sito: {e}",
                    "data": {"error": str(e)}
                }
        
        return {
            "success": True,
            "intent": "open_website",
            "text": "🌐 Quale sito vuoi aprire? Dimmi il nome o l'URL.",
            "data": {}
        }
    
    async def _handle_open_app(self, text: str) -> Dict[str, Any]:
        """Open an application"""
        import subprocess
        import os
        
        text_lower = text.lower()
        
        # Common Windows applications
        apps = {
            "notepad": "notepad.exe",
            "blocco note": "notepad.exe",
            "calcolatrice": "calc.exe",
            "calculator": "calc.exe",
            "esplora risorse": "explorer.exe",
            "explorer": "explorer.exe",
            "file explorer": "explorer.exe",
            "paint": "mspaint.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "outlook": "outlook.exe",
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "edge": "msedge.exe",
            "browser": "msedge.exe",
            "terminale": "wt.exe",
            "terminal": "wt.exe",
            "powershell": "powershell.exe",
            "cmd": "cmd.exe",
            "prompt": "cmd.exe",
            "impostazioni": "ms-settings:",
            "settings": "ms-settings:",
            "task manager": "taskmgr.exe",
            "gestione attività": "taskmgr.exe",
            "spotify": "spotify.exe",
            "discord": "discord.exe",
            "teams": "teams.exe",
            "vs code": "code.exe",
            "vscode": "code.exe",
            "visual studio code": "code.exe",
        }
        
        app_to_open = None
        app_name = None
        
        for name, exe in apps.items():
            if name in text_lower:
                app_to_open = exe
                app_name = name.capitalize()
                break
        
        if app_to_open:
            try:
                if app_to_open.startswith("ms-"):
                    os.startfile(app_to_open)
                else:
                    subprocess.Popen(app_to_open, shell=True)
                
                response = f"📱 Sto aprendo {app_name}..."
                return {
                    "success": True,
                    "intent": "open_app",
                    "text": response,
                    "tts_text": self.personality.wrap_for_tts(response),
                    "data": {"app": app_name, "exe": app_to_open}
                }
            except Exception as e:
                return {
                    "success": False,
                    "intent": "open_app",
                    "text": f"❌ Non riesco ad aprire {app_name}: {e}",
                    "data": {"error": str(e)}
                }
        
        return {
            "success": True,
            "intent": "open_app",
            "text": "📱 Quale applicazione vuoi aprire? Dimmi il nome del programma.",
            "data": {}
        }
    
    async def _handle_screenshot(self) -> Dict[str, Any]:
        """Take a screenshot"""
        try:
            import subprocess
            from pathlib import Path
            
            # Use Windows Snipping Tool
            screenshots_dir = Path.home() / "Pictures" / "Screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = screenshots_dir / f"screenshot_{timestamp}.png"
            
            # Try using PowerShell to capture screen
            ps_script = f'''
            Add-Type -AssemblyName System.Windows.Forms
            $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
            $bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
            $graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
            $bitmap.Save("{filename}")
            '''
            
            subprocess.run(["powershell", "-Command", ps_script], capture_output=True)
            
            response = f"📸 Screenshot salvato in: {filename}"
            return {
                "success": True,
                "intent": "screenshot",
                "text": response,
                "tts_text": self.personality.wrap_for_tts("Screenshot catturato!"),
                "data": {"path": str(filename)}
            }
        except Exception as e:
            # Fallback: open snipping tool
            try:
                import subprocess
                subprocess.Popen("snippingtool", shell=True)
                return {
                    "success": True,
                    "intent": "screenshot",
                    "text": "📸 Ho aperto lo Strumento di cattura. Seleziona l'area da catturare.",
                    "data": {}
                }
            except:
                return {
                    "success": False,
                    "intent": "screenshot",
                    "text": f"❌ Errore screenshot: {e}",
                    "data": {"error": str(e)}
                }
    
    async def _handle_volume(self, text: str) -> Dict[str, Any]:
        """Control system volume"""
        text_lower = text.lower()
        
        try:
            import subprocess
            
            if any(word in text_lower for word in ["muto", "mute", "silenzia", "silenzio"]):
                # Mute
                subprocess.run([
                    "powershell", "-Command",
                    "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"
                ], capture_output=True)
                response = "🔇 Volume disattivato"
                
            elif any(word in text_lower for word in ["alza", "aumenta", "su", "più alto"]):
                # Volume up
                for _ in range(5):  # 5 steps up
                    subprocess.run([
                        "powershell", "-Command",
                        "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"
                    ], capture_output=True)
                response = "🔊 Volume aumentato"
                
            elif any(word in text_lower for word in ["abbassa", "diminuisci", "giù", "più basso"]):
                # Volume down
                for _ in range(5):  # 5 steps down
                    subprocess.run([
                        "powershell", "-Command",
                        "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"
                    ], capture_output=True)
                response = "🔉 Volume diminuito"
                
            elif any(word in text_lower for word in ["massimo", "max", "al massimo"]):
                for _ in range(25):
                    subprocess.run([
                        "powershell", "-Command",
                        "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"
                    ], capture_output=True)
                response = "🔊 Volume al massimo"
                
            elif any(word in text_lower for word in ["minimo", "min", "al minimo"]):
                for _ in range(25):
                    subprocess.run([
                        "powershell", "-Command",
                        "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"
                    ], capture_output=True)
                response = "🔈 Volume al minimo"
            else:
                response = "🔊 Cosa vuoi fare con il volume? Posso alzarlo, abbassarlo, o metterlo in muto."
            
            return {
                "success": True,
                "intent": "volume",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(response),
                "data": {}
            }
        except Exception as e:
            return {
                "success": False,
                "intent": "volume",
                "text": f"❌ Errore nel controllo volume: {e}",
                "data": {"error": str(e)}
            }
    
    async def _handle_battery(self) -> Dict[str, Any]:
        """Get battery status"""
        try:
            import psutil
            
            battery = psutil.sensors_battery()
            if battery:
                percent = battery.percent
                plugged = battery.power_plugged
                
                status = "🔌 in carica" if plugged else "🔋 a batteria"
                
                # Time remaining
                if battery.secsleft > 0 and not plugged:
                    hours = battery.secsleft // 3600
                    minutes = (battery.secsleft % 3600) // 60
                    time_left = f" - circa {hours}h {minutes}m rimanenti" if hours > 0 else f" - circa {minutes} minuti rimanenti"
                else:
                    time_left = ""
                
                # Emoji based on level
                if percent > 80:
                    emoji = "🔋"
                elif percent > 50:
                    emoji = "🔋"
                elif percent > 20:
                    emoji = "🪫"
                else:
                    emoji = "⚠️🪫"
                
                response = f"{emoji} Batteria al {percent}% {status}{time_left}"
            else:
                response = "💻 Questo dispositivo non ha una batteria (probabilmente è un PC desktop)"
            
            return {
                "success": True,
                "intent": "battery",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(response),
                "data": {"percent": battery.percent if battery else None, "plugged": battery.power_plugged if battery else None}
            }
        except Exception as e:
            return {
                "success": False,
                "intent": "battery",
                "text": f"❌ Errore nel leggere lo stato della batteria: {e}",
                "data": {"error": str(e)}
            }
    
    async def _handle_note(self, text: str) -> Dict[str, Any]:
        """Save or retrieve notes"""
        text_lower = text.lower()
        
        # Notes are stored in memory
        if not hasattr(self, '_notes'):
            self._notes = []
        
        # Check if saving a note
        if any(word in text_lower for word in ["salva", "ricorda", "annota", "segna", "prendi nota"]):
            # Extract the note content
            note_content = text
            for prefix in ["salva nota", "ricorda che", "annota", "segna che", "prendi nota", 
                          "ricordami", "salva", "nota"]:
                if prefix in text_lower:
                    note_content = text_lower.split(prefix, 1)[-1].strip()
                    break
            
            if note_content and len(note_content) > 2:
                self._notes.append({
                    "content": note_content,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Also save to memory manager if available
                if hasattr(self, 'memory'):
                    await self.memory.add_to_context("notes", self._notes)
                
                response = f"📝 Ho salvato: \"{note_content}\""
                return {
                    "success": True,
                    "intent": "note_save",
                    "text": response,
                    "tts_text": self.personality.wrap_for_tts("Nota salvata!"),
                    "data": {"note": note_content, "total_notes": len(self._notes)}
                }
        
        # Check if listing notes
        elif any(word in text_lower for word in ["mostra note", "leggi note", "quali note", "le mie note", "elenco note"]):
            if self._notes:
                notes_text = "\n".join([f"• {n['content']}" for n in self._notes[-10:]])  # Last 10
                response = f"📝 Le tue note:\n{notes_text}"
            else:
                response = "📝 Non hai ancora salvato nessuna nota."
            
            return {
                "success": True,
                "intent": "note_list",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(f"Hai {len(self._notes)} note salvate"),
                "data": {"notes": self._notes}
            }
        
        # Check if clearing notes
        elif any(word in text_lower for word in ["cancella note", "elimina note", "svuota note"]):
            count = len(self._notes)
            self._notes = []
            response = f"🗑️ Ho cancellato {count} note."
            return {
                "success": True,
                "intent": "note_clear",
                "text": response,
                "tts_text": self.personality.wrap_for_tts("Note cancellate"),
                "data": {"deleted": count}
            }
        
        return {
            "success": True,
            "intent": "note",
            "text": "📝 Posso salvare note per te. Dimmi 'salva nota...' oppure 'mostra le mie note'.",
            "data": {}
        }
    
    async def _handle_timer(self, text: str) -> Dict[str, Any]:
        """Set a timer"""
        import re
        
        text_lower = text.lower()
        
        # Extract duration
        minutes = 0
        seconds = 0
        
        # Pattern: X minuti, X secondi, X min, X sec
        min_match = re.search(r'(\d+)\s*(minut[io]?|min)', text_lower)
        sec_match = re.search(r'(\d+)\s*(second[io]?|sec)', text_lower)
        
        if min_match:
            minutes = int(min_match.group(1))
        if sec_match:
            seconds = int(sec_match.group(1))
        
        total_seconds = minutes * 60 + seconds
        
        if total_seconds > 0:
            # Create async timer
            async def timer_callback():
                await asyncio.sleep(total_seconds)
                # This will trigger a notification - simplified version
                logger.info(f"⏰ Timer completato! ({minutes}m {seconds}s)")
            
            asyncio.create_task(timer_callback())
            
            time_str = ""
            if minutes > 0:
                time_str += f"{minutes} minuti"
            if seconds > 0:
                time_str += f" {seconds} secondi" if time_str else f"{seconds} secondi"
            
            response = f"⏱️ Timer impostato per {time_str}. Ti avviserò quando sarà finito!"
            return {
                "success": True,
                "intent": "timer",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(f"Timer impostato per {time_str}"),
                "data": {"minutes": minutes, "seconds": seconds, "total_seconds": total_seconds}
            }
        
        return {
            "success": True,
            "intent": "timer",
            "text": "⏱️ Per quanto tempo vuoi il timer? Dimmi ad esempio 'timer 5 minuti'.",
            "data": {}
        }
    
    async def _handle_empty_trash(self) -> Dict[str, Any]:
        """Empty the recycle bin"""
        try:
            import subprocess
            
            # PowerShell command to empty recycle bin
            result = subprocess.run([
                "powershell", "-Command",
                "Clear-RecycleBin -Force -ErrorAction SilentlyContinue"
            ], capture_output=True, text=True)
            
            response = "🗑️ Cestino svuotato!"
            return {
                "success": True,
                "intent": "empty_trash",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(response),
                "data": {}
            }
        except Exception as e:
            return {
                "success": False,
                "intent": "empty_trash",
                "text": f"❌ Errore nello svuotare il cestino: {e}",
                "data": {"error": str(e)}
            }
    
    async def _handle_search_files(self, text: str) -> Dict[str, Any]:
        """Search for files on the computer"""
        import subprocess
        
        text_lower = text.lower()
        
        # Extract search query
        search_query = text
        for prefix in ["cerca file", "trova file", "cerca", "trova", "dove è", "dov'è"]:
            if prefix in text_lower:
                search_query = text_lower.split(prefix, 1)[-1].strip()
                break
        
        if search_query and len(search_query) > 1:
            try:
                # Open Windows Search with query
                import webbrowser
                search_url = f"search-ms:query={search_query}"
                webbrowser.open(search_url)
                
                response = f"🔍 Ho aperto la ricerca per: \"{search_query}\""
                return {
                    "success": True,
                    "intent": "search_files",
                    "text": response,
                    "tts_text": self.personality.wrap_for_tts(f"Cerco {search_query}"),
                    "data": {"query": search_query}
                }
            except Exception as e:
                return {
                    "success": False,
                    "intent": "search_files",
                    "text": f"❌ Errore nella ricerca: {e}",
                    "data": {"error": str(e)}
                }
        
        return {
            "success": True,
            "intent": "search_files",
            "text": "🔍 Cosa vuoi cercare? Dimmi il nome del file.",
            "data": {}
        }
    
    async def _handle_lock_pc(self) -> Dict[str, Any]:
        """Lock the PC"""
        try:
            import subprocess
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], capture_output=True)
            
            return {
                "success": True,
                "intent": "lock_pc",
                "text": "🔒 PC bloccato!",
                "tts_text": self.personality.wrap_for_tts("PC bloccato"),
                "data": {}
            }
        except Exception as e:
            return {
                "success": False,
                "intent": "lock_pc",
                "text": f"❌ Errore: {e}",
                "data": {"error": str(e)}
            }
    
    async def _handle_shutdown(self, text: str) -> Dict[str, Any]:
        """Shutdown, restart or sleep PC"""
        text_lower = text.lower()
        
        try:
            import subprocess
            
            if any(word in text_lower for word in ["riavvia", "restart", "riavvio"]):
                response = "🔄 Riavvio il PC tra 10 secondi... Salva il tuo lavoro!"
                subprocess.Popen(["shutdown", "/r", "/t", "10"])
                action = "restart"
                
            elif any(word in text_lower for word in ["spegni", "shutdown", "arresta"]):
                response = "⏻ Spengo il PC tra 10 secondi... Salva il tuo lavoro!"
                subprocess.Popen(["shutdown", "/s", "/t", "10"])
                action = "shutdown"
                
            elif any(word in text_lower for word in ["sospendi", "sleep", "iberna", "standby"]):
                response = "😴 Metto il PC in sospensione..."
                subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"])
                action = "sleep"
                
            elif "annulla" in text_lower:
                subprocess.run(["shutdown", "/a"], capture_output=True)
                response = "✅ Operazione annullata!"
                action = "cancel"
            else:
                return {
                    "success": True,
                    "intent": "shutdown",
                    "text": "⏻ Vuoi spegnere, riavviare o mettere in sospensione il PC?",
                    "data": {}
                }
            
            return {
                "success": True,
                "intent": "shutdown",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(response),
                "data": {"action": action}
            }
        except Exception as e:
            return {
                "success": False,
                "intent": "shutdown",
                "text": f"❌ Errore: {e}",
                "data": {"error": str(e)}
            }
    
    # ============== COMANDI INFORMATIVI ==============
    
    async def _handle_wikipedia(self, text: str) -> Dict[str, Any]:
        """Search Wikipedia for information"""
        import httpx
        
        text_lower = text.lower()
        
        # Extract search term
        search_term = text
        for prefix in ["chi è", "chi era", "cos'è", "cosa è", "wikipedia", 
                       "dimmi di", "parlami di", "spiegami", "cerca"]:
            if prefix in text_lower:
                search_term = text_lower.split(prefix, 1)[-1].strip()
                break
        
        # Clean up search term
        search_term = search_term.strip("?!.,")
        
        if not search_term or len(search_term) < 2:
            return {
                "success": False,
                "intent": "wikipedia",
                "text": "❓ Di cosa vuoi sapere? Dimmi un argomento da cercare.",
                "data": {}
            }
        
        try:
            # Use Wikipedia API
            url = f"https://it.wikipedia.org/api/rest_v1/page/summary/{search_term.replace(' ', '_')}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                
                if resp.status_code == 200:
                    data = resp.json()
                    title = data.get("title", search_term)
                    extract = data.get("extract", "")
                    
                    if extract:
                        # Truncate if too long
                        if len(extract) > 500:
                            extract = extract[:500] + "..."
                        
                        response = f"📚 **{title}**\n\n{extract}"
                        tts = f"{title}. {extract[:200]}"
                    else:
                        response = f"📚 Ho trovato '{title}' ma non ci sono informazioni dettagliate."
                        tts = f"Ho trovato {title}"
                else:
                    # Try search API
                    search_url = f"https://it.wikipedia.org/w/api.php?action=opensearch&search={search_term}&limit=3&format=json"
                    search_resp = await client.get(search_url)
                    search_data = search_resp.json()
                    
                    if len(search_data) > 1 and search_data[1]:
                        suggestions = ", ".join(search_data[1][:3])
                        response = f"🔍 Non ho trovato '{search_term}'. Forse intendevi: {suggestions}?"
                        tts = f"Non ho trovato risultati esatti. Forse intendevi {search_data[1][0]}?"
                    else:
                        response = f"❓ Non ho trovato informazioni su '{search_term}' su Wikipedia."
                        tts = "Mi dispiace, non ho trovato informazioni"
            
            return {
                "success": True,
                "intent": "wikipedia",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(tts),
                "data": {"search_term": search_term}
            }
            
        except Exception as e:
            return {
                "success": False,
                "intent": "wikipedia",
                "text": f"❌ Errore nella ricerca: {e}",
                "data": {"error": str(e)}
            }
    
    async def _handle_translate(self, text: str) -> Dict[str, Any]:
        """Translate text using LibreTranslate or MyMemory API"""
        import httpx
        
        text_lower = text.lower()
        
        # Detect source and target language
        target_lang = "en"  # Default to English
        source_lang = "it"  # Default from Italian
        
        lang_map = {
            "inglese": "en", "english": "en",
            "italiano": "it", "italian": "it",
            "francese": "fr", "french": "fr",
            "spagnolo": "es", "spanish": "es",
            "tedesco": "de", "german": "de",
            "portoghese": "pt", "portuguese": "pt",
            "russo": "ru", "russian": "ru",
            "cinese": "zh", "chinese": "zh",
            "giapponese": "ja", "japanese": "ja",
        }
        
        for lang_name, lang_code in lang_map.items():
            if f"in {lang_name}" in text_lower:
                target_lang = lang_code
            if f"da {lang_name}" in text_lower or f"dall'{lang_name}" in text_lower:
                source_lang = lang_code
        
        # Extract text to translate
        translate_text = text
        for prefix in ["traduci", "traduzione", "come si dice", "traduci in", 
                       f"in {target_lang}", "tradurre"]:
            if prefix in text_lower:
                parts = text_lower.split(prefix, 1)
                if len(parts) > 1:
                    translate_text = parts[1].strip()
                    # Remove language specifier
                    for lang in lang_map.keys():
                        translate_text = translate_text.replace(f"in {lang}", "").strip()
                    break
        
        translate_text = translate_text.strip("\"'?!.,")
        
        if not translate_text or len(translate_text) < 2:
            return {
                "success": False,
                "intent": "translate",
                "text": "❓ Cosa vuoi tradurre? Esempio: 'traduci ciao in inglese'",
                "data": {}
            }
        
        try:
            # Use MyMemory API (free, no key required)
            url = f"https://api.mymemory.translated.net/get?q={translate_text}&langpair={source_lang}|{target_lang}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                data = resp.json()
                
                if data.get("responseStatus") == 200:
                    translation = data.get("responseData", {}).get("translatedText", "")
                    
                    lang_names = {v: k for k, v in lang_map.items()}
                    target_name = lang_names.get(target_lang, target_lang).capitalize()
                    
                    response = f"🌍 **Traduzione in {target_name}:**\n\n\"{translate_text}\" → \"{translation}\""
                    tts = f"In {target_name}: {translation}"
                else:
                    response = f"❌ Errore nella traduzione"
                    tts = "Mi dispiace, non sono riuscito a tradurre"
            
            return {
                "success": True,
                "intent": "translate",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(tts),
                "data": {"original": translate_text, "translated": translation, "target": target_lang}
            }
            
        except Exception as e:
            return {
                "success": False,
                "intent": "translate",
                "text": f"❌ Errore nella traduzione: {e}",
                "data": {"error": str(e)}
            }
    
    async def _handle_news(self, text: str = "") -> Dict[str, Any]:
        """Get latest news headlines"""
        import httpx
        
        try:
            # Use RSS feeds from Italian news sources
            # Alternative: use a news API
            response = "📰 **Ultime Notizie**\n\n"
            response += "Per le notizie in tempo reale, ti consiglio di visitare:\n"
            response += "• 🇮🇹 [ANSA](https://www.ansa.it)\n"
            response += "• 🇮🇹 [Repubblica](https://www.repubblica.it)\n"
            response += "• 🇮🇹 [Corriere](https://www.corriere.it)\n"
            response += "• 🌍 [Google News](https://news.google.com/topstories?hl=it)\n\n"
            response += "💡 Posso aprire uno di questi siti per te! Dì 'apri ansa' o 'apri google news'"
            
            return {
                "success": True,
                "intent": "news",
                "text": response,
                "tts_text": self.personality.wrap_for_tts("Per le ultime notizie ti consiglio di visitare ANSA o Repubblica. Posso aprirli per te!"),
                "data": {}
            }
            
        except Exception as e:
            return {
                "success": False,
                "intent": "news",
                "text": f"❌ Errore: {e}",
                "data": {"error": str(e)}
            }
    
    async def _handle_currency(self, text: str) -> Dict[str, Any]:
        """Convert currencies using free API"""
        import httpx
        import re
        
        text_lower = text.lower()
        
        # Currency codes
        currencies = {
            "euro": "EUR", "eur": "EUR", "€": "EUR",
            "dollaro": "USD", "dollari": "USD", "usd": "USD", "$": "USD",
            "sterlina": "GBP", "sterline": "GBP", "gbp": "GBP", "£": "GBP",
            "franco": "CHF", "franchi": "CHF", "chf": "CHF",
            "yen": "JPY", "jpy": "JPY", "¥": "JPY",
            "yuan": "CNY", "cny": "CNY",
            "bitcoin": "BTC", "btc": "BTC", "₿": "BTC",
        }
        
        # Extract amount and currencies
        amount = 1.0
        from_currency = "EUR"
        to_currency = "USD"
        
        # Find numbers in text
        numbers = re.findall(r'[\d.,]+', text)
        if numbers:
            try:
                amount = float(numbers[0].replace(',', '.'))
            except:
                amount = 1.0
        
        # Find currencies
        for name, code in currencies.items():
            if name in text_lower:
                if "in" in text_lower:
                    parts = text_lower.split("in")
                    if name in parts[0]:
                        from_currency = code
                    elif name in parts[1]:
                        to_currency = code
                else:
                    from_currency = code
        
        try:
            # Use exchangerate-api.com (free tier)
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                data = resp.json()
                
                if "rates" in data:
                    rate = data["rates"].get(to_currency, 1)
                    result = amount * rate
                    
                    response = f"💱 **Conversione Valuta**\n\n"
                    response += f"{amount:.2f} {from_currency} = **{result:.2f} {to_currency}**\n"
                    response += f"📈 Tasso: 1 {from_currency} = {rate:.4f} {to_currency}"
                    
                    tts = f"{amount:.2f} {from_currency} equivalgono a {result:.2f} {to_currency}"
                else:
                    response = "❌ Impossibile recuperare i tassi di cambio"
                    tts = "Mi dispiace, non riesco a recuperare i tassi di cambio"
            
            return {
                "success": True,
                "intent": "currency",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(tts),
                "data": {"amount": amount, "from": from_currency, "to": to_currency, "result": result if 'result' in dir() else None}
            }
            
        except Exception as e:
            return {
                "success": False,
                "intent": "currency",
                "text": f"❌ Errore nella conversione: {e}",
                "data": {"error": str(e)}
            }
    
    async def _handle_definition(self, text: str) -> Dict[str, Any]:
        """Get word definition"""
        import httpx
        
        text_lower = text.lower()
        
        # Extract word
        word = text
        for prefix in ["definizione di", "definizione", "significa", "significato di", 
                       "significato", "definisci", "cosa significa"]:
            if prefix in text_lower:
                word = text_lower.split(prefix, 1)[-1].strip()
                break
        
        word = word.strip("?!.,\"'")
        
        if not word or len(word) < 2:
            return {
                "success": False,
                "intent": "definition",
                "text": "❓ Di quale parola vuoi la definizione?",
                "data": {}
            }
        
        try:
            # Use Free Dictionary API
            url = f"https://api.dictionaryapi.dev/api/v2/entries/it/{word}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        entry = data[0]
                        meanings = entry.get("meanings", [])
                        
                        response = f"📖 **{word.capitalize()}**\n\n"
                        
                        for meaning in meanings[:2]:  # Max 2 meanings
                            part_of_speech = meaning.get("partOfSpeech", "")
                            definitions = meaning.get("definitions", [])
                            
                            if definitions:
                                response += f"*{part_of_speech}*: {definitions[0].get('definition', '')}\n"
                        
                        tts = f"{word}: {meanings[0]['definitions'][0]['definition']}" if meanings else f"Definizione di {word}"
                    else:
                        response = f"❓ Non ho trovato la definizione di '{word}'"
                        tts = "Non ho trovato la definizione"
                else:
                    # Try English dictionary as fallback
                    url_en = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
                    resp_en = await client.get(url_en)
                    
                    if resp_en.status_code == 200:
                        data = resp_en.json()[0]
                        meanings = data.get("meanings", [])
                        definition = meanings[0]["definitions"][0]["definition"] if meanings else "No definition"
                        
                        response = f"📖 **{word.capitalize()}** (English)\n\n{definition}"
                        tts = f"{word}: {definition}"
                    else:
                        response = f"❓ Non ho trovato la definizione di '{word}'"
                        tts = "Non ho trovato la definizione"
            
            return {
                "success": True,
                "intent": "definition",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(tts),
                "data": {"word": word}
            }
            
        except Exception as e:
            return {
                "success": False,
                "intent": "definition",
                "text": f"❌ Errore: {e}",
                "data": {"error": str(e)}
            }
    
    async def _handle_joke(self) -> Dict[str, Any]:
        """Tell a joke"""
        import random
        
        jokes = [
            "Perché i programmatori preferiscono il buio? Perché la luce attira i bug! 🐛",
            "Come si chiama un pesce senza occhi? Un pssce! 🐟",
            "Cosa dice un bit all'altro? Ci si vede al prossimo byte! 💾",
            "Perché il computer è andato dal dottore? Perché aveva un virus! 🤒",
            "Cosa fa un informatico quando ha fame? Ordina dei cookies! 🍪",
            "Come si chiamano i gatti esperti di computer? Mouse hunter! 🐱",
            "Perché il libro di matematica era triste? Perché aveva troppi problemi! 📚",
            "Cosa dice una stampante stressata? Sono sotto pressione! 🖨️",
            "Perché gli sviluppatori portano gli occhiali? Perché non riescono a C# (vedere nitido)! 👓",
            "Quanti programmatori servono per cambiare una lampadina? Nessuno, è un problema hardware! 💡",
            "Perché il database è andato dallo psicologo? Aveva troppe relazioni complicate! 🗄️",
            "Cosa ha detto il router al computer? Smettila di mandarmi pacchetti! 📦",
        ]
        
        joke = random.choice(jokes)
        
        return {
            "success": True,
            "intent": "joke",
            "text": f"😄 {joke}",
            "tts_text": self.personality.wrap_for_tts(joke),
            "data": {},
            "avatar_expression": "happy"
        }
    
    async def _handle_quote(self) -> Dict[str, Any]:
        """Get an inspirational quote"""
        import random
        
        quotes = [
            ("L'unico modo di fare un ottimo lavoro è amare quello che fai.", "Steve Jobs"),
            ("La semplicità è la sofisticazione suprema.", "Leonardo da Vinci"),
            ("Il futuro appartiene a coloro che credono nella bellezza dei propri sogni.", "Eleanor Roosevelt"),
            ("Non è la specie più forte a sopravvivere, né la più intelligente, ma quella più reattiva ai cambiamenti.", "Charles Darwin"),
            ("Sii il cambiamento che vuoi vedere nel mondo.", "Mahatma Gandhi"),
            ("L'immaginazione è più importante della conoscenza.", "Albert Einstein"),
            ("La vita è quello che ti accade mentre sei impegnato a fare altri progetti.", "John Lennon"),
            ("Il successo non è definitivo, il fallimento non è fatale: è il coraggio di continuare che conta.", "Winston Churchill"),
            ("Ogni grande sogno inizia con un sognatore.", "Harriet Tubman"),
            ("La creatività è l'intelligenza che si diverte.", "Albert Einstein"),
            ("Non smettere mai di imparare, perché la vita non smette mai di insegnare.", "Anonimo"),
            ("Il modo migliore per predire il futuro è crearlo.", "Peter Drucker"),
        ]
        
        quote, author = random.choice(quotes)
        
        response = f"💭 *\"{quote}\"*\n\n— {author}"
        tts = f"{quote}. {author}"
        
        return {
            "success": True,
            "intent": "quote",
            "text": response,
            "tts_text": self.personality.wrap_for_tts(tts),
            "data": {"quote": quote, "author": author},
            "avatar_expression": "thinking"
        }
    
    async def _handle_weather_forecast(self, text: str) -> Dict[str, Any]:
        """Get extended weather forecast"""
        import httpx
        
        text_lower = text.lower()
        
        # Extract city
        cities = {
            "roma": (41.9028, 12.4964, "Roma"),
            "milano": (45.4642, 9.1900, "Milano"),
            "napoli": (40.8518, 14.2681, "Napoli"),
            "torino": (45.0703, 7.6869, "Torino"),
            "firenze": (43.7696, 11.2558, "Firenze"),
            "bologna": (44.4949, 11.3426, "Bologna"),
            "venezia": (45.4408, 12.3155, "Venezia"),
            "palermo": (38.1157, 13.3615, "Palermo"),
        }
        
        city_name = "Roma"
        lat, lon = 41.9028, 12.4964
        
        for city_key, (city_lat, city_lon, city_display) in cities.items():
            if city_key in text_lower:
                lat, lon = city_lat, city_lon
                city_name = city_display
                break
        
        try:
            # Get 7-day forecast
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=Europe/Rome"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                data = resp.json()
            
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            weather_codes = daily.get("weather_code", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            rain_probs = daily.get("precipitation_probability_max", [])
            
            weather_emojis = {
                0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️", 
                45: "🌫️", 48: "🌫️",
                51: "🌧️", 53: "🌧️", 55: "🌧️",
                61: "🌧️", 63: "🌧️", 65: "🌧️",
                71: "❄️", 73: "❄️", 75: "❄️",
                95: "⛈️", 96: "⛈️", 99: "⛈️",
            }
            
            giorni = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
            
            response = f"📅 **Previsioni per {city_name}**\n\n"
            
            for i in range(min(5, len(dates))):  # Next 5 days
                date = dates[i]
                emoji = weather_emojis.get(weather_codes[i], "🌡️")
                max_t = max_temps[i] if i < len(max_temps) else "?"
                min_t = min_temps[i] if i < len(min_temps) else "?"
                rain = rain_probs[i] if i < len(rain_probs) else 0
                
                # Parse date to get day name
                from datetime import datetime as dt
                day_date = dt.strptime(date, "%Y-%m-%d")
                day_name = giorni[day_date.weekday()]
                
                response += f"{emoji} **{day_name} {day_date.day}/{day_date.month}**: {min_t}° - {max_t}°"
                if rain > 30:
                    response += f" 🌧️{rain}%"
                response += "\n"
            
            tts = f"Previsioni per {city_name}: domani previste massime di {max_temps[1] if len(max_temps) > 1 else max_temps[0]} gradi"
            
            return {
                "success": True,
                "intent": "weather_forecast",
                "text": response,
                "tts_text": self.personality.wrap_for_tts(tts),
                "data": {"city": city_name, "forecast": daily}
            }
            
        except Exception as e:
            return {
                "success": False,
                "intent": "weather_forecast",
                "text": f"❌ Errore nelle previsioni: {e}",
                "data": {"error": str(e)}
            }
    
    # ============== FINE COMANDI INFORMATIVI ==============

    async def _handle_pilot_command(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle pilot mode commands with enhanced capabilities
        Pilot mode can execute complex actions and control the system
        ALL critical actions require internal logical confirmation
        """
        import psutil
        import subprocess
        
        logger.info(f"🚀 PILOT COMMAND: {text}")
        text_lower = text.lower()
        
        # Parse pilot commands
        if any(word in text_lower for word in ["apri", "avvia", "lancia", "esegui app"]):
            # Application launch commands
            return await self._pilot_launch_app(text)
        
        elif any(word in text_lower for word in ["cerca", "trova", "ricerca"]):
            # Search commands
            return await self._pilot_search(text)
        
        elif any(word in text_lower for word in ["routine", "automazione", "automatizza", "esegui routine"]):
            # Automation/routine commands
            return await self._pilot_execute_routine(text)
        
        elif any(word in text_lower for word in ["calcola", "percentuale", "quanto", "calcolo"]):
            # Calculation and percentage commands
            return await self._pilot_calculate(text)
        
        elif any(word in text_lower for word in ["proponi", "suggerisci", "decisione", "consiglia"]):
            # Decision proposal
            return await self._pilot_propose_decision(text)
        
        elif any(word in text_lower for word in ["analizza", "scansiona", "controlla"]):
            # Deep analysis with internal confirmation
            confirmation = await self._pilot_internal_confirm("analisi approfondita del sistema")
            if not confirmation["approved"]:
                return self._pilot_confirmation_denied(confirmation)
            
            result = await self._handle_system_status()
            result["pilot_mode"] = True
            result["internal_confirmation"] = confirmation
            result["text"] = "🚀 PILOT: " + result["text"]
            return result
        
        elif any(word in text_lower for word in ["ottimizza", "migliora", "velocizza"]):
            # Optimization with confirmation for critical actions
            return await self._pilot_optimize_advanced(text)
        
        elif any(word in text_lower for word in ["stato", "status", "report"]):
            # System report
            result = await self._handle_system_status()
            
            # Add extra pilot info
            processes = len(psutil.pids())
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            
            result["data"]["pilot_info"] = {
                "active_processes": processes,
                "uptime_hours": round(uptime.total_seconds() / 3600, 1),
                "boot_time": boot_time.isoformat()
            }
            result["pilot_mode"] = True
            result["text"] = f"🚀 PILOT REPORT: {result['text']} Processi attivi: {processes}. Sistema attivo da {round(uptime.total_seconds() / 3600, 1)} ore."
            return result
        
        elif any(word in text_lower for word in ["che ora", "ora", "orario"]):
            result = await self._handle_time()
            result["pilot_mode"] = True
            result["text"] = "🚀 " + result["text"]
            return result
        
        elif any(word in text_lower for word in ["comanda", "controlla tutto", "gestisci", "prendi il controllo"]):
            # Full app control mode
            return await self._pilot_full_control(text)
        
        elif any(word in text_lower for word in ["chatgpt", "chiedi a gpt", "chiedi all'ai", "chiedi a openai", "domanda a chatgpt"]):
            # Direct ChatGPT query
            return await self._pilot_ask_chatgpt(text)
        
        elif any(word in text_lower for word in ["perplexity", "cerca con perplexity", "chiedi a perplexity"]):
            # Perplexity AI search
            return await self._pilot_ask_perplexity(text)
        
        elif self._is_question(text_lower):
            # If it looks like a question, search for answer via AI
            return await self._pilot_answer_question(text)
        
        else:
            # Use reasoning for complex queries in pilot mode
            return await self._pilot_reason(text)
    
    def _is_question(self, text: str) -> bool:
        """Check if text is a question that should be answered via AI search"""
        question_indicators = [
            "?", "come ", "cosa ", "perché ", "quando ", "dove ", "chi ", "quale ",
            "quanto ", "cos'è", "che cos'è", "spiega", "spiegami", "dimmi",
            "what ", "how ", "why ", "when ", "where ", "who ", "which ",
            "explain", "tell me"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators)
    
    async def _pilot_launch_app(self, text: str) -> Dict[str, Any]:
        """Launch applications in pilot mode"""
        import subprocess
        
        text_lower = text.lower()
        
        # Map of common app names to executables
        app_map = {
            "browser": "start chrome",
            "chrome": "start chrome",
            "firefox": "start firefox",
            "notepad": "notepad",
            "blocco note": "notepad",
            "calcolatrice": "calc",
            "calculator": "calc",
            "esplora": "explorer",
            "explorer": "explorer",
            "terminale": "cmd",
            "terminal": "cmd",
            "powershell": "powershell"
        }
        
        launched = None
        for app_name, command in app_map.items():
            if app_name in text_lower:
                try:
                    subprocess.Popen(command, shell=True)
                    launched = app_name
                    break
                except Exception as e:
                    logger.error(f"Failed to launch {app_name}: {e}")
        
        if launched:
            response_text = f"🚀 PILOT: Ho avviato {launched}."
            return {
                "success": True,
                "intent": "pilot_launch",
                "pilot_mode": True,
                "text": response_text,
                "tts_text": self.personality.wrap_for_tts(f"Avviato {launched}."),
                "data": {"launched": launched}
            }
        else:
            response_text = f"🚀 PILOT: App non trovata. Opzioni: browser, notepad, calcolatrice, terminale."
            return {
                "success": False,
                "intent": "pilot_launch",
                "pilot_mode": True,
                "text": response_text,
                "tts_text": self.personality.wrap_for_tts("App non trovata. Specifica."),
                "data": {}
            }
    
    async def _pilot_search(self, text: str) -> Dict[str, Any]:
        """Handle search commands in pilot mode"""
        import webbrowser
        
        # Extract search query
        search_terms = text.lower()
        for prefix in ["cerca", "trova", "ricerca"]:
            if prefix in search_terms:
                search_terms = search_terms.split(prefix, 1)[-1].strip()
                break
        
        if search_terms:
            # Open web search
            search_url = f"https://www.google.com/search?q={search_terms.replace(' ', '+')}"
            webbrowser.open(search_url)
            
            response_text = f"🚀 PILOT: Ricerca aperta per '{search_terms}'."
            return {
                "success": True,
                "intent": "pilot_search",
                "pilot_mode": True,
                "text": response_text,
                "tts_text": self.personality.wrap_for_tts(f"Ricerca avviata per {search_terms}."),
                "data": {"query": search_terms, "url": search_url}
            }
        
        return {
            "success": False,
            "intent": "pilot_search",
            "pilot_mode": True,
            "text": "🚀 PILOT: Specifica cosa cercare.",
            "tts_text": self.personality.wrap_for_tts("Specifica cosa cercare."),
            "data": {}
        }
    
    async def _pilot_optimize(self, text: str) -> Dict[str, Any]:
        """Execute optimization in pilot mode"""
        optimizations = await self.optimizer.get_optimizations("system")
        
        # In pilot mode, try to apply optimizations
        applied = []
        for opt in optimizations[:3]:  # Apply top 3
            logger.info(f"🚀 PILOT applying optimization: {opt.get('name', 'unknown')}")
            applied.append(opt.get("name", "optimization"))
        
        if applied:
            return {
                "success": True,
                "intent": "pilot_optimize",
                "pilot_mode": True,
                "text": f"🚀 PILOT: Ho analizzato il sistema. Ottimizzazioni disponibili: {', '.join(applied)}.",
                "data": {"optimizations": optimizations}
            }
        
        return {
            "success": True,
            "intent": "pilot_optimize",
            "pilot_mode": True,
            "text": "🚀 PILOT: Il sistema è già ottimizzato. Nessun intervento necessario.",
            "data": {}
        }
    
    async def _pilot_reason(self, text: str) -> Dict[str, Any]:
        """
        Use autonomous reasoning in pilot mode.
        
        STEP 9 ENHANCEMENT:
        - Ragiona autonomamente PRIMA di rispondere
        - Esegue automaticamente le azioni se richieste
        - Risponde SOLO al completamento del ragionamento
        """
        logger.info(f"🚀 PILOT reasoning on: {text}")
        
        # ========== FASE 1: RAGIONAMENTO COMPLETO ==========
        # Non risponde finché il ragionamento non è completo
        thinking_result = await self.reasoning.autonomous_think(
            topic=text,
            context={"mode": "pilot", "auto_execute": True},
            depth=4  # Deep thinking in pilot mode
        )
        
        # ========== FASE 2: ANALISI AZIONI RICHIESTE ==========
        conclusion = thinking_result.get("conclusion", {})
        recommendations = conclusion.get("recommendations", [])
        actions_executed = []
        
        # Check if any actions should be auto-executed
        text_lower = text.lower()
        
        # Auto-execute safe actions in pilot mode
        if any(w in text_lower for w in ["esegui", "fai", "avvia", "attiva", "lancia"]):
            # Try to identify and execute the action
            action_result = await self._pilot_auto_execute_action(text, recommendations)
            if action_result["executed"]:
                actions_executed.append(action_result)
        
        # ========== FASE 3: FORMATTAZIONE RISPOSTA FINALE ==========
        response_parts = []
        
        # Main conclusion
        main_response = thinking_result.get("response", conclusion.get("statement", "Comando elaborato."))
        response_parts.append(main_response)
        
        # Add executed actions info
        if actions_executed:
            response_parts.append("\n\n⚡ **Azioni eseguite automaticamente:**")
            for action in actions_executed:
                response_parts.append(f"  • {action['description']}")
        
        # Add recommendations if not executed
        remaining_recs = [r for r in recommendations if not any(r.lower() in str(a).lower() for a in actions_executed)]
        if remaining_recs:
            response_parts.append("\n\n💡 **Suggerimenti:**")
            for i, rec in enumerate(remaining_recs[:3], 1):
                response_parts.append(f"  {i}. {rec}")
        
        final_response = "\n".join(response_parts)
        
        # TTS version (concise)
        tts_response = main_response
        if actions_executed:
            tts_response += f" Ho eseguito automaticamente {len(actions_executed)} azioni."
        
        return {
            "success": True,
            "intent": "pilot_reason",
            "pilot_mode": True,
            "text": f"🚀 PILOT: {final_response}",
            "tts_text": self.personality.wrap_for_tts(tts_response),
            "data": {
                "thinking_result": thinking_result,
                "actions_executed": actions_executed,
                "thinking_time": thinking_result.get("thinking_time_seconds", 0),
                "reasoning_complete": True
            }
        }
    
    async def _pilot_auto_execute_action(self, command: str, recommendations: list) -> Dict[str, Any]:
        """
        Automatically execute actions in pilot mode.
        Only executes safe, non-destructive actions.
        """
        import subprocess
        import webbrowser
        
        command_lower = command.lower()
        executed = False
        description = ""
        result = None
        
        # Safe action mappings
        if "browser" in command_lower or "chrome" in command_lower:
            subprocess.Popen("start chrome", shell=True)
            executed = True
            description = "Avviato browser Chrome"
        
        elif "notepad" in command_lower or "blocco note" in command_lower:
            subprocess.Popen("notepad", shell=True)
            executed = True
            description = "Aperto Blocco Note"
        
        elif "calcolatrice" in command_lower or "calculator" in command_lower:
            subprocess.Popen("calc", shell=True)
            executed = True
            description = "Aperta Calcolatrice"
        
        elif "cerca" in command_lower:
            # Extract search term
            search_term = command_lower.split("cerca", 1)[-1].strip()
            if search_term:
                url = f"https://www.google.com/search?q={search_term.replace(' ', '+')}"
                webbrowser.open(url)
                executed = True
                description = f"Ricerca avviata per: {search_term}"
                result = {"url": url}
        
        elif "apri" in command_lower and ("sito" in command_lower or "pagina" in command_lower):
            # Extract URL from recommendations or command
            for rec in recommendations:
                if "http" in rec.lower():
                    import re
                    urls = re.findall(r'https?://\S+', rec)
                    if urls:
                        webbrowser.open(urls[0])
                        executed = True
                        description = f"Aperto sito: {urls[0]}"
                        result = {"url": urls[0]}
                        break
        
        return {
            "executed": executed,
            "description": description,
            "result": result
        }
    
    # ========== PILOT MODE ADVANCED METHODS ==========
    
    async def _pilot_internal_confirm(self, action: str) -> Dict[str, Any]:
        """
        Internal logical confirmation for critical actions.
        Gideon uses reasoning to validate if an action should proceed.
        """
        logger.info(f"🔐 PILOT CONFIRMATION: Evaluating action '{action}'")
        
        # Use reasoning engine to evaluate the action
        evaluation_prompt = f"""
        Valuta se questa azione è sicura e appropriata da eseguire: "{action}"
        
        Considera:
        1. L'azione è reversibile?
        2. Può causare perdita di dati?
        3. Richiede privilegi elevati?
        4. È nell'interesse dell'utente?
        
        Rispondi con una valutazione breve.
        """
        
        thinking_result = await self.reasoning.autonomous_think(evaluation_prompt)
        
        # For now, approve most non-destructive actions
        # In production, this would have more sophisticated logic
        dangerous_words = ["elimina", "cancella", "formatta", "distruggi", "rimuovi tutto"]
        is_dangerous = any(word in action.lower() for word in dangerous_words)
        
        confirmation = {
            "action": action,
            "approved": not is_dangerous,
            "reasoning": thinking_result.get("response", "Azione valutata"),
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.95 if not is_dangerous else 0.3
        }
        
        if is_dangerous:
            confirmation["reason_denied"] = "Azione potenzialmente distruttiva rilevata"
            logger.warning(f"🚫 PILOT DENIED: {action}")
        else:
            logger.info(f"✅ PILOT APPROVED: {action}")
        
        return confirmation
    
    def _pilot_confirmation_denied(self, confirmation: Dict) -> Dict[str, Any]:
        """Return denial response for unapproved actions"""
        return {
            "success": False,
            "intent": "pilot_denied",
            "pilot_mode": True,
            "text": f"🚫 PILOT: Ho analizzato la richiesta '{confirmation['action']}' ma non posso eseguirla. {confirmation.get('reason_denied', 'Richiesta conferma manuale.')}",
            "data": confirmation
        }
    
    async def _pilot_execute_routine(self, text: str) -> Dict[str, Any]:
        """
        Execute automation routines.
        Routines are predefined multi-step automations.
        """
        import psutil
        
        logger.info(f"⚙️ PILOT ROUTINE: {text}")
        
        # Available routines
        routines = {
            "pulizia": {
                "name": "Pulizia Sistema",
                "steps": [
                    "Analisi file temporanei",
                    "Identificazione cache obsolete", 
                    "Report spazio recuperabile"
                ],
                "action": self._routine_cleanup
            },
            "performance": {
                "name": "Ottimizzazione Performance",
                "steps": [
                    "Analisi processi attivi",
                    "Identificazione processi pesanti",
                    "Suggerimenti ottimizzazione"
                ],
                "action": self._routine_performance
            },
            "monitoraggio": {
                "name": "Monitoraggio Continuo",
                "steps": [
                    "Attivazione sensori sistema",
                    "Raccolta metriche real-time",
                    "Report stato continuo"
                ],
                "action": self._routine_monitor
            },
            "backup": {
                "name": "Backup Rapido",
                "steps": [
                    "Identificazione file critici",
                    "Verifica spazio disponibile",
                    "Preparazione backup"
                ],
                "action": self._routine_backup_check
            },
            "sicurezza": {
                "name": "Check Sicurezza",
                "steps": [
                    "Verifica connessioni attive",
                    "Analisi porte aperte",
                    "Report sicurezza"
                ],
                "action": self._routine_security
            }
        }
        
        # Find requested routine
        text_lower = text.lower()
        selected_routine = None
        
        for key, routine in routines.items():
            if key in text_lower or routine["name"].lower() in text_lower:
                selected_routine = (key, routine)
                break
        
        if selected_routine:
            key, routine = selected_routine
            
            # Request internal confirmation
            confirmation = await self._pilot_internal_confirm(f"esecuzione routine {routine['name']}")
            if not confirmation["approved"]:
                return self._pilot_confirmation_denied(confirmation)
            
            # Execute routine
            result = await routine["action"]()
            
            return {
                "success": True,
                "intent": "pilot_routine",
                "pilot_mode": True,
                "text": f"🚀 PILOT ROUTINE [{routine['name']}]: {result['summary']}",
                "data": {
                    "routine": routine["name"],
                    "steps_executed": routine["steps"],
                    "result": result,
                    "confirmation": confirmation
                }
            }
        else:
            # List available routines
            routine_list = ", ".join([r["name"] for r in routines.values()])
            return {
                "success": True,
                "intent": "pilot_routine_list",
                "pilot_mode": True,
                "text": f"🚀 PILOT: Routine disponibili: {routine_list}. Quale vuoi eseguire?",
                "data": {"available_routines": list(routines.keys())}
            }
    
    async def _routine_cleanup(self) -> Dict[str, Any]:
        """Cleanup routine - analyze temp files"""
        import os
        import tempfile
        
        temp_dir = tempfile.gettempdir()
        temp_files = []
        total_size = 0
        
        try:
            for item in os.listdir(temp_dir)[:50]:  # Limit to 50 items
                item_path = os.path.join(temp_dir, item)
                try:
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path)
                        total_size += size
                        temp_files.append({"name": item, "size_kb": round(size/1024, 2)})
                except:
                    pass
        except Exception as e:
            logger.error(f"Cleanup routine error: {e}")
        
        return {
            "summary": f"Trovati {len(temp_files)} file temporanei. Spazio recuperabile: {round(total_size/(1024*1024), 2)} MB",
            "temp_files_count": len(temp_files),
            "recoverable_mb": round(total_size/(1024*1024), 2)
        }
    
    async def _routine_performance(self) -> Dict[str, Any]:
        """Performance routine - analyze heavy processes"""
        import psutil
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                if info['cpu_percent'] and info['cpu_percent'] > 1:
                    processes.append(info)
            except:
                pass
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
        top_processes = processes[:5]
        
        summary_parts = []
        for p in top_processes:
            summary_parts.append(f"{p['name']} ({p['cpu_percent']:.1f}% CPU)")
        
        return {
            "summary": f"Top processi per CPU: {', '.join(summary_parts) if summary_parts else 'nessun processo significativo'}",
            "top_processes": top_processes,
            "total_analyzed": len(processes)
        }
    
    async def _routine_monitor(self) -> Dict[str, Any]:
        """Monitoring routine - real-time metrics"""
        import psutil
        
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()
        
        return {
            "summary": f"CPU: {cpu}% | RAM: {mem.percent}% | Disco: {disk.percent}% | Rete TX: {round(net.bytes_sent/(1024*1024), 1)}MB",
            "metrics": {
                "cpu_percent": cpu,
                "memory_percent": mem.percent,
                "disk_percent": disk.percent,
                "network_sent_mb": round(net.bytes_sent/(1024*1024), 2),
                "network_recv_mb": round(net.bytes_recv/(1024*1024), 2)
            }
        }
    
    async def _routine_backup_check(self) -> Dict[str, Any]:
        """Backup check routine - analyze critical files"""
        import os
        
        home = os.path.expanduser("~")
        documents = os.path.join(home, "Documents")
        
        doc_count = 0
        doc_size = 0
        
        try:
            if os.path.exists(documents):
                for root, dirs, files in os.walk(documents):
                    doc_count += len(files)
                    for f in files:
                        try:
                            doc_size += os.path.getsize(os.path.join(root, f))
                        except:
                            pass
                    if doc_count > 1000:  # Limit scan
                        break
        except Exception as e:
            logger.error(f"Backup check error: {e}")
        
        return {
            "summary": f"Documenti trovati: {doc_count}. Dimensione totale: {round(doc_size/(1024*1024), 2)} MB. Pronto per backup.",
            "documents_count": doc_count,
            "size_mb": round(doc_size/(1024*1024), 2)
        }
    
    async def _routine_security(self) -> Dict[str, Any]:
        """Security check routine - analyze connections"""
        import psutil
        
        connections = psutil.net_connections(kind='inet')
        established = [c for c in connections if c.status == 'ESTABLISHED']
        listening = [c for c in connections if c.status == 'LISTEN']
        
        # Get unique remote addresses
        remote_ips = set()
        for conn in established:
            if conn.raddr:
                remote_ips.add(conn.raddr.ip)
        
        return {
            "summary": f"Connessioni attive: {len(established)}. Porte in ascolto: {len(listening)}. IP remoti unici: {len(remote_ips)}.",
            "established_connections": len(established),
            "listening_ports": len(listening),
            "unique_remote_ips": len(remote_ips)
        }
    
    async def _pilot_calculate(self, text: str) -> Dict[str, Any]:
        """
        Handle calculations and percentages.
        Can perform system-related calculations.
        """
        import psutil
        import re
        
        logger.info(f"🧮 PILOT CALCULATE: {text}")
        text_lower = text.lower()
        
        result_data = {}
        result_text = ""
        
        # Check for specific calculation types
        if any(word in text_lower for word in ["cpu", "processore", "utilizzo"]):
            cpu = psutil.cpu_percent(interval=1)
            result_text = f"Utilizzo CPU attuale: {cpu}%. "
            if cpu < 30:
                result_text += f"Capacità disponibile: {100-cpu}% - sistema leggero."
            elif cpu < 70:
                result_text += f"Capacità disponibile: {100-cpu}% - carico moderato."
            else:
                result_text += f"Capacità disponibile: {100-cpu}% - carico elevato, considera ottimizzazione."
            
            result_data = {"cpu_percent": cpu, "available_percent": 100-cpu}
        
        elif any(word in text_lower for word in ["memoria", "ram", "memory"]):
            mem = psutil.virtual_memory()
            result_text = f"RAM utilizzata: {mem.percent}%. "
            result_text += f"Disponibile: {round(mem.available/(1024**3), 2)} GB su {round(mem.total/(1024**3), 2)} GB totali. "
            result_text += f"Efficienza memoria: {round((1 - mem.percent/100) * 100, 1)}%"
            
            result_data = {
                "used_percent": mem.percent,
                "available_gb": round(mem.available/(1024**3), 2),
                "total_gb": round(mem.total/(1024**3), 2),
                "efficiency": round((1 - mem.percent/100) * 100, 1)
            }
        
        elif any(word in text_lower for word in ["disco", "spazio", "storage"]):
            disk = psutil.disk_usage('/')
            result_text = f"Disco utilizzato: {disk.percent}%. "
            result_text += f"Spazio libero: {round(disk.free/(1024**3), 2)} GB su {round(disk.total/(1024**3), 2)} GB. "
            
            if disk.percent > 80:
                result_text += "⚠️ Consiglio pulizia disco."
            else:
                result_text += "Spazio sufficiente."
            
            result_data = {
                "used_percent": disk.percent,
                "free_gb": round(disk.free/(1024**3), 2),
                "total_gb": round(disk.total/(1024**3), 2)
            }
        
        elif any(word in text_lower for word in ["efficienza", "prestazioni", "performance"]):
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate overall efficiency score
            cpu_score = 100 - cpu
            mem_score = 100 - mem.percent
            disk_score = 100 - disk.percent
            
            overall = (cpu_score * 0.4 + mem_score * 0.4 + disk_score * 0.2)
            
            result_text = f"Efficienza sistema complessiva: {overall:.1f}%. "
            result_text += f"(CPU: {cpu_score:.0f}%, RAM: {mem_score:.0f}%, Disco: {disk_score:.0f}%)"
            
            if overall > 70:
                result_text += " - Sistema performante! ✅"
            elif overall > 40:
                result_text += " - Prestazioni moderate. Alcune ottimizzazioni consigliate."
            else:
                result_text += " - ⚠️ Prestazioni basse. Ottimizzazione necessaria."
            
            result_data = {
                "overall_efficiency": round(overall, 1),
                "cpu_score": round(cpu_score, 1),
                "memory_score": round(mem_score, 1),
                "disk_score": round(disk_score, 1)
            }
        
        else:
            # Try to extract numbers for basic calculations
            numbers = re.findall(r'\d+\.?\d*', text)
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                
                if "percentuale" in text_lower or "%" in text:
                    pct = (a / b) * 100 if b != 0 else 0
                    result_text = f"{a} è il {pct:.2f}% di {b}."
                    result_data = {"value": a, "total": b, "percentage": round(pct, 2)}
                else:
                    result_text = f"Numeri trovati: {a} e {b}. Somma: {a+b}, Differenza: {a-b}, Prodotto: {a*b}"
                    if b != 0:
                        result_text += f", Divisione: {a/b:.2f}"
                    result_data = {"a": a, "b": b, "sum": a+b, "diff": a-b, "product": a*b}
            else:
                result_text = "Dimmi cosa vuoi calcolare: utilizzo CPU, memoria, disco, efficienza sistema, o fornisci dei numeri."
                result_data = {"available_calculations": ["cpu", "memoria", "disco", "efficienza", "percentuale"]}
        
        return {
            "success": True,
            "intent": "pilot_calculate",
            "pilot_mode": True,
            "text": f"🧮 PILOT: {result_text}",
            "data": result_data
        }
    
    async def _pilot_propose_decision(self, text: str) -> Dict[str, Any]:
        """
        Propose decisions based on system analysis.
        Uses reasoning to make smart suggestions.
        """
        import psutil
        
        logger.info(f"💡 PILOT DECISION: {text}")
        
        # Gather system state for decision making
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        decisions = []
        
        # CPU-based decisions
        if cpu > 80:
            decisions.append({
                "area": "CPU",
                "problem": f"Utilizzo CPU elevato ({cpu}%)",
                "suggestion": "Chiudere applicazioni non essenziali o identificare processi pesanti",
                "priority": "alta",
                "action": "routine performance"
            })
        elif cpu < 10:
            decisions.append({
                "area": "CPU",
                "observation": f"CPU quasi inattiva ({cpu}%)",
                "suggestion": "Sistema pronto per task intensivi",
                "priority": "info"
            })
        
        # Memory-based decisions
        if mem.percent > 80:
            decisions.append({
                "area": "Memoria",
                "problem": f"RAM quasi esaurita ({mem.percent}%)",
                "suggestion": "Liberare memoria chiudendo applicazioni o aumentare swap",
                "priority": "alta"
            })
        elif mem.percent > 60:
            decisions.append({
                "area": "Memoria",
                "observation": f"Memoria in uso moderato ({mem.percent}%)",
                "suggestion": "Monitorare l'utilizzo, evitare di aprire troppe applicazioni",
                "priority": "media"
            })
        
        # Disk-based decisions
        if disk.percent > 90:
            decisions.append({
                "area": "Disco",
                "problem": f"Spazio disco critico ({disk.percent}%)",
                "suggestion": "Eseguire pulizia disco immediatamente",
                "priority": "critica",
                "action": "routine pulizia"
            })
        elif disk.percent > 75:
            decisions.append({
                "area": "Disco",
                "observation": f"Spazio disco in esaurimento ({disk.percent}%)",
                "suggestion": "Pianificare pulizia disco a breve",
                "priority": "media"
            })
        
        # Use reasoning for additional insights
        if text and len(text) > 20:
            reasoning = await self.reasoning.autonomous_think(
                f"Analizza questa richiesta e proponi una decisione: {text}"
            )
            if reasoning.get("response"):
                decisions.append({
                    "area": "Analisi Richiesta",
                    "suggestion": reasoning["response"],
                    "priority": "info"
                })
        
        # Build response
        if decisions:
            high_priority = [d for d in decisions if d.get("priority") in ["alta", "critica"]]
            
            response_parts = []
            for d in decisions[:3]:  # Top 3 decisions
                if "problem" in d:
                    response_parts.append(f"[{d['area']}] ⚠️ {d['problem']}: {d['suggestion']}")
                else:
                    response_parts.append(f"[{d['area']}] {d.get('observation', '')}: {d['suggestion']}")
            
            result_text = " | ".join(response_parts)
            
            if high_priority:
                result_text = "🚨 AZIONI CONSIGLIATE: " + result_text
            else:
                result_text = "💡 ANALISI: " + result_text
        else:
            result_text = "Sistema in buone condizioni. Nessuna azione urgente necessaria. Continuo il monitoraggio."
            decisions = [{"area": "Generale", "suggestion": "Tutto ok", "priority": "info"}]
        
        return {
            "success": True,
            "intent": "pilot_decision",
            "pilot_mode": True,
            "text": f"🚀 PILOT: {result_text}",
            "data": {
                "decisions": decisions,
                "system_state": {
                    "cpu": cpu,
                    "memory": mem.percent,
                    "disk": disk.percent
                }
            }
        }
    
    async def _pilot_optimize_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced optimization with internal confirmation"""
        import psutil
        
        # First, confirm the optimization action
        confirmation = await self._pilot_internal_confirm("ottimizzazione avanzata del sistema")
        if not confirmation["approved"]:
            return self._pilot_confirmation_denied(confirmation)
        
        # Get system state
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        
        optimizations_applied = []
        optimizations_suggested = []
        
        # Analyze and suggest/apply optimizations
        if cpu > 70:
            heavy_procs = []
            for proc in psutil.process_iter(['name', 'cpu_percent']):
                try:
                    if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 10:
                        heavy_procs.append(proc.info['name'])
                except:
                    pass
            
            if heavy_procs:
                optimizations_suggested.append({
                    "type": "cpu",
                    "action": f"Considera chiudere: {', '.join(heavy_procs[:3])}",
                    "impact": "alto"
                })
        
        if mem.percent > 70:
            optimizations_suggested.append({
                "type": "memory",
                "action": "Liberare memoria: chiudere tab browser o app inutilizzate",
                "impact": "medio"
            })
        
        # Get optimizer suggestions
        opt_result = await self.optimizer.get_optimizations("performance")
        
        if optimizations_suggested:
            suggestions = "; ".join([o["action"] for o in optimizations_suggested])
            result_text = f"Analisi completata. Suggerimenti: {suggestions}"
        else:
            result_text = "Sistema già ottimizzato. Prestazioni al massimo."
        
        return {
            "success": True,
            "intent": "pilot_optimize_advanced",
            "pilot_mode": True,
            "text": f"🚀 PILOT OPTIMIZATION: {result_text}",
            "data": {
                "optimizations_suggested": optimizations_suggested,
                "optimizations_applied": optimizations_applied,
                "confirmation": confirmation,
                "current_state": {"cpu": cpu, "memory": mem.percent}
            }
        }
    
    async def _pilot_full_control(self, text: str) -> Dict[str, Any]:
        """
        Full app control mode - Gideon takes autonomous control
        """
        logger.info("🎮 PILOT FULL CONTROL ACTIVATED")
        
        # Confirm full control mode
        confirmation = await self._pilot_internal_confirm("attivazione controllo completo")
        if not confirmation["approved"]:
            return self._pilot_confirmation_denied(confirmation)
        
        # Gather all system info
        status = await self._handle_system_status()
        decisions = await self._pilot_propose_decision("valuta stato generale")
        
        response_text = """Controllo completo attivato. Sto monitorando attivamente:
        
• Sistema: Operativo ✅
• CPU/RAM/Disco: Sotto controllo
• Decisioni autonome: Abilitate
• Conferma interna: Attiva per azioni critiche

Sono pronto ad eseguire qualsiasi comando. Dimmi cosa devo fare e me ne occuperò io."""

        return {
            "success": True,
            "intent": "pilot_full_control",
            "pilot_mode": True,
            "text": f"🚀 PILOT: {response_text}",
            "data": {
                "control_active": True,
                "confirmation": confirmation,
                "system_status": status.get("data", {}),
                "pending_decisions": decisions.get("data", {}).get("decisions", [])
            }
        }
    
    # ========== AI SEARCH METHODS ==========
    
    async def _pilot_ask_chatgpt(self, text: str) -> Dict[str, Any]:
        """
        Ask ChatGPT a question - uses API if available, otherwise browser
        """
        logger.info(f"🤖 PILOT ChatGPT query: {text}")
        
        # Extract the actual question
        question = text.lower()
        prefixes = ["chatgpt", "chiedi a gpt", "chiedi all'ai", "chiedi a openai", "domanda a chatgpt"]
        for prefix in prefixes:
            if prefix in question:
                question = question.split(prefix, 1)[-1].strip()
                break
        
        if not question or len(question) < 3:
            return {
                "success": False,
                "intent": "pilot_chatgpt",
                "pilot_mode": True,
                "text": "🤖 PILOT: Cosa vuoi chiedere a ChatGPT?",
                "data": {}
            }
        
        # Use AI search service
        result = await self.ai_search.search_answer(question)
        
        if result["success"]:
            if result["source"] == "openai":
                return {
                    "success": True,
                    "intent": "pilot_chatgpt",
                    "pilot_mode": True,
                    "text": f"🤖 PILOT (ChatGPT): {result['answer']}",
                    "data": {
                        "source": "openai_api",
                        "model": result.get("model", "unknown"),
                        "usage": result.get("usage", {})
                    }
                }
            else:
                return {
                    "success": True,
                    "intent": "pilot_chatgpt",
                    "pilot_mode": True,
                    "text": f"🌐 PILOT: {result['answer']}",
                    "data": {
                        "source": "browser",
                        "url": result.get("url", ""),
                        "clipboard_copied": result.get("clipboard_copied", False)
                    }
                }
        
        return {
            "success": False,
            "intent": "pilot_chatgpt",
            "pilot_mode": True,
            "text": f"🤖 PILOT: Non sono riuscito a ottenere una risposta. {result.get('error', '')}",
            "data": result
        }
    
    async def _pilot_ask_perplexity(self, text: str) -> Dict[str, Any]:
        """
        Ask Perplexity AI - opens browser with web search
        """
        logger.info(f"🔍 PILOT Perplexity query: {text}")
        
        # Extract the question
        question = text.lower()
        prefixes = ["perplexity", "cerca con perplexity", "chiedi a perplexity"]
        for prefix in prefixes:
            if prefix in question:
                question = question.split(prefix, 1)[-1].strip()
                break
        
        if not question or len(question) < 3:
            return {
                "success": False,
                "intent": "pilot_perplexity",
                "pilot_mode": True,
                "text": "🔍 PILOT: Cosa vuoi cercare con Perplexity?",
                "data": {}
            }
        
        result = await self.ai_search.ask_perplexity(question)
        
        return {
            "success": True,
            "intent": "pilot_perplexity",
            "pilot_mode": True,
            "text": f"🔍 PILOT: {result['answer']}",
            "data": {
                "source": "perplexity",
                "url": result.get("url", ""),
                "question": question
            }
        }
    
    async def _pilot_answer_question(self, text: str) -> Dict[str, Any]:
        """
        Answer a question using AI search.
        Automatically triggered when pilot mode detects a question.
        """
        logger.info(f"❓ PILOT answering question: {text}")
        
        # First try OpenAI API if available
        if self.ai_search.openai_available:
            result = await self.ai_search.search_answer(text, use_browser_fallback=False)
            
            if result["success"]:
                return {
                    "success": True,
                    "intent": "pilot_answer",
                    "pilot_mode": True,
                    "text": f"🤖 PILOT: {result['answer']}",
                    "data": {
                        "source": result["source"],
                        "question": text
                    }
                }
        
        # Fallback to internal reasoning
        reasoning_result = await self.reasoning.autonomous_think(text)
        
        # If reasoning doesn't provide a good answer, suggest browser
        response = reasoning_result.get("response", "")
        
        if len(response) < 50 or "non so" in response.lower():
            # Offer to search online
            return {
                "success": True,
                "intent": "pilot_answer_suggest",
                "pilot_mode": True,
                "text": f"🤔 PILOT: {response}\n\n💡 Vuoi che cerchi su ChatGPT o Perplexity per una risposta più completa?",
                "data": {
                    "source": "reasoning",
                    "question": text,
                    "can_search_online": True
                }
            }
        
        return {
            "success": True,
            "intent": "pilot_answer",
            "pilot_mode": True,
            "text": f"🧠 PILOT: {response}",
            "data": {
                "source": "reasoning",
                "question": text,
                "reasoning_data": reasoning_result
            }
        }
    
    async def _handle_system_status(self) -> Dict[str, Any]:
        """Handle system status query with personality"""
        # Collect system metrics
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status = {
            "cpu": {
                "usage_percent": cpu_percent,
                "status": "ok" if cpu_percent < 80 else "warning"
            },
            "memory": {
                "usage_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2),
                "status": "ok" if memory.percent < 80 else "warning"
            },
            "disk": {
                "usage_percent": disk.percent,
                "free_gb": round(disk.free / (1024**3), 2),
                "status": "ok" if disk.percent < 90 else "warning"
            }
        }
        
        # Use personality for natural response
        text = self.personality.get_status_response(
            cpu_percent, 
            memory.percent, 
            disk.percent
        )
        
        return {
            "success": True,
            "intent": "status",
            "text": text,
            "tts_text": self.personality.wrap_for_tts(text),
            "data": status
        }
    
    async def _handle_analysis_request(self, entities: Dict) -> Dict[str, Any]:
        """Handle analysis request with personality"""
        target = entities.get("target", "system")
        
        logger.info(f"🔍 Starting analysis of: {target}")
        
        # Add thinking phrase
        thinking = self.personality.get_thinking_phrase()
        
        # Trigger optimizer analysis
        analysis = await self.optimizer.analyze(target)
        
        # Format response based on personality level
        level = self.personality.current_level.value
        
        if level == "pilot":
            text = f"Analisi {target}. "
            if analysis.get("issues"):
                text += f"Problemi: {len(analysis['issues'])}. "
            if analysis.get("score"):
                text += f"Score: {analysis['score']:.0%}. "
            if analysis.get("optimizations"):
                text += f"Ottimizzazioni: {len(analysis['optimizations'])}."
        elif level == "advanced":
            text = f"Analisi tecnica completata per '{target}'. "
            if analysis.get("issues"):
                text += f"Rilevate {len(analysis['issues'])} anomalie da investigare. "
            if analysis.get("score"):
                text += f"Indice di efficienza calcolato: {analysis['score']:.1%}. "
            if analysis.get("optimizations"):
                text += f"Identificate {len(analysis['optimizations'])} ottimizzazioni applicabili."
        else:
            ack = self.personality.get_acknowledgment()
            text = f"{ack} Ho analizzato {target}. "
            if analysis.get("issues"):
                issues_count = len(analysis["issues"])
                text += f"Ho trovato {issues_count} cose da sistemare. "
            if analysis.get("score"):
                score = analysis["score"]
                if score > 0.8:
                    text += f"Il sistema sta andando bene! ({score:.0%}) 😊 "
                elif score > 0.5:
                    text += f"C'è margine di miglioramento ({score:.0%}). "
                else:
                    text += f"Ci sono alcune criticità da affrontare ({score:.0%}). "
            if analysis.get("optimizations"):
                text += f"Ho {len(analysis['optimizations'])} suggerimenti per te!"
        
        return {
            "success": True,
            "intent": "analysis",
            "text": text,
            "tts_text": self.personality.wrap_for_tts(text),
            "data": analysis
        }
    
    async def _handle_optimization(self, entities: Dict) -> Dict[str, Any]:
        """Handle optimization request with personality"""
        target = entities.get("target", "system")
        level = self.personality.current_level.value
        
        optimizations = await self.optimizer.get_optimizations(target)
        
        if not optimizations:
            if level == "pilot":
                no_opt_text = f"Nessuna ottimizzazione necessaria per {target}. Sistema efficiente."
            elif level == "advanced":
                no_opt_text = f"Analisi completata: {target} opera già a livelli ottimali. Nessun intervento richiesto."
            else:
                no_opt_text = f"Ottima notizia! 🎉 {target.capitalize()} funziona già al massimo. Non serve fare nulla!"
            
            return {
                "success": True,
                "intent": "optimization",
                "text": no_opt_text,
                "tts_text": self.personality.wrap_for_tts(no_opt_text),
                "data": []
            }
        
        # Format optimizations based on personality
        if level == "pilot":
            text = f"{len(optimizations)} ottimizzazioni per {target}:"
            for i, opt in enumerate(optimizations[:3], 1):
                impact = opt.get("impact_percent", 0)
                text += f"\n{i}. {opt['description']} (+{impact:.0f}%)"
        elif level == "advanced":
            text = f"Report ottimizzazione per '{target}' - {len(optimizations)} interventi identificati:\n"
            for i, opt in enumerate(optimizations[:3], 1):
                impact = opt.get("impact_percent", 0)
                priority = opt.get("priority", "media")
                text += f"\n{i}. [{priority.upper()}] {opt['description']} (impatto: +{impact:.1f}%)"
        else:
            text = f"Ho trovato {len(optimizations)} modi per migliorare {target}! 🚀\n"
            for i, opt in enumerate(optimizations[:3], 1):
                impact = opt.get("impact_percent", 0)
                text += f"\n{i}. {opt['description']} - potremmo guadagnare circa {impact:.0f}%"
            text += "\n\nVuoi che ne applichi qualcuna?"
        
        return {
            "success": True,
            "intent": "optimization",
            "text": text,
            "tts_text": self.personality.wrap_for_tts(text),
            "data": optimizations
        }
    
    async def _handle_calculation(self, text: str, entities: Dict) -> Dict[str, Any]:
        """Handle calculation requests - SIMPLIFIED: only exact results"""
        logger.info(f"🧮 Performing calculations for: {text}")
        
        # First try to detect simple math expressions
        import re
        
        # Try to find simple math in text like "2+2", "10 per 5", "100/4", etc.
        text_lower = text.lower()
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = None
            
            if any(op in text_lower for op in ["più", "+", "somma", "aggiungi"]):
                result = a + b
            elif any(op in text_lower for op in ["meno", "-", "sottrai", "differenza"]):
                result = a - b
            elif any(op in text_lower for op in ["per", "x", "*", "moltiplica", "moltiplicato"]):
                result = a * b
            elif any(op in text_lower for op in ["diviso", "/", "dividi", "divisione"]):
                result = a / b if b != 0 else None
            elif any(op in text_lower for op in ["potenza", "elevato", "^", "alla"]):
                result = a ** b
            
            if result is not None:
                # Format result simply
                if isinstance(result, float) and result == int(result):
                    response = f"{int(result)}"
                elif isinstance(result, float):
                    response = f"{result:.2f}"
                else:
                    response = f"{result}"
                
                return {
                    "success": True,
                    "intent": "calculation",
                    "text": response,
                    "tts_text": f"Il risultato è {response}",
                    "data": {"result": result, "operands": [a, b]}
                }
        
        # For single number - try other operations
        if len(numbers) == 1:
            a = float(numbers[0])
            result = None
            
            if "radice" in text_lower or "sqrt" in text_lower:
                import math
                result = math.sqrt(a)
            elif "quadrato" in text_lower and "radice" not in text_lower:
                result = a ** 2
            elif "cubo" in text_lower:
                result = a ** 3
            elif "doppio" in text_lower:
                result = a * 2
            elif "triplo" in text_lower:
                result = a * 3
            elif "metà" in text_lower or "meta" in text_lower:
                result = a / 2
            
            if result is not None:
                if isinstance(result, float) and result == int(result):
                    response = f"{int(result)}"
                elif isinstance(result, float):
                    response = f"{result:.2f}"
                else:
                    response = f"{result}"
                
                return {
                    "success": True,
                    "intent": "calculation",
                    "text": response,
                    "tts_text": f"Il risultato è {response}",
                    "data": {"result": result}
                }
        
        # Check if user explicitly asked for system stats
        system_keywords = ["efficienza", "cpu", "memoria", "ram", "disco", "sistema", "risorse", "performance", "prestazioni"]
        if any(kw in text_lower for kw in system_keywords):
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            max_usage = max(cpu, memory.percent, disk.percent)
            efficiency = 100 - max_usage
            
            # Simplified response for system stats
            if "efficienza" in text_lower:
                response = f"{efficiency:.1f}%"
            elif "cpu" in text_lower:
                response = f"{cpu:.1f}%"
            elif "memoria" in text_lower or "ram" in text_lower:
                response = f"{memory.percent:.1f}%"
            elif "disco" in text_lower:
                response = f"{disk.percent:.1f}%"
            else:
                response = f"CPU: {cpu:.1f}% | RAM: {memory.percent:.1f}% | Disco: {disk.percent:.1f}%"
            
            return {
                "success": True,
                "intent": "system_stats",
                "text": response,
                "tts_text": response,
                "data": {
                    "efficiency": efficiency,
                    "cpu_percent": cpu,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent
                }
            }
        
        # No calculation found - return failure so AI can handle it
        return {
            "success": False,
            "intent": "unknown",
            "text": None,
            "reason": "no_calculation_found"
        }
    
    async def _handle_control_command(self, entities: Dict) -> Dict[str, Any]:
        """Handle control command (requires Pilot mode)"""
        if not self.pilot_mode_active:
            return {
                "success": False,
                "intent": "control",
                "text": "I comandi di controllo richiedono la modalità Pilot. Attivala prima di procedere.",
                "requires_pilot": True
            }
        
        action = entities.get("action")
        target = entities.get("target")
        
        # Execute control action
        result = await self._execute_control_action(action, target)
        
        return {
            "success": result["success"],
            "intent": "control",
            "text": result["message"],
            "data": result
        }
    
    async def _handle_information_query(self, text: str, entities: Dict) -> Dict[str, Any]:
        """Handle general information query using AI"""
        # Use OpenAI for complex queries
        if settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Get relevant memory context
            context = await self.memory.get_relevant_context(text)
            
            messages = [
                {"role": "system", "content": "Sei Gideon, un assistente IA avanzato. Rispondi in modo conciso e professionale in italiano."},
                {"role": "user", "content": text}
            ]
            
            if context:
                messages.insert(1, {"role": "system", "content": f"Contesto: {context}"})
            
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            return {
                "success": True,
                "intent": "information",
                "text": answer,
                "data": {"tokens_used": response.usage.total_tokens}
            }
        
        return await self._handle_general_conversation(text)
    
    async def _handle_chatgpt_query(self, text: str) -> Dict[str, Any]:
        """Handle AI query request using multi-provider system"""
        logger.info(f"🤖 AI query: {text}")
        
        # Extract the actual question
        question = text.lower()
        prefixes = ["chatgpt", "chiedi a gpt", "chiedi all'ai", "cerca su chatgpt", 
                   "domanda a chatgpt", "chiedi a claude", "chiedi a gemini", "chiedi a groq"]
        for prefix in prefixes:
            if prefix in question:
                question = question.split(prefix, 1)[-1].strip()
                break
        
        if not question or len(question) < 3:
            providers = self.ai_manager.get_available_providers()
            provider_list = ", ".join([p["name"] for p in providers]) if providers else "nessuno configurato"
            return {
                "success": True,
                "intent": "ai_prompt",
                "text": f"🤖 Cosa vuoi chiedermi? Provider disponibili: {provider_list}",
                "data": {"available_providers": [p["id"] for p in providers]}
            }
        
        # Detect if user wants a specific provider
        provider = None
        if "claude" in text.lower():
            provider = "anthropic"
        elif "gemini" in text.lower():
            provider = "google"
        elif "groq" in text.lower() or "llama" in text.lower():
            provider = "groq"
        elif "ollama" in text.lower() or "locale" in text.lower():
            provider = "ollama"
        
        # OTTIMIZZAZIONE: Skip history per risposte veloci su domande semplici
        conversation_history = None
        if len(question) > 100 or any(kw in question for kw in ["precedente", "prima", "dicevi", "abbiamo"]):
            # Solo se sembra far riferimento a conversazione precedente
            history = await self.memory.get_conversation_history(limit=3)
            conversation_history = [
                {"role": "user" if h.get("role") == "user" else "assistant", "content": h.get("content", "")}
                for h in history
            ]
        
        # Use the new multi-provider AI system with dynamic parameters
        max_tokens = getattr(self, '_current_max_tokens', 300)
        temperature = getattr(self, '_current_temperature', 0.5)
        
        response = await self.ai_manager.generate(
            prompt=question,
            provider=provider,
            conversation_history=conversation_history,
            max_tokens=max_tokens,
            temperature=temperature,
            fallback=True
        )
        
        if response.success:
            provider_emoji = {
                "OpenAI": "🤖",
                "Anthropic Claude": "🧠",
                "Google Gemini": "💎",
                "Groq": "⚡",
                "Ollama (Local)": "🏠"
            }.get(response.provider, "🤖")
            
            return {
                "success": True,
                "intent": "ai_response",
                "text": f"{provider_emoji} {response.provider} risponde:\n\n{response.content}",
                "data": {
                    "source": response.provider.lower().replace(" ", "_"),
                    "model": response.model,
                    "tokens_used": response.tokens_used
                }
            }
        
        # Fallback to browser if no provider available
        result = await self.ai_search.search_answer(question)
        if result["success"]:
            return {
                "success": True,
                "intent": "ai_browser",
                "text": f"🌐 {result['answer']}",
                "data": {
                    "source": "browser",
                    "url": result.get("url", "")
                }
            }
        
        return {
            "success": False,
            "intent": "ai_error",
            "text": "🤖 Non sono riuscito a ottenere una risposta. Configura almeno un provider AI nel file .env",
            "data": {"available_providers": self.ai_manager.get_status()}
        }
    
    async def _handle_perplexity_query(self, text: str) -> Dict[str, Any]:
        """Handle Perplexity AI query request"""
        logger.info(f"🔍 Perplexity query: {text}")
        
        # Extract the question
        question = text.lower()
        prefixes = ["perplexity", "cerca con perplexity", "chiedi a perplexity"]
        for prefix in prefixes:
            if prefix in question:
                question = question.split(prefix, 1)[-1].strip()
                break
        
        if not question or len(question) < 3:
            return {
                "success": True,
                "intent": "perplexity_prompt",
                "text": "🔍 Cosa vuoi cercare con Perplexity? Fammi la tua domanda.",
                "data": {}
            }
        
        result = await self.ai_search.ask_perplexity(question)
        
        return {
            "success": True,
            "intent": "perplexity",
            "text": f"🔍 {result['answer']}",
            "data": {
                "source": "perplexity",
                "url": result.get("url", ""),
                "question": question
            }
        }
    
    async def _handle_general_conversation(self, text: str) -> Dict[str, Any]:
        """Handle general conversation with smart local responses"""
        
        text_lower = text.lower()
        
        # Check for calculation requests first
        calc_result = await self.nlp.handle_calculation(text)
        if calc_result:
            return calc_result
        
        # Try to get a smart local response (no API needed)
        intent_result = await self.nlp.extract_intent(text)
        intent = intent_result.get("intent", "conversation")
        
        # Get smart response for known intents
        smart_response = self.nlp.get_smart_response(intent)
        if smart_response:
            return {
                "success": True,
                "intent": intent,
                "text": smart_response,
                "tts_text": self.personality.wrap_for_tts(smart_response),
                "data": {"source": "smart_local"}
            }
        
        # Check if it requires thinking
        thinking_keywords = ['perché', 'come mai', 'spiegami', 'valuta', 'considera']
        if any(keyword in text_lower for keyword in thinking_keywords):
            # Trigger autonomous thinking
            return await self._handle_autonomous_thinking(text, {})
        
        # Get multi-turn context for adaptive responses
        context = await self.memory.get_relevant_context(text)
        multi_turn = context.get("multi_turn", {}) if context else {}
        
        # Adapt response based on conversation context
        response_text = await self._generate_contextual_response(text, multi_turn)
        
        return {
            "success": True,
            "intent": "conversation",
            "text": response_text,
            "tts_text": self.personality.wrap_for_tts(response_text),
            "context_aware": True,
            "turn_count": multi_turn.get("turn_count", 0)
        }
    
    async def _generate_contextual_response(self, text: str, multi_turn: Dict) -> str:
        """Generate response using AI provider (OpenRouter/etc) with conversation context"""
        
        # Try to use AI provider for intelligent response
        if self.ai_manager.has_available_provider():
            try:
                # Get conversation history for context
                history = await self.memory.get_conversation_history(limit=5)
                conversation_history = [
                    {"role": "user" if h.get("role") == "user" else "assistant", "content": h.get("content", "")}
                    for h in history
                ]
                
                # Build system prompt based on personality level
                level = self.personality.current_level.value
                system_prompts = {
                    "pilot": """Sei GIDEON, un'intelligenza artificiale autonoma e proattiva in modalità Pilot.
Rispondi in modo conciso, diretto e operativo. Sei come J.A.R.V.I.S. di Iron Man.
Parla in italiano. Sii efficiente e vai dritto al punto.""",
                    
                    "advanced": """Sei GIDEON, un assistente IA avanzato in modalità analitica.
Fornisci risposte dettagliate e tecniche quando appropriato.
Parla in italiano in modo professionale ma accessibile.""",
                    
                    "normal": """Sei GIDEON, un assistente IA amichevole e intelligente.
Rispondi in italiano in modo naturale, utile e conversazionale.
Sei disponibile, chiaro e ti piace aiutare. Puoi fare calcoli, dare informazioni e conversare."""
                }
                
                system_prompt = system_prompts.get(level, system_prompts["normal"])
                
                # Add context about active topic if available
                active_topic = multi_turn.get("active_topic")
                if active_topic:
                    system_prompt += f"\n\nArgomento corrente della conversazione: {active_topic}"
                
                # Get dynamic parameters
                max_tokens = getattr(self, '_current_max_tokens', 300)
                temperature = getattr(self, '_current_temperature', 0.5)
                
                # Generate response using AI
                response = await self.ai_manager.generate(
                    prompt=text,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    fallback=True
                )
                
                if response.success and response.content:
                    logger.info(f"🤖 AI response generated via {response.provider}")
                    return response.content
                    
            except Exception as e:
                logger.warning(f"AI generation failed: {e}, falling back to local response")
        
        # Fallback to local responses if AI not available
        return self._get_fallback_response(text, multi_turn)
    
    def _get_fallback_response(self, text: str, multi_turn: Dict) -> str:
        """Fallback local response when AI is not available"""
        is_follow_up = multi_turn.get("is_follow_up", False)
        referenced_context = multi_turn.get("referenced_context", {})
        active_topic = multi_turn.get("active_topic")
        turn_count = multi_turn.get("turn_count", 0)
        level = self.personality.current_level.value
        
        # Handle continuation requests
        if referenced_context.get("continuation_request"):
            return f"Continuando su '{active_topic}'... Cosa vuoi sapere in particolare?"
        
        # Handle comparison requests
        if referenced_context.get("comparison_request"):
            return f"Posso fare un confronto. Dimmi cosa vuoi comparare."
        
        # Handle pronoun references
        if referenced_context.get("has_pronoun_reference") and active_topic:
            return f"Ti riferisci a '{active_topic}'? Dimmi di più."
        
        # Handle follow-up
        if is_follow_up:
            return "Come posso aiutarti ulteriormente?"
        
        # Default response
        greeting = self.personality.get_greeting()
        return f"{greeting} Come posso aiutarti?"
    
    async def _handle_smart_optimization(self, text: str, entities: Dict) -> Dict[str, Any]:
        """
        Handle optimization requests with intelligent analysis and proposals.
        
        STEP 9: Propone ottimizzazioni in modo proattivo con calcoli e percentuali.
        """
        logger.info(f"🎯 Smart Optimization request: {text}")
        
        import psutil
        
        # Collect current system metrics
        cpu = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Calculate efficiency metrics
        max_usage = max(cpu, memory.percent, disk.percent)
        current_efficiency = 100 - max_usage
        target_efficiency = 90.0
        
        # Generate optimization proposals based on analysis
        proposals = []
        potential_gain = 0
        
        if cpu > 60:
            gain = (cpu - 60) * 0.3
            potential_gain += gain
            proposals.append({
                "area": "CPU",
                "current": f"{cpu:.1f}%",
                "issue": "Utilizzo CPU moderato-alto",
                "action": "Identificare e ottimizzare processi intensivi",
                "expected_gain": f"+{gain:.1f}% efficienza",
                "priority": "alta" if cpu > 80 else "media"
            })
        
        if memory.percent > 70:
            gain = (memory.percent - 70) * 0.25
            potential_gain += gain
            proposals.append({
                "area": "Memoria",
                "current": f"{memory.percent:.1f}%",
                "issue": "Memoria in uso elevata",
                "action": "Ottimizzare caching e liberare memoria non utilizzata",
                "expected_gain": f"+{gain:.1f}% efficienza",
                "priority": "alta" if memory.percent > 85 else "media"
            })
        
        if disk.percent > 80:
            gain = (disk.percent - 80) * 0.2
            potential_gain += gain
            proposals.append({
                "area": "Disco",
                "current": f"{disk.percent:.1f}%",
                "issue": "Spazio disco limitato",
                "action": "Pulire file temporanei e cache",
                "expected_gain": f"+{gain:.1f}% efficienza",
                "priority": "alta" if disk.percent > 90 else "media"
            })
        
        # Calculate projected efficiency
        projected_efficiency = min(current_efficiency + potential_gain, 100)
        improvement_percent = projected_efficiency - current_efficiency
        
        # Format response
        response_text = "🎯 **ANALISI OTTIMIZZAZIONE**\n\n"
        
        # Current state
        response_text += f"📊 **Stato Attuale:**\n"
        response_text += f"   • Efficienza sistema: {current_efficiency:.1f}%\n"
        response_text += f"   • CPU: {cpu:.1f}% | Memoria: {memory.percent:.1f}% | Disco: {disk.percent:.1f}%\n\n"
        
        if proposals:
            response_text += f"💡 **Proposte di Ottimizzazione:**\n"
            for i, prop in enumerate(proposals, 1):
                priority_icon = "🔴" if prop["priority"] == "alta" else "🟡"
                response_text += f"\n{i}. {priority_icon} **{prop['area']}** ({prop['current']})\n"
                response_text += f"   → {prop['action']}\n"
                response_text += f"   📈 Guadagno stimato: {prop['expected_gain']}\n"
            
            response_text += f"\n⚡ **Risultato Atteso:**\n"
            response_text += f"   • Efficienza proiettata: {projected_efficiency:.1f}%\n"
            response_text += f"   • Miglioramento totale: +{improvement_percent:.1f}%\n"
        else:
            response_text += "✅ **Sistema già ottimizzato!**\n"
            response_text += f"   Efficienza attuale: {current_efficiency:.1f}%\n"
            response_text += "   Non sono necessari interventi immediati.\n"
        
        # Add recommendation
        if proposals:
            top_priority = max(proposals, key=lambda x: 1 if x["priority"] == "alta" else 0)
            response_text += f"\n📌 **Raccomandazione:** Inizia da {top_priority['area']} per il massimo impatto."
        
        # TTS version (concise)
        if proposals:
            tts_text = f"Ho analizzato il sistema. L'efficienza attuale è {current_efficiency:.0f}%. "
            tts_text += f"Ho {len(proposals)} proposte di ottimizzazione che potrebbero migliorare le prestazioni del {improvement_percent:.0f}%."
        else:
            tts_text = f"Il sistema è già ottimizzato con un'efficienza del {current_efficiency:.0f}%."
        
        return {
            "success": True,
            "intent": "optimization",
            "text": response_text,
            "tts_text": self.personality.wrap_for_tts(tts_text),
            "data": {
                "current_efficiency": current_efficiency,
                "projected_efficiency": projected_efficiency,
                "improvement_percent": improvement_percent,
                "proposals": proposals,
                "metrics": {
                    "cpu": cpu,
                    "memory": memory.percent,
                    "disk": disk.percent
                },
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def _handle_smart_response(self, text: str, entities: Dict, intent_result: Dict) -> Dict[str, Any]:
        """
        Handle general questions with intelligent, context-aware responses.
        Uses AI providers (OpenRouter, OpenAI, etc.) for intelligent responses.
        """
        logger.info(f"💬 Smart response for: {text}")
        
        intent = intent_result.get("intent", "unknown")
        confidence = intent_result.get("confidence", 0.5)
        
        # Try calculation first (no API needed)
        calc_result = await self.nlp.handle_calculation(text)
        if calc_result:
            return calc_result
        
        # Use AI Manager for intelligent response (OpenRouter, OpenAI, etc.)
        has_provider = self.ai_manager.has_available_provider()
        logger.info(f"🔍 AI Provider check: has_available_provider = {has_provider}")
        
        if has_provider:
            try:
                # Get conversation history for context
                history = await self.memory.get_conversation_history(limit=5)
                conversation_history = [
                    {"role": "user" if h.get("role") == "user" else "assistant", "content": h.get("content", "")}
                    for h in history
                ]
                
                # Build contextual system prompt
                system_prompt = f"""Sei GIDEON, un assistente IA avanzato e amichevole.

Caratteristiche:
- Rispondi sempre in italiano in modo naturale e conversazionale
- Fornisci risposte utili, precise e informative
- Sii disponibile e cordiale
- Se non sai qualcosa, dillo onestamente
- Puoi fare calcoli, dare informazioni, consigliare e conversare

Livello operativo attuale: {self.current_level}"""
                
                logger.info(f"🤖 Generating AI response via {self.ai_manager.default_provider}...")
                
                response = await self.ai_manager.generate(
                    prompt=text,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    fallback=True,
                    max_tokens=1000,
                    temperature=0.7
                )
                
                logger.info(f"📊 AI response received: success={response.success}, content_len={len(response.content) if response.content else 0}")
                
                if response.success and response.content:
                    logger.info(f"✅ AI response from {response.provider}: {len(response.content)} chars")
                    return {
                        "success": True,
                        "intent": intent,
                        "text": response.content,
                        "tts_text": self.personality.wrap_for_tts(response.content),
                        "data": {
                            "source": response.provider.lower().replace(" ", "_"),
                            "model": response.model,
                            "tokens_used": response.tokens_used,
                            "confidence": confidence
                        }
                    }
                else:
                    logger.warning(f"AI response failed: {response.finish_reason}")
                    
            except Exception as e:
                logger.warning(f"AI generation error: {e}")
        
        # Fallback to smart local response
        smart_response = self.nlp.get_smart_response(intent)
        if smart_response and confidence > 0.6:
            return {
                "success": True,
                "intent": intent,
                "text": smart_response,
                "tts_text": self.personality.wrap_for_tts(smart_response),
                "data": {"source": "smart_local", "confidence": confidence}
            }
        
        # Ultimate fallback
        return await self._handle_general_conversation(text)

    async def _handle_autonomous_thinking(self, text: str, entities: Dict) -> Dict[str, Any]:
        """
        Handle autonomous thinking request using AI (OpenRouter/OpenAI/etc).
        
        STEP 9 Enhancement: Ragionamento autonomo con AI per risposte intelligenti.
        """
        logger.info(f"🧠 Activating autonomous thinking mode for: {text}")
        
        # First, try to use AI for intelligent response
        has_provider = self.ai_manager.has_available_provider()
        logger.info(f"🔍 Autonomous thinking - AI Provider check: has_available_provider = {has_provider}")
        
        if has_provider:
            try:
                # Get conversation history for context
                history = await self.memory.get_conversation_history(limit=5)
                conversation_history = [
                    {"role": "user" if h.get("role") == "user" else "assistant", "content": h.get("content", "")}
                    for h in history
                ]
                
                # Build contextual system prompt
                system_prompt = f"""Sei GIDEON, un assistente IA avanzato con capacità di ragionamento autonomo.

Quando rispondi:
- Rispondi sempre in italiano in modo naturale e conversazionale
- Fornisci risposte approfondite, utili e informative
- Per domande complesse, spiega il ragionamento passo per passo
- Usa esempi pratici quando appropriato
- Sii disponibile e cordiale
- Se non sai qualcosa con certezza, dillo onestamente

Livello operativo: {self.current_level}"""
                
                logger.info(f"🤖 Generating AI response via {self.ai_manager.default_provider}...")
                
                response = await self.ai_manager.generate(
                    prompt=text,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    fallback=True,
                    max_tokens=1500,
                    temperature=0.7
                )
                
                logger.info(f"📊 Autonomous AI response received: success={response.success}, content_len={len(response.content) if response.content else 0}, finish_reason={response.finish_reason}")
                
                if response.success and response.content:
                    logger.info(f"✅ AI response from {response.provider}: {len(response.content)} chars")
                    return {
                        "success": True,
                        "intent": "autonomous_thinking",
                        "text": response.content,
                        "tts_text": self.personality.wrap_for_tts(response.content),
                        "data": {
                            "source": response.provider.lower().replace(" ", "_"),
                            "model": response.model,
                            "tokens_used": response.tokens_used,
                            "reasoning_steps": ["AI autonomous reasoning completed"],
                            "confidence": 0.9
                        },
                        "reasoning_quality": "buona"
                    }
                else:
                    logger.warning(f"AI response failed: {response.finish_reason}, falling back to local reasoning")
                    
            except Exception as e:
                logger.warning(f"AI generation error: {e}, falling back to local reasoning")
        
        # Fallback to local reasoning engine if AI unavailable
        logger.info("📝 Using local reasoning engine as fallback...")
        
        # Determine thinking depth based on request complexity
        depth = 3
        text_lower = text.lower()
        
        # Auto-detect complexity for depth
        if any(w in text_lower for w in ["approfonditamente", "dettagliato", "completo", "analisi completa"]):
            depth = 5
        elif any(w in text_lower for w in ["veloce", "rapido", "breve"]):
            depth = 2
        elif len(text.split()) > 15:  # Long queries need deeper thinking
            depth = 4
        
        # Extract the core topic/question
        topic = text
        
        # Build comprehensive context
        context = {
            "request_time": datetime.now().isoformat(),
            "mode": "autonomous_reasoning",
            "user_query": text,
            "entities": entities,
            "personality_level": self.current_level,
            "system_state": await self._get_quick_system_state()
        }
        
        # ========== STEP 1: Pre-processing and understanding ==========
        logger.info("🧠 Step 1: Comprensione della domanda...")
        
        # ========== STEP 2: Autonomous reasoning ==========
        logger.info("🧠 Step 2: Ragionamento autonomo in corso...")
        thought_result = await self.reasoning.autonomous_think(
            topic=topic,
            context=context,
            depth=depth
        )
        
        # ========== STEP 3: Extract and validate conclusions ==========
        logger.info("🧠 Step 3: Validazione conclusioni...")
        conclusion = thought_result["conclusion"]
        thought_chain = thought_result["thought_chain"]
        
        # ========== STEP 4: Format natural response ==========
        # Create a human-like response from the conclusion
        response_text = self._format_reasoning_response(
            conclusion=conclusion,
            thought_chain=thought_chain,
            thinking_time=thought_result["thinking_time_seconds"],
            query=text
        )
        
        # TTS version (more conversational)
        tts_text = self._format_reasoning_tts(conclusion, thought_chain)
        
        # Build reasoning steps list from thought chain
        reasoning_steps_list = []
        for i, step in enumerate(thought_chain[:5], 1):  # Max 5 steps
            action = step.get('action', 'analysis')
            thought = step.get('thought', '')[:100]
            if thought:
                reasoning_steps_list.append(f"Step {i}: {action} - {thought}")
        if not reasoning_steps_list:
            reasoning_steps_list = ["AI reasoning completed"]
        
        return {
            "success": True,
            "intent": "autonomous_thinking",
            "text": response_text,
            "tts_text": self.personality.wrap_for_tts(tts_text),
            "data": {
                "thought_process": thought_result,
                "conclusion": conclusion,
                "thinking_depth": depth,
                "chain_summary": self._format_thought_chain(thought_chain),
                "reasoning_steps": reasoning_steps_list,
                "reasoning_count": len(thought_chain)
            },
            "reasoning_quality": conclusion.get("reasoning_quality", "buona")
        }
    
    def _format_reasoning_response(
        self,
        conclusion: Dict,
        thought_chain: List[Dict],
        thinking_time: float,
        query: str
    ) -> str:
        """Format reasoning result into a natural response"""
        response = ""
        
        # Intro based on personality
        if self.current_level == "pilot":
            response += "📊 **ANALISI COMPLETATA**\n\n"
        elif self.current_level == "advanced":
            response += "🔬 Ho analizzato la tua richiesta:\n\n"
        else:
            response += ""  # Normal mode: direct response
        
        # Main conclusion
        response += f"{conclusion['statement']}\n"
        
        # Key insights from thought chain
        key_insights = [step for step in thought_chain if step.get('action') == 'pattern_analysis' or step.get('action') == 'logical_reasoning']
        if key_insights:
            response += "\n📌 **Punti chiave:**\n"
            for insight in key_insights[:2]:
                if insight.get('thought'):
                    response += f"• {insight['thought']}\n"
        
        # Recommendations
        if conclusion.get("recommendations"):
            response += "\n💡 **Suggerimenti:**\n"
            for i, rec in enumerate(conclusion["recommendations"][:3], 1):
                response += f"{i}. {rec}\n"
        
        # Confidence and timing
        confidence_emoji = "🎯" if conclusion['confidence'] > 0.8 else "📊" if conclusion['confidence'] > 0.6 else "🤔"
        response += f"\n{confidence_emoji} Confidenza: {conclusion['confidence']*100:.0f}%"
        response += f" | ⏱️ Elaborazione: {thinking_time:.1f}s"
        
        return response
    
    def _format_reasoning_tts(self, conclusion: Dict, thought_chain: List[Dict]) -> str:
        """Format reasoning for TTS (spoken response)"""
        # Keep TTS concise and natural
        response = conclusion['statement']
        
        # Add one key recommendation if available
        if conclusion.get("recommendations") and len(conclusion["recommendations"]) > 0:
            response += f". Ti consiglio di {conclusion['recommendations'][0].lower()}"
        
        return response
    
    async def _get_quick_system_state(self) -> Dict[str, Any]:
        """Get quick system state for reasoning context"""
        import psutil
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0
            }
        except:
            return {}
    
    def _format_thought_chain(self, chain: List[Dict]) -> str:
        """Format thought chain for display"""
        summary = "Processo di ragionamento:\n"
        for step in chain:
            summary += f"  {step['step']}. {step['action']}: {step.get('thought', 'N/A')}\n"
        return summary
    
    async def _handle_pilot_activation(self, text: str) -> Dict[str, Any]:
        """Handle Pilot mode activation request"""
        return {
            "success": True,
            "intent": "pilot_activation",
            "text": "Per attivare la modalità Pilot, pronuncia la frase di autorizzazione.",
            "requires_auth": True,
            "auth_phrase_required": True
        }
    
    async def _handle_pilot_deactivation(self) -> Dict[str, Any]:
        """Handle Pilot mode deactivation request"""
        if self.pilot_mode_active:
            await self.deactivate_pilot_mode()
            response_text = "🔒 Modalità Pilot disattivata. Tornato in modalità normale."
            return {
                "success": True,
                "intent": "pilot_deactivation",
                "text": response_text,
                "tts_text": self.personality.wrap_for_tts("Modalità Pilot disattivata. Tornato in modalità normale."),
                "pilot_mode": False
            }
        else:
            return {
                "success": True,
                "intent": "pilot_deactivation",
                "text": "La modalità Pilot non è attiva.",
                "tts_text": self.personality.wrap_for_tts("La modalità Pilot non è attiva."),
                "pilot_mode": False
            }
    
    async def activate_pilot_mode(self, auth_phrase: str) -> bool:
        """Activate Pilot mode with authentication"""
        if auth_phrase == settings.PILOT_PHRASE:
            self.pilot_mode_active = True
            self.pilot_activated_at = datetime.now()
            self.current_level = "pilot"
            self.action_manager.set_pilot_mode(True)
            self.personality.set_level("pilot")
            logger.warning("🚨 PILOT MODE ACTIVATED")
            return True
        return False
    
    async def deactivate_pilot_mode(self):
        """Deactivate Pilot mode"""
        self.pilot_mode_active = False
        self.pilot_activated_at = None
        self.current_level = "normal"
        self.action_manager.set_pilot_mode(False)
        self.personality.set_level("normal")
        logger.info("✅ Pilot mode deactivated")
    
    def set_operation_level(self, level: str):
        """
        Set operation level and sync all components.
        
        Levels:
        - NORMAL: Friendly assistant, basic capabilities
        - ADVANCED: Deep reasoning, analysis, optimization suggestions
        - PILOT: Full autonomy, proactive actions, automatic execution (Gideon style)
        """
        valid_levels = ["normal", "advanced", "pilot"]
        if level.lower() in valid_levels:
            old_level = self.current_level
            self.current_level = level.lower()
            self.personality.set_level(level.lower())
            
            # Get level config
            config = self.get_level_config()
            
            if level.lower() == "pilot":
                self.pilot_mode_active = True
                self.action_manager.set_pilot_mode(True)
                logger.info("🚀 PILOT MODE: Full autonomy activated - Gideon style")
            elif level.lower() == "advanced":
                self.pilot_mode_active = False
                self.action_manager.set_pilot_mode(False)
                logger.info("⚡ ADVANCED MODE: Deep reasoning activated")
            else:
                self.pilot_mode_active = False
                self.action_manager.set_pilot_mode(False)
                logger.info("🟢 NORMAL MODE: Friendly assistant")
                
            logger.info(f"📊 Level changed: {old_level.upper()} → {level.upper()} (autonomy: {config['autonomy']*100:.0f}%)")
            
            return {
                "level": level.lower(),
                "config": config,
                "pilot_active": self.pilot_mode_active
            }
    
    def _get_expression_for_intent(self, intent: str) -> str:
        """Get avatar expression for intent"""
        expressions = {
            "time": "neutral",
            "status": "focused",
            "analysis": "thinking",
            "optimization": "confident",
            "control": "serious",
            "information": "friendly",
            "conversation": "happy",
            "error": "concerned"
        }
        return expressions.get(intent, "neutral")
    
    async def _execute_control_action(self, action: str, target: str) -> Dict[str, Any]:
        """Execute a control action (Pilot mode only)"""
        logger.warning(f"⚠️ Executing control action: {action} on {target}")
        
        # Simulated control actions
        actions_map = {
            "restart": f"Riavvio di {target} in corso...",
            "stop": f"Arresto di {target} in corso...",
            "start": f"Avvio di {target} in corso...",
            "deploy": f"Deployment di {target} in corso...",
        }
        
        message = actions_map.get(action, f"Esecuzione di {action} su {target}")
        
        return {
            "success": True,
            "message": message,
            "action": action,
            "target": target,
            "executed_at": datetime.now().isoformat()
        }
    
    async def analyze_system(self, target: str = "system") -> Dict[str, Any]:
        """Perform complete system analysis"""
        logger.info(f"🔍 Analyzing {target}...")
        
        analysis = await self.optimizer.comprehensive_analysis(target)
        
        return analysis
