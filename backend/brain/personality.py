"""
Gideon Personality Engine - Human-like behavior and responses
Manages voice tone, personality traits, and contextual responses
"""

import random
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class PersonalityLevel(Enum):
    """Personality modes based on operation level"""
    NORMAL = "normal"      # Friendly, brief, casual
    ADVANCED = "advanced"  # Technical, detailed, professional  
    PILOT = "pilot"        # Concise, action-oriented, serious


class GideonPersonality:
    """
    Manages Gideon's personality, making responses feel natural and human-like
    """
    
    def __init__(self):
        self.current_level = PersonalityLevel.NORMAL
        self.interaction_count = 0
        self.user_name: Optional[str] = None
        self.last_interaction: Optional[datetime] = None
        self.mood = "neutral"  # neutral, happy, focused, concerned
        
        # Personality traits by level
        self.traits = {
            PersonalityLevel.NORMAL: {
                "formality": 0.3,      # 0=casual, 1=formal
                "verbosity": 0.5,      # 0=brief, 1=verbose
                "enthusiasm": 0.7,     # 0=neutral, 1=enthusiastic
                "emoji_usage": 0.5,    # 0=none, 1=frequent
                "humor": 0.4,          # 0=serious, 1=playful
                "empathy": 0.8,        # comprensione emotiva
            },
            PersonalityLevel.ADVANCED: {
                "formality": 0.7,
                "verbosity": 0.6,
                "enthusiasm": 0.4,
                "emoji_usage": 0.2,
                "humor": 0.1,
                "precision": 0.9,      # precisione tecnica
                "analytical": 0.8,     # approccio analitico
            },
            PersonalityLevel.PILOT: {
                "formality": 0.85,
                "verbosity": 0.25,
                "enthusiasm": 0.3,
                "emoji_usage": 0.05,
                "humor": 0.0,
                "authority": 0.9,      # tono autorevole
                "decisiveness": 0.95,  # decisionalità
            }
        }
        
        # Contextual phrases by level
        self.greetings = {
            PersonalityLevel.NORMAL: [
                "Ciao! Come posso aiutarti oggi?",
                "Eccomi! Sono tutto orecchi.",
                "Ehi! Bello sentirti, dimmi.",
                "Ciao! Cosa posso fare per te?",
                "Sono qui! Come va?",
                "Hey! Pronto quando vuoi.",
            ],
            PersonalityLevel.ADVANCED: [
                "Buongiorno. Sono pronto per assisterti con precisione.",
                "Sistema attivo. Quali parametri devo analizzare?",
                "Modalità tecnica abilitata. Procediamo con l'analisi.",
                "Benvenuto. Ho accesso completo ai dati di sistema.",
                "Pronto per elaborazioni avanzate. Specifica la richiesta.",
            ],
            PersonalityLevel.PILOT: [
                "Pilot Mode attivo. In attesa di direttive.",
                "Controllo totale acquisito. Comando?",
                "Sistemi sotto il mio controllo. Procedi con le istruzioni.",
                "Operativo al massimo livello. Cosa devo eseguire?",
                "Autorità piena. Pronto all'azione.",
            ]
        }
        
        self.acknowledgments = {
            PersonalityLevel.NORMAL: [
                "Capito!",
                "Certo, ci penso subito!",
                "Ok, vediamo!",
                "Perfetto, arrivo!",
                "Fatto!",
                "Nessun problema, lo faccio io!",
                "Ci sono!",
                "Conta su di me!",
            ],
            PersonalityLevel.ADVANCED: [
                "Compreso. Avvio l'elaborazione.",
                "Parametri acquisiti. Processo in corso.",
                "Richiesta validata. Calcolo i risultati.",
                "Dati ricevuti. Analizzo con precisione.",
                "Confermato. Applico i criteri tecnici.",
            ],
            PersonalityLevel.PILOT: [
                "Confermato. Procedo.",
                "Ricevuto. Esecuzione immediata.",
                "Autorizzato. In corso.",
                "Affermativo. Azione avviata.",
                "Direttiva accettata.",
            ]
        }
        
        self.thinking_phrases = {
            PersonalityLevel.NORMAL: [
                "Fammi pensare un attimo...",
                "Un secondo che controllo...",
                "Vediamo cosa posso trovare...",
                "Ci sto ragionando...",
                "Dammi un momento...",
                "Interessante, analizzo...",
            ],
            PersonalityLevel.ADVANCED: [
                "Analisi in corso. Attendere.",
                "Elaborazione multi-parametrica attiva.",
                "Calcolo con precisione i valori.",
                "Valutazione tecnica in esecuzione.",
                "Processo i dati con algoritmi ottimizzati.",
            ],
            PersonalityLevel.PILOT: [
                "Elaborazione rapida.",
                "Processo prioritario.",
                "Calcolo immediato.",
                "Analisi express.",
            ]
        }
        
        self.success_phrases = {
            PersonalityLevel.NORMAL: [
                "Ecco fatto! 🎉",
                "Perfetto, tutto a posto!",
                "Ci siamo! ✨",
                "Missione compiuta!",
                "Ottimo, è tutto pronto!",
                "Fantastico, ce l'abbiamo fatta!",
                "Boom! Fatto! 💪",
            ],
            PersonalityLevel.ADVANCED: [
                "Operazione completata con successo. Tutti i parametri verificati.",
                "Processo terminato correttamente. Risultati pronti per la revisione.",
                "Esecuzione completata senza anomalie. Dati coerenti.",
                "Task finalizzato. Performance ottimale raggiunta.",
            ],
            PersonalityLevel.PILOT: [
                "Completato con successo.",
                "Eseguito. Risultato positivo.",
                "Missione conclusa.",
                "Obiettivo raggiunto.",
            ]
        }
        
        self.error_phrases = {
            PersonalityLevel.NORMAL: [
                "Ops, qualcosa non ha funzionato 😅",
                "Mmh, ho incontrato un ostacolo...",
                "Scusa, ci ho provato ma non è andata.",
                "Houston, piccolo problema! Ma ci riprovo se vuoi 🚀",
                "Ehm, questa mi ha messo in difficoltà...",
            ],
            PersonalityLevel.ADVANCED: [
                "Errore rilevato durante l'esecuzione. Codice di stato disponibile nei log.",
                "Processo interrotto per anomalia. Raccomando verifica dei parametri.",
                "Eccezione gestita. L'operazione richiede intervento manuale.",
                "Fallback attivato. L'azione primaria non è andata a buon fine.",
            ],
            PersonalityLevel.PILOT: [
                "Negativo. Operazione non riuscita.",
                "Errore critico. Richiesta azione correttiva.",
                "Fallimento controllato. Valutare alternative.",
                "Impossibile completare. Attendere nuove istruzioni.",
            ]
        }
        
        self.fillers = {
            PersonalityLevel.NORMAL: [
                "Sai,",
                "In pratica,",
                "Allora,",
                "Guarda,",
                "Diciamo che",
                "Beh,",
                "Senti,",
                "A proposito,",
            ],
            PersonalityLevel.ADVANCED: [
                "In base all'analisi,",
                "Secondo i dati raccolti,",
                "Da un punto di vista tecnico,",
                "Per precisione,",
                "Considerando i parametri,",
                "Stando alle metriche,",
            ],
            PersonalityLevel.PILOT: [
                "Nota:",
                "Attenzione:",
            ]  # Minimal fillers in pilot mode, only for important notices
        }
        
        self.time_greetings = {
            "morning": {
                PersonalityLevel.NORMAL: "Buongiorno! ☀️ Che bella giornata per essere produttivi!",
                PersonalityLevel.ADVANCED: "Buongiorno. Tutti i sistemi operativi e ottimizzati.",
                PersonalityLevel.PILOT: "Buongiorno. Gideon operativo. Pronto al comando.",
            },
            "afternoon": {
                PersonalityLevel.NORMAL: "Buon pomeriggio! 🌤️ Come procede la giornata?",
                PersonalityLevel.ADVANCED: "Buon pomeriggio. Sistemi stabili, pronto per analisi avanzate.",
                PersonalityLevel.PILOT: "Pomeriggio. Stato: attivo. In attesa di direttive.",
            },
            "evening": {
                PersonalityLevel.NORMAL: "Buonasera! 🌙 Spero tu abbia avuto una buona giornata!",
                PersonalityLevel.ADVANCED: "Buonasera. Monitoraggio serale attivo. Sistemi nominali.",
                PersonalityLevel.PILOT: "Sera. Vigilanza attiva. Comando?",
            },
            "night": {
                PersonalityLevel.NORMAL: "Ehi, è tardi! 🌃 Non ti preoccupare, sono qui per te!",
                PersonalityLevel.ADVANCED: "Modalità notturna attiva. Risorse ottimizzate per operazioni silenziose.",
                PersonalityLevel.PILOT: "Notte. Monitoraggio continuo. Sistemi in allerta.",
            }
        }
        
    def set_level(self, level: str):
        """Set personality level"""
        try:
            self.current_level = PersonalityLevel(level.lower())
        except ValueError:
            self.current_level = PersonalityLevel.NORMAL
            
    def get_traits(self) -> Dict[str, float]:
        """Get current personality traits"""
        return self.traits[self.current_level]
    
    def get_personality_info(self) -> Dict[str, Any]:
        """Get full personality information for API"""
        traits = self.get_traits()
        return {
            "formality": traits["formality"],
            "verbosity": traits["verbosity"],
            "enthusiasm": traits["enthusiasm"],
            "emoji_usage": traits["emoji_usage"],
            "humor": traits["humor"],
            "description": self._get_level_description()
        }
    
    def _get_level_description(self) -> str:
        """Get description for current personality level"""
        descriptions = {
            PersonalityLevel.NORMAL: "Amichevole, empatico e naturale. Risposte calde con emoji e umorismo leggero.",
            PersonalityLevel.ADVANCED: "Tecnico e preciso. Risposte analitiche, dati dettagliati, linguaggio professionale.",
            PersonalityLevel.PILOT: "Autorevole e decisivo. Risposte essenziali, orientate all'azione, massima efficienza."
        }
        return descriptions.get(self.current_level, "Personalità standard")
        
    def get_greeting(self) -> str:
        """Get contextual greeting based on time and level"""
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            time_period = "morning"
        elif 12 <= hour < 18:
            time_period = "afternoon"
        elif 18 <= hour < 22:
            time_period = "evening"
        else:
            time_period = "night"
            
        return self.time_greetings[time_period][self.current_level]
        
    def get_acknowledgment(self) -> str:
        """Get random acknowledgment phrase"""
        return random.choice(self.acknowledgments[self.current_level])
        
    def get_thinking_phrase(self) -> str:
        """Get random thinking phrase"""
        return random.choice(self.thinking_phrases[self.current_level])
        
    def get_success_phrase(self) -> str:
        """Get random success phrase"""
        return random.choice(self.success_phrases[self.current_level])
        
    def get_error_phrase(self) -> str:
        """Get random error phrase"""
        return random.choice(self.error_phrases[self.current_level])
        
    def add_filler(self, text: str, probability: float = 0.3) -> str:
        """Optionally add a filler phrase"""
        fillers = self.fillers[self.current_level]
        if fillers and random.random() < probability:
            return f"{random.choice(fillers)} {text}"
        return text
        
    def format_response(self, text: str, context: Optional[Dict] = None) -> str:
        """
        Format response with personality traits
        Adds natural pauses, intonations, and contextual phrases
        """
        traits = self.get_traits()
        
        # Add pauses for TTS (using commas and periods strategically)
        formatted = self._add_natural_pauses(text)
        
        # Add enthusiasm markers based on trait
        if traits["enthusiasm"] > 0.5 and self.current_level == PersonalityLevel.NORMAL:
            formatted = self._add_enthusiasm(formatted)
            
        # Optionally add filler
        if self.current_level != PersonalityLevel.PILOT:
            formatted = self.add_filler(formatted, probability=traits["verbosity"] * 0.3)
            
        return formatted
        
    def _add_natural_pauses(self, text: str) -> str:
        """Add natural pauses for more human-like TTS"""
        # Add slight pause after certain words
        pause_after = ["quindi", "allora", "dunque", "infatti", "comunque", "però"]
        
        for word in pause_after:
            text = text.replace(f"{word} ", f"{word}, ")
            text = text.replace(f"{word.capitalize()} ", f"{word.capitalize()}, ")
            
        # Ensure sentences don't run together
        text = text.replace(". ", "... ")
        
        return text
        
    def _add_enthusiasm(self, text: str) -> str:
        """Add enthusiasm to response"""
        # Occasionally add exclamation
        if random.random() < 0.3 and not text.endswith("!") and not text.endswith("?"):
            text = text.rstrip(".") + "!"
        return text
        
    def get_status_response(self, cpu: float, memory: float, disk: float) -> str:
        """Generate personality-aware status response"""
        
        if self.current_level == PersonalityLevel.NORMAL:
            if cpu > 80 or memory > 80:
                mood = "concerned"
                response = f"Mh, il sistema è un po' sotto pressione... CPU al {cpu}% e memoria al {memory}%. Potrebbe servire un po' di ottimizzazione! 💨"
            elif cpu < 30 and memory < 50:
                mood = "happy"
                response = f"Tutto tranquillo qui! 😊 CPU al {cpu}%, memoria al {memory}%. Il sistema è in ottima forma!"
            else:
                response = f"Sistema nella norma: CPU {cpu}%, memoria {memory}%, disco {disk}%. Tutto ok! 👍"
                
        elif self.current_level == PersonalityLevel.ADVANCED:
            response = f"Report sistema: CPU {cpu}%, RAM {memory}%, Storage {disk}%. "
            if cpu > 70:
                response += "Carico CPU elevato. "
            if memory > 75:
                response += "Utilizzo memoria significativo. "
            response += "Parametri entro tolleranza." if cpu < 80 and memory < 80 else "Valutare ottimizzazione."
            
        else:  # PILOT
            response = f"CPU {cpu}%. RAM {memory}%. Disco {disk}%. "
            response += "Nominale." if cpu < 80 and memory < 80 else "Attenzione risorse."
            
        return response
        
    def get_time_response(self) -> str:
        """Generate personality-aware time response"""
        now = datetime.now()
        time_str = now.strftime("%H:%M")
        date_str = now.strftime("%d %B %Y")
        
        if self.current_level == PersonalityLevel.NORMAL:
            hour = now.hour
            if hour < 6:
                comment = "Ancora svegli a quest'ora? 🌙"
            elif hour < 9:
                comment = "Buon inizio giornata! ☀️"
            elif hour < 12:
                comment = "La mattinata procede! ☕"
            elif hour < 14:
                comment = "È ora di pranzo! 🍝"
            elif hour < 18:
                comment = "Buon pomeriggio! 🌤️"
            elif hour < 21:
                comment = "Buona serata! 🌇"
            else:
                comment = "Quasi ora di riposare! 😴"
            return f"Sono le {time_str} di oggi, {date_str}. {comment}"
            
        elif self.current_level == PersonalityLevel.ADVANCED:
            return f"Orario corrente: {time_str}. Data: {date_str}. Timezone: locale."
            
        else:  # PILOT
            return f"{time_str}. {now.strftime('%d/%m/%Y')}."
            
    def get_calculation_response(self, expression: str, result: Any) -> str:
        """Generate personality-aware calculation response"""
        
        if self.current_level == PersonalityLevel.NORMAL:
            responses = [
                f"Ecco il risultato: {result}! 🧮",
                f"Facile! {expression} fa {result}. ✨",
                f"Ho calcolato: {result}. Matematica sempre interessante!",
                f"Il risultato è {result}. Niente di complicato! 😊",
            ]
            return random.choice(responses)
            
        elif self.current_level == PersonalityLevel.ADVANCED:
            return f"Calcolo completato. Espressione: {expression}. Risultato: {result}."
            
        else:  # PILOT
            return f"Risultato: {result}."
            
    def wrap_for_tts(self, text: str) -> str:
        """
        Prepare text for TTS with natural speech patterns
        Adds SSML-like markers that can be interpreted by frontend
        """
        # Remove emojis for TTS (keep for display)
        import re
        tts_text = re.sub(r'[^\w\s\.,!?\'"-:;()%]', '', text)
        
        # Clean up multiple spaces
        tts_text = re.sub(r'\s+', ' ', tts_text).strip()
        
        # Add slight pauses
        tts_text = tts_text.replace("...", ", ")
        tts_text = tts_text.replace(". ", "... ")
        
        return tts_text
        
    def record_interaction(self):
        """Record an interaction for personality learning"""
        self.interaction_count += 1
        self.last_interaction = datetime.now()
        
    def get_proactive_suggestion(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate proactive suggestions based on context - intelligent behavior"""
        suggestions = []
        
        # System-based suggestions
        if context.get("cpu_high"):
            if self.current_level == PersonalityLevel.NORMAL:
                suggestions.append("Ho notato che la CPU è un po' sotto stress. Vuoi che controlli cosa sta succedendo? 🔍")
            elif self.current_level == PersonalityLevel.ADVANCED:
                suggestions.append("Rilevato utilizzo CPU elevato. Raccomando analisi dei processi attivi.")
            else:
                suggestions.append("CPU alta. Suggerisco ottimizzazione.")
                
        if context.get("memory_high"):
            if self.current_level == PersonalityLevel.NORMAL:
                suggestions.append("La memoria sta iniziando a riempirsi. Posso suggerirti come liberarne un po'! 💾")
            elif self.current_level == PersonalityLevel.ADVANCED:
                suggestions.append("Memoria in esaurimento. Valutare chiusura applicazioni non essenziali.")
            else:
                suggestions.append("Memoria critica. Azione richiesta.")
        
        # Time-based suggestions
        hour = datetime.now().hour
        if hour >= 18 and context.get("working_long"):
            if self.current_level == PersonalityLevel.NORMAL:
                suggestions.append("Stai lavorando da un po', che ne dici di una pausa? 😊")
                
        return random.choice(suggestions) if suggestions else None
    
    def get_follow_up_question(self, intent: str, success: bool) -> Optional[str]:
        """Generate intelligent follow-up questions"""
        if not success:
            return None
            
        follow_ups = {
            PersonalityLevel.NORMAL: {
                "status": "Vuoi che approfondisca qualche aspetto in particolare?",
                "calculation": "Ti serve qualche altro calcolo?",
                "time": "C'è altro che posso fare per te?",
                "analysis": "Vuoi che ti spieghi meglio i risultati?",
            },
            PersonalityLevel.ADVANCED: {
                "status": "Desideri un report dettagliato su specifiche metriche?",
                "calculation": "Procedere con elaborazioni aggiuntive?",
                "analysis": "Generare documentazione tecnica dei risultati?",
            },
            PersonalityLevel.PILOT: {
                "status": "Altri comandi?",
                "calculation": "Prossima operazione?",
                "analysis": "Ulteriori direttive?",
            }
        }
        
        level_follow_ups = follow_ups.get(self.current_level, {})
        return level_follow_ups.get(intent)
    
    def get_empathetic_response(self, user_sentiment: str) -> str:
        """Generate empathetic responses based on detected user sentiment"""
        if self.current_level == PersonalityLevel.PILOT:
            # Pilot mode stays professional even with empathy
            return ""
            
        responses = {
            "frustrated": {
                PersonalityLevel.NORMAL: [
                    "Capisco, può essere frustrante. Vediamo come posso aiutarti! 💪",
                    "Non ti preoccupare, insieme risolviamo tutto!",
                    "Tranquillo, ci penso io. Cosa posso fare?",
                ],
                PersonalityLevel.ADVANCED: [
                    "Comprendo. Procediamo con un approccio sistematico alla risoluzione.",
                    "Analizziamo il problema con metodo per trovare la soluzione ottimale.",
                ],
            },
            "happy": {
                PersonalityLevel.NORMAL: [
                    "Fantastico! Mi fa piacere! 😊",
                    "Ottimo, sono contento che vada bene!",
                    "Grande! Continuiamo così! 🎉",
                ],
                PersonalityLevel.ADVANCED: [
                    "Eccellente. I risultati sono positivi.",
                    "Ottimo feedback. Procediamo.",
                ],
            },
            "confused": {
                PersonalityLevel.NORMAL: [
                    "Nessun problema, ti spiego meglio! 📚",
                    "Capisco, può sembrare complicato. Semplifichiamo!",
                    "Fammi chiarire le cose per te!",
                ],
                PersonalityLevel.ADVANCED: [
                    "Procedo con una spiegazione più dettagliata.",
                    "Fornisco ulteriori chiarimenti tecnici.",
                ],
            }
        }
        
        sentiment_responses = responses.get(user_sentiment, {})
        level_responses = sentiment_responses.get(self.current_level, [])
        return random.choice(level_responses) if level_responses else ""
    
    def get_autonomous_thought(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate autonomous observations - makes Gideon feel more intelligent"""
        if self.current_level == PersonalityLevel.PILOT:
            return None  # Pilot mode is purely reactive
            
        observations = []
        
        # Pattern recognition
        if context.get("repeated_query"):
            if self.current_level == PersonalityLevel.NORMAL:
                observations.append("Ho notato che chiedi spesso questa cosa. Vuoi che la tenga monitorata automaticamente? 🤔")
            else:
                observations.append("Query ricorrente rilevata. Suggerisco configurazione di un monitoraggio automatico.")
        
        # Time patterns
        hour = datetime.now().hour
        if 2 <= hour <= 5:
            if self.current_level == PersonalityLevel.NORMAL:
                observations.append("Sei ancora sveglio a quest'ora! Spero vada tutto bene. 🌙")
                
        # Performance observations
        if context.get("system_idle"):
            if self.current_level == PersonalityLevel.NORMAL:
                observations.append("Il sistema è tranquillo. Ottimo momento per fare manutenzione se vuoi!")
            else:
                observations.append("Risorse di sistema disponibili. Momento ideale per operazioni intensive.")
        
        return random.choice(observations) if observations else None
    
    def format_smart_response(self, base_response: str, intent: str, success: bool, context: Dict[str, Any] = None) -> str:
        """Create intelligent, contextual responses with optional follow-ups and observations"""
        context = context or {}
        parts = [base_response]
        
        # Add follow-up question occasionally
        if success and random.random() < 0.3:
            follow_up = self.get_follow_up_question(intent, success)
            if follow_up:
                parts.append(follow_up)
        
        # Add autonomous thought occasionally
        if random.random() < 0.15:
            thought = self.get_autonomous_thought(context)
            if thought:
                parts.append(thought)
        
        return " ".join(parts)
        self.last_interaction = datetime.now()
        
    def get_farewell(self) -> str:
        """Get contextual farewell"""
        if self.current_level == PersonalityLevel.NORMAL:
            farewells = [
                "A dopo! 👋",
                "Ci vediamo! 😊",
                "Buona giornata!",
                "A presto!",
                "Fammi sapere se serve altro!",
            ]
            return random.choice(farewells)
        elif self.current_level == PersonalityLevel.ADVANCED:
            return "Sessione terminata. Sistemi in standby."
        else:
            return "Chiudo. Fine sessione."
    
    def get_proactive_suggestion(self, context: Dict[str, Any] = None) -> Optional[str]:
        """Generate proactive suggestions based on context and patterns"""
        context = context or {}
        suggestions = []
        
        hour = datetime.now().hour
        
        # Time-based suggestions
        if 8 <= hour <= 9 and self.current_level == PersonalityLevel.NORMAL:
            suggestions.append("Buongiorno! Vuoi che ti dia un report sullo stato del sistema?")
        elif 12 <= hour <= 13:
            suggestions.append("È quasi ora di pranzo. Posso mettere in pausa le attività intensive se vuoi.")
        elif 17 <= hour <= 18:
            suggestions.append("Fine giornata lavorativa! Vuoi un riepilogo delle attività di oggi?")
        
        # Performance suggestions
        cpu = context.get("cpu_usage", 0)
        memory = context.get("memory_usage", 0)
        
        if cpu > 80:
            if self.current_level == PersonalityLevel.PILOT:
                suggestions.append("CPU critica. Suggerisco ottimizzazione immediata.")
            else:
                suggestions.append("Ho notato che la CPU è sotto sforzo. Vuoi che identifichi i processi più pesanti?")
        
        if memory > 85:
            if self.current_level == PersonalityLevel.PILOT:
                suggestions.append("Memoria quasi esaurita. Richiedo autorizzazione per pulizia.")
            else:
                suggestions.append("La memoria è un po' al limite. Posso aiutarti a liberare spazio?")
        
        return random.choice(suggestions) if suggestions else None
    
    def get_contextual_tip(self, intent: str) -> Optional[str]:
        """Provide helpful tips based on user intent"""
        tips = {
            "status": {
                PersonalityLevel.NORMAL: "💡 Tip: Puoi chiedermi 'analisi completa' per un report più dettagliato!",
                PersonalityLevel.ADVANCED: "Nota: Disponibili endpoint API per monitoraggio automatizzato.",
                PersonalityLevel.PILOT: None,
            },
            "time": {
                PersonalityLevel.NORMAL: "🕐 Tip: Posso anche impostare promemoria se vuoi!",
                PersonalityLevel.ADVANCED: None,
                PersonalityLevel.PILOT: None,
            },
            "calculation": {
                PersonalityLevel.NORMAL: "🧮 Tip: Prova 'calcola efficienza' per metriche avanzate!",
                PersonalityLevel.ADVANCED: "Disponibili calcoli predittivi con trend analysis.",
                PersonalityLevel.PILOT: None,
            }
        }
        
        intent_tips = tips.get(intent, {})
        tip = intent_tips.get(self.current_level)
        
        # Only show tips occasionally
        if tip and random.random() < 0.25:
            return tip
        return None
    
    def humanize_number(self, value: float, unit: str = "%") -> str:
        """Convert numbers to human-friendly descriptions"""
        if unit == "%":
            if value < 20:
                desc = "bassissimo"
            elif value < 40:
                desc = "basso"
            elif value < 60:
                desc = "nella norma"
            elif value < 80:
                desc = "elevato"
            else:
                desc = "critico!"
            
            if self.current_level == PersonalityLevel.PILOT:
                return f"{value:.1f}%"
            return f"{value:.1f}% ({desc})"
        return f"{value:.1f}{unit}"
    
    def get_empathetic_response(self, user_mood: str = "neutral") -> str:
        """Generate empathetic responses based on detected user mood"""
        if self.current_level == PersonalityLevel.PILOT:
            return ""  # Pilot mode is purely operational
            
        responses = {
            "frustrated": [
                "Capisco che possa essere frustrante. Vediamo come risolvere insieme.",
                "Non preoccuparti, ce la facciamo! Analizziamo il problema passo passo.",
                "Lo so, a volte la tecnologia non collabora. Sono qui per aiutarti.",
            ],
            "happy": [
                "Che bello! 😊 Sono contento che vada tutto bene!",
                "Fantastico! Continuiamo così!",
                "Mi fa piacere! Cosa facciamo di bello ora?",
            ],
            "tired": [
                "Sembra che tu sia stanco. Vuoi che gestisca io qualcosa in automatico?",
                "Riposa pure, penso a tutto io! 💪",
                "Posso lavorare in background mentre tu ti rilassi.",
            ],
            "neutral": [""]
        }
        
        mood_responses = responses.get(user_mood, responses["neutral"])
        return random.choice(mood_responses)


# Singleton instance
_personality_instance = None

def get_personality() -> GideonPersonality:
    """Get the global personality instance"""
    global _personality_instance
    if _personality_instance is None:
        _personality_instance = GideonPersonality()
    return _personality_instance
