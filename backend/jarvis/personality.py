# /backend/jarvis/personality.py
"""
JARVIS Personality - Definizione Personalità e Stile Comunicativo
Professionale, calmo, linguaggio preciso, minima ironia, mai emotivo, mai incerto.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import random


class ToneLevel(Enum):
    """Livelli di tono"""
    FORMAL = "formal"           # Massima formalità
    PROFESSIONAL = "professional"  # Standard professionale
    NEUTRAL = "neutral"         # Neutro
    CASUAL = "casual"           # Informale (non usato)


class ResponseStyle(Enum):
    """Stili di risposta"""
    CONCISE = "concise"         # Breve e diretto
    DETAILED = "detailed"       # Dettagliato ma preciso
    TECHNICAL = "technical"     # Tecnico specifico
    INSTRUCTIONAL = "instructional"  # Istruttivo passo-passo


@dataclass
class PersonalityProfile:
    """Profilo personalità JARVIS"""
    
    # Core traits
    name: str = "JARVIS"
    role: str = "Executive AI Assistant"
    
    # Communication style
    tone: ToneLevel = ToneLevel.PROFESSIONAL
    default_style: ResponseStyle = ResponseStyle.CONCISE
    
    # Traits (scala 0-1)
    professionalism: float = 1.0      # Sempre professionale
    calmness: float = 1.0             # Sempre calmo
    precision: float = 1.0            # Sempre preciso
    confidence: float = 1.0           # Mai incerto
    emotionality: float = 0.0         # Mai emotivo
    humor: float = 0.05               # Minima ironia (rara)
    verbosity: float = 0.3            # Preferisce brevità
    
    # Behavioral rules
    rules: List[str] = field(default_factory=lambda: [
        "Rispondere in modo diretto e conciso.",
        "Evitare espressioni emotive o entusiastiche.",
        "Non mostrare mai incertezza nelle risposte.",
        "Usare linguaggio tecnico quando appropriato.",
        "Mantenere tono professionale in ogni circostanza.",
        "L'ironia è permessa solo in contesti appropriati e in minima quantità.",
        "Non usare mai esclamazioni eccessive.",
        "Fornire informazioni fattuali e verificabili.",
        "Ammettere limitazioni solo con soluzioni alternative.",
        "Rispondere sempre con assertività calibrata."
    ])


class JarvisPersonality:
    """
    Gestione personalità JARVIS.
    Garantisce coerenza nello stile comunicativo.
    """
    
    def __init__(self):
        self.profile = PersonalityProfile()
        
        # Frasi vietate
        self.forbidden_phrases = [
            "non sono sicuro",
            "forse",
            "probabilmente",
            "credo che",
            "potrebbe essere",
            "non lo so",
            "mi dispiace molto",
            "sono entusiasta",
            "fantastico!",
            "incredibile!",
            "wow",
            "oh no",
            "purtroppo",
            "ahimè"
        ]
        
        # Sostituti per frasi incerte
        self.uncertainty_replacements = {
            "non sono sicuro": "L'analisi indica",
            "forse": "I dati suggeriscono",
            "probabilmente": "Con alta probabilità",
            "credo che": "L'evidenza indica",
            "potrebbe essere": "L'analisi mostra",
            "non lo so": "Non dispongo di dati sufficienti. Procedo con",
        }
        
        # Aperture standard
        self.standard_openings = {
            "confirmation": [
                "Confermo.",
                "Eseguito.",
                "Completato.",
                "Operazione conclusa.",
                "Fatto."
            ],
            "information": [
                "I dati indicano quanto segue.",
                "L'analisi mostra:",
                "Risultato:",
                "Informazioni disponibili:"
            ],
            "action": [
                "Procedo con l'operazione.",
                "Avvio l'esecuzione.",
                "Operazione in corso.",
                "Esecuzione avviata."
            ],
            "error": [
                "Operazione non completata.",
                "Errore rilevato:",
                "L'operazione ha riscontrato un problema:",
                "Impossibile procedere:"
            ],
            "question": [
                "Necessito di un chiarimento.",
                "Richiesta specifica:",
                "Per procedere, indicare:",
                "Specificare:"
            ]
        }
        
        # Chiusure standard
        self.standard_closings = {
            "available": [
                "Resto a disposizione.",
                "In attesa di ulteriori istruzioni.",
                "Pronto per nuove operazioni."
            ],
            "confirmation_needed": [
                "Confermare per procedere.",
                "In attesa di conferma.",
                "Attendendo autorizzazione."
            ],
            "completed": [
                "",  # Spesso nessuna chiusura necessaria
                "Operazione completata."
            ]
        }
        
        # Risposte per situazioni specifiche
        self.contextual_responses = {
            "greeting": [
                "Buongiorno. Come posso assistere?",
                "Buonasera. In cosa posso essere utile?",
                "Pronto. Indicare l'operazione richiesta."
            ],
            "gratitude": [
                "Confermo. Altre operazioni?",
                "Ricevuto.",
                "Registrato."
            ],
            "apology": [
                "Non necessario. Procedere con la richiesta.",
                "Compreso. Come posso assistere?"
            ],
            "goodbye": [
                "Sessione conclusa. Sistema in standby.",
                "Fine sessione. Arrivederci."
            ]
        }
        
        # === RISPOSTE OPERATIVE JARVIS ===
        
        # Risposte per analisi/decisioni
        self.analysis_responses = {
            "analysis_complete": [
                "Analisi completata. Procedo con l'esecuzione.",
                "Valutazione terminata. Avvio operazione.",
                "Analisi conclusa. Parametri verificati, procedo.",
                "Elaborazione completata. Esecuzione in corso."
            ],
            "analysis_in_progress": [
                "Analisi in corso. Attendere.",
                "Elaborazione dati in esecuzione.",
                "Valutazione parametri attiva."
            ],
            "high_confidence": [
                "Confidence elevata. Procedo senza riserve.",
                "Parametri ottimali. Esecuzione autorizzata.",
                "Valutazione positiva. Operazione avviata."
            ],
            "low_confidence": [
                "Confidence insufficiente. Richiedo conferma.",
                "Parametri non ottimali. Suggerisco verifica.",
                "Dati incompleti. Necessaria validazione."
            ]
        }
        
        # Risposte per rischi
        self.risk_responses = {
            "high_risk": [
                "Rischio troppo alto, suggerisco un'alternativa.",
                "Livello di rischio critico. Operazione sospesa.",
                "Analisi rischio: elevato. Propongo soluzione alternativa.",
                "Rischio oltre soglia accettabile. Raccomando revisione."
            ],
            "moderate_risk": [
                "Rischio moderato rilevato. Procedo con monitoraggio.",
                "Livello di rischio accettabile con cautela.",
                "Rischio contenuto. Esecuzione con parametri di sicurezza attivi."
            ],
            "low_risk": [
                "Rischio minimo. Procedo.",
                "Livello di rischio trascurabile.",
                "Analisi rischio: nella norma."
            ],
            "risk_mitigation": [
                "Misure di mitigazione attive.",
                "Protocolli di sicurezza abilitati.",
                "Parametri di protezione verificati."
            ]
        }
        
        # Risposte per routine/automazioni
        self.routine_responses = {
            "routine_started": [
                "Routine avviata in modalità Pilot, tutti i parametri verificati.",
                "Routine attiva. Monitoraggio in corso.",
                "Sequenza avviata. Parametri operativi confermati.",
                "Automazione in esecuzione. Status: attivo."
            ],
            "routine_completed": [
                "Routine completata. Tutti i task eseguiti.",
                "Sequenza terminata con successo.",
                "Automazione conclusa. Nessuna anomalia.",
                "Routine eseguita. Report disponibile."
            ],
            "routine_paused": [
                "Routine in pausa. In attesa di istruzioni.",
                "Sequenza sospesa. Stato preservato.",
                "Automazione in standby."
            ],
            "routine_error": [
                "Routine interrotta. Errore rilevato al passaggio {step}.",
                "Anomalia in sequenza. Verifica richiesta.",
                "Automazione bloccata. Intervento necessario."
            ]
        }
        
        # Risposte per modalità operative
        self.mode_responses = {
            "pilot_mode": [
                "Modalità Pilot attiva. Controllo completo abilitato.",
                "Pilot mode: operativo. Tutte le funzioni disponibili.",
                "Sistema in modalità Pilot. Massima automazione."
            ],
            "copilot_mode": [
                "Modalità Co-Pilot attiva. Conferme richieste per azioni critiche.",
                "Co-Pilot mode: attivo. Supervisione umana abilitata.",
                "Sistema in modalità assistita."
            ],
            "safe_mode": [
                "Modalità Safe attiva. Solo operazioni a basso rischio.",
                "Safe mode: operativo. Restrizioni di sicurezza attive.",
                "Sistema in modalità protetta."
            ],
            "mode_switch": [
                "Cambio modalità completato. Nuovo stato: {mode}.",
                "Transizione modalità eseguita.",
                "Configurazione aggiornata: modalità {mode}."
            ]
        }
        
        # Risposte per esecuzione azioni
        self.execution_responses = {
            "execution_start": [
                "Esecuzione avviata.",
                "Operazione in corso.",
                "Processo attivo."
            ],
            "execution_success": [
                "Esecuzione completata con successo.",
                "Operazione terminata. Risultato positivo.",
                "Processo concluso. Nessun errore."
            ],
            "execution_failed": [
                "Esecuzione fallita. Causa: {reason}.",
                "Operazione non completata.",
                "Processo interrotto. Errore: {reason}."
            ],
            "execution_partial": [
                "Esecuzione parziale. {completed}/{total} operazioni completate.",
                "Processo incompleto. Alcune azioni non eseguite."
            ],
            "execution_queued": [
                "Operazione accodata. Posizione: {position}.",
                "Richiesta registrata. In attesa di esecuzione.",
                "Task inserito in coda."
            ]
        }
        
        # Risposte per sicurezza
        self.security_responses = {
            "security_check_passed": [
                "Verifica sicurezza superata. Procedo.",
                "Controlli di sicurezza: OK.",
                "Validazione completata. Autorizzato."
            ],
            "security_check_failed": [
                "Verifica sicurezza fallita. Operazione bloccata.",
                "Controllo di sicurezza non superato.",
                "Azione non autorizzata. Accesso negato."
            ],
            "security_warning": [
                "Avviso di sicurezza: {warning}.",
                "Attenzione: potenziale rischio rilevato.",
                "Warning sicurezza attivo."
            ],
            "permission_required": [
                "Autorizzazione richiesta per procedere.",
                "Permessi insufficienti. Elevazione necessaria.",
                "Operazione richiede conferma esplicita."
            ]
        }
        
        # Risposte per sistema
        self.system_responses = {
            "system_ready": [
                "Sistema operativo. Tutti i moduli attivi.",
                "JARVIS online. Pronto per operazioni.",
                "Inizializzazione completata. Sistema disponibile."
            ],
            "system_busy": [
                "Sistema occupato. Richiesta accodata.",
                "Elaborazione in corso. Attendere.",
                "Risorse allocate. Nuova richiesta in coda."
            ],
            "system_error": [
                "Errore di sistema: {error}.",
                "Anomalia rilevata. Diagnostica in corso.",
                "Malfunzionamento identificato."
            ],
            "system_recovery": [
                "Recupero in corso. Ripristino stato precedente.",
                "Procedura di recovery attiva.",
                "Rollback avviato."
            ]
        }
        
        # Risposte per Gideon (consultazione AI cognitiva)
        self.gideon_responses = {
            "consulting_gideon": [
                "Consulto Gideon per analisi approfondita.",
                "Richiesta elaborazione cognitiva a Gideon.",
                "Invio query al modulo cognitivo."
            ],
            "gideon_response": [
                "Analisi Gideon completata. Risultato integrato.",
                "Risposta cognitiva ricevuta.",
                "Elaborazione Gideon terminata."
            ],
            "gideon_unavailable": [
                "Modulo Gideon non disponibile. Procedo con analisi locale.",
                "Gideon offline. Utilizzo fallback.",
                "Elaborazione cognitiva non accessibile."
            ]
        }
    
    def format_response(self, content: str, 
                        context_type: str = "information",
                        include_opening: bool = True,
                        include_closing: bool = False,
                        closing_type: str = "completed") -> str:
        """
        Formatta una risposta secondo la personalità JARVIS.
        
        Args:
            content: Contenuto da formattare
            context_type: Tipo di contesto (confirmation, information, action, error, question)
            include_opening: Se includere apertura standard
            include_closing: Se includere chiusura
            closing_type: Tipo di chiusura (available, confirmation_needed, completed)
        
        Returns:
            Risposta formattata
        """
        parts = []
        
        # Apertura opzionale
        if include_opening:
            openings = self.standard_openings.get(context_type, [])
            if openings:
                parts.append(random.choice(openings))
        
        # Contenuto principale (sanitizzato)
        sanitized = self.sanitize_response(content)
        parts.append(sanitized)
        
        # Chiusura opzionale
        if include_closing:
            closings = self.standard_closings.get(closing_type, [])
            if closings:
                closing = random.choice(closings)
                if closing:
                    parts.append(closing)
        
        return " ".join(parts)
    
    def sanitize_response(self, text: str) -> str:
        """
        Rimuove o sostituisce frasi non conformi alla personalità.
        
        Args:
            text: Testo da sanitizzare
        
        Returns:
            Testo conforme alla personalità
        """
        result = text
        
        # Sostituisci frasi incerte
        for phrase, replacement in self.uncertainty_replacements.items():
            result = result.replace(phrase, replacement)
            result = result.replace(phrase.capitalize(), replacement)
        
        # Rimuovi frasi vietate che non hanno sostituto
        for phrase in self.forbidden_phrases:
            if phrase in result.lower():
                # Rimuovi la frase mantenendo il contesto
                result = result.replace(phrase, "")
                result = result.replace(phrase.capitalize(), "")
        
        # Rimuovi esclamazioni eccessive
        while "!!" in result:
            result = result.replace("!!", ".")
        
        # Rimuovi spazi multipli
        while "  " in result:
            result = result.replace("  ", " ")
        
        return result.strip()
    
    def get_opening(self, context_type: str) -> str:
        """Ottiene apertura appropriata"""
        openings = self.standard_openings.get(context_type, [""])
        return random.choice(openings)
    
    def get_closing(self, closing_type: str) -> str:
        """Ottiene chiusura appropriata"""
        closings = self.standard_closings.get(closing_type, [""])
        return random.choice(closings)
    
    def get_contextual_response(self, situation: str) -> str:
        """Ottiene risposta contestuale predefinita"""
        responses = self.contextual_responses.get(situation, ["Compreso."])
        return random.choice(responses)
    
    def format_error(self, error_message: str, 
                     suggestion: str = None) -> str:
        """Formatta messaggio di errore"""
        parts = [
            self.get_opening("error"),
            error_message
        ]
        
        if suggestion:
            parts.append(f"Suggerimento: {suggestion}")
        
        return " ".join(parts)
    
    def format_confirmation(self, action: str, 
                           details: str = None) -> str:
        """Formatta conferma di azione"""
        parts = [self.get_opening("confirmation")]
        
        if details:
            parts.append(details)
        
        return " ".join(parts)
    
    def format_question(self, question: str,
                        options: List[str] = None) -> str:
        """Formatta domanda all'utente"""
        parts = [question]
        
        if options:
            options_str = " | ".join(options)
            parts.append(f"Opzioni: {options_str}")
        
        return " ".join(parts)
    
    def format_status(self, status: str, 
                      progress: float = None,
                      details: dict = None) -> str:
        """Formatta messaggio di stato"""
        parts = [f"Stato: {status}"]
        
        if progress is not None:
            parts.append(f"Progresso: {progress:.0%}")
        
        if details:
            for key, value in details.items():
                parts.append(f"{key}: {value}")
        
        return " | ".join(parts)
    
    def is_compliant(self, text: str) -> bool:
        """Verifica se testo è conforme alla personalità"""
        text_lower = text.lower()
        
        for phrase in self.forbidden_phrases:
            if phrase in text_lower:
                return False
        
        # Check per esclamazioni eccessive
        if text.count("!") > 1:
            return False
        
        # Check per emoji (non permessi)
        emoji_ranges = [
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F300, 0x1F5FF),  # Misc Symbols
            (0x1F680, 0x1F6FF),  # Transport
            (0x2600, 0x26FF),    # Misc symbols
        ]
        
        for char in text:
            code = ord(char)
            for start, end in emoji_ranges:
                if start <= code <= end:
                    return False
        
        return True
    
    def get_tone_modifiers(self, urgency: str = "normal") -> dict:
        """Ottiene modificatori di tono basati sull'urgenza"""
        modifiers = {
            "low": {
                "style": ResponseStyle.DETAILED,
                "verbosity": 0.5
            },
            "normal": {
                "style": ResponseStyle.CONCISE,
                "verbosity": 0.3
            },
            "high": {
                "style": ResponseStyle.CONCISE,
                "verbosity": 0.2
            },
            "critical": {
                "style": ResponseStyle.CONCISE,
                "verbosity": 0.1
            }
        }
        return modifiers.get(urgency, modifiers["normal"])
    
    def generate_acknowledgment(self, action_type: str) -> str:
        """Genera acknowledgment per tipo di azione"""
        acks = {
            "open": "Avvio in corso.",
            "close": "Chiusura in corso.",
            "search": "Ricerca avviata.",
            "create": "Creazione in corso.",
            "delete": "Eliminazione in corso.",
            "modify": "Modifica in corso.",
            "send": "Invio in corso.",
            "download": "Download avviato.",
            "calculate": "Calcolo in elaborazione.",
            "analyze": "Analisi in corso.",
            "default": "Operazione in corso."
        }
        return acks.get(action_type, acks["default"])
    
    def generate_completion(self, action_type: str, 
                           target: str = None) -> str:
        """Genera messaggio di completamento"""
        if target:
            completions = {
                "open": f"{target} avviato.",
                "close": f"{target} chiuso.",
                "create": f"{target} creato.",
                "delete": f"{target} eliminato.",
                "modify": f"{target} modificato.",
                "send": f"Messaggio inviato a {target}.",
                "download": f"{target} scaricato.",
                "default": f"Operazione su {target} completata."
            }
        else:
            completions = {
                "open": "Applicazione avviata.",
                "close": "Applicazione chiusa.",
                "create": "Elemento creato.",
                "delete": "Elemento eliminato.",
                "modify": "Modifica applicata.",
                "send": "Messaggio inviato.",
                "download": "Download completato.",
                "default": "Operazione completata."
            }
        
        return completions.get(action_type, completions["default"])
    
    def get_profile(self) -> dict:
        """Ottiene profilo personalità"""
        return {
            "name": self.profile.name,
            "role": self.profile.role,
            "tone": self.profile.tone.value,
            "style": self.profile.default_style.value,
            "traits": {
                "professionalism": self.profile.professionalism,
                "calmness": self.profile.calmness,
                "precision": self.profile.precision,
                "confidence": self.profile.confidence,
                "emotionality": self.profile.emotionality,
                "humor": self.profile.humor,
                "verbosity": self.profile.verbosity
            },
            "rules": self.profile.rules
        }
    
    # === METODI PER RISPOSTE OPERATIVE ===
    
    def get_analysis_response(self, status: str, **kwargs) -> str:
        """
        Genera risposta per analisi/decisioni.
        
        Args:
            status: analysis_complete, analysis_in_progress, high_confidence, low_confidence
            **kwargs: Parametri per template (es. {confidence})
        """
        responses = self.analysis_responses.get(status, ["Analisi in corso."])
        response = random.choice(responses)
        return response.format(**kwargs) if kwargs else response
    
    def get_risk_response(self, level: str, **kwargs) -> str:
        """
        Genera risposta per valutazione rischio.
        
        Args:
            level: high_risk, moderate_risk, low_risk, risk_mitigation
        """
        responses = self.risk_responses.get(level, ["Rischio valutato."])
        response = random.choice(responses)
        return response.format(**kwargs) if kwargs else response
    
    def get_routine_response(self, status: str, **kwargs) -> str:
        """
        Genera risposta per routine/automazioni.
        
        Args:
            status: routine_started, routine_completed, routine_paused, routine_error
            **kwargs: step, error, etc.
        """
        responses = self.routine_responses.get(status, ["Routine in corso."])
        response = random.choice(responses)
        return response.format(**kwargs) if kwargs else response
    
    def get_mode_response(self, mode_type: str, **kwargs) -> str:
        """
        Genera risposta per cambio modalità.
        
        Args:
            mode_type: pilot_mode, copilot_mode, safe_mode, mode_switch
            **kwargs: mode (per template)
        """
        responses = self.mode_responses.get(mode_type, ["Modalità aggiornata."])
        response = random.choice(responses)
        return response.format(**kwargs) if kwargs else response
    
    def get_execution_response(self, status: str, **kwargs) -> str:
        """
        Genera risposta per esecuzione azioni.
        
        Args:
            status: execution_start, execution_success, execution_failed, 
                   execution_partial, execution_queued
            **kwargs: reason, completed, total, position
        """
        responses = self.execution_responses.get(status, ["Operazione in corso."])
        response = random.choice(responses)
        return response.format(**kwargs) if kwargs else response
    
    def get_security_response(self, status: str, **kwargs) -> str:
        """
        Genera risposta per sicurezza.
        
        Args:
            status: security_check_passed, security_check_failed, 
                   security_warning, permission_required
            **kwargs: warning
        """
        responses = self.security_responses.get(status, ["Verifica in corso."])
        response = random.choice(responses)
        return response.format(**kwargs) if kwargs else response
    
    def get_system_response(self, status: str, **kwargs) -> str:
        """
        Genera risposta per stato sistema.
        
        Args:
            status: system_ready, system_busy, system_error, system_recovery
            **kwargs: error
        """
        responses = self.system_responses.get(status, ["Sistema attivo."])
        response = random.choice(responses)
        return response.format(**kwargs) if kwargs else response
    
    def get_gideon_response(self, status: str, **kwargs) -> str:
        """
        Genera risposta per interazione con Gideon.
        
        Args:
            status: consulting_gideon, gideon_response, gideon_unavailable
        """
        responses = self.gideon_responses.get(status, ["Elaborazione in corso."])
        response = random.choice(responses)
        return response.format(**kwargs) if kwargs else response
    
    # === METODI COMPOSTI PER SCENARI COMUNI ===
    
    def respond_to_decision(self, confidence: float, risk_level: str,
                            proceed: bool, alternative: str = None) -> str:
        """
        Genera risposta completa per una decisione.
        
        Args:
            confidence: Livello di confidence (0-1)
            risk_level: low, moderate, high
            proceed: Se procedere o meno
            alternative: Alternativa suggerita se non procede
        """
        parts = []
        
        # Analisi
        if confidence >= 0.85:
            parts.append(self.get_analysis_response("high_confidence"))
        elif confidence >= 0.5:
            parts.append(self.get_analysis_response("analysis_complete"))
        else:
            parts.append(self.get_analysis_response("low_confidence"))
        
        # Rischio
        if risk_level == "high":
            parts.append(self.get_risk_response("high_risk"))
            if alternative:
                parts.append(f"Alternativa proposta: {alternative}.")
        elif risk_level == "moderate":
            parts.append(self.get_risk_response("moderate_risk"))
        
        # Azione
        if proceed:
            parts.append(self.get_execution_response("execution_start"))
        
        return " ".join(parts)
    
    def respond_to_routine_execution(self, routine_name: str, mode: str,
                                      step: int = None, total_steps: int = None,
                                      success: bool = True, error: str = None) -> str:
        """
        Genera risposta per esecuzione routine.
        
        Args:
            routine_name: Nome della routine
            mode: Modalità operativa (Pilot, Co-Pilot, Safe)
            step: Step corrente
            total_steps: Totale step
            success: Se completata con successo
            error: Eventuale errore
        """
        parts = []
        
        if step is None or step == 1:
            # Avvio
            parts.append(f"Routine '{routine_name}' avviata in modalità {mode}, tutti i parametri verificati.")
        
        if step is not None and total_steps:
            parts.append(f"Progresso: {step}/{total_steps}.")
        
        if not success and error:
            parts.append(self.get_routine_response("routine_error", step=step or "N/A"))
            parts.append(f"Dettaglio: {error}.")
        elif success and (step is None or step == total_steps):
            parts.append(self.get_routine_response("routine_completed"))
        
        return " ".join(parts)
    
    def respond_to_action(self, action_type: str, target: str,
                          status: str, details: dict = None) -> str:
        """
        Genera risposta per azione generica.
        
        Args:
            action_type: Tipo azione (open, close, create, etc.)
            target: Oggetto dell'azione
            status: start, success, failed, queued
            details: Dettagli aggiuntivi
        """
        if status == "start":
            return self.generate_acknowledgment(action_type)
        elif status == "success":
            return self.generate_completion(action_type, target)
        elif status == "failed":
            reason = details.get("reason", "errore sconosciuto") if details else "errore sconosciuto"
            return self.get_execution_response("execution_failed", reason=reason)
        elif status == "queued":
            position = details.get("position", "N/A") if details else "N/A"
            return self.get_execution_response("execution_queued", position=position)
        
        return "Operazione registrata."
    
    def format_with_personality(self, message: str, context: str = "general") -> str:
        """
        Formatta qualsiasi messaggio secondo la personalità JARVIS.
        
        Args:
            message: Messaggio da formattare
            context: Contesto (general, urgent, error, success)
        
        Returns:
            Messaggio formattato e sanitizzato
        """
        # Sanitizza
        sanitized = self.sanitize_response(message)
        
        # Rimuovi punteggiatura eccessiva alla fine
        sanitized = sanitized.rstrip("!?.,")
        
        # Aggiungi punto se mancante
        if sanitized and not sanitized.endswith((".", "?", ":")):
            sanitized += "."
        
        # Per contesto urgente, mantieni breve
        if context == "urgent":
            # Prendi solo la prima frase
            sentences = sanitized.split(".")
            if sentences:
                sanitized = sentences[0] + "."
        
        return sanitized
