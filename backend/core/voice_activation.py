"""
🎤 GIDEON 3.0 - Voice Activation Manager
Gestione frasi di attivazione e trigger vocali
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum


class ActivationTrigger(Enum):
    """Tipi di trigger di attivazione"""
    MODE_CHANGE = "mode_change"
    LEVEL_CHANGE = "level_change"
    COMMAND = "command"
    WAKE_WORD = "wake_word"


class VoiceActivationManager:
    """
    Gestisce le frasi di attivazione vocale per cambiare modalità e livelli
    
    Trigger predefiniti:
    - "Assistente modalità pilota" → PILOT mode
    - "Assistente secondo te" → Modalità NORMAL o ADVANCED (toggle)
    - "Assistente modalità executive" → EXECUTIVE mode
    - "Assistente modalità copilot" → COPILOT mode
    - "Assistente pausa" → PASSIVE mode
    """
    
    def __init__(self, mode_manager=None):
        self.mode_manager = mode_manager
        self._last_activation = None
        self._activation_history = []
        
        # Trigger phrases per modalità operative
        self._mode_triggers = {
            # PILOT Mode
            "pilot": [
                r"assistente\s+modalit[àa]\s+pilot[ao]?",
                r"attiva\s+modalit[àa]\s+pilot[ao]?",
                r"vai\s+in\s+pilot",
                r"pilot\s+mode",
                r"modalit[àa]\s+autonoma",
                r"prendi\s+il\s+controllo",
                r"hands[\s\-]?free"
            ],
            # EXECUTIVE Mode (Jarvis)
            "executive": [
                r"assistente\s+modalit[àa]\s+executive",
                r"attiva\s+jarvis",
                r"jarvis\s+mode",
                r"modalit[àa]\s+jarvis",
                r"supervisione\s+globale",
                r"prendi\s+il\s+comando\s+totale"
            ],
            # COPILOT Mode
            "copilot": [
                r"assistente\s+modalit[àa]\s+copilot",
                r"torna\s+in\s+copilot",
                r"modalit[àa]\s+assistita",
                r"chiedi\s+conferma"
            ],
            # PASSIVE Mode
            "passive": [
                r"assistente\s+pausa",
                r"assistente\s+stop",
                r"modalit[àa]\s+passiva",
                r"solo\s+ascolto",
                r"non\s+fare\s+niente",
                r"fermati"
            ]
        }
        
        # Trigger phrases per livello risposta
        self._level_triggers = {
            # Toggle NORMAL/ADVANCED
            "toggle": [
                r"assistente\s+secondo\s+te",
                r"cambia\s+livello",
                r"pi[uù]\s+dettagli",
                r"meno\s+dettagli"
            ],
            # Forza NORMAL
            "normal": [
                r"modalit[àa]\s+normale",
                r"risposte\s+semplici",
                r"parla\s+semplice",
                r"sii\s+breve"
            ],
            # Forza ADVANCED
            "advanced": [
                r"modalit[àa]\s+avanzata",
                r"analisi\s+completa",
                r"dettagli\s+tecnici",
                r"sii\s+dettagliato",
                r"approfondisci"
            ]
        }
        
        # Wake words generici (attivano l'ascolto)
        self._wake_words = [
            r"^(hey\s+)?gideon",
            r"^(hey\s+)?jarvis",
            r"^assistente",
            r"^ehi\s+assistente"
        ]
        
        # Comandi speciali
        self._special_commands = {
            "emergency_stop": [
                r"emergenza",
                r"stop\s+tutto",
                r"kill\s+switch",
                r"ferma\s+tutto\s+subito"
            ],
            "status": [
                r"che\s+modalit[àa]\s+sei",
                r"stato\s+sistema",
                r"come\s+stai"
            ]
        }
        
        # Callbacks per attivazioni
        self._activation_callbacks: List[Callable] = []
    
    def process_input(self, text: str) -> Optional[Dict]:
        """
        Processa input testuale/vocale e rileva trigger
        
        Args:
            text: Testo da analizzare
            
        Returns:
            Dict con tipo trigger e azione, o None se nessun trigger
        """
        text_lower = text.lower().strip()
        
        result = None
        
        # Check wake words
        if self._check_wake_word(text_lower):
            # Rimuovi wake word per analisi successiva
            text_lower = self._remove_wake_word(text_lower)
        
        # Check mode triggers
        mode_result = self._check_mode_trigger(text_lower)
        if mode_result:
            result = mode_result
        
        # Check level triggers
        level_result = self._check_level_trigger(text_lower)
        if level_result:
            result = level_result
        
        # Check special commands
        special_result = self._check_special_commands(text_lower)
        if special_result:
            result = special_result
        
        # Log e callback se trovato trigger
        if result:
            self._last_activation = datetime.now()
            self._activation_history.append({
                "text": text,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Notifica callbacks
            for callback in self._activation_callbacks:
                try:
                    callback(result)
                except Exception:
                    pass
        
        return result
    
    def _check_wake_word(self, text: str) -> bool:
        """Verifica presenza wake word"""
        for pattern in self._wake_words:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _remove_wake_word(self, text: str) -> str:
        """Rimuove wake word dal testo"""
        for pattern in self._wake_words:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        return text
    
    def _check_mode_trigger(self, text: str) -> Optional[Dict]:
        """Verifica trigger per cambio modalità"""
        for mode, patterns in self._mode_triggers.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return {
                        "type": ActivationTrigger.MODE_CHANGE,
                        "target_mode": mode,
                        "pattern_matched": pattern,
                        "original_text": text
                    }
        return None
    
    def _check_level_trigger(self, text: str) -> Optional[Dict]:
        """Verifica trigger per cambio livello risposta"""
        for level, patterns in self._level_triggers.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return {
                        "type": ActivationTrigger.LEVEL_CHANGE,
                        "target_level": level,
                        "pattern_matched": pattern,
                        "original_text": text
                    }
        return None
    
    def _check_special_commands(self, text: str) -> Optional[Dict]:
        """Verifica comandi speciali"""
        for command, patterns in self._special_commands.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return {
                        "type": ActivationTrigger.COMMAND,
                        "command": command,
                        "pattern_matched": pattern,
                        "original_text": text
                    }
        return None
    
    async def execute_activation(self, activation: Dict) -> Dict:
        """
        Esegue l'azione corrispondente al trigger rilevato
        
        Args:
            activation: Risultato da process_input()
            
        Returns:
            Risultato dell'esecuzione
        """
        if not activation:
            return {"success": False, "error": "Nessuna attivazione"}
        
        trigger_type = activation.get("type")
        
        # Cambio modalità
        if trigger_type == ActivationTrigger.MODE_CHANGE:
            if not self.mode_manager:
                return {"success": False, "error": "ModeManager non configurato"}
            
            target = activation.get("target_mode")
            
            if target == "pilot":
                result = await self.mode_manager.go_pilot("Attivazione vocale")
            elif target == "executive":
                result = await self.mode_manager.go_executive("Attivazione vocale")
            elif target == "copilot":
                result = await self.mode_manager.go_copilot("Attivazione vocale")
            elif target == "passive":
                result = await self.mode_manager.go_passive("Attivazione vocale")
            else:
                return {"success": False, "error": f"Modalità sconosciuta: {target}"}
            
            return {
                "success": True,
                "action": "mode_change",
                "new_mode": target,
                "response": self._get_mode_response(target),
                **result
            }
        
        # Cambio livello risposta
        elif trigger_type == ActivationTrigger.LEVEL_CHANGE:
            if not self.mode_manager:
                return {"success": False, "error": "ModeManager non configurato"}
            
            target = activation.get("target_level")
            
            if target == "toggle":
                # Toggle tra NORMAL e ADVANCED
                if self.mode_manager.is_normal():
                    result = self.mode_manager.go_advanced("Attivazione vocale - toggle")
                    new_level = "advanced"
                else:
                    result = self.mode_manager.go_normal("Attivazione vocale - toggle")
                    new_level = "normal"
            elif target == "normal":
                result = self.mode_manager.go_normal("Attivazione vocale")
                new_level = "normal"
            elif target == "advanced":
                result = self.mode_manager.go_advanced("Attivazione vocale")
                new_level = "advanced"
            else:
                return {"success": False, "error": f"Livello sconosciuto: {target}"}
            
            return {
                "success": True,
                "action": "level_change",
                "new_level": new_level,
                "response": self._get_level_response(new_level),
                **result
            }
        
        # Comandi speciali
        elif trigger_type == ActivationTrigger.COMMAND:
            command = activation.get("command")
            
            if command == "emergency_stop":
                return {
                    "success": True,
                    "action": "emergency_stop",
                    "response": "⚠️ Attivazione emergenza! Fermo tutto.",
                    "requires_emergency_system": True
                }
            elif command == "status":
                status = self._get_system_status()
                return {
                    "success": True,
                    "action": "status",
                    "response": status,
                    "status": self.mode_manager.get_full_status() if self.mode_manager else {}
                }
        
        return {"success": False, "error": "Tipo attivazione non gestito"}
    
    def _get_mode_response(self, mode: str) -> str:
        """Risposta vocale per cambio modalità"""
        responses = {
            "pilot": "🚀 Modalità Pilot attivata. Ho il controllo totale, hands-free.",
            "executive": "👔 Modalità Executive attivata. Jarvis in supervisione globale.",
            "copilot": "🤝 Modalità Copilot attivata. Ti chiederò conferma per le azioni.",
            "passive": "🔇 Modalità Passiva attivata. Sono in ascolto ma non eseguirò azioni."
        }
        return responses.get(mode, f"Modalità {mode} attivata.")
    
    def _get_level_response(self, level: str) -> str:
        """Risposta vocale per cambio livello"""
        responses = {
            "normal": "💬 Risponderò in modo semplice e diretto.",
            "advanced": "🔬 Risponderò con analisi dettagliate e tecniche."
        }
        return responses.get(level, f"Livello {level} attivato.")
    
    def _get_system_status(self) -> str:
        """Genera risposta stato sistema"""
        if not self.mode_manager:
            return "Sistema attivo."
        
        mode = self.mode_manager.mode_name
        level = self.mode_manager.response_level_name
        
        return f"Sono in modalità {mode} con risposte {level}."
    
    # === Gestione trigger personalizzati ===
    
    def add_mode_trigger(self, mode: str, pattern: str):
        """Aggiunge trigger per modalità"""
        if mode not in self._mode_triggers:
            self._mode_triggers[mode] = []
        self._mode_triggers[mode].append(pattern)
    
    def add_level_trigger(self, level: str, pattern: str):
        """Aggiunge trigger per livello"""
        if level not in self._level_triggers:
            self._level_triggers[level] = []
        self._level_triggers[level].append(pattern)
    
    def add_wake_word(self, pattern: str):
        """Aggiunge wake word"""
        self._wake_words.append(pattern)
    
    def add_special_command(self, command: str, patterns: List[str]):
        """Aggiunge comando speciale"""
        self._special_commands[command] = patterns
    
    def on_activation(self, callback: Callable):
        """Registra callback per attivazioni"""
        self._activation_callbacks.append(callback)
    
    def get_all_triggers(self) -> Dict:
        """Restituisce tutti i trigger configurati"""
        return {
            "mode_triggers": self._mode_triggers,
            "level_triggers": self._level_triggers,
            "wake_words": self._wake_words,
            "special_commands": self._special_commands
        }
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Storico attivazioni"""
        return self._activation_history[-limit:]
    
    def get_status(self) -> Dict:
        """Stato del manager"""
        return {
            "last_activation": self._last_activation.isoformat() if self._last_activation else None,
            "total_activations": len(self._activation_history),
            "mode_triggers_count": sum(len(v) for v in self._mode_triggers.values()),
            "level_triggers_count": sum(len(v) for v in self._level_triggers.values()),
            "wake_words_count": len(self._wake_words)
        }
