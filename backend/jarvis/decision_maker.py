"""
⚖️ JARVIS CORE - Decision Maker
Valuta alternative e prende decisioni finali
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import asyncio


class DecisionConfidence(Enum):
    """Livelli di confidenza decisione"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


class DecisionOutcome(Enum):
    """Possibili esiti della decisione"""
    EXECUTE = "execute"              # Esegui immediatamente
    CONFIRM = "confirm"              # Richiedi conferma
    SUGGEST = "suggest"              # Suggerisci alternative
    CLARIFY = "clarify"              # Chiedi chiarimenti
    REJECT = "reject"                # Rifiuta (troppo rischioso)
    DELEGATE = "delegate"            # Delega a Gideon per analisi


class Alternative:
    """Rappresenta un'alternativa di azione"""
    
    def __init__(self, action: Dict, score: float, reasoning: str):
        self.action = action
        self.score = score  # 0.0 - 1.0
        self.reasoning = reasoning
        self.risks: List[str] = []
        self.benefits: List[str] = []
        self.estimated_time: float = 0.0  # secondi
        self.reversible: bool = True
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "score": self.score,
            "reasoning": self.reasoning,
            "risks": self.risks,
            "benefits": self.benefits,
            "estimated_time": self.estimated_time,
            "reversible": self.reversible
        }


class Decision:
    """Rappresenta una decisione presa"""
    
    def __init__(self, outcome: DecisionOutcome, chosen_action: Dict,
                 confidence: float, reasoning: str):
        self.id = f"dec_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.outcome = outcome
        self.chosen_action = chosen_action
        self.confidence = confidence
        self.reasoning = reasoning
        self.alternatives: List[Alternative] = []
        self.timestamp = datetime.now()
        self.context: Dict = {}
        self.user_confirmed: Optional[bool] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "outcome": self.outcome.value,
            "chosen_action": self.chosen_action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "timestamp": self.timestamp.isoformat(),
            "user_confirmed": self.user_confirmed
        }


class DecisionMaker:
    """
    Valuta alternative e prende decisioni per Jarvis Supervisor
    
    Processo decisionale:
    1. Riceve intent interpretato
    2. Genera alternative possibili
    3. Valuta rischi/benefici di ogni alternativa
    4. Classifica le alternative
    5. Decide l'esito (execute, confirm, suggest, clarify, reject)
    6. Restituisce decisione con spiegazione
    """
    
    def __init__(self, mode_manager=None, security_manager=None):
        self.mode_manager = mode_manager
        self.security = security_manager
        self.automation = None  # Collegamento ad Automation Layer
        
        self._decision_history: List[Decision] = []
        self._max_history = 100
        
        # Soglie di confidenza per esiti
        self._confidence_thresholds = {
            DecisionOutcome.EXECUTE: 0.8,      # Alta confidenza -> esegui
            DecisionOutcome.CONFIRM: 0.5,      # Media -> conferma
            DecisionOutcome.SUGGEST: 0.3,      # Bassa -> suggerisci
            DecisionOutcome.CLARIFY: 0.2,      # Molto bassa -> chiarisci
            DecisionOutcome.REJECT: 0.0        # Rifiuta se rischioso
        }
        
        # Azioni rischiose che richiedono sempre conferma
        self._risky_actions = {
            "shutdown", "restart", "delete_file", "format",
            "uninstall", "kill_process", "modify_system"
        }
        
        # Azioni safe che possono essere eseguite automaticamente
        self._safe_actions = {
            "open_app", "open_url", "search_web", "notify",
            "set_volume", "play_music", "time", "date", "weather"
        }
        
        # Pesi per valutazione
        self._evaluation_weights = {
            "confidence": 0.3,
            "safety": 0.25,
            "relevance": 0.2,
            "user_preference": 0.15,
            "efficiency": 0.1
        }
    
    def set_automation(self, automation_layer):
        """Collega l'Automation Layer per verificare capabilities"""
        self.automation = automation_layer
    
    async def decide(self, intent: Dict, context: Dict = None) -> Decision:
        """
        Prende una decisione basata sull'intent
        
        Args:
            intent: Intent interpretato (da IntentInterpreter)
            context: Contesto aggiuntivo (modalità, storico, etc.)
            
        Returns:
            Decision con esito e spiegazione
        """
        context = context or {}
        
        # 1. Genera alternative
        alternatives = await self._generate_alternatives(intent, context)
        
        # 2. Valuta ogni alternativa
        for alt in alternatives:
            await self._evaluate_alternative(alt, context)
        
        # 3. Classifica alternative per score
        alternatives.sort(key=lambda x: x.score, reverse=True)
        
        # 4. Scegli la migliore
        best = alternatives[0] if alternatives else None
        
        # 5. Determina esito
        outcome, reasoning = await self._determine_outcome(best, intent, context)
        
        # 6. Crea decisione
        decision = Decision(
            outcome=outcome,
            chosen_action=best.action if best else {},
            confidence=best.score if best else 0.0,
            reasoning=reasoning
        )
        decision.alternatives = alternatives
        decision.context = context
        
        # Salva in history
        self._decision_history.append(decision)
        if len(self._decision_history) > self._max_history:
            self._decision_history = self._decision_history[-self._max_history:]
        
        return decision
    
    async def _generate_alternatives(self, intent: Dict, context: Dict) -> List[Alternative]:
        """Genera alternative possibili per l'intent"""
        alternatives = []
        
        intent_name = intent.get("name", "unknown")
        entities = intent.get("entities", {})
        original_confidence = intent.get("confidence", 0.5)
        
        # Alternativa principale (azione diretta)
        primary_action = self._intent_to_action(intent_name, entities)
        if primary_action:
            alt = Alternative(
                action=primary_action,
                score=original_confidence,
                reasoning=f"Azione diretta per intent '{intent_name}'"
            )
            alternatives.append(alt)
        
        # Alternative da intent secondari
        for alt_intent in intent.get("alternatives", []):
            alt_action = self._intent_to_action(
                alt_intent.get("name"), 
                alt_intent.get("entities", {})
            )
            if alt_action:
                alt = Alternative(
                    action=alt_action,
                    score=alt_intent.get("confidence", 0.3),
                    reasoning=f"Alternativa: {alt_intent.get('name')}"
                )
                alternatives.append(alt)
        
        # Se nessuna alternativa, suggerisci help
        if not alternatives:
            alternatives.append(Alternative(
                action={"action": "help", "params": {}},
                score=0.2,
                reasoning="Nessuna azione chiara - suggerisco aiuto"
            ))
        
        return alternatives
    
    def _intent_to_action(self, intent_name: str, entities: Dict) -> Optional[Dict]:
        """Converte intent in azione"""
        action_map = {
            "shutdown": {"action": "shutdown", "params": {}},
            "restart": {"action": "restart", "params": {}},
            "lock": {"action": "lock", "params": {}},
            "sleep": {"action": "sleep", "params": {}},
            "open_app": {"action": "open_app", "params": {"name": entities.get("app_name", "")}},
            "close_app": {"action": "close_app", "params": {"name": entities.get("app_name", "")}},
            "open_file": {"action": "open_file", "params": {"path": entities.get("filename", "")}},
            "delete_file": {"action": "delete_file", "params": {"path": entities.get("filename", "")}},
            "search_web": {"action": "search_web", "params": {"query": entities.get("query", "")}},
            "open_url": {"action": "open_url", "params": {"url": entities.get("url", "")}},
            "weather": {"action": "weather", "params": {"location": entities.get("location", "")}},
            "time": {"action": "time", "params": {}},
            "date": {"action": "date", "params": {}},
            "volume": {"action": "set_volume", "params": {"level": entities.get("level", 50)}},
            "mute": {"action": "mute", "params": {}},
            "calculate": {"action": "calculate", "params": {"expression": entities.get("expression", "")}},
            "greeting": {"action": "respond", "params": {"type": "greeting"}},
            "thanks": {"action": "respond", "params": {"type": "thanks"}},
            "help": {"action": "help", "params": {}}
        }
        return action_map.get(intent_name)
    
    async def _evaluate_alternative(self, alt: Alternative, context: Dict):
        """Valuta un'alternativa calcolando score, rischi e benefici"""
        action_type = alt.action.get("action", "")
        
        # Safety score
        if action_type in self._risky_actions:
            alt.risks.append("Azione potenzialmente rischiosa")
            alt.reversible = False
            safety_score = 0.3
        elif action_type in self._safe_actions:
            alt.benefits.append("Azione sicura")
            safety_score = 1.0
        else:
            safety_score = 0.7
        
        # Mode compatibility
        mode_score = 1.0
        if self.mode_manager:
            can_exec = self.mode_manager.can_execute(action_type)
            if not can_exec.get("allowed"):
                alt.risks.append(f"Non permesso in modalità {self.mode_manager.mode_name}")
                mode_score = 0.2
            elif can_exec.get("requires_confirmation"):
                alt.risks.append("Richiede conferma")
                mode_score = 0.7
        
        # Security check
        security_score = 1.0
        if self.security:
            sec_check = await self.security.check_permission(action_type)
            if not sec_check.get("allowed"):
                alt.risks.append("Bloccato dalla sicurezza")
                security_score = 0.1
            elif sec_check.get("requires_confirmation"):
                security_score = 0.6
        
        # Calcola score finale pesato
        weights = self._evaluation_weights
        alt.score = (
            alt.score * weights["confidence"] +
            safety_score * weights["safety"] +
            mode_score * weights["relevance"] +
            security_score * weights["efficiency"]
        )
        
        # Benefici generici
        if alt.reversible:
            alt.benefits.append("Azione reversibile")
        
        # Stima tempo
        alt.estimated_time = self._estimate_time(action_type)
    
    def _estimate_time(self, action_type: str) -> float:
        """Stima tempo esecuzione in secondi"""
        time_estimates = {
            "open_app": 2.0,
            "close_app": 1.0,
            "search_web": 3.0,
            "open_url": 2.0,
            "shutdown": 10.0,
            "restart": 30.0,
            "weather": 2.0,
            "calculate": 0.1,
            "respond": 0.5
        }
        return time_estimates.get(action_type, 1.0)
    
    async def _determine_outcome(self, best: Optional[Alternative], 
                                  intent: Dict, context: Dict) -> Tuple[DecisionOutcome, str]:
        """Determina l'esito della decisione"""
        
        if not best:
            return DecisionOutcome.CLARIFY, "Non ho capito cosa vuoi fare. Puoi riformulare?"
        
        action_type = best.action.get("action", "")
        confidence = best.score
        
        # Check modalità
        current_mode = "copilot"
        if self.mode_manager:
            current_mode = self.mode_manager.mode.value
        
        # EXECUTIVE/PILOT: più autonomia
        if current_mode in ["executive", "pilot"]:
            if action_type not in self._risky_actions:
                return DecisionOutcome.EXECUTE, f"Eseguo: {action_type}"
            elif current_mode == "executive":
                return DecisionOutcome.EXECUTE, f"Executive mode: eseguo anche {action_type}"
            else:
                return DecisionOutcome.CONFIRM, f"Azione rischiosa in Pilot: confermi {action_type}?"
        
        # COPILOT: chiede conferma per azioni non-safe
        if current_mode == "copilot":
            if action_type in self._safe_actions and confidence >= 0.6:
                return DecisionOutcome.EXECUTE, f"Azione sicura: eseguo {action_type}"
            elif confidence >= self._confidence_thresholds[DecisionOutcome.CONFIRM]:
                return DecisionOutcome.CONFIRM, f"Vuoi che esegua {action_type}?"
            else:
                return DecisionOutcome.SUGGEST, f"Ho capito {action_type}, ma non sono sicuro. Confermi?"
        
        # PASSIVE: solo suggerimenti
        if current_mode == "passive":
            return DecisionOutcome.SUGGEST, f"Suggerisco: {action_type} (modalità passiva)"
        
        # Default basato su confidenza
        if confidence >= self._confidence_thresholds[DecisionOutcome.EXECUTE]:
            return DecisionOutcome.EXECUTE, f"Alta confidenza: eseguo {action_type}"
        elif confidence >= self._confidence_thresholds[DecisionOutcome.CONFIRM]:
            return DecisionOutcome.CONFIRM, f"Confermi {action_type}?"
        elif confidence >= self._confidence_thresholds[DecisionOutcome.SUGGEST]:
            return DecisionOutcome.SUGGEST, f"Intendevi {action_type}?"
        else:
            return DecisionOutcome.CLARIFY, "Non sono sicuro di aver capito. Puoi essere più specifico?"
    
    def confirm_decision(self, decision_id: str) -> Optional[Decision]:
        """Conferma una decisione in attesa"""
        for dec in self._decision_history:
            if dec.id == decision_id:
                dec.user_confirmed = True
                dec.outcome = DecisionOutcome.EXECUTE
                return dec
        return None
    
    def reject_decision(self, decision_id: str) -> Optional[Decision]:
        """Rifiuta una decisione"""
        for dec in self._decision_history:
            if dec.id == decision_id:
                dec.user_confirmed = False
                dec.outcome = DecisionOutcome.REJECT
                return dec
        return None
    
    def get_pending_decisions(self) -> List[Decision]:
        """Ottiene decisioni in attesa di conferma"""
        return [d for d in self._decision_history 
                if d.outcome == DecisionOutcome.CONFIRM and d.user_confirmed is None]
    
    def get_history(self, limit: int = 20) -> List[dict]:
        """Storico decisioni"""
        return [d.to_dict() for d in self._decision_history[-limit:]]
    
    def get_stats(self) -> Dict:
        """Statistiche decisioni"""
        outcomes = {}
        for dec in self._decision_history:
            outcome = dec.outcome.value
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        return {
            "total_decisions": len(self._decision_history),
            "outcomes": outcomes,
            "avg_confidence": sum(d.confidence for d in self._decision_history) / max(len(self._decision_history), 1),
            "pending": len(self.get_pending_decisions())
        }
