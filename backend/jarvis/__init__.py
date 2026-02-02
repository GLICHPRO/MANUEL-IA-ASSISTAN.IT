"""
🤖 JARVIS - Executive AI

Jarvis è l'EXECUTIVE AI del sistema:
- Comprende l'intento umano (NLU avanzato)
- Prende decisioni intelligenti
- Orchestrata tutti i moduli del sistema
- Esegue azioni con feedback loop

Architettura:
┌─────────────────────────────────────────────────────┐
│                    JARVIS                            │
│              (Executive AI Core)                     │
│                                                      │
│    ┌─────────────────────────────────────────┐      │
│    │          EXECUTIVE ENGINE               │      │
│    │  understand → decide → orchestrate →    │      │
│    │              execute                    │      │
│    └─────────────────────────────────────────┘      │
│                        │                            │
│         ┌──────────────┼──────────────┐            │
│         ▼              ▼              ▼            │
│    ┌─────────┐   ┌──────────┐   ┌──────────┐      │
│    │ GIDEON  │   │AUTOMATION│   │  MEMORY  │      │
│    │Cognitive│   │Executive │   │ Context  │      │
│    └─────────┘   └──────────┘   └──────────┘      │
└─────────────────────────────────────────────────────┘

Ruolo: Comprendere l'intento umano, decidere, orchestrare moduli, eseguire azioni.
"""

from .intent_interpreter import IntentInterpreter, Intent, IntentCategory, SlotType, Slot
from .decision_maker import DecisionMaker, Decision, DecisionOutcome
from .executive_ai import ExecutiveAI, ExecutionPriority, ExecutionStatus, ExecutionTask
from .priority_manager import PriorityManager, Priority, Urgency, Importance, PrioritizedTask
from .decision_engine import DecisionEngine, DecisionStrategy, RiskLevel, Alternative, GideonConsultation
from .security_validator import SecurityValidator, ThreatLevel, PermissionLevel, SecurityAction, RiskAssessment
from .execution_controller import ExecutionController, ExecutionState, RoutineType, AppControlAction, Routine
from .executive_memory import ExecutiveMemory, OutcomeType, LearningSignal, StrategyUpdate, Pattern
from .confidence_evaluator import ConfidenceEvaluator, ConfidenceLevel, ConfidenceFactor, RecommendationAction, ConfidenceAssessment
from .audit_logger import AuditLogger, AuditEventType, AuditSeverity, RollbackStatus, AuditEntry
from .personality import JarvisPersonality, ToneLevel, ResponseStyle, PersonalityProfile
from .dev_automations import DevAutomations, get_dev_automations, SystemLoadMonitor

__all__ = [
    # Intent Interpreter (NLU)
    'IntentInterpreter', 'Intent', 'IntentCategory', 'SlotType', 'Slot',
    # Decision Maker (legacy)
    'DecisionMaker', 'Decision', 'DecisionOutcome',
    # Decision Engine (advanced)
    'DecisionEngine', 'DecisionStrategy', 'RiskLevel', 'Alternative', 'GideonConsultation',
    # Security Validator
    'SecurityValidator', 'ThreatLevel', 'PermissionLevel', 'SecurityAction', 'RiskAssessment',
    # Execution Controller
    'ExecutionController', 'ExecutionState', 'RoutineType', 'AppControlAction', 'Routine',
    # Executive Memory
    'ExecutiveMemory', 'OutcomeType', 'LearningSignal', 'StrategyUpdate', 'Pattern',
    # Confidence Evaluator
    'ConfidenceEvaluator', 'ConfidenceLevel', 'ConfidenceFactor', 'RecommendationAction', 'ConfidenceAssessment',
    # Audit Logger
    'AuditLogger', 'AuditEventType', 'AuditSeverity', 'RollbackStatus', 'AuditEntry',
    # Personality
    'JarvisPersonality', 'ToneLevel', 'ResponseStyle', 'PersonalityProfile',
    # Executive AI
    'ExecutiveAI', 'ExecutionPriority', 'ExecutionStatus', 'ExecutionTask',
    # Priority Manager
    'PriorityManager', 'Priority', 'Urgency', 'Importance', 'PrioritizedTask',
    # Dev Automations
    'DevAutomations', 'get_dev_automations', 'SystemLoadMonitor',
    # Supervisor
    'JarvisSupervisor', 'JarvisCore'
]


class JarvisSupervisor(ExecutiveAI):
    """
    🤖 JARVIS - Executive AI (Supervisor)
    
    Estende ExecutiveAI con funzionalità di supervisione:
    - Comprensione intent umano avanzata
    - Decisioni intelligenti con contesto
    - Orchestrazione di Gideon + Automation Layer
    - Esecuzione con feedback loop
    
    Pipeline completa:
    1. understand() - Comprendi l'intento
    2. decide() - Decidi cosa fare  
    3. orchestrate() - Pianifica esecuzione
    4. execute() - Esegui con monitoraggio
    """
    
    def __init__(self, gideon_core=None, automation_layer=None, mode_manager=None):
        super().__init__()
        
        # Link moduli
        if gideon_core:
            self.link_gideon(gideon_core)
        if automation_layer:
            self.link_automation(automation_layer)
        if mode_manager:
            self.link_mode_manager(mode_manager)
        
        # Legacy attributes per compatibilità
        self.pending_decisions = []
    
    def set_gideon(self, gideon_core):
        """Collega il modulo Gideon (legacy wrapper)"""
        self.link_gideon(gideon_core)
    
    def set_automation(self, automation_layer):
        """Collega l'Automation Layer (legacy wrapper)"""
        self.link_automation(automation_layer)
    
    # ========== LEGACY METHODS (compatibilità) ==========
    
    async def process_legacy(self, text: str, context: dict = None) -> dict:
        """
        Pipeline principale legacy: Input → Analisi → Decisione → Esecuzione
        
        Usa il nuovo process() da ExecutiveAI che è più completo.
        
        Args:
            text: Input utente (testo o trascrizione)
            context: Contesto aggiuntivo
            
        Returns:
            Risultato completo del processo
        """
        # Usa la pipeline avanzata di ExecutiveAI
        result = await super().process(text)
        
        # Adatta output per compatibilità
        if "phases" not in result:
            result["phases"] = {}
        if "output" not in result:
            result["output"] = result.get("response")
            
        return result
    
    # Il nuovo process() è ereditato da ExecutiveAI
    # con pipeline: understand → decide → orchestrate → execute
    
    async def quick_command(self, text: str) -> dict:
        """
        Comando rapido per azioni semplici e chiare
        Usa quick() da ExecutiveAI
        """
        return await self.quick(text)
    
    def _map_intent_to_action(self, intent: Intent) -> dict:
        """Mappa intent ad alta confidenza direttamente ad azione"""
        quick_map = {
            "open_app": {"type": "open_application", "app": intent.entities.get("app")},
            "close_app": {"type": "close_application", "app": intent.entities.get("app")},
            "search_web": {"type": "web_search", "query": intent.entities.get("query")},
            "time": {"type": "get_time"},
            "date": {"type": "get_date"},
            "volume_up": {"type": "volume", "action": "up"},
            "volume_down": {"type": "volume", "action": "down"},
            "mute": {"type": "volume", "action": "mute"},
        }
        return quick_map.get(intent.name)
    
    # ========== CONFIRMATIONS ==========
    
    async def confirm(self, decision_id: str) -> dict:
        """Conferma una decisione pending"""
        for pending in self.pending_decisions:
            if pending["id"] == decision_id:
                self.pending_decisions.remove(pending)
                decision = pending["decision"]
                
                if self.automation:
                    return await self.automation.execute(decision.chosen_action)
                return {"error": "Automation Layer non disponibile"}
        
        return {"error": "Decisione non trovata"}
    
    def reject(self, decision_id: str) -> dict:
        """Rifiuta una decisione pending"""
        for pending in self.pending_decisions:
            if pending["id"] == decision_id:
                self.pending_decisions.remove(pending)
                return {"rejected": True}
        
        return {"error": "Decisione non trovata"}
    
    def get_pending(self) -> list:
        """Lista decisioni in attesa"""
        return [{
            "id": p["id"],
            "action": p["decision"].chosen_action,
            "reasoning": p["decision"].reasoning
        } for p in self.pending_decisions]
    
    # ========== DELEGATION ==========
    
    async def ask_gideon(self, query: str, context: dict = None) -> dict:
        """Chiede analisi/previsione a Gideon"""
        if not self.gideon:
            return {"error": "Gideon non disponibile"}
        return await self.gideon.get_recommendation(query, context)
    
    async def tell_automation(self, action: dict) -> dict:
        """Ordina esecuzione ad Automation Layer"""
        if not self.automation:
            return {"error": "Automation Layer non disponibile"}
        return await self.automation.execute(action)
    
    # ========== STATUS ==========
    
    def get_supervisor_status(self) -> dict:
        """Stato completo del Supervisor (usa get_status() da ExecutiveAI)"""
        base_status = super().get_status()
        
        # Aggiungi info specifiche supervisor
        base_status["pending_decisions_legacy"] = len(self.pending_decisions)
        base_status["mode"] = self.mode_manager.mode_name if self.mode_manager else "unknown"
        
        return base_status


# Alias per retrocompatibilità
JarvisCore = JarvisSupervisor
