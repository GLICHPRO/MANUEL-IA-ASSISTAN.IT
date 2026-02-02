"""
🤖 JARVIS - Executive AI Engine

Il cuore esecutivo di Jarvis:
- Comprensione intent umano avanzata
- Decisioni intelligenti basate su contesto
- Orchestrazione di tutti i moduli del sistema
- Esecuzione azioni con feedback loop

Ruolo: EXECUTIVE AI - il cervello decisionale e operativo
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import re
import json


class ExecutionPriority(Enum):
    """Priorità di esecuzione"""
    CRITICAL = 1    # Emergenze, sicurezza
    HIGH = 2        # Comandi diretti utente
    NORMAL = 3      # Operazioni standard
    LOW = 4         # Background tasks
    DEFERRED = 5    # Eseguibili quando idle


class ExecutionStatus(Enum):
    """Stati di esecuzione"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ExecutionTask:
    """Rappresenta un task in esecuzione"""
    id: str
    action: Dict
    priority: ExecutionPriority
    status: ExecutionStatus
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3


@dataclass
class ConversationTurn:
    """Un turno di conversazione"""
    user_input: str
    intent: Dict
    response: str
    action_taken: Optional[Dict]
    timestamp: datetime = field(default_factory=datetime.now)


class ExecutiveAI:
    """
    🧠 JARVIS Executive AI
    
    Responsabilità:
    1. COMPRENSIONE: Analizza e comprende l'intento umano
    2. DECISIONE: Valuta opzioni e prende decisioni
    3. ORCHESTRAZIONE: Coordina Gideon, Automation e altri moduli
    4. ESECUZIONE: Esegue azioni e monitora risultati
    5. APPRENDIMENTO: Impara dalle interazioni
    """
    
    def __init__(self):
        # Moduli collegati
        self.gideon = None          # Modulo cognitivo/predittivo
        self.automation = None       # Layer esecutivo
        self.mode_manager = None     # Gestione modalità
        self.emergency = None        # Sistema emergenza
        self.memory = None           # Sistema memoria
        
        # Componenti interni
        from .intent_interpreter import IntentInterpreter
        from .decision_maker import DecisionMaker
        
        self.interpreter = IntentInterpreter()
        self.decision_maker = DecisionMaker()
        
        # Stato
        self.is_active = True
        self.current_task: Optional[ExecutionTask] = None
        self.task_queue: List[ExecutionTask] = []
        self.execution_history: List[ExecutionTask] = []
        
        # Conversazione
        self.conversation_history: List[ConversationTurn] = []
        self.context = {
            "user_name": None,
            "last_topic": None,
            "pending_clarification": None,
            "preferences": {},
            "session_start": datetime.now()
        }
        
        # Plugins e handlers custom
        self.intent_handlers: Dict[str, Callable] = {}
        self.pre_execution_hooks: List[Callable] = []
        self.post_execution_hooks: List[Callable] = []
        
        # Statistiche
        self.stats = {
            "total_commands": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_response_time": 0.0,
            "most_used_intents": {}
        }
        
        self._register_default_handlers()
    
    # ========== MODULI LINKING ==========
    
    def link_gideon(self, gideon_core):
        """Collega modulo cognitivo Gideon"""
        self.gideon = gideon_core
    
    def link_automation(self, automation_layer):
        """Collega Automation Layer"""
        self.automation = automation_layer
        self.decision_maker.set_automation(automation_layer)
    
    def link_mode_manager(self, mode_manager):
        """Collega Mode Manager"""
        self.mode_manager = mode_manager
        self.decision_maker.mode_manager = mode_manager
    
    def link_emergency(self, emergency_system):
        """Collega Emergency System"""
        self.emergency = emergency_system
    
    def link_memory(self, memory_manager):
        """Collega Memory Manager"""
        self.memory = memory_manager
    
    # ========== MAIN PROCESSING PIPELINE ==========
    
    async def understand(self, text: str, audio_features: Dict = None) -> Dict:
        """
        FASE 1: Comprendi l'intento umano
        
        Analizza:
        - Testo (parole, sintassi)
        - Contesto conversazione
        - Storico interazioni
        - Features audio (se disponibili)
        
        Returns:
            Intent arricchito con context
        """
        start_time = datetime.now()
        
        # Pre-processing
        text = self._preprocess_input(text)
        
        # Interpreta intent base
        intent = self.interpreter.interpret(text, self.context)
        
        # Arricchisci con contesto conversazione
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            if self._is_follow_up(text, last_turn):
                intent = self._merge_with_previous(intent, last_turn)
        
        # Risolvi riferimenti anaforici (lui, lei, quello, etc.)
        intent = self._resolve_anaphora(intent, text)
        
        # Chiedi analisi a Gideon se intent ambiguo
        if intent.confidence < 0.7 and self.gideon:
            gideon_insight = await self.gideon.analyze_intent(text, intent.to_dict())
            intent = self._enhance_with_gideon(intent, gideon_insight)
        
        return {
            "intent": intent.to_dict(),
            "original_text": text,
            "context_used": self.context.copy(),
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
        }
    
    async def decide(self, understanding: Dict) -> Dict:
        """
        FASE 2: Prendi una decisione
        
        Valuta:
        - Azioni possibili
        - Rischi e benefici
        - Modalità operativa corrente
        - Preferenze utente
        
        Returns:
            Decision con azione scelta e alternatives
        """
        intent = understanding["intent"]
        
        # Controlla handler custom
        if intent["name"] in self.intent_handlers:
            custom_action = await self.intent_handlers[intent["name"]](intent, self.context)
            if custom_action:
                return {
                    "outcome": "execute",
                    "action": custom_action,
                    "confidence": 0.95,
                    "reasoning": "Handler custom",
                    "source": "custom_handler"
                }
        
        # Verifica modalità operativa
        if self.mode_manager:
            mode_decision = self._check_mode_constraints(intent)
            if mode_decision:
                return mode_decision
        
        # Decisione standard
        decision = await self.decision_maker.decide(intent, {
            "conversation_context": self.context,
            "history": [t.intent for t in self.conversation_history[-5:]],
            "gideon_available": self.gideon is not None,
            "automation_available": self.automation is not None
        })
        
        # Se decisione richiede analisi Gideon
        from .decision_maker import DecisionOutcome
        if decision.outcome == DecisionOutcome.DELEGATE and self.gideon:
            gideon_recommendation = await self.gideon.get_recommendation(
                understanding["original_text"], 
                {"intent": intent}
            )
            decision = self._incorporate_gideon_recommendation(decision, gideon_recommendation)
        
        return decision.to_dict()
    
    async def orchestrate(self, decision: Dict) -> Dict:
        """
        FASE 3: Orchestrazione moduli
        
        Coordina:
        - Quale modulo deve agire
        - In che ordine
        - Con quali parametri
        
        Returns:
            Execution plan
        """
        action = decision.get("chosen_action", decision.get("action", {}))
        
        # Determina il modulo esecutore
        executor = self._determine_executor(action)
        
        # Prepara execution plan
        plan = {
            "action": action,
            "executor": executor,
            "pre_conditions": [],
            "post_conditions": [],
            "fallback": None,
            "timeout": 30.0
        }
        
        # Aggiungi pre-condizioni se necessario
        if action.get("type") in ["delete_file", "modify_system"]:
            plan["pre_conditions"].append({
                "type": "backup",
                "target": action.get("target")
            })
        
        # Chiedi a Gideon analisi predittiva
        if self.gideon and action.get("type") in ["automation", "workflow"]:
            prediction = await self.gideon.predict_outcome(action)
            if prediction.get("risk_level", 0) > 0.7:
                plan["requires_confirmation"] = True
                plan["risk_warning"] = prediction.get("warning")
        
        # Imposta fallback
        plan["fallback"] = self._get_fallback_action(action)
        
        return plan
    
    async def execute(self, plan: Dict) -> Dict:
        """
        FASE 4: Esecuzione azione
        
        Esegue:
        - Pre-condizioni
        - Azione principale
        - Post-condizioni
        - Gestisce errori e fallback
        
        Returns:
            Execution result
        """
        start_time = datetime.now()
        
        # Crea task
        task = ExecutionTask(
            id=f"task_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            action=plan["action"],
            priority=ExecutionPriority.NORMAL,
            status=ExecutionStatus.PENDING
        )
        
        self.current_task = task
        
        try:
            # Run pre-execution hooks
            for hook in self.pre_execution_hooks:
                await hook(task, plan)
            
            # Esegui pre-condizioni
            for pre in plan.get("pre_conditions", []):
                await self._execute_condition(pre)
            
            task.status = ExecutionStatus.RUNNING
            task.started_at = datetime.now()
            
            # Esecuzione principale
            executor = plan.get("executor", "automation")
            
            if executor == "automation" and self.automation:
                result = await self.automation.execute(plan["action"])
            elif executor == "gideon" and self.gideon:
                result = await self.gideon.execute_analysis(plan["action"])
            else:
                result = await self._execute_internal(plan["action"])
            
            # Esegui post-condizioni
            for post in plan.get("post_conditions", []):
                await self._execute_condition(post)
            
            task.status = ExecutionStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Run post-execution hooks
            for hook in self.post_execution_hooks:
                await hook(task, result)
            
            # Aggiorna statistiche
            self.stats["successful_executions"] += 1
            
            return {
                "success": True,
                "task_id": task.id,
                "result": result,
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            task.status = ExecutionStatus.FAILED
            task.error = str(e)
            
            # Tenta fallback
            if plan.get("fallback") and task.retries < task.max_retries:
                task.retries += 1
                return await self.execute({"action": plan["fallback"], "executor": plan["executor"]})
            
            self.stats["failed_executions"] += 1
            
            return {
                "success": False,
                "task_id": task.id,
                "error": str(e),
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
        
        finally:
            self.current_task = None
            self.execution_history.append(task)
            if len(self.execution_history) > 100:
                self.execution_history.pop(0)
    
    async def process(self, text: str, audio_features: Dict = None) -> Dict:
        """
        🚀 PIPELINE COMPLETA: Input → Output
        
        Orchestration flow:
        1. understand() - Comprendi l'intento
        2. decide() - Decidi cosa fare
        3. orchestrate() - Pianifica l'esecuzione
        4. execute() - Esegui l'azione
        5. respond() - Genera risposta
        
        Returns:
            Complete result with response
        """
        pipeline_start = datetime.now()
        self.stats["total_commands"] += 1
        
        result = {
            "input": text,
            "phases": {},
            "response": None,
            "executed": False
        }
        
        try:
            # Check emergency
            if self.emergency and self.emergency.is_killed:
                return {
                    "input": text,
                    "response": "Sistema in modalità emergenza. Tutti i comandi sono sospesi.",
                    "executed": False,
                    "emergency": True
                }
            
            # FASE 1: UNDERSTAND
            understanding = await self.understand(text, audio_features)
            result["phases"]["understand"] = understanding
            
            # FASE 2: DECIDE
            decision = await self.decide(understanding)
            result["phases"]["decide"] = decision
            
            # Gestisci esiti non-execute
            if decision.get("outcome") not in ["execute", "EXECUTE"]:
                result["response"] = self._generate_non_execute_response(decision)
                return result
            
            # FASE 3: ORCHESTRATE
            plan = await self.orchestrate(decision)
            result["phases"]["orchestrate"] = plan
            
            # Richiede conferma?
            if plan.get("requires_confirmation"):
                result["response"] = f"⚠️ {plan.get('risk_warning', 'Azione rischiosa')}. Confermi?"
                result["pending_confirmation"] = True
                result["pending_action"] = plan
                return result
            
            # FASE 4: EXECUTE
            execution = await self.execute(plan)
            result["phases"]["execute"] = execution
            result["executed"] = execution.get("success", False)
            
            # FASE 5: RESPOND
            result["response"] = self._generate_response(understanding, execution)
            
            # Salva turno conversazione
            self._save_conversation_turn(text, understanding["intent"], result["response"], 
                                        plan["action"] if result["executed"] else None)
            
            # Update stats
            exec_time = (datetime.now() - pipeline_start).total_seconds() * 1000
            self._update_response_time(exec_time)
            
            result["total_time_ms"] = exec_time
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            result["response"] = f"Mi dispiace, si è verificato un errore: {str(e)}"
            return result
    
    # ========== QUICK COMMANDS ==========
    
    async def quick(self, text: str) -> Dict:
        """
        Comando rapido per azioni semplici
        Salta fasi di analisi approfondita
        """
        intent = self.interpreter.interpret(text)
        
        if intent.confidence >= 0.85:
            action = self._quick_action_map(intent)
            if action:
                if self.automation:
                    result = await self.automation.execute(action)
                    return {
                        "success": True,
                        "result": result,
                        "response": self._quick_response(intent, result)
                    }
        
        # Fallback a pipeline completa
        return await self.process(text)
    
    def _quick_action_map(self, intent) -> Optional[Dict]:
        """Mappa intent ad alta confidenza ad azione rapida"""
        quick_map = {
            "time": {"type": "get_time"},
            "date": {"type": "get_date"},
            "open_app": {"type": "open_application", "app": intent.entities.get("app_name")},
            "close_app": {"type": "close_application", "app": intent.entities.get("app_name")},
            "search_web": {"type": "web_search", "query": intent.entities.get("query")},
            "volume_up": {"type": "system", "action": "volume_up"},
            "volume_down": {"type": "system", "action": "volume_down"},
            "mute": {"type": "system", "action": "mute"},
            "screenshot": {"type": "system", "action": "screenshot"},
        }
        return quick_map.get(intent.name)
    
    # ========== HANDLERS REGISTRATION ==========
    
    def register_intent_handler(self, intent_name: str, handler: Callable):
        """Registra handler custom per un intent"""
        self.intent_handlers[intent_name] = handler
    
    def register_pre_hook(self, hook: Callable):
        """Registra hook pre-esecuzione"""
        self.pre_execution_hooks.append(hook)
    
    def register_post_hook(self, hook: Callable):
        """Registra hook post-esecuzione"""
        self.post_execution_hooks.append(hook)
    
    def _register_default_handlers(self):
        """Registra handlers di default"""
        
        async def greeting_handler(intent, context):
            hour = datetime.now().hour
            if hour < 12:
                greeting = "Buongiorno"
            elif hour < 18:
                greeting = "Buon pomeriggio"
            else:
                greeting = "Buonasera"
            
            name = context.get("user_name", "")
            return {
                "type": "respond",
                "message": f"{greeting}{', ' + name if name else ''}! Come posso aiutarti?"
            }
        
        async def thanks_handler(intent, context):
            return {
                "type": "respond",
                "message": "Di nulla! Sono qui per aiutarti."
            }
        
        async def help_handler(intent, context):
            return {
                "type": "respond",
                "message": """Ecco cosa posso fare:
🖥️ Controllo sistema: apri/chiudi app, volume, screenshot
🌐 Web: cerca su Google, apri siti
📁 File: apri, crea, cerca file
⏰ Info: ora, data, meteo
🤖 Automazioni: esegui workflow e macro
Dimmi pure cosa ti serve!"""
            }
        
        self.intent_handlers["greeting"] = greeting_handler
        self.intent_handlers["thanks"] = thanks_handler
        self.intent_handlers["help"] = help_handler
    
    # ========== HELPER METHODS ==========
    
    def _preprocess_input(self, text: str) -> str:
        """Pre-processa input utente"""
        text = text.strip()
        # Rimuovi punteggiatura multipla
        text = re.sub(r'([.!?])\1+', r'\1', text)
        return text
    
    def _is_follow_up(self, text: str, last_turn: ConversationTurn) -> bool:
        """Verifica se è un follow-up della conversazione precedente"""
        follow_up_indicators = [
            r'^(e|anche|poi|inoltre|ancora)\b',
            r'^(sì|si|ok|va bene|certo)\b',
            r'^(no|non|niente)\b',
            r'^(quello|quella|quelli|quelle)\b',
            r'^(lui|lei|esso|essa)\b'
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in follow_up_indicators)
    
    def _merge_with_previous(self, intent, last_turn: ConversationTurn):
        """Unisce intent corrente con contesto precedente"""
        if last_turn.action_taken:
            intent.entities["previous_action"] = last_turn.action_taken
            intent.entities["previous_topic"] = last_turn.intent.get("name")
        return intent
    
    def _resolve_anaphora(self, intent, text: str):
        """Risolve riferimenti anaforici"""
        pronouns = {
            r'\b(quello|quella)\b': "last_mentioned_object",
            r'\b(lui|lei)\b': "last_mentioned_person",
            r'\b(lì|là)\b': "last_mentioned_location"
        }
        
        for pattern, ref_type in pronouns.items():
            if re.search(pattern, text.lower()):
                if ref_type in self.context:
                    intent.entities[ref_type] = self.context[ref_type]
        
        return intent
    
    def _enhance_with_gideon(self, intent, gideon_insight: Dict):
        """Arricchisce intent con insight da Gideon"""
        if gideon_insight:
            if gideon_insight.get("suggested_intent"):
                # Gideon suggerisce un intent diverso
                if gideon_insight.get("confidence", 0) > intent.confidence:
                    intent.name = gideon_insight["suggested_intent"]
                    intent.confidence = gideon_insight["confidence"]
            
            if gideon_insight.get("additional_entities"):
                intent.entities.update(gideon_insight["additional_entities"])
        
        return intent
    
    def _check_mode_constraints(self, intent: Dict) -> Optional[Dict]:
        """Verifica vincoli della modalità operativa"""
        if not self.mode_manager:
            return None
        
        mode = self.mode_manager.current_mode
        
        # In PASSIVE, blocca esecuzioni
        if mode.value == "passive" and intent.get("category") not in ["info", "conversation"]:
            return {
                "outcome": "reject",
                "reasoning": "Sistema in modalità passiva. Solo query informative consentite.",
                "confidence": 1.0
            }
        
        # In COPILOT, richiedi sempre conferma per azioni
        if mode.value == "copilot" and intent.get("category") not in ["info", "conversation"]:
            return {
                "outcome": "confirm",
                "reasoning": "Modalità copilot: richiesta conferma per eseguire.",
                "confidence": 0.8,
                "chosen_action": self._intent_to_action(intent)
            }
        
        return None
    
    def _incorporate_gideon_recommendation(self, decision, recommendation: Dict):
        """Incorpora raccomandazione Gideon nella decisione"""
        if recommendation.get("recommendation"):
            from .decision_maker import Alternative, DecisionOutcome
            
            alt = Alternative(
                action=recommendation["recommendation"],
                score=recommendation.get("confidence", 0.7),
                reasoning=f"Raccomandato da Gideon: {recommendation.get('reasoning', '')}"
            )
            decision.alternatives.insert(0, alt)
            
            if recommendation.get("confidence", 0) > decision.confidence:
                decision.chosen_action = recommendation["recommendation"]
                decision.confidence = recommendation["confidence"]
                decision.outcome = DecisionOutcome.EXECUTE
        
        return decision
    
    def _determine_executor(self, action: Dict) -> str:
        """Determina quale modulo deve eseguire l'azione"""
        action_type = action.get("type", "")
        
        if action_type in ["respond", "inform"]:
            return "internal"
        elif action_type in ["analyze", "predict", "simulate"]:
            return "gideon"
        else:
            return "automation"
    
    def _get_fallback_action(self, action: Dict) -> Optional[Dict]:
        """Ottiene azione fallback"""
        fallbacks = {
            "open_application": {"type": "notify", "message": "Impossibile aprire l'applicazione"},
            "web_search": {"type": "open_url", "url": "https://www.google.com"},
        }
        return fallbacks.get(action.get("type"))
    
    async def _execute_condition(self, condition: Dict):
        """Esegue una pre/post condizione"""
        if condition["type"] == "backup":
            # Logica backup
            pass
        elif condition["type"] == "notify":
            # Notifica
            pass
    
    async def _execute_internal(self, action: Dict) -> Dict:
        """Esegue azione internamente (senza moduli esterni)"""
        action_type = action.get("type", "")
        
        if action_type == "respond":
            return {"success": True, "message": action.get("message", "")}
        elif action_type == "get_time":
            return {"success": True, "time": datetime.now().strftime("%H:%M:%S")}
        elif action_type == "get_date":
            return {"success": True, "date": datetime.now().strftime("%d/%m/%Y")}
        else:
            return {"success": False, "error": f"Azione interna sconosciuta: {action_type}"}
    
    def _intent_to_action(self, intent: Dict) -> Dict:
        """Converte intent in azione"""
        return {
            "type": intent.get("name", "unknown"),
            "params": intent.get("entities", {})
        }
    
    def _generate_non_execute_response(self, decision: Dict) -> str:
        """Genera risposta per decisioni non-execute"""
        outcome = decision.get("outcome", "")
        
        if outcome in ["confirm", "CONFIRM"]:
            action = decision.get("chosen_action", {})
            return f"Vuoi che esegua: {action.get('type', 'questa azione')}?"
        elif outcome in ["suggest", "SUGGEST"]:
            suggestions = decision.get("alternatives", [])
            if suggestions:
                return "Ecco alcune opzioni:\n" + "\n".join(
                    f"• {s.get('reasoning', s.get('action', {}).get('type', ''))}" 
                    for s in suggestions[:3]
                )
            return "Non ho trovato un'azione appropriata. Puoi riformulare?"
        elif outcome in ["clarify", "CLARIFY"]:
            return decision.get("reasoning", "Puoi essere più specifico?")
        elif outcome in ["reject", "REJECT"]:
            return decision.get("reasoning", "Non posso eseguire questa azione.")
        else:
            return "Non ho capito. Puoi ripetere?"
    
    def _generate_response(self, understanding: Dict, execution: Dict) -> str:
        """Genera risposta per esecuzione completata"""
        if execution.get("success"):
            result = execution.get("result", {})
            
            # Risposte specifiche per tipo
            if result.get("message"):
                return result["message"]
            elif result.get("time"):
                return f"🕐 Sono le {result['time']}"
            elif result.get("date"):
                return f"📅 Oggi è {result['date']}"
            elif result.get("type") == "application_opened":
                return f"✅ Ho aperto {result.get('app_name', 'l\'applicazione')}"
            else:
                return "✅ Fatto!"
        else:
            error = execution.get("error", "errore sconosciuto")
            return f"❌ Mi dispiace, c'è stato un problema: {error}"
    
    def _quick_response(self, intent, result: Dict) -> str:
        """Genera risposta rapida"""
        if result.get("time"):
            return f"🕐 {result['time']}"
        elif result.get("date"):
            return f"📅 {result['date']}"
        else:
            return "✅ Fatto!"
    
    def _save_conversation_turn(self, text: str, intent: Dict, response: str, action: Optional[Dict]):
        """Salva turno conversazione"""
        turn = ConversationTurn(
            user_input=text,
            intent=intent,
            response=response,
            action_taken=action
        )
        self.conversation_history.append(turn)
        
        # Mantieni solo ultimi 50 turni
        if len(self.conversation_history) > 50:
            self.conversation_history.pop(0)
        
        # Aggiorna contesto
        self.context["last_topic"] = intent.get("name")
        
        # Traccia intent più usati
        intent_name = intent.get("name", "unknown")
        self.stats["most_used_intents"][intent_name] = \
            self.stats["most_used_intents"].get(intent_name, 0) + 1
    
    def _update_response_time(self, time_ms: float):
        """Aggiorna media tempo risposta"""
        total = self.stats["successful_executions"] + self.stats["failed_executions"]
        if total > 0:
            current_avg = self.stats["average_response_time"]
            self.stats["average_response_time"] = \
                (current_avg * (total - 1) + time_ms) / total
    
    # ========== STATUS & DIAGNOSTICS ==========
    
    def get_status(self) -> Dict:
        """Stato completo Executive AI"""
        return {
            "is_active": self.is_active,
            "current_task": self.current_task.id if self.current_task else None,
            "queue_size": len(self.task_queue),
            "modules": {
                "gideon": self.gideon is not None,
                "automation": self.automation is not None,
                "mode_manager": self.mode_manager is not None,
                "emergency": self.emergency is not None,
                "memory": self.memory is not None
            },
            "context": {
                "user_name": self.context.get("user_name"),
                "last_topic": self.context.get("last_topic"),
                "session_duration_min": (datetime.now() - self.context["session_start"]).seconds / 60
            },
            "stats": self.stats,
            "conversation_length": len(self.conversation_history),
            "registered_handlers": list(self.intent_handlers.keys())
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Ottiene storico conversazione"""
        return [
            {
                "user": t.user_input,
                "intent": t.intent.get("name"),
                "response": t.response,
                "action": t.action_taken,
                "timestamp": t.timestamp.isoformat()
            }
            for t in self.conversation_history[-limit:]
        ]
    
    def clear_context(self):
        """Reset contesto conversazione"""
        self.context = {
            "user_name": self.context.get("user_name"),
            "last_topic": None,
            "pending_clarification": None,
            "preferences": self.context.get("preferences", {}),
            "session_start": datetime.now()
        }
        self.conversation_history.clear()
    
    def set_user_name(self, name: str):
        """Imposta nome utente"""
        self.context["user_name"] = name
    
    def set_preference(self, key: str, value: Any):
        """Imposta preferenza utente"""
        self.context["preferences"][key] = value
