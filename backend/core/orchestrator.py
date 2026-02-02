"""
🎭 ORCHESTRATOR - Coordinatore Gideon 3.0 + Jarvis Core
Garantisce il flusso: Input → Analisi → Decisione → Esecuzione → Risposta

Pipeline Cognitiva:
1. Jarvis interpreta intent
2. Gideon analizza, prevede, simula
3. Jarvis valuta alternative e decide
4. Jarvis esegue (se approvato)
5. Risposta all'utente
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class ProcessingPhase(Enum):
    """Fasi del processing"""
    RECEIVED = "received"
    INTERPRETING = "interpreting"    # Nuovo: Jarvis interpreta intent
    ANALYZING = "analyzing"
    PREDICTING = "predicting"
    SIMULATING = "simulating"
    RANKING = "ranking"
    DECIDING = "deciding"
    CONFIRMING = "confirming"        # Nuovo: attesa conferma
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


class ProcessingContext:
    """Contesto di una richiesta in elaborazione"""
    
    def __init__(self, request_id: str, user_input: str):
        self.request_id = request_id
        self.user_input = user_input
        self.phase = ProcessingPhase.RECEIVED
        self.started_at = datetime.now()
        self.completed_at = None
        
        # Risultati delle varie fasi
        self.intent = None           # Nuovo: intent interpretato
        self.analysis = None
        self.predictions = None
        self.simulations = None
        self.ranking = None
        self.recommendation = None
        self.decision = None
        self.execution_result = None
        self.final_response = None
        
        # Metadata
        self.error = None
        self.phases_timing = {}
        self.cognitive_trace = []    # Nuovo: traccia ragionamento
    
    def set_phase(self, phase: ProcessingPhase):
        """Aggiorna la fase corrente"""
        if self.phase != phase:
            self.phases_timing[self.phase.value] = datetime.now()
            self.cognitive_trace.append({
                "phase": phase.value,
                "timestamp": datetime.now().isoformat()
            })
        self.phase = phase
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "user_input": self.user_input,
            "phase": self.phase.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "has_intent": self.intent is not None,
            "has_analysis": self.analysis is not None,
            "has_recommendation": self.recommendation is not None,
            "has_decision": self.decision is not None,
            "has_execution": self.execution_result is not None,
            "error": self.error,
            "cognitive_trace": self.cognitive_trace
        }


class Orchestrator:
    """
    Orchestratore principale del sistema
    
    Architettura:
    ┌─────────────────────────────────────────────┐
    │                  JARVIS                      │
    │              (Supervisor)                    │
    │                                              │
    │  ┌─────────────┐       ┌─────────────────┐  │
    │  │   GIDEON    │       │ AUTOMATION LAYER│  │
    │  │  Cognitivo  │ ───▶  │    Esecutivo    │  │
    │  │  Predittivo │       │                 │  │
    │  └─────────────┘       └─────────────────┘  │
    └─────────────────────────────────────────────┘
    
    Pipeline:
    1. JARVIS interpreta l'intent dell'utente
    2. GIDEON analizza, prevede, simula scenari (se necessario)
    3. JARVIS prende decisione
    4. AUTOMATION LAYER esegue (se approvato)
    """
    
    def __init__(self, gideon_core=None, jarvis_core=None, 
                 automation_layer=None, mode_manager=None, 
                 emergency_system=None, action_logger=None):
        self.gideon = gideon_core
        self.jarvis = jarvis_core
        self.automation = automation_layer
        self.mode_manager = mode_manager
        self.emergency = emergency_system
        self.logger = action_logger
        
        # Collega i moduli a Jarvis se disponibili
        if self.jarvis:
            if self.gideon:
                self.jarvis.set_gideon(self.gideon)
            if self.automation:
                self.jarvis.set_automation(self.automation)
        
        self._request_counter = 0
        self._active_contexts: Dict[str, ProcessingContext] = {}
        self._history: List[ProcessingContext] = []
        self._max_history = 100
        
        # Callbacks per aggiornamenti di stato
        self._phase_callbacks: List = []
        
        # Configurazione pipeline
        self._use_cognitive_pipeline = True
        self._quick_commands_enabled = True
    
    def _generate_request_id(self) -> str:
        """Genera ID univoco per richiesta"""
        self._request_counter += 1
        return f"req_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._request_counter}"
    
    async def process(self, user_input: str, context: dict = None) -> dict:
        """
        Processa una richiesta con pipeline cognitiva completa
        
        Args:
            user_input: Input dell'utente (testo o trascrizione vocale)
            context: Contesto aggiuntivo
            
        Returns:
            Risposta completa solo dopo ragionamento terminato
        """
        request_id = self._generate_request_id()
        ctx = ProcessingContext(request_id, user_input)
        self._active_contexts[request_id] = ctx
        context = context or {}
        
        try:
            # Check emergenza
            if self.emergency and self.emergency.is_killed:
                ctx.error = "Sistema in stato KILL - operazioni sospese"
                ctx.phase = ProcessingPhase.ERROR
                return self._build_error_response(ctx)
            
            # ========================================
            # FASE 0: QUICK COMMAND (opzionale)
            # ========================================
            if self._quick_commands_enabled and self.jarvis:
                quick_result = await self._try_quick_command(user_input, ctx)
                if quick_result:
                    return quick_result
            
            # ========================================
            # FASE 1: JARVIS INTERPRETA INTENT
            # ========================================
            ctx.set_phase(ProcessingPhase.INTERPRETING)
            await self._notify_phase(ctx)
            
            if self.jarvis and hasattr(self.jarvis, 'interpreter'):
                ctx.intent = self.jarvis.interpreter.interpret(user_input, context)
                ctx.cognitive_trace.append({
                    "phase": "intent",
                    "result": ctx.intent.to_dict() if hasattr(ctx.intent, 'to_dict') else ctx.intent
                })
            else:
                # Fallback: intent base
                ctx.intent = {
                    "name": "unknown",
                    "confidence": 0.3,
                    "entities": {},
                    "text": user_input
                }
            
            # ========================================
            # FASE 2: GIDEON ANALIZZA (se complesso)
            # ========================================
            intent_confidence = getattr(ctx.intent, 'confidence', 0.5) if hasattr(ctx.intent, 'confidence') else ctx.intent.get('confidence', 0.5)
            needs_deep_analysis = intent_confidence < 0.7 or self._is_complex_request(user_input)
            
            if self.gideon and needs_deep_analysis:
                ctx.set_phase(ProcessingPhase.ANALYZING)
                await self._notify_phase(ctx)
                
                intent_dict = ctx.intent.to_dict() if hasattr(ctx.intent, 'to_dict') else ctx.intent
                
                # Analisi
                ctx.analysis = await self.gideon.analyze({
                    "input": user_input,
                    "intent": intent_dict,
                    "context": context
                })
                
                # Previsioni
                ctx.set_phase(ProcessingPhase.PREDICTING)
                await self._notify_phase(ctx)
                ctx.predictions = await self.gideon.predict({
                    "input": user_input,
                    "intent": intent_dict,
                    "analysis": ctx.analysis
                })
                
                # Simulazione scenari
                ctx.set_phase(ProcessingPhase.SIMULATING)
                await self._notify_phase(ctx)
                ctx.simulations = await self.gideon.simulate({
                    "input": user_input,
                    "analysis": ctx.analysis,
                    "predictions": ctx.predictions
                })
                
                # Ranking opzioni
                ctx.set_phase(ProcessingPhase.RANKING)
                await self._notify_phase(ctx)
                options = ctx.simulations.get("scenarios", [])
                ctx.ranking = await self.gideon.rank_options(options)
                
                # Genera raccomandazione
                ctx.recommendation = await self.gideon.get_recommendation({
                    "input": user_input,
                    "intent": intent_dict,
                    "analysis": ctx.analysis,
                    "predictions": ctx.predictions,
                    "simulations": ctx.simulations,
                    "ranking": ctx.ranking
                })
                
                ctx.cognitive_trace.append({
                    "phase": "gideon_analysis",
                    "recommendation": ctx.recommendation
                })
            
            # ========================================
            # FASE 3: JARVIS DECIDE
            # ========================================
            ctx.set_phase(ProcessingPhase.DECIDING)
            await self._notify_phase(ctx)
            
            if self.jarvis and hasattr(self.jarvis, 'decision_maker'):
                intent_dict = ctx.intent.to_dict() if hasattr(ctx.intent, 'to_dict') else ctx.intent
                
                # Arricchisci contesto con analisi Gideon
                decision_context = {
                    **context,
                    "gideon_recommendation": ctx.recommendation,
                    "gideon_analysis": ctx.analysis,
                    "gideon_predictions": ctx.predictions
                }
                
                ctx.decision = await self.jarvis.decision_maker.decide(
                    intent_dict, 
                    decision_context
                )
                
                ctx.cognitive_trace.append({
                    "phase": "decision",
                    "outcome": ctx.decision.outcome.value if hasattr(ctx.decision, 'outcome') else str(ctx.decision),
                    "confidence": ctx.decision.confidence if hasattr(ctx.decision, 'confidence') else 0
                })
            else:
                # Fallback: decisione basata su mode_manager
                ctx.decision = await self._legacy_decide(ctx)
            
            # ========================================
            # FASE 4: ESECUZIONE (se approvata)
            # ========================================
            should_execute = self._should_execute(ctx)
            
            if should_execute and self.jarvis:
                ctx.set_phase(ProcessingPhase.EXECUTING)
                await self._notify_phase(ctx)
                
                # Determina azione da eseguire
                action = self._get_action_to_execute(ctx)
                
                if action:
                    ctx.execution_result = await self.jarvis.executor.execute(action)
                    
                    # Log azione
                    if self.logger:
                        await self.logger.log_action(
                            action_type=action.get("action", "unknown"),
                            params=action.get("params", {}),
                            result=ctx.execution_result,
                            reasoning=ctx.decision.reasoning if hasattr(ctx.decision, 'reasoning') else None,
                            mode=self.mode_manager.mode_name if self.mode_manager else None
                        )
                    
                    ctx.cognitive_trace.append({
                        "phase": "execution",
                        "action": action,
                        "success": ctx.execution_result.get("success", False)
                    })
            
            # ========================================
            # FASE 5: RISPOSTA FINALE
            # ========================================
            ctx.set_phase(ProcessingPhase.COMPLETED)
            ctx.completed_at = datetime.now()
            
            ctx.final_response = self._build_response(ctx)
            
            # Salva in history
            self._history.append(ctx)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            
            return ctx.final_response
            
        except Exception as e:
            ctx.error = str(e)
            ctx.phase = ProcessingPhase.ERROR
            return self._build_error_response(ctx)
        
        finally:
            if request_id in self._active_contexts:
                del self._active_contexts[request_id]
    
    async def _try_quick_command(self, user_input: str, ctx: ProcessingContext) -> Optional[dict]:
        """Prova a eseguire come comando rapido"""
        quick_intents = {"time", "date", "greeting", "thanks", "help"}
        
        intent = self.jarvis.interpreter.interpret(user_input)
        
        if intent.name in quick_intents and intent.confidence >= 0.8:
            result = await self.jarvis.quick_command(user_input)
            if result:
                ctx.set_phase(ProcessingPhase.COMPLETED)
                ctx.completed_at = datetime.now()
                ctx.intent = intent
                ctx.execution_result = result
                ctx.cognitive_trace.append({"phase": "quick_command", "intent": intent.name})
                return self._build_response(ctx)
        
        return None
    
    def _is_complex_request(self, text: str) -> bool:
        """Determina se richiede analisi approfondita"""
        complex_keywords = [
            "analizza", "confronta", "valuta", "consiglia", 
            "migliore", "ottimizza", "pianifica", "prevedi",
            "simula", "scenario", "rischi", "alternative"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in complex_keywords)
    
    async def _legacy_decide(self, ctx: ProcessingContext) -> dict:
        """Decisione legacy basata solo su mode_manager"""
        should_execute = False
        
        if ctx.recommendation and self.mode_manager:
            mode_check = self.mode_manager.can_execute(
                ctx.recommendation.get("action", "respond")
            )
            
            if not mode_check["allowed"]:
                return {"execute": False, "reason": mode_check["reason"]}
            elif mode_check["requires_confirmation"]:
                return {"execute": False, "reason": "Richiede conferma", "pending": True}
            else:
                return {"execute": True, "reason": "Approvato"}
        
        return {"execute": False, "reason": "Nessuna raccomandazione"}
    
    def _should_execute(self, ctx: ProcessingContext) -> bool:
        """Determina se eseguire l'azione"""
        if not ctx.decision:
            return False
        
        # Nuovo sistema con DecisionOutcome
        if hasattr(ctx.decision, 'outcome'):
            from jarvis.decision_maker import DecisionOutcome
            return ctx.decision.outcome == DecisionOutcome.EXECUTE
        
        # Legacy
        return ctx.decision.get("execute", False)
    
    def _get_action_to_execute(self, ctx: ProcessingContext) -> Optional[dict]:
        """Ottiene l'azione da eseguire"""
        # Dal nuovo decision maker
        if hasattr(ctx.decision, 'chosen_action'):
            return ctx.decision.chosen_action
        
        # Dalla raccomandazione Gideon
        if ctx.recommendation:
            return {
                "action": ctx.recommendation.get("action", "respond"),
                "params": ctx.recommendation.get("params", {})
            }
        
        return None
    
    def _build_response(self, ctx: ProcessingContext) -> dict:
        """Costruisce la risposta finale"""
        response = {
            "success": True,
            "request_id": ctx.request_id,
            "input": ctx.user_input,
            "processing_time_ms": int((ctx.completed_at - ctx.started_at).total_seconds() * 1000),
        }
        
        # Intent info
        if ctx.intent:
            intent_dict = ctx.intent.to_dict() if hasattr(ctx.intent, 'to_dict') else ctx.intent
            response["intent"] = {
                "name": intent_dict.get("name"),
                "confidence": intent_dict.get("confidence"),
                "category": intent_dict.get("category")
            }
        
        # Decision info
        if ctx.decision:
            if hasattr(ctx.decision, 'outcome'):
                response["decision"] = {
                    "outcome": ctx.decision.outcome.value,
                    "confidence": ctx.decision.confidence,
                    "reasoning": ctx.decision.reasoning
                }
            else:
                response["decision"] = ctx.decision
        
        # Contenuto risposta
        if ctx.execution_result and ctx.execution_result.get("success"):
            response["response"] = ctx.execution_result.get("output", "Operazione completata")
            response["action_executed"] = True
        elif hasattr(ctx.decision, 'outcome'):
            from jarvis.decision_maker import DecisionOutcome
            
            if ctx.decision.outcome == DecisionOutcome.CONFIRM:
                response["requires_confirmation"] = True
                response["pending_id"] = ctx.decision.id
                response["response"] = ctx.decision.reasoning
            elif ctx.decision.outcome == DecisionOutcome.SUGGEST:
                response["suggestions"] = [a.to_dict() for a in ctx.decision.alternatives[:3]]
                response["response"] = ctx.decision.reasoning
            elif ctx.decision.outcome == DecisionOutcome.CLARIFY:
                response["needs_clarification"] = True
                response["response"] = ctx.decision.reasoning
            elif ctx.decision.outcome == DecisionOutcome.REJECT:
                response["rejected"] = True
                response["response"] = ctx.decision.reasoning
            else:
                response["response"] = ctx.decision.reasoning
        elif ctx.recommendation:
            response["response"] = ctx.recommendation.get("response", 
                                   ctx.recommendation.get("content", ""))
            response["action_executed"] = False
        else:
            response["response"] = "Non ho potuto elaborare la richiesta"
        
        # Metadata analisi Gideon (opzionale)
        if ctx.analysis:
            response["gideon_analysis"] = {
                "intent": ctx.analysis.get("intent"),
                "confidence": ctx.analysis.get("confidence"),
                "entities": ctx.analysis.get("entities", [])
            }
        
        # Cognitive trace (per debug/trasparenza)
        response["cognitive_trace"] = ctx.cognitive_trace
        
        return response
    
    def _build_error_response(self, ctx: ProcessingContext) -> dict:
        """Costruisce risposta di errore"""
        return {
            "success": False,
            "request_id": ctx.request_id,
            "input": ctx.user_input,
            "error": ctx.error,
            "phase": ctx.phase.value,
            "response": f"Si è verificato un errore: {ctx.error}"
        }
    
    async def _notify_phase(self, ctx: ProcessingContext):
        """Notifica cambio fase ai listeners"""
        for callback in self._phase_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ctx)
                else:
                    callback(ctx)
            except Exception:
                pass
    
    def on_phase_change(self, callback):
        """Registra callback per cambio fase"""
        self._phase_callbacks.append(callback)
    
    def get_active_requests(self) -> List[dict]:
        """Richieste attualmente in elaborazione"""
        return [ctx.to_dict() for ctx in self._active_contexts.values()]
    
    def get_history(self, limit: int = 20) -> List[dict]:
        """Storico richieste"""
        return [ctx.to_dict() for ctx in self._history[-limit:]]
    
    def get_status(self) -> dict:
        """Stato dell'orchestratore"""
        return {
            "active_requests": len(self._active_contexts),
            "total_processed": self._request_counter,
            "history_size": len(self._history),
            "gideon_active": self.gideon is not None,
            "jarvis_active": self.jarvis is not None,
            "mode_manager_active": self.mode_manager is not None,
            "emergency_active": self.emergency is not None,
            "cognitive_pipeline": self._use_cognitive_pipeline,
            "quick_commands": self._quick_commands_enabled
        }
    
    # ========== METODI DI CONTROLLO PIPELINE ==========
    
    def enable_cognitive_pipeline(self, enabled: bool = True):
        """Abilita/disabilita pipeline cognitiva completa"""
        self._use_cognitive_pipeline = enabled
    
    def enable_quick_commands(self, enabled: bool = True):
        """Abilita/disabilita comandi rapidi"""
        self._quick_commands_enabled = enabled
    
    async def process_with_trace(self, user_input: str, context: dict = None) -> dict:
        """
        Processa con trace dettagliato del ragionamento
        Utile per debug e trasparenza
        """
        result = await self.process(user_input, context)
        
        # Aggiungi trace esteso
        if self._history:
            last_ctx = self._history[-1]
            result["detailed_trace"] = {
                "phases": last_ctx.phases_timing,
                "intent_raw": last_ctx.intent.to_dict() if hasattr(last_ctx.intent, 'to_dict') else last_ctx.intent,
                "gideon_analysis": last_ctx.analysis,
                "gideon_predictions": last_ctx.predictions,
                "decision_full": last_ctx.decision.to_dict() if hasattr(last_ctx.decision, 'to_dict') else last_ctx.decision
            }
        
        return result
    
    async def confirm_pending(self, pending_id: str) -> dict:
        """Conferma azione pending ed esegui"""
        if self.jarvis:
            result = self.jarvis.confirm_pending(pending_id)
            if result.get("confirmed") and result.get("action"):
                exec_result = await self.jarvis.executor.execute(result["action"])
                return {
                    "success": True,
                    "confirmed": True,
                    "executed": True,
                    "result": exec_result
                }
            return result
        return {"success": False, "error": "Jarvis non attivo"}
    
    async def reject_pending(self, pending_id: str) -> dict:
        """Rifiuta azione pending"""
        if self.jarvis:
            return self.jarvis.reject_pending(pending_id)
        return {"success": False, "error": "Jarvis non attivo"}
