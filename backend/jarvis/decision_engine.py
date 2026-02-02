"""
🧠 JARVIS CORE - Decision Engine (Avanzato)

Motore decisionale che:
- Valuta multiple alternative con scoring multi-criterio
- Consulta Gideon per analisi predittiva
- Applica strategie decisionali configurabili
- Sceglie la strategia ottimale con spiegazione

Processo:
1. Ricevi intent + context
2. Genera alternative possibili
3. Consulta Gideon per previsioni
4. Valuta rischi/benefici
5. Applica strategia decisionale
6. Scegli azione ottimale
7. Restituisci decisione con reasoning
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import random


# ========== ENUMS ==========

class DecisionStrategy(Enum):
    """Strategie decisionali disponibili"""
    CONSERVATIVE = "conservative"   # Minimizza rischi, preferisce sicurezza
    BALANCED = "balanced"           # Bilancia rischi/benefici
    AGGRESSIVE = "aggressive"       # Massimizza efficienza, tollera rischi
    USER_CENTRIC = "user_centric"   # Priorità preferenze utente
    SPEED = "speed"                 # Minimizza tempo esecuzione
    GIDEON_GUIDED = "gideon_guided" # Segue raccomandazioni Gideon


class DecisionOutcome(Enum):
    """Possibili esiti della decisione"""
    EXECUTE = "execute"              # Esegui immediatamente
    EXECUTE_MONITORED = "execute_monitored"  # Esegui con monitoraggio
    CONFIRM = "confirm"              # Richiedi conferma utente
    SUGGEST = "suggest"              # Suggerisci senza eseguire
    CLARIFY = "clarify"              # Chiedi chiarimenti
    REJECT = "reject"                # Rifiuta (troppo rischioso)
    DELEGATE = "delegate"            # Delega a Gideon
    DEFER = "defer"                  # Rimanda (condizioni non ottimali)
    SPLIT = "split"                  # Dividi in sotto-azioni


class RiskLevel(Enum):
    """Livelli di rischio"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ConfidenceLevel(Enum):
    """Livelli di confidenza"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


# ========== DATACLASSES ==========

@dataclass
class Alternative:
    """Rappresenta un'alternativa di azione"""
    id: str
    action: Dict
    score: float = 0.0
    reasoning: str = ""
    
    # Valutazioni
    confidence: float = 0.5
    risk_level: RiskLevel = RiskLevel.LOW
    safety_score: float = 0.8
    efficiency_score: float = 0.5
    user_preference_score: float = 0.5
    gideon_score: float = 0.5
    
    # Metadata
    risks: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    
    estimated_time: float = 1.0  # secondi
    reversible: bool = True
    requires_confirmation: bool = False
    
    # Gideon analysis
    gideon_recommendation: Optional[str] = None
    predicted_success_rate: float = 0.8
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action": self.action,
            "score": round(self.score, 3),
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level.name,
            "safety_score": round(self.safety_score, 3),
            "efficiency_score": round(self.efficiency_score, 3),
            "risks": self.risks,
            "benefits": self.benefits,
            "estimated_time": self.estimated_time,
            "reversible": self.reversible,
            "gideon_recommendation": self.gideon_recommendation,
            "predicted_success_rate": round(self.predicted_success_rate, 3)
        }


@dataclass
class GideonConsultation:
    """Risultato consultazione Gideon"""
    consulted: bool = False
    recommendation: str = ""
    confidence: float = 0.0
    predictions: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    suggested_alternatives: List[Dict] = field(default_factory=list)
    context_insights: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "consulted": self.consulted,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 3),
            "predictions": self.predictions,
            "warnings": self.warnings,
            "suggested_alternatives": self.suggested_alternatives
        }


@dataclass
class Decision:
    """Rappresenta una decisione finale"""
    id: str
    outcome: DecisionOutcome
    chosen_action: Dict
    confidence: float
    reasoning: str
    strategy_used: DecisionStrategy
    
    alternatives: List[Alternative] = field(default_factory=list)
    gideon_consultation: Optional[GideonConsultation] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    execution_deadline: Optional[datetime] = None
    user_confirmed: Optional[bool] = None
    
    # Tracking
    context: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "outcome": self.outcome.value,
            "chosen_action": self.chosen_action,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "strategy_used": self.strategy_used.value,
            "alternatives": [a.to_dict() for a in self.alternatives[:5]],
            "gideon_consultation": self.gideon_consultation.to_dict() if self.gideon_consultation else None,
            "timestamp": self.timestamp.isoformat(),
            "user_confirmed": self.user_confirmed
        }


# ========== DECISION ENGINE ==========

class DecisionEngine:
    """
    🧠 Motore Decisionale Avanzato
    
    Valuta alternative, consulta Gideon, sceglie strategia ottimale.
    
    Features:
    - Multi-criteria scoring
    - Gideon consultation per previsioni
    - Strategie configurabili
    - Risk assessment integrato
    - Learning dalle decisioni passate
    """
    
    def __init__(self, mode_manager=None, security_validator=None):
        self.mode_manager = mode_manager
        self.security = security_validator
        self.gideon = None  # Collegamento a Gideon Core
        
        # Strategy attiva
        self._current_strategy = DecisionStrategy.BALANCED
        
        # Decision history
        self._decision_history: List[Decision] = []
        self._max_history = 200
        
        # Pesi per scoring (configurabili per strategia)
        self._scoring_weights = {
            DecisionStrategy.CONSERVATIVE: {
                "confidence": 0.2,
                "safety": 0.35,
                "efficiency": 0.1,
                "user_preference": 0.15,
                "gideon": 0.2
            },
            DecisionStrategy.BALANCED: {
                "confidence": 0.25,
                "safety": 0.25,
                "efficiency": 0.2,
                "user_preference": 0.15,
                "gideon": 0.15
            },
            DecisionStrategy.AGGRESSIVE: {
                "confidence": 0.3,
                "safety": 0.1,
                "efficiency": 0.35,
                "user_preference": 0.1,
                "gideon": 0.15
            },
            DecisionStrategy.USER_CENTRIC: {
                "confidence": 0.2,
                "safety": 0.2,
                "efficiency": 0.1,
                "user_preference": 0.35,
                "gideon": 0.15
            },
            DecisionStrategy.SPEED: {
                "confidence": 0.25,
                "safety": 0.15,
                "efficiency": 0.4,
                "user_preference": 0.1,
                "gideon": 0.1
            },
            DecisionStrategy.GIDEON_GUIDED: {
                "confidence": 0.15,
                "safety": 0.2,
                "efficiency": 0.15,
                "user_preference": 0.1,
                "gideon": 0.4
            }
        }
        
        # Soglie per outcome
        self._outcome_thresholds = {
            "execute": 0.75,
            "execute_monitored": 0.6,
            "confirm": 0.45,
            "suggest": 0.3,
            "clarify": 0.15
        }
        
        # Action classification
        self._action_risks = {
            # Critical risk
            "shutdown": RiskLevel.CRITICAL,
            "restart": RiskLevel.CRITICAL,
            "format": RiskLevel.CRITICAL,
            "delete_system": RiskLevel.CRITICAL,
            
            # High risk
            "delete_file": RiskLevel.HIGH,
            "uninstall": RiskLevel.HIGH,
            "modify_registry": RiskLevel.HIGH,
            "kill_process": RiskLevel.HIGH,
            
            # Medium risk
            "create_file": RiskLevel.MEDIUM,
            "move_file": RiskLevel.MEDIUM,
            "close_app": RiskLevel.MEDIUM,
            "run_command": RiskLevel.MEDIUM,
            "send_email": RiskLevel.MEDIUM,
            
            # Low risk
            "open_app": RiskLevel.LOW,
            "open_file": RiskLevel.LOW,
            "open_url": RiskLevel.LOW,
            "copy_file": RiskLevel.LOW,
            "set_volume": RiskLevel.LOW,
            
            # No risk
            "search_web": RiskLevel.NONE,
            "weather": RiskLevel.NONE,
            "time": RiskLevel.NONE,
            "date": RiskLevel.NONE,
            "calculate": RiskLevel.NONE,
            "respond": RiskLevel.NONE,
            "notify": RiskLevel.NONE
        }
        
        # Time estimates (secondi)
        self._time_estimates = {
            "shutdown": 15.0,
            "restart": 45.0,
            "open_app": 3.0,
            "close_app": 1.0,
            "search_web": 2.0,
            "open_url": 2.0,
            "open_file": 2.0,
            "create_file": 1.0,
            "delete_file": 0.5,
            "copy_file": 2.0,
            "move_file": 2.0,
            "send_email": 3.0,
            "weather": 2.0,
            "calculate": 0.1,
            "respond": 0.5,
            "notify": 0.3
        }
        
        # User preferences (learned)
        self._user_preferences: Dict[str, float] = {}
        
        # Stats
        self._stats = {
            "total_decisions": 0,
            "outcomes": {},
            "avg_confidence": 0.0,
            "gideon_consultations": 0,
            "strategies_used": {}
        }
    
    def link_gideon(self, gideon_core):
        """Collega Gideon Core per consultazioni"""
        self.gideon = gideon_core
    
    def set_strategy(self, strategy: DecisionStrategy):
        """Imposta strategia decisionale"""
        self._current_strategy = strategy
    
    # ========== MAIN DECISION FLOW ==========
    
    async def decide(self, intent: Dict, context: Dict = None) -> Decision:
        """
        🎯 Processo decisionale principale
        
        Pipeline:
        1. Genera alternative
        2. Consulta Gideon (se disponibile)
        3. Valuta ogni alternativa
        4. Applica strategia
        5. Scegli azione ottimale
        6. Determina outcome
        7. Crea e restituisci Decision
        
        Args:
            intent: Intent interpretato
            context: Contesto (mode, history, user_prefs, etc.)
            
        Returns:
            Decision completa con reasoning
        """
        context = context or {}
        start_time = datetime.now()
        
        # 1. Genera alternative
        alternatives = await self._generate_alternatives(intent, context)
        
        # 2. Consulta Gideon per analisi predittiva
        gideon_consultation = await self._consult_gideon(intent, alternatives, context)
        
        # 3. Valuta ogni alternativa
        for alt in alternatives:
            await self._evaluate_alternative(alt, context, gideon_consultation)
        
        # 4. Applica strategia e calcola score finale
        strategy = self._select_strategy(intent, context)
        for alt in alternatives:
            alt.score = self._calculate_final_score(alt, strategy)
        
        # 5. Ordina per score
        alternatives.sort(key=lambda x: x.score, reverse=True)
        
        # 6. Scegli la migliore e determina outcome
        best = alternatives[0] if alternatives else None
        outcome, reasoning = await self._determine_outcome(best, intent, context, gideon_consultation)
        
        # 7. Crea Decision
        decision = Decision(
            id=f"dec_{start_time.strftime('%Y%m%d%H%M%S%f')}",
            outcome=outcome,
            chosen_action=best.action if best else {},
            confidence=best.score if best else 0.0,
            reasoning=reasoning,
            strategy_used=strategy,
            alternatives=alternatives,
            gideon_consultation=gideon_consultation,
            context=context
        )
        
        # Track decision
        self._track_decision(decision)
        
        return decision
    
    async def quick_decide(self, intent: Dict) -> Decision:
        """Decisione rapida per azioni semplici (senza consultare Gideon)"""
        action_type = intent.get("name", "unknown")
        confidence = intent.get("confidence", 0.5)
        entities = intent.get("entities", {})
        
        # Skip Gideon per azioni safe
        risk = self._action_risks.get(action_type, RiskLevel.MEDIUM)
        
        if risk in [RiskLevel.NONE, RiskLevel.LOW] and confidence > 0.7:
            action = self._intent_to_action(action_type, entities)
            
            return Decision(
                id=f"dec_quick_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                outcome=DecisionOutcome.EXECUTE,
                chosen_action=action or {},
                confidence=confidence,
                reasoning=f"Quick decision: {action_type} (low risk, high confidence)",
                strategy_used=DecisionStrategy.SPEED,
                alternatives=[],
                gideon_consultation=None
            )
        
        # Fallback a decisione completa
        return await self.decide(intent, {})
    
    # ========== ALTERNATIVE GENERATION ==========
    
    async def _generate_alternatives(self, intent: Dict, context: Dict) -> List[Alternative]:
        """Genera alternative possibili"""
        alternatives = []
        alt_counter = 0
        
        intent_name = intent.get("name", "unknown")
        entities = intent.get("entities", {})
        confidence = intent.get("confidence", 0.5)
        
        # 1. Azione primaria
        primary_action = self._intent_to_action(intent_name, entities)
        if primary_action:
            alt = Alternative(
                id=f"alt_{alt_counter}",
                action=primary_action,
                confidence=confidence,
                reasoning=f"Azione diretta: {intent_name}"
            )
            alternatives.append(alt)
            alt_counter += 1
        
        # 2. Alternative da intent secondari
        for alt_intent in intent.get("alternatives", []):
            alt_action = self._intent_to_action(
                alt_intent.get("name"),
                alt_intent.get("entities", {})
            )
            if alt_action:
                alt = Alternative(
                    id=f"alt_{alt_counter}",
                    action=alt_action,
                    confidence=alt_intent.get("confidence", 0.3),
                    reasoning=f"Alternativa: {alt_intent.get('name')}"
                )
                alternatives.append(alt)
                alt_counter += 1
        
        # 3. Alternative contestuali
        contextual_alts = await self._generate_contextual_alternatives(intent, context)
        for action, reason in contextual_alts:
            alt = Alternative(
                id=f"alt_{alt_counter}",
                action=action,
                confidence=0.4,
                reasoning=reason
            )
            alternatives.append(alt)
            alt_counter += 1
        
        # 4. Fallback help
        if not alternatives:
            alternatives.append(Alternative(
                id="alt_fallback",
                action={"action": "help", "params": {}},
                confidence=0.2,
                reasoning="Fallback: nessuna azione chiara"
            ))
        
        return alternatives
    
    async def _generate_contextual_alternatives(self, intent: Dict, context: Dict) -> List[Tuple[Dict, str]]:
        """Genera alternative basate sul contesto"""
        alternatives = []
        intent_name = intent.get("name", "")
        
        # Se cerca app, suggerisci anche ricerca web
        if intent_name == "open_app":
            app = intent.get("entities", {}).get("app_name", "")
            alternatives.append((
                {"action": "search_web", "params": {"query": f"{app} download"}},
                f"Alternativa: cerca {app} online"
            ))
        
        # Se cerca file, suggerisci ricerca
        if intent_name == "open_file":
            filename = intent.get("entities", {}).get("filename", "")
            alternatives.append((
                {"action": "search_file", "params": {"query": filename}},
                f"Alternativa: cerca {filename} nel sistema"
            ))
        
        # Se errore, suggerisci help
        if intent_name == "unknown":
            alternatives.append((
                {"action": "help", "params": {}},
                "Suggerimento: mostra aiuto"
            ))
        
        return alternatives
    
    def _intent_to_action(self, intent_name: str, entities: Dict) -> Optional[Dict]:
        """Converte intent in azione"""
        action_map = {
            # System
            "shutdown": {"action": "shutdown", "params": {}},
            "restart": {"action": "restart", "params": {}},
            "lock": {"action": "lock", "params": {}},
            "sleep": {"action": "sleep", "params": {}},
            "screenshot": {"action": "screenshot", "params": {}},
            
            # Apps
            "open_app": {"action": "open_app", "params": {"name": entities.get("app_name", "")}},
            "close_app": {"action": "close_app", "params": {"name": entities.get("app_name", "")}},
            
            # Files
            "open_file": {"action": "open_file", "params": {"path": entities.get("file_name", "")}},
            "create_file": {"action": "create_file", "params": {"path": entities.get("file_name", "")}},
            "delete_file": {"action": "delete_file", "params": {"path": entities.get("file_name", "")}},
            
            # Web
            "search_web": {"action": "search_web", "params": {"query": entities.get("query", "")}},
            "open_url": {"action": "open_url", "params": {"url": entities.get("url", "")}},
            
            # Info
            "weather": {"action": "weather", "params": {"location": entities.get("location", "")}},
            "time": {"action": "time", "params": {}},
            "date": {"action": "date", "params": {}},
            "calculate": {"action": "calculate", "params": {"expression": entities.get("expression", "")}},
            
            # Media
            "volume_up": {"action": "volume_up", "params": {}},
            "volume_down": {"action": "volume_down", "params": {}},
            "set_volume": {"action": "set_volume", "params": {"level": entities.get("level", 50)}},
            "mute": {"action": "mute", "params": {}},
            "play_music": {"action": "play_music", "params": {"track": entities.get("track", "")}},
            
            # Conversation
            "greeting": {"action": "respond", "params": {"type": "greeting"}},
            "thanks": {"action": "respond", "params": {"type": "thanks"}},
            "help": {"action": "help", "params": {}},
            
            # Automation
            "run_routine": {"action": "run_routine", "params": {"name": entities.get("routine_name", "")}},
            "create_routine": {"action": "create_routine", "params": {"name": entities.get("routine_name", "")}}
        }
        return action_map.get(intent_name)
    
    # ========== GIDEON CONSULTATION ==========
    
    async def _consult_gideon(self, intent: Dict, alternatives: List[Alternative], 
                              context: Dict) -> Optional[GideonConsultation]:
        """Consulta Gideon per analisi predittiva"""
        if not self.gideon:
            return None
        
        consultation = GideonConsultation(consulted=True)
        self._stats["gideon_consultations"] += 1
        
        try:
            # Chiedi raccomandazione a Gideon
            gideon_response = await self._ask_gideon_recommendation(intent, context)
            
            if gideon_response:
                consultation.recommendation = gideon_response.get("recommendation", "")
                consultation.confidence = gideon_response.get("confidence", 0.5)
                consultation.predictions = gideon_response.get("predictions", {})
                consultation.warnings = gideon_response.get("warnings", [])
                consultation.suggested_alternatives = gideon_response.get("alternatives", [])
                consultation.context_insights = gideon_response.get("context_insights", {})
                
        except Exception as e:
            consultation.warnings.append(f"Errore consultazione Gideon: {str(e)}")
        
        return consultation
    
    async def _ask_gideon_recommendation(self, intent: Dict, context: Dict) -> Optional[Dict]:
        """Chiede raccomandazione specifica a Gideon"""
        if not self.gideon:
            return None
        
        try:
            # Prepara query per Gideon
            query = {
                "type": "decision_support",
                "intent": intent,
                "context": context,
                "request": "analyze_and_recommend"
            }
            
            # Se Gideon ha get_recommendation
            if hasattr(self.gideon, 'get_recommendation'):
                return await self.gideon.get_recommendation(
                    f"Analizza l'intent '{intent.get('name')}' e fornisci raccomandazione",
                    context
                )
            
            # Fallback: analisi basica
            return {
                "recommendation": f"Procedi con {intent.get('name')}",
                "confidence": 0.6,
                "predictions": {
                    "success_probability": 0.8,
                    "estimated_impact": "low"
                },
                "warnings": [],
                "alternatives": []
            }
            
        except Exception:
            return None
    
    # ========== ALTERNATIVE EVALUATION ==========
    
    async def _evaluate_alternative(self, alt: Alternative, context: Dict,
                                    gideon: Optional[GideonConsultation]):
        """Valuta un'alternativa su multiple dimensioni"""
        action_type = alt.action.get("action", "")
        
        # 1. Risk assessment
        alt.risk_level = self._action_risks.get(action_type, RiskLevel.MEDIUM)
        alt.safety_score = self._calculate_safety_score(alt)
        
        # 2. Efficiency
        alt.estimated_time = self._time_estimates.get(action_type, 2.0)
        alt.efficiency_score = self._calculate_efficiency_score(alt)
        
        # 3. Reversibility
        alt.reversible = self._is_reversible(action_type)
        
        # 4. User preference
        alt.user_preference_score = self._get_user_preference(action_type)
        
        # 5. Gideon score
        if gideon and gideon.consulted:
            alt.gideon_score = self._calculate_gideon_score(alt, gideon)
            alt.gideon_recommendation = gideon.recommendation
            alt.predicted_success_rate = gideon.predictions.get("success_probability", 0.8)
        
        # 6. Mode compatibility
        mode_penalty = await self._check_mode_compatibility(alt, context)
        alt.safety_score *= mode_penalty
        
        # 7. Security check
        if self.security:
            security_result = await self.security.validate_action(alt.action)
            if not security_result.get("allowed", True):
                alt.risks.append(f"Bloccato: {security_result.get('reason', 'security')}")
                alt.safety_score *= 0.1
                alt.requires_confirmation = True
        
        # 8. Populate risks/benefits
        self._populate_risks_benefits(alt)
    
    def _calculate_safety_score(self, alt: Alternative) -> float:
        """Calcola safety score basato su risk level"""
        risk_to_safety = {
            RiskLevel.NONE: 1.0,
            RiskLevel.LOW: 0.9,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.4,
            RiskLevel.CRITICAL: 0.1
        }
        base_score = risk_to_safety.get(alt.risk_level, 0.5)
        
        # Bonus se reversibile
        if alt.reversible:
            base_score = min(1.0, base_score + 0.1)
        
        return base_score
    
    def _calculate_efficiency_score(self, alt: Alternative) -> float:
        """Calcola efficiency score basato su tempo stimato"""
        time = alt.estimated_time
        
        if time < 0.5:
            return 1.0
        elif time < 2.0:
            return 0.9
        elif time < 5.0:
            return 0.7
        elif time < 15.0:
            return 0.5
        else:
            return 0.3
    
    def _is_reversible(self, action_type: str) -> bool:
        """Verifica se azione è reversibile"""
        irreversible = {
            "shutdown", "restart", "delete_file", "format",
            "send_email", "delete_system", "uninstall"
        }
        return action_type not in irreversible
    
    def _get_user_preference(self, action_type: str) -> float:
        """Ottiene preferenza utente per tipo azione"""
        return self._user_preferences.get(action_type, 0.5)
    
    def _calculate_gideon_score(self, alt: Alternative, gideon: GideonConsultation) -> float:
        """Calcola score basato su raccomandazione Gideon"""
        action_type = alt.action.get("action", "")
        
        # Se Gideon ha suggerito questa azione
        for suggested in gideon.suggested_alternatives:
            if suggested.get("action") == action_type:
                return 0.9
        
        # Se ci sono warning per questa azione
        for warning in gideon.warnings:
            if action_type.lower() in warning.lower():
                return 0.3
        
        # Score base dalla confidenza Gideon
        return gideon.confidence * 0.8
    
    async def _check_mode_compatibility(self, alt: Alternative, context: Dict) -> float:
        """Verifica compatibilità con modalità operativa"""
        if not self.mode_manager:
            return 1.0
        
        action_type = alt.action.get("action", "")
        
        try:
            can_exec = self.mode_manager.can_execute(action_type)
            
            if not can_exec.get("allowed", True):
                alt.risks.append(f"Non permesso in modalità {self.mode_manager.mode_name}")
                return 0.2
            
            if can_exec.get("requires_confirmation"):
                alt.requires_confirmation = True
                return 0.7
            
            return 1.0
            
        except Exception:
            return 0.8
    
    def _populate_risks_benefits(self, alt: Alternative):
        """Popola rischi e benefici"""
        action_type = alt.action.get("action", "")
        
        # Risks
        if alt.risk_level == RiskLevel.CRITICAL:
            alt.risks.append("Azione critica - può causare perdita dati")
        elif alt.risk_level == RiskLevel.HIGH:
            alt.risks.append("Azione ad alto rischio")
        
        if not alt.reversible:
            alt.risks.append("Azione non reversibile")
        
        if alt.estimated_time > 10:
            alt.risks.append("Tempo esecuzione lungo")
        
        # Benefits
        if alt.risk_level == RiskLevel.NONE:
            alt.benefits.append("Azione completamente sicura")
        
        if alt.reversible:
            alt.benefits.append("Azione reversibile")
        
        if alt.efficiency_score > 0.8:
            alt.benefits.append("Esecuzione rapida")
        
        if alt.gideon_score > 0.7:
            alt.benefits.append("Raccomandato da Gideon")
    
    # ========== STRATEGY & SCORING ==========
    
    def _select_strategy(self, intent: Dict, context: Dict) -> DecisionStrategy:
        """Seleziona strategia ottimale per il contesto"""
        # Check explicit strategy in context
        if "strategy" in context:
            try:
                return DecisionStrategy(context["strategy"])
            except ValueError:
                pass
        
        # Strategia basata su modalità
        if self.mode_manager:
            mode = self.mode_manager.mode.value if hasattr(self.mode_manager, 'mode') else "copilot"
            
            if mode == "executive":
                return DecisionStrategy.AGGRESSIVE
            elif mode == "pilot":
                return DecisionStrategy.BALANCED
            elif mode == "passive":
                return DecisionStrategy.CONSERVATIVE
        
        # Default
        return self._current_strategy
    
    def _calculate_final_score(self, alt: Alternative, strategy: DecisionStrategy) -> float:
        """Calcola score finale usando pesi della strategia"""
        weights = self._scoring_weights.get(strategy, self._scoring_weights[DecisionStrategy.BALANCED])
        
        score = (
            alt.confidence * weights["confidence"] +
            alt.safety_score * weights["safety"] +
            alt.efficiency_score * weights["efficiency"] +
            alt.user_preference_score * weights["user_preference"] +
            alt.gideon_score * weights["gideon"]
        )
        
        # Penalty per risks
        risk_penalty = len(alt.risks) * 0.05
        score = max(0, score - risk_penalty)
        
        # Bonus per benefits
        benefit_bonus = len(alt.benefits) * 0.02
        score = min(1.0, score + benefit_bonus)
        
        return score
    
    # ========== OUTCOME DETERMINATION ==========
    
    async def _determine_outcome(self, best: Optional[Alternative], intent: Dict,
                                  context: Dict, gideon: Optional[GideonConsultation]
                                  ) -> Tuple[DecisionOutcome, str]:
        """Determina l'outcome finale"""
        
        if not best:
            return DecisionOutcome.CLARIFY, "Non ho capito cosa vuoi fare. Puoi riformulare?"
        
        action_type = best.action.get("action", "")
        score = best.score
        
        # Override per azioni critiche
        if best.risk_level == RiskLevel.CRITICAL:
            return DecisionOutcome.CONFIRM, f"⚠️ Azione critica '{action_type}': confermi?"
        
        # Check Gideon warnings
        if gideon and gideon.warnings:
            for warning in gideon.warnings:
                if "pericoloso" in warning.lower() or "danger" in warning.lower():
                    return DecisionOutcome.REJECT, f"Gideon sconsiglia: {warning}"
        
        # Check se richiede conferma
        if best.requires_confirmation:
            return DecisionOutcome.CONFIRM, f"Confermi l'esecuzione di {action_type}?"
        
        # Basato su score
        if score >= self._outcome_thresholds["execute"]:
            return DecisionOutcome.EXECUTE, f"✅ Eseguo {action_type} (confidence: {score:.0%})"
        
        elif score >= self._outcome_thresholds["execute_monitored"]:
            return DecisionOutcome.EXECUTE_MONITORED, f"🔍 Eseguo {action_type} con monitoraggio"
        
        elif score >= self._outcome_thresholds["confirm"]:
            return DecisionOutcome.CONFIRM, f"Vuoi che esegua {action_type}?"
        
        elif score >= self._outcome_thresholds["suggest"]:
            return DecisionOutcome.SUGGEST, f"Suggerisco: {action_type}"
        
        else:
            return DecisionOutcome.CLARIFY, f"Non sono sicuro. Intendevi {action_type}?"
    
    # ========== TRACKING & LEARNING ==========
    
    def _track_decision(self, decision: Decision):
        """Traccia decisione per statistiche e learning"""
        self._decision_history.append(decision)
        
        if len(self._decision_history) > self._max_history:
            self._decision_history = self._decision_history[-self._max_history:]
        
        # Update stats
        self._stats["total_decisions"] += 1
        
        outcome = decision.outcome.value
        self._stats["outcomes"][outcome] = self._stats["outcomes"].get(outcome, 0) + 1
        
        strategy = decision.strategy_used.value
        self._stats["strategies_used"][strategy] = self._stats["strategies_used"].get(strategy, 0) + 1
        
        # Running average confidence
        total = self._stats["total_decisions"]
        self._stats["avg_confidence"] = (
            (self._stats["avg_confidence"] * (total - 1) + decision.confidence) / total
        )
    
    def learn_user_preference(self, action_type: str, positive: bool):
        """Impara preferenza utente da feedback"""
        current = self._user_preferences.get(action_type, 0.5)
        
        if positive:
            self._user_preferences[action_type] = min(1.0, current + 0.05)
        else:
            self._user_preferences[action_type] = max(0.0, current - 0.05)
    
    # ========== CONFIRMATION ==========
    
    def confirm_decision(self, decision_id: str) -> Optional[Decision]:
        """Conferma una decisione in attesa"""
        for dec in self._decision_history:
            if dec.id == decision_id:
                dec.user_confirmed = True
                dec.outcome = DecisionOutcome.EXECUTE
                
                # Learn positive preference
                action = dec.chosen_action.get("action", "")
                if action:
                    self.learn_user_preference(action, True)
                
                return dec
        return None
    
    def reject_decision(self, decision_id: str, reason: str = "") -> Optional[Decision]:
        """Rifiuta una decisione"""
        for dec in self._decision_history:
            if dec.id == decision_id:
                dec.user_confirmed = False
                dec.outcome = DecisionOutcome.REJECT
                dec.metadata["rejection_reason"] = reason
                
                # Learn negative preference
                action = dec.chosen_action.get("action", "")
                if action:
                    self.learn_user_preference(action, False)
                
                return dec
        return None
    
    def get_pending_decisions(self) -> List[Decision]:
        """Ottiene decisioni in attesa di conferma"""
        return [d for d in self._decision_history 
                if d.outcome == DecisionOutcome.CONFIRM and d.user_confirmed is None]
    
    # ========== PUBLIC API ==========
    
    def get_history(self, limit: int = 20) -> List[dict]:
        """Storico decisioni"""
        return [d.to_dict() for d in self._decision_history[-limit:]]
    
    def get_stats(self) -> Dict:
        """Statistiche motore decisionale"""
        return {
            **self._stats,
            "current_strategy": self._current_strategy.value,
            "pending_decisions": len(self.get_pending_decisions()),
            "gideon_linked": self.gideon is not None,
            "security_linked": self.security is not None
        }
    
    def get_status(self) -> Dict:
        """Stato completo"""
        return {
            "strategy": self._current_strategy.value,
            "total_decisions": self._stats["total_decisions"],
            "avg_confidence": round(self._stats["avg_confidence"], 3),
            "gideon_consultations": self._stats["gideon_consultations"],
            "pending": len(self.get_pending_decisions()),
            "gideon_available": self.gideon is not None
        }
