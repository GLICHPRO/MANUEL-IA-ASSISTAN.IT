"""
📡 Sistema di Comunicazione Jarvis ↔ Gideon

Gestisce la comunicazione bidirezionale strutturata e loggata tra:
- JARVIS (Executive AI): Invia obiettivi, vincoli, richieste di analisi
- GIDEON (Predictive AI): Risponde con esiti, ranking, probabilità, rischi

Tutte le comunicazioni sono:
- Strutturate (formato definito)
- Loggate (tracciabilità completa)
- Tipizzate (enum per tipi messaggio)
- Timestampate (cronologia precisa)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from datetime import datetime
import uuid
import json
import logging
from pathlib import Path


# Setup logging dedicato
comm_logger = logging.getLogger("jarvis_gideon_comm")
comm_logger.setLevel(logging.DEBUG)

# File handler per persistenza
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / "communication.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
))
comm_logger.addHandler(file_handler)


# === ENUMS ===

class MessageType(Enum):
    """Tipi di messaggio nella comunicazione"""
    # Jarvis → Gideon
    OBJECTIVE = "objective"           # Obiettivo da analizzare
    CONSTRAINT = "constraint"         # Vincolo da considerare
    ANALYSIS_REQUEST = "analysis_request"  # Richiesta di analisi
    SCENARIO_REQUEST = "scenario_request"  # Richiesta scenari
    RISK_CHECK = "risk_check"         # Verifica rischi
    SIMULATION_REQUEST = "simulation_request"  # Richiesta simulazione
    RANKING_REQUEST = "ranking_request"  # Richiesta classifica
    VALIDATION_REQUEST = "validation_request"  # Validazione decisione
    
    # Gideon → Jarvis
    ANALYSIS_RESULT = "analysis_result"  # Risultato analisi
    SCENARIO_RESULT = "scenario_result"  # Scenari generati
    RISK_ASSESSMENT = "risk_assessment"  # Valutazione rischi
    SIMULATION_RESULT = "simulation_result"  # Risultati simulazione
    RANKING_RESULT = "ranking_result"  # Classifica
    PROBABILITY_RESULT = "probability_result"  # Calcolo probabilità
    RECOMMENDATION = "recommendation"  # Raccomandazione
    WARNING = "warning"               # Avviso
    VALIDATION_RESULT = "validation_result"  # Esito validazione


class MessagePriority(Enum):
    """Priorità del messaggio"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class MessageStatus(Enum):
    """Stato del messaggio"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Sender(Enum):
    """Mittente del messaggio"""
    JARVIS = "jarvis"
    GIDEON = "gideon"
    SYSTEM = "system"


# === DATA CLASSES ===

@dataclass
class Objective:
    """Obiettivo inviato da Jarvis a Gideon"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    priority: int = 5
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "constraints": self.constraints,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "context": self.context,
            "metadata": self.metadata
        }


@dataclass
class Constraint:
    """Vincolo operativo"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = ""  # time, resource, safety, policy, technical
    description: str = ""
    value: Any = None
    is_hard: bool = True  # Hard constraint = non negoziabile
    weight: float = 1.0  # Per soft constraints
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "value": self.value,
            "is_hard": self.is_hard,
            "weight": self.weight
        }


@dataclass 
class GideonResponse:
    """Risposta strutturata di Gideon"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    request_id: str = ""  # ID della richiesta originale
    response_type: MessageType = MessageType.ANALYSIS_RESULT
    
    # Contenuto principale
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Metriche chiave
    probability: Optional[float] = None  # Probabilità successo
    confidence: float = 0.0  # Confidenza nella risposta
    risk_level: str = "unknown"  # low, medium, high, critical
    risk_score: float = 0.0
    
    # Ranking e opzioni
    ranking: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    alternatives: List[Dict] = field(default_factory=list)
    
    # Warnings e note
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    
    # Flag per Jarvis
    requires_action: bool = False
    suggested_action: str = ""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "response_type": self.response_type.value,
            "summary": self.summary,
            "details": self.details,
            "probability": self.probability,
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 3),
            "ranking": self.ranking,
            "recommendations": self.recommendations,
            "alternatives": self.alternatives,
            "warnings": self.warnings,
            "notes": self.notes,
            "requires_action": self.requires_action,
            "suggested_action": self.suggested_action,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CommunicationMessage:
    """Messaggio di comunicazione completo"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: Sender = Sender.SYSTEM
    recipient: Sender = Sender.SYSTEM
    message_type: MessageType = MessageType.ANALYSIS_REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    
    # Payload
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Correlazione
    correlation_id: Optional[str] = None  # Per tracciare request/response
    in_reply_to: Optional[str] = None  # ID messaggio a cui risponde
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender.value,
            "recipient": self.recipient.value,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "in_reply_to": self.in_reply_to,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }
    
    def to_log_string(self) -> str:
        """Formato per logging"""
        return (
            f"[{self.id}] {self.sender.value.upper()} → {self.recipient.value.upper()} | "
            f"{self.message_type.value} | Priority: {self.priority.name} | "
            f"Status: {self.status.value}"
        )


# === COMMUNICATION CHANNEL ===

class CommunicationChannel:
    """
    Canale di comunicazione bidirezionale Jarvis ↔ Gideon
    
    Gestisce:
    - Invio/ricezione messaggi strutturati
    - Logging completo
    - Correlazione request/response
    - Code di messaggi
    - Handler per tipi specifici
    """
    
    def __init__(self):
        self.message_history: List[CommunicationMessage] = []
        self.pending_messages: Dict[str, CommunicationMessage] = {}
        self.handlers: Dict[MessageType, List[Callable]] = {}
        self.correlation_map: Dict[str, List[str]] = {}  # correlation_id -> [message_ids]
        
        # Statistiche
        self.stats = {
            "jarvis_to_gideon": 0,
            "gideon_to_jarvis": 0,
            "total_messages": 0,
            "completed": 0,
            "failed": 0
        }
        
        comm_logger.info("Communication Channel initialized")
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Registra handler per tipo di messaggio"""
        if message_type not in self.handlers:
            self.handlers[message_type] = []
        self.handlers[message_type].append(handler)
        comm_logger.debug(f"Handler registered for {message_type.value}")
    
    def send(self, 
             sender: Sender,
             recipient: Sender,
             message_type: MessageType,
             payload: Dict[str, Any],
             priority: MessagePriority = MessagePriority.NORMAL,
             correlation_id: str = None,
             in_reply_to: str = None,
             metadata: Dict = None) -> CommunicationMessage:
        """
        Invia un messaggio nel canale
        
        Returns:
            CommunicationMessage con ID per tracking
        """
        message = CommunicationMessage(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id or str(uuid.uuid4())[:8],
            in_reply_to=in_reply_to,
            metadata=metadata or {}
        )
        
        # Aggiorna statistiche
        self.stats["total_messages"] += 1
        if sender == Sender.JARVIS:
            self.stats["jarvis_to_gideon"] += 1
        elif sender == Sender.GIDEON:
            self.stats["gideon_to_jarvis"] += 1
        
        # Traccia correlazione
        if message.correlation_id not in self.correlation_map:
            self.correlation_map[message.correlation_id] = []
        self.correlation_map[message.correlation_id].append(message.id)
        
        # Aggiungi a pending
        self.pending_messages[message.id] = message
        
        # Log
        comm_logger.info(f"SEND: {message.to_log_string()}")
        comm_logger.debug(f"Payload: {json.dumps(payload, default=str)[:500]}")
        
        # Salva in history
        self.message_history.append(message)
        
        return message
    
    def receive(self, message_id: str) -> Optional[CommunicationMessage]:
        """Recupera messaggio per ID"""
        return self.pending_messages.get(message_id)
    
    def process(self, message_id: str) -> bool:
        """Marca messaggio come in elaborazione"""
        if message_id in self.pending_messages:
            msg = self.pending_messages[message_id]
            msg.status = MessageStatus.PROCESSING
            msg.processed_at = datetime.now()
            comm_logger.info(f"PROCESSING: {msg.to_log_string()}")
            return True
        return False
    
    def complete(self, message_id: str, success: bool = True) -> bool:
        """Marca messaggio come completato"""
        if message_id in self.pending_messages:
            msg = self.pending_messages[message_id]
            msg.status = MessageStatus.COMPLETED if success else MessageStatus.FAILED
            msg.completed_at = datetime.now()
            
            if success:
                self.stats["completed"] += 1
            else:
                self.stats["failed"] += 1
            
            # Rimuovi da pending
            del self.pending_messages[message_id]
            
            comm_logger.info(f"COMPLETED: {msg.to_log_string()}")
            return True
        return False
    
    def get_correlated_messages(self, correlation_id: str) -> List[CommunicationMessage]:
        """Ottiene tutti i messaggi correlati (request + response chain)"""
        message_ids = self.correlation_map.get(correlation_id, [])
        return [
            msg for msg in self.message_history 
            if msg.id in message_ids
        ]
    
    def get_conversation(self, correlation_id: str) -> List[Dict]:
        """Ottiene conversazione formattata"""
        messages = self.get_correlated_messages(correlation_id)
        return [
            {
                "sender": msg.sender.value,
                "type": msg.message_type.value,
                "summary": msg.payload.get("summary", msg.payload.get("description", "")),
                "timestamp": msg.created_at.isoformat()
            }
            for msg in sorted(messages, key=lambda m: m.created_at)
        ]
    
    def get_history(self, 
                    sender: Sender = None,
                    message_type: MessageType = None,
                    limit: int = 50) -> List[CommunicationMessage]:
        """Ottiene storia messaggi filtrata"""
        messages = self.message_history
        
        if sender:
            messages = [m for m in messages if m.sender == sender]
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        return messages[-limit:]
    
    def get_statistics(self) -> Dict:
        """Statistiche comunicazione"""
        return {
            **self.stats,
            "pending": len(self.pending_messages),
            "history_size": len(self.message_history),
            "active_correlations": len(self.correlation_map)
        }


# === JARVIS COMMUNICATOR ===

class JarvisCommunicator:
    """
    Interfaccia di comunicazione lato Jarvis
    
    Jarvis usa questa classe per:
    - Inviare obiettivi e vincoli a Gideon
    - Richiedere analisi, simulazioni, ranking
    - Ricevere risposte strutturate
    """
    
    def __init__(self, channel: CommunicationChannel):
        self.channel = channel
        self.pending_requests: Dict[str, str] = {}  # correlation_id -> description
    
    def send_objective(self, objective: Objective,
                       constraints: List[Constraint] = None) -> str:
        """
        Invia obiettivo a Gideon per analisi
        
        Returns:
            correlation_id per tracking
        """
        payload = {
            "objective": objective.to_dict(),
            "constraints": [c.to_dict() for c in (constraints or [])]
        }
        
        # Mappa priorità obiettivo a MessagePriority
        if objective.priority >= 9:
            priority = MessagePriority.CRITICAL
        elif objective.priority >= 7:
            priority = MessagePriority.HIGH
        elif objective.priority >= 4:
            priority = MessagePriority.NORMAL
        else:
            priority = MessagePriority.LOW
        
        msg = self.channel.send(
            sender=Sender.JARVIS,
            recipient=Sender.GIDEON,
            message_type=MessageType.OBJECTIVE,
            payload=payload,
            priority=priority
        )
        
        self.pending_requests[msg.correlation_id] = objective.description
        
        comm_logger.info(
            f"JARVIS → GIDEON: Objective '{objective.description[:50]}' "
            f"[correlation: {msg.correlation_id}]"
        )
        
        return msg.correlation_id
    
    def request_analysis(self, 
                        context: Dict[str, Any],
                        analysis_type: str = "comprehensive",
                        priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Richiede analisi a Gideon"""
        payload = {
            "context": context,
            "analysis_type": analysis_type,
            "requested_outputs": ["summary", "risks", "recommendations"]
        }
        
        msg = self.channel.send(
            sender=Sender.JARVIS,
            recipient=Sender.GIDEON,
            message_type=MessageType.ANALYSIS_REQUEST,
            payload=payload,
            priority=priority
        )
        
        return msg.correlation_id
    
    def request_scenarios(self,
                         objective: str,
                         constraints: List[Dict] = None,
                         num_scenarios: int = 5) -> str:
        """Richiede generazione scenari"""
        payload = {
            "objective": objective,
            "constraints": constraints or [],
            "num_scenarios": num_scenarios,
            "include_risks": True,
            "include_probabilities": True
        }
        
        msg = self.channel.send(
            sender=Sender.JARVIS,
            recipient=Sender.GIDEON,
            message_type=MessageType.SCENARIO_REQUEST,
            payload=payload,
            priority=MessagePriority.NORMAL
        )
        
        return msg.correlation_id
    
    def request_risk_check(self,
                          action: Dict[str, Any],
                          context: Dict[str, Any] = None) -> str:
        """Richiede verifica rischi per azione"""
        payload = {
            "action": action,
            "context": context or {},
            "check_types": ["safety", "technical", "policy", "resource"]
        }
        
        msg = self.channel.send(
            sender=Sender.JARVIS,
            recipient=Sender.GIDEON,
            message_type=MessageType.RISK_CHECK,
            payload=payload,
            priority=MessagePriority.HIGH
        )
        
        return msg.correlation_id
    
    def request_simulation(self,
                          scenario: Dict[str, Any],
                          iterations: int = 1000,
                          include_sensitivity: bool = True) -> str:
        """Richiede simulazione Monte Carlo"""
        payload = {
            "scenario": scenario,
            "iterations": iterations,
            "include_sensitivity": include_sensitivity,
            "confidence_levels": [0.90, 0.95, 0.99]
        }
        
        msg = self.channel.send(
            sender=Sender.JARVIS,
            recipient=Sender.GIDEON,
            message_type=MessageType.SIMULATION_REQUEST,
            payload=payload,
            priority=MessagePriority.NORMAL
        )
        
        return msg.correlation_id
    
    def request_ranking(self,
                       options: List[Dict],
                       criteria: List[str] = None,
                       method: str = "topsis") -> str:
        """Richiede ranking opzioni"""
        payload = {
            "options": options,
            "criteria": criteria or ["probability", "risk", "cost", "time"],
            "method": method,
            "return_top": 5
        }
        
        msg = self.channel.send(
            sender=Sender.JARVIS,
            recipient=Sender.GIDEON,
            message_type=MessageType.RANKING_REQUEST,
            payload=payload,
            priority=MessagePriority.NORMAL
        )
        
        return msg.correlation_id
    
    def request_validation(self,
                          decision: Dict[str, Any],
                          constraints: List[Dict] = None) -> str:
        """Richiede validazione decisione prima dell'esecuzione"""
        payload = {
            "decision": decision,
            "constraints": constraints or [],
            "validation_level": "strict"
        }
        
        msg = self.channel.send(
            sender=Sender.JARVIS,
            recipient=Sender.GIDEON,
            message_type=MessageType.VALIDATION_REQUEST,
            payload=payload,
            priority=MessagePriority.HIGH
        )
        
        return msg.correlation_id


# === GIDEON COMMUNICATOR ===

class GideonCommunicator:
    """
    Interfaccia di comunicazione lato Gideon
    
    Gideon usa questa classe per:
    - Ricevere richieste da Jarvis
    - Inviare risposte strutturate
    - Emettere warning e raccomandazioni
    """
    
    def __init__(self, channel: CommunicationChannel):
        self.channel = channel
    
    def respond_analysis(self,
                        request_id: str,
                        correlation_id: str,
                        summary: str,
                        details: Dict,
                        probability: float = None,
                        confidence: float = 0.8,
                        risk_level: str = "medium",
                        risk_score: float = 0.5,
                        recommendations: List[str] = None) -> GideonResponse:
        """Risponde con risultato analisi"""
        
        response = GideonResponse(
            request_id=request_id,
            response_type=MessageType.ANALYSIS_RESULT,
            summary=summary,
            details=details,
            probability=probability,
            confidence=confidence,
            risk_level=risk_level,
            risk_score=risk_score,
            recommendations=recommendations or []
        )
        
        msg = self.channel.send(
            sender=Sender.GIDEON,
            recipient=Sender.JARVIS,
            message_type=MessageType.ANALYSIS_RESULT,
            payload=response.to_dict(),
            correlation_id=correlation_id,
            in_reply_to=request_id
        )
        
        comm_logger.info(
            f"GIDEON → JARVIS: Analysis result [prob: {probability}, "
            f"risk: {risk_level}] [correlation: {correlation_id}]"
        )
        
        return response
    
    def respond_scenarios(self,
                         request_id: str,
                         correlation_id: str,
                         scenarios: List[Dict],
                         ranking: List[Dict] = None,
                         best_scenario: Dict = None) -> GideonResponse:
        """Risponde con scenari generati"""
        
        response = GideonResponse(
            request_id=request_id,
            response_type=MessageType.SCENARIO_RESULT,
            summary=f"Generati {len(scenarios)} scenari. "
                    f"Scenario migliore: {best_scenario.get('name', 'N/A') if best_scenario else 'N/A'}",
            details={"scenarios": scenarios, "best": best_scenario},
            ranking=ranking or [],
            confidence=0.85
        )
        
        if best_scenario:
            response.probability = best_scenario.get("probability", 0.5)
            response.risk_level = best_scenario.get("risk_level", "medium")
        
        msg = self.channel.send(
            sender=Sender.GIDEON,
            recipient=Sender.JARVIS,
            message_type=MessageType.SCENARIO_RESULT,
            payload=response.to_dict(),
            correlation_id=correlation_id,
            in_reply_to=request_id
        )
        
        return response
    
    def respond_risk_assessment(self,
                               request_id: str,
                               correlation_id: str,
                               risk_level: str,
                               risk_score: float,
                               risks: List[Dict],
                               mitigations: List[str] = None,
                               proceed_recommendation: bool = True) -> GideonResponse:
        """Risponde con valutazione rischi"""
        
        response = GideonResponse(
            request_id=request_id,
            response_type=MessageType.RISK_ASSESSMENT,
            summary=f"Rischio {risk_level} (score: {risk_score:.2f}). "
                    f"{'Procedi con cautela.' if proceed_recommendation else 'Sconsiglio procedere.'}",
            details={"risks": risks, "mitigations": mitigations or []},
            risk_level=risk_level,
            risk_score=risk_score,
            recommendations=mitigations or [],
            requires_action=not proceed_recommendation,
            suggested_action="review_risks" if not proceed_recommendation else ""
        )
        
        # Aggiungi warning se rischio alto
        if risk_score > 0.6:
            response.warnings.append(f"Rischio elevato: {risk_score:.0%}")
        
        msg = self.channel.send(
            sender=Sender.GIDEON,
            recipient=Sender.JARVIS,
            message_type=MessageType.RISK_ASSESSMENT,
            payload=response.to_dict(),
            correlation_id=correlation_id,
            in_reply_to=request_id,
            priority=MessagePriority.HIGH if risk_score > 0.6 else MessagePriority.NORMAL
        )
        
        return response
    
    def respond_simulation(self,
                          request_id: str,
                          correlation_id: str,
                          mean: float,
                          std: float,
                          confidence_intervals: Dict[str, tuple],
                          iterations: int,
                          convergence: bool = True) -> GideonResponse:
        """Risponde con risultati simulazione"""
        
        response = GideonResponse(
            request_id=request_id,
            response_type=MessageType.SIMULATION_RESULT,
            summary=f"Simulazione {iterations} iterazioni. "
                    f"Media: {mean:.2f}, Std: {std:.2f}. "
                    f"{'Convergenza raggiunta.' if convergence else 'Non convergente.'}",
            details={
                "mean": mean,
                "std": std,
                "confidence_intervals": confidence_intervals,
                "iterations": iterations,
                "converged": convergence
            },
            probability=mean if 0 <= mean <= 1 else None,
            confidence=0.95 if convergence else 0.7
        )
        
        if not convergence:
            response.warnings.append("Simulazione non convergente - risultati instabili")
        
        msg = self.channel.send(
            sender=Sender.GIDEON,
            recipient=Sender.JARVIS,
            message_type=MessageType.SIMULATION_RESULT,
            payload=response.to_dict(),
            correlation_id=correlation_id,
            in_reply_to=request_id
        )
        
        return response
    
    def respond_ranking(self,
                       request_id: str,
                       correlation_id: str,
                       ranking: List[Dict],
                       method_used: str,
                       top_recommendation: Dict = None) -> GideonResponse:
        """Risponde con classifica"""
        
        response = GideonResponse(
            request_id=request_id,
            response_type=MessageType.RANKING_RESULT,
            summary=f"Classifica dei {len(ranking)} scenari pronta (metodo: {method_used}). "
                    f"Top: {top_recommendation.get('name', 'N/A') if top_recommendation else ranking[0].get('name', 'N/A') if ranking else 'N/A'}",
            details={"method": method_used},
            ranking=ranking,
            confidence=0.88
        )
        
        if top_recommendation:
            response.recommendations.append(
                f"Raccomando: {top_recommendation.get('name')} "
                f"(score: {top_recommendation.get('score', 0):.2f})"
            )
        
        msg = self.channel.send(
            sender=Sender.GIDEON,
            recipient=Sender.JARVIS,
            message_type=MessageType.RANKING_RESULT,
            payload=response.to_dict(),
            correlation_id=correlation_id,
            in_reply_to=request_id
        )
        
        return response
    
    def respond_validation(self,
                          request_id: str,
                          correlation_id: str,
                          is_valid: bool,
                          issues: List[str] = None,
                          suggestions: List[str] = None) -> GideonResponse:
        """Risponde con esito validazione"""
        
        response = GideonResponse(
            request_id=request_id,
            response_type=MessageType.VALIDATION_RESULT,
            summary=f"Validazione: {'APPROVATO ✓' if is_valid else 'NON APPROVATO ✗'}. "
                    f"{len(issues or [])} problemi rilevati.",
            details={"valid": is_valid, "issues": issues or []},
            confidence=0.95,
            requires_action=not is_valid,
            suggested_action="fix_issues" if not is_valid else "proceed",
            recommendations=suggestions or [],
            warnings=issues or []
        )
        
        msg = self.channel.send(
            sender=Sender.GIDEON,
            recipient=Sender.JARVIS,
            message_type=MessageType.VALIDATION_RESULT,
            payload=response.to_dict(),
            correlation_id=correlation_id,
            in_reply_to=request_id,
            priority=MessagePriority.HIGH
        )
        
        return response
    
    def send_warning(self,
                    message: str,
                    severity: str = "medium",
                    context: Dict = None,
                    correlation_id: str = None) -> str:
        """Invia warning proattivo a Jarvis"""
        
        priority = {
            "low": MessagePriority.LOW,
            "medium": MessagePriority.NORMAL,
            "high": MessagePriority.HIGH,
            "critical": MessagePriority.CRITICAL
        }.get(severity, MessagePriority.NORMAL)
        
        payload = {
            "message": message,
            "severity": severity,
            "context": context or {},
            "requires_attention": severity in ["high", "critical"]
        }
        
        msg = self.channel.send(
            sender=Sender.GIDEON,
            recipient=Sender.JARVIS,
            message_type=MessageType.WARNING,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id
        )
        
        comm_logger.warning(f"GIDEON WARNING [{severity}]: {message}")
        
        return msg.correlation_id
    
    def send_recommendation(self,
                           action: str,
                           probability: float,
                           reasoning: str,
                           alternatives: List[Dict] = None,
                           correlation_id: str = None) -> str:
        """Invia raccomandazione proattiva"""
        
        payload = {
            "recommended_action": action,
            "probability": probability,
            "reasoning": reasoning,
            "alternatives": alternatives or []
        }
        
        msg = self.channel.send(
            sender=Sender.GIDEON,
            recipient=Sender.JARVIS,
            message_type=MessageType.RECOMMENDATION,
            payload=payload,
            priority=MessagePriority.NORMAL,
            correlation_id=correlation_id
        )
        
        comm_logger.info(f"GIDEON RECOMMENDATION: {action} (prob: {probability:.0%})")
        
        return msg.correlation_id


# === COMMUNICATION BRIDGE ===

class CommunicationBridge:
    """
    Bridge completo per comunicazione Jarvis ↔ Gideon
    
    Fornisce accesso unificato a entrambi i lati della comunicazione.
    Da usare nell'Orchestrator per coordinare i due sistemi.
    """
    
    def __init__(self):
        self.channel = CommunicationChannel()
        self.jarvis = JarvisCommunicator(self.channel)
        self.gideon = GideonCommunicator(self.channel)
        
        comm_logger.info("Communication Bridge initialized - Jarvis ↔ Gideon ready")
    
    def get_conversation_log(self, correlation_id: str = None) -> List[Dict]:
        """Ottiene log conversazione formattato"""
        if correlation_id:
            return self.channel.get_conversation(correlation_id)
        
        # Ultime conversazioni
        recent = self.channel.get_history(limit=20)
        return [msg.to_dict() for msg in recent]
    
    def get_statistics(self) -> Dict:
        """Statistiche complete"""
        return {
            "channel": self.channel.get_statistics(),
            "pending_jarvis_requests": len(self.jarvis.pending_requests)
        }
    
    def export_log(self, filepath: str = None) -> str:
        """Esporta log comunicazioni in JSON"""
        if filepath is None:
            filepath = str(log_dir / f"comm_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        data = {
            "exported_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "messages": [msg.to_dict() for msg in self.channel.message_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        comm_logger.info(f"Communication log exported to {filepath}")
        return filepath
