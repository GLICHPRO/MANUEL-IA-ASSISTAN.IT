"""
🔗 GIDEON 3.0 - Agent Bus
Sistema di comunicazione multi-agente
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
import uuid
import json


class MessageType(Enum):
    """Tipi di messaggi tra agenti"""
    REQUEST = "request"         # Richiesta di azione/info
    RESPONSE = "response"       # Risposta a una richiesta
    EVENT = "event"             # Notifica evento
    BROADCAST = "broadcast"     # Messaggio a tutti
    COMMAND = "command"         # Comando diretto
    STATUS = "status"           # Aggiornamento stato


class MessagePriority(Enum):
    """Priorità messaggi"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AgentMessage:
    """Messaggio tra agenti"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    sender: str = ""
    recipient: str = ""  # "" = broadcast
    content: dict = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = None  # Per collegare richiesta-risposta
    ttl: int = 60  # Time to live in secondi
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "ttl": self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentMessage':
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "request")),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            content=data.get("content", {}),
            priority=MessagePriority(data.get("priority", 1)),
            correlation_id=data.get("correlation_id"),
            ttl=data.get("ttl", 60)
        )


class Agent:
    """
    Base class per agenti nel sistema
    """
    
    def __init__(self, agent_id: str, name: str):
        self.id = agent_id
        self.name = name
        self.is_active = False
        self._bus: Optional['AgentBus'] = None
        self._message_handlers: Dict[MessageType, Callable] = {}
    
    def set_bus(self, bus: 'AgentBus'):
        """Collega l'agente al bus"""
        self._bus = bus
    
    async def start(self):
        """Avvia l'agente"""
        self.is_active = True
    
    async def stop(self):
        """Ferma l'agente"""
        self.is_active = False
    
    async def send(self, recipient: str, content: dict, 
                   msg_type: MessageType = MessageType.REQUEST,
                   priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Invia un messaggio"""
        if not self._bus:
            raise RuntimeError("Agent not connected to bus")
        
        msg = AgentMessage(
            type=msg_type,
            sender=self.id,
            recipient=recipient,
            content=content,
            priority=priority
        )
        
        await self._bus.publish(msg)
        return msg.id
    
    async def broadcast(self, content: dict, 
                        msg_type: MessageType = MessageType.BROADCAST):
        """Invia messaggio a tutti gli agenti"""
        return await self.send("", content, msg_type)
    
    async def request(self, recipient: str, content: dict, 
                      timeout: float = 30.0) -> Optional[dict]:
        """Invia richiesta e attende risposta"""
        if not self._bus:
            raise RuntimeError("Agent not connected to bus")
        
        return await self._bus.request(self.id, recipient, content, timeout)
    
    async def receive(self, message: AgentMessage):
        """
        Riceve un messaggio - da sovrascrivere nelle sottoclassi
        """
        handler = self._message_handlers.get(message.type)
        if handler:
            return await handler(message)
        return None
    
    def on_message_type(self, msg_type: MessageType, handler: Callable):
        """Registra handler per tipo di messaggio"""
        self._message_handlers[msg_type] = handler
    
    def get_status(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "is_active": self.is_active
        }


class AgentBus:
    """
    Bus di comunicazione tra agenti
    Gestisce routing, queuing e delivery dei messaggi
    """
    
    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._queues: Dict[str, asyncio.Queue] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._message_history: List[AgentMessage] = []
        self._max_history = 1000
        self._subscribers: Dict[str, List[str]] = {}  # topic -> agent_ids
        self._running = False
        self._processor_tasks: Dict[str, asyncio.Task] = {}
    
    async def start(self):
        """Avvia il bus"""
        self._running = True
        # Avvia processor per ogni agente
        for agent_id in self._agents:
            self._start_processor(agent_id)
    
    async def stop(self):
        """Ferma il bus"""
        self._running = False
        for task in self._processor_tasks.values():
            task.cancel()
        self._processor_tasks.clear()
    
    def register_agent(self, agent: Agent) -> bool:
        """Registra un agente nel bus"""
        if agent.id in self._agents:
            return False
        
        self._agents[agent.id] = agent
        self._queues[agent.id] = asyncio.Queue()
        agent.set_bus(self)
        
        # Avvia processor se bus è running
        if self._running:
            self._start_processor(agent.id)
        
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Rimuove un agente dal bus"""
        if agent_id not in self._agents:
            return False
        
        # Ferma processor
        if agent_id in self._processor_tasks:
            self._processor_tasks[agent_id].cancel()
            del self._processor_tasks[agent_id]
        
        del self._agents[agent_id]
        del self._queues[agent_id]
        
        # Rimuovi da subscribers
        for topic in self._subscribers:
            if agent_id in self._subscribers[topic]:
                self._subscribers[topic].remove(agent_id)
        
        return True
    
    def _start_processor(self, agent_id: str):
        """Avvia task di processing per un agente"""
        async def processor():
            queue = self._queues[agent_id]
            agent = self._agents[agent_id]
            
            while self._running:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Verifica TTL
                    age = (datetime.now() - msg.timestamp).total_seconds()
                    if age > msg.ttl:
                        continue  # Messaggio scaduto
                    
                    # Consegna al agent
                    response = await agent.receive(msg)
                    
                    # Se è una richiesta, invia risposta
                    if msg.type == MessageType.REQUEST and response:
                        await self._send_response(msg, response)
                    
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Error processing message for {agent_id}: {e}")
        
        self._processor_tasks[agent_id] = asyncio.create_task(processor())
    
    async def publish(self, message: AgentMessage):
        """Pubblica un messaggio"""
        # Salva in history
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history = self._message_history[-self._max_history:]
        
        # Routing
        if message.recipient:
            # Messaggio diretto
            if message.recipient in self._queues:
                await self._queues[message.recipient].put(message)
        else:
            # Broadcast
            for agent_id, queue in self._queues.items():
                if agent_id != message.sender:
                    await queue.put(message)
    
    async def request(self, sender: str, recipient: str, 
                      content: dict, timeout: float = 30.0) -> Optional[dict]:
        """Invia richiesta e attende risposta"""
        msg = AgentMessage(
            type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            content=content
        )
        
        # Crea future per la risposta
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[msg.id] = future
        
        # Pubblica richiesta
        await self.publish(msg)
        
        try:
            # Attendi risposta
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return None
        finally:
            if msg.id in self._pending_requests:
                del self._pending_requests[msg.id]
    
    async def _send_response(self, request: AgentMessage, response_content: dict):
        """Invia risposta a una richiesta"""
        response = AgentMessage(
            type=MessageType.RESPONSE,
            sender=request.recipient,
            recipient=request.sender,
            content=response_content,
            correlation_id=request.id
        )
        
        # Completa il future se esiste
        if request.id in self._pending_requests:
            self._pending_requests[request.id].set_result(response_content)
        
        await self.publish(response)
    
    # === Pub/Sub per topics ===
    
    def subscribe(self, agent_id: str, topic: str):
        """Sottoscrivi un agente a un topic"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        if agent_id not in self._subscribers[topic]:
            self._subscribers[topic].append(agent_id)
    
    def unsubscribe(self, agent_id: str, topic: str):
        """Rimuovi sottoscrizione"""
        if topic in self._subscribers and agent_id in self._subscribers[topic]:
            self._subscribers[topic].remove(agent_id)
    
    async def publish_to_topic(self, sender: str, topic: str, content: dict):
        """Pubblica a tutti i sottoscrittori di un topic"""
        if topic not in self._subscribers:
            return
        
        for agent_id in self._subscribers[topic]:
            if agent_id != sender and agent_id in self._queues:
                msg = AgentMessage(
                    type=MessageType.EVENT,
                    sender=sender,
                    recipient=agent_id,
                    content={"topic": topic, **content}
                )
                await self._queues[agent_id].put(msg)
    
    # === Utility ===
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Ottiene un agente per ID"""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[dict]:
        """Lista tutti gli agenti"""
        return [agent.get_status() for agent in self._agents.values()]
    
    def get_message_history(self, limit: int = 50) -> List[dict]:
        """Storico messaggi"""
        return [m.to_dict() for m in self._message_history[-limit:]]
    
    def get_status(self) -> dict:
        """Stato del bus"""
        return {
            "running": self._running,
            "agents_count": len(self._agents),
            "agents": list(self._agents.keys()),
            "pending_requests": len(self._pending_requests),
            "topics": {k: len(v) for k, v in self._subscribers.items()},
            "message_history_size": len(self._message_history)
        }


# === Agenti predefiniti ===

class GideonAgent(Agent):
    """Agente Gideon 3.0 - Analisi e Previsioni"""
    
    def __init__(self, gideon_core=None):
        super().__init__("gideon", "Gideon 3.0")
        self.core = gideon_core
    
    async def receive(self, message: AgentMessage):
        if message.type == MessageType.REQUEST:
            action = message.content.get("action")
            
            if action == "analyze" and self.core:
                return await self.core.analyze(message.content.get("data", {}))
            elif action == "predict" and self.core:
                return await self.core.predict(message.content.get("context", {}))
            elif action == "simulate" and self.core:
                return await self.core.simulate(message.content.get("scenario", {}))
        
        return {"received": True}


class JarvisAgent(Agent):
    """Agente Jarvis Core - Esecuzione e Controllo"""
    
    def __init__(self, jarvis_core=None):
        super().__init__("jarvis", "Jarvis Core")
        self.core = jarvis_core
    
    async def receive(self, message: AgentMessage):
        if message.type == MessageType.REQUEST:
            action = message.content.get("action")
            
            if action == "execute" and self.core:
                return await self.core.execute_recommendation(
                    message.content.get("recommendation", {})
                )
        
        return {"received": True}
