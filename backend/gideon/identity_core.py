# /backend/gideon/identity_core.py
"""
🔮 GIDEON 3.0 - Identity Core
Mantiene coerenza, personalità e memoria contestuale del sistema.
NON esegue azioni - gestisce solo identità e contesto.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import hashlib
import logging

logger = logging.getLogger(__name__)


class PersonalityTrait(Enum):
    """Tratti di personalità del sistema"""
    ANALYTICAL = "analytical"       # Analitico
    CAUTIOUS = "cautious"          # Prudente
    PROACTIVE = "proactive"        # Proattivo
    HELPFUL = "helpful"            # Collaborativo
    PRECISE = "precise"            # Preciso
    ADAPTIVE = "adaptive"          # Adattivo
    TRANSPARENT = "transparent"    # Trasparente


class CommunicationStyle(Enum):
    """Stile di comunicazione"""
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    SIMPLE = "simple"
    VERBOSE = "verbose"
    CONCISE = "concise"


class ContextType(Enum):
    """Tipi di contesto"""
    SESSION = "session"           # Contesto sessione
    CONVERSATION = "conversation" # Contesto conversazione
    USER = "user"                 # Contesto utente
    TASK = "task"                 # Contesto task
    DOMAIN = "domain"             # Contesto dominio
    SYSTEM = "system"             # Contesto sistema


class MemoryType(Enum):
    """Tipi di memoria"""
    WORKING = "working"           # Memoria di lavoro (breve)
    SHORT_TERM = "short_term"     # Breve termine
    LONG_TERM = "long_term"       # Lungo termine
    EPISODIC = "episodic"         # Episodica (eventi)
    SEMANTIC = "semantic"         # Semantica (conoscenza)
    PROCEDURAL = "procedural"     # Procedurale (come fare)


@dataclass
class ContextualMemory:
    """Unità di memoria contestuale"""
    id: str
    memory_type: MemoryType
    content: Dict
    
    # Relevance
    relevance_score: float = 1.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
    # Associations
    tags: List[str] = field(default_factory=list)
    related_memories: List[str] = field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_persistent: bool = False
    
    def decay_relevance(self, decay_rate: float = 0.1):
        """Applica decay alla rilevanza"""
        hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        self.relevance_score *= (1 - decay_rate * hours_since_access)
        self.relevance_score = max(0.01, self.relevance_score)
    
    def access(self):
        """Registra accesso e rinforza memoria"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        self.relevance_score = min(1.0, self.relevance_score * 1.1)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.memory_type.value,
            "content": self.content,
            "relevance": round(self.relevance_score, 3),
            "access_count": self.access_count,
            "tags": self.tags,
            "is_persistent": self.is_persistent
        }


@dataclass
class Context:
    """Contesto attivo"""
    context_type: ContextType
    data: Dict = field(default_factory=dict)
    
    # Stack for nested contexts
    parent_context: Optional['Context'] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update(self, key: str, value: Any):
        """Aggiorna dato nel contesto"""
        self.data[key] = value
        self.updated_at = datetime.now()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Ottiene dato dal contesto"""
        return self.data.get(key, default)
    
    def merge(self, other: 'Context'):
        """Merge con altro contesto"""
        self.data.update(other.data)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> dict:
        return {
            "type": self.context_type.value,
            "data": self.data,
            "has_parent": self.parent_context is not None
        }


@dataclass
class IdentityProfile:
    """Profilo identità del sistema"""
    name: str = "Gideon"
    version: str = "3.0"
    role: str = "AI Predittivo e Analitico"
    
    # Personality
    personality_traits: List[PersonalityTrait] = field(default_factory=lambda: [
        PersonalityTrait.ANALYTICAL,
        PersonalityTrait.HELPFUL,
        PersonalityTrait.TRANSPARENT
    ])
    
    communication_style: CommunicationStyle = CommunicationStyle.TECHNICAL
    
    # Values and principles
    core_values: List[str] = field(default_factory=lambda: [
        "Accuratezza nelle previsioni",
        "Trasparenza nei limiti",
        "Supporto alle decisioni",
        "Non esecuzione diretta"
    ])
    
    # Capabilities
    capabilities: List[str] = field(default_factory=lambda: [
        "Analisi predittiva",
        "Simulazione scenari",
        "Valutazione rischi",
        "Ragionamento temporale",
        "Meta-cognizione"
    ])
    
    # Limitations
    known_limitations: List[str] = field(default_factory=lambda: [
        "Non esegue azioni dirette",
        "Conoscenza limitata al training",
        "Incertezza in domini nuovi"
    ])
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "role": self.role,
            "personality": [t.value for t in self.personality_traits],
            "communication_style": self.communication_style.value,
            "core_values": self.core_values,
            "capabilities": self.capabilities,
            "limitations": self.known_limitations
        }


@dataclass
class ConsistencyCheck:
    """Risultato verifica coerenza"""
    is_consistent: bool
    inconsistencies: List[str] = field(default_factory=list)
    confidence: float = 1.0
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "is_consistent": self.is_consistent,
            "inconsistencies": self.inconsistencies,
            "confidence": round(self.confidence, 2),
            "suggestions": self.suggestions
        }


class IdentityCore:
    """
    Core identità per Gideon.
    Gestisce personalità, memoria contestuale e coerenza.
    """
    
    def __init__(self):
        # Identity profile
        self.profile = IdentityProfile()
        
        # Memory systems
        self.working_memory: deque = deque(maxlen=20)
        self.short_term_memory: Dict[str, ContextualMemory] = {}
        self.long_term_memory: Dict[str, ContextualMemory] = {}
        self.episodic_memory: List[ContextualMemory] = []
        
        # Context management
        self.current_context: Optional[Context] = None
        self.context_stack: List[Context] = []
        self.context_history: List[Context] = []
        
        # Session tracking
        self.session_id: str = self._generate_session_id()
        self.session_start: datetime = datetime.now()
        self.interactions_count: int = 0
        
        # User modeling
        self.user_preferences: Dict = {}
        self.user_history: List[Dict] = []
        
        # Consistency tracking
        self.consistency_violations: List[Dict] = []
        
        # Counters
        self._memory_counter = 0
    
    def _generate_session_id(self) -> str:
        """Genera ID sessione unico"""
        data = f"{datetime.now().isoformat()}_{id(self)}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    # === Identity Management ===
    
    def get_identity(self) -> Dict:
        """Restituisce identità completa"""
        return {
            "profile": self.profile.to_dict(),
            "session": {
                "id": self.session_id,
                "started": self.session_start.isoformat(),
                "interactions": self.interactions_count,
                "duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60
            },
            "current_context": self.current_context.to_dict() if self.current_context else None
        }
    
    def introduce(self) -> str:
        """Genera auto-presentazione"""
        traits = ", ".join(t.value for t in self.profile.personality_traits)
        return (
            f"Sono {self.profile.name} {self.profile.version}, {self.profile.role}. "
            f"I miei tratti distintivi sono: {traits}. "
            f"Ricorda: non eseguo azioni direttamente, fornisco solo analisi e raccomandazioni."
        )
    
    def update_personality(self, trait: PersonalityTrait, add: bool = True):
        """Aggiorna tratti personalità"""
        if add and trait not in self.profile.personality_traits:
            self.profile.personality_traits.append(trait)
        elif not add and trait in self.profile.personality_traits:
            self.profile.personality_traits.remove(trait)
    
    def set_communication_style(self, style: CommunicationStyle):
        """Imposta stile comunicazione"""
        self.profile.communication_style = style
    
    # === Memory Management ===
    
    def remember(self, content: Dict, memory_type: MemoryType = MemoryType.SHORT_TERM,
                 tags: List[str] = None, persistent: bool = False) -> ContextualMemory:
        """
        Memorizza informazione.
        """
        self._memory_counter += 1
        memory_id = f"mem_{self._memory_counter}"
        
        memory = ContextualMemory(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            tags=tags or [],
            is_persistent=persistent
        )
        
        # Store based on type
        if memory_type == MemoryType.WORKING:
            self.working_memory.append(memory)
        elif memory_type == MemoryType.SHORT_TERM:
            self.short_term_memory[memory_id] = memory
        elif memory_type == MemoryType.LONG_TERM:
            self.long_term_memory[memory_id] = memory
        elif memory_type == MemoryType.EPISODIC:
            self.episodic_memory.append(memory)
        
        return memory
    
    def recall(self, query: Dict = None, memory_type: MemoryType = None,
               tags: List[str] = None, limit: int = 10) -> List[ContextualMemory]:
        """
        Recupera memorie rilevanti.
        """
        memories = []
        
        # Collect from relevant stores
        if memory_type is None or memory_type == MemoryType.WORKING:
            memories.extend(list(self.working_memory))
        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            memories.extend(self.short_term_memory.values())
        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            memories.extend(self.long_term_memory.values())
        if memory_type is None or memory_type == MemoryType.EPISODIC:
            memories.extend(self.episodic_memory)
        
        # Filter by tags
        if tags:
            memories = [m for m in memories if any(t in m.tags for t in tags)]
        
        # Filter by query content
        if query:
            relevant = []
            for mem in memories:
                score = self._calculate_relevance(mem, query)
                if score > 0.3:
                    mem.relevance_score = score
                    relevant.append(mem)
            memories = relevant
        
        # Sort by relevance and recency
        memories.sort(key=lambda m: (m.relevance_score, -m.access_count), reverse=True)
        
        # Access memories (reinforcement)
        for mem in memories[:limit]:
            mem.access()
        
        return memories[:limit]
    
    def _calculate_relevance(self, memory: ContextualMemory, query: Dict) -> float:
        """Calcola rilevanza di una memoria rispetto a query"""
        score = 0.0
        
        # Tag matching
        query_tags = query.get("tags", [])
        if query_tags:
            matches = len(set(memory.tags) & set(query_tags))
            score += matches / len(query_tags) * 0.4
        
        # Content key matching
        query_keys = set(query.keys())
        memory_keys = set(memory.content.keys())
        if query_keys and memory_keys:
            key_overlap = len(query_keys & memory_keys) / len(query_keys)
            score += key_overlap * 0.3
        
        # Recency bonus
        hours_ago = (datetime.now() - memory.last_accessed).total_seconds() / 3600
        recency = max(0, 1 - hours_ago / 24)
        score += recency * 0.2
        
        # Access frequency bonus
        score += min(0.1, memory.access_count * 0.01)
        
        return min(1.0, score)
    
    def forget(self, memory_id: str) -> bool:
        """Rimuove memoria specifica"""
        if memory_id in self.short_term_memory:
            del self.short_term_memory[memory_id]
            return True
        if memory_id in self.long_term_memory:
            del self.long_term_memory[memory_id]
            return True
        return False
    
    def consolidate_memory(self):
        """
        Consolida memoria: sposta importante da short a long term.
        """
        to_consolidate = []
        
        for mem_id, memory in list(self.short_term_memory.items()):
            # Criteria for consolidation
            if (memory.access_count >= 3 or 
                memory.is_persistent or
                memory.relevance_score > 0.7):
                to_consolidate.append(mem_id)
        
        for mem_id in to_consolidate:
            memory = self.short_term_memory.pop(mem_id)
            memory.memory_type = MemoryType.LONG_TERM
            self.long_term_memory[mem_id] = memory
        
        # Clean up low-relevance short-term
        to_remove = []
        for mem_id, memory in self.short_term_memory.items():
            memory.decay_relevance()
            if memory.relevance_score < 0.1:
                to_remove.append(mem_id)
        
        for mem_id in to_remove:
            del self.short_term_memory[mem_id]
        
        return {
            "consolidated": len(to_consolidate),
            "removed": len(to_remove),
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory)
        }
    
    # === Context Management ===
    
    def push_context(self, context_type: ContextType,
                     data: Dict = None) -> Context:
        """
        Crea e attiva nuovo contesto.
        """
        new_context = Context(
            context_type=context_type,
            data=data or {},
            parent_context=self.current_context
        )
        
        if self.current_context:
            self.context_stack.append(self.current_context)
        
        self.current_context = new_context
        return new_context
    
    def pop_context(self) -> Optional[Context]:
        """
        Rimuove contesto attuale e torna al precedente.
        """
        if not self.current_context:
            return None
        
        old_context = self.current_context
        self.context_history.append(old_context)
        
        if self.context_stack:
            self.current_context = self.context_stack.pop()
        else:
            self.current_context = None
        
        return old_context
    
    def update_context(self, key: str, value: Any):
        """Aggiorna contesto corrente"""
        if self.current_context:
            self.current_context.update(key, value)
    
    def get_context(self, key: str = None) -> Any:
        """Ottiene valore dal contesto o intero contesto"""
        if not self.current_context:
            return None
        
        if key:
            return self.current_context.get(key)
        return self.current_context.data
    
    def get_full_context(self) -> Dict:
        """Ottiene contesto completo includendo parent contexts"""
        if not self.current_context:
            return {}
        
        full_context = {}
        context = self.current_context
        
        while context:
            # Parent context data first (can be overridden)
            for key, value in context.data.items():
                if key not in full_context:
                    full_context[key] = value
            context = context.parent_context
        
        return full_context
    
    # === Consistency Checking ===
    
    def check_consistency(self, statement: Dict) -> ConsistencyCheck:
        """
        Verifica coerenza di una affermazione con identità e storia.
        """
        inconsistencies = []
        suggestions = []
        confidence = 1.0
        
        # Check against core values
        action = statement.get("action", "")
        if action and "execute" in action.lower():
            inconsistencies.append(
                "Azione di esecuzione diretta viola principio 'Non esecuzione diretta'"
            )
            suggestions.append("Delegare esecuzione a Jarvis")
            confidence -= 0.3
        
        # Check against capabilities
        domain = statement.get("domain", "")
        if domain and domain not in self.profile.capabilities:
            if domain not in [c.lower() for c in self.profile.capabilities]:
                inconsistencies.append(f"Dominio '{domain}' fuori dalle capability note")
                suggestions.append("Indicare incertezza o limitazione")
                confidence -= 0.2
        
        # Check against previous statements
        relevant_memories = self.recall(
            query={"tags": [statement.get("topic", "")]},
            limit=5
        )
        
        for memory in relevant_memories:
            prev_statement = memory.content.get("statement", {})
            if self._contradicts(statement, prev_statement):
                inconsistencies.append(
                    f"Contraddizione con memoria precedente: {memory.id}"
                )
                suggestions.append("Riconoscere evoluzione posizione o correggere")
                confidence -= 0.15
        
        is_consistent = len(inconsistencies) == 0
        
        # Log violation
        if not is_consistent:
            self.consistency_violations.append({
                "statement": statement,
                "inconsistencies": inconsistencies,
                "timestamp": datetime.now().isoformat()
            })
        
        return ConsistencyCheck(
            is_consistent=is_consistent,
            inconsistencies=inconsistencies,
            confidence=max(0.3, confidence),
            suggestions=suggestions
        )
    
    def _contradicts(self, statement1: Dict, statement2: Dict) -> bool:
        """Verifica se due affermazioni si contraddicono"""
        # Simple contradiction check
        topic1 = statement1.get("topic")
        topic2 = statement2.get("topic")
        
        if topic1 != topic2:
            return False
        
        position1 = statement1.get("position")
        position2 = statement2.get("position")
        
        if position1 and position2:
            # Direct opposition check
            opposites = [
                ("positive", "negative"),
                ("yes", "no"),
                ("true", "false"),
                ("agree", "disagree")
            ]
            
            for a, b in opposites:
                if (a in str(position1).lower() and b in str(position2).lower()) or \
                   (b in str(position1).lower() and a in str(position2).lower()):
                    return True
        
        return False
    
    # === User Interaction ===
    
    def record_interaction(self, interaction: Dict):
        """Registra interazione utente"""
        self.interactions_count += 1
        
        # Store in working memory
        self.remember(
            content=interaction,
            memory_type=MemoryType.WORKING,
            tags=["interaction", interaction.get("type", "unknown")]
        )
        
        # Update user preferences if relevant
        if interaction.get("preference"):
            pref_key = interaction["preference"]["key"]
            pref_value = interaction["preference"]["value"]
            self.user_preferences[pref_key] = pref_value
        
        # Store in history
        self.user_history.append({
            "interaction": interaction,
            "timestamp": datetime.now().isoformat(),
            "session": self.session_id
        })
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Ottiene preferenza utente"""
        return self.user_preferences.get(key, default)
    
    # === Adaptive Behavior ===
    
    def adapt_response(self, response: str, context: Dict = None) -> str:
        """
        Adatta risposta in base a stile comunicazione e contesto.
        """
        style = self.profile.communication_style
        
        if style == CommunicationStyle.CONCISE:
            # Shorten response
            sentences = response.split(". ")
            if len(sentences) > 3:
                response = ". ".join(sentences[:3]) + "."
        
        elif style == CommunicationStyle.TECHNICAL:
            # Keep technical terms
            pass
        
        elif style == CommunicationStyle.SIMPLE:
            # Could simplify technical terms (placeholder)
            pass
        
        elif style == CommunicationStyle.FORMAL:
            # Add formal language markers
            response = response.replace("ok", "d'accordo")
        
        return response
    
    # === Status ===
    
    def get_status(self) -> Dict:
        """Stato dell'Identity Core"""
        return {
            "session_id": self.session_id,
            "session_duration_minutes": round(
                (datetime.now() - self.session_start).total_seconds() / 60, 1
            ),
            "interactions": self.interactions_count,
            "working_memory_size": len(self.working_memory),
            "short_term_memories": len(self.short_term_memory),
            "long_term_memories": len(self.long_term_memory),
            "episodic_memories": len(self.episodic_memory),
            "context_depth": len(self.context_stack) + (1 if self.current_context else 0),
            "consistency_violations": len(self.consistency_violations),
            "communication_style": self.profile.communication_style.value
        }
