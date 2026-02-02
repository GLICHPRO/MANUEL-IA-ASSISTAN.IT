"""
🧠 Memory System - Memoria e Apprendimento

Sistema di memoria a due livelli:
- Short-Term Memory (STM): gestione comandi in tempo reale
- Long-Term Memory (LTM): apprendimento e strategie persistenti

Caratteristiche:
- Decay temporale per STM
- Consolidamento da STM a LTM
- Pattern recognition per apprendimento
- Aggiornamento strategie basato su risultati
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import uuid
import json
import hashlib
import math
import logging
from pathlib import Path


# Logger
memory_logger = logging.getLogger("memory_system")
memory_logger.setLevel(logging.DEBUG)


# === ENUMS ===

class MemoryType(Enum):
    """Tipo di memoria"""
    COMMAND = "command"           # Comando eseguito
    RESULT = "result"             # Risultato operazione
    CONTEXT = "context"           # Contesto situazionale
    PATTERN = "pattern"           # Pattern riconosciuto
    STRATEGY = "strategy"         # Strategia appresa
    PREFERENCE = "preference"     # Preferenza utente
    ERROR = "error"               # Errore/fallimento
    SUCCESS = "success"           # Successo
    INTERACTION = "interaction"   # Interazione utente


class MemoryPriority(Enum):
    """Priorità memoria"""
    CRITICAL = 5    # Mai dimenticare
    HIGH = 4        # Importante
    MEDIUM = 3      # Normale
    LOW = 2         # Poco importante
    EPHEMERAL = 1   # Temporaneo


class LearningType(Enum):
    """Tipo di apprendimento"""
    REINFORCEMENT = "reinforcement"   # Rinforzo positivo/negativo
    ASSOCIATION = "association"       # Associazione pattern
    CORRECTION = "correction"         # Correzione errore
    OPTIMIZATION = "optimization"     # Ottimizzazione strategia


# === DATA CLASSES ===

@dataclass
class MemoryItem:
    """Singolo elemento di memoria"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Contenuto
    content: Any = None
    memory_type: MemoryType = MemoryType.COMMAND
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Priorità e decadimento
    priority: MemoryPriority = MemoryPriority.MEDIUM
    strength: float = 1.0  # 0.0 - 1.0, decade nel tempo
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # TTL per STM (None = no decay)
    ttl_seconds: Optional[int] = None
    
    # Embedding per similarity search (opzionale)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "tags": self.tags,
            "priority": self.priority.value,
            "strength": round(self.strength, 3),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }
    
    def access(self):
        """Registra accesso alla memoria"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        # Rinforza memoria con accesso
        self.strength = min(1.0, self.strength + 0.1)
    
    def decay(self, factor: float = 0.95):
        """Applica decadimento alla memoria"""
        self.strength *= factor
    
    def is_expired(self) -> bool:
        """Verifica se memoria è scaduta"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def content_hash(self) -> str:
        """Hash del contenuto per deduplicazione"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.md5(content_str.encode()).hexdigest()[:12]


@dataclass
class LearnedPattern:
    """Pattern appreso dall'esperienza"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Pattern
    pattern_type: str = ""  # "command_sequence", "error_recovery", "optimization"
    trigger: Dict[str, Any] = field(default_factory=dict)  # Cosa attiva il pattern
    response: Dict[str, Any] = field(default_factory=dict)  # Risposta appresa
    
    # Statistiche
    occurrences: int = 0
    successes: int = 0
    failures: int = 0
    
    # Confidence
    confidence: float = 0.5
    
    # Timing
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def success_rate(self) -> float:
        """Calcola tasso di successo"""
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5
    
    def update_confidence(self):
        """Aggiorna confidence basata su statistiche"""
        rate = self.success_rate()
        # Bayesian update con prior
        n = self.occurrences
        prior = 0.5
        self.confidence = (prior + n * rate) / (1 + n)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "trigger": self.trigger,
            "response": self.response,
            "occurrences": self.occurrences,
            "successes": self.successes,
            "failures": self.failures,
            "confidence": round(self.confidence, 3),
            "success_rate": round(self.success_rate(), 3)
        }


@dataclass
class Strategy:
    """Strategia appresa per situazione specifica"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    name: str = ""
    description: str = ""
    
    # Condizioni di applicabilità
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Azioni da eseguire
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance
    usage_count: int = 0
    success_count: int = 0
    avg_execution_time: float = 0.0
    
    # Priorità e stato
    priority: int = 5
    is_active: bool = True
    
    # Versioning
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def effectiveness(self) -> float:
        """Calcola efficacia strategia"""
        if self.usage_count == 0:
            return 0.5
        return self.success_count / self.usage_count
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "conditions": self.conditions,
            "actions": self.actions,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "effectiveness": round(self.effectiveness(), 3),
            "priority": self.priority,
            "is_active": self.is_active,
            "version": self.version
        }


# === SHORT-TERM MEMORY ===

class ShortTermMemory:
    """
    Memoria a breve termine.
    Gestisce comandi e contesti in tempo reale.
    
    Caratteristiche:
    - Capacità limitata (sliding window)
    - Decay temporale automatico
    - Accesso rapido O(1)
    """
    
    def __init__(self, capacity: int = 100, decay_interval: int = 60):
        self.capacity = capacity
        self.decay_interval = decay_interval  # secondi
        
        # Storage
        self.memories: Dict[str, MemoryItem] = {}
        self.recent_queue: deque = deque(maxlen=capacity)
        
        # Indici per ricerca rapida
        self.by_type: Dict[MemoryType, List[str]] = {t: [] for t in MemoryType}
        self.by_tag: Dict[str, List[str]] = {}
        
        # Working memory (contesto attivo)
        self.working_context: Dict[str, Any] = {}
        
        # Timestamp ultimo decay
        self.last_decay = datetime.now()
        
        memory_logger.info(f"STM initialized: capacity={capacity}")
    
    def store(self, content: Any, 
             memory_type: MemoryType = MemoryType.COMMAND,
             tags: List[str] = None,
             priority: MemoryPriority = MemoryPriority.MEDIUM,
             ttl: int = 300,  # 5 minuti default
             context: Dict = None) -> MemoryItem:
        """
        Memorizza elemento nella STM.
        """
        # Applica decay se necessario
        self._check_decay()
        
        # Crea memoria
        memory = MemoryItem(
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            priority=priority,
            ttl_seconds=ttl,
            context=context or {}
        )
        
        # Evita duplicati recenti
        content_hash = memory.content_hash()
        for recent_id in list(self.recent_queue)[-10:]:
            if recent_id in self.memories:
                if self.memories[recent_id].content_hash() == content_hash:
                    # Aggiorna esistente invece di duplicare
                    self.memories[recent_id].access()
                    return self.memories[recent_id]
        
        # Gestisci overflow
        if len(self.memories) >= self.capacity:
            self._evict_oldest()
        
        # Salva
        self.memories[memory.id] = memory
        self.recent_queue.append(memory.id)
        
        # Aggiorna indici
        self.by_type[memory_type].append(memory.id)
        for tag in memory.tags:
            if tag not in self.by_tag:
                self.by_tag[tag] = []
            self.by_tag[tag].append(memory.id)
        
        memory_logger.debug(f"STM stored: {memory.id} ({memory_type.value})")
        
        return memory
    
    def recall(self, memory_id: str) -> Optional[MemoryItem]:
        """Recupera memoria per ID"""
        memory = self.memories.get(memory_id)
        if memory and not memory.is_expired():
            memory.access()
            return memory
        return None
    
    def recall_recent(self, n: int = 10, 
                     memory_type: MemoryType = None) -> List[MemoryItem]:
        """Recupera ultime N memorie"""
        self._check_decay()
        
        result = []
        for memory_id in reversed(list(self.recent_queue)):
            if len(result) >= n:
                break
            
            memory = self.memories.get(memory_id)
            if memory and not memory.is_expired():
                if memory_type is None or memory.memory_type == memory_type:
                    memory.access()
                    result.append(memory)
        
        return result
    
    def recall_by_type(self, memory_type: MemoryType, 
                      limit: int = 20) -> List[MemoryItem]:
        """Recupera memorie per tipo"""
        result = []
        for memory_id in reversed(self.by_type.get(memory_type, [])):
            if len(result) >= limit:
                break
            memory = self.memories.get(memory_id)
            if memory and not memory.is_expired():
                memory.access()
                result.append(memory)
        return result
    
    def recall_by_tag(self, tag: str, limit: int = 20) -> List[MemoryItem]:
        """Recupera memorie per tag"""
        result = []
        for memory_id in reversed(self.by_tag.get(tag, [])):
            if len(result) >= limit:
                break
            memory = self.memories.get(memory_id)
            if memory and not memory.is_expired():
                memory.access()
                result.append(memory)
        return result
    
    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Ricerca semplice nelle memorie"""
        query_lower = query.lower()
        results = []
        
        for memory in self.memories.values():
            if memory.is_expired():
                continue
            
            # Cerca nel contenuto
            content_str = str(memory.content).lower()
            if query_lower in content_str:
                results.append((memory, 1.0))
                continue
            
            # Cerca nei tag
            for tag in memory.tags:
                if query_lower in tag.lower():
                    results.append((memory, 0.8))
                    break
        
        # Ordina per rilevanza e strength
        results.sort(key=lambda x: (x[1], x[0].strength), reverse=True)
        
        return [m for m, _ in results[:limit]]
    
    def set_working_context(self, key: str, value: Any):
        """Imposta variabile nel contesto di lavoro"""
        self.working_context[key] = value
    
    def get_working_context(self, key: str, default: Any = None) -> Any:
        """Recupera variabile dal contesto di lavoro"""
        return self.working_context.get(key, default)
    
    def clear_working_context(self):
        """Pulisce contesto di lavoro"""
        self.working_context.clear()
    
    def _check_decay(self):
        """Applica decay periodico"""
        now = datetime.now()
        elapsed = (now - self.last_decay).total_seconds()
        
        if elapsed >= self.decay_interval:
            decay_factor = 0.95 ** (elapsed / self.decay_interval)
            
            expired = []
            for memory_id, memory in self.memories.items():
                if memory.is_expired():
                    expired.append(memory_id)
                elif memory.priority != MemoryPriority.CRITICAL:
                    memory.decay(decay_factor)
                    if memory.strength < 0.1:
                        expired.append(memory_id)
            
            # Rimuovi scadute
            for memory_id in expired:
                self._remove(memory_id)
            
            self.last_decay = now
            
            if expired:
                memory_logger.debug(f"STM decay: removed {len(expired)} items")
    
    def _evict_oldest(self):
        """Rimuove memoria più vecchia/debole"""
        if not self.memories:
            return
        
        # Trova candidato: bassa priorità + bassa strength + vecchio
        candidates = [
            (m, m.priority.value * 10 + m.strength + m.access_count * 0.1)
            for m in self.memories.values()
            if m.priority != MemoryPriority.CRITICAL
        ]
        
        if candidates:
            candidates.sort(key=lambda x: x[1])
            self._remove(candidates[0][0].id)
    
    def _remove(self, memory_id: str):
        """Rimuove memoria"""
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        
        # Rimuovi da indici
        if memory_id in self.by_type.get(memory.memory_type, []):
            self.by_type[memory.memory_type].remove(memory_id)
        
        for tag in memory.tags:
            if tag in self.by_tag and memory_id in self.by_tag[tag]:
                self.by_tag[tag].remove(memory_id)
        
        # Rimuovi da storage
        del self.memories[memory_id]
    
    def get_stats(self) -> Dict:
        """Statistiche STM"""
        type_counts = {t.value: len(ids) for t, ids in self.by_type.items()}
        
        strengths = [m.strength for m in self.memories.values()]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0
        
        return {
            "total_items": len(self.memories),
            "capacity": self.capacity,
            "utilization": round(len(self.memories) / self.capacity, 2),
            "by_type": type_counts,
            "average_strength": round(avg_strength, 3),
            "working_context_keys": list(self.working_context.keys())
        }


# === LONG-TERM MEMORY ===

class LongTermMemory:
    """
    Memoria a lungo termine.
    Apprendimento e strategie persistenti.
    
    Caratteristiche:
    - Persistenza su file
    - Pattern recognition
    - Strategy learning
    - No decay (ma prioritizzazione)
    """
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("data/memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage principale
        self.memories: Dict[str, MemoryItem] = {}
        self.patterns: Dict[str, LearnedPattern] = {}
        self.strategies: Dict[str, Strategy] = {}
        
        # Indici
        self.by_type: Dict[MemoryType, List[str]] = {t: [] for t in MemoryType}
        self.by_tag: Dict[str, List[str]] = {}
        
        # Carica da disco
        self._load()
        
        memory_logger.info(f"LTM initialized: {len(self.memories)} memories, "
                          f"{len(self.patterns)} patterns, {len(self.strategies)} strategies")
    
    def store(self, content: Any,
             memory_type: MemoryType = MemoryType.PATTERN,
             tags: List[str] = None,
             priority: MemoryPriority = MemoryPriority.MEDIUM,
             context: Dict = None) -> MemoryItem:
        """Memorizza nella LTM"""
        memory = MemoryItem(
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            priority=priority,
            context=context or {},
            ttl_seconds=None  # Nessun decay in LTM
        )
        
        # Salva
        self.memories[memory.id] = memory
        
        # Aggiorna indici
        self.by_type[memory_type].append(memory.id)
        for tag in memory.tags:
            if tag not in self.by_tag:
                self.by_tag[tag] = []
            self.by_tag[tag].append(memory.id)
        
        # Persist
        self._save()
        
        memory_logger.info(f"LTM stored: {memory.id} ({memory_type.value})")
        
        return memory
    
    def recall(self, memory_id: str) -> Optional[MemoryItem]:
        """Recupera memoria per ID"""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access()
        return memory
    
    def recall_by_type(self, memory_type: MemoryType,
                      limit: int = 50) -> List[MemoryItem]:
        """Recupera memorie per tipo"""
        result = []
        for memory_id in self.by_type.get(memory_type, []):
            if len(result) >= limit:
                break
            memory = self.memories.get(memory_id)
            if memory:
                result.append(memory)
        return result
    
    def search(self, query: str, limit: int = 20) -> List[MemoryItem]:
        """Ricerca nelle memorie"""
        query_lower = query.lower()
        results = []
        
        for memory in self.memories.values():
            content_str = str(memory.content).lower()
            score = 0
            
            if query_lower in content_str:
                score = 1.0
            else:
                # Partial match
                words = query_lower.split()
                matches = sum(1 for w in words if w in content_str)
                score = matches / len(words) if words else 0
            
            if score > 0.3:
                results.append((memory, score))
        
        results.sort(key=lambda x: (x[1], x[0].access_count), reverse=True)
        return [m for m, _ in results[:limit]]
    
    # === Pattern Management ===
    
    def learn_pattern(self, pattern_type: str,
                     trigger: Dict[str, Any],
                     response: Dict[str, Any],
                     success: bool = True) -> LearnedPattern:
        """Apprende o aggiorna un pattern"""
        # Cerca pattern simile esistente
        trigger_hash = hashlib.md5(
            json.dumps(trigger, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        existing = None
        for pattern in self.patterns.values():
            if pattern.pattern_type == pattern_type:
                existing_hash = hashlib.md5(
                    json.dumps(pattern.trigger, sort_keys=True).encode()
                ).hexdigest()[:12]
                if existing_hash == trigger_hash:
                    existing = pattern
                    break
        
        if existing:
            # Aggiorna esistente
            existing.occurrences += 1
            if success:
                existing.successes += 1
            else:
                existing.failures += 1
            existing.last_seen = datetime.now()
            existing.update_confidence()
            pattern = existing
        else:
            # Crea nuovo
            pattern = LearnedPattern(
                pattern_type=pattern_type,
                trigger=trigger,
                response=response,
                occurrences=1,
                successes=1 if success else 0,
                failures=0 if success else 1
            )
            pattern.update_confidence()
            self.patterns[pattern.id] = pattern
        
        self._save()
        
        memory_logger.info(f"Pattern learned: {pattern.id} ({pattern_type}) "
                          f"confidence={pattern.confidence:.2f}")
        
        return pattern
    
    def match_pattern(self, trigger: Dict[str, Any],
                     pattern_type: str = None) -> Optional[LearnedPattern]:
        """Trova pattern che corrisponde al trigger"""
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns.values():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            
            # Calcola match score
            score = self._pattern_match_score(trigger, pattern.trigger)
            
            # Pesa con confidence
            weighted_score = score * pattern.confidence
            
            if weighted_score > best_score and weighted_score > 0.5:
                best_score = weighted_score
                best_match = pattern
        
        return best_match
    
    def _pattern_match_score(self, query: Dict, pattern: Dict) -> float:
        """Calcola similarità tra trigger e pattern"""
        if not query or not pattern:
            return 0.0
        
        matches = 0
        total = len(pattern)
        
        for key, value in pattern.items():
            if key in query:
                if query[key] == value:
                    matches += 1
                elif isinstance(value, str) and isinstance(query[key], str):
                    if value.lower() in query[key].lower():
                        matches += 0.7
        
        return matches / total if total > 0 else 0.0
    
    def get_patterns(self, pattern_type: str = None,
                    min_confidence: float = 0.0) -> List[LearnedPattern]:
        """Recupera patterns"""
        patterns = []
        for pattern in self.patterns.values():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            if pattern.confidence >= min_confidence:
                patterns.append(pattern)
        
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns
    
    # === Strategy Management ===
    
    def create_strategy(self, name: str,
                       conditions: Dict[str, Any],
                       actions: List[Dict[str, Any]],
                       description: str = "",
                       priority: int = 5) -> Strategy:
        """Crea nuova strategia"""
        strategy = Strategy(
            name=name,
            description=description,
            conditions=conditions,
            actions=actions,
            priority=priority
        )
        
        self.strategies[strategy.id] = strategy
        self._save()
        
        memory_logger.info(f"Strategy created: {strategy.id} - {name}")
        
        return strategy
    
    def update_strategy(self, strategy_id: str,
                       success: bool,
                       execution_time: float = 0.0) -> Optional[Strategy]:
        """Aggiorna statistiche strategia dopo uso"""
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return None
        
        strategy.usage_count += 1
        if success:
            strategy.success_count += 1
        
        # Media mobile del tempo
        if execution_time > 0:
            n = strategy.usage_count
            strategy.avg_execution_time = (
                (strategy.avg_execution_time * (n - 1) + execution_time) / n
            )
        
        strategy.updated_at = datetime.now()
        strategy.version += 1
        
        self._save()
        
        return strategy
    
    def find_strategy(self, context: Dict[str, Any]) -> Optional[Strategy]:
        """Trova strategia applicabile al contesto"""
        candidates = []
        
        for strategy in self.strategies.values():
            if not strategy.is_active:
                continue
            
            # Verifica condizioni
            match_score = self._condition_match_score(context, strategy.conditions)
            
            if match_score > 0.6:
                # Pesa con efficacia
                score = match_score * strategy.effectiveness() * strategy.priority
                candidates.append((strategy, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _condition_match_score(self, context: Dict, conditions: Dict) -> float:
        """Calcola quanto il contesto soddisfa le condizioni"""
        if not conditions:
            return 1.0
        
        matches = 0
        for key, expected in conditions.items():
            if key in context:
                if context[key] == expected:
                    matches += 1
                elif isinstance(expected, list) and context[key] in expected:
                    matches += 1
        
        return matches / len(conditions)
    
    def get_strategies(self, active_only: bool = True) -> List[Strategy]:
        """Recupera strategie"""
        strategies = list(self.strategies.values())
        if active_only:
            strategies = [s for s in strategies if s.is_active]
        strategies.sort(key=lambda s: (s.priority, s.effectiveness()), reverse=True)
        return strategies
    
    # === Persistence ===
    
    def _save(self):
        """Salva su disco"""
        try:
            # Salva memories
            memories_data = {
                mid: {
                    "content": m.content,
                    "memory_type": m.memory_type.value,
                    "tags": m.tags,
                    "priority": m.priority.value,
                    "strength": m.strength,
                    "created_at": m.created_at.isoformat(),
                    "access_count": m.access_count,
                    "context": m.context
                }
                for mid, m in self.memories.items()
            }
            
            with open(self.storage_path / "memories.json", "w") as f:
                json.dump(memories_data, f, indent=2)
            
            # Salva patterns
            patterns_data = {
                pid: p.to_dict()
                for pid, p in self.patterns.items()
            }
            
            with open(self.storage_path / "patterns.json", "w") as f:
                json.dump(patterns_data, f, indent=2)
            
            # Salva strategies
            strategies_data = {
                sid: s.to_dict()
                for sid, s in self.strategies.items()
            }
            
            with open(self.storage_path / "strategies.json", "w") as f:
                json.dump(strategies_data, f, indent=2)
                
        except Exception as e:
            memory_logger.error(f"Failed to save LTM: {e}")
    
    def _load(self):
        """Carica da disco"""
        # Load memories
        memories_file = self.storage_path / "memories.json"
        if memories_file.exists():
            try:
                with open(memories_file) as f:
                    data = json.load(f)
                
                for mid, mdata in data.items():
                    memory = MemoryItem(
                        id=mid,
                        content=mdata["content"],
                        memory_type=MemoryType(mdata["memory_type"]),
                        tags=mdata.get("tags", []),
                        priority=MemoryPriority(mdata["priority"]),
                        strength=mdata.get("strength", 1.0),
                        context=mdata.get("context", {})
                    )
                    memory.created_at = datetime.fromisoformat(mdata["created_at"])
                    memory.access_count = mdata.get("access_count", 0)
                    
                    self.memories[mid] = memory
                    self.by_type[memory.memory_type].append(mid)
                    for tag in memory.tags:
                        if tag not in self.by_tag:
                            self.by_tag[tag] = []
                        self.by_tag[tag].append(mid)
                        
            except Exception as e:
                memory_logger.error(f"Failed to load memories: {e}")
        
        # Load patterns
        patterns_file = self.storage_path / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    data = json.load(f)
                
                for pid, pdata in data.items():
                    pattern = LearnedPattern(
                        id=pid,
                        pattern_type=pdata["pattern_type"],
                        trigger=pdata["trigger"],
                        response=pdata["response"],
                        occurrences=pdata.get("occurrences", 1),
                        successes=pdata.get("successes", 0),
                        failures=pdata.get("failures", 0),
                        confidence=pdata.get("confidence", 0.5)
                    )
                    self.patterns[pid] = pattern
                    
            except Exception as e:
                memory_logger.error(f"Failed to load patterns: {e}")
        
        # Load strategies
        strategies_file = self.storage_path / "strategies.json"
        if strategies_file.exists():
            try:
                with open(strategies_file) as f:
                    data = json.load(f)
                
                for sid, sdata in data.items():
                    strategy = Strategy(
                        id=sid,
                        name=sdata["name"],
                        description=sdata.get("description", ""),
                        conditions=sdata["conditions"],
                        actions=sdata["actions"],
                        usage_count=sdata.get("usage_count", 0),
                        success_count=sdata.get("success_count", 0),
                        priority=sdata.get("priority", 5),
                        is_active=sdata.get("is_active", True),
                        version=sdata.get("version", 1)
                    )
                    self.strategies[sid] = strategy
                    
            except Exception as e:
                memory_logger.error(f"Failed to load strategies: {e}")
    
    def get_stats(self) -> Dict:
        """Statistiche LTM"""
        return {
            "total_memories": len(self.memories),
            "total_patterns": len(self.patterns),
            "total_strategies": len(self.strategies),
            "active_strategies": sum(1 for s in self.strategies.values() if s.is_active),
            "memories_by_type": {t.value: len(ids) for t, ids in self.by_type.items()},
            "storage_path": str(self.storage_path)
        }


# === MEMORY SYSTEM (Combined) ===

class MemorySystem:
    """
    Sistema di memoria unificato.
    Coordina STM e LTM con consolidamento automatico.
    """
    
    def __init__(self, stm_capacity: int = 100, ltm_path: Path = None):
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.ltm = LongTermMemory(storage_path=ltm_path)
        
        # Configurazione consolidamento
        self.consolidation_threshold = 3  # Accessi per consolidare
        self.consolidation_interval = 300  # secondi
        self.last_consolidation = datetime.now()
        
        # Learning callbacks
        self.learning_hooks: List[Callable] = []
        
        memory_logger.info("MemorySystem initialized")
    
    # === STM Operations ===
    
    def remember(self, content: Any,
                memory_type: MemoryType = MemoryType.COMMAND,
                tags: List[str] = None,
                priority: MemoryPriority = MemoryPriority.MEDIUM,
                context: Dict = None) -> MemoryItem:
        """Memorizza nella STM"""
        return self.stm.store(
            content=content,
            memory_type=memory_type,
            tags=tags,
            priority=priority,
            context=context
        )
    
    def recall_recent(self, n: int = 10,
                     memory_type: MemoryType = None) -> List[MemoryItem]:
        """Recupera memorie recenti dalla STM"""
        return self.stm.recall_recent(n, memory_type)
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Recupera dal working context"""
        return self.stm.get_working_context(key, default)
    
    def set_context(self, key: str, value: Any):
        """Imposta nel working context"""
        self.stm.set_working_context(key, value)
    
    # === LTM Operations ===
    
    def store_permanent(self, content: Any,
                       memory_type: MemoryType = MemoryType.PATTERN,
                       tags: List[str] = None,
                       context: Dict = None) -> MemoryItem:
        """Memorizza permanentemente nella LTM"""
        return self.ltm.store(
            content=content,
            memory_type=memory_type,
            tags=tags,
            priority=MemoryPriority.HIGH,
            context=context
        )
    
    def search_all(self, query: str, limit: int = 20) -> List[MemoryItem]:
        """Cerca in STM e LTM"""
        stm_results = self.stm.search(query, limit // 2)
        ltm_results = self.ltm.search(query, limit // 2)
        
        # Unisci e ordina
        all_results = stm_results + ltm_results
        all_results.sort(key=lambda m: m.strength * (m.access_count + 1), reverse=True)
        
        return all_results[:limit]
    
    # === Learning ===
    
    def learn_from_result(self, command: Dict[str, Any],
                         result: Dict[str, Any],
                         success: bool):
        """
        Apprende dai risultati delle operazioni.
        Crea/aggiorna pattern basati su esperienza.
        """
        # Memorizza in STM
        memory_type = MemoryType.SUCCESS if success else MemoryType.ERROR
        self.remember(
            content={"command": command, "result": result, "success": success},
            memory_type=memory_type,
            tags=[command.get("type", "unknown")],
            priority=MemoryPriority.HIGH if not success else MemoryPriority.MEDIUM
        )
        
        # Apprendi pattern
        pattern = self.ltm.learn_pattern(
            pattern_type="command_result",
            trigger=command,
            response=result,
            success=success
        )
        
        # Notifica hooks
        for hook in self.learning_hooks:
            try:
                hook(command, result, success, pattern)
            except Exception as e:
                memory_logger.error(f"Learning hook error: {e}")
        
        # Check consolidamento
        self._check_consolidation()
        
        return pattern
    
    def learn_strategy(self, name: str,
                      conditions: Dict[str, Any],
                      actions: List[Dict[str, Any]],
                      from_experience: bool = False) -> Strategy:
        """Apprende nuova strategia"""
        strategy = self.ltm.create_strategy(
            name=name,
            conditions=conditions,
            actions=actions,
            description="Learned from experience" if from_experience else ""
        )
        
        memory_logger.info(f"New strategy learned: {name}")
        
        return strategy
    
    def suggest_strategy(self, context: Dict[str, Any]) -> Optional[Strategy]:
        """Suggerisce strategia basata su contesto"""
        return self.ltm.find_strategy(context)
    
    def record_strategy_result(self, strategy_id: str,
                              success: bool,
                              execution_time: float = 0.0):
        """Registra risultato uso strategia"""
        self.ltm.update_strategy(strategy_id, success, execution_time)
    
    # === Consolidation ===
    
    def _check_consolidation(self):
        """Controlla se consolidare memorie da STM a LTM"""
        now = datetime.now()
        elapsed = (now - self.last_consolidation).total_seconds()
        
        if elapsed < self.consolidation_interval:
            return
        
        self._consolidate()
        self.last_consolidation = now
    
    def _consolidate(self):
        """Consolida memorie importanti da STM a LTM"""
        consolidated = 0
        
        for memory in list(self.stm.memories.values()):
            # Criteri per consolidamento:
            # - Accessi frequenti
            # - Alta priorità
            # - Alta forza
            
            should_consolidate = (
                memory.access_count >= self.consolidation_threshold or
                memory.priority == MemoryPriority.CRITICAL or
                (memory.priority == MemoryPriority.HIGH and memory.strength > 0.7)
            )
            
            if should_consolidate:
                # Copia in LTM
                self.ltm.store(
                    content=memory.content,
                    memory_type=memory.memory_type,
                    tags=memory.tags,
                    priority=memory.priority,
                    context=memory.context
                )
                consolidated += 1
        
        if consolidated > 0:
            memory_logger.info(f"Consolidated {consolidated} memories to LTM")
    
    def force_consolidation(self):
        """Forza consolidamento immediato"""
        self._consolidate()
    
    # === Hooks ===
    
    def add_learning_hook(self, hook: Callable):
        """Aggiunge hook per learning events"""
        self.learning_hooks.append(hook)
    
    # === Stats ===
    
    def get_stats(self) -> Dict:
        """Statistiche complete sistema memoria"""
        return {
            "stm": self.stm.get_stats(),
            "ltm": self.ltm.get_stats(),
            "consolidation_threshold": self.consolidation_threshold,
            "last_consolidation": self.last_consolidation.isoformat()
        }
    
    def get_memory_summary(self) -> str:
        """Riassunto testuale della memoria"""
        stm_stats = self.stm.get_stats()
        ltm_stats = self.ltm.get_stats()
        
        return (
            f"🧠 Memory System Status\n"
            f"  STM: {stm_stats['total_items']}/{stm_stats['capacity']} items "
            f"(avg strength: {stm_stats['average_strength']:.2f})\n"
            f"  LTM: {ltm_stats['total_memories']} memories, "
            f"{ltm_stats['total_patterns']} patterns, "
            f"{ltm_stats['active_strategies']} strategies"
        )
