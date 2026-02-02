"""
⚡ JARVIS - Priority Manager

Assegna urgenza e importanza alle azioni:
- Valuta priorità in base a contesto, deadline, impatto
- Gestisce coda di azioni con prioritizzazione dinamica
- Supporta interruzioni per emergenze
- Bilanciamento carico di lavoro

Matrice Eisenhower:
┌──────────────────────────────────────────────────┐
│              URGENTE │ NON URGENTE               │
├──────────────────────┼───────────────────────────┤
│ IMPORTANTE │ DO NOW  │ SCHEDULE                  │
│            │ (P1)    │ (P2)                      │
├────────────┼─────────┼───────────────────────────┤
│ NON IMPORT │ DELEGATE│ ELIMINATE/DEFER           │
│            │ (P3)    │ (P4)                      │
└──────────────────────┴───────────────────────────┘
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import heapq
import asyncio
import uuid


class Priority(Enum):
    """Livelli di priorità"""
    CRITICAL = 1    # Emergenze, sicurezza
    HIGH = 2        # Azioni urgenti e importanti
    NORMAL = 3      # Azioni standard
    LOW = 4         # Background tasks
    DEFERRED = 5    # Eseguibili quando idle


class Urgency(Enum):
    """Livelli di urgenza temporale"""
    IMMEDIATE = 1      # Ora
    VERY_URGENT = 2    # Entro 1 minuto
    URGENT = 3         # Entro 5 minuti
    SOON = 4           # Entro 30 minuti
    NORMAL = 5         # Entro 1 ora
    LOW = 6            # Entro oggi
    WHENEVER = 7       # Quando possibile


class Importance(Enum):
    """Livelli di importanza"""
    CRITICAL = 1    # Impatto sistema/sicurezza
    HIGH = 2        # Richiesta diretta utente
    MEDIUM = 3      # Task automatici importanti
    LOW = 4         # Background/maintenance
    MINIMAL = 5     # Nice-to-have


@dataclass(order=True)
class PrioritizedTask:
    """Task con priorità per la coda"""
    priority_score: float = field(compare=True)
    task_id: str = field(compare=False)
    action: Dict = field(compare=False)
    priority: Priority = field(compare=False)
    urgency: Urgency = field(compare=False)
    importance: Importance = field(compare=False)
    created_at: datetime = field(compare=False, default_factory=datetime.now)
    deadline: Optional[datetime] = field(compare=False, default=None)
    source: str = field(compare=False, default="user")  # user, system, automation
    metadata: Dict = field(compare=False, default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "action": self.action,
            "priority": self.priority.name,
            "urgency": self.urgency.name,
            "importance": self.importance.name,
            "priority_score": self.priority_score,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "source": self.source,
            "metadata": self.metadata
        }


class PriorityManager:
    """
    ⚡ Gestisce priorità e scheduling delle azioni
    
    Features:
    - Calcolo priorità multi-fattore
    - Coda prioritizzata con heap
    - Aging: le task vecchie aumentano priorità
    - Preemption per emergenze
    - Deadline tracking
    - Load balancing
    """
    
    def __init__(self):
        # Coda prioritizzata (min-heap, score basso = priorità alta)
        self._queue: List[PrioritizedTask] = []
        self._task_map: Dict[str, PrioritizedTask] = {}
        
        # Task in esecuzione
        self._running_tasks: Dict[str, PrioritizedTask] = {}
        self._max_concurrent = 3
        
        # Configurazione priorità
        self._priority_config = {
            # Pesi per calcolo score
            "weights": {
                "priority": 0.35,
                "urgency": 0.30,
                "importance": 0.25,
                "aging": 0.10
            },
            # Aging: incremento priorità per minuto di attesa
            "aging_rate": 0.01,
            # Deadline boost: moltiplicatore quando vicino deadline
            "deadline_boost": {
                "imminent": 2.0,    # < 1 min
                "close": 1.5,       # < 5 min
                "near": 1.2         # < 30 min
            }
        }
        
        # Mappatura intent → priorità default
        self._intent_priorities = {
            # CRITICAL
            "emergency_stop": Priority.CRITICAL,
            "shutdown": Priority.HIGH,
            "security_alert": Priority.CRITICAL,
            
            # HIGH - Comandi diretti utente
            "open_app": Priority.HIGH,
            "close_app": Priority.HIGH,
            "search_web": Priority.HIGH,
            "open_url": Priority.HIGH,
            
            # NORMAL - Operazioni standard
            "time": Priority.NORMAL,
            "date": Priority.NORMAL,
            "weather": Priority.NORMAL,
            "calculate": Priority.NORMAL,
            "greeting": Priority.NORMAL,
            
            # LOW - Background
            "run_routine": Priority.LOW,
            "create_routine": Priority.LOW,
            "sync_data": Priority.LOW,
            
            # DEFERRED
            "cleanup": Priority.DEFERRED,
            "optimize": Priority.DEFERRED
        }
        
        # Mappatura category → importanza default
        self._category_importance = {
            "system": Importance.HIGH,
            "security": Importance.CRITICAL,
            "app": Importance.MEDIUM,
            "web": Importance.MEDIUM,
            "info": Importance.LOW,
            "automation": Importance.MEDIUM,
            "conversation": Importance.LOW,
            "media": Importance.LOW
        }
        
        # Statistiche
        self._stats = {
            "total_queued": 0,
            "total_completed": 0,
            "total_cancelled": 0,
            "average_wait_time": 0.0,
            "priority_distribution": {p.name: 0 for p in Priority}
        }
        
        # Callbacks
        self._on_task_ready: Optional[Callable] = None
        self._on_preemption: Optional[Callable] = None
    
    # ========== PRIORITY CALCULATION ==========
    
    def calculate_priority(self, action: Dict, context: Dict = None) -> PrioritizedTask:
        """
        Calcola priorità per un'azione
        
        Args:
            action: Azione da prioritizzare
            context: Contesto (intent, source, deadline, etc.)
            
        Returns:
            PrioritizedTask con score calcolato
        """
        context = context or {}
        
        # Determina componenti priorità
        intent_name = context.get("intent", {}).get("name", "unknown")
        category = context.get("intent", {}).get("category", "unknown")
        source = context.get("source", "user")
        deadline = context.get("deadline")
        
        # Priority base da intent
        priority = self._intent_priorities.get(intent_name, Priority.NORMAL)
        
        # Override se specificato esplicitamente
        if "priority" in context:
            priority = Priority[context["priority"].upper()]
        
        # Importanza da categoria
        importance = self._category_importance.get(category, Importance.MEDIUM)
        
        # Urgenza da deadline o default
        urgency = self._calculate_urgency(deadline, context)
        
        # Boost per richiesta diretta utente
        if source == "user":
            if priority.value > Priority.HIGH.value:
                priority = Priority.HIGH
        
        # Crea task
        task = PrioritizedTask(
            priority_score=0.0,  # Calcolato dopo
            task_id=str(uuid.uuid4())[:8],
            action=action,
            priority=priority,
            urgency=urgency,
            importance=importance,
            deadline=datetime.fromisoformat(deadline) if isinstance(deadline, str) else deadline,
            source=source,
            metadata=context.get("metadata", {})
        )
        
        # Calcola score finale
        task.priority_score = self._compute_score(task)
        
        return task
    
    def _calculate_urgency(self, deadline: Optional[datetime], context: Dict) -> Urgency:
        """Calcola urgenza basata su deadline e contesto"""
        
        # Se deadline specificato
        if deadline:
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline)
            
            time_left = (deadline - datetime.now()).total_seconds()
            
            if time_left <= 0:
                return Urgency.IMMEDIATE
            elif time_left <= 60:
                return Urgency.VERY_URGENT
            elif time_left <= 300:
                return Urgency.URGENT
            elif time_left <= 1800:
                return Urgency.SOON
            elif time_left <= 3600:
                return Urgency.NORMAL
            else:
                return Urgency.LOW
        
        # Default basato su source
        source = context.get("source", "user")
        if source == "user":
            return Urgency.URGENT  # Utente si aspetta risposta rapida
        elif source == "automation":
            return Urgency.NORMAL
        else:
            return Urgency.WHENEVER
    
    def _compute_score(self, task: PrioritizedTask) -> float:
        """
        Calcola score numerico per ordinamento
        Score più basso = priorità più alta
        """
        weights = self._priority_config["weights"]
        
        # Componenti normalizzati (1-5 range → 0-1)
        priority_score = (task.priority.value - 1) / 4
        urgency_score = (task.urgency.value - 1) / 6
        importance_score = (task.importance.value - 1) / 4
        
        # Aging factor (aumenta con tempo in coda)
        age_minutes = (datetime.now() - task.created_at).total_seconds() / 60
        aging_factor = max(0, 1 - (age_minutes * self._priority_config["aging_rate"]))
        
        # Score base
        score = (
            weights["priority"] * priority_score +
            weights["urgency"] * urgency_score +
            weights["importance"] * importance_score +
            weights["aging"] * aging_factor
        )
        
        # Deadline boost
        if task.deadline:
            time_left = (task.deadline - datetime.now()).total_seconds()
            boosts = self._priority_config["deadline_boost"]
            
            if time_left <= 60:
                score /= boosts["imminent"]
            elif time_left <= 300:
                score /= boosts["close"]
            elif time_left <= 1800:
                score /= boosts["near"]
        
        return score
    
    # ========== QUEUE MANAGEMENT ==========
    
    def enqueue(self, action: Dict, context: Dict = None) -> PrioritizedTask:
        """
        Aggiunge task alla coda prioritizzata
        
        Returns:
            Task creato con ID
        """
        task = self.calculate_priority(action, context)
        
        heapq.heappush(self._queue, task)
        self._task_map[task.task_id] = task
        
        # Stats
        self._stats["total_queued"] += 1
        self._stats["priority_distribution"][task.priority.name] += 1
        
        # Check preemption
        if task.priority == Priority.CRITICAL:
            self._handle_preemption(task)
        
        # Notify
        if self._on_task_ready:
            asyncio.create_task(self._notify_task_ready(task))
        
        return task
    
    def dequeue(self) -> Optional[PrioritizedTask]:
        """
        Rimuove e ritorna task con priorità più alta
        """
        if not self._queue:
            return None
        
        # Re-score per aging prima di dequeue
        self._refresh_scores()
        
        task = heapq.heappop(self._queue)
        del self._task_map[task.task_id]
        
        return task
    
    def peek(self) -> Optional[PrioritizedTask]:
        """Guarda task con priorità più alta senza rimuoverlo"""
        if not self._queue:
            return None
        return self._queue[0]
    
    def get_task(self, task_id: str) -> Optional[PrioritizedTask]:
        """Ottiene task per ID"""
        return self._task_map.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancella task dalla coda"""
        if task_id not in self._task_map:
            return False
        
        task = self._task_map[task_id]
        self._queue.remove(task)
        heapq.heapify(self._queue)
        del self._task_map[task_id]
        
        self._stats["total_cancelled"] += 1
        
        return True
    
    def update_priority(self, task_id: str, new_priority: Priority) -> bool:
        """Aggiorna priorità di un task in coda"""
        if task_id not in self._task_map:
            return False
        
        task = self._task_map[task_id]
        
        # Rimuovi dalla coda
        self._queue.remove(task)
        
        # Aggiorna priorità e score
        task.priority = new_priority
        task.priority_score = self._compute_score(task)
        
        # Re-inserisci
        heapq.heappush(self._queue, task)
        
        return True
    
    def _refresh_scores(self):
        """Ricalcola score per aging"""
        for task in self._queue:
            task.priority_score = self._compute_score(task)
        heapq.heapify(self._queue)
    
    def _handle_preemption(self, critical_task: PrioritizedTask):
        """Gestisce preemption per task critici"""
        if self._on_preemption and self._running_tasks:
            # Trova task meno prioritario in esecuzione
            lowest_priority_task = max(
                self._running_tasks.values(),
                key=lambda t: t.priority.value
            )
            
            if lowest_priority_task.priority.value > critical_task.priority.value:
                asyncio.create_task(
                    self._on_preemption(lowest_priority_task, critical_task)
                )
    
    async def _notify_task_ready(self, task: PrioritizedTask):
        """Notifica che un task è pronto"""
        if self._on_task_ready:
            await self._on_task_ready(task)
    
    # ========== BATCH OPERATIONS ==========
    
    def get_next_batch(self, count: int = 5) -> List[PrioritizedTask]:
        """
        Ottiene prossimi N task da eseguire
        Non li rimuove dalla coda
        """
        self._refresh_scores()
        return sorted(self._queue, key=lambda t: t.priority_score)[:count]
    
    def get_by_priority(self, priority: Priority) -> List[PrioritizedTask]:
        """Ottiene tutti i task di una certa priorità"""
        return [t for t in self._queue if t.priority == priority]
    
    def get_by_source(self, source: str) -> List[PrioritizedTask]:
        """Ottiene task per source"""
        return [t for t in self._queue if t.source == source]
    
    def get_overdue(self) -> List[PrioritizedTask]:
        """Ottiene task con deadline scaduta"""
        now = datetime.now()
        return [t for t in self._queue if t.deadline and t.deadline < now]
    
    def get_urgent(self) -> List[PrioritizedTask]:
        """Ottiene task urgenti"""
        return [t for t in self._queue 
                if t.urgency.value <= Urgency.URGENT.value]
    
    # ========== EISENHOWER MATRIX ==========
    
    def get_eisenhower_matrix(self) -> Dict[str, List[PrioritizedTask]]:
        """
        Classifica task secondo matrice Eisenhower
        
        Returns:
            Dict con quadranti: do_now, schedule, delegate, eliminate
        """
        matrix = {
            "do_now": [],      # Urgente + Importante
            "schedule": [],    # Non urgente + Importante
            "delegate": [],    # Urgente + Non importante
            "eliminate": []    # Non urgente + Non importante
        }
        
        for task in self._queue:
            is_urgent = task.urgency.value <= Urgency.URGENT.value
            is_important = task.importance.value <= Importance.MEDIUM.value
            
            if is_urgent and is_important:
                matrix["do_now"].append(task)
            elif not is_urgent and is_important:
                matrix["schedule"].append(task)
            elif is_urgent and not is_important:
                matrix["delegate"].append(task)
            else:
                matrix["eliminate"].append(task)
        
        return matrix
    
    # ========== SCHEDULING HELPERS ==========
    
    def estimate_wait_time(self, task_id: str) -> Optional[float]:
        """
        Stima tempo di attesa per un task (secondi)
        
        Returns:
            Secondi stimati, None se task non trovato
        """
        if task_id not in self._task_map:
            return None
        
        task = self._task_map[task_id]
        
        # Conta task con priorità maggiore
        tasks_ahead = sum(
            1 for t in self._queue 
            if t.priority_score < task.priority_score
        )
        
        # Stima basata su tempo medio esecuzione (assumi 2 sec/task)
        avg_execution_time = 2.0
        concurrent_slots = max(1, self._max_concurrent - len(self._running_tasks))
        
        return (tasks_ahead / concurrent_slots) * avg_execution_time
    
    def can_execute_now(self) -> bool:
        """Verifica se possiamo eseguire un nuovo task"""
        return len(self._running_tasks) < self._max_concurrent
    
    def mark_running(self, task_id: str):
        """Marca task come in esecuzione"""
        if task_id in self._task_map:
            task = self._task_map[task_id]
            self._running_tasks[task_id] = task
    
    def mark_completed(self, task_id: str):
        """Marca task come completato"""
        if task_id in self._running_tasks:
            task = self._running_tasks.pop(task_id)
            
            # Update stats
            self._stats["total_completed"] += 1
            wait_time = (datetime.now() - task.created_at).total_seconds()
            
            # Running average
            n = self._stats["total_completed"]
            self._stats["average_wait_time"] = (
                (self._stats["average_wait_time"] * (n-1) + wait_time) / n
            )
    
    # ========== CONFIGURATION ==========
    
    def set_intent_priority(self, intent_name: str, priority: Priority):
        """Configura priorità default per un intent"""
        self._intent_priorities[intent_name] = priority
    
    def set_category_importance(self, category: str, importance: Importance):
        """Configura importanza default per una categoria"""
        self._category_importance[category] = importance
    
    def set_max_concurrent(self, max_tasks: int):
        """Imposta numero massimo di task concorrenti"""
        self._max_concurrent = max(1, max_tasks)
    
    def set_on_task_ready(self, callback: Callable):
        """Imposta callback per nuovo task pronto"""
        self._on_task_ready = callback
    
    def set_on_preemption(self, callback: Callable):
        """Imposta callback per preemption"""
        self._on_preemption = callback
    
    # ========== STATUS & STATS ==========
    
    def get_queue_status(self) -> Dict:
        """Stato della coda"""
        return {
            "queue_size": len(self._queue),
            "running_tasks": len(self._running_tasks),
            "max_concurrent": self._max_concurrent,
            "can_execute": self.can_execute_now(),
            "next_task": self.peek().to_dict() if self.peek() else None,
            "urgent_count": len(self.get_urgent()),
            "overdue_count": len(self.get_overdue())
        }
    
    def get_stats(self) -> Dict:
        """Statistiche operative"""
        return {
            **self._stats,
            "queue_size": len(self._queue),
            "running_count": len(self._running_tasks)
        }
    
    def get_priority_summary(self) -> Dict[str, int]:
        """Distribuzione task per priorità"""
        summary = {p.name: 0 for p in Priority}
        for task in self._queue:
            summary[task.priority.name] += 1
        return summary
    
    def clear_queue(self) -> int:
        """Svuota la coda, ritorna numero task rimossi"""
        count = len(self._queue)
        self._queue.clear()
        self._task_map.clear()
        return count


# ========== PRIORITY DECORATOR ==========

def with_priority(priority: Priority = Priority.NORMAL, 
                  importance: Importance = Importance.MEDIUM,
                  urgency: Urgency = Urgency.NORMAL):
    """
    Decorator per assegnare priorità a funzioni/handler
    
    Usage:
        @with_priority(Priority.HIGH, Importance.HIGH)
        async def handle_user_command(action):
            ...
    """
    def decorator(func):
        func._priority = priority
        func._importance = importance
        func._urgency = urgency
        return func
    return decorator
