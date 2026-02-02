"""
🎭 Memoria Episodica - Tracciamento Eventi, Scenari e Contesti

Sistema di memoria che registra e organizza:
- Episodi (sequenze di eventi correlati)
- Scenari (contesti situazionali)
- Timeline degli eventi
- Relazioni causali tra eventi

Permette di:
- Ricostruire cosa è successo
- Comprendere il contesto di decisioni passate
- Identificare pattern temporali
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import json
import logging
from pathlib import Path


# Logger
episodic_logger = logging.getLogger("episodic_memory")
episodic_logger.setLevel(logging.DEBUG)


# === ENUMS ===

class EventType(Enum):
    """Tipi di evento"""
    USER_INPUT = "user_input"           # Input dell'utente
    SYSTEM_ACTION = "system_action"     # Azione del sistema
    DECISION = "decision"               # Decisione presa
    RESULT = "result"                   # Risultato
    ERROR = "error"                     # Errore
    STATE_CHANGE = "state_change"       # Cambio di stato
    CONTEXT_SHIFT = "context_shift"     # Cambio di contesto
    MILESTONE = "milestone"             # Punto importante
    OBSERVATION = "observation"         # Osservazione


class EpisodeStatus(Enum):
    """Stato dell'episodio"""
    ACTIVE = "active"         # In corso
    COMPLETED = "completed"   # Completato
    INTERRUPTED = "interrupted"  # Interrotto
    FAILED = "failed"         # Fallito


class ScenarioType(Enum):
    """Tipi di scenario"""
    TASK = "task"                 # Esecuzione task
    CONVERSATION = "conversation" # Conversazione
    TROUBLESHOOTING = "troubleshooting"  # Risoluzione problema
    LEARNING = "learning"         # Sessione apprendimento
    EXPLORATION = "exploration"   # Esplorazione
    ROUTINE = "routine"           # Routine abituale


# === DATA CLASSES ===

@dataclass
class Event:
    """Singolo evento nella timeline"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Tipo e contenuto
    event_type: EventType = EventType.OBSERVATION
    content: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    # Relazioni
    episode_id: Optional[str] = None
    parent_event_id: Optional[str] = None  # Evento che ha causato questo
    caused_by: List[str] = field(default_factory=list)  # Eventi correlati
    
    # Contesto
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Importanza (1-10)
    importance: int = 5
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[int] = None
    
    # Metadata
    source: str = "system"  # Chi ha generato l'evento
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "content": self.content,
            "description": self.description,
            "episode_id": self.episode_id,
            "parent_event_id": self.parent_event_id,
            "context": self.context,
            "tags": self.tags,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "source": self.source
        }
    
    def age_seconds(self) -> float:
        """Età dell'evento in secondi"""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class Episode:
    """
    Episodio - sequenza di eventi correlati che formano una storia coerente.
    Es: "Sessione di debug del bug #123"
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Identificazione
    title: str = ""
    description: str = ""
    scenario_type: ScenarioType = ScenarioType.TASK
    
    # Eventi
    events: List[Event] = field(default_factory=list)
    event_count: int = 0
    
    # Stato
    status: EpisodeStatus = EpisodeStatus.ACTIVE
    
    # Contesto iniziale e finale
    initial_context: Dict[str, Any] = field(default_factory=dict)
    final_context: Dict[str, Any] = field(default_factory=dict)
    
    # Goal dell'episodio (opzionale)
    goal: Optional[str] = None
    goal_achieved: Optional[bool] = None
    
    # Relazioni
    parent_episode_id: Optional[str] = None  # Episodio padre
    related_episodes: List[str] = field(default_factory=list)
    
    # Tags per categorizzazione
    tags: List[str] = field(default_factory=list)
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    
    # Metriche
    success_events: int = 0
    error_events: int = 0
    
    def add_event(self, event: Event):
        """Aggiunge evento all'episodio"""
        event.episode_id = self.id
        self.events.append(event)
        self.event_count += 1
        
        if event.event_type == EventType.ERROR:
            self.error_events += 1
        elif event.event_type == EventType.RESULT:
            # Gestisci content come stringa o dict
            if isinstance(event.content, dict):
                if event.content.get("success", False):
                    self.success_events += 1
            elif isinstance(event.content, str) and "success" in event.content.lower():
                self.success_events += 1
    
    def close(self, status: EpisodeStatus = EpisodeStatus.COMPLETED,
             goal_achieved: bool = None):
        """Chiude l'episodio"""
        self.status = status
        self.ended_at = datetime.now()
        if goal_achieved is not None:
            self.goal_achieved = goal_achieved
    
    def duration_seconds(self) -> float:
        """Durata dell'episodio"""
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    def success_rate(self) -> float:
        """Tasso di successo degli eventi"""
        total = self.success_events + self.error_events
        return self.success_events / total if total > 0 else 0.5
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "scenario_type": self.scenario_type.value,
            "status": self.status.value,
            "event_count": self.event_count,
            "goal": self.goal,
            "goal_achieved": self.goal_achieved,
            "tags": self.tags,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds(),
            "success_rate": round(self.success_rate(), 3)
        }
    
    def get_summary(self) -> str:
        """Riassunto testuale dell'episodio"""
        status_emoji = {
            EpisodeStatus.ACTIVE: "🔄",
            EpisodeStatus.COMPLETED: "✅",
            EpisodeStatus.INTERRUPTED: "⏸️",
            EpisodeStatus.FAILED: "❌"
        }
        
        return (
            f"{status_emoji.get(self.status, '❓')} {self.title}\n"
            f"   Tipo: {self.scenario_type.value}\n"
            f"   Eventi: {self.event_count} ({self.error_events} errori)\n"
            f"   Durata: {self.duration_seconds():.1f}s"
        )


@dataclass
class Scenario:
    """
    Scenario - contesto situazionale che raggruppa episodi simili.
    Es: "Debug sessioni", "Interazioni email"
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    name: str = ""
    description: str = ""
    scenario_type: ScenarioType = ScenarioType.TASK
    
    # Episodi in questo scenario
    episode_ids: List[str] = field(default_factory=list)
    episode_count: int = 0
    
    # Pattern comuni
    common_patterns: List[Dict[str, Any]] = field(default_factory=list)
    typical_duration: float = 0.0  # Media durata episodi
    
    # Statistiche
    total_events: int = 0
    success_rate: float = 0.5
    
    # Timing
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def add_episode(self, episode: Episode):
        """Aggiunge episodio allo scenario"""
        self.episode_ids.append(episode.id)
        self.episode_count += 1
        self.total_events += episode.event_count
        self.last_seen = datetime.now()
        
        # Aggiorna media durata
        n = self.episode_count
        self.typical_duration = (
            (self.typical_duration * (n - 1) + episode.duration_seconds()) / n
        )
        
        # Aggiorna success rate
        self.success_rate = (
            (self.success_rate * (n - 1) + episode.success_rate()) / n
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scenario_type": self.scenario_type.value,
            "episode_count": self.episode_count,
            "total_events": self.total_events,
            "typical_duration": round(self.typical_duration, 1),
            "success_rate": round(self.success_rate, 3),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }


# === EPISODIC MEMORY ===

class EpisodicMemory:
    """
    Sistema di memoria episodica.
    Traccia e organizza eventi, episodi e scenari.
    """
    
    def __init__(self, storage_path: Path = None, 
                max_events: int = 10000,
                max_episodes: int = 1000):
        self.storage_path = storage_path or Path("data/episodic_memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_events = max_events
        self.max_episodes = max_episodes
        
        # Storage
        self.events: Dict[str, Event] = {}
        self.episodes: Dict[str, Episode] = {}
        self.scenarios: Dict[str, Scenario] = {}
        
        # Episodio attivo corrente
        self.active_episode: Optional[Episode] = None
        
        # Timeline (ordinata per timestamp)
        self.timeline: List[str] = []  # Event IDs
        
        # Indici
        self.events_by_type: Dict[EventType, List[str]] = {t: [] for t in EventType}
        self.events_by_episode: Dict[str, List[str]] = defaultdict(list)
        self.episodes_by_scenario: Dict[str, List[str]] = defaultdict(list)
        
        # Carica dati esistenti
        self._load()
        
        episodic_logger.info(f"EpisodicMemory initialized: {len(self.events)} events, "
                           f"{len(self.episodes)} episodes, {len(self.scenarios)} scenarios")
    
    # === Event Management ===
    
    def record_event(self, 
                    event_type: EventType,
                    content: Dict[str, Any],
                    description: str = "",
                    importance: int = 5,
                    tags: List[str] = None,
                    context: Dict[str, Any] = None,
                    parent_event_id: str = None,
                    source: str = "system") -> Event:
        """Registra un nuovo evento"""
        event = Event(
            event_type=event_type,
            content=content,
            description=description or self._generate_description(event_type, content),
            importance=importance,
            tags=tags or [],
            context=context or {},
            parent_event_id=parent_event_id,
            source=source
        )
        
        # Aggiungi all'episodio attivo se esiste
        if self.active_episode:
            self.active_episode.add_event(event)
        
        # Storage
        self.events[event.id] = event
        self.timeline.append(event.id)
        self.events_by_type[event_type].append(event.id)
        
        if event.episode_id:
            self.events_by_episode[event.episode_id].append(event.id)
        
        # Gestisci overflow
        if len(self.events) > self.max_events:
            self._cleanup_old_events()
        
        # Persist periodicamente
        if len(self.events) % 100 == 0:
            self._save()
        
        episodic_logger.debug(f"Event recorded: {event.id} ({event_type.value})")
        
        return event
    
    def _generate_description(self, event_type: EventType, content: Any) -> str:
        """Genera descrizione automatica"""
        # Gestisce contenuto stringa
        if isinstance(content, str):
            return f"{event_type.value}: {content[:50]}"
        
        # Gestisce contenuto dict
        if not isinstance(content, dict):
            return f"{event_type.value}: {str(content)[:50]}"
        
        if event_type == EventType.USER_INPUT:
            return f"User: {content.get('text', content.get('command', 'input'))}"
        elif event_type == EventType.SYSTEM_ACTION:
            return f"Action: {content.get('action', 'unknown')}"
        elif event_type == EventType.DECISION:
            return f"Decision: {content.get('decision', content.get('outcome', 'made'))}"
        elif event_type == EventType.RESULT:
            status = "Success" if content.get('success', False) else "Failed"
            return f"Result: {status}"
        elif event_type == EventType.ERROR:
            return f"Error: {content.get('error', content.get('message', 'unknown'))}"
        else:
            return f"{event_type.value}: {str(content)[:50]}"
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Recupera evento per ID"""
        return self.events.get(event_id)
    
    def get_recent_events(self, n: int = 20, 
                         event_type: EventType = None) -> List[Event]:
        """Recupera eventi recenti"""
        if event_type:
            event_ids = self.events_by_type.get(event_type, [])[-n:]
        else:
            event_ids = self.timeline[-n:]
        
        return [self.events[eid] for eid in reversed(event_ids) if eid in self.events]
    
    def get_events_in_timerange(self, 
                               start: datetime, 
                               end: datetime = None) -> List[Event]:
        """Recupera eventi in un intervallo temporale"""
        end = end or datetime.now()
        
        events = []
        for event_id in self.timeline:
            event = self.events.get(event_id)
            if event and start <= event.timestamp <= end:
                events.append(event)
        
        return events
    
    def search_events(self, query: str, limit: int = 20) -> List[Event]:
        """Cerca eventi per contenuto"""
        query_lower = query.lower()
        results = []
        
        for event in self.events.values():
            score = 0
            
            # Cerca in descrizione
            if query_lower in event.description.lower():
                score = 1.0
            # Cerca in content
            elif query_lower in json.dumps(event.content).lower():
                score = 0.8
            # Cerca in tags
            elif any(query_lower in tag.lower() for tag in event.tags):
                score = 0.6
            
            if score > 0:
                results.append((event, score * event.importance))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in results[:limit]]
    
    # === Episode Management ===
    
    def start_episode(self, 
                     title: str,
                     scenario_type: ScenarioType = ScenarioType.TASK,
                     goal: str = None,
                     tags: List[str] = None,
                     initial_context: Dict = None) -> Episode:
        """Inizia un nuovo episodio"""
        # Chiudi episodio precedente se attivo
        if self.active_episode:
            self.close_episode(EpisodeStatus.INTERRUPTED)
        
        episode = Episode(
            title=title,
            scenario_type=scenario_type,
            goal=goal,
            tags=tags or [],
            initial_context=initial_context or {}
        )
        
        self.episodes[episode.id] = episode
        self.active_episode = episode
        
        # Registra evento di inizio
        self.record_event(
            EventType.MILESTONE,
            content={"action": "episode_start", "title": title, "goal": goal},
            description=f"Episode started: {title}",
            importance=7,
            tags=["episode_boundary"]
        )
        
        episodic_logger.info(f"Episode started: {episode.id} - {title}")
        
        return episode
    
    def close_episode(self, 
                     status: EpisodeStatus = EpisodeStatus.COMPLETED,
                     goal_achieved: bool = None,
                     final_context: Dict = None,
                     outcome: Dict = None) -> Optional[Episode]:
        """Chiude l'episodio attivo"""
        if not self.active_episode:
            return None
        
        episode = self.active_episode
        episode.close(status, goal_achieved)
        
        if final_context:
            episode.final_context = final_context
        
        if outcome:
            episode.outcome = outcome
        
        # Registra evento di fine
        self.record_event(
            EventType.MILESTONE,
            content={
                "action": "episode_end",
                "status": status.value,
                "goal_achieved": goal_achieved,
                "duration": episode.duration_seconds()
            },
            description=f"Episode ended: {episode.title} ({status.value})",
            importance=7,
            tags=["episode_boundary"]
        )
        
        # Associa a scenario se appropriato
        self._associate_to_scenario(episode)
        
        self.active_episode = None
        
        # Gestisci overflow
        if len(self.episodes) > self.max_episodes:
            self._cleanup_old_episodes()
        
        self._save()
        
        episodic_logger.info(f"Episode closed: {episode.id} - {status.value}")
        
        return episode
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Recupera episodio"""
        return self.episodes.get(episode_id)
    
    def get_recent_episodes(self, n: int = 10,
                           status: EpisodeStatus = None,
                           scenario_type: ScenarioType = None) -> List[Episode]:
        """Recupera episodi recenti"""
        episodes = list(self.episodes.values())
        
        if status:
            episodes = [e for e in episodes if e.status == status]
        if scenario_type:
            episodes = [e for e in episodes if e.scenario_type == scenario_type]
        
        episodes.sort(key=lambda e: e.started_at, reverse=True)
        return episodes[:n]
    
    def get_episode_events(self, episode_id: str) -> List[Event]:
        """Recupera tutti gli eventi di un episodio"""
        episode = self.episodes.get(episode_id)
        if not episode:
            return []
        
        return sorted(episode.events, key=lambda e: e.timestamp)
    
    # === Scenario Management ===
    
    def _associate_to_scenario(self, episode: Episode):
        """Associa episodio a uno scenario esistente o ne crea uno nuovo"""
        # Cerca scenario simile
        best_match = None
        best_score = 0.0
        
        for scenario in self.scenarios.values():
            if scenario.scenario_type != episode.scenario_type:
                continue
            
            # Calcola similarità basata su tags
            if episode.tags and scenario.name:
                common_tags = set(episode.tags) & set(scenario.name.lower().split())
                if common_tags:
                    score = len(common_tags) / max(len(episode.tags), 1)
                    if score > best_score:
                        best_score = score
                        best_match = scenario
        
        if best_match and best_score > 0.3:
            best_match.add_episode(episode)
            self.episodes_by_scenario[best_match.id].append(episode.id)
        else:
            # Crea nuovo scenario
            scenario = Scenario(
                name=episode.title,
                description=episode.description,
                scenario_type=episode.scenario_type
            )
            scenario.add_episode(episode)
            
            self.scenarios[scenario.id] = scenario
            self.episodes_by_scenario[scenario.id].append(episode.id)
    
    def create_scenario(self,
                       name: str,
                       scenario_type: ScenarioType = ScenarioType.TASK,
                       description: str = "",
                       context: Dict = None,
                       tags: List[str] = None) -> Scenario:
        """Crea un nuovo scenario"""
        scenario = Scenario(
            name=name,
            description=description,
            scenario_type=scenario_type
        )
        
        # Aggiungi context come pattern se fornito
        if context:
            scenario.common_patterns = [context]
        
        self.scenarios[scenario.id] = scenario
        
        episodic_logger.info(f"Scenario created: {scenario.id} - {name}")
        
        return scenario
    
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Recupera scenario"""
        return self.scenarios.get(scenario_id)
    
    def get_scenarios(self, scenario_type: ScenarioType = None) -> List[Scenario]:
        """Recupera tutti gli scenari"""
        scenarios = list(self.scenarios.values())
        
        if scenario_type:
            scenarios = [s for s in scenarios if s.scenario_type == scenario_type]
        
        scenarios.sort(key=lambda s: s.last_seen, reverse=True)
        return scenarios
    
    # === Analysis ===
    
    def get_event_chain(self, event_id: str, depth: int = 5) -> List[Event]:
        """Ricostruisce la catena di eventi causali"""
        chain = []
        current_id = event_id
        
        while current_id and len(chain) < depth:
            event = self.events.get(current_id)
            if not event:
                break
            
            chain.append(event)
            current_id = event.parent_event_id
        
        return list(reversed(chain))
    
    def get_timeline_summary(self, hours: int = 24) -> Dict:
        """Riassunto della timeline recente"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            e for e in self.events.values()
            if e.timestamp >= cutoff
        ]
        
        by_type = defaultdict(int)
        by_hour = defaultdict(int)
        
        for event in recent_events:
            by_type[event.event_type.value] += 1
            hour = event.timestamp.strftime("%Y-%m-%d %H:00")
            by_hour[hour] += 1
        
        return {
            "total_events": len(recent_events),
            "by_type": dict(by_type),
            "by_hour": dict(sorted(by_hour.items())),
            "timerange": f"Last {hours} hours"
        }
    
    def find_similar_episodes(self, episode: Episode, limit: int = 5) -> List[Tuple[Episode, float]]:
        """Trova episodi simili"""
        results = []
        
        for other in self.episodes.values():
            if other.id == episode.id:
                continue
            
            score = 0.0
            
            # Stesso tipo scenario
            if other.scenario_type == episode.scenario_type:
                score += 0.3
            
            # Tags comuni
            if episode.tags and other.tags:
                common = len(set(episode.tags) & set(other.tags))
                score += common * 0.2
            
            # Success rate simile
            rate_diff = abs(episode.success_rate() - other.success_rate())
            score += (1 - rate_diff) * 0.2
            
            # Durata simile
            if other.duration_seconds() > 0:
                duration_ratio = min(
                    episode.duration_seconds() / other.duration_seconds(),
                    other.duration_seconds() / episode.duration_seconds()
                )
                score += duration_ratio * 0.1
            
            if score > 0.3:
                results.append((other, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    # === Cleanup ===
    
    def _cleanup_old_events(self):
        """Rimuove eventi vecchi"""
        # Mantieni il 80% più recente
        keep_count = int(self.max_events * 0.8)
        
        if len(self.timeline) > keep_count:
            remove_ids = set(self.timeline[:-keep_count])
            
            for event_id in remove_ids:
                if event_id in self.events:
                    del self.events[event_id]
            
            self.timeline = self.timeline[-keep_count:]
            
            # Ricostruisci indici
            self._rebuild_indices()
            
            episodic_logger.info(f"Cleaned up {len(remove_ids)} old events")
    
    def _cleanup_old_episodes(self):
        """Rimuove episodi vecchi"""
        keep_count = int(self.max_episodes * 0.8)
        
        episodes_sorted = sorted(
            self.episodes.values(),
            key=lambda e: e.started_at,
            reverse=True
        )
        
        keep_ids = {e.id for e in episodes_sorted[:keep_count]}
        remove_ids = set(self.episodes.keys()) - keep_ids
        
        for episode_id in remove_ids:
            del self.episodes[episode_id]
        
        episodic_logger.info(f"Cleaned up {len(remove_ids)} old episodes")
    
    def _rebuild_indices(self):
        """Ricostruisce indici dopo cleanup"""
        self.events_by_type = {t: [] for t in EventType}
        self.events_by_episode = defaultdict(list)
        
        for event_id in self.timeline:
            if event_id not in self.events:
                continue
            
            event = self.events[event_id]
            self.events_by_type[event.event_type].append(event_id)
            
            if event.episode_id:
                self.events_by_episode[event.episode_id].append(event_id)
    
    # === Persistence ===
    
    def _save(self):
        """Salva su disco"""
        try:
            # Salva eventi (solo i più recenti per efficienza)
            recent_events = {
                eid: e.to_dict()
                for eid, e in list(self.events.items())[-5000:]
            }
            
            with open(self.storage_path / "events.json", "w") as f:
                json.dump(recent_events, f)
            
            # Salva episodi
            episodes_data = {eid: e.to_dict() for eid, e in self.episodes.items()}
            
            with open(self.storage_path / "episodes.json", "w") as f:
                json.dump(episodes_data, f)
            
            # Salva scenari
            scenarios_data = {sid: s.to_dict() for sid, s in self.scenarios.items()}
            
            with open(self.storage_path / "scenarios.json", "w") as f:
                json.dump(scenarios_data, f)
                
        except Exception as e:
            episodic_logger.error(f"Failed to save episodic memory: {e}")
    
    def _load(self):
        """Carica da disco"""
        # Load events
        events_file = self.storage_path / "events.json"
        if events_file.exists():
            try:
                with open(events_file) as f:
                    data = json.load(f)
                
                for eid, edata in data.items():
                    event = Event(
                        id=eid,
                        event_type=EventType(edata["event_type"]),
                        content=edata.get("content", {}),
                        description=edata.get("description", ""),
                        episode_id=edata.get("episode_id"),
                        parent_event_id=edata.get("parent_event_id"),
                        context=edata.get("context", {}),
                        tags=edata.get("tags", []),
                        importance=edata.get("importance", 5),
                        source=edata.get("source", "system")
                    )
                    event.timestamp = datetime.fromisoformat(edata["timestamp"])
                    
                    self.events[eid] = event
                    self.timeline.append(eid)
                    self.events_by_type[event.event_type].append(eid)
                    
            except Exception as e:
                episodic_logger.error(f"Failed to load events: {e}")
        
        # Load episodes
        episodes_file = self.storage_path / "episodes.json"
        if episodes_file.exists():
            try:
                with open(episodes_file) as f:
                    data = json.load(f)
                
                for eid, edata in data.items():
                    episode = Episode(
                        id=eid,
                        title=edata.get("title", ""),
                        description=edata.get("description", ""),
                        scenario_type=ScenarioType(edata.get("scenario_type", "task")),
                        status=EpisodeStatus(edata.get("status", "completed")),
                        goal=edata.get("goal"),
                        goal_achieved=edata.get("goal_achieved"),
                        tags=edata.get("tags", [])
                    )
                    episode.started_at = datetime.fromisoformat(edata["started_at"])
                    if edata.get("ended_at"):
                        episode.ended_at = datetime.fromisoformat(edata["ended_at"])
                    episode.event_count = edata.get("event_count", 0)
                    
                    self.episodes[eid] = episode
                    
            except Exception as e:
                episodic_logger.error(f"Failed to load episodes: {e}")
        
        # Load scenarios
        scenarios_file = self.storage_path / "scenarios.json"
        if scenarios_file.exists():
            try:
                with open(scenarios_file) as f:
                    data = json.load(f)
                
                for sid, sdata in data.items():
                    scenario = Scenario(
                        id=sid,
                        name=sdata.get("name", ""),
                        description=sdata.get("description", ""),
                        scenario_type=ScenarioType(sdata.get("scenario_type", "task")),
                        episode_count=sdata.get("episode_count", 0),
                        total_events=sdata.get("total_events", 0),
                        typical_duration=sdata.get("typical_duration", 0),
                        success_rate=sdata.get("success_rate", 0.5)
                    )
                    self.scenarios[sid] = scenario
                    
            except Exception as e:
                episodic_logger.error(f"Failed to load scenarios: {e}")
    
    # === Stats ===
    
    def get_stats(self) -> Dict:
        """Statistiche memoria episodica"""
        return {
            "total_events": len(self.events),
            "total_episodes": len(self.episodes),
            "total_scenarios": len(self.scenarios),
            "active_episode": self.active_episode.id if self.active_episode else None,
            "events_by_type": {
                t.value: len(ids) for t, ids in self.events_by_type.items()
            },
            "completed_episodes": sum(
                1 for e in self.episodes.values()
                if e.status == EpisodeStatus.COMPLETED
            ),
            "timeline_span": self._get_timeline_span()
        }
    
    def _get_timeline_span(self) -> str:
        """Intervallo temporale della timeline"""
        if not self.timeline:
            return "empty"
        
        first = self.events.get(self.timeline[0])
        last = self.events.get(self.timeline[-1])
        
        if first and last:
            span = last.timestamp - first.timestamp
            return f"{span.days}d {span.seconds // 3600}h"
        
        return "unknown"
