"""
💡 GIDEON 3.0 - Smart Suggestion Engine
Sistema di suggerimenti intelligenti basato su contesto, storia e pattern
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class SuggestionType(Enum):
    """Tipi di suggerimenti"""
    ACTION = "action"           # Azione da eseguire
    OPTIMIZATION = "optimization"  # Ottimizzazione sistema
    AUTOMATION = "automation"   # Proposta automazione
    WARNING = "warning"         # Avviso preventivo
    TIP = "tip"                # Consiglio generico
    LEARNING = "learning"       # Suggerimento educativo


class SuggestionPriority(Enum):
    """Priorità suggerimenti"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Suggestion:
    """Struttura di un suggerimento"""
    id: str
    type: SuggestionType
    priority: SuggestionPriority
    title: str
    description: str
    action: Optional[dict] = None
    context_match: float = 0.0
    confidence: float = 0.0
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "action": self.action,
            "context_match": round(self.context_match, 2),
            "confidence": round(self.confidence, 2),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat()
        }


class SuggestionEngine:
    """
    Motore di suggerimenti intelligenti di Gideon 3.0
    
    Features:
    - Pattern recognition su azioni utente
    - Context-aware suggestions
    - Proactive recommendations
    - Learning from feedback
    """
    
    def __init__(self):
        self.suggestions: Dict[str, Suggestion] = {}
        self.action_history: List[dict] = []
        self.pattern_cache: Dict[str, dict] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.feedback_scores: Dict[str, float] = {}  # suggestion_id -> score
        
        # Pattern templates
        self.patterns = {
            "frequent_action": self._detect_frequent_actions,
            "time_based": self._detect_time_patterns,
            "sequence": self._detect_sequences,
            "resource": self._detect_resource_patterns
        }
        
        # Regole di suggerimento
        self.suggestion_rules = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Inizializza regole di suggerimento predefinite"""
        self.suggestion_rules = [
            {
                "id": "repeated_search",
                "condition": lambda ctx: ctx.get("search_count", 0) > 3,
                "suggestion": {
                    "type": SuggestionType.AUTOMATION,
                    "title": "Ricerche frequenti rilevate",
                    "description": "Noto che cerchi spesso informazioni simili. Vuoi creare una routine automatica?",
                    "priority": SuggestionPriority.MEDIUM
                }
            },
            {
                "id": "high_cpu",
                "condition": lambda ctx: ctx.get("cpu_percent", 0) > 80,
                "suggestion": {
                    "type": SuggestionType.OPTIMIZATION,
                    "title": "CPU sotto carico",
                    "description": "Il processore è molto utilizzato. Vuoi che analizzi quali processi consumano più risorse?",
                    "action": {"type": "analyze_processes"},
                    "priority": SuggestionPriority.HIGH
                }
            },
            {
                "id": "memory_warning",
                "condition": lambda ctx: ctx.get("memory_percent", 0) > 85,
                "suggestion": {
                    "type": SuggestionType.WARNING,
                    "title": "Memoria quasi piena",
                    "description": "La RAM sta per esaurirsi. Consiglio di chiudere alcune applicazioni.",
                    "action": {"type": "list_memory_heavy"},
                    "priority": SuggestionPriority.HIGH
                }
            },
            {
                "id": "evening_routine",
                "condition": lambda ctx: 17 <= datetime.now().hour <= 19 and ctx.get("work_session_long", False),
                "suggestion": {
                    "type": SuggestionType.TIP,
                    "title": "Fine giornata lavorativa",
                    "description": "Hai lavorato a lungo. È un buon momento per salvare tutto e fare una pausa.",
                    "priority": SuggestionPriority.LOW
                }
            },
            {
                "id": "browser_tabs",
                "condition": lambda ctx: ctx.get("browser_tabs", 0) > 20,
                "suggestion": {
                    "type": SuggestionType.OPTIMIZATION,
                    "title": "Troppe schede aperte",
                    "description": "Hai molte schede aperte nel browser. Vuoi che raggruppi quelle simili?",
                    "action": {"type": "organize_tabs"},
                    "priority": SuggestionPriority.MEDIUM
                }
            }
        ]
    
    # ============================================
    # CORE SUGGESTION GENERATION
    # ============================================
    
    async def generate(self, context: dict) -> List[Suggestion]:
        """
        Genera suggerimenti basati sul contesto attuale
        
        Args:
            context: Contesto corrente (risorse, azioni recenti, ora, etc.)
            
        Returns:
            Lista di suggerimenti ordinati per rilevanza
        """
        suggestions = []
        
        # 1. Suggerimenti basati su regole
        rule_suggestions = await self._apply_rules(context)
        suggestions.extend(rule_suggestions)
        
        # 2. Suggerimenti basati su pattern
        pattern_suggestions = await self._analyze_patterns(context)
        suggestions.extend(pattern_suggestions)
        
        # 3. Suggerimenti proattivi
        proactive_suggestions = await self._generate_proactive(context)
        suggestions.extend(proactive_suggestions)
        
        # Rimuovi duplicati e ordina
        unique_suggestions = self._deduplicate(suggestions)
        sorted_suggestions = self._sort_by_relevance(unique_suggestions)
        
        # Salva nel cache
        for s in sorted_suggestions[:10]:
            self.suggestions[s.id] = s
        
        return sorted_suggestions[:5]  # Top 5 suggerimenti
    
    async def _apply_rules(self, context: dict) -> List[Suggestion]:
        """Applica regole di suggerimento"""
        suggestions = []
        
        for rule in self.suggestion_rules:
            try:
                if rule["condition"](context):
                    sugg_data = rule["suggestion"]
                    suggestion = Suggestion(
                        id=f"rule_{rule['id']}_{datetime.now().strftime('%H%M')}",
                        type=sugg_data["type"],
                        priority=sugg_data["priority"],
                        title=sugg_data["title"],
                        description=sugg_data["description"],
                        action=sugg_data.get("action"),
                        confidence=0.8,
                        context_match=0.9
                    )
                    suggestions.append(suggestion)
            except Exception:
                pass  # Ignora errori nelle regole
        
        return suggestions
    
    async def _analyze_patterns(self, context: dict) -> List[Suggestion]:
        """Analizza pattern nelle azioni dell'utente"""
        suggestions = []
        
        for pattern_name, detector in self.patterns.items():
            detected = await detector(context)
            if detected:
                suggestions.extend(detected)
        
        return suggestions
    
    async def _generate_proactive(self, context: dict) -> List[Suggestion]:
        """Genera suggerimenti proattivi basati su previsioni"""
        suggestions = []
        
        # Suggerimento basato sull'ora del giorno
        hour = datetime.now().hour
        
        if 6 <= hour < 9:
            suggestions.append(Suggestion(
                id=f"proactive_morning_{hour}",
                type=SuggestionType.TIP,
                priority=SuggestionPriority.LOW,
                title="Buongiorno!",
                description="Inizia la giornata controllando la tua to-do list.",
                confidence=0.6,
                context_match=0.5
            ))
        elif 12 <= hour < 14:
            suggestions.append(Suggestion(
                id=f"proactive_lunch_{hour}",
                type=SuggestionType.TIP,
                priority=SuggestionPriority.LOW,
                title="Pausa pranzo",
                description="È ora di pranzo. Ricorda di fare una pausa!",
                confidence=0.5,
                context_match=0.4
            ))
        
        return suggestions
    
    # ============================================
    # PATTERN DETECTION
    # ============================================
    
    async def _detect_frequent_actions(self, context: dict) -> List[Suggestion]:
        """Rileva azioni eseguite frequentemente"""
        if len(self.action_history) < 5:
            return []
        
        # Conta azioni recenti
        action_counts = defaultdict(int)
        for action in self.action_history[-20:]:
            action_type = action.get("type", "unknown")
            action_counts[action_type] += 1
        
        suggestions = []
        for action_type, count in action_counts.items():
            if count >= 3:
                suggestions.append(Suggestion(
                    id=f"frequent_{action_type}_{datetime.now().strftime('%H%M')}",
                    type=SuggestionType.AUTOMATION,
                    priority=SuggestionPriority.MEDIUM,
                    title=f"Azione frequente: {action_type}",
                    description=f"Hai eseguito '{action_type}' {count} volte di recente. Vuoi automatizzarla?",
                    action={"type": "create_automation", "action": action_type},
                    confidence=min(count / 10, 0.9),
                    context_match=0.8
                ))
        
        return suggestions
    
    async def _detect_time_patterns(self, context: dict) -> List[Suggestion]:
        """Rileva pattern temporali"""
        suggestions = []
        
        # Analizza se certe azioni avvengono a orari specifici
        hourly_actions = defaultdict(lambda: defaultdict(int))
        
        for action in self.action_history:
            action_time = action.get("timestamp")
            if action_time:
                try:
                    dt = datetime.fromisoformat(action_time)
                    hour = dt.hour
                    action_type = action.get("type", "unknown")
                    hourly_actions[hour][action_type] += 1
                except:
                    pass
        
        current_hour = datetime.now().hour
        
        # Se c'è un pattern forte per quest'ora
        if current_hour in hourly_actions:
            for action_type, count in hourly_actions[current_hour].items():
                if count >= 2:
                    suggestions.append(Suggestion(
                        id=f"time_pattern_{action_type}_{current_hour}",
                        type=SuggestionType.AUTOMATION,
                        priority=SuggestionPriority.MEDIUM,
                        title=f"Azione abituale alle {current_hour}:00",
                        description=f"Solitamente esegui '{action_type}' a quest'ora. Vuoi che lo faccia io?",
                        action={"type": action_type},
                        confidence=min(count / 5, 0.85),
                        context_match=0.85
                    ))
        
        return suggestions
    
    async def _detect_sequences(self, context: dict) -> List[Suggestion]:
        """Rileva sequenze di azioni comuni"""
        if len(self.action_history) < 3:
            return []
        
        # Trova sequenze di 2 azioni
        sequences = defaultdict(int)
        
        for i in range(len(self.action_history) - 1):
            seq = (
                self.action_history[i].get("type"),
                self.action_history[i + 1].get("type")
            )
            sequences[seq] += 1
        
        suggestions = []
        for (action1, action2), count in sequences.items():
            if count >= 2 and action1 and action2:
                suggestions.append(Suggestion(
                    id=f"sequence_{action1}_{action2}",
                    type=SuggestionType.AUTOMATION,
                    priority=SuggestionPriority.MEDIUM,
                    title="Sequenza rilevata",
                    description=f"Spesso dopo '{action1}' esegui '{action2}'. Vuoi creare una macro?",
                    action={"type": "create_macro", "steps": [action1, action2]},
                    confidence=min(count / 5, 0.8),
                    context_match=0.75
                ))
        
        return suggestions[:3]  # Max 3 suggerimenti di sequenza
    
    async def _detect_resource_patterns(self, context: dict) -> List[Suggestion]:
        """Rileva pattern nell'uso risorse"""
        suggestions = []
        
        cpu = context.get("cpu_percent", 0)
        memory = context.get("memory_percent", 0)
        
        if cpu > 60 and cpu < 80:
            suggestions.append(Suggestion(
                id=f"resource_cpu_moderate",
                type=SuggestionType.TIP,
                priority=SuggestionPriority.LOW,
                title="CPU moderatamente utilizzata",
                description="Il processore è attivo ma non sovraccarico. Buon momento per task intensivi.",
                confidence=0.6,
                context_match=0.5
            ))
        
        if memory > 70 and memory < 85:
            suggestions.append(Suggestion(
                id=f"resource_memory_moderate",
                type=SuggestionType.TIP,
                priority=SuggestionPriority.LOW,
                title="Memoria in uso",
                description="La RAM è occupata al 70%+. Considera di non aprire altre app pesanti.",
                confidence=0.6,
                context_match=0.5
            ))
        
        return suggestions
    
    # ============================================
    # FEEDBACK AND LEARNING
    # ============================================
    
    def record_feedback(self, suggestion_id: str, accepted: bool, 
                        rating: float = None):
        """Registra feedback su un suggerimento"""
        if rating is None:
            rating = 1.0 if accepted else 0.0
        
        self.feedback_scores[suggestion_id] = rating
        
        # Aggiorna preferenze utente
        if suggestion_id in self.suggestions:
            sugg = self.suggestions[suggestion_id]
            pref_key = f"type_{sugg.type.value}"
            current = self.user_preferences.get(pref_key, 0.5)
            # Media mobile
            self.user_preferences[pref_key] = current * 0.7 + rating * 0.3
    
    def record_action(self, action: dict):
        """Registra un'azione eseguita per pattern learning"""
        action["timestamp"] = datetime.now().isoformat()
        self.action_history.append(action)
        
        # Mantieni solo ultime 100 azioni
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
    
    # ============================================
    # UTILITY FUNCTIONS
    # ============================================
    
    def _deduplicate(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Rimuove suggerimenti duplicati"""
        seen_titles = set()
        unique = []
        
        for s in suggestions:
            if s.title not in seen_titles:
                seen_titles.add(s.title)
                unique.append(s)
        
        return unique
    
    def _sort_by_relevance(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Ordina suggerimenti per rilevanza complessiva"""
        def relevance_score(s: Suggestion) -> float:
            # Combina priorità, confidenza e match
            priority_score = s.priority.value / 4.0
            
            # Applica preferenze utente
            pref_key = f"type_{s.type.value}"
            user_pref = self.user_preferences.get(pref_key, 0.5)
            
            return (
                priority_score * 0.3 +
                s.confidence * 0.3 +
                s.context_match * 0.2 +
                user_pref * 0.2
            )
        
        return sorted(suggestions, key=relevance_score, reverse=True)
    
    def add_rule(self, rule_id: str, condition: callable, 
                 suggestion_type: SuggestionType,
                 title: str, description: str,
                 priority: SuggestionPriority = SuggestionPriority.MEDIUM,
                 action: dict = None):
        """Aggiunge una nuova regola di suggerimento"""
        self.suggestion_rules.append({
            "id": rule_id,
            "condition": condition,
            "suggestion": {
                "type": suggestion_type,
                "title": title,
                "description": description,
                "priority": priority,
                "action": action
            }
        })
    
    def get_suggestion(self, suggestion_id: str) -> Optional[dict]:
        """Ottiene un suggerimento per ID"""
        if suggestion_id in self.suggestions:
            return self.suggestions[suggestion_id].to_dict()
        return None
    
    def get_active_suggestions(self) -> List[dict]:
        """Restituisce suggerimenti attivi (non scaduti)"""
        now = datetime.now()
        active = []
        
        for s in self.suggestions.values():
            if s.expires_at is None or s.expires_at > now:
                active.append(s.to_dict())
        
        return active
    
    def get_statistics(self) -> dict:
        """Statistiche del motore di suggerimenti"""
        accepted = sum(1 for s in self.feedback_scores.values() if s > 0.5)
        total = len(self.feedback_scores)
        
        return {
            "total_suggestions_generated": len(self.suggestions),
            "actions_recorded": len(self.action_history),
            "rules_active": len(self.suggestion_rules),
            "feedback_received": total,
            "acceptance_rate": accepted / total if total > 0 else 0,
            "user_preferences": self.user_preferences
        }
