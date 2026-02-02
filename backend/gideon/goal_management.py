# /backend/gideon/goal_management.py
"""
🔮 GIDEON 3.0 - Goal Management
Gestisce obiettivi, priorità e tracking del progresso.
NON esegue azioni - fornisce solo gestione e analisi obiettivi.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
import logging
import uuid

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Stati di un obiettivo"""
    DRAFT = "draft"               # In definizione
    ACTIVE = "active"             # Attivo
    IN_PROGRESS = "in_progress"   # In esecuzione
    BLOCKED = "blocked"           # Bloccato
    PAUSED = "paused"            # In pausa
    COMPLETED = "completed"       # Completato
    FAILED = "failed"            # Fallito
    CANCELLED = "cancelled"       # Cancellato


class GoalPriority(Enum):
    """Priorità obiettivo"""
    CRITICAL = "critical"         # P0 - Critico
    HIGH = "high"                # P1 - Alta
    MEDIUM = "medium"            # P2 - Media
    LOW = "low"                  # P3 - Bassa
    NICE_TO_HAVE = "nice_to_have"  # P4 - Opzionale


class GoalType(Enum):
    """Tipi di obiettivo"""
    OUTCOME = "outcome"           # Risultato finale
    MILESTONE = "milestone"       # Punto intermedio
    TASK = "task"                # Compito specifico
    HABIT = "habit"              # Abitudine ricorrente
    METRIC = "metric"            # Metrica da raggiungere
    CONSTRAINT = "constraint"     # Vincolo da rispettare


class AlignmentType(Enum):
    """Allineamento tra obiettivi"""
    SUPPORTS = "supports"         # A supporta B
    CONFLICTS = "conflicts"       # A in conflitto con B
    REQUIRES = "requires"         # A richiede B
    ENABLES = "enables"           # A abilita B
    NEUTRAL = "neutral"           # Indipendenti


@dataclass
class GoalMetric:
    """Metrica per misurare progresso obiettivo"""
    name: str
    target_value: float
    current_value: float = 0.0
    unit: str = ""
    
    # Direction
    higher_is_better: bool = True
    
    # Thresholds
    min_acceptable: Optional[float] = None
    warning_threshold: Optional[float] = None
    
    # History
    history: List[Tuple[datetime, float]] = field(default_factory=list)
    
    def progress_percentage(self) -> float:
        """Calcola percentuale progresso"""
        if self.target_value == 0:
            return 100.0 if self.current_value > 0 else 0.0
        
        if self.higher_is_better:
            return min(100.0, (self.current_value / self.target_value) * 100)
        else:
            if self.current_value == 0:
                return 100.0
            return min(100.0, (self.target_value / self.current_value) * 100)
    
    def is_achieved(self) -> bool:
        """Verifica se metrica è raggiunta"""
        if self.higher_is_better:
            return self.current_value >= self.target_value
        else:
            return self.current_value <= self.target_value
    
    def record_value(self, value: float):
        """Registra nuovo valore"""
        self.current_value = value
        self.history.append((datetime.now(), value))
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target": self.target_value,
            "current": self.current_value,
            "unit": self.unit,
            "progress": round(self.progress_percentage(), 1),
            "achieved": self.is_achieved()
        }


@dataclass
class Goal:
    """Definizione di un obiettivo"""
    id: str
    name: str
    description: str
    goal_type: GoalType
    
    # Status
    status: GoalStatus = GoalStatus.DRAFT
    priority: GoalPriority = GoalPriority.MEDIUM
    
    # Progress
    progress: float = 0.0  # 0-100
    metrics: List[GoalMetric] = field(default_factory=list)
    
    # Timeline
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Hierarchy
    parent_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    # Context
    context: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    owner: str = "system"
    
    # Success criteria
    success_criteria: List[str] = field(default_factory=list)
    acceptance_threshold: float = 80.0  # % progress per completamento
    
    def update_progress(self):
        """Aggiorna progresso basato su metriche e sub-goals"""
        if self.metrics:
            self.progress = statistics.mean(m.progress_percentage() for m in self.metrics)
    
    def is_overdue(self) -> bool:
        """Verifica se obiettivo è in ritardo"""
        if self.deadline and self.status not in [GoalStatus.COMPLETED, GoalStatus.CANCELLED]:
            return datetime.now() > self.deadline
        return False
    
    def days_until_deadline(self) -> Optional[int]:
        """Giorni rimanenti alla deadline"""
        if self.deadline:
            delta = self.deadline - datetime.now()
            return delta.days
        return None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.goal_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": round(self.progress, 1),
            "metrics": [m.to_dict() for m in self.metrics],
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "is_overdue": self.is_overdue(),
            "days_until_deadline": self.days_until_deadline(),
            "parent_id": self.parent_id,
            "sub_goals": self.sub_goals,
            "depends_on": self.depends_on,
            "tags": self.tags
        }


@dataclass
class GoalAlignment:
    """Allineamento tra due obiettivi"""
    goal_a_id: str
    goal_b_id: str
    alignment_type: AlignmentType
    strength: float = 0.5  # 0-1
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            "goal_a": self.goal_a_id,
            "goal_b": self.goal_b_id,
            "type": self.alignment_type.value,
            "strength": round(self.strength, 2),
            "description": self.description
        }


@dataclass
class GoalRecommendation:
    """Raccomandazione per un obiettivo"""
    goal_id: str
    recommendation_type: str
    title: str
    description: str
    priority: int = 5
    impact: float = 0.5
    effort: float = 0.5
    
    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "type": self.recommendation_type,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "impact": round(self.impact, 2),
            "effort": round(self.effort, 2)
        }


class GoalManagement:
    """
    Sistema di gestione obiettivi per Gideon.
    Organizza, traccia e ottimizza il raggiungimento degli obiettivi.
    """
    
    def __init__(self):
        # Storage
        self.goals: Dict[str, Goal] = {}
        self.alignments: List[GoalAlignment] = []
        
        # Templates
        self.goal_templates: Dict[str, Dict] = {}
        self._register_default_templates()
        
        # History
        self.completed_goals: List[Goal] = []
        
        # Priority weights
        self.priority_weights = {
            GoalPriority.CRITICAL: 5,
            GoalPriority.HIGH: 4,
            GoalPriority.MEDIUM: 3,
            GoalPriority.LOW: 2,
            GoalPriority.NICE_TO_HAVE: 1
        }
    
    def _register_default_templates(self):
        """Registra template obiettivi predefiniti"""
        self.goal_templates = {
            "system_optimization": {
                "type": GoalType.OUTCOME,
                "metrics": [
                    {"name": "performance_score", "target": 90, "unit": "%"},
                    {"name": "error_rate", "target": 1, "higher_is_better": False, "unit": "%"}
                ],
                "success_criteria": ["Performance score > 90%", "Error rate < 1%"]
            },
            "task_completion": {
                "type": GoalType.TASK,
                "metrics": [
                    {"name": "completion", "target": 100, "unit": "%"}
                ],
                "success_criteria": ["Task completed successfully"]
            },
            "learning_goal": {
                "type": GoalType.METRIC,
                "metrics": [
                    {"name": "accuracy", "target": 85, "unit": "%"},
                    {"name": "samples_processed", "target": 1000, "unit": ""}
                ],
                "success_criteria": ["Accuracy > 85%", "Processed 1000+ samples"]
            },
            "automation_workflow": {
                "type": GoalType.OUTCOME,
                "metrics": [
                    {"name": "automation_rate", "target": 80, "unit": "%"},
                    {"name": "manual_interventions", "target": 5, "higher_is_better": False, "unit": ""}
                ],
                "success_criteria": ["80%+ automated", "< 5 manual interventions"]
            }
        }
    
    # === Goal Creation ===
    
    def create_goal(self, name: str, description: str,
                    goal_type: GoalType = GoalType.OUTCOME,
                    priority: GoalPriority = GoalPriority.MEDIUM,
                    deadline: datetime = None,
                    template: str = None,
                    **kwargs) -> Goal:
        """
        Crea un nuovo obiettivo.
        """
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        
        goal = Goal(
            id=goal_id,
            name=name,
            description=description,
            goal_type=goal_type,
            priority=priority,
            deadline=deadline,
            parent_id=kwargs.get("parent_id"),
            depends_on=kwargs.get("depends_on", []),
            tags=kwargs.get("tags", []),
            owner=kwargs.get("owner", "system"),
            context=kwargs.get("context", {})
        )
        
        # Apply template
        if template and template in self.goal_templates:
            self._apply_template(goal, template)
        
        # Add custom metrics
        for metric_data in kwargs.get("metrics", []):
            goal.metrics.append(GoalMetric(**metric_data))
        
        # Add success criteria
        goal.success_criteria.extend(kwargs.get("success_criteria", []))
        
        self.goals[goal_id] = goal
        logger.info(f"Goal creato: {goal_id} - {name}")
        
        return goal
    
    def _apply_template(self, goal: Goal, template_name: str):
        """Applica template a obiettivo"""
        template = self.goal_templates[template_name]
        
        # Set type
        goal.goal_type = template.get("type", goal.goal_type)
        
        # Add metrics
        for metric_data in template.get("metrics", []):
            goal.metrics.append(GoalMetric(
                name=metric_data["name"],
                target_value=metric_data["target"],
                unit=metric_data.get("unit", ""),
                higher_is_better=metric_data.get("higher_is_better", True)
            ))
        
        # Add criteria
        goal.success_criteria.extend(template.get("success_criteria", []))
    
    def create_sub_goal(self, parent_id: str, name: str, description: str,
                        **kwargs) -> Optional[Goal]:
        """Crea sub-obiettivo"""
        if parent_id not in self.goals:
            return None
        
        parent = self.goals[parent_id]
        
        sub_goal = self.create_goal(
            name=name,
            description=description,
            goal_type=GoalType.TASK,
            priority=parent.priority,
            parent_id=parent_id,
            **kwargs
        )
        
        parent.sub_goals.append(sub_goal.id)
        return sub_goal
    
    # === Goal Updates ===
    
    def update_status(self, goal_id: str, status: GoalStatus) -> bool:
        """Aggiorna stato obiettivo"""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        old_status = goal.status
        goal.status = status
        
        # Track timestamps
        if status == GoalStatus.IN_PROGRESS and not goal.started_at:
            goal.started_at = datetime.now()
        elif status == GoalStatus.COMPLETED:
            goal.completed_at = datetime.now()
            goal.progress = 100.0
            self.completed_goals.append(goal)
        
        logger.info(f"Goal {goal_id}: {old_status.value} → {status.value}")
        return True
    
    def update_metric(self, goal_id: str, metric_name: str, value: float) -> bool:
        """Aggiorna valore metrica"""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        
        for metric in goal.metrics:
            if metric.name == metric_name:
                metric.record_value(value)
                goal.update_progress()
                
                # Check auto-completion
                if goal.progress >= goal.acceptance_threshold:
                    if all(m.is_achieved() for m in goal.metrics):
                        self.update_status(goal_id, GoalStatus.COMPLETED)
                
                return True
        
        return False
    
    def update_progress(self, goal_id: str, progress: float) -> bool:
        """Aggiorna progresso manualmente"""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        goal.progress = max(0, min(100, progress))
        
        if progress >= goal.acceptance_threshold:
            self.update_status(goal_id, GoalStatus.COMPLETED)
        elif progress > 0 and goal.status == GoalStatus.ACTIVE:
            self.update_status(goal_id, GoalStatus.IN_PROGRESS)
        
        return True
    
    # === Goal Relationships ===
    
    def add_dependency(self, goal_id: str, depends_on_id: str) -> bool:
        """Aggiunge dipendenza"""
        if goal_id not in self.goals or depends_on_id not in self.goals:
            return False
        
        self.goals[goal_id].depends_on.append(depends_on_id)
        self.goals[depends_on_id].blocks.append(goal_id)
        
        # Add alignment
        self.alignments.append(GoalAlignment(
            goal_a_id=goal_id,
            goal_b_id=depends_on_id,
            alignment_type=AlignmentType.REQUIRES,
            strength=1.0
        ))
        
        return True
    
    def add_alignment(self, goal_a_id: str, goal_b_id: str,
                      alignment_type: AlignmentType,
                      strength: float = 0.5) -> bool:
        """Aggiunge allineamento tra obiettivi"""
        if goal_a_id not in self.goals or goal_b_id not in self.goals:
            return False
        
        self.alignments.append(GoalAlignment(
            goal_a_id=goal_a_id,
            goal_b_id=goal_b_id,
            alignment_type=alignment_type,
            strength=strength
        ))
        
        return True
    
    def find_conflicts(self) -> List[GoalAlignment]:
        """Trova conflitti tra obiettivi"""
        return [a for a in self.alignments if a.alignment_type == AlignmentType.CONFLICTS]
    
    # === Goal Analysis ===
    
    def prioritize_goals(self, goals: List[Goal] = None) -> List[Goal]:
        """
        Prioritizza obiettivi considerando urgenza, importanza e dipendenze.
        """
        if goals is None:
            goals = [g for g in self.goals.values() 
                    if g.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]]
        
        def priority_score(goal: Goal) -> float:
            score = 0.0
            
            # Priority weight
            score += self.priority_weights[goal.priority] * 10
            
            # Urgency (deadline proximity)
            if goal.deadline:
                days = goal.days_until_deadline()
                if days is not None:
                    if days < 0:  # Overdue
                        score += 50
                    elif days < 1:
                        score += 30
                    elif days < 7:
                        score += 20
                    else:
                        score += max(0, 10 - days)
            
            # Progress (prefer goals near completion)
            if 80 <= goal.progress < 100:
                score += 15
            
            # Dependencies (goals that block others are higher)
            score += len(goal.blocks) * 5
            
            # Blocked dependencies reduce score
            blocked_deps = sum(1 for d in goal.depends_on 
                             if d in self.goals and 
                             self.goals[d].status != GoalStatus.COMPLETED)
            score -= blocked_deps * 10
            
            return score
        
        return sorted(goals, key=priority_score, reverse=True)
    
    def analyze_goal_health(self, goal_id: str) -> Dict:
        """Analizza salute di un obiettivo"""
        if goal_id not in self.goals:
            return {"error": "Goal not found"}
        
        goal = self.goals[goal_id]
        
        # Calculate health indicators
        indicators = {
            "progress_health": self._assess_progress_health(goal),
            "timeline_health": self._assess_timeline_health(goal),
            "dependency_health": self._assess_dependency_health(goal),
            "metric_health": self._assess_metric_health(goal)
        }
        
        # Overall health
        overall = statistics.mean(indicators.values())
        
        # Generate recommendations
        recommendations = self._generate_recommendations(goal, indicators)
        
        return {
            "goal_id": goal_id,
            "overall_health": round(overall, 2),
            "indicators": indicators,
            "status": goal.status.value,
            "progress": goal.progress,
            "is_at_risk": overall < 0.5,
            "recommendations": [r.to_dict() for r in recommendations]
        }
    
    def _assess_progress_health(self, goal: Goal) -> float:
        """Valuta salute del progresso"""
        if goal.status == GoalStatus.COMPLETED:
            return 1.0
        
        if not goal.started_at:
            return 0.5  # Not started
        
        # Time elapsed vs progress
        elapsed = (datetime.now() - goal.started_at).total_seconds()
        
        if goal.deadline:
            total_time = (goal.deadline - goal.started_at).total_seconds()
            if total_time > 0:
                expected_progress = (elapsed / total_time) * 100
                
                if goal.progress >= expected_progress:
                    return 1.0
                else:
                    return max(0, goal.progress / expected_progress)
        
        return 0.7  # No deadline to compare
    
    def _assess_timeline_health(self, goal: Goal) -> float:
        """Valuta salute della timeline"""
        if not goal.deadline:
            return 0.8  # No deadline
        
        if goal.is_overdue():
            return 0.0
        
        days = goal.days_until_deadline()
        if days is None:
            return 0.8
        
        if days > 30:
            return 1.0
        elif days > 7:
            return 0.8
        elif days > 1:
            return 0.5
        else:
            return 0.3
    
    def _assess_dependency_health(self, goal: Goal) -> float:
        """Valuta salute delle dipendenze"""
        if not goal.depends_on:
            return 1.0
        
        completed = sum(1 for d in goal.depends_on 
                       if d in self.goals and 
                       self.goals[d].status == GoalStatus.COMPLETED)
        
        return completed / len(goal.depends_on)
    
    def _assess_metric_health(self, goal: Goal) -> float:
        """Valuta salute delle metriche"""
        if not goal.metrics:
            return 0.7  # No metrics
        
        achieved = sum(1 for m in goal.metrics if m.is_achieved())
        return achieved / len(goal.metrics)
    
    def _generate_recommendations(self, goal: Goal,
                                   indicators: Dict) -> List[GoalRecommendation]:
        """Genera raccomandazioni per obiettivo"""
        recommendations = []
        
        # Progress issues
        if indicators["progress_health"] < 0.5:
            recommendations.append(GoalRecommendation(
                goal_id=goal.id,
                recommendation_type="progress",
                title="Accelerare progresso",
                description="Il progresso è sotto le aspettative. Considera di allocare più risorse.",
                priority=8,
                impact=0.8,
                effort=0.6
            ))
        
        # Timeline issues
        if indicators["timeline_health"] < 0.5:
            recommendations.append(GoalRecommendation(
                goal_id=goal.id,
                recommendation_type="timeline",
                title="Rivedere deadline",
                description="La deadline è a rischio. Valuta estensione o riduzione scope.",
                priority=9,
                impact=0.7,
                effort=0.3
            ))
        
        # Dependency issues
        if indicators["dependency_health"] < 0.5:
            recommendations.append(GoalRecommendation(
                goal_id=goal.id,
                recommendation_type="dependency",
                title="Sbloccare dipendenze",
                description="Ci sono dipendenze non completate. Prioritizza il loro completamento.",
                priority=7,
                impact=0.9,
                effort=0.7
            ))
        
        # Metric issues
        if indicators["metric_health"] < 0.5:
            failing_metrics = [m.name for m in goal.metrics if not m.is_achieved()]
            recommendations.append(GoalRecommendation(
                goal_id=goal.id,
                recommendation_type="metrics",
                title="Migliorare metriche",
                description=f"Metriche sotto target: {', '.join(failing_metrics)}",
                priority=6,
                impact=0.6,
                effort=0.5
            ))
        
        return sorted(recommendations, key=lambda r: r.priority, reverse=True)
    
    # === Goal Queries ===
    
    def get_goal(self, goal_id: str) -> Optional[Dict]:
        """Ottiene obiettivo per ID"""
        if goal_id in self.goals:
            return self.goals[goal_id].to_dict()
        return None
    
    def list_goals(self, status: GoalStatus = None,
                   priority: GoalPriority = None,
                   tags: List[str] = None) -> List[Dict]:
        """Lista obiettivi con filtri"""
        results = []
        
        for goal in self.goals.values():
            if status and goal.status != status:
                continue
            if priority and goal.priority != priority:
                continue
            if tags and not any(t in goal.tags for t in tags):
                continue
            
            results.append(goal.to_dict())
        
        return results
    
    def get_goal_tree(self, root_id: str = None) -> Dict:
        """Ottiene albero obiettivi"""
        def build_tree(goal_id: str) -> Dict:
            goal = self.goals.get(goal_id)
            if not goal:
                return None
            
            return {
                "id": goal.id,
                "name": goal.name,
                "status": goal.status.value,
                "progress": goal.progress,
                "children": [build_tree(sid) for sid in goal.sub_goals if sid in self.goals]
            }
        
        if root_id:
            return build_tree(root_id)
        
        # Find root goals (no parent)
        roots = [g for g in self.goals.values() if not g.parent_id]
        return {
            "roots": [build_tree(g.id) for g in roots]
        }
    
    def get_active_goals(self) -> List[Dict]:
        """Ottiene obiettivi attivi prioritizzati"""
        active = [g for g in self.goals.values() 
                 if g.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]]
        prioritized = self.prioritize_goals(active)
        return [g.to_dict() for g in prioritized]
    
    def get_overdue_goals(self) -> List[Dict]:
        """Ottiene obiettivi in ritardo"""
        return [g.to_dict() for g in self.goals.values() if g.is_overdue()]
    
    def get_blocked_goals(self) -> List[Dict]:
        """Ottiene obiettivi bloccati"""
        blocked = []
        
        for goal in self.goals.values():
            if goal.status == GoalStatus.BLOCKED:
                blocked.append(goal.to_dict())
            elif goal.depends_on:
                # Check if blocked by incomplete deps
                incomplete_deps = [d for d in goal.depends_on 
                                  if d in self.goals and 
                                  self.goals[d].status != GoalStatus.COMPLETED]
                if incomplete_deps:
                    goal_dict = goal.to_dict()
                    goal_dict["blocked_by"] = incomplete_deps
                    blocked.append(goal_dict)
        
        return blocked
    
    # === Statistics ===
    
    def get_statistics(self) -> Dict:
        """Statistiche obiettivi"""
        total = len(self.goals)
        
        by_status = {}
        for status in GoalStatus:
            by_status[status.value] = sum(1 for g in self.goals.values() if g.status == status)
        
        by_priority = {}
        for priority in GoalPriority:
            by_priority[priority.value] = sum(1 for g in self.goals.values() if g.priority == priority)
        
        avg_progress = statistics.mean(g.progress for g in self.goals.values()) if self.goals else 0
        
        overdue = sum(1 for g in self.goals.values() if g.is_overdue())
        
        return {
            "total_goals": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "average_progress": round(avg_progress, 1),
            "overdue_count": overdue,
            "completed_count": len(self.completed_goals),
            "alignments_count": len(self.alignments),
            "conflicts": len(self.find_conflicts())
        }
    
    def get_status(self) -> Dict:
        """Stato del goal management"""
        return {
            "total_goals": len(self.goals),
            "active_goals": sum(1 for g in self.goals.values() 
                              if g.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]),
            "templates": list(self.goal_templates.keys()),
            "alignments": len(self.alignments)
        }
