# /backend/gideon/meta_cognition.py
"""
🔮 GIDEON 3.0 - Meta-Cognition
Valuta propri limiti, incertezze e capacità di auto-valutazione.
NON esegue azioni - fornisce solo analisi meta-cognitiva.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
import math
import logging

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Tipi di incertezza"""
    EPISTEMIC = "epistemic"         # Mancanza di conoscenza
    ALEATORIC = "aleatoric"         # Variabilità intrinseca
    MODEL = "model"                 # Limitazioni del modello
    DATA = "data"                   # Qualità/quantità dati
    CONTEXTUAL = "contextual"       # Dipende dal contesto


class LimitationType(Enum):
    """Tipi di limitazione"""
    KNOWLEDGE = "knowledge"         # Conoscenza limitata
    CAPABILITY = "capability"       # Capacità limitata
    RESOURCE = "resource"           # Risorse limitate
    TEMPORAL = "temporal"           # Vincoli temporali
    SCOPE = "scope"                 # Fuori scope
    COMPLEXITY = "complexity"       # Troppo complesso


class ConfidenceCalibration(Enum):
    """Calibrazione della confidenza"""
    OVERCONFIDENT = "overconfident"   # Troppo sicuro
    UNDERCONFIDENT = "underconfident" # Troppo insicuro
    WELL_CALIBRATED = "well_calibrated"  # Ben calibrato


class ReflectionType(Enum):
    """Tipi di riflessione"""
    PERFORMANCE = "performance"     # Su performance passata
    DECISION = "decision"           # Su decisioni prese
    PREDICTION = "prediction"       # Su previsioni fatte
    LIMITATION = "limitation"       # Su propri limiti
    IMPROVEMENT = "improvement"     # Su come migliorare


@dataclass
class UncertaintyAssessment:
    """Valutazione dell'incertezza"""
    domain: str
    uncertainty_type: UncertaintyType
    level: float  # 0-1
    
    # Sources
    sources: List[str] = field(default_factory=list)
    
    # Reducibility
    is_reducible: bool = True
    reduction_strategy: str = ""
    
    # Impact
    impact_on_decision: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "type": self.uncertainty_type.value,
            "level": round(self.level, 3),
            "sources": self.sources,
            "is_reducible": self.is_reducible,
            "reduction_strategy": self.reduction_strategy,
            "impact": round(self.impact_on_decision, 3)
        }


@dataclass
class LimitationAwareness:
    """Consapevolezza di una limitazione"""
    id: str
    limitation_type: LimitationType
    description: str
    
    # Severity
    severity: float = 0.5  # 0-1
    
    # Scope
    affected_domains: List[str] = field(default_factory=list)
    
    # Workarounds
    workarounds: List[str] = field(default_factory=list)
    mitigation_possible: bool = True
    
    # When triggered
    trigger_conditions: List[str] = field(default_factory=list)
    
    identified_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.limitation_type.value,
            "description": self.description,
            "severity": round(self.severity, 2),
            "affected_domains": self.affected_domains,
            "workarounds": self.workarounds,
            "mitigation_possible": self.mitigation_possible,
            "trigger_conditions": self.trigger_conditions
        }


@dataclass
class SelfReflection:
    """Riflessione su azione/decisione passata"""
    id: str
    reflection_type: ReflectionType
    subject: str  # Cosa si sta riflettendo
    
    # Assessment
    what_went_well: List[str] = field(default_factory=list)
    what_went_wrong: List[str] = field(default_factory=list)
    
    # Confidence calibration
    predicted_confidence: float = 0.0
    actual_outcome: float = 0.0
    calibration: ConfidenceCalibration = ConfidenceCalibration.WELL_CALIBRATED
    
    # Learnings
    lessons_learned: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    
    # Context
    context: Dict = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.reflection_type.value,
            "subject": self.subject,
            "what_went_well": self.what_went_well,
            "what_went_wrong": self.what_went_wrong,
            "predicted_confidence": round(self.predicted_confidence, 2),
            "actual_outcome": round(self.actual_outcome, 2),
            "calibration": self.calibration.value,
            "lessons_learned": self.lessons_learned,
            "improvements": self.improvements
        }


@dataclass
class KnowsAbout:
    """Rappresenta cosa il sistema sa di sapere/non sapere"""
    domain: str
    knowledge_level: float  # 0-1
    confidence_in_knowledge: float  # Meta-confidence
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Known unknowns
    known_unknowns: List[str] = field(default_factory=list)
    
    # Blind spots
    potential_blind_spots: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "knowledge_level": round(self.knowledge_level, 2),
            "confidence": round(self.confidence_in_knowledge, 2),
            "known_unknowns": self.known_unknowns,
            "potential_blind_spots": self.potential_blind_spots
        }


@dataclass
class MetaCognitiveState:
    """Stato meta-cognitivo complessivo"""
    # Confidence calibration
    overall_calibration: ConfidenceCalibration
    calibration_score: float  # -1 to 1 (0 = perfect)
    
    # Self-awareness
    self_awareness_score: float  # 0-1
    
    # Uncertainty
    total_uncertainty: float
    uncertainty_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Limitations
    active_limitations: List[str] = field(default_factory=list)
    
    # Recommendations
    meta_recommendations: List[str] = field(default_factory=list)
    
    assessed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "overall_calibration": self.overall_calibration.value,
            "calibration_score": round(self.calibration_score, 3),
            "self_awareness": round(self.self_awareness_score, 2),
            "total_uncertainty": round(self.total_uncertainty, 2),
            "uncertainty_breakdown": self.uncertainty_breakdown,
            "active_limitations": self.active_limitations,
            "recommendations": self.meta_recommendations
        }


class MetaCognition:
    """
    Sistema di meta-cognizione per Gideon.
    Valuta propri limiti, incertezze e capacità.
    """
    
    def __init__(self):
        # Counters (must be before _register_default_limitations)
        self._reflection_counter = 0
        self._limitation_counter = 0
        
        # Knowledge tracking
        self.knowledge_domains: Dict[str, KnowsAbout] = {}
        
        # Limitations
        self.limitations: Dict[str, LimitationAwareness] = {}
        self._register_default_limitations()
        
        # Reflections history
        self.reflections: List[SelfReflection] = []
        
        # Prediction tracking (for calibration)
        self.predictions: List[Dict] = []  # {predicted, actual, domain}
        
        # Uncertainty models
        self.uncertainty_sources: Dict[str, float] = {
            "data_quality": 0.1,
            "model_accuracy": 0.15,
            "context_completeness": 0.2,
            "temporal_relevance": 0.1,
            "external_factors": 0.25
        }
    
    def _register_default_limitations(self):
        """Registra limitazioni note del sistema"""
        default_limits = [
            {
                "type": LimitationType.KNOWLEDGE,
                "description": "Conoscenza limitata a dati di training",
                "severity": 0.6,
                "domains": ["real_time_events", "personal_context"],
                "workarounds": ["Richiedere contesto aggiuntivo", "Indicare incertezza"]
            },
            {
                "type": LimitationType.CAPABILITY,
                "description": "Non può eseguire azioni nel mondo reale direttamente",
                "severity": 0.7,
                "domains": ["physical_actions", "external_systems"],
                "workarounds": ["Delegare a JARVIS", "Suggerire azioni"]
            },
            {
                "type": LimitationType.TEMPORAL,
                "description": "Conoscenza non aggiornata in tempo reale",
                "severity": 0.5,
                "domains": ["current_events", "live_data"],
                "workarounds": ["Richiedere dati aggiornati", "Indicare timestamp"]
            },
            {
                "type": LimitationType.COMPLEXITY,
                "description": "Difficoltà con problemi altamente non-lineari",
                "severity": 0.4,
                "domains": ["chaotic_systems", "multi_variable_optimization"],
                "workarounds": ["Scomporre in sotto-problemi", "Usare simulazioni"]
            },
            {
                "type": LimitationType.SCOPE,
                "description": "Non può garantire risultati fuori dal dominio addestrato",
                "severity": 0.6,
                "domains": ["novel_domains", "edge_cases"],
                "workarounds": ["Indicare bassa confidenza", "Suggerire verifica umana"]
            }
        ]
        
        for i, limit_data in enumerate(default_limits):
            self._limitation_counter += 1
            limitation = LimitationAwareness(
                id=f"limit_{self._limitation_counter}",
                limitation_type=limit_data["type"],
                description=limit_data["description"],
                severity=limit_data["severity"],
                affected_domains=limit_data["domains"],
                workarounds=limit_data["workarounds"]
            )
            self.limitations[limitation.id] = limitation
    
    # === Uncertainty Assessment ===
    
    def assess_uncertainty(self, domain: str, context: Dict = None) -> UncertaintyAssessment:
        """
        Valuta l'incertezza in un dominio.
        
        Args:
            domain: Dominio da valutare
            context: Contesto aggiuntivo
        
        Returns:
            UncertaintyAssessment
        """
        context = context or {}
        
        # Calculate uncertainty components
        epistemic = self._assess_epistemic_uncertainty(domain, context)
        aleatoric = self._assess_aleatoric_uncertainty(domain, context)
        model = self._assess_model_uncertainty(domain, context)
        
        # Determine primary type
        uncertainties = {
            UncertaintyType.EPISTEMIC: epistemic,
            UncertaintyType.ALEATORIC: aleatoric,
            UncertaintyType.MODEL: model
        }
        
        primary_type = max(uncertainties.keys(), key=lambda k: uncertainties[k])
        total_uncertainty = statistics.mean(uncertainties.values())
        
        # Identify sources
        sources = self._identify_uncertainty_sources(domain, context)
        
        # Is it reducible?
        is_reducible = primary_type != UncertaintyType.ALEATORIC
        
        # Reduction strategy
        strategy = self._get_reduction_strategy(primary_type, domain)
        
        return UncertaintyAssessment(
            domain=domain,
            uncertainty_type=primary_type,
            level=total_uncertainty,
            sources=sources,
            is_reducible=is_reducible,
            reduction_strategy=strategy,
            impact_on_decision=total_uncertainty * 0.7
        )
    
    def _assess_epistemic_uncertainty(self, domain: str, context: Dict) -> float:
        """Valuta incertezza epistemica (mancanza di conoscenza)"""
        # Check knowledge level
        knowledge = self.knowledge_domains.get(domain)
        if not knowledge:
            return 0.7  # Unknown domain
        
        return 1 - knowledge.knowledge_level
    
    def _assess_aleatoric_uncertainty(self, domain: str, context: Dict) -> float:
        """Valuta incertezza aleatoria (variabilità intrinseca)"""
        # Some domains are inherently uncertain
        high_variance_domains = ["user_behavior", "market", "weather", "network"]
        
        if any(d in domain.lower() for d in high_variance_domains):
            return 0.6
        
        return 0.2  # Default low aleatoric uncertainty
    
    def _assess_model_uncertainty(self, domain: str, context: Dict) -> float:
        """Valuta incertezza del modello"""
        # Based on prediction history
        domain_predictions = [p for p in self.predictions if p.get("domain") == domain]
        
        if len(domain_predictions) < 5:
            return 0.4  # Insufficient data
        
        # Calculate error variance
        errors = [abs(p["predicted"] - p["actual"]) for p in domain_predictions]
        return min(1.0, statistics.stdev(errors) if len(errors) > 1 else 0.3)
    
    def _identify_uncertainty_sources(self, domain: str, context: Dict) -> List[str]:
        """Identifica fonti di incertezza"""
        sources = []
        
        # Data quality
        if context.get("data_quality", 0.8) < 0.7:
            sources.append("Qualità dati bassa")
        
        # Completeness
        if context.get("completeness", 0.8) < 0.7:
            sources.append("Informazioni incomplete")
        
        # Novelty
        if domain not in self.knowledge_domains:
            sources.append("Dominio non familiare")
        
        # Temporal
        if context.get("data_age_hours", 0) > 24:
            sources.append("Dati non recenti")
        
        # External factors
        if context.get("external_dependencies", []):
            sources.append("Dipendenze esterne non controllabili")
        
        return sources
    
    def _get_reduction_strategy(self, uncertainty_type: UncertaintyType, domain: str) -> str:
        """Strategia per ridurre l'incertezza"""
        strategies = {
            UncertaintyType.EPISTEMIC: "Raccogliere più informazioni sul dominio",
            UncertaintyType.ALEATORIC: "Aumentare campionamento o usare intervalli",
            UncertaintyType.MODEL: "Calibrare modello con più esempi",
            UncertaintyType.DATA: "Migliorare qualità e quantità dati",
            UncertaintyType.CONTEXTUAL: "Richiedere più contesto"
        }
        return strategies.get(uncertainty_type, "Analisi approfondita richiesta")
    
    # === Limitation Awareness ===
    
    def check_limitations(self, task: Dict) -> List[LimitationAwareness]:
        """
        Verifica quali limitazioni si applicano a un task.
        """
        applicable = []
        
        task_domains = task.get("domains", [])
        task_type = task.get("type", "")
        
        for limitation in self.limitations.values():
            # Check domain overlap
            if any(d in limitation.affected_domains for d in task_domains):
                applicable.append(limitation)
                continue
            
            # Check trigger conditions
            for trigger in limitation.trigger_conditions:
                if trigger in task_type.lower():
                    applicable.append(limitation)
                    break
        
        return applicable
    
    def acknowledge_limitation(self, task: Dict) -> Dict:
        """
        Riconosce limitazioni per un task e suggerisce workaround.
        """
        limitations = self.check_limitations(task)
        
        if not limitations:
            return {
                "has_limitations": False,
                "can_proceed": True,
                "confidence_modifier": 1.0
            }
        
        # Calculate confidence modifier
        total_severity = sum(l.severity for l in limitations)
        confidence_modifier = max(0.3, 1 - total_severity * 0.3)
        
        # Collect workarounds
        workarounds = []
        for l in limitations:
            workarounds.extend(l.workarounds)
        
        return {
            "has_limitations": True,
            "can_proceed": all(l.mitigation_possible for l in limitations),
            "confidence_modifier": confidence_modifier,
            "limitations": [l.to_dict() for l in limitations],
            "suggested_workarounds": list(set(workarounds)),
            "disclaimer": self._generate_disclaimer(limitations)
        }
    
    def _generate_disclaimer(self, limitations: List[LimitationAwareness]) -> str:
        """Genera disclaimer basato su limitazioni"""
        if not limitations:
            return ""
        
        primary = max(limitations, key=lambda l: l.severity)
        
        disclaimers = {
            LimitationType.KNOWLEDGE: "Nota: La mia conoscenza in questo dominio potrebbe essere limitata.",
            LimitationType.CAPABILITY: "Nota: Non posso eseguire questa azione direttamente.",
            LimitationType.TEMPORAL: "Nota: Le informazioni potrebbero non essere aggiornate.",
            LimitationType.COMPLEXITY: "Nota: Questo problema è complesso e la soluzione potrebbe essere approssimata.",
            LimitationType.SCOPE: "Nota: Questo task è ai limiti delle mie capacità."
        }
        
        return disclaimers.get(primary.limitation_type, 
                              "Nota: Potrebbero esserci limitazioni in questa analisi.")
    
    # === Self-Reflection ===
    
    def reflect(self, subject: str, reflection_type: ReflectionType,
                predicted_confidence: float, actual_outcome: float,
                context: Dict = None) -> SelfReflection:
        """
        Esegue una riflessione su un'azione/decisione passata.
        """
        self._reflection_counter += 1
        
        # Determine calibration
        diff = predicted_confidence - actual_outcome
        if diff > 0.15:
            calibration = ConfidenceCalibration.OVERCONFIDENT
        elif diff < -0.15:
            calibration = ConfidenceCalibration.UNDERCONFIDENT
        else:
            calibration = ConfidenceCalibration.WELL_CALIBRATED
        
        # Analyze what went well/wrong
        went_well, went_wrong = self._analyze_outcome(
            predicted_confidence, actual_outcome, context
        )
        
        # Generate lessons
        lessons = self._extract_lessons(calibration, went_wrong, context)
        
        # Suggest improvements
        improvements = self._suggest_improvements(calibration, lessons)
        
        reflection = SelfReflection(
            id=f"refl_{self._reflection_counter}",
            reflection_type=reflection_type,
            subject=subject,
            what_went_well=went_well,
            what_went_wrong=went_wrong,
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_outcome,
            calibration=calibration,
            lessons_learned=lessons,
            improvements=improvements,
            context=context or {}
        )
        
        self.reflections.append(reflection)
        
        # Track prediction for calibration
        domain = (context or {}).get("domain", "general")
        self.predictions.append({
            "predicted": predicted_confidence,
            "actual": actual_outcome,
            "domain": domain,
            "timestamp": datetime.now()
        })
        
        return reflection
    
    def _analyze_outcome(self, predicted: float, actual: float,
                         context: Dict) -> Tuple[List[str], List[str]]:
        """Analizza cosa è andato bene/male"""
        went_well = []
        went_wrong = []
        
        if abs(predicted - actual) < 0.1:
            went_well.append("Previsione accurata")
        
        if actual >= 0.7:
            went_well.append("Outcome positivo raggiunto")
        elif actual < 0.3:
            went_wrong.append("Outcome significativamente negativo")
        
        if predicted > actual + 0.2:
            went_wrong.append("Eccesso di confidenza nella previsione")
        elif predicted < actual - 0.2:
            went_wrong.append("Sottostima delle possibilità di successo")
        
        # Context-based analysis
        if context:
            if context.get("time_pressure") and actual < 0.5:
                went_wrong.append("Pressione temporale ha impattato negativamente")
            if context.get("good_data_quality") and actual >= 0.7:
                went_well.append("Buona qualità dei dati ha supportato decisione")
        
        return went_well, went_wrong
    
    def _extract_lessons(self, calibration: ConfidenceCalibration,
                         went_wrong: List[str], context: Dict) -> List[str]:
        """Estrae lezioni dalla riflessione"""
        lessons = []
        
        if calibration == ConfidenceCalibration.OVERCONFIDENT:
            lessons.append("Necessario essere più cauti nelle previsioni")
            lessons.append("Considerare più scenari negativi")
        elif calibration == ConfidenceCalibration.UNDERCONFIDENT:
            lessons.append("Posso fidarmi di più delle mie analisi")
            lessons.append("L'incertezza era sovrastimata")
        
        for issue in went_wrong:
            if "confidenza" in issue.lower():
                lessons.append("Rivedere criteri di confidence scoring")
            if "temporale" in issue.lower():
                lessons.append("Allocare più tempo per analisi complesse")
        
        return lessons
    
    def _suggest_improvements(self, calibration: ConfidenceCalibration,
                              lessons: List[str]) -> List[str]:
        """Suggerisce miglioramenti"""
        improvements = []
        
        if calibration == ConfidenceCalibration.OVERCONFIDENT:
            improvements.append("Applicare margine di sicurezza alle previsioni")
            improvements.append("Richiedere validazione esterna per alta confidenza")
        elif calibration == ConfidenceCalibration.UNDERCONFIDENT:
            improvements.append("Ridurre penalità per incertezza in domini noti")
        
        improvements.append("Continuare a tracciare previsioni per calibrazione")
        
        return improvements
    
    # === Meta-Cognitive State ===
    
    def get_meta_state(self) -> MetaCognitiveState:
        """
        Ottiene stato meta-cognitivo complessivo.
        """
        # Calculate calibration
        calibration, calibration_score = self._calculate_calibration()
        
        # Self-awareness score
        awareness = self._calculate_self_awareness()
        
        # Total uncertainty
        total_uncertainty = self._calculate_total_uncertainty()
        
        # Active limitations
        active_limits = [l.id for l in self.limitations.values() if l.severity > 0.5]
        
        # Generate meta-recommendations
        recommendations = self._generate_meta_recommendations(
            calibration, awareness, total_uncertainty
        )
        
        return MetaCognitiveState(
            overall_calibration=calibration,
            calibration_score=calibration_score,
            self_awareness_score=awareness,
            total_uncertainty=total_uncertainty,
            uncertainty_breakdown={s: v for s, v in self.uncertainty_sources.items()},
            active_limitations=active_limits,
            meta_recommendations=recommendations
        )
    
    def _calculate_calibration(self) -> Tuple[ConfidenceCalibration, float]:
        """Calcola calibrazione complessiva"""
        if len(self.predictions) < 5:
            return ConfidenceCalibration.WELL_CALIBRATED, 0.0
        
        errors = [p["predicted"] - p["actual"] for p in self.predictions[-50:]]
        mean_error = statistics.mean(errors)
        
        if mean_error > 0.1:
            return ConfidenceCalibration.OVERCONFIDENT, mean_error
        elif mean_error < -0.1:
            return ConfidenceCalibration.UNDERCONFIDENT, mean_error
        else:
            return ConfidenceCalibration.WELL_CALIBRATED, mean_error
    
    def _calculate_self_awareness(self) -> float:
        """Calcola livello di auto-consapevolezza"""
        factors = []
        
        # Based on reflections
        if self.reflections:
            recent_reflections = len([r for r in self.reflections 
                                     if (datetime.now() - r.created_at).days < 7])
            factors.append(min(1.0, recent_reflections / 10))
        else:
            factors.append(0.3)
        
        # Based on known limitations
        factors.append(min(1.0, len(self.limitations) / 5))
        
        # Based on knowledge tracking
        factors.append(min(1.0, len(self.knowledge_domains) / 10))
        
        return statistics.mean(factors)
    
    def _calculate_total_uncertainty(self) -> float:
        """Calcola incertezza totale"""
        return sum(self.uncertainty_sources.values()) / len(self.uncertainty_sources)
    
    def _generate_meta_recommendations(self, calibration: ConfidenceCalibration,
                                        awareness: float,
                                        uncertainty: float) -> List[str]:
        """Genera raccomandazioni meta-cognitive"""
        recommendations = []
        
        if calibration == ConfidenceCalibration.OVERCONFIDENT:
            recommendations.append("Considerare scenari alternativi più frequentemente")
        elif calibration == ConfidenceCalibration.UNDERCONFIDENT:
            recommendations.append("Fidarsi di più delle proprie analisi in domini noti")
        
        if awareness < 0.5:
            recommendations.append("Aumentare frequenza di auto-riflessione")
        
        if uncertainty > 0.5:
            recommendations.append("Raccogliere più dati prima di decisioni importanti")
        
        recommendations.append("Mantenere tracciamento delle previsioni per calibrazione")
        
        return recommendations
    
    # === Knowledge Tracking ===
    
    def update_knowledge(self, domain: str, knowledge_level: float,
                         known_unknowns: List[str] = None):
        """Aggiorna conoscenza su un dominio"""
        if domain in self.knowledge_domains:
            existing = self.knowledge_domains[domain]
            # Smooth update
            existing.knowledge_level = existing.knowledge_level * 0.7 + knowledge_level * 0.3
            existing.last_updated = datetime.now()
            if known_unknowns:
                existing.known_unknowns.extend(known_unknowns)
        else:
            self.knowledge_domains[domain] = KnowsAbout(
                domain=domain,
                knowledge_level=knowledge_level,
                confidence_in_knowledge=0.6,  # Initial meta-confidence
                known_unknowns=known_unknowns or []
            )
    
    def what_i_know(self, domain: str = None) -> Dict:
        """Riporta cosa il sistema sa di sapere"""
        if domain:
            knowledge = self.knowledge_domains.get(domain)
            if knowledge:
                return knowledge.to_dict()
            return {"domain": domain, "knowledge_level": 0, "status": "unknown"}
        
        return {
            "domains": [k.to_dict() for k in self.knowledge_domains.values()],
            "total_domains": len(self.knowledge_domains),
            "average_knowledge": statistics.mean(
                k.knowledge_level for k in self.knowledge_domains.values()
            ) if self.knowledge_domains else 0
        }
    
    def what_i_dont_know(self) -> Dict:
        """Riporta cosa il sistema sa di NON sapere"""
        known_unknowns = []
        blind_spots = []
        
        for knowledge in self.knowledge_domains.values():
            known_unknowns.extend([{
                "domain": knowledge.domain,
                "unknown": u
            } for u in knowledge.known_unknowns])
            
            blind_spots.extend([{
                "domain": knowledge.domain,
                "blind_spot": b
            } for b in knowledge.potential_blind_spots])
        
        return {
            "known_unknowns": known_unknowns,
            "potential_blind_spots": blind_spots,
            "limitations": [l.to_dict() for l in self.limitations.values()]
        }
    
    # === Status ===
    
    def get_status(self) -> Dict:
        """Stato del sistema meta-cognitivo"""
        calibration, score = self._calculate_calibration()
        
        return {
            "knowledge_domains": len(self.knowledge_domains),
            "known_limitations": len(self.limitations),
            "reflections_count": len(self.reflections),
            "predictions_tracked": len(self.predictions),
            "calibration": calibration.value,
            "calibration_score": round(score, 3),
            "self_awareness": round(self._calculate_self_awareness(), 2)
        }
