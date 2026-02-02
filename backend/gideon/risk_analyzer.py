# /backend/gideon/risk_analyzer.py
"""
🔮 GIDEON 3.0 - Risk Analyzer
Analizza rischi e propone mitigazioni per decisioni e azioni.
NON esegue azioni - fornisce solo analisi e raccomandazioni.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Livelli di rischio"""
    NEGLIGIBLE = "negligible"   # Trascurabile
    LOW = "low"                 # Basso
    MODERATE = "moderate"       # Moderato
    HIGH = "high"               # Alto
    CRITICAL = "critical"       # Critico
    CATASTROPHIC = "catastrophic"  # Catastrofico


class RiskCategory(Enum):
    """Categorie di rischio"""
    TECHNICAL = "technical"         # Errori tecnici, bug
    OPERATIONAL = "operational"     # Problemi operativi
    SECURITY = "security"           # Rischi sicurezza
    DATA = "data"                   # Perdita/corruzione dati
    PERFORMANCE = "performance"     # Degradazione performance
    AVAILABILITY = "availability"   # Disponibilità sistema
    COMPLIANCE = "compliance"       # Conformità normativa
    FINANCIAL = "financial"         # Impatto economico
    REPUTATIONAL = "reputational"   # Danno reputazionale
    USER_EXPERIENCE = "user_experience"  # Impatto utente


class MitigationType(Enum):
    """Tipi di mitigazione"""
    AVOID = "avoid"           # Evita il rischio
    REDUCE = "reduce"         # Riduci probabilità/impatto
    TRANSFER = "transfer"     # Trasferisci (es. backup)
    ACCEPT = "accept"         # Accetta con monitoraggio
    CONTINGENCY = "contingency"  # Piano di emergenza


@dataclass
class Risk:
    """Definizione di un rischio"""
    id: str
    name: str
    description: str
    category: RiskCategory
    
    # Valutazione
    probability: float          # 0.0 - 1.0
    impact: float               # 0.0 - 1.0 (severità)
    detectability: float        # 0.0 - 1.0 (facilità di rilevamento)
    
    # Calcolati
    risk_score: float = 0.0     # probability * impact
    rpn: float = 0.0            # Risk Priority Number
    level: RiskLevel = RiskLevel.LOW
    
    # Context
    triggers: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    
    # Timing
    identified_at: datetime = field(default_factory=datetime.now)
    
    def calculate_scores(self):
        """Calcola score e RPN"""
        self.risk_score = self.probability * self.impact
        # RPN = Severity * Occurrence * Detection (1-10 scale mapped)
        self.rpn = self.impact * 10 * self.probability * 10 * (1 - self.detectability) * 10
        self.level = self._determine_level()
    
    def _determine_level(self) -> RiskLevel:
        """Determina livello rischio"""
        if self.risk_score < 0.1:
            return RiskLevel.NEGLIGIBLE
        elif self.risk_score < 0.25:
            return RiskLevel.LOW
        elif self.risk_score < 0.50:
            return RiskLevel.MODERATE
        elif self.risk_score < 0.75:
            return RiskLevel.HIGH
        elif self.risk_score < 0.90:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.CATASTROPHIC
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "probability": round(self.probability, 3),
            "impact": round(self.impact, 3),
            "detectability": round(self.detectability, 3),
            "risk_score": round(self.risk_score, 3),
            "rpn": round(self.rpn, 1),
            "level": self.level.value,
            "triggers": self.triggers,
            "affected_components": self.affected_components
        }


@dataclass
class Mitigation:
    """Strategia di mitigazione"""
    id: str
    risk_id: str
    name: str
    description: str
    mitigation_type: MitigationType
    
    # Efficacia
    probability_reduction: float = 0.0   # Riduzione probabilità
    impact_reduction: float = 0.0        # Riduzione impatto
    effectiveness: float = 0.0           # Overall effectiveness
    
    # Costi
    implementation_cost: float = 0.0     # 0-1 (relativo)
    maintenance_cost: float = 0.0
    time_to_implement: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    # Priority
    priority: int = 5                    # 1-10
    
    # Actions
    steps: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    def calculate_effectiveness(self):
        """Calcola efficacia complessiva"""
        self.effectiveness = (self.probability_reduction + self.impact_reduction) / 2
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "risk_id": self.risk_id,
            "name": self.name,
            "description": self.description,
            "type": self.mitigation_type.value,
            "probability_reduction": round(self.probability_reduction, 3),
            "impact_reduction": round(self.impact_reduction, 3),
            "effectiveness": round(self.effectiveness, 3),
            "implementation_cost": round(self.implementation_cost, 3),
            "time_to_implement_hours": self.time_to_implement.total_seconds() / 3600,
            "priority": self.priority,
            "steps": self.steps
        }


@dataclass
class RiskAssessment:
    """Valutazione rischio completa"""
    scenario_id: str
    overall_risk_level: RiskLevel
    overall_risk_score: float
    
    risks: List[Risk] = field(default_factory=list)
    mitigations: List[Mitigation] = field(default_factory=list)
    
    # Summary
    total_risks: int = 0
    critical_risks: int = 0
    mitigated_risks: int = 0
    residual_risk: float = 0.0
    
    # Recommendation
    proceed_recommendation: str = ""
    confidence: float = 0.0
    
    assessed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "overall_risk_level": self.overall_risk_level.value,
            "overall_risk_score": round(self.overall_risk_score, 3),
            "risks": [r.to_dict() for r in self.risks],
            "mitigations": [m.to_dict() for m in self.mitigations],
            "summary": {
                "total_risks": self.total_risks,
                "critical_risks": self.critical_risks,
                "mitigated_risks": self.mitigated_risks,
                "residual_risk": round(self.residual_risk, 3)
            },
            "recommendation": self.proceed_recommendation,
            "confidence": round(self.confidence, 3),
            "assessed_at": self.assessed_at.isoformat()
        }


class RiskAnalyzer:
    """
    Analizzatore di rischi avanzato per Gideon.
    Identifica, valuta e propone mitigazioni per i rischi.
    """
    
    def __init__(self):
        # Risk templates
        self.risk_templates: Dict[str, Dict] = {}
        self._register_default_templates()
        
        # Mitigation strategies
        self.mitigation_strategies: Dict[RiskCategory, List[Dict]] = {}
        self._register_default_strategies()
        
        # History
        self.assessment_history: List[RiskAssessment] = []
        
        # Thresholds
        self.risk_thresholds = {
            RiskLevel.NEGLIGIBLE: 0.1,
            RiskLevel.LOW: 0.25,
            RiskLevel.MODERATE: 0.50,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 0.90
        }
        
        # Counter
        self._risk_counter = 0
        self._mitigation_counter = 0
    
    def _register_default_templates(self):
        """Registra template rischi comuni"""
        self.risk_templates = {
            "system_failure": {
                "name": "System Failure",
                "category": RiskCategory.TECHNICAL,
                "base_probability": 0.1,
                "base_impact": 0.8,
                "detectability": 0.7,
                "triggers": ["resource_exhaustion", "bug", "dependency_failure"],
                "components": ["core_system"]
            },
            "data_loss": {
                "name": "Data Loss",
                "category": RiskCategory.DATA,
                "base_probability": 0.05,
                "base_impact": 0.95,
                "detectability": 0.5,
                "triggers": ["disk_failure", "corruption", "accidental_deletion"],
                "components": ["database", "file_system"]
            },
            "security_breach": {
                "name": "Security Breach",
                "category": RiskCategory.SECURITY,
                "base_probability": 0.15,
                "base_impact": 0.9,
                "detectability": 0.4,
                "triggers": ["vulnerability", "misconfiguration", "social_engineering"],
                "components": ["auth", "network", "api"]
            },
            "performance_degradation": {
                "name": "Performance Degradation",
                "category": RiskCategory.PERFORMANCE,
                "base_probability": 0.3,
                "base_impact": 0.4,
                "detectability": 0.8,
                "triggers": ["high_load", "memory_leak", "slow_queries"],
                "components": ["api", "database"]
            },
            "service_unavailability": {
                "name": "Service Unavailability",
                "category": RiskCategory.AVAILABILITY,
                "base_probability": 0.2,
                "base_impact": 0.7,
                "detectability": 0.9,
                "triggers": ["crash", "network_issue", "overload"],
                "components": ["web_server", "api"]
            },
            "user_error": {
                "name": "User Error",
                "category": RiskCategory.OPERATIONAL,
                "base_probability": 0.4,
                "base_impact": 0.3,
                "detectability": 0.6,
                "triggers": ["misclick", "wrong_input", "confusion"],
                "components": ["ui", "workflows"]
            },
            "api_failure": {
                "name": "External API Failure",
                "category": RiskCategory.TECHNICAL,
                "base_probability": 0.25,
                "base_impact": 0.5,
                "detectability": 0.85,
                "triggers": ["timeout", "rate_limit", "api_change"],
                "components": ["integrations"]
            },
            "resource_exhaustion": {
                "name": "Resource Exhaustion",
                "category": RiskCategory.PERFORMANCE,
                "base_probability": 0.2,
                "base_impact": 0.6,
                "detectability": 0.75,
                "triggers": ["memory_full", "cpu_spike", "disk_full"],
                "components": ["system"]
            }
        }
    
    def _register_default_strategies(self):
        """Registra strategie di mitigazione default"""
        self.mitigation_strategies = {
            RiskCategory.TECHNICAL: [
                {"name": "Automated Testing", "type": MitigationType.REDUCE,
                 "prob_red": 0.4, "impact_red": 0.2, "cost": 0.3},
                {"name": "Code Review", "type": MitigationType.REDUCE,
                 "prob_red": 0.3, "impact_red": 0.1, "cost": 0.2},
                {"name": "Rollback Capability", "type": MitigationType.CONTINGENCY,
                 "prob_red": 0.0, "impact_red": 0.6, "cost": 0.4}
            ],
            RiskCategory.DATA: [
                {"name": "Automated Backup", "type": MitigationType.TRANSFER,
                 "prob_red": 0.1, "impact_red": 0.8, "cost": 0.3},
                {"name": "Data Validation", "type": MitigationType.REDUCE,
                 "prob_red": 0.5, "impact_red": 0.2, "cost": 0.2},
                {"name": "Replication", "type": MitigationType.TRANSFER,
                 "prob_red": 0.2, "impact_red": 0.7, "cost": 0.5}
            ],
            RiskCategory.SECURITY: [
                {"name": "Access Control", "type": MitigationType.REDUCE,
                 "prob_red": 0.5, "impact_red": 0.3, "cost": 0.3},
                {"name": "Encryption", "type": MitigationType.REDUCE,
                 "prob_red": 0.3, "impact_red": 0.5, "cost": 0.4},
                {"name": "Security Audit", "type": MitigationType.REDUCE,
                 "prob_red": 0.4, "impact_red": 0.2, "cost": 0.5}
            ],
            RiskCategory.PERFORMANCE: [
                {"name": "Caching", "type": MitigationType.REDUCE,
                 "prob_red": 0.4, "impact_red": 0.3, "cost": 0.2},
                {"name": "Load Balancing", "type": MitigationType.REDUCE,
                 "prob_red": 0.3, "impact_red": 0.4, "cost": 0.4},
                {"name": "Performance Monitoring", "type": MitigationType.REDUCE,
                 "prob_red": 0.2, "impact_red": 0.2, "cost": 0.2}
            ],
            RiskCategory.AVAILABILITY: [
                {"name": "Redundancy", "type": MitigationType.TRANSFER,
                 "prob_red": 0.5, "impact_red": 0.6, "cost": 0.6},
                {"name": "Health Checks", "type": MitigationType.REDUCE,
                 "prob_red": 0.3, "impact_red": 0.3, "cost": 0.2},
                {"name": "Failover System", "type": MitigationType.CONTINGENCY,
                 "prob_red": 0.2, "impact_red": 0.7, "cost": 0.5}
            ],
            RiskCategory.OPERATIONAL: [
                {"name": "User Training", "type": MitigationType.REDUCE,
                 "prob_red": 0.5, "impact_red": 0.2, "cost": 0.3},
                {"name": "Confirmation Dialogs", "type": MitigationType.REDUCE,
                 "prob_red": 0.4, "impact_red": 0.3, "cost": 0.1},
                {"name": "Undo Capability", "type": MitigationType.CONTINGENCY,
                 "prob_red": 0.1, "impact_red": 0.6, "cost": 0.3}
            ]
        }
    
    # === Risk Identification ===
    
    def identify_risks(self, scenario: Dict) -> List[Risk]:
        """
        Identifica rischi potenziali in uno scenario.
        
        Args:
            scenario: Descrizione dello scenario
        
        Returns:
            Lista di rischi identificati
        """
        risks = []
        
        # Analizza action type
        action_type = scenario.get("action_type", "generic")
        components = scenario.get("components", [])
        context = scenario.get("context", {})
        
        # Check each template
        for template_id, template in self.risk_templates.items():
            relevance = self._calculate_relevance(template, action_type, components, context)
            
            if relevance > 0.3:  # Threshold di rilevanza
                risk = self._create_risk_from_template(template_id, template, relevance, context)
                risks.append(risk)
        
        # Aggiungi rischi custom dal contesto
        custom_risks = context.get("custom_risks", [])
        for cr in custom_risks:
            risk = self._create_custom_risk(cr)
            risks.append(risk)
        
        # Ordina per score
        for r in risks:
            r.calculate_scores()
        
        risks.sort(key=lambda x: x.risk_score, reverse=True)
        
        return risks
    
    def _calculate_relevance(self, template: Dict, action_type: str,
                             components: List[str], context: Dict) -> float:
        """Calcola rilevanza di un template per lo scenario"""
        relevance = 0.0
        
        # Check components overlap
        template_components = template.get("components", [])
        if components:
            overlap = len(set(components) & set(template_components))
            relevance += overlap * 0.3
        
        # Check triggers
        template_triggers = template.get("triggers", [])
        scenario_triggers = context.get("triggers", [])
        if scenario_triggers:
            trigger_overlap = len(set(scenario_triggers) & set(template_triggers))
            relevance += trigger_overlap * 0.3
        
        # Base relevance per categoria
        category = template.get("category")
        if category == RiskCategory.TECHNICAL:
            relevance += 0.2  # Sempre un po' rilevante
        
        # Action-specific
        if "system" in action_type.lower():
            if category in [RiskCategory.TECHNICAL, RiskCategory.AVAILABILITY]:
                relevance += 0.2
        elif "data" in action_type.lower():
            if category == RiskCategory.DATA:
                relevance += 0.3
        elif "security" in action_type.lower():
            if category == RiskCategory.SECURITY:
                relevance += 0.3
        
        return min(1.0, relevance)
    
    def _create_risk_from_template(self, template_id: str, template: Dict,
                                    relevance: float, context: Dict) -> Risk:
        """Crea rischio da template"""
        self._risk_counter += 1
        
        # Adjust probability based on context
        base_prob = template["base_probability"]
        system_health = context.get("system_health", 0.8)
        complexity = context.get("complexity", 0.5)
        
        adjusted_prob = base_prob * (1 + (1 - system_health) + complexity * 0.5) * relevance
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))
        
        return Risk(
            id=f"risk_{self._risk_counter}",
            name=template["name"],
            description=f"Rischio: {template['name']} (rilevanza: {relevance:.2f})",
            category=template["category"],
            probability=adjusted_prob,
            impact=template["base_impact"],
            detectability=template["detectability"],
            triggers=template.get("triggers", []),
            affected_components=template.get("components", [])
        )
    
    def _create_custom_risk(self, risk_data: Dict) -> Risk:
        """Crea rischio custom"""
        self._risk_counter += 1
        
        return Risk(
            id=f"risk_{self._risk_counter}",
            name=risk_data.get("name", "Custom Risk"),
            description=risk_data.get("description", ""),
            category=RiskCategory(risk_data.get("category", "technical")),
            probability=risk_data.get("probability", 0.5),
            impact=risk_data.get("impact", 0.5),
            detectability=risk_data.get("detectability", 0.5)
        )
    
    # === Risk Assessment ===
    
    def assess(self, scenario: Dict) -> RiskAssessment:
        """
        Esegue valutazione rischio completa.
        
        Args:
            scenario: Scenario da valutare
        
        Returns:
            RiskAssessment completo
        """
        # Identifica rischi
        risks = self.identify_risks(scenario)
        
        # Genera mitigazioni
        mitigations = []
        for risk in risks:
            risk_mitigations = self.generate_mitigations(risk)
            mitigations.extend(risk_mitigations)
        
        # Calcola overall score
        if risks:
            overall_score = statistics.mean(r.risk_score for r in risks)
            max_score = max(r.risk_score for r in risks)
            # Peso extra per rischi critici
            overall_score = overall_score * 0.6 + max_score * 0.4
        else:
            overall_score = 0.0
        
        # Conta rischi critici
        critical_count = sum(1 for r in risks if r.level in [RiskLevel.CRITICAL, RiskLevel.CATASTROPHIC])
        
        # Calcola rischio residuo (dopo mitigazioni)
        residual = self._calculate_residual_risk(risks, mitigations)
        
        # Genera recommendation
        recommendation, confidence = self._generate_recommendation(overall_score, critical_count, residual)
        
        assessment = RiskAssessment(
            scenario_id=scenario.get("id", "unknown"),
            overall_risk_level=self._score_to_level(overall_score),
            overall_risk_score=overall_score,
            risks=risks,
            mitigations=mitigations,
            total_risks=len(risks),
            critical_risks=critical_count,
            mitigated_risks=len(set(m.risk_id for m in mitigations)),
            residual_risk=residual,
            proceed_recommendation=recommendation,
            confidence=confidence
        )
        
        # Store in history
        self.assessment_history.append(assessment)
        
        return assessment
    
    # === Mitigation Generation ===
    
    def generate_mitigations(self, risk: Risk) -> List[Mitigation]:
        """
        Genera strategie di mitigazione per un rischio.
        
        Args:
            risk: Rischio da mitigare
        
        Returns:
            Lista di mitigazioni
        """
        mitigations = []
        
        # Get strategies for category
        strategies = self.mitigation_strategies.get(risk.category, [])
        
        for strategy in strategies:
            # Skip se non efficace abbastanza
            min_effectiveness = 0.2
            effectiveness = (strategy["prob_red"] + strategy["impact_red"]) / 2
            if effectiveness < min_effectiveness:
                continue
            
            self._mitigation_counter += 1
            
            mitigation = Mitigation(
                id=f"mit_{self._mitigation_counter}",
                risk_id=risk.id,
                name=strategy["name"],
                description=f"Mitiga {risk.name} tramite {strategy['name']}",
                mitigation_type=strategy["type"],
                probability_reduction=strategy["prob_red"],
                impact_reduction=strategy["impact_red"],
                implementation_cost=strategy["cost"],
                steps=self._generate_steps(strategy, risk)
            )
            
            mitigation.calculate_effectiveness()
            mitigation.priority = self._calculate_priority(risk, mitigation)
            
            mitigations.append(mitigation)
        
        # Ordina per priority
        mitigations.sort(key=lambda x: x.priority, reverse=True)
        
        return mitigations
    
    def _generate_steps(self, strategy: Dict, risk: Risk) -> List[str]:
        """Genera step di implementazione"""
        steps = []
        
        mit_type = strategy["type"]
        
        if mit_type == MitigationType.REDUCE:
            steps = [
                f"Identificare cause principali di {risk.name}",
                f"Implementare {strategy['name']}",
                "Verificare efficacia della mitigazione",
                "Monitorare metriche correlate"
            ]
        elif mit_type == MitigationType.TRANSFER:
            steps = [
                f"Configurare {strategy['name']}",
                "Testare meccanismo di trasferimento",
                "Documentare procedura di recovery",
                "Schedulare test periodici"
            ]
        elif mit_type == MitigationType.CONTINGENCY:
            steps = [
                f"Definire trigger per {strategy['name']}",
                "Documentare procedura di emergenza",
                "Comunicare al team",
                "Testare piano periodicamente"
            ]
        elif mit_type == MitigationType.AVOID:
            steps = [
                f"Identificare alternative a {risk.affected_components}",
                "Valutare impatto del cambiamento",
                "Implementare soluzione alternativa",
                "Rimuovere componente rischioso"
            ]
        else:
            steps = [
                "Valutare rischio",
                "Documentare accettazione",
                "Monitorare continuamente"
            ]
        
        return steps
    
    def _calculate_priority(self, risk: Risk, mitigation: Mitigation) -> int:
        """Calcola priorità mitigazione"""
        # Higher risk = higher priority
        risk_factor = risk.risk_score * 5
        
        # Higher effectiveness = higher priority
        eff_factor = mitigation.effectiveness * 3
        
        # Lower cost = higher priority
        cost_factor = (1 - mitigation.implementation_cost) * 2
        
        priority = int(risk_factor + eff_factor + cost_factor)
        return max(1, min(10, priority))
    
    # === Risk Calculations ===
    
    def _calculate_residual_risk(self, risks: List[Risk],
                                  mitigations: List[Mitigation]) -> float:
        """Calcola rischio residuo dopo mitigazioni"""
        if not risks:
            return 0.0
        
        residual_scores = []
        
        for risk in risks:
            # Find mitigations for this risk
            risk_mits = [m for m in mitigations if m.risk_id == risk.id]
            
            if not risk_mits:
                residual_scores.append(risk.risk_score)
            else:
                # Apply best mitigation
                best_mit = max(risk_mits, key=lambda m: m.effectiveness)
                
                new_prob = risk.probability * (1 - best_mit.probability_reduction)
                new_impact = risk.impact * (1 - best_mit.impact_reduction)
                residual_scores.append(new_prob * new_impact)
        
        return statistics.mean(residual_scores)
    
    def _score_to_level(self, score: float) -> RiskLevel:
        """Converte score in livello"""
        if score < self.risk_thresholds[RiskLevel.NEGLIGIBLE]:
            return RiskLevel.NEGLIGIBLE
        elif score < self.risk_thresholds[RiskLevel.LOW]:
            return RiskLevel.LOW
        elif score < self.risk_thresholds[RiskLevel.MODERATE]:
            return RiskLevel.MODERATE
        elif score < self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif score < self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.CATASTROPHIC
    
    def _generate_recommendation(self, score: float, critical: int,
                                  residual: float) -> Tuple[str, float]:
        """Genera raccomandazione"""
        confidence = 0.7
        
        if critical > 0:
            if residual > 0.5:
                return "STOP - Rischi critici senza mitigazione adeguata", 0.9
            else:
                return "PROCEED_WITH_MITIGATIONS - Applicare tutte le mitigazioni", 0.8
        
        if score < 0.25:
            return "PROCEED - Rischio accettabile", confidence
        elif score < 0.50:
            return "PROCEED_WITH_MONITORING - Monitorare attentamente", confidence
        elif score < 0.75:
            return "EVALUATE - Valutare alternative o mitigazioni", confidence
        else:
            return "RECONSIDER - Ripianificare o abbandonare", confidence
    
    # === Analysis Methods ===
    
    def analyze_risk_matrix(self, risks: List[Risk]) -> Dict:
        """
        Genera matrice di rischio (probabilità x impatto).
        """
        matrix = {
            "low_low": [],
            "low_high": [],
            "high_low": [],
            "high_high": []
        }
        
        for risk in risks:
            if risk.probability < 0.5:
                if risk.impact < 0.5:
                    matrix["low_low"].append(risk.id)
                else:
                    matrix["low_high"].append(risk.id)
            else:
                if risk.impact < 0.5:
                    matrix["high_low"].append(risk.id)
                else:
                    matrix["high_high"].append(risk.id)
        
        return {
            "matrix": matrix,
            "priority_order": ["high_high", "low_high", "high_low", "low_low"],
            "counts": {k: len(v) for k, v in matrix.items()}
        }
    
    def compare_scenarios(self, scenarios: List[Dict]) -> Dict:
        """Confronta rischi tra scenari"""
        assessments = [self.assess(s) for s in scenarios]
        
        comparison = {
            "scenarios": [a.scenario_id for a in assessments],
            "risk_scores": {a.scenario_id: a.overall_risk_score for a in assessments},
            "risk_levels": {a.scenario_id: a.overall_risk_level.value for a in assessments},
            "critical_counts": {a.scenario_id: a.critical_risks for a in assessments},
            "residual_risks": {a.scenario_id: a.residual_risk for a in assessments}
        }
        
        # Find safest
        safest = min(assessments, key=lambda a: a.overall_risk_score)
        comparison["safest_scenario"] = safest.scenario_id
        
        # Find riskiest
        riskiest = max(assessments, key=lambda a: a.overall_risk_score)
        comparison["riskiest_scenario"] = riskiest.scenario_id
        
        return comparison
    
    def trend_analysis(self, scenario_type: str = None) -> Dict:
        """Analizza trend rischi nel tempo"""
        relevant = self.assessment_history
        if scenario_type:
            relevant = [a for a in relevant if scenario_type in a.scenario_id]
        
        if len(relevant) < 2:
            return {"trend": "insufficient_data", "data_points": len(relevant)}
        
        scores = [a.overall_risk_score for a in relevant]
        
        # Calculate trend
        avg_early = statistics.mean(scores[:len(scores)//2])
        avg_late = statistics.mean(scores[len(scores)//2:])
        
        if avg_late > avg_early * 1.1:
            trend = "increasing"
        elif avg_late < avg_early * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "early_average": avg_early,
            "late_average": avg_late,
            "data_points": len(relevant),
            "current_score": scores[-1] if scores else 0
        }
    
    # === Status ===
    
    def get_status(self) -> Dict:
        """Stato dell'analyzer"""
        return {
            "risk_templates": len(self.risk_templates),
            "mitigation_strategies": sum(len(v) for v in self.mitigation_strategies.values()),
            "assessments_performed": len(self.assessment_history),
            "risks_identified": self._risk_counter,
            "mitigations_generated": self._mitigation_counter
        }
