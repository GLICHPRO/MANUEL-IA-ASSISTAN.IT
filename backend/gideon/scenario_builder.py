# /backend/gideon/scenario_builder.py
"""
🔮 GIDEON 3.0 - Scenario Builder
Costruzione di scenari multipli per analisi predittiva e decisionale.
Non esegue azioni - genera scenari per simulazione e valutazione.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import copy
import hashlib
import logging

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Tipi di scenario"""
    BASELINE = "baseline"           # Scenario base/attuale
    OPTIMISTIC = "optimistic"       # Scenario ottimistico
    PESSIMISTIC = "pessimistic"     # Scenario pessimistico
    ALTERNATIVE = "alternative"     # Scenario alternativo
    WHAT_IF = "what_if"            # Scenario ipotetico
    CONTINGENCY = "contingency"     # Piano di contingenza
    STRESS_TEST = "stress_test"     # Test sotto stress
    MONTE_CARLO = "monte_carlo"     # Per simulazione Monte Carlo


class ScenarioStatus(Enum):
    """Stato dello scenario"""
    DRAFT = "draft"
    READY = "ready"
    SIMULATING = "simulating"
    SIMULATED = "simulated"
    VALIDATED = "validated"
    ARCHIVED = "archived"


class VariableType(Enum):
    """Tipi di variabili scenario"""
    CONSTANT = "constant"           # Valore fisso
    RANGE = "range"                 # Range di valori
    DISTRIBUTION = "distribution"   # Distribuzione statistica
    DEPENDENT = "dependent"         # Dipende da altre variabili
    EXTERNAL = "external"           # Fonte esterna
    TIME_SERIES = "time_series"     # Serie temporale


@dataclass
class ScenarioVariable:
    """Variabile di uno scenario"""
    name: str
    var_type: VariableType
    value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    distribution: Optional[str] = None  # normal, uniform, exponential
    distribution_params: Dict = field(default_factory=dict)
    depends_on: Optional[str] = None
    dependency_func: Optional[str] = None
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.var_type.value,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "distribution": self.distribution,
            "distribution_params": self.distribution_params,
            "depends_on": self.depends_on,
            "unit": self.unit,
            "description": self.description
        }


@dataclass
class ScenarioConstraint:
    """Vincolo su uno scenario"""
    name: str
    expression: str  # Es: "cpu_usage < 90"
    variables: List[str]
    is_hard: bool = True  # Hard = must satisfy, Soft = preferibile
    penalty: float = 1.0  # Penalità se violato (per soft constraints)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "expression": self.expression,
            "variables": self.variables,
            "is_hard": self.is_hard,
            "penalty": self.penalty
        }


@dataclass
class ScenarioOutcome:
    """Possibile outcome di uno scenario"""
    name: str
    probability: float
    impact_score: float  # -1.0 (molto negativo) a +1.0 (molto positivo)
    description: str
    conditions: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "probability": self.probability,
            "impact_score": self.impact_score,
            "description": self.description,
            "conditions": self.conditions,
            "metrics": self.metrics
        }


@dataclass
class Scenario:
    """Definizione completa di uno scenario"""
    id: str
    name: str
    description: str
    scenario_type: ScenarioType
    status: ScenarioStatus = ScenarioStatus.DRAFT
    
    # Core elements
    variables: Dict[str, ScenarioVariable] = field(default_factory=dict)
    constraints: List[ScenarioConstraint] = field(default_factory=list)
    outcomes: List[ScenarioOutcome] = field(default_factory=list)
    
    # Context
    context: Dict = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    
    # Relationships
    parent_id: Optional[str] = None
    derived_from: Optional[str] = None
    related_scenarios: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = "gideon"
    tags: List[str] = field(default_factory=list)
    
    # Simulation results (populated after simulation)
    simulation_results: Optional[Dict] = None
    confidence_score: float = 0.0
    risk_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.scenario_type.value,
            "status": self.status.value,
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "constraints": [c.to_dict() for c in self.constraints],
            "outcomes": [o.to_dict() for o in self.outcomes],
            "context": self.context,
            "assumptions": self.assumptions,
            "parent_id": self.parent_id,
            "derived_from": self.derived_from,
            "related_scenarios": self.related_scenarios,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "simulation_results": self.simulation_results,
            "confidence_score": self.confidence_score,
            "risk_score": self.risk_score
        }


@dataclass
class ScenarioComparison:
    """Confronto tra scenari"""
    scenarios: List[str]  # IDs
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {scenario_id: value}
    rankings: Dict[str, List[str]]  # metric_name -> [scenario_ids ordered]
    best_overall: str
    worst_overall: str
    trade_offs: List[Dict]
    
    def to_dict(self) -> dict:
        return {
            "scenarios": self.scenarios,
            "metrics": self.metrics,
            "rankings": self.rankings,
            "best_overall": self.best_overall,
            "worst_overall": self.worst_overall,
            "trade_offs": self.trade_offs
        }


class ScenarioBuilder:
    """
    Costruttore di scenari per Gideon.
    Crea, gestisce e manipola scenari multipli per analisi predittiva.
    """
    
    def __init__(self):
        # Storage
        self.scenarios: Dict[str, Scenario] = {}
        self.scenario_groups: Dict[str, List[str]] = {}  # group_name -> [scenario_ids]
        
        # Templates
        self.templates: Dict[str, Dict] = {}
        self._register_default_templates()
        
        # Counter
        self._counter = 0
        
        # Variable transformers
        self.transformers: Dict[str, Callable] = {}
        self._register_default_transformers()
    
    def _register_default_templates(self):
        """Registra template scenario predefiniti"""
        self.templates["system_action"] = {
            "variables": {
                "success_rate": {"type": "range", "min": 0.0, "max": 1.0, "default": 0.8},
                "execution_time": {"type": "range", "min": 0.1, "max": 60.0, "default": 1.0},
                "resource_usage": {"type": "range", "min": 0.0, "max": 100.0, "default": 30.0},
                "risk_level": {"type": "range", "min": 0.0, "max": 1.0, "default": 0.2}
            },
            "constraints": [
                {"name": "resource_limit", "expression": "resource_usage < 90", "is_hard": True}
            ],
            "outcomes": [
                {"name": "success", "probability": 0.8, "impact": 0.8},
                {"name": "partial_success", "probability": 0.15, "impact": 0.3},
                {"name": "failure", "probability": 0.05, "impact": -0.5}
            ]
        }
        
        self.templates["automation_workflow"] = {
            "variables": {
                "steps_count": {"type": "constant", "default": 5},
                "step_success_rate": {"type": "range", "min": 0.9, "max": 0.99, "default": 0.95},
                "total_time": {"type": "range", "min": 1.0, "max": 300.0, "default": 30.0},
                "rollback_capability": {"type": "constant", "default": True}
            },
            "constraints": [
                {"name": "time_limit", "expression": "total_time < 300", "is_hard": False}
            ],
            "outcomes": [
                {"name": "workflow_complete", "probability": 0.85, "impact": 1.0},
                {"name": "partial_complete", "probability": 0.10, "impact": 0.4},
                {"name": "workflow_failed", "probability": 0.05, "impact": -0.7}
            ]
        }
        
        self.templates["resource_allocation"] = {
            "variables": {
                "cpu_usage": {"type": "distribution", "distribution": "normal", "mean": 50, "std": 15},
                "memory_usage": {"type": "distribution", "distribution": "normal", "mean": 60, "std": 20},
                "disk_io": {"type": "range", "min": 0.0, "max": 100.0, "default": 30.0},
                "network_latency": {"type": "distribution", "distribution": "exponential", "lambda": 0.1}
            },
            "constraints": [
                {"name": "cpu_threshold", "expression": "cpu_usage < 85", "is_hard": True},
                {"name": "memory_threshold", "expression": "memory_usage < 90", "is_hard": True}
            ],
            "outcomes": [
                {"name": "optimal", "probability": 0.6, "impact": 0.9},
                {"name": "acceptable", "probability": 0.30, "impact": 0.5},
                {"name": "degraded", "probability": 0.10, "impact": -0.3}
            ]
        }
        
        self.templates["decision_branch"] = {
            "variables": {
                "option_count": {"type": "constant", "default": 3},
                "confidence_threshold": {"type": "range", "min": 0.5, "max": 0.95, "default": 0.7},
                "risk_tolerance": {"type": "range", "min": 0.0, "max": 1.0, "default": 0.3}
            },
            "constraints": [
                {"name": "min_confidence", "expression": "confidence > confidence_threshold", "is_hard": True}
            ],
            "outcomes": [
                {"name": "optimal_choice", "probability": 0.5, "impact": 0.9},
                {"name": "suboptimal_choice", "probability": 0.35, "impact": 0.4},
                {"name": "poor_choice", "probability": 0.15, "impact": -0.4}
            ]
        }
    
    def _register_default_transformers(self):
        """Registra trasformatori variabili"""
        self.transformers["percent_to_decimal"] = lambda x: x / 100.0
        self.transformers["decimal_to_percent"] = lambda x: x * 100.0
        self.transformers["invert"] = lambda x: 1.0 - x
        self.transformers["square"] = lambda x: x ** 2
        self.transformers["sqrt"] = lambda x: x ** 0.5
    
    # === Scenario Creation ===
    
    def create_scenario(self, name: str, description: str,
                        scenario_type: ScenarioType = ScenarioType.BASELINE,
                        template: str = None,
                        context: Dict = None) -> Scenario:
        """
        Crea un nuovo scenario.
        
        Args:
            name: Nome dello scenario
            description: Descrizione
            scenario_type: Tipo di scenario
            template: Template da usare (opzionale)
            context: Contesto aggiuntivo
        
        Returns:
            Scenario creato
        """
        self._counter += 1
        scenario_id = f"scn_{self._counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        scenario = Scenario(
            id=scenario_id,
            name=name,
            description=description,
            scenario_type=scenario_type,
            context=context or {}
        )
        
        # Applica template se specificato
        if template and template in self.templates:
            self._apply_template(scenario, template)
        
        self.scenarios[scenario_id] = scenario
        logger.info(f"Scenario creato: {scenario_id} - {name}")
        return scenario
    
    def _apply_template(self, scenario: Scenario, template_name: str):
        """Applica template a scenario"""
        template = self.templates[template_name]
        
        # Variables
        for var_name, var_config in template.get("variables", {}).items():
            var_type = VariableType(var_config.get("type", "constant"))
            scenario.variables[var_name] = ScenarioVariable(
                name=var_name,
                var_type=var_type,
                value=var_config.get("default"),
                min_value=var_config.get("min"),
                max_value=var_config.get("max"),
                distribution=var_config.get("distribution"),
                distribution_params={
                    k: v for k, v in var_config.items() 
                    if k not in ["type", "default", "min", "max", "distribution"]
                }
            )
        
        # Constraints
        for cons_config in template.get("constraints", []):
            scenario.constraints.append(ScenarioConstraint(
                name=cons_config["name"],
                expression=cons_config["expression"],
                variables=self._extract_variables(cons_config["expression"]),
                is_hard=cons_config.get("is_hard", True)
            ))
        
        # Outcomes
        for out_config in template.get("outcomes", []):
            scenario.outcomes.append(ScenarioOutcome(
                name=out_config["name"],
                probability=out_config["probability"],
                impact_score=out_config["impact"],
                description=f"Outcome: {out_config['name']}"
            ))
    
    def _extract_variables(self, expression: str) -> List[str]:
        """Estrae nomi variabili da espressione"""
        import re
        # Trova parole che non sono operatori o numeri
        tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
        operators = {'and', 'or', 'not', 'if', 'else', 'True', 'False'}
        return [t for t in tokens if t not in operators]
    
    # === Scenario Manipulation ===
    
    def add_variable(self, scenario_id: str, name: str, 
                     var_type: VariableType, value: Any, **kwargs) -> bool:
        """Aggiunge variabile a scenario"""
        if scenario_id not in self.scenarios:
            return False
        
        scenario = self.scenarios[scenario_id]
        scenario.variables[name] = ScenarioVariable(
            name=name,
            var_type=var_type,
            value=value,
            min_value=kwargs.get("min_value"),
            max_value=kwargs.get("max_value"),
            distribution=kwargs.get("distribution"),
            distribution_params=kwargs.get("distribution_params", {}),
            depends_on=kwargs.get("depends_on"),
            unit=kwargs.get("unit", ""),
            description=kwargs.get("description", "")
        )
        scenario.updated_at = datetime.now()
        return True
    
    def add_constraint(self, scenario_id: str, name: str,
                       expression: str, is_hard: bool = True) -> bool:
        """Aggiunge vincolo a scenario"""
        if scenario_id not in self.scenarios:
            return False
        
        scenario = self.scenarios[scenario_id]
        scenario.constraints.append(ScenarioConstraint(
            name=name,
            expression=expression,
            variables=self._extract_variables(expression),
            is_hard=is_hard
        ))
        scenario.updated_at = datetime.now()
        return True
    
    def add_outcome(self, scenario_id: str, name: str,
                    probability: float, impact: float,
                    description: str = "") -> bool:
        """Aggiunge outcome a scenario"""
        if scenario_id not in self.scenarios:
            return False
        
        scenario = self.scenarios[scenario_id]
        scenario.outcomes.append(ScenarioOutcome(
            name=name,
            probability=probability,
            impact_score=impact,
            description=description
        ))
        scenario.updated_at = datetime.now()
        return True
    
    def set_variable_value(self, scenario_id: str, var_name: str, value: Any) -> bool:
        """Imposta valore di una variabile"""
        if scenario_id not in self.scenarios:
            return False
        
        scenario = self.scenarios[scenario_id]
        if var_name in scenario.variables:
            scenario.variables[var_name].value = value
            scenario.updated_at = datetime.now()
            return True
        return False
    
    # === Scenario Derivation ===
    
    def derive_scenario(self, base_id: str, name: str,
                        scenario_type: ScenarioType,
                        modifications: Dict = None) -> Optional[Scenario]:
        """
        Deriva nuovo scenario da uno esistente.
        
        Args:
            base_id: ID scenario base
            name: Nome nuovo scenario
            scenario_type: Tipo del nuovo scenario
            modifications: Modifiche da applicare {var_name: new_value}
        
        Returns:
            Nuovo scenario derivato
        """
        if base_id not in self.scenarios:
            return None
        
        base = self.scenarios[base_id]
        
        # Deep copy
        new_scenario = self.create_scenario(
            name=name,
            description=f"Derivato da {base.name}",
            scenario_type=scenario_type,
            context=copy.deepcopy(base.context)
        )
        
        # Copy elements
        new_scenario.variables = copy.deepcopy(base.variables)
        new_scenario.constraints = copy.deepcopy(base.constraints)
        new_scenario.outcomes = copy.deepcopy(base.outcomes)
        new_scenario.assumptions = base.assumptions.copy()
        new_scenario.derived_from = base_id
        new_scenario.parent_id = base_id
        
        # Apply modifications
        if modifications:
            for var_name, new_value in modifications.items():
                if var_name in new_scenario.variables:
                    new_scenario.variables[var_name].value = new_value
        
        # Link scenarios
        base.related_scenarios.append(new_scenario.id)
        new_scenario.related_scenarios.append(base_id)
        
        return new_scenario
    
    def create_optimistic_pessimistic(self, base_id: str,
                                       optimistic_factor: float = 1.2,
                                       pessimistic_factor: float = 0.8) -> Tuple[Scenario, Scenario]:
        """
        Crea scenari ottimistico e pessimistico da baseline.
        
        Args:
            base_id: ID scenario base
            optimistic_factor: Moltiplicatore per valori positivi (ottimistico)
            pessimistic_factor: Moltiplicatore per valori positivi (pessimistico)
        
        Returns:
            Tuple (scenario_ottimistico, scenario_pessimistico)
        """
        if base_id not in self.scenarios:
            return None, None
        
        base = self.scenarios[base_id]
        
        # Optimistic modifications
        opt_mods = {}
        pess_mods = {}
        
        for var_name, var in base.variables.items():
            if isinstance(var.value, (int, float)):
                # Per variabili positive (success rate, etc.)
                if "success" in var_name.lower() or "rate" in var_name.lower():
                    opt_mods[var_name] = min(1.0, var.value * optimistic_factor)
                    pess_mods[var_name] = max(0.0, var.value * pessimistic_factor)
                # Per variabili negative (risk, error, etc.)
                elif "risk" in var_name.lower() or "error" in var_name.lower():
                    opt_mods[var_name] = max(0.0, var.value * pessimistic_factor)
                    pess_mods[var_name] = min(1.0, var.value * optimistic_factor)
                # Per altre variabili
                else:
                    opt_mods[var_name] = var.value * optimistic_factor
                    pess_mods[var_name] = var.value * pessimistic_factor
        
        optimistic = self.derive_scenario(
            base_id, f"{base.name} (Ottimistico)",
            ScenarioType.OPTIMISTIC, opt_mods
        )
        
        pessimistic = self.derive_scenario(
            base_id, f"{base.name} (Pessimistico)",
            ScenarioType.PESSIMISTIC, pess_mods
        )
        
        return optimistic, pessimistic
    
    def create_what_if(self, base_id: str, what_if: str,
                       changes: Dict) -> Optional[Scenario]:
        """
        Crea scenario what-if.
        
        Args:
            base_id: ID scenario base
            what_if: Descrizione ipotesi ("What if CPU usage doubles?")
            changes: Modifiche alle variabili
        """
        scenario = self.derive_scenario(
            base_id,
            f"What-If: {what_if}",
            ScenarioType.WHAT_IF,
            changes
        )
        
        if scenario:
            scenario.assumptions.append(what_if)
        
        return scenario
    
    # === Scenario Analysis ===
    
    def validate_scenario(self, scenario_id: str) -> Dict:
        """
        Valida uno scenario verificando vincoli e completezza.
        
        Returns:
            Report di validazione
        """
        if scenario_id not in self.scenarios:
            return {"valid": False, "error": "Scenario non trovato"}
        
        scenario = self.scenarios[scenario_id]
        issues = []
        warnings = []
        
        # Check variables
        for var_name, var in scenario.variables.items():
            if var.value is None and var.var_type != VariableType.DEPENDENT:
                issues.append(f"Variabile '{var_name}' senza valore")
            
            if var.min_value is not None and var.value is not None:
                if var.value < var.min_value:
                    issues.append(f"'{var_name}' sotto valore minimo")
            
            if var.max_value is not None and var.value is not None:
                if var.value > var.max_value:
                    issues.append(f"'{var_name}' sopra valore massimo")
        
        # Check outcomes probabilities
        total_prob = sum(o.probability for o in scenario.outcomes)
        if abs(total_prob - 1.0) > 0.01:
            warnings.append(f"Probabilità outcomes non sommano a 1 ({total_prob:.2f})")
        
        # Check constraints (simplified)
        for constraint in scenario.constraints:
            for var in constraint.variables:
                if var not in scenario.variables:
                    issues.append(f"Vincolo '{constraint.name}' usa variabile sconosciuta '{var}'")
        
        valid = len(issues) == 0
        
        if valid:
            scenario.status = ScenarioStatus.READY
        
        return {
            "valid": valid,
            "scenario_id": scenario_id,
            "issues": issues,
            "warnings": warnings,
            "status": scenario.status.value
        }
    
    def compare_scenarios(self, scenario_ids: List[str],
                          metrics: List[str] = None) -> ScenarioComparison:
        """
        Confronta più scenari.
        
        Args:
            scenario_ids: Lista ID scenari da confrontare
            metrics: Metriche da usare (default: tutte le variabili comuni)
        
        Returns:
            ScenarioComparison con ranking e trade-offs
        """
        scenarios = [self.scenarios[sid] for sid in scenario_ids if sid in self.scenarios]
        
        if len(scenarios) < 2:
            return None
        
        # Find common variables as metrics
        if not metrics:
            common_vars = set(scenarios[0].variables.keys())
            for s in scenarios[1:]:
                common_vars &= set(s.variables.keys())
            metrics = list(common_vars)
        
        # Collect metric values
        metric_values = {}
        for metric in metrics:
            metric_values[metric] = {}
            for s in scenarios:
                if metric in s.variables:
                    metric_values[metric][s.id] = s.variables[metric].value
        
        # Rank per metric
        rankings = {}
        for metric, values in metric_values.items():
            # Sort descending (higher is better by default)
            sorted_ids = sorted(values.keys(), key=lambda x: values.get(x, 0), reverse=True)
            rankings[metric] = sorted_ids
        
        # Calculate overall score
        scores = {s.id: 0 for s in scenarios}
        for metric, ranked_ids in rankings.items():
            for rank, sid in enumerate(ranked_ids):
                scores[sid] += len(ranked_ids) - rank
        
        best = max(scores.keys(), key=lambda x: scores[x])
        worst = min(scores.keys(), key=lambda x: scores[x])
        
        # Identify trade-offs
        trade_offs = []
        for i, s1 in enumerate(scenarios):
            for s2 in scenarios[i+1:]:
                s1_better = []
                s2_better = []
                for metric in metrics:
                    v1 = metric_values[metric].get(s1.id, 0)
                    v2 = metric_values[metric].get(s2.id, 0)
                    if v1 > v2:
                        s1_better.append(metric)
                    elif v2 > v1:
                        s2_better.append(metric)
                
                if s1_better and s2_better:
                    trade_offs.append({
                        "scenarios": [s1.id, s2.id],
                        f"{s1.id}_better_at": s1_better,
                        f"{s2.id}_better_at": s2_better
                    })
        
        return ScenarioComparison(
            scenarios=scenario_ids,
            metrics=metric_values,
            rankings=rankings,
            best_overall=best,
            worst_overall=worst,
            trade_offs=trade_offs
        )
    
    # === Batch Operations ===
    
    def create_scenario_set(self, name: str, base_config: Dict,
                            variations: List[Dict]) -> List[str]:
        """
        Crea set di scenari con variazioni sistematiche.
        
        Args:
            name: Nome base
            base_config: Configurazione base
            variations: Lista di variazioni [{var_name: [values]}]
        
        Returns:
            Lista ID scenari creati
        """
        created = []
        
        # Create base
        base = self.create_scenario(
            f"{name} - Base",
            f"Scenario base per set {name}",
            ScenarioType.BASELINE
        )
        
        for var_name, value in base_config.items():
            self.add_variable(base.id, var_name, VariableType.CONSTANT, value)
        
        created.append(base.id)
        
        # Create variations
        for i, variation in enumerate(variations):
            var_scenario = self.derive_scenario(
                base.id,
                f"{name} - Variazione {i+1}",
                ScenarioType.ALTERNATIVE,
                variation
            )
            if var_scenario:
                created.append(var_scenario.id)
        
        # Group them
        self.scenario_groups[name] = created
        
        return created
    
    def create_sensitivity_scenarios(self, base_id: str,
                                      variable: str,
                                      values: List[Any]) -> List[str]:
        """
        Crea scenari per analisi di sensitività su una variabile.
        """
        if base_id not in self.scenarios:
            return []
        
        created = []
        for value in values:
            scenario = self.derive_scenario(
                base_id,
                f"Sensitivity: {variable}={value}",
                ScenarioType.WHAT_IF,
                {variable: value}
            )
            if scenario:
                created.append(scenario.id)
        
        return created
    
    # === Query & Retrieval ===
    
    def get_scenario(self, scenario_id: str) -> Optional[Dict]:
        """Ottiene scenario per ID"""
        if scenario_id in self.scenarios:
            return self.scenarios[scenario_id].to_dict()
        return None
    
    def list_scenarios(self, scenario_type: ScenarioType = None,
                       status: ScenarioStatus = None,
                       tags: List[str] = None) -> List[Dict]:
        """Lista scenari con filtri opzionali"""
        results = []
        
        for scenario in self.scenarios.values():
            if scenario_type and scenario.scenario_type != scenario_type:
                continue
            if status and scenario.status != status:
                continue
            if tags and not any(t in scenario.tags for t in tags):
                continue
            
            results.append({
                "id": scenario.id,
                "name": scenario.name,
                "type": scenario.scenario_type.value,
                "status": scenario.status.value,
                "created_at": scenario.created_at.isoformat()
            })
        
        return results
    
    def get_scenario_tree(self, root_id: str) -> Dict:
        """Ottiene albero di scenari derivati"""
        if root_id not in self.scenarios:
            return None
        
        def build_tree(scenario_id: str) -> Dict:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                return None
            
            children = [
                build_tree(sid) for sid in self.scenarios.keys()
                if self.scenarios[sid].parent_id == scenario_id
            ]
            
            return {
                "id": scenario.id,
                "name": scenario.name,
                "type": scenario.scenario_type.value,
                "children": [c for c in children if c]
            }
        
        return build_tree(root_id)
    
    def get_status(self) -> Dict:
        """Stato del builder"""
        return {
            "total_scenarios": len(self.scenarios),
            "by_type": {
                t.value: sum(1 for s in self.scenarios.values() if s.scenario_type == t)
                for t in ScenarioType
            },
            "by_status": {
                s.value: sum(1 for sc in self.scenarios.values() if sc.status == s)
                for s in ScenarioStatus
            },
            "groups": {k: len(v) for k, v in self.scenario_groups.items()},
            "templates": list(self.templates.keys())
        }
