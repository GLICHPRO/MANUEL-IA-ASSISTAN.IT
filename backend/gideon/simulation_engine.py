# /backend/gideon/simulation_engine.py
"""
🔮 GIDEON 3.0 - Simulation Engine
Motore di simulazione parallela e predittiva per analisi scenari.
Non esegue azioni - simula e predice outcomes.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import random
import math
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import copy

logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Modalità di simulazione"""
    SINGLE = "single"               # Singola esecuzione
    MONTE_CARLO = "monte_carlo"     # Monte Carlo sampling
    SENSITIVITY = "sensitivity"     # Analisi di sensitività
    PARALLEL = "parallel"           # Esecuzione parallela
    BAYESIAN = "bayesian"          # Inferenza Bayesiana
    PREDICTIVE = "predictive"       # Simulazione predittiva temporale


class DistributionType(Enum):
    """Tipi di distribuzione statistica"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    TRIANGULAR = "triangular"
    BETA = "beta"
    POISSON = "poisson"
    LOG_NORMAL = "log_normal"


@dataclass
class SimulationResult:
    """Risultato di una singola simulazione"""
    iteration: int
    values: Dict[str, float]
    outcome: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SimulationSummary:
    """Sommario di simulazione multi-iterazione"""
    scenario_id: str
    mode: SimulationMode
    iterations: int
    
    # Statistical summaries per variable
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Outcome distribution
    outcome_distribution: Dict[str, int] = field(default_factory=dict)
    outcome_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Overall metrics
    success_rate: float = 0.0
    expected_value: float = 0.0
    risk_score: float = 0.0
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    confidence_level: float = 0.95
    
    # Execution info
    execution_time_ms: float = 0.0
    completed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "mode": self.mode.value,
            "iterations": self.iterations,
            "statistics": self.statistics,
            "outcome_distribution": self.outcome_distribution,
            "outcome_probabilities": self.outcome_probabilities,
            "success_rate": self.success_rate,
            "expected_value": self.expected_value,
            "risk_score": self.risk_score,
            "confidence_intervals": self.confidence_intervals,
            "confidence_level": self.confidence_level,
            "execution_time_ms": self.execution_time_ms,
            "completed_at": self.completed_at.isoformat()
        }


@dataclass
class SensitivityResult:
    """Risultato analisi di sensitività"""
    variable: str
    values_tested: List[Any]
    outcomes: Dict[Any, Dict]  # value -> outcome_stats
    elasticity: float  # Sensibilità relativa
    critical_thresholds: List[Tuple[float, str]]  # (threshold, outcome_change)
    
    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "values_tested": self.values_tested,
            "outcomes": self.outcomes,
            "elasticity": self.elasticity,
            "critical_thresholds": self.critical_thresholds
        }


@dataclass
class PredictiveResult:
    """Risultato simulazione predittiva temporale"""
    scenario_id: str
    time_horizon: timedelta
    time_steps: int
    
    # Trajectory data
    trajectories: Dict[str, List[float]] = field(default_factory=dict)
    timestamps: List[datetime] = field(default_factory=list)
    
    # Predictions
    final_predictions: Dict[str, float] = field(default_factory=dict)
    prediction_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Trend analysis
    trends: Dict[str, str] = field(default_factory=dict)  # variable -> trend type
    turning_points: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "time_horizon_seconds": self.time_horizon.total_seconds(),
            "time_steps": self.time_steps,
            "trajectories": self.trajectories,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "final_predictions": self.final_predictions,
            "prediction_intervals": self.prediction_intervals,
            "trends": self.trends,
            "turning_points": self.turning_points
        }


class SimulationEngine:
    """
    Motore di simulazione avanzato per Gideon.
    Supporta simulazioni parallele, Monte Carlo, sensitività e predittive.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        
        # Random state
        self.seed: Optional[int] = None
        
        # Results cache
        self.results_cache: Dict[str, SimulationSummary] = {}
        
        # Custom distributions
        self.custom_distributions: Dict[str, Callable] = {}
        
        # Lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Counters
        self._total_simulations = 0
        self._total_iterations = 0
    
    def set_seed(self, seed: int):
        """Imposta seed per riproducibilità"""
        self.seed = seed
        random.seed(seed)
    
    # === Distribution Sampling ===
    
    def sample_distribution(self, dist_type: DistributionType,
                           params: Dict) -> float:
        """
        Genera campione da distribuzione.
        
        Args:
            dist_type: Tipo distribuzione
            params: Parametri distribuzione
        
        Returns:
            Valore campionato
        """
        if dist_type == DistributionType.UNIFORM:
            return random.uniform(params.get("min", 0), params.get("max", 1))
        
        elif dist_type == DistributionType.NORMAL:
            return random.gauss(params.get("mean", 0), params.get("std", 1))
        
        elif dist_type == DistributionType.EXPONENTIAL:
            return random.expovariate(params.get("lambda", 1))
        
        elif dist_type == DistributionType.TRIANGULAR:
            return random.triangular(
                params.get("min", 0),
                params.get("max", 1),
                params.get("mode", 0.5)
            )
        
        elif dist_type == DistributionType.BETA:
            return random.betavariate(
                params.get("alpha", 2),
                params.get("beta", 2)
            )
        
        elif dist_type == DistributionType.POISSON:
            # Approximation using rejection method
            lam = params.get("lambda", 1)
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= random.random()
            return k - 1
        
        elif dist_type == DistributionType.LOG_NORMAL:
            return random.lognormvariate(
                params.get("mean", 0),
                params.get("sigma", 1)
            )
        
        return 0.0
    
    def sample_variable(self, variable: Dict) -> float:
        """Campiona valore da definizione variabile"""
        var_type = variable.get("type", "constant")
        
        if var_type == "constant":
            return variable.get("value", 0)
        
        elif var_type == "range":
            return random.uniform(
                variable.get("min_value", 0),
                variable.get("max_value", 1)
            )
        
        elif var_type == "distribution":
            dist_type = DistributionType(variable.get("distribution", "normal"))
            return self.sample_distribution(dist_type, variable.get("distribution_params", {}))
        
        return variable.get("value", 0)
    
    # === Core Simulation ===
    
    def simulate_single(self, scenario: Dict) -> SimulationResult:
        """
        Esegue singola simulazione di uno scenario.
        
        Args:
            scenario: Definizione scenario con variabili e outcomes
        
        Returns:
            SimulationResult
        """
        self._total_simulations += 1
        
        # Sample all variables
        sampled_values = {}
        for var_name, var_def in scenario.get("variables", {}).items():
            if isinstance(var_def, dict):
                sampled_values[var_name] = self.sample_variable(var_def)
            else:
                sampled_values[var_name] = var_def
        
        # Evaluate constraints
        constraints_satisfied = True
        for constraint in scenario.get("constraints", []):
            if not self._evaluate_constraint(constraint, sampled_values):
                constraints_satisfied = False
                break
        
        # Determine outcome
        outcomes = scenario.get("outcomes", [])
        selected_outcome = self._select_outcome(outcomes, sampled_values)
        
        # Calculate metrics
        metrics = self._calculate_metrics(sampled_values, selected_outcome, scenario)
        
        return SimulationResult(
            iteration=self._total_simulations,
            values=sampled_values,
            outcome=selected_outcome.get("name", "unknown"),
            success=selected_outcome.get("impact_score", 0) > 0 and constraints_satisfied,
            metrics=metrics
        )
    
    def _evaluate_constraint(self, constraint: Dict, values: Dict) -> bool:
        """Valuta vincolo su valori"""
        expression = constraint.get("expression", "True")
        
        # Simple evaluation (in production, use safer eval)
        try:
            # Create safe namespace
            safe_dict = {"__builtins__": {}}
            safe_dict.update(values)
            return eval(expression, safe_dict)
        except:
            return True
    
    def _select_outcome(self, outcomes: List[Dict], values: Dict) -> Dict:
        """Seleziona outcome basato su probabilità"""
        if not outcomes:
            return {"name": "unknown", "probability": 1.0, "impact_score": 0}
        
        # Adjust probabilities based on values (simplified)
        adjusted = []
        for outcome in outcomes:
            prob = outcome.get("probability", 0.5)
            
            # Adjust based on conditions
            for condition in outcome.get("conditions", []):
                try:
                    safe_dict = {"__builtins__": {}}
                    safe_dict.update(values)
                    if eval(condition, safe_dict):
                        prob *= 1.2  # Increase if condition met
                except:
                    pass
            
            adjusted.append((outcome, prob))
        
        # Normalize
        total = sum(p for _, p in adjusted)
        if total > 0:
            adjusted = [(o, p/total) for o, p in adjusted]
        
        # Random selection
        r = random.random()
        cumulative = 0
        for outcome, prob in adjusted:
            cumulative += prob
            if r <= cumulative:
                return outcome
        
        return outcomes[0]
    
    def _calculate_metrics(self, values: Dict, outcome: Dict,
                          scenario: Dict) -> Dict[str, float]:
        """Calcola metriche simulazione"""
        metrics = {
            "outcome_impact": outcome.get("impact_score", 0),
            "risk_exposure": self._calculate_risk(values, scenario),
            "efficiency": self._calculate_efficiency(values)
        }
        return metrics
    
    def _calculate_risk(self, values: Dict, scenario: Dict) -> float:
        """Calcola risk score"""
        risk = 0.0
        risk_vars = ["risk", "error", "failure"]
        
        for var_name, value in values.items():
            if any(rv in var_name.lower() for rv in risk_vars):
                if isinstance(value, (int, float)):
                    risk += value
        
        return min(1.0, risk / max(1, len(values)))
    
    def _calculate_efficiency(self, values: Dict) -> float:
        """Calcola efficiency score"""
        efficiency_vars = ["success", "rate", "efficiency", "performance"]
        scores = []
        
        for var_name, value in values.items():
            if any(ev in var_name.lower() for ev in efficiency_vars):
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    scores.append(value)
        
        return statistics.mean(scores) if scores else 0.5
    
    # === Monte Carlo Simulation ===
    
    def simulate_monte_carlo(self, scenario: Dict,
                             iterations: int = 1000,
                             confidence_level: float = 0.95) -> SimulationSummary:
        """
        Esegue simulazione Monte Carlo.
        
        Args:
            scenario: Definizione scenario
            iterations: Numero iterazioni
            confidence_level: Livello confidenza per intervalli
        
        Returns:
            SimulationSummary
        """
        start_time = datetime.now()
        results: List[SimulationResult] = []
        
        for i in range(iterations):
            result = self.simulate_single(scenario)
            result.iteration = i
            results.append(result)
        
        self._total_iterations += iterations
        
        summary = self._aggregate_results(
            scenario.get("id", "unknown"),
            results,
            SimulationMode.MONTE_CARLO,
            confidence_level
        )
        
        summary.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Cache result
        cache_key = f"{scenario.get('id')}_{iterations}"
        self.results_cache[cache_key] = summary
        
        return summary
    
    def simulate_monte_carlo_parallel(self, scenario: Dict,
                                       iterations: int = 1000,
                                       confidence_level: float = 0.95) -> SimulationSummary:
        """
        Simulazione Monte Carlo con esecuzione parallela.
        """
        start_time = datetime.now()
        results: List[SimulationResult] = []
        
        # Divide iterations among workers
        chunk_size = max(1, iterations // self.max_workers)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for worker_id in range(self.max_workers):
                start_iter = worker_id * chunk_size
                end_iter = min(start_iter + chunk_size, iterations)
                
                if start_iter >= iterations:
                    break
                
                future = executor.submit(
                    self._run_monte_carlo_chunk,
                    scenario,
                    start_iter,
                    end_iter
                )
                futures.append(future)
            
            for future in as_completed(futures):
                chunk_results = future.result()
                results.extend(chunk_results)
        
        self._total_iterations += len(results)
        
        summary = self._aggregate_results(
            scenario.get("id", "unknown"),
            results,
            SimulationMode.PARALLEL,
            confidence_level
        )
        
        summary.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return summary
    
    def _run_monte_carlo_chunk(self, scenario: Dict,
                                start_iter: int, end_iter: int) -> List[SimulationResult]:
        """Esegue chunk di simulazioni Monte Carlo"""
        results = []
        for i in range(start_iter, end_iter):
            result = self.simulate_single(scenario)
            result.iteration = i
            results.append(result)
        return results
    
    def _aggregate_results(self, scenario_id: str,
                           results: List[SimulationResult],
                           mode: SimulationMode,
                           confidence_level: float) -> SimulationSummary:
        """Aggrega risultati simulazione"""
        if not results:
            return SimulationSummary(
                scenario_id=scenario_id,
                mode=mode,
                iterations=0
            )
        
        summary = SimulationSummary(
            scenario_id=scenario_id,
            mode=mode,
            iterations=len(results),
            confidence_level=confidence_level
        )
        
        # Aggregate variable statistics
        all_vars = set()
        for r in results:
            all_vars.update(r.values.keys())
        
        for var in all_vars:
            values = [r.values.get(var, 0) for r in results if var in r.values]
            if values:
                summary.statistics[var] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values)
                }
                
                # Confidence interval
                summary.confidence_intervals[var] = self._calculate_confidence_interval(
                    values, confidence_level
                )
        
        # Outcome distribution
        for r in results:
            summary.outcome_distribution[r.outcome] = \
                summary.outcome_distribution.get(r.outcome, 0) + 1
        
        # Convert to probabilities
        total = len(results)
        for outcome, count in summary.outcome_distribution.items():
            summary.outcome_probabilities[outcome] = count / total
        
        # Overall metrics
        summary.success_rate = sum(1 for r in results if r.success) / total
        
        impact_scores = [r.metrics.get("outcome_impact", 0) for r in results]
        summary.expected_value = statistics.mean(impact_scores)
        
        risk_scores = [r.metrics.get("risk_exposure", 0) for r in results]
        summary.risk_score = statistics.mean(risk_scores)
        
        return summary
    
    def _calculate_confidence_interval(self, values: List[float],
                                        confidence: float) -> Tuple[float, float]:
        """Calcola intervallo di confidenza"""
        if len(values) < 2:
            return (values[0] if values else 0, values[0] if values else 0)
        
        mean = statistics.mean(values)
        std_err = statistics.stdev(values) / math.sqrt(len(values))
        
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        margin = z * std_err
        return (mean - margin, mean + margin)
    
    # === Sensitivity Analysis ===
    
    def analyze_sensitivity(self, scenario: Dict,
                            variable: str,
                            values: List[Any],
                            iterations_per_value: int = 100) -> SensitivityResult:
        """
        Analisi di sensitività su una variabile.
        
        Args:
            scenario: Scenario base
            variable: Nome variabile da analizzare
            values: Valori da testare
            iterations_per_value: Iterazioni per ogni valore
        
        Returns:
            SensitivityResult
        """
        outcomes = {}
        
        for value in values:
            # Modify scenario
            modified = copy.deepcopy(scenario)
            if variable in modified.get("variables", {}):
                if isinstance(modified["variables"][variable], dict):
                    modified["variables"][variable]["value"] = value
                    modified["variables"][variable]["type"] = "constant"
                else:
                    modified["variables"][variable] = value
            
            # Run simulations
            summary = self.simulate_monte_carlo(
                modified,
                iterations=iterations_per_value,
                confidence_level=0.95
            )
            
            outcomes[value] = {
                "success_rate": summary.success_rate,
                "expected_value": summary.expected_value,
                "risk_score": summary.risk_score,
                "outcome_probabilities": summary.outcome_probabilities
            }
        
        # Calculate elasticity
        elasticity = self._calculate_elasticity(values, outcomes)
        
        # Find critical thresholds
        thresholds = self._find_thresholds(values, outcomes)
        
        return SensitivityResult(
            variable=variable,
            values_tested=values,
            outcomes=outcomes,
            elasticity=elasticity,
            critical_thresholds=thresholds
        )
    
    def _calculate_elasticity(self, values: List[Any],
                               outcomes: Dict) -> float:
        """Calcola elasticità (sensibilità relativa)"""
        if len(values) < 2:
            return 0.0
        
        # Use success_rate as primary metric
        rates = [outcomes[v].get("success_rate", 0) for v in values]
        
        # Calculate average % change in output / % change in input
        changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0 and rates[i-1] != 0:
                input_change = (values[i] - values[i-1]) / values[i-1]
                output_change = (rates[i] - rates[i-1]) / rates[i-1]
                if input_change != 0:
                    changes.append(abs(output_change / input_change))
        
        return statistics.mean(changes) if changes else 0.0
    
    def _find_thresholds(self, values: List[Any],
                          outcomes: Dict) -> List[Tuple[float, str]]:
        """Trova soglie critiche dove l'outcome cambia"""
        thresholds = []
        
        rates = [(v, outcomes[v].get("success_rate", 0)) for v in values]
        rates.sort(key=lambda x: x[0])
        
        for i in range(1, len(rates)):
            prev_rate = rates[i-1][1]
            curr_rate = rates[i][1]
            
            # Significant change (>10% shift)
            if abs(curr_rate - prev_rate) > 0.1:
                threshold = (rates[i-1][0] + rates[i][0]) / 2
                direction = "increasing" if curr_rate > prev_rate else "decreasing"
                thresholds.append((threshold, f"success_rate_{direction}"))
        
        return thresholds
    
    # === Predictive Simulation ===
    
    def simulate_predictive(self, scenario: Dict,
                            time_horizon: timedelta,
                            time_steps: int = 10,
                            trend_functions: Dict[str, Callable] = None) -> PredictiveResult:
        """
        Simulazione predittiva nel tempo.
        
        Args:
            scenario: Scenario con variabili
            time_horizon: Orizzonte temporale
            time_steps: Numero di step temporali
            trend_functions: Funzioni trend personalizzate per variabili
        
        Returns:
            PredictiveResult con traiettorie e previsioni
        """
        start_time = datetime.now()
        step_duration = time_horizon / time_steps
        
        result = PredictiveResult(
            scenario_id=scenario.get("id", "unknown"),
            time_horizon=time_horizon,
            time_steps=time_steps
        )
        
        # Initialize trajectories
        variables = scenario.get("variables", {})
        for var_name, var_def in variables.items():
            initial_value = var_def.get("value", 0) if isinstance(var_def, dict) else var_def
            result.trajectories[var_name] = [initial_value]
        
        # Generate timestamps
        result.timestamps = [start_time]
        for i in range(1, time_steps + 1):
            result.timestamps.append(start_time + step_duration * i)
        
        # Simulate each time step
        for step in range(1, time_steps + 1):
            for var_name in result.trajectories:
                prev_value = result.trajectories[var_name][-1]
                
                # Apply trend function if provided
                if trend_functions and var_name in trend_functions:
                    new_value = trend_functions[var_name](prev_value, step, time_steps)
                else:
                    # Default: slight random walk with mean reversion
                    new_value = self._default_trend(prev_value, variables.get(var_name, {}))
                
                result.trajectories[var_name].append(new_value)
        
        # Calculate final predictions
        for var_name, trajectory in result.trajectories.items():
            result.final_predictions[var_name] = trajectory[-1]
            
            # Prediction interval based on trajectory variance
            if len(trajectory) > 1:
                std_dev = statistics.stdev(trajectory)
                last_val = trajectory[-1]
                result.prediction_intervals[var_name] = (
                    last_val - 1.96 * std_dev,
                    last_val + 1.96 * std_dev
                )
        
        # Analyze trends
        for var_name, trajectory in result.trajectories.items():
            result.trends[var_name] = self._detect_trend(trajectory)
        
        # Find turning points
        result.turning_points = self._find_turning_points(result.trajectories, result.timestamps)
        
        return result
    
    def _default_trend(self, prev_value: float, var_def: Dict) -> float:
        """Trend di default: random walk con mean reversion"""
        mean = var_def.get("value", prev_value) if isinstance(var_def, dict) else var_def
        
        # Mean reversion factor
        reversion = 0.1 * (mean - prev_value)
        
        # Random component
        noise = random.gauss(0, abs(mean) * 0.05 + 0.01)
        
        return prev_value + reversion + noise
    
    def _detect_trend(self, trajectory: List[float]) -> str:
        """Rileva tipo di trend"""
        if len(trajectory) < 3:
            return "stable"
        
        # Calculate differences
        diffs = [trajectory[i+1] - trajectory[i] for i in range(len(trajectory)-1)]
        avg_diff = statistics.mean(diffs)
        
        # Calculate acceleration
        if len(diffs) > 1:
            accel = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            avg_accel = statistics.mean(accel)
        else:
            avg_accel = 0
        
        # Classify
        if abs(avg_diff) < 0.01 * abs(statistics.mean(trajectory)):
            return "stable"
        elif avg_diff > 0:
            if avg_accel > 0:
                return "accelerating_up"
            elif avg_accel < 0:
                return "decelerating_up"
            else:
                return "linear_up"
        else:
            if avg_accel < 0:
                return "accelerating_down"
            elif avg_accel > 0:
                return "decelerating_down"
            else:
                return "linear_down"
    
    def _find_turning_points(self, trajectories: Dict[str, List[float]],
                              timestamps: List[datetime]) -> List[Dict]:
        """Trova punti di svolta nelle traiettorie"""
        turning_points = []
        
        for var_name, trajectory in trajectories.items():
            for i in range(1, len(trajectory) - 1):
                prev_diff = trajectory[i] - trajectory[i-1]
                next_diff = trajectory[i+1] - trajectory[i]
                
                # Sign change indicates turning point
                if prev_diff * next_diff < 0:
                    turning_points.append({
                        "variable": var_name,
                        "index": i,
                        "timestamp": timestamps[i].isoformat(),
                        "value": trajectory[i],
                        "type": "peak" if prev_diff > 0 else "trough"
                    })
        
        return turning_points
    
    # === Batch Operations ===
    
    def simulate_batch(self, scenarios: List[Dict],
                       mode: SimulationMode = SimulationMode.MONTE_CARLO,
                       iterations: int = 1000) -> Dict[str, SimulationSummary]:
        """
        Simula batch di scenari in parallelo.
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for scenario in scenarios:
                scenario_id = scenario.get("id", str(len(futures)))
                
                if mode == SimulationMode.MONTE_CARLO:
                    future = executor.submit(
                        self.simulate_monte_carlo,
                        scenario, iterations
                    )
                else:
                    future = executor.submit(
                        self.simulate_single,
                        scenario
                    )
                
                futures[future] = scenario_id
            
            for future in as_completed(futures):
                scenario_id = futures[future]
                results[scenario_id] = future.result()
        
        return results
    
    def compare_simulations(self, summaries: List[SimulationSummary]) -> Dict:
        """
        Confronta risultati di più simulazioni.
        """
        if not summaries:
            return {}
        
        comparison = {
            "scenarios": [s.scenario_id for s in summaries],
            "success_rates": {s.scenario_id: s.success_rate for s in summaries},
            "expected_values": {s.scenario_id: s.expected_value for s in summaries},
            "risk_scores": {s.scenario_id: s.risk_score for s in summaries},
            "rankings": {}
        }
        
        # Rank by success rate
        by_success = sorted(summaries, key=lambda x: x.success_rate, reverse=True)
        comparison["rankings"]["by_success"] = [s.scenario_id for s in by_success]
        
        # Rank by expected value
        by_value = sorted(summaries, key=lambda x: x.expected_value, reverse=True)
        comparison["rankings"]["by_expected_value"] = [s.scenario_id for s in by_value]
        
        # Rank by risk (lower is better)
        by_risk = sorted(summaries, key=lambda x: x.risk_score)
        comparison["rankings"]["by_risk"] = [s.scenario_id for s in by_risk]
        
        # Best overall (weighted)
        scores = {}
        for s in summaries:
            scores[s.scenario_id] = (
                s.success_rate * 0.4 +
                s.expected_value * 0.4 +
                (1 - s.risk_score) * 0.2
            )
        comparison["best_overall"] = max(scores.keys(), key=lambda x: scores[x])
        
        return comparison
    
    # === Status & Info ===
    
    def get_status(self) -> Dict:
        """Stato del motore"""
        return {
            "max_workers": self.max_workers,
            "seed": self.seed,
            "total_simulations": self._total_simulations,
            "total_iterations": self._total_iterations,
            "cached_results": len(self.results_cache),
            "custom_distributions": list(self.custom_distributions.keys())
        }
    
    def clear_cache(self):
        """Pulisce cache risultati"""
        with self._lock:
            self.results_cache.clear()
