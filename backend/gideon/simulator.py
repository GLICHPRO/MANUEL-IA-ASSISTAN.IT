"""
🎮 GIDEON 3.0 - Simulator
Sistema di simulazione scenari con analisi probabilistica avanzata
Monte Carlo, Bayesian inference, Confidence intervals
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import random
import math
from dataclasses import dataclass, field
from enum import Enum


class SimulationType(Enum):
    """Tipi di simulazione disponibili"""
    DETERMINISTIC = "deterministic"
    MONTE_CARLO = "monte_carlo"
    BAYESIAN = "bayesian"
    SENSITIVITY = "sensitivity"


@dataclass
class SimulationConfig:
    """Configurazione per simulazioni"""
    iterations: int = 1000
    confidence_level: float = 0.95
    seed: Optional[int] = None
    parallel: bool = False


@dataclass
class ConfidenceInterval:
    """Intervallo di confidenza"""
    lower: float
    upper: float
    confidence: float
    mean: float
    std_dev: float
    
    def to_dict(self) -> dict:
        return {
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "confidence": self.confidence,
            "mean": round(self.mean, 4),
            "std_dev": round(self.std_dev, 4),
            "range": round(self.upper - self.lower, 4)
        }


@dataclass
class MonteCarloResult:
    """Risultato simulazione Monte Carlo"""
    samples: List[float]
    mean: float
    median: float
    std_dev: float
    confidence_interval: ConfidenceInterval
    percentiles: Dict[int, float]
    success_probability: float
    
    def to_dict(self) -> dict:
        return {
            "iterations": len(self.samples),
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std_dev": round(self.std_dev, 4),
            "confidence_interval": self.confidence_interval.to_dict(),
            "percentiles": {k: round(v, 4) for k, v in self.percentiles.items()},
            "success_probability": round(self.success_probability, 4)
        }


class Simulator:
    """
    Motore di simulazione avanzato di Gideon 3.0
    
    Features:
    - Simulazione Monte Carlo
    - Analisi Bayesiana
    - Intervalli di confidenza
    - Analisi di sensitività
    - Probabilità condizionate
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.simulations = []
        self.prior_knowledge: Dict[str, Dict] = {}  # Bayesian priors
        
        self.scenario_templates = {
            "system_action": self._simulate_system_action,
            "resource_change": self._simulate_resource_change,
            "automation": self._simulate_automation,
            "user_workflow": self._simulate_workflow
        }
        
        # Inizializza generatore random con seed se specificato
        if self.config.seed:
            random.seed(self.config.seed)
            
    # ============================================
    # MONTE CARLO SIMULATION
    # ============================================
    
    async def monte_carlo(self, scenario: dict, iterations: int = None) -> MonteCarloResult:
        """
        Esegue simulazione Monte Carlo per analisi probabilistica
        
        Args:
            scenario: Scenario da simulare
            iterations: Numero di iterazioni (default: config.iterations)
            
        Returns:
            MonteCarloResult con statistiche complete
        """
        n = iterations or self.config.iterations
        samples = []
        successes = 0
        
        for _ in range(n):
            result = await self._single_simulation(scenario)
            samples.append(result["score"])
            if result.get("success", False):
                successes += 1
        
        # Calcola statistiche
        samples.sort()
        mean = sum(samples) / n
        median = samples[n // 2] if n % 2 else (samples[n//2 - 1] + samples[n//2]) / 2
        variance = sum((x - mean) ** 2 for x in samples) / n
        std_dev = math.sqrt(variance)
        
        # Intervallo di confidenza
        ci = self._calculate_confidence_interval(samples, self.config.confidence_level)
        
        # Percentili
        percentiles = {
            5: samples[int(n * 0.05)],
            25: samples[int(n * 0.25)],
            50: median,
            75: samples[int(n * 0.75)],
            95: samples[int(n * 0.95)]
        }
        
        return MonteCarloResult(
            samples=samples,
            mean=mean,
            median=median,
            std_dev=std_dev,
            confidence_interval=ci,
            percentiles=percentiles,
            success_probability=successes / n
        )
    
    async def _single_simulation(self, scenario: dict) -> dict:
        """Singola iterazione di simulazione con variabilità"""
        base_prob = scenario.get("probability", 0.7)
        base_score = scenario.get("score", 0.5)
        
        # Aggiungi rumore gaussiano
        noise = random.gauss(0, 0.1)
        actual_prob = max(0, min(1, base_prob + noise))
        
        success = random.random() < actual_prob
        
        # Score con variabilità
        score_noise = random.gauss(0, 0.05)
        actual_score = base_score + (0.2 if success else -0.2) + score_noise
        actual_score = max(0, min(1, actual_score))
        
        return {
            "success": success,
            "score": actual_score,
            "probability": actual_prob
        }
    
    def _calculate_confidence_interval(self, samples: List[float], 
                                       confidence: float) -> ConfidenceInterval:
        """Calcola intervallo di confidenza"""
        n = len(samples)
        mean = sum(samples) / n
        variance = sum((x - mean) ** 2 for x in samples) / n
        std_dev = math.sqrt(variance)
        
        # Z-score per livello di confidenza
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        margin = z * (std_dev / math.sqrt(n))
        
        return ConfidenceInterval(
            lower=mean - margin,
            upper=mean + margin,
            confidence=confidence,
            mean=mean,
            std_dev=std_dev
        )
    
    # ============================================
    # BAYESIAN ANALYSIS
    # ============================================
    
    async def bayesian_update(self, scenario_id: str, observation: dict) -> dict:
        """
        Aggiorna credenze su uno scenario usando Bayes
        
        P(H|E) = P(E|H) * P(H) / P(E)
        """
        # Recupera prior o inizializza
        if scenario_id not in self.prior_knowledge:
            self.prior_knowledge[scenario_id] = {
                "success_prob": 0.5,  # Prior uniforme
                "observations": 0,
                "successes": 0
            }
        
        prior = self.prior_knowledge[scenario_id]
        
        # Aggiorna con osservazione (Beta-Binomial model)
        was_success = observation.get("success", False)
        prior["observations"] += 1
        if was_success:
            prior["successes"] += 1
        
        # Posterior usando Beta distribution (conjugate prior)
        alpha = prior["successes"] + 1
        beta = prior["observations"] - prior["successes"] + 1
        
        posterior_mean = alpha / (alpha + beta)
        posterior_variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        
        prior["success_prob"] = posterior_mean
        
        return {
            "scenario_id": scenario_id,
            "posterior_probability": round(posterior_mean, 4),
            "posterior_variance": round(posterior_variance, 6),
            "confidence": 1 - math.sqrt(posterior_variance),
            "observations": prior["observations"],
            "alpha": alpha,
            "beta": beta
        }
    
    async def bayesian_predict(self, scenario: dict) -> dict:
        """Predice outcome usando conoscenza Bayesiana"""
        scenario_id = scenario.get("id", "unknown")
        
        if scenario_id in self.prior_knowledge:
            prior = self.prior_knowledge[scenario_id]
            predicted_prob = prior["success_prob"]
            confidence = min(prior["observations"] / 100, 1.0)  # Cresce con osservazioni
        else:
            predicted_prob = scenario.get("probability", 0.5)
            confidence = 0.3
        
        return {
            "scenario_id": scenario_id,
            "predicted_success_probability": round(predicted_prob, 4),
            "confidence": round(confidence, 4),
            "has_prior_knowledge": scenario_id in self.prior_knowledge,
            "recommendation": "proceed" if predicted_prob > 0.6 else "caution" if predicted_prob > 0.4 else "avoid"
        }
    
    # ============================================
    # SENSITIVITY ANALYSIS
    # ============================================
    
    async def sensitivity_analysis(self, scenario: dict, 
                                   parameter: str,
                                   range_values: List[float] = None) -> dict:
        """
        Analisi di sensitività: come varia l'outcome al variare di un parametro
        """
        if range_values is None:
            base_value = scenario.get(parameter, 0.5)
            range_values = [base_value * x for x in [0.5, 0.75, 1.0, 1.25, 1.5]]
        
        results = []
        
        for value in range_values:
            test_scenario = scenario.copy()
            test_scenario[parameter] = value
            
            mc_result = await self.monte_carlo(test_scenario, iterations=100)
            
            results.append({
                "parameter_value": round(value, 4),
                "mean_score": mc_result.mean,
                "success_probability": mc_result.success_probability,
                "std_dev": mc_result.std_dev
            })
        
        # Calcola elasticità
        if len(results) > 1:
            base_idx = len(results) // 2
            elasticity = (results[-1]["mean_score"] - results[0]["mean_score"]) / \
                        (range_values[-1] - range_values[0])
        else:
            elasticity = 0
        
        return {
            "parameter": parameter,
            "base_value": scenario.get(parameter, 0.5),
            "results": results,
            "elasticity": round(elasticity, 4),
            "sensitivity": "high" if abs(elasticity) > 0.5 else "medium" if abs(elasticity) > 0.2 else "low"
        }
    
    # ============================================
    # SCENARIO COMPARISON
    # ============================================
    
    async def what_if_analysis(self, base_scenario: dict, 
                               modifications: List[dict]) -> dict:
        """
        Analisi what-if: confronta scenario base con varianti
        """
        base_result = await self.monte_carlo(base_scenario, iterations=500)
        
        comparisons = [{
            "name": "base",
            "modifications": {},
            "result": base_result.to_dict()
        }]
        
        for mod in modifications:
            modified_scenario = base_scenario.copy()
            modified_scenario.update(mod.get("changes", {}))
            
            mod_result = await self.monte_carlo(modified_scenario, iterations=500)
            
            comparisons.append({
                "name": mod.get("name", "variant"),
                "modifications": mod.get("changes", {}),
                "result": mod_result.to_dict(),
                "delta_mean": round(mod_result.mean - base_result.mean, 4),
                "delta_success": round(mod_result.success_probability - base_result.success_probability, 4)
            })
        
        # Trova la migliore variante
        best = max(comparisons, key=lambda x: x["result"]["mean"])
        
        return {
            "base_scenario": base_scenario,
            "comparisons": comparisons,
            "best_variant": best["name"],
            "recommendation": f"La variante '{best['name']}' offre i risultati migliori"
        }
        
    async def run(self, scenario: dict) -> dict:
        """
        Esegue una simulazione di scenario
        
        Args:
            scenario: Dizionario con parametri dello scenario
            
        Returns:
            Risultati della simulazione
        """
        scenario_type = scenario.get("type", "generic")
        
        simulation_result = {
            "scenario_id": f"sim_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": scenario_type,
            "input": scenario,
            "timestamp": datetime.now().isoformat(),
            "outcomes": [],
            "best_outcome": None,
            "worst_outcome": None,
            "expected_outcome": None
        }
        
        # Esegui simulazione specifica se disponibile
        if scenario_type in self.scenario_templates:
            outcomes = await self.scenario_templates[scenario_type](scenario)
        else:
            outcomes = await self._simulate_generic(scenario)
            
        simulation_result["outcomes"] = outcomes
        
        # Determina outcomes principali
        if outcomes:
            sorted_outcomes = sorted(outcomes, key=lambda x: x.get("score", 0), reverse=True)
            simulation_result["best_outcome"] = sorted_outcomes[0]
            simulation_result["worst_outcome"] = sorted_outcomes[-1]
            simulation_result["expected_outcome"] = self._calculate_expected(outcomes)
        
        # Salva simulazione
        self.simulations.append(simulation_result)
        
        return simulation_result
    
    async def generate_scenarios(self, predictions: dict) -> list:
        """
        Genera scenari possibili basati sulle previsioni
        
        Args:
            predictions: Output del Predictor
            
        Returns:
            Lista di scenari con probabilità
        """
        scenarios = []
        
        # Estrai azioni suggerite dalle previsioni
        combined = predictions.get("combined", {})
        suggested_actions = combined.get("all_suggested_actions", [])
        
        for action in suggested_actions[:5]:  # Top 5 azioni
            scenario = await self._generate_scenario_for_action(action)
            scenarios.append(scenario)
        
        # Aggiungi scenario "do nothing"
        scenarios.append({
            "id": "no_action",
            "name": "Nessuna azione",
            "description": "Mantieni lo stato attuale",
            "action": None,
            "probability": 1.0,
            "risk": 0.0,
            "benefit": 0.0,
            "score": 0.3
        })
        
        # Ordina per score
        scenarios.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return scenarios
    
    async def compare_scenarios(self, scenarios: list) -> dict:
        """Confronta più scenari e restituisce analisi comparativa"""
        if not scenarios:
            return {"error": "Nessuno scenario da confrontare"}
        
        comparison = {
            "total_scenarios": len(scenarios),
            "best_by_benefit": None,
            "lowest_risk": None,
            "best_balanced": None,
            "comparison_matrix": []
        }
        
        # Trova migliori per categoria
        sorted_by_benefit = sorted(scenarios, key=lambda x: x.get("benefit", 0), reverse=True)
        sorted_by_risk = sorted(scenarios, key=lambda x: x.get("risk", 1))
        sorted_by_score = sorted(scenarios, key=lambda x: x.get("score", 0), reverse=True)
        
        comparison["best_by_benefit"] = sorted_by_benefit[0] if sorted_by_benefit else None
        comparison["lowest_risk"] = sorted_by_risk[0] if sorted_by_risk else None
        comparison["best_balanced"] = sorted_by_score[0] if sorted_by_score else None
        
        # Crea matrice comparativa
        for scenario in scenarios:
            comparison["comparison_matrix"].append({
                "id": scenario.get("id"),
                "name": scenario.get("name"),
                "benefit": scenario.get("benefit", 0),
                "risk": scenario.get("risk", 0),
                "score": scenario.get("score", 0),
                "rank_benefit": sorted_by_benefit.index(scenario) + 1,
                "rank_risk": sorted_by_risk.index(scenario) + 1,
                "rank_overall": sorted_by_score.index(scenario) + 1
            })
            
        return comparison
    
    async def _simulate_system_action(self, scenario: dict) -> list:
        """Simula un'azione di sistema"""
        action = scenario.get("action", "unknown")
        outcomes = []
        
        # Outcome positivo
        outcomes.append({
            "type": "success",
            "probability": 0.85,
            "description": f"Azione '{action}' completata con successo",
            "impact": {
                "cpu": -5,  # Riduzione uso CPU
                "memory": -3,
                "time_ms": 500
            },
            "score": 0.85
        })
        
        # Outcome neutro
        outcomes.append({
            "type": "partial",
            "probability": 0.10,
            "description": f"Azione '{action}' completata parzialmente",
            "impact": {
                "cpu": 0,
                "memory": 0,
                "time_ms": 1000
            },
            "score": 0.50
        })
        
        # Outcome negativo
        outcomes.append({
            "type": "failure",
            "probability": 0.05,
            "description": f"Azione '{action}' fallita",
            "impact": {
                "cpu": 2,
                "memory": 1,
                "time_ms": 200
            },
            "score": 0.10
        })
        
        return outcomes
    
    async def _simulate_resource_change(self, scenario: dict) -> list:
        """Simula cambiamento risorse"""
        resource = scenario.get("resource", "memory")
        change = scenario.get("change", 0)
        
        outcomes = []
        
        if change < 0:  # Liberazione risorse
            outcomes.append({
                "type": "success",
                "probability": 0.90,
                "description": f"Liberati {abs(change)}% di {resource}",
                "impact": {resource: change},
                "score": 0.90
            })
        else:  # Consumo risorse
            outcomes.append({
                "type": "warning",
                "probability": 0.70,
                "description": f"Consumo {change}% di {resource}",
                "impact": {resource: change},
                "score": 0.50
            })
            
        return outcomes
    
    async def _simulate_automation(self, scenario: dict) -> list:
        """Simula un'automazione"""
        automation_name = scenario.get("name", "automazione")
        steps = scenario.get("steps", 1)
        
        # Probabilità successo decresce con la complessità
        success_prob = max(0.5, 0.95 - (steps * 0.05))
        
        return [
            {
                "type": "success",
                "probability": success_prob,
                "description": f"Automazione '{automation_name}' completata ({steps} step)",
                "score": success_prob
            },
            {
                "type": "failure",
                "probability": 1 - success_prob,
                "description": f"Automazione '{automation_name}' interrotta",
                "score": 0.1
            }
        ]
    
    async def _simulate_workflow(self, scenario: dict) -> list:
        """Simula un workflow utente"""
        return [
            {
                "type": "completed",
                "probability": 0.80,
                "description": "Workflow completato",
                "score": 0.80
            },
            {
                "type": "interrupted",
                "probability": 0.15,
                "description": "Workflow interrotto dall'utente",
                "score": 0.40
            },
            {
                "type": "error",
                "probability": 0.05,
                "description": "Errore durante il workflow",
                "score": 0.10
            }
        ]
    
    async def _simulate_generic(self, scenario: dict) -> list:
        """Simulazione generica"""
        return [
            {
                "type": "success",
                "probability": 0.70,
                "description": "Operazione riuscita",
                "score": 0.70
            },
            {
                "type": "partial",
                "probability": 0.20,
                "description": "Operazione parziale",
                "score": 0.40
            },
            {
                "type": "failure",
                "probability": 0.10,
                "description": "Operazione fallita",
                "score": 0.10
            }
        ]
    
    async def _generate_scenario_for_action(self, action: dict) -> dict:
        """Genera uno scenario per un'azione suggerita"""
        action_name = action.get("action", "unknown")
        priority = action.get("priority", 0.5)
        
        # Calcola metriche scenario
        benefit = priority * 0.8 + random.uniform(0, 0.2)
        risk = (1 - priority) * 0.3 + random.uniform(0, 0.1)
        score = benefit * 0.7 - risk * 0.3
        
        return {
            "id": f"scenario_{action_name}",
            "name": action.get("reason", action_name),
            "description": f"Esegui azione: {action_name}",
            "action": action_name,
            "probability": priority,
            "risk": round(risk, 2),
            "benefit": round(benefit, 2),
            "score": round(score, 2)
        }
    
    def _calculate_expected(self, outcomes: list) -> dict:
        """Calcola l'outcome atteso (media pesata)"""
        if not outcomes:
            return {}
        
        total_prob = sum(o.get("probability", 0) for o in outcomes)
        if total_prob == 0:
            return outcomes[0]
        
        expected_score = sum(
            o.get("score", 0) * o.get("probability", 0) 
            for o in outcomes
        ) / total_prob
        
        return {
            "type": "expected",
            "probability": 1.0,
            "description": "Risultato atteso (media pesata)",
            "score": round(expected_score, 2)
        }
    
    # ============================================
    # ADVANCED PROBABILISTIC ANALYSIS
    # ============================================
    
    async def risk_analysis(self, scenario: dict) -> dict:
        """
        Analisi dettagliata del rischio di uno scenario
        
        Calcola:
        - Value at Risk (VaR)
        - Expected Shortfall (CVaR)
        - Probabilità di perdita
        - Risk-adjusted return
        """
        mc_result = await self.monte_carlo(scenario, iterations=1000)
        samples = mc_result.samples
        
        # Value at Risk (95%)
        var_95 = sorted(samples)[int(len(samples) * 0.05)]
        
        # Expected Shortfall (media delle perdite oltre VaR)
        tail_losses = [s for s in samples if s <= var_95]
        cvar = sum(tail_losses) / len(tail_losses) if tail_losses else var_95
        
        # Probabilità di fallimento (score < 0.3)
        failure_prob = len([s for s in samples if s < 0.3]) / len(samples)
        
        # Risk-adjusted return (Sharpe-like ratio)
        risk_free = 0.3  # baseline score
        if mc_result.std_dev > 0:
            sharpe = (mc_result.mean - risk_free) / mc_result.std_dev
        else:
            sharpe = 0
        
        return {
            "scenario": scenario.get("id", "unknown"),
            "expected_score": round(mc_result.mean, 4),
            "volatility": round(mc_result.std_dev, 4),
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar, 4),
            "failure_probability": round(failure_prob, 4),
            "risk_adjusted_return": round(sharpe, 4),
            "risk_level": "high" if failure_prob > 0.3 else "medium" if failure_prob > 0.1 else "low",
            "recommendation": "proceed" if sharpe > 0.5 and failure_prob < 0.2 else "caution" if sharpe > 0 else "avoid"
        }
    
    async def multi_scenario_simulation(self, scenarios: List[dict]) -> dict:
        """
        Simula multipli scenari in parallelo e compara risultati
        """
        results = []
        
        for scenario in scenarios:
            mc = await self.monte_carlo(scenario, iterations=500)
            risk = await self.risk_analysis(scenario)
            
            results.append({
                "scenario_id": scenario.get("id"),
                "scenario_name": scenario.get("name"),
                "monte_carlo": mc.to_dict(),
                "risk_analysis": risk
            })
        
        # Trova il migliore per vari criteri
        best_expected = max(results, key=lambda x: x["monte_carlo"]["mean"])
        lowest_risk = min(results, key=lambda x: x["risk_analysis"]["failure_probability"])
        best_risk_adjusted = max(results, key=lambda x: x["risk_analysis"]["risk_adjusted_return"])
        
        return {
            "total_scenarios": len(scenarios),
            "results": results,
            "recommendations": {
                "best_expected_value": best_expected["scenario_id"],
                "lowest_risk": lowest_risk["scenario_id"],
                "best_risk_adjusted": best_risk_adjusted["scenario_id"]
            },
            "summary": f"Consiglio '{best_risk_adjusted['scenario_name']}' per il miglior rapporto rischio/rendimento"
        }
    
    async def outcome_tree(self, scenario: dict, depth: int = 3) -> dict:
        """
        Genera albero decisionale degli outcome possibili
        """
        def build_tree(current_scenario: dict, current_depth: int) -> dict:
            if current_depth == 0:
                return {
                    "scenario": current_scenario,
                    "probability": current_scenario.get("probability", 0.5),
                    "score": current_scenario.get("score", 0.5)
                }
            
            # Genera rami
            success_branch = current_scenario.copy()
            success_branch["score"] = min(1, current_scenario.get("score", 0.5) + 0.15)
            success_branch["probability"] = current_scenario.get("probability", 0.7)
            
            failure_branch = current_scenario.copy()
            failure_branch["score"] = max(0, current_scenario.get("score", 0.5) - 0.2)
            failure_branch["probability"] = 1 - current_scenario.get("probability", 0.7)
            
            return {
                "scenario": current_scenario,
                "probability": current_scenario.get("probability", 0.5),
                "branches": [
                    {"type": "success", **build_tree(success_branch, current_depth - 1)},
                    {"type": "failure", **build_tree(failure_branch, current_depth - 1)}
                ]
            }
        
        tree = build_tree(scenario, depth)
        
        # Calcola valore atteso del tree
        def calculate_ev(node: dict, path_prob: float = 1.0) -> float:
            if "branches" not in node:
                return node["score"] * path_prob
            
            total = 0
            for branch in node["branches"]:
                branch_prob = branch["probability"] * path_prob
                total += calculate_ev(branch, branch_prob)
            return total
        
        tree["expected_value"] = round(calculate_ev(tree), 4)
        
        return tree
    
    def get_simulation_history(self, limit: int = 10) -> List[dict]:
        """Restituisce storico simulazioni"""
        return self.simulations[-limit:]
    
    def get_statistics(self) -> dict:
        """Statistiche generali del simulatore"""
        return {
            "total_simulations": len(self.simulations),
            "prior_scenarios": len(self.prior_knowledge),
            "config": {
                "default_iterations": self.config.iterations,
                "confidence_level": self.config.confidence_level
            }
        }
