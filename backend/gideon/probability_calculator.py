# /backend/gideon/probability_calculator.py
"""
🔮 GIDEON 3.0 - Probability Calculator
Calcola probabilità di successo e fallimento per decisioni e azioni.
NON esegue azioni - fornisce solo analisi probabilistica.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math
import statistics
import logging

logger = logging.getLogger(__name__)


class ProbabilityMethod(Enum):
    """Metodi di calcolo probabilità"""
    FREQUENTIST = "frequentist"         # Basato su frequenza storica
    BAYESIAN = "bayesian"               # Inferenza Bayesiana
    SUBJECTIVE = "subjective"           # Stima soggettiva/esperti
    COMBINED = "combined"               # Combinazione pesata
    MONTE_CARLO = "monte_carlo"         # Basato su simulazione
    MACHINE_LEARNING = "ml"             # Basato su modello ML


class ConfidenceLevel(Enum):
    """Livelli di confidenza"""
    VERY_LOW = "very_low"       # < 50%
    LOW = "low"                 # 50-65%
    MODERATE = "moderate"       # 65-80%
    HIGH = "high"               # 80-90%
    VERY_HIGH = "very_high"     # > 90%


@dataclass
class ProbabilityFactor:
    """Fattore che influenza la probabilità"""
    name: str
    weight: float  # 0.0 - 1.0
    value: float   # 0.0 - 1.0 (contributo positivo)
    confidence: float = 0.8
    description: str = ""
    source: str = "calculated"
    
    def weighted_contribution(self) -> float:
        return self.weight * self.value * self.confidence


@dataclass
class ProbabilityResult:
    """Risultato calcolo probabilità"""
    success_probability: float
    failure_probability: float
    confidence_level: ConfidenceLevel
    confidence_score: float
    method: ProbabilityMethod
    
    # Dettagli
    factors: List[ProbabilityFactor] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Intervalli
    success_interval: Tuple[float, float] = (0.0, 1.0)
    
    # Metadata
    calculated_at: datetime = field(default_factory=datetime.now)
    data_points: int = 0
    
    def to_dict(self) -> dict:
        return {
            "success_probability": round(self.success_probability, 4),
            "failure_probability": round(self.failure_probability, 4),
            "confidence_level": self.confidence_level.value,
            "confidence_score": round(self.confidence_score, 4),
            "method": self.method.value,
            "factors": [
                {"name": f.name, "weight": f.weight, "value": f.value, "contribution": f.weighted_contribution()}
                for f in self.factors
            ],
            "breakdown": self.breakdown,
            "success_interval": [round(x, 4) for x in self.success_interval],
            "calculated_at": self.calculated_at.isoformat(),
            "data_points": self.data_points
        }


@dataclass 
class ConditionalProbability:
    """Probabilità condizionale P(A|B)"""
    event_a: str
    given_b: str
    probability: float
    joint_probability: float  # P(A ∩ B)
    marginal_b: float         # P(B)
    
    def to_dict(self) -> dict:
        return {
            "event": self.event_a,
            "given": self.given_b,
            "probability": round(self.probability, 4),
            "joint": round(self.joint_probability, 4),
            "marginal": round(self.marginal_b, 4)
        }


@dataclass
class BayesianUpdate:
    """Aggiornamento Bayesiano"""
    prior: float
    likelihood: float
    evidence: float
    posterior: float
    update_factor: float
    
    def to_dict(self) -> dict:
        return {
            "prior": round(self.prior, 4),
            "likelihood": round(self.likelihood, 4),
            "evidence": round(self.evidence, 4),
            "posterior": round(self.posterior, 4),
            "update_factor": round(self.update_factor, 4)
        }


class ProbabilityCalculator:
    """
    Calcolatore di probabilità avanzato per Gideon.
    Supporta metodi frequentisti, Bayesiani e combinati.
    """
    
    def __init__(self):
        # Prior defaults
        self.default_priors: Dict[str, float] = {
            "action_success": 0.75,
            "system_stable": 0.90,
            "user_satisfied": 0.80,
            "resource_available": 0.85,
            "network_connected": 0.95,
            "file_exists": 0.70,
            "permission_granted": 0.80,
            "api_responsive": 0.90
        }
        
        # Likelihood tables
        self.likelihood_tables: Dict[str, Dict[str, float]] = {}
        
        # Evidence history
        self.evidence_history: List[Dict] = []
        
        # Factor weights
        self.factor_weights = {
            "historical_success": 0.25,
            "system_state": 0.20,
            "resource_availability": 0.15,
            "complexity": 0.15,
            "user_context": 0.10,
            "time_constraints": 0.10,
            "external_dependencies": 0.05
        }
    
    # === Core Probability Calculation ===
    
    def calculate(self, context: Dict,
                  method: ProbabilityMethod = ProbabilityMethod.COMBINED) -> ProbabilityResult:
        """
        Calcola probabilità di successo per un'azione/decisione.
        
        Args:
            context: Contesto con informazioni sull'azione
            method: Metodo di calcolo
        
        Returns:
            ProbabilityResult completo
        """
        if method == ProbabilityMethod.FREQUENTIST:
            return self._calculate_frequentist(context)
        elif method == ProbabilityMethod.BAYESIAN:
            return self._calculate_bayesian(context)
        elif method == ProbabilityMethod.SUBJECTIVE:
            return self._calculate_subjective(context)
        elif method == ProbabilityMethod.COMBINED:
            return self._calculate_combined(context)
        else:
            return self._calculate_combined(context)
    
    def _calculate_frequentist(self, context: Dict) -> ProbabilityResult:
        """Calcolo frequentista basato su dati storici"""
        history = context.get("history", [])
        action_type = context.get("action_type", "generic")
        
        if not history:
            # Usa prior di default
            success_prob = self.default_priors.get(action_type, 0.70)
            data_points = 0
        else:
            # Calcola da frequenza
            successes = sum(1 for h in history if h.get("success", False))
            data_points = len(history)
            success_prob = successes / data_points if data_points > 0 else 0.5
        
        # Confidence basata su sample size
        confidence = self._sample_size_confidence(data_points)
        
        # Intervallo di confidenza (Wilson score)
        interval = self._wilson_interval(success_prob, data_points)
        
        return ProbabilityResult(
            success_probability=success_prob,
            failure_probability=1 - success_prob,
            confidence_level=self._get_confidence_level(confidence),
            confidence_score=confidence,
            method=ProbabilityMethod.FREQUENTIST,
            success_interval=interval,
            data_points=data_points,
            breakdown={"frequentist": success_prob}
        )
    
    def _calculate_bayesian(self, context: Dict) -> ProbabilityResult:
        """Calcolo Bayesiano con aggiornamento prior"""
        action_type = context.get("action_type", "action_success")
        prior = context.get("prior", self.default_priors.get(action_type, 0.5))
        
        # Raccogli evidenze
        evidence_factors = self._extract_evidence(context)
        
        # Aggiorna prior con ogni evidenza
        posterior = prior
        updates = []
        
        for evidence in evidence_factors:
            likelihood = evidence.get("likelihood", 0.5)
            marginal = evidence.get("marginal", 0.5)
            
            if marginal > 0:
                # Bayes: P(H|E) = P(E|H) * P(H) / P(E)
                new_posterior = (likelihood * posterior) / marginal
                new_posterior = max(0.01, min(0.99, new_posterior))  # Clamp
                
                updates.append(BayesianUpdate(
                    prior=posterior,
                    likelihood=likelihood,
                    evidence=marginal,
                    posterior=new_posterior,
                    update_factor=new_posterior / posterior if posterior > 0 else 1.0
                ))
                
                posterior = new_posterior
        
        # Confidence da numero di update
        confidence = min(0.95, 0.5 + len(updates) * 0.1)
        
        return ProbabilityResult(
            success_probability=posterior,
            failure_probability=1 - posterior,
            confidence_level=self._get_confidence_level(confidence),
            confidence_score=confidence,
            method=ProbabilityMethod.BAYESIAN,
            breakdown={
                "prior": prior,
                "posterior": posterior,
                "updates": len(updates)
            }
        )
    
    def _calculate_subjective(self, context: Dict) -> ProbabilityResult:
        """Stima soggettiva basata su fattori"""
        factors = []
        
        # System state factor
        system_health = context.get("system_health", 0.8)
        factors.append(ProbabilityFactor(
            name="system_state",
            weight=self.factor_weights["system_state"],
            value=system_health,
            description="Health dello stato sistema"
        ))
        
        # Resource availability
        resources = context.get("resources_available", 0.9)
        factors.append(ProbabilityFactor(
            name="resource_availability",
            weight=self.factor_weights["resource_availability"],
            value=resources,
            description="Disponibilità risorse"
        ))
        
        # Complexity factor (inversamente proporzionale)
        complexity = context.get("complexity", 0.5)
        factors.append(ProbabilityFactor(
            name="complexity",
            weight=self.factor_weights["complexity"],
            value=1 - complexity,  # Lower complexity = higher success
            description="Inversamente proporzionale alla complessità"
        ))
        
        # Historical success
        hist_success = context.get("historical_success_rate", 0.75)
        factors.append(ProbabilityFactor(
            name="historical_success",
            weight=self.factor_weights["historical_success"],
            value=hist_success,
            description="Tasso di successo storico"
        ))
        
        # User context
        user_exp = context.get("user_experience", 0.7)
        factors.append(ProbabilityFactor(
            name="user_context",
            weight=self.factor_weights["user_context"],
            value=user_exp,
            description="Esperienza/contesto utente"
        ))
        
        # Time constraints
        time_pressure = context.get("time_pressure", 0.3)
        factors.append(ProbabilityFactor(
            name="time_constraints",
            weight=self.factor_weights["time_constraints"],
            value=1 - time_pressure,
            description="Assenza di pressione temporale"
        ))
        
        # External dependencies
        ext_reliability = context.get("external_reliability", 0.85)
        factors.append(ProbabilityFactor(
            name="external_dependencies",
            weight=self.factor_weights["external_dependencies"],
            value=ext_reliability,
            description="Affidabilità dipendenze esterne"
        ))
        
        # Calcola probabilità pesata
        total_weight = sum(f.weight for f in factors)
        success_prob = sum(f.weighted_contribution() for f in factors) / total_weight
        
        # Confidence media
        confidence = statistics.mean(f.confidence for f in factors)
        
        return ProbabilityResult(
            success_probability=success_prob,
            failure_probability=1 - success_prob,
            confidence_level=self._get_confidence_level(confidence),
            confidence_score=confidence,
            method=ProbabilityMethod.SUBJECTIVE,
            factors=factors,
            breakdown={f.name: f.weighted_contribution() for f in factors}
        )
    
    def _calculate_combined(self, context: Dict) -> ProbabilityResult:
        """Combinazione pesata di tutti i metodi"""
        # Calcola con tutti i metodi
        freq = self._calculate_frequentist(context)
        bayes = self._calculate_bayesian(context)
        subj = self._calculate_subjective(context)
        
        # Pesi basati sulla confidenza di ogni metodo
        weights = {
            "frequentist": freq.confidence_score * (1 + freq.data_points * 0.01),
            "bayesian": bayes.confidence_score,
            "subjective": subj.confidence_score * 0.8  # Leggermente penalizzato
        }
        
        total_weight = sum(weights.values())
        
        # Probabilità combinata
        combined_prob = (
            weights["frequentist"] * freq.success_probability +
            weights["bayesian"] * bayes.success_probability +
            weights["subjective"] * subj.success_probability
        ) / total_weight
        
        # Confidence combinata
        combined_confidence = statistics.mean([
            freq.confidence_score,
            bayes.confidence_score,
            subj.confidence_score
        ])
        
        # Intervallo combinato
        intervals = [freq.success_interval, (bayes.success_probability * 0.9, bayes.success_probability * 1.1)]
        combined_interval = (
            min(i[0] for i in intervals),
            max(i[1] for i in intervals)
        )
        
        return ProbabilityResult(
            success_probability=combined_prob,
            failure_probability=1 - combined_prob,
            confidence_level=self._get_confidence_level(combined_confidence),
            confidence_score=combined_confidence,
            method=ProbabilityMethod.COMBINED,
            factors=subj.factors,  # Usa fattori dal soggettivo
            breakdown={
                "frequentist": freq.success_probability,
                "bayesian": bayes.success_probability,
                "subjective": subj.success_probability,
                "weights": weights
            },
            success_interval=combined_interval,
            data_points=freq.data_points
        )
    
    # === Utility Functions ===
    
    def _extract_evidence(self, context: Dict) -> List[Dict]:
        """Estrae evidenze dal contesto per Bayes"""
        evidence = []
        
        # System state evidence
        if "system_health" in context:
            health = context["system_health"]
            evidence.append({
                "name": "system_health",
                "likelihood": 0.9 if health > 0.8 else 0.5 if health > 0.5 else 0.2,
                "marginal": 0.7
            })
        
        # Recent failures
        if "recent_failures" in context:
            failures = context["recent_failures"]
            evidence.append({
                "name": "recent_failures",
                "likelihood": max(0.1, 1 - failures * 0.2),
                "marginal": 0.6
            })
        
        # Resource availability
        if "resources_available" in context:
            res = context["resources_available"]
            evidence.append({
                "name": "resources",
                "likelihood": res,
                "marginal": 0.75
            })
        
        return evidence
    
    def _sample_size_confidence(self, n: int) -> float:
        """Confidenza basata su sample size"""
        if n == 0:
            return 0.3
        elif n < 10:
            return 0.4 + n * 0.03
        elif n < 50:
            return 0.7 + (n - 10) * 0.005
        elif n < 100:
            return 0.9 + (n - 50) * 0.001
        else:
            return min(0.98, 0.95 + (n - 100) * 0.0001)
    
    def _wilson_interval(self, p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
        """Calcola intervallo di Wilson per proporzioni"""
        if n == 0:
            return (0.0, 1.0)
        
        denominator = 1 + z*z/n
        centre = p + z*z/(2*n)
        adjustment = z * math.sqrt((p*(1-p) + z*z/(4*n))/n)
        
        lower = max(0, (centre - adjustment) / denominator)
        upper = min(1, (centre + adjustment) / denominator)
        
        return (lower, upper)
    
    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Converte score in livello"""
        if score < 0.50:
            return ConfidenceLevel.VERY_LOW
        elif score < 0.65:
            return ConfidenceLevel.LOW
        elif score < 0.80:
            return ConfidenceLevel.MODERATE
        elif score < 0.90:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    # === Advanced Calculations ===
    
    def conditional_probability(self, event_a: str, given_b: str,
                                 joint_data: Dict) -> ConditionalProbability:
        """
        Calcola P(A|B) = P(A ∩ B) / P(B)
        """
        # Estrai probabilità dal dato
        p_joint = joint_data.get("joint", 0.0)
        p_b = joint_data.get("marginal_b", 0.5)
        
        if p_b > 0:
            p_a_given_b = p_joint / p_b
        else:
            p_a_given_b = 0.0
        
        return ConditionalProbability(
            event_a=event_a,
            given_b=given_b,
            probability=p_a_given_b,
            joint_probability=p_joint,
            marginal_b=p_b
        )
    
    def chain_probability(self, events: List[Dict]) -> float:
        """
        Calcola probabilità di una catena di eventi (indipendenti).
        P(A ∩ B ∩ C) = P(A) * P(B) * P(C)
        """
        if not events:
            return 1.0
        
        total = 1.0
        for event in events:
            prob = event.get("probability", 0.5)
            total *= prob
        
        return total
    
    def at_least_one_success(self, probabilities: List[float]) -> float:
        """
        Probabilità che almeno un evento abbia successo.
        P(almeno 1) = 1 - P(tutti falliscono)
        """
        if not probabilities:
            return 0.0
        
        all_fail = 1.0
        for p in probabilities:
            all_fail *= (1 - p)
        
        return 1 - all_fail
    
    def expected_attempts(self, success_prob: float) -> float:
        """
        Numero atteso di tentativi per un successo.
        E[X] = 1/p (distribuzione geometrica)
        """
        if success_prob <= 0:
            return float('inf')
        return 1 / success_prob
    
    def success_in_n_attempts(self, success_prob: float, n: int) -> float:
        """
        Probabilità di almeno un successo in n tentativi.
        """
        return 1 - (1 - success_prob) ** n
    
    # === Scenario Analysis ===
    
    def analyze_scenario(self, scenario: Dict) -> Dict:
        """
        Analisi probabilistica completa di uno scenario.
        """
        # Calcola probabilità base
        base_result = self.calculate(scenario, ProbabilityMethod.COMBINED)
        
        # Calcola varianti
        optimistic = self._adjust_for_optimism(scenario)
        pessimistic = self._adjust_for_pessimism(scenario)
        
        opt_result = self.calculate(optimistic, ProbabilityMethod.COMBINED)
        pess_result = self.calculate(pessimistic, ProbabilityMethod.COMBINED)
        
        # Expected value
        outcomes = scenario.get("outcomes", [])
        expected_value = self._calculate_expected_value(base_result.success_probability, outcomes)
        
        return {
            "base_case": base_result.to_dict(),
            "optimistic_case": opt_result.to_dict(),
            "pessimistic_case": pess_result.to_dict(),
            "expected_value": expected_value,
            "probability_range": {
                "low": pess_result.success_probability,
                "mid": base_result.success_probability,
                "high": opt_result.success_probability
            },
            "recommendation": self._generate_recommendation(base_result)
        }
    
    def _adjust_for_optimism(self, context: Dict) -> Dict:
        """Aggiusta contesto per scenario ottimistico"""
        optimistic = context.copy()
        for key, value in optimistic.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                optimistic[key] = min(1.0, value * 1.15)
        return optimistic
    
    def _adjust_for_pessimism(self, context: Dict) -> Dict:
        """Aggiusta contesto per scenario pessimistico"""
        pessimistic = context.copy()
        for key, value in pessimistic.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                pessimistic[key] = max(0.0, value * 0.85)
        return pessimistic
    
    def _calculate_expected_value(self, success_prob: float, outcomes: List[Dict]) -> float:
        """Calcola valore atteso"""
        if not outcomes:
            return success_prob  # Semplice se no outcomes definiti
        
        expected = 0.0
        for outcome in outcomes:
            prob = outcome.get("probability", 0.5)
            value = outcome.get("value", 0.0)
            expected += prob * value
        
        return expected
    
    def _generate_recommendation(self, result: ProbabilityResult) -> str:
        """Genera raccomandazione basata sul risultato"""
        prob = result.success_probability
        conf = result.confidence_score
        
        if prob >= 0.85 and conf >= 0.7:
            return "PROCEED - Alta probabilità di successo con buona confidenza"
        elif prob >= 0.70 and conf >= 0.6:
            return "PROCEED_WITH_CAUTION - Probabilità accettabile, monitorare"
        elif prob >= 0.50:
            return "EVALUATE_ALTERNATIVES - Probabilità marginale, considerare alternative"
        else:
            return "AVOID - Bassa probabilità di successo, ripianificare"
    
    # === Status ===
    
    def get_status(self) -> Dict:
        """Stato del calcolatore"""
        return {
            "default_priors": len(self.default_priors),
            "factor_weights": self.factor_weights,
            "evidence_history_size": len(self.evidence_history),
            "likelihood_tables": len(self.likelihood_tables)
        }
