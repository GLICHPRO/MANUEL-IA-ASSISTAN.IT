# /backend/jarvis/confidence_evaluator.py
"""
JARVIS Confidence Evaluator - Valutazione Affidabilità Decisioni
Valuta la confidence delle decisioni basandosi su molteplici fattori.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math
import logging

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Livelli di confidence"""
    VERY_HIGH = "very_high"      # 0.9 - 1.0
    HIGH = "high"                # 0.75 - 0.9
    MODERATE = "moderate"        # 0.5 - 0.75
    LOW = "low"                  # 0.25 - 0.5
    VERY_LOW = "very_low"        # 0.0 - 0.25
    UNCERTAIN = "uncertain"      # Non calcolabile


class ConfidenceFactor(Enum):
    """Fattori che influenzano la confidence"""
    INTENT_CLARITY = "intent_clarity"           # Chiarezza dell'intento
    CONTEXT_COMPLETENESS = "context_completeness"  # Completezza del contesto
    HISTORICAL_SUCCESS = "historical_success"    # Successo storico
    DATA_QUALITY = "data_quality"               # Qualità dei dati
    AMBIGUITY = "ambiguity"                     # Livello di ambiguità
    RISK_LEVEL = "risk_level"                   # Livello di rischio
    USER_FEEDBACK = "user_feedback"             # Feedback utente
    STRATEGY_RELIABILITY = "strategy_reliability"  # Affidabilità strategia
    TIME_PRESSURE = "time_pressure"             # Pressione temporale
    RESOURCE_AVAILABILITY = "resource_availability"  # Disponibilità risorse


class RecommendationAction(Enum):
    """Azioni raccomandate basate sulla confidence"""
    PROCEED = "proceed"                    # Procedi con l'azione
    PROCEED_WITH_CAUTION = "proceed_with_caution"  # Procedi con cautela
    REQUEST_CONFIRMATION = "request_confirmation"  # Richiedi conferma
    GATHER_MORE_INFO = "gather_more_info"  # Raccogli più informazioni
    DEFER = "defer"                        # Rimanda la decisione
    ABORT = "abort"                        # Interrompi l'azione


@dataclass
class ConfidenceAssessment:
    """Valutazione completa della confidence"""
    overall_score: float
    level: ConfidenceLevel
    factors: Dict[str, float]
    weights: Dict[str, float]
    recommendation: RecommendationAction
    reasoning: str
    warnings: List[str]
    suggestions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "level": self.level.value,
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "recommendation": self.recommendation.value,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class HistoricalRecord:
    """Record storico per calcolo confidence"""
    intent: str
    action_type: str
    confidence_predicted: float
    outcome_success: bool
    timestamp: datetime


class ConfidenceEvaluator:
    """
    Valutatore di Confidence per JARVIS.
    Analizza molteplici fattori per determinare l'affidabilità delle decisioni.
    """
    
    def __init__(self, executive_memory=None):
        self.executive_memory = executive_memory
        
        # Pesi default per i fattori
        self.default_weights = {
            ConfidenceFactor.INTENT_CLARITY.value: 0.20,
            ConfidenceFactor.CONTEXT_COMPLETENESS.value: 0.15,
            ConfidenceFactor.HISTORICAL_SUCCESS.value: 0.20,
            ConfidenceFactor.DATA_QUALITY.value: 0.10,
            ConfidenceFactor.AMBIGUITY.value: 0.10,
            ConfidenceFactor.RISK_LEVEL.value: 0.10,
            ConfidenceFactor.USER_FEEDBACK.value: 0.05,
            ConfidenceFactor.STRATEGY_RELIABILITY.value: 0.05,
            ConfidenceFactor.TIME_PRESSURE.value: 0.025,
            ConfidenceFactor.RESOURCE_AVAILABILITY.value: 0.025
        }
        
        # Soglie per livelli di confidence
        self.thresholds = {
            ConfidenceLevel.VERY_HIGH: 0.90,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.MODERATE: 0.50,
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.VERY_LOW: 0.0
        }
        
        # Soglie per raccomandazioni
        self.action_thresholds = {
            RecommendationAction.PROCEED: 0.85,
            RecommendationAction.PROCEED_WITH_CAUTION: 0.70,
            RecommendationAction.REQUEST_CONFIRMATION: 0.50,
            RecommendationAction.GATHER_MORE_INFO: 0.30,
            RecommendationAction.DEFER: 0.15,
            RecommendationAction.ABORT: 0.0
        }
        
        # Historical data per calibrazione
        self.historical_records: List[HistoricalRecord] = []
        self.max_history = 1000
        
        # Calibrazione
        self.calibration_offset = 0.0
        self.calibration_scale = 1.0
        
        # Cache per performance
        self._cache: Dict[str, Tuple[ConfidenceAssessment, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
    
    def evaluate(self, intent: dict, context: dict, 
                 decision: dict = None, 
                 custom_weights: Dict[str, float] = None) -> ConfidenceAssessment:
        """
        Valuta la confidence di una decisione.
        
        Args:
            intent: Dizionario dell'intento interpretato
            context: Contesto corrente
            decision: Decisione da valutare (opzionale)
            custom_weights: Pesi personalizzati (opzionale)
        
        Returns:
            ConfidenceAssessment con valutazione completa
        """
        # Check cache
        cache_key = self._generate_cache_key(intent, context)
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached
        
        # Usa pesi custom o default
        weights = custom_weights or self.default_weights.copy()
        
        # Calcola ogni fattore
        factors = {}
        warnings = []
        suggestions = []
        
        # 1. Intent Clarity
        factors[ConfidenceFactor.INTENT_CLARITY.value] = self._evaluate_intent_clarity(
            intent, warnings, suggestions
        )
        
        # 2. Context Completeness
        factors[ConfidenceFactor.CONTEXT_COMPLETENESS.value] = self._evaluate_context_completeness(
            context, warnings, suggestions
        )
        
        # 3. Historical Success
        factors[ConfidenceFactor.HISTORICAL_SUCCESS.value] = self._evaluate_historical_success(
            intent, warnings, suggestions
        )
        
        # 4. Data Quality
        factors[ConfidenceFactor.DATA_QUALITY.value] = self._evaluate_data_quality(
            intent, context, warnings, suggestions
        )
        
        # 5. Ambiguity
        factors[ConfidenceFactor.AMBIGUITY.value] = self._evaluate_ambiguity(
            intent, warnings, suggestions
        )
        
        # 6. Risk Level
        factors[ConfidenceFactor.RISK_LEVEL.value] = self._evaluate_risk(
            decision, warnings, suggestions
        )
        
        # 7. User Feedback
        factors[ConfidenceFactor.USER_FEEDBACK.value] = self._evaluate_user_feedback(
            intent, warnings, suggestions
        )
        
        # 8. Strategy Reliability
        factors[ConfidenceFactor.STRATEGY_RELIABILITY.value] = self._evaluate_strategy_reliability(
            decision, warnings, suggestions
        )
        
        # 9. Time Pressure
        factors[ConfidenceFactor.TIME_PRESSURE.value] = self._evaluate_time_pressure(
            context, warnings, suggestions
        )
        
        # 10. Resource Availability
        factors[ConfidenceFactor.RESOURCE_AVAILABILITY.value] = self._evaluate_resource_availability(
            context, warnings, suggestions
        )
        
        # Calcola score pesato
        overall_score = self._calculate_weighted_score(factors, weights)
        
        # Applica calibrazione
        overall_score = self._apply_calibration(overall_score)
        
        # Determina livello
        level = self._determine_level(overall_score)
        
        # Determina raccomandazione
        recommendation = self._determine_recommendation(overall_score, factors, context)
        
        # Genera reasoning
        reasoning = self._generate_reasoning(overall_score, level, factors, recommendation)
        
        # Crea assessment
        assessment = ConfidenceAssessment(
            overall_score=overall_score,
            level=level,
            factors=factors,
            weights=weights,
            recommendation=recommendation,
            reasoning=reasoning,
            warnings=warnings,
            suggestions=suggestions,
            metadata={
                "intent_name": intent.get("name", "unknown"),
                "decision_type": decision.get("type") if decision else None
            }
        )
        
        # Cache result
        self._cache[cache_key] = (assessment, datetime.now())
        
        logger.info(f"Confidence valutata: {overall_score:.2f} ({level.value})")
        return assessment
    
    def _evaluate_intent_clarity(self, intent: dict, 
                                  warnings: List[str], 
                                  suggestions: List[str]) -> float:
        """Valuta chiarezza dell'intento"""
        score = 0.5  # Base
        
        # Confidence dell'intent interpreter
        intent_confidence = intent.get("confidence", 0.5)
        score = intent_confidence
        
        # Penalità se intento generico
        generic_intents = ["unknown", "general", "other", "fallback"]
        if intent.get("name", "").lower() in generic_intents:
            score *= 0.5
            warnings.append("Intento non specifico rilevato.")
            suggestions.append("Riformulare la richiesta in modo più specifico.")
        
        # Bonus se ha entità chiare
        entities = intent.get("entities", {})
        if entities:
            entity_bonus = min(0.2, len(entities) * 0.05)
            score = min(1.0, score + entity_bonus)
        
        # Penalità per ambiguità
        if intent.get("alternatives") and len(intent.get("alternatives", [])) > 2:
            score *= 0.9
            warnings.append("Molteplici interpretazioni possibili.")
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_context_completeness(self, context: dict,
                                        warnings: List[str],
                                        suggestions: List[str]) -> float:
        """Valuta completezza del contesto"""
        if not context:
            warnings.append("Contesto non disponibile.")
            suggestions.append("Fornire maggiori informazioni sul contesto.")
            return 0.3
        
        # Campi attesi
        expected_fields = [
            "time_of_day", "user_id", "session_id", 
            "app_context", "urgency", "history"
        ]
        
        present = sum(1 for f in expected_fields if f in context and context[f])
        completeness = present / len(expected_fields)
        
        # Campi critici
        critical_fields = ["user_id", "session_id"]
        critical_present = sum(1 for f in critical_fields if f in context)
        if critical_present < len(critical_fields):
            completeness *= 0.8
            warnings.append("Campi critici del contesto mancanti.")
        
        return completeness
    
    def _evaluate_historical_success(self, intent: dict,
                                      warnings: List[str],
                                      suggestions: List[str]) -> float:
        """Valuta successo storico per questo tipo di intento"""
        intent_name = intent.get("name", "unknown")
        
        # Filtra record rilevanti
        relevant_records = [
            r for r in self.historical_records
            if r.intent == intent_name
        ]
        
        if len(relevant_records) < 5:
            # Dati insufficienti
            return 0.5  # Neutral
        
        # Calcola success rate con decay temporale
        now = datetime.now()
        weighted_success = 0.0
        total_weight = 0.0
        
        for record in relevant_records:
            age_days = (now - record.timestamp).days
            weight = math.exp(-age_days / 30)  # Decay esponenziale
            
            weighted_success += weight * (1.0 if record.outcome_success else 0.0)
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        success_rate = weighted_success / total_weight
        
        if success_rate < 0.5:
            warnings.append(f"Success rate storico basso per '{intent_name}': {success_rate:.0%}")
            suggestions.append("Considerare approccio alternativo.")
        
        return success_rate
    
    def _evaluate_data_quality(self, intent: dict, context: dict,
                                warnings: List[str],
                                suggestions: List[str]) -> float:
        """Valuta qualità dei dati"""
        score = 1.0
        
        # Check per valori null/empty
        null_count = 0
        total_fields = 0
        
        for key, value in intent.items():
            total_fields += 1
            if value is None or value == "" or value == []:
                null_count += 1
        
        for key, value in context.items():
            total_fields += 1
            if value is None or value == "" or value == []:
                null_count += 1
        
        if total_fields > 0:
            null_ratio = null_count / total_fields
            score = 1.0 - (null_ratio * 0.5)  # Max 50% penalità
        
        # Check consistenza tipi
        if "confidence" in intent:
            conf = intent["confidence"]
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                score *= 0.9
                warnings.append("Valore di confidence non valido.")
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_ambiguity(self, intent: dict,
                             warnings: List[str],
                             suggestions: List[str]) -> float:
        """Valuta livello di ambiguità (score alto = bassa ambiguità)"""
        score = 1.0
        
        # Penalità per alternative multiple
        alternatives = intent.get("alternatives", [])
        if alternatives:
            alt_count = len(alternatives)
            if alt_count > 0:
                # Controlla differenza di confidence
                main_conf = intent.get("confidence", 0.5)
                for alt in alternatives[:3]:
                    alt_conf = alt.get("confidence", 0)
                    diff = main_conf - alt_conf
                    if diff < 0.2:  # Alternative molto vicine
                        score *= 0.85
                        
            if alt_count > 3:
                warnings.append("Alta ambiguità: molte interpretazioni alternative.")
                suggestions.append("Specificare meglio l'intento.")
        
        # Check per parole ambigue nel testo originale
        ambiguous_words = ["forse", "probabilmente", "potrebbe", "qualcosa", "qualche"]
        original_text = intent.get("original_text", "").lower()
        for word in ambiguous_words:
            if word in original_text:
                score *= 0.95
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_risk(self, decision: dict,
                        warnings: List[str],
                        suggestions: List[str]) -> float:
        """Valuta rischio (score alto = basso rischio)"""
        if not decision:
            return 0.7  # Default moderato
        
        risk_level = decision.get("risk_level", "low")
        
        risk_scores = {
            "minimal": 1.0,
            "low": 0.9,
            "moderate": 0.7,
            "high": 0.4,
            "critical": 0.2
        }
        
        score = risk_scores.get(risk_level, 0.7)
        
        if score < 0.5:
            warnings.append(f"Rischio elevato: {risk_level}")
            suggestions.append("Richiedere conferma utente prima di procedere.")
        
        return score
    
    def _evaluate_user_feedback(self, intent: dict,
                                 warnings: List[str],
                                 suggestions: List[str]) -> float:
        """Valuta basandosi su feedback utente precedenti"""
        if not self.executive_memory:
            return 0.5  # Neutral senza memoria
        
        intent_name = intent.get("name", "unknown")
        
        # Ottieni preferenze utente
        prefs = self.executive_memory.get_user_preferences("default", intent_name)
        
        if not prefs:
            return 0.5  # Nessun dato
        
        # Media delle confidence delle preferenze
        if prefs:
            avg_confidence = sum(p.get("confidence", 0.5) for p in prefs) / len(prefs)
            return avg_confidence
        
        return 0.5
    
    def _evaluate_strategy_reliability(self, decision: dict,
                                        warnings: List[str],
                                        suggestions: List[str]) -> float:
        """Valuta affidabilità della strategia usata"""
        if not decision:
            return 0.5
        
        strategy = decision.get("strategy_used", "unknown")
        
        if not self.executive_memory:
            return 0.5
        
        # Ottieni profilo strategia
        strategies = getattr(self.executive_memory, 'strategies', {})
        if strategy in strategies:
            profile = strategies[strategy]
            effectiveness = getattr(profile, 'effectiveness_score', 0.5)
            
            if effectiveness < 0.4:
                warnings.append(f"Strategia '{strategy}' ha bassa efficacia storica.")
            
            return effectiveness
        
        return 0.5
    
    def _evaluate_time_pressure(self, context: dict,
                                 warnings: List[str],
                                 suggestions: List[str]) -> float:
        """Valuta pressione temporale (score alto = nessuna pressione)"""
        urgency = context.get("urgency", "normal")
        
        urgency_scores = {
            "low": 1.0,
            "normal": 0.8,
            "high": 0.5,
            "critical": 0.3,
            "immediate": 0.2
        }
        
        score = urgency_scores.get(urgency, 0.8)
        
        if score < 0.5:
            warnings.append(f"Alta pressione temporale: {urgency}")
            suggestions.append("Decisione rapida potrebbe ridurre accuratezza.")
        
        return score
    
    def _evaluate_resource_availability(self, context: dict,
                                         warnings: List[str],
                                         suggestions: List[str]) -> float:
        """Valuta disponibilità risorse"""
        # Check risorse sistema
        resources = context.get("system_resources", {})
        
        if not resources:
            return 0.7  # Default
        
        cpu = resources.get("cpu_percent", 50)
        memory = resources.get("memory_percent", 50)
        
        # Penalità per risorse limitate
        score = 1.0
        
        if cpu > 90:
            score *= 0.7
            warnings.append("CPU al limite.")
        elif cpu > 70:
            score *= 0.9
        
        if memory > 90:
            score *= 0.7
            warnings.append("Memoria al limite.")
        elif memory > 70:
            score *= 0.9
        
        return max(0.0, min(1.0, score))
    
    def _calculate_weighted_score(self, factors: Dict[str, float],
                                   weights: Dict[str, float]) -> float:
        """Calcola score pesato"""
        total_score = 0.0
        total_weight = 0.0
        
        for factor_name, factor_score in factors.items():
            weight = weights.get(factor_name, 0.1)
            total_score += factor_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return total_score / total_weight
    
    def _apply_calibration(self, score: float) -> float:
        """Applica calibrazione basata su performance storica"""
        calibrated = (score * self.calibration_scale) + self.calibration_offset
        return max(0.0, min(1.0, calibrated))
    
    def _determine_level(self, score: float) -> ConfidenceLevel:
        """Determina livello di confidence"""
        for level, threshold in self.thresholds.items():
            if score >= threshold:
                return level
        return ConfidenceLevel.VERY_LOW
    
    def _determine_recommendation(self, score: float, 
                                   factors: Dict[str, float],
                                   context: dict) -> RecommendationAction:
        """Determina azione raccomandata"""
        # Override per rischio alto
        risk_score = factors.get(ConfidenceFactor.RISK_LEVEL.value, 0.7)
        if risk_score < 0.3:
            return RecommendationAction.REQUEST_CONFIRMATION
        
        # Override per ambiguità alta
        ambiguity_score = factors.get(ConfidenceFactor.AMBIGUITY.value, 0.7)
        if ambiguity_score < 0.4:
            return RecommendationAction.GATHER_MORE_INFO
        
        # Basato su score
        for action, threshold in self.action_thresholds.items():
            if score >= threshold:
                return action
        
        return RecommendationAction.ABORT
    
    def _generate_reasoning(self, score: float, level: ConfidenceLevel,
                            factors: Dict[str, float],
                            recommendation: RecommendationAction) -> str:
        """Genera reasoning professionale per la valutazione"""
        # Trova fattori più bassi
        sorted_factors = sorted(factors.items(), key=lambda x: x[1])
        lowest = sorted_factors[:2]
        highest = sorted_factors[-2:]
        
        parts = [
            f"Valutazione confidence: {score:.0%} ({level.value})."
        ]
        
        if level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
            parts.append("I parametri analizzati indicano elevata affidabilità.")
        elif level == ConfidenceLevel.MODERATE:
            parts.append("Affidabilità nella norma, alcuni fattori richiedono attenzione.")
        else:
            parts.append("Affidabilità limitata. Si raccomanda cautela.")
        
        # Fattori critici
        if lowest[0][1] < 0.5:
            factor_name = lowest[0][0].replace("_", " ").title()
            parts.append(f"Fattore critico: {factor_name} ({lowest[0][1]:.0%}).")
        
        # Raccomandazione
        action_text = {
            RecommendationAction.PROCEED: "Procedere con l'esecuzione.",
            RecommendationAction.PROCEED_WITH_CAUTION: "Procedere con monitoraggio attivo.",
            RecommendationAction.REQUEST_CONFIRMATION: "Richiedere conferma utente.",
            RecommendationAction.GATHER_MORE_INFO: "Acquisire informazioni aggiuntive.",
            RecommendationAction.DEFER: "Rimandare la decisione.",
            RecommendationAction.ABORT: "Interrompere l'operazione."
        }
        
        parts.append(f"Raccomandazione: {action_text.get(recommendation, 'Valutare manualmente.')}")
        
        return " ".join(parts)
    
    def _generate_cache_key(self, intent: dict, context: dict) -> str:
        """Genera chiave cache"""
        import hashlib
        import json
        content = json.dumps({
            "intent": intent.get("name"),
            "confidence": intent.get("confidence"),
            "context_keys": sorted(context.keys())
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    # === Calibrazione ===
    
    def record_outcome(self, intent_name: str, action_type: str,
                       predicted_confidence: float, success: bool):
        """Registra outcome per calibrazione futura"""
        record = HistoricalRecord(
            intent=intent_name,
            action_type=action_type,
            confidence_predicted=predicted_confidence,
            outcome_success=success,
            timestamp=datetime.now()
        )
        
        self.historical_records.append(record)
        
        # Mantieni limite
        if len(self.historical_records) > self.max_history:
            self.historical_records = self.historical_records[-self.max_history:]
        
        # Ricalibra periodicamente
        if len(self.historical_records) % 100 == 0:
            self._recalibrate()
    
    def _recalibrate(self):
        """Ricalibra il sistema basandosi sui risultati"""
        if len(self.historical_records) < 50:
            return
        
        # Calcola errore medio
        errors = []
        for record in self.historical_records[-200:]:
            actual = 1.0 if record.outcome_success else 0.0
            error = actual - record.confidence_predicted
            errors.append(error)
        
        if errors:
            mean_error = sum(errors) / len(errors)
            self.calibration_offset = mean_error * 0.5  # Correzione parziale
            
            logger.info(f"Calibrazione aggiornata: offset={self.calibration_offset:.3f}")
    
    # === Quick Methods ===
    
    def quick_evaluate(self, intent: dict) -> Tuple[float, ConfidenceLevel]:
        """Valutazione rapida senza contesto completo"""
        intent_conf = intent.get("confidence", 0.5)
        
        # Aggiusta per fattori noti
        if intent.get("alternatives"):
            intent_conf *= 0.9
        
        level = self._determine_level(intent_conf)
        return intent_conf, level
    
    def is_confident(self, score: float, threshold: float = 0.7) -> bool:
        """Verifica se score supera soglia"""
        return score >= threshold
    
    def get_status(self) -> dict:
        """Stato del valutatore"""
        return {
            "historical_records": len(self.historical_records),
            "calibration_offset": round(self.calibration_offset, 4),
            "calibration_scale": round(self.calibration_scale, 4),
            "cache_size": len(self._cache),
            "thresholds": {k.value: v for k, v in self.thresholds.items()}
        }
