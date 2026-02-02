# /backend/gideon/self_correction.py
"""
🔮 GIDEON 3.0 - Self-Correction Engine
Rileva e corregge previsioni errate, apprende dagli errori.
NON esegue azioni - fornisce solo correzioni e calibrazioni.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
import math
import logging

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Tipi di errore nelle previsioni"""
    OVERESTIMATE = "overestimate"       # Sovrastima
    UNDERESTIMATE = "underestimate"     # Sottostima
    WRONG_DIRECTION = "wrong_direction" # Direzione sbagliata
    TIMING_ERROR = "timing_error"       # Errore temporale
    MISSING_FACTOR = "missing_factor"   # Fattore non considerato
    MODEL_BIAS = "model_bias"           # Bias del modello
    DATA_QUALITY = "data_quality"       # Problemi dati
    CONTEXT_SHIFT = "context_shift"     # Cambio contesto


class CorrectionStrategy(Enum):
    """Strategie di correzione"""
    BIAS_ADJUSTMENT = "bias_adjustment"       # Aggiusta bias
    WEIGHT_RECALIBRATION = "weight_recalibration"  # Ricalibra pesi
    MODEL_UPDATE = "model_update"             # Aggiorna modello
    FACTOR_INCLUSION = "factor_inclusion"     # Includi nuovo fattore
    OUTLIER_HANDLING = "outlier_handling"     # Gestisci outlier
    ENSEMBLE_BLEND = "ensemble_blend"         # Combina modelli
    TEMPORAL_SMOOTHING = "temporal_smoothing" # Smoothing temporale


class SeverityLevel(Enum):
    """Gravità dell'errore"""
    NEGLIGIBLE = "negligible"  # < 5%
    MINOR = "minor"            # 5-15%
    MODERATE = "moderate"      # 15-30%
    SIGNIFICANT = "significant" # 30-50%
    SEVERE = "severe"          # > 50%


@dataclass
class PredictionRecord:
    """Record di una previsione"""
    id: str
    domain: str
    predicted_value: float
    actual_value: Optional[float] = None
    
    # Context
    context: Dict = field(default_factory=dict)
    factors_used: List[str] = field(default_factory=list)
    confidence: float = 0.5
    
    # Timing
    predicted_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    # Error analysis
    error: Optional[float] = None
    error_type: Optional[ErrorType] = None
    corrected: bool = False
    
    def calculate_error(self) -> Optional[float]:
        """Calcola errore se abbiamo valore effettivo"""
        if self.actual_value is not None:
            self.error = self.actual_value - self.predicted_value
            return self.error
        return None
    
    def error_percentage(self) -> Optional[float]:
        """Errore percentuale"""
        if self.actual_value is not None and self.actual_value != 0:
            return abs(self.error or 0) / abs(self.actual_value) * 100
        return None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "predicted": self.predicted_value,
            "actual": self.actual_value,
            "error": self.error,
            "error_percentage": self.error_percentage(),
            "confidence": self.confidence,
            "error_type": self.error_type.value if self.error_type else None,
            "corrected": self.corrected
        }


@dataclass
class CorrectionAction:
    """Azione correttiva da applicare"""
    id: str
    strategy: CorrectionStrategy
    domain: str
    
    # Parameters
    adjustment_value: float = 0.0
    weight_changes: Dict[str, float] = field(default_factory=dict)
    new_factors: List[str] = field(default_factory=list)
    
    # Impact
    expected_improvement: float = 0.0
    confidence: float = 0.5
    
    # Validation
    applied: bool = False
    validated: bool = False
    validation_result: Optional[float] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "domain": self.domain,
            "adjustment": self.adjustment_value,
            "weight_changes": self.weight_changes,
            "new_factors": self.new_factors,
            "expected_improvement": round(self.expected_improvement, 3),
            "applied": self.applied,
            "validated": self.validated
        }


@dataclass
class ErrorPattern:
    """Pattern di errore ricorrente"""
    pattern_id: str
    domain: str
    error_type: ErrorType
    
    # Statistics
    frequency: int = 0
    avg_magnitude: float = 0.0
    conditions: List[str] = field(default_factory=list)
    
    # Suggested fix
    recommended_strategy: CorrectionStrategy = CorrectionStrategy.BIAS_ADJUSTMENT
    fix_confidence: float = 0.5
    
    last_seen: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "domain": self.domain,
            "error_type": self.error_type.value,
            "frequency": self.frequency,
            "avg_magnitude": round(self.avg_magnitude, 3),
            "conditions": self.conditions,
            "recommended_strategy": self.recommended_strategy.value,
            "fix_confidence": round(self.fix_confidence, 2)
        }


@dataclass
class CalibrationState:
    """Stato di calibrazione per un dominio"""
    domain: str
    
    # Bias tracking
    current_bias: float = 0.0
    bias_history: List[float] = field(default_factory=list)
    
    # Accuracy metrics
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Square Error
    mape: float = 0.0  # Mean Absolute Percentage Error
    
    # Weights
    factor_weights: Dict[str, float] = field(default_factory=dict)
    
    # Calibration
    is_calibrated: bool = False
    last_calibration: Optional[datetime] = None
    calibration_quality: float = 0.5
    
    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "current_bias": round(self.current_bias, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 2),
            "factor_weights": self.factor_weights,
            "is_calibrated": self.is_calibrated,
            "calibration_quality": round(self.calibration_quality, 2)
        }


class SelfCorrectionEngine:
    """
    Motore di auto-correzione per Gideon.
    Monitora previsioni, rileva errori e applica correzioni.
    """
    
    def __init__(self):
        # Prediction tracking
        self.predictions: Dict[str, PredictionRecord] = {}
        self.prediction_history: List[PredictionRecord] = []
        
        # Calibration state per domain
        self.calibrations: Dict[str, CalibrationState] = {}
        
        # Error patterns
        self.error_patterns: Dict[str, ErrorPattern] = {}
        
        # Correction actions
        self.corrections: List[CorrectionAction] = []
        self.pending_corrections: List[CorrectionAction] = []
        
        # Configuration
        self.error_threshold = 0.1  # 10% per trigger correction
        self.min_samples_for_calibration = 10
        self.bias_smoothing_factor = 0.3
        
        # Counters
        self._prediction_counter = 0
        self._correction_counter = 0
        self._pattern_counter = 0
    
    # === Prediction Tracking ===
    
    def record_prediction(self, domain: str, predicted_value: float,
                          confidence: float = 0.5,
                          context: Dict = None,
                          factors: List[str] = None) -> PredictionRecord:
        """
        Registra una nuova previsione per tracking.
        """
        self._prediction_counter += 1
        pred_id = f"pred_{self._prediction_counter}"
        
        record = PredictionRecord(
            id=pred_id,
            domain=domain,
            predicted_value=predicted_value,
            confidence=confidence,
            context=context or {},
            factors_used=factors or []
        )
        
        self.predictions[pred_id] = record
        
        # Apply any pending bias correction
        if domain in self.calibrations:
            cal = self.calibrations[domain]
            if cal.current_bias != 0:
                # Adjust prediction based on known bias
                corrected = predicted_value - cal.current_bias
                record.context["original_prediction"] = predicted_value
                record.predicted_value = corrected
        
        return record
    
    def resolve_prediction(self, prediction_id: str,
                           actual_value: float) -> Optional[Dict]:
        """
        Risolve previsione con valore effettivo e analizza errore.
        """
        if prediction_id not in self.predictions:
            return None
        
        record = self.predictions[prediction_id]
        record.actual_value = actual_value
        record.resolved_at = datetime.now()
        record.calculate_error()
        
        # Analyze error
        error_analysis = self._analyze_error(record)
        record.error_type = error_analysis["error_type"]
        
        # Move to history
        self.prediction_history.append(record)
        
        # Update calibration
        self._update_calibration(record)
        
        # Check for patterns
        self._detect_patterns(record)
        
        # Generate correction if needed
        correction = None
        if abs(record.error_percentage() or 0) > self.error_threshold * 100:
            correction = self._generate_correction(record, error_analysis)
        
        return {
            "prediction": record.to_dict(),
            "error_analysis": error_analysis,
            "correction": correction.to_dict() if correction else None
        }
    
    def _analyze_error(self, record: PredictionRecord) -> Dict:
        """Analizza tipo e causa dell'errore"""
        error = record.error or 0
        error_pct = record.error_percentage() or 0
        
        # Determine error type
        if error > 0:
            error_type = ErrorType.UNDERESTIMATE
        elif error < 0:
            error_type = ErrorType.OVERESTIMATE
        else:
            error_type = None
        
        # Determine severity
        if error_pct < 5:
            severity = SeverityLevel.NEGLIGIBLE
        elif error_pct < 15:
            severity = SeverityLevel.MINOR
        elif error_pct < 30:
            severity = SeverityLevel.MODERATE
        elif error_pct < 50:
            severity = SeverityLevel.SIGNIFICANT
        else:
            severity = SeverityLevel.SEVERE
        
        # Identify potential causes
        causes = []
        
        # Check if context changed
        if record.context.get("context_changed"):
            causes.append("context_shift")
            error_type = ErrorType.CONTEXT_SHIFT
        
        # Check confidence vs error
        if record.confidence > 0.8 and error_pct > 20:
            causes.append("overconfident_model")
            error_type = ErrorType.MODEL_BIAS
        
        # Check for missing factors
        if not record.factors_used:
            causes.append("missing_factors")
        
        return {
            "error_type": error_type,
            "severity": severity.value if severity else None,
            "error_value": error,
            "error_percentage": error_pct,
            "potential_causes": causes,
            "needs_correction": error_pct > self.error_threshold * 100
        }
    
    # === Calibration ===
    
    def _update_calibration(self, record: PredictionRecord):
        """Aggiorna stato calibrazione per il dominio"""
        domain = record.domain
        
        if domain not in self.calibrations:
            self.calibrations[domain] = CalibrationState(domain=domain)
        
        cal = self.calibrations[domain]
        
        # Update bias with exponential smoothing
        if record.error is not None:
            cal.bias_history.append(record.error)
            cal.current_bias = (
                self.bias_smoothing_factor * record.error +
                (1 - self.bias_smoothing_factor) * cal.current_bias
            )
        
        # Calculate metrics on history
        domain_predictions = [p for p in self.prediction_history 
                            if p.domain == domain and p.error is not None]
        
        if len(domain_predictions) >= self.min_samples_for_calibration:
            errors = [p.error for p in domain_predictions[-50:]]  # Last 50
            abs_errors = [abs(e) for e in errors]
            
            cal.mae = statistics.mean(abs_errors)
            cal.rmse = math.sqrt(statistics.mean([e**2 for e in errors]))
            
            # MAPE
            pct_errors = [p.error_percentage() for p in domain_predictions[-50:] 
                         if p.error_percentage() is not None]
            cal.mape = statistics.mean(pct_errors) if pct_errors else 0
            
            # Calibration quality
            cal.is_calibrated = cal.mape < 15
            cal.calibration_quality = max(0, 1 - cal.mape / 100)
            cal.last_calibration = datetime.now()
    
    def calibrate_domain(self, domain: str, force: bool = False) -> Dict:
        """
        Esegue calibrazione completa per un dominio.
        """
        domain_predictions = [p for p in self.prediction_history 
                            if p.domain == domain and p.error is not None]
        
        if len(domain_predictions) < self.min_samples_for_calibration and not force:
            return {
                "success": False,
                "reason": f"Insufficienti campioni ({len(domain_predictions)}/{self.min_samples_for_calibration})"
            }
        
        if domain not in self.calibrations:
            self.calibrations[domain] = CalibrationState(domain=domain)
        
        cal = self.calibrations[domain]
        
        # Calculate optimal bias correction
        errors = [p.error for p in domain_predictions]
        cal.current_bias = statistics.mean(errors)
        
        # Analyze factor importance
        factor_errors = {}
        for pred in domain_predictions:
            for factor in pred.factors_used:
                if factor not in factor_errors:
                    factor_errors[factor] = []
                factor_errors[factor].append(abs(pred.error or 0))
        
        # Lower error = higher weight
        for factor, errs in factor_errors.items():
            avg_err = statistics.mean(errs)
            cal.factor_weights[factor] = max(0.1, 1 - avg_err)
        
        cal.is_calibrated = True
        cal.last_calibration = datetime.now()
        
        return {
            "success": True,
            "domain": domain,
            "samples_used": len(domain_predictions),
            "bias_correction": cal.current_bias,
            "factor_weights": cal.factor_weights,
            "calibration_quality": cal.calibration_quality
        }
    
    # === Pattern Detection ===
    
    def _detect_patterns(self, record: PredictionRecord):
        """Rileva pattern di errore ricorrenti"""
        if not record.error_type:
            return
        
        domain = record.domain
        error_type = record.error_type
        
        # Pattern key
        pattern_key = f"{domain}_{error_type.value}"
        
        if pattern_key not in self.error_patterns:
            self._pattern_counter += 1
            self.error_patterns[pattern_key] = ErrorPattern(
                pattern_id=f"pattern_{self._pattern_counter}",
                domain=domain,
                error_type=error_type
            )
        
        pattern = self.error_patterns[pattern_key]
        pattern.frequency += 1
        pattern.last_seen = datetime.now()
        
        # Update average magnitude
        if record.error_percentage():
            pattern.avg_magnitude = (
                pattern.avg_magnitude * (pattern.frequency - 1) +
                record.error_percentage()
            ) / pattern.frequency
        
        # Record conditions
        for key, value in record.context.items():
            condition = f"{key}={value}"
            if condition not in pattern.conditions:
                pattern.conditions.append(condition)
        
        # Determine fix strategy
        pattern.recommended_strategy = self._recommend_strategy(pattern)
        pattern.fix_confidence = min(0.9, pattern.frequency * 0.1)
    
    def _recommend_strategy(self, pattern: ErrorPattern) -> CorrectionStrategy:
        """Raccomanda strategia di correzione per pattern"""
        if pattern.error_type == ErrorType.MODEL_BIAS:
            return CorrectionStrategy.BIAS_ADJUSTMENT
        elif pattern.error_type in [ErrorType.OVERESTIMATE, ErrorType.UNDERESTIMATE]:
            if pattern.frequency > 5:
                return CorrectionStrategy.WEIGHT_RECALIBRATION
            return CorrectionStrategy.BIAS_ADJUSTMENT
        elif pattern.error_type == ErrorType.MISSING_FACTOR:
            return CorrectionStrategy.FACTOR_INCLUSION
        elif pattern.error_type == ErrorType.CONTEXT_SHIFT:
            return CorrectionStrategy.MODEL_UPDATE
        elif pattern.error_type == ErrorType.TIMING_ERROR:
            return CorrectionStrategy.TEMPORAL_SMOOTHING
        else:
            return CorrectionStrategy.ENSEMBLE_BLEND
    
    # === Correction Generation ===
    
    def _generate_correction(self, record: PredictionRecord,
                             error_analysis: Dict) -> CorrectionAction:
        """Genera azione correttiva"""
        self._correction_counter += 1
        
        # Determine strategy
        error_type = error_analysis.get("error_type")
        strategy = CorrectionStrategy.BIAS_ADJUSTMENT
        
        if error_type:
            pattern_key = f"{record.domain}_{error_type.value}"
            if pattern_key in self.error_patterns:
                strategy = self.error_patterns[pattern_key].recommended_strategy
        
        # Calculate adjustment
        adjustment = 0.0
        weight_changes = {}
        
        if strategy == CorrectionStrategy.BIAS_ADJUSTMENT:
            adjustment = -(record.error or 0) * 0.5  # Gradual correction
        
        elif strategy == CorrectionStrategy.WEIGHT_RECALIBRATION:
            # Adjust weights based on error direction
            for factor in record.factors_used:
                if record.error and record.error > 0:
                    weight_changes[factor] = 1.1  # Increase weight
                else:
                    weight_changes[factor] = 0.9  # Decrease weight
        
        # Expected improvement
        expected_improvement = min(0.5, abs(record.error_percentage() or 0) * 0.3 / 100)
        
        correction = CorrectionAction(
            id=f"corr_{self._correction_counter}",
            strategy=strategy,
            domain=record.domain,
            adjustment_value=adjustment,
            weight_changes=weight_changes,
            expected_improvement=expected_improvement,
            confidence=0.6
        )
        
        self.pending_corrections.append(correction)
        return correction
    
    def apply_correction(self, correction_id: str) -> bool:
        """Applica una correzione pendente"""
        correction = None
        for c in self.pending_corrections:
            if c.id == correction_id:
                correction = c
                break
        
        if not correction:
            return False
        
        domain = correction.domain
        
        if domain not in self.calibrations:
            self.calibrations[domain] = CalibrationState(domain=domain)
        
        cal = self.calibrations[domain]
        
        # Apply based on strategy
        if correction.strategy == CorrectionStrategy.BIAS_ADJUSTMENT:
            cal.current_bias += correction.adjustment_value
        
        elif correction.strategy == CorrectionStrategy.WEIGHT_RECALIBRATION:
            for factor, multiplier in correction.weight_changes.items():
                if factor in cal.factor_weights:
                    cal.factor_weights[factor] *= multiplier
                else:
                    cal.factor_weights[factor] = multiplier
        
        correction.applied = True
        self.corrections.append(correction)
        self.pending_corrections.remove(correction)
        
        return True
    
    def validate_correction(self, correction_id: str,
                           improvement_observed: float) -> Dict:
        """Valida efficacia di una correzione"""
        correction = None
        for c in self.corrections:
            if c.id == correction_id:
                correction = c
                break
        
        if not correction:
            return {"success": False, "reason": "Correction not found"}
        
        correction.validated = True
        correction.validation_result = improvement_observed
        
        # Compare with expected
        success = improvement_observed >= correction.expected_improvement * 0.5
        
        return {
            "success": True,
            "correction_id": correction_id,
            "expected_improvement": correction.expected_improvement,
            "actual_improvement": improvement_observed,
            "effective": success
        }
    
    # === Correction Suggestions ===
    
    def suggest_corrections(self, domain: str = None) -> List[Dict]:
        """Suggerisce correzioni basate su pattern rilevati"""
        suggestions = []
        
        patterns = self.error_patterns.values()
        if domain:
            patterns = [p for p in patterns if p.domain == domain]
        
        for pattern in patterns:
            if pattern.frequency >= 3:  # Minimum occurrences
                suggestions.append({
                    "domain": pattern.domain,
                    "error_type": pattern.error_type.value,
                    "frequency": pattern.frequency,
                    "avg_magnitude": pattern.avg_magnitude,
                    "recommended_strategy": pattern.recommended_strategy.value,
                    "confidence": pattern.fix_confidence,
                    "priority": self._calculate_priority(pattern)
                })
        
        return sorted(suggestions, key=lambda s: s["priority"], reverse=True)
    
    def _calculate_priority(self, pattern: ErrorPattern) -> float:
        """Calcola priorità di correzione"""
        # Higher frequency + higher magnitude = higher priority
        return pattern.frequency * 0.3 + pattern.avg_magnitude * 0.5 + pattern.fix_confidence * 0.2
    
    # === Correction Application ===
    
    def correct_prediction(self, predicted_value: float,
                           domain: str,
                           context: Dict = None) -> Dict:
        """
        Applica correzioni note a una nuova previsione.
        """
        corrected = predicted_value
        corrections_applied = []
        
        # Apply bias correction
        if domain in self.calibrations:
            cal = self.calibrations[domain]
            if cal.current_bias != 0:
                corrected -= cal.current_bias
                corrections_applied.append({
                    "type": "bias_correction",
                    "value": -cal.current_bias
                })
        
        # Apply pattern-based corrections
        for pattern in self.error_patterns.values():
            if pattern.domain == domain and pattern.frequency >= 5:
                if pattern.error_type == ErrorType.OVERESTIMATE:
                    adjustment = -pattern.avg_magnitude / 100 * corrected * 0.3
                    corrected += adjustment
                    corrections_applied.append({
                        "type": "pattern_correction",
                        "pattern": pattern.error_type.value,
                        "value": adjustment
                    })
                elif pattern.error_type == ErrorType.UNDERESTIMATE:
                    adjustment = pattern.avg_magnitude / 100 * corrected * 0.3
                    corrected += adjustment
                    corrections_applied.append({
                        "type": "pattern_correction",
                        "pattern": pattern.error_type.value,
                        "value": adjustment
                    })
        
        return {
            "original": predicted_value,
            "corrected": corrected,
            "corrections_applied": corrections_applied,
            "correction_magnitude": corrected - predicted_value,
            "confidence_boost": len(corrections_applied) * 0.05
        }
    
    # === Metrics ===
    
    def get_accuracy_metrics(self, domain: str = None) -> Dict:
        """Ottiene metriche di accuratezza"""
        predictions = self.prediction_history
        if domain:
            predictions = [p for p in predictions if p.domain == domain]
        
        resolved = [p for p in predictions if p.error is not None]
        
        if not resolved:
            return {"error": "No resolved predictions"}
        
        errors = [p.error for p in resolved]
        abs_errors = [abs(e) for e in errors]
        pct_errors = [p.error_percentage() for p in resolved if p.error_percentage()]
        
        return {
            "total_predictions": len(predictions),
            "resolved_predictions": len(resolved),
            "mae": statistics.mean(abs_errors),
            "rmse": math.sqrt(statistics.mean([e**2 for e in errors])),
            "mape": statistics.mean(pct_errors) if pct_errors else None,
            "bias": statistics.mean(errors),
            "std_error": statistics.stdev(errors) if len(errors) > 1 else 0,
            "accuracy_rate": sum(1 for p in pct_errors if p < 15) / len(pct_errors) if pct_errors else 0
        }
    
    def get_status(self) -> Dict:
        """Stato del motore di auto-correzione"""
        return {
            "total_predictions": len(self.predictions) + len(self.prediction_history),
            "pending_predictions": len(self.predictions),
            "resolved_predictions": len(self.prediction_history),
            "calibrated_domains": sum(1 for c in self.calibrations.values() if c.is_calibrated),
            "error_patterns_detected": len(self.error_patterns),
            "corrections_applied": len(self.corrections),
            "pending_corrections": len(self.pending_corrections)
        }
