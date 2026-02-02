"""
🔄 SELF-CORRECTION ENGINE
==========================
GIDEON monitora e corregge il proprio comportamento:
- Rileva pattern problematici
- Auto-calibra confidenza
- Corregge bias emersi
- Impara dagli errori

"Mi sono accorto che tendo a sovrastimare - sto correggendo"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class CorrectionType(Enum):
    """Tipi di correzione"""
    CONFIDENCE_CALIBRATION = "confidence"
    BIAS_CORRECTION = "bias"
    PATTERN_ADJUSTMENT = "pattern"
    ERROR_LEARNING = "error"
    FEEDBACK_INTEGRATION = "feedback"


class BiasType(Enum):
    """Tipi di bias rilevabili"""
    OVERCONFIDENCE = "overconfidence"
    UNDERCONFIDENCE = "underconfidence"
    RECENCY_BIAS = "recency"
    CONFIRMATION_BIAS = "confirmation"
    AVAILABILITY_BIAS = "availability"
    ANCHORING = "anchoring"


class FeedbackType(Enum):
    """Tipi di feedback"""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"


@dataclass
class Prediction:
    """Predizione registrata"""
    prediction_id: str
    content: str
    confidence: float
    category: str
    timestamp: datetime
    actual_outcome: Optional[str] = None
    was_correct: Optional[bool] = None
    feedback: Optional[FeedbackType] = None


@dataclass
class CorrectionAction:
    """Azione correttiva"""
    correction_id: str
    correction_type: CorrectionType
    description: str
    before_value: Any
    after_value: Any
    reason: str
    applied_at: datetime = field(default_factory=datetime.now)


@dataclass
class BiasAlert:
    """Alert per bias rilevato"""
    bias_type: BiasType
    severity: float  # 0-1
    evidence: List[str]
    suggested_correction: str
    detected_at: datetime = field(default_factory=datetime.now)


class SelfCorrectionEngine:
    """
    Engine di auto-correzione per GIDEON.
    
    Monitora e corregge automaticamente:
    - Calibrazione confidenza
    - Bias sistematici
    - Pattern problematici
    - Errori ricorrenti
    """
    
    # Soglie
    CONFIDENCE_CALIBRATION_THRESHOLD = 0.15  # 15% discrepanza
    BIAS_DETECTION_THRESHOLD = 0.2  # 20% bias
    MIN_SAMPLES_FOR_CALIBRATION = 10
    
    def __init__(self):
        self.predictions: List[Prediction] = []
        self.corrections: List[CorrectionAction] = []
        self.bias_alerts: List[BiasAlert] = []
        
        # Calibrazione confidenza per categoria
        self.confidence_calibration: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Tracking errori per pattern
        self.error_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Feedback stats
        self.feedback_stats: Dict[str, Dict[FeedbackType, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        # Bias tracking
        self.confidence_history: List[Tuple[float, bool]] = []  # (confidence, was_correct)
        
    def record_prediction(
        self,
        content: str,
        confidence: float,
        category: str = "general"
    ) -> str:
        """Registra una predizione"""
        
        # Applica calibrazione
        calibrated_confidence = self._apply_calibration(confidence, category)
        
        prediction = Prediction(
            prediction_id=f"pred_{datetime.now().timestamp()}",
            content=content,
            confidence=calibrated_confidence,
            category=category,
            timestamp=datetime.now()
        )
        
        self.predictions.append(prediction)
        
        logger.debug(f"🔄 Predizione registrata: {prediction.prediction_id}")
        
        return prediction.prediction_id
    
    def record_outcome(
        self,
        prediction_id: str,
        actual_outcome: str,
        was_correct: bool,
        feedback: Optional[FeedbackType] = None
    ):
        """Registra outcome di una predizione"""
        
        # Trova predizione
        pred = next(
            (p for p in self.predictions if p.prediction_id == prediction_id),
            None
        )
        
        if not pred:
            logger.warning(f"Predizione non trovata: {prediction_id}")
            return
        
        pred.actual_outcome = actual_outcome
        pred.was_correct = was_correct
        pred.feedback = feedback
        
        # Aggiorna tracking
        self.confidence_history.append((pred.confidence, was_correct))
        
        if feedback:
            self.feedback_stats[pred.category][feedback] += 1
        
        # Registra errore se presente
        if not was_correct:
            self.error_patterns[pred.category].append(datetime.now())
        
        # Trigger analisi
        asyncio.create_task(self._analyze_and_correct())
    
    async def _analyze_and_correct(self):
        """Analizza dati e applica correzioni"""
        
        # 1. Analizza calibrazione confidenza
        await self._analyze_confidence_calibration()
        
        # 2. Rileva bias
        await self._detect_biases()
        
        # 3. Analizza pattern errori
        await self._analyze_error_patterns()
    
    async def _analyze_confidence_calibration(self):
        """Analizza e corregge calibrazione confidenza"""
        
        # Raggruppa per categoria
        by_category: Dict[str, List[Prediction]] = defaultdict(list)
        
        for pred in self.predictions:
            if pred.was_correct is not None:
                by_category[pred.category].append(pred)
        
        for category, preds in by_category.items():
            if len(preds) < self.MIN_SAMPLES_FOR_CALIBRATION:
                continue
            
            # Calcola calibrazione
            avg_confidence = statistics.mean(p.confidence for p in preds)
            actual_accuracy = sum(1 for p in preds if p.was_correct) / len(preds)
            
            # Discrepanza
            discrepancy = avg_confidence - actual_accuracy
            
            if abs(discrepancy) > self.CONFIDENCE_CALIBRATION_THRESHOLD:
                # Applica correzione
                old_calibration = self.confidence_calibration[category]
                
                # Calibrazione: se sovrastimo, riduci; se sottostimo, aumenta
                adjustment = 1 - (discrepancy / 2)  # Correzione graduale
                new_calibration = old_calibration * adjustment
                
                self.confidence_calibration[category] = new_calibration
                
                # Log correzione
                correction = CorrectionAction(
                    correction_id=f"calib_{datetime.now().timestamp()}",
                    correction_type=CorrectionType.CONFIDENCE_CALIBRATION,
                    description=f"Calibrazione confidenza per '{category}'",
                    before_value=old_calibration,
                    after_value=new_calibration,
                    reason=f"Discrepanza rilevata: {discrepancy:.2%} (conf: {avg_confidence:.2%}, acc: {actual_accuracy:.2%})"
                )
                
                self.corrections.append(correction)
                
                logger.info(f"🔄 Calibrazione applicata: {category} -> {new_calibration:.3f}")
    
    async def _detect_biases(self):
        """Rileva bias sistematici"""
        
        if len(self.confidence_history) < self.MIN_SAMPLES_FOR_CALIBRATION:
            return
        
        recent = self.confidence_history[-50:]  # Ultimi 50
        
        # Check overconfidence
        high_conf = [(c, cor) for c, cor in recent if c > 0.8]
        if high_conf:
            high_conf_accuracy = sum(1 for _, cor in high_conf if cor) / len(high_conf)
            if high_conf_accuracy < 0.8:  # Meno dell'80% corretto quando conf > 80%
                self._report_bias(
                    BiasType.OVERCONFIDENCE,
                    severity=(0.8 - high_conf_accuracy),
                    evidence=[
                        f"Confidenza media alta: {statistics.mean(c for c, _ in high_conf):.2%}",
                        f"Accuratezza effettiva: {high_conf_accuracy:.2%}"
                    ]
                )
        
        # Check underconfidence
        low_conf = [(c, cor) for c, cor in recent if c < 0.5]
        if low_conf:
            low_conf_accuracy = sum(1 for _, cor in low_conf if cor) / len(low_conf)
            if low_conf_accuracy > 0.7:  # Più del 70% corretto quando conf < 50%
                self._report_bias(
                    BiasType.UNDERCONFIDENCE,
                    severity=(low_conf_accuracy - 0.5),
                    evidence=[
                        f"Confidenza media bassa: {statistics.mean(c for c, _ in low_conf):.2%}",
                        f"Accuratezza effettiva: {low_conf_accuracy:.2%}"
                    ]
                )
        
        # Check recency bias (ultimi risultati influenzano troppo)
        if len(recent) >= 20:
            first_half_acc = sum(1 for _, cor in recent[:10] if cor) / 10
            second_half_acc = sum(1 for _, cor in recent[10:20] if cor) / 10
            
            if abs(first_half_acc - second_half_acc) > 0.3:
                self._report_bias(
                    BiasType.RECENCY_BIAS,
                    severity=abs(first_half_acc - second_half_acc),
                    evidence=[
                        f"Accuratezza prima metà: {first_half_acc:.2%}",
                        f"Accuratezza seconda metà: {second_half_acc:.2%}"
                    ]
                )
    
    def _report_bias(
        self,
        bias_type: BiasType,
        severity: float,
        evidence: List[str]
    ):
        """Riporta bias rilevato"""
        
        suggestions = {
            BiasType.OVERCONFIDENCE: "Ridurre confidenza baseline del 10-15%",
            BiasType.UNDERCONFIDENCE: "Aumentare confidenza baseline del 10-15%",
            BiasType.RECENCY_BIAS: "Dare più peso a dati storici",
            BiasType.CONFIRMATION_BIAS: "Cercare attivamente evidenze contrarie",
            BiasType.AVAILABILITY_BIAS: "Consultare database completo, non solo casi recenti",
            BiasType.ANCHORING: "Resettare valutazione da zero"
        }
        
        alert = BiasAlert(
            bias_type=bias_type,
            severity=severity,
            evidence=evidence,
            suggested_correction=suggestions.get(bias_type, "Valutazione manuale richiesta")
        )
        
        self.bias_alerts.append(alert)
        
        logger.warning(f"🔄 BIAS RILEVATO: {bias_type.value} (sev: {severity:.2%})")
    
    async def _analyze_error_patterns(self):
        """Analizza pattern negli errori"""
        
        now = datetime.now()
        
        for category, errors in self.error_patterns.items():
            # Filtra errori recenti (ultima settimana)
            recent_errors = [e for e in errors if now - e < timedelta(days=7)]
            
            if len(recent_errors) >= 5:  # Pattern significativo
                # Check clustering temporale
                if self._errors_clustered(recent_errors):
                    logger.warning(f"🔄 Pattern errori rilevato in '{category}': {len(recent_errors)} errori recenti")
                    
                    # Applica correzione automatica
                    old_calib = self.confidence_calibration[category]
                    new_calib = old_calib * 0.9  # Riduci 10%
                    self.confidence_calibration[category] = new_calib
                    
                    self.corrections.append(CorrectionAction(
                        correction_id=f"pattern_{datetime.now().timestamp()}",
                        correction_type=CorrectionType.PATTERN_ADJUSTMENT,
                        description=f"Riduzione confidenza per pattern errori in '{category}'",
                        before_value=old_calib,
                        after_value=new_calib,
                        reason=f"{len(recent_errors)} errori nell'ultima settimana"
                    ))
    
    def _errors_clustered(self, errors: List[datetime]) -> bool:
        """Verifica se errori sono raggruppati temporalmente"""
        
        if len(errors) < 3:
            return False
        
        errors_sorted = sorted(errors)
        intervals = [
            (errors_sorted[i+1] - errors_sorted[i]).total_seconds()
            for i in range(len(errors_sorted) - 1)
        ]
        
        # Clustering se intervallo medio < 4 ore
        avg_interval = statistics.mean(intervals)
        return avg_interval < 4 * 3600
    
    def _apply_calibration(self, confidence: float, category: str) -> float:
        """Applica calibrazione a confidenza"""
        
        calibration = self.confidence_calibration[category]
        calibrated = confidence * calibration
        
        # Clamp a [0, 1]
        return max(0.0, min(1.0, calibrated))
    
    def get_calibration_for(self, category: str) -> float:
        """Ottiene calibrazione per categoria"""
        return self.confidence_calibration[category]
    
    def get_active_biases(self) -> List[BiasAlert]:
        """Ritorna bias attivi (ultimi 7 giorni)"""
        
        cutoff = datetime.now() - timedelta(days=7)
        return [b for b in self.bias_alerts if b.detected_at > cutoff]
    
    def get_corrections_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Ritorna storia correzioni"""
        
        return [
            {
                'id': c.correction_id,
                'type': c.correction_type.value,
                'description': c.description,
                'before': c.before_value,
                'after': c.after_value,
                'reason': c.reason,
                'applied_at': c.applied_at.isoformat()
            }
            for c in self.corrections[-limit:]
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Ritorna statistiche self-correction"""
        
        total_preds = len(self.predictions)
        evaluated = [p for p in self.predictions if p.was_correct is not None]
        
        accuracy = (
            sum(1 for p in evaluated if p.was_correct) / len(evaluated)
            if evaluated else 0
        )
        
        avg_confidence = (
            statistics.mean(p.confidence for p in evaluated)
            if evaluated else 0
        )
        
        return {
            'total_predictions': total_preds,
            'evaluated_predictions': len(evaluated),
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'calibration_gap': avg_confidence - accuracy if evaluated else 0,
            'total_corrections': len(self.corrections),
            'active_biases': len(self.get_active_biases()),
            'calibrations_by_category': dict(self.confidence_calibration)
        }
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        stats = self.get_stats()
        biases = self.get_active_biases()
        
        bias_section = ""
        if biases:
            bias_section = "\n## ⚠️ Bias Attivi\n"
            for b in biases[:3]:
                bias_section += f"- **{b.bias_type.value}** (sev: {b.severity:.1%}): {b.suggested_correction}\n"
        
        return f"""
# 🔄 Self-Correction Engine Status

## 📊 Statistiche
| Metrica | Valore |
|---------|--------|
| Predizioni totali | {stats['total_predictions']} |
| Predizioni valutate | {stats['evaluated_predictions']} |
| Accuratezza | {stats['accuracy']:.1%} |
| Confidenza media | {stats['avg_confidence']:.1%} |
| Gap calibrazione | {stats['calibration_gap']:.1%} |
| Correzioni applicate | {stats['total_corrections']} |

{bias_section}

## 🔧 Calibrazioni Attive
{chr(10).join(f"- **{cat}**: {cal:.3f}" for cat, cal in stats['calibrations_by_category'].items()) or '- Nessuna calibrazione attiva'}

## 📜 Ultime Correzioni
{chr(10).join(f"- [{c['type']}] {c['description']}" for c in self.get_corrections_history(5)) or '- Nessuna correzione recente'}
"""


# Import necessario per asyncio
import asyncio
from typing import Tuple
