"""
🧠 GRACEFUL DEGRADATION ENGINE
==============================
Quando i dati sono incompleti o caotici, riduce progressivamente
le funzionalità mantenendo solo analisi essenziali ad alta confidenza.

Livelli di Operatività:
- L1: FULL - Tutte le funzionalità attive
- L2: REDUCED - Funzionalità non critiche disabilitate  
- L3: ESSENTIAL - Solo analisi core
- L4: MINIMAL - Solo logging e supporto passivo
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class OperationalLevel(Enum):
    """Livelli di operatività del sistema"""
    L1_FULL = 1        # Tutte le funzionalità
    L2_REDUCED = 2     # Ridotto
    L3_ESSENTIAL = 3   # Essenziale
    L4_MINIMAL = 4     # Minimo (safe state)


@dataclass
class DegradationTrigger:
    """Trigger per degradazione"""
    trigger_type: str  # 'data_quality', 'sensor_failure', 'latency', 'confidence'
    threshold: float
    current_value: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FeatureStatus:
    """Stato di una funzionalità"""
    name: str
    enabled: bool
    level_required: OperationalLevel
    last_check: datetime = field(default_factory=datetime.now)
    reason_disabled: Optional[str] = None


class GracefulDegradationEngine:
    """
    Engine per degradazione graceful del sistema.
    
    Principi:
    - Mai fallire completamente
    - Ridurre progressivamente le capacità
    - Mantenere sempre logging e spiegazione
    - Evitare risposte "sicure ma sbagliate"
    """
    
    def __init__(self):
        self.current_level = OperationalLevel.L1_FULL
        self.triggers: List[DegradationTrigger] = []
        self.features: Dict[str, FeatureStatus] = {}
        self.history: List[Dict[str, Any]] = []
        self._initialize_features()
        
    def _initialize_features(self):
        """Inizializza le feature con i loro livelli richiesti"""
        feature_config = {
            # L1 - Full: tutto attivo
            'advanced_prediction': OperationalLevel.L1_FULL,
            'multi_tool_orchestration': OperationalLevel.L1_FULL,
            'autonomous_actions': OperationalLevel.L1_FULL,
            'complex_simulations': OperationalLevel.L1_FULL,
            
            # L2 - Reduced: funzionalità avanzate disabilitate
            'real_time_analysis': OperationalLevel.L2_REDUCED,
            'external_integrations': OperationalLevel.L2_REDUCED,
            'background_processing': OperationalLevel.L2_REDUCED,
            
            # L3 - Essential: solo core
            'basic_analysis': OperationalLevel.L3_ESSENTIAL,
            'query_response': OperationalLevel.L3_ESSENTIAL,
            'alert_generation': OperationalLevel.L3_ESSENTIAL,
            
            # L4 - Minimal: sempre attivo
            'logging': OperationalLevel.L4_MINIMAL,
            'explanation': OperationalLevel.L4_MINIMAL,
            'passive_support': OperationalLevel.L4_MINIMAL,
            'human_notification': OperationalLevel.L4_MINIMAL,
        }
        
        for name, level in feature_config.items():
            self.features[name] = FeatureStatus(
                name=name,
                enabled=True,
                level_required=level
            )
    
    async def evaluate_system_health(self, metrics: Dict[str, Any]) -> OperationalLevel:
        """
        Valuta la salute del sistema e determina il livello operativo.
        
        Args:
            metrics: Metriche di sistema (data_quality, latency, sensor_status, etc.)
            
        Returns:
            Livello operativo raccomandato
        """
        triggers = []
        
        # Check qualità dati
        data_quality = metrics.get('data_quality', 1.0)
        if data_quality < 0.3:
            triggers.append(DegradationTrigger(
                trigger_type='data_quality',
                threshold=0.3,
                current_value=data_quality,
                severity='critical'
            ))
        elif data_quality < 0.5:
            triggers.append(DegradationTrigger(
                trigger_type='data_quality',
                threshold=0.5,
                current_value=data_quality,
                severity='high'
            ))
        elif data_quality < 0.7:
            triggers.append(DegradationTrigger(
                trigger_type='data_quality',
                threshold=0.7,
                current_value=data_quality,
                severity='medium'
            ))
            
        # Check latenza
        latency_ms = metrics.get('latency_ms', 0)
        if latency_ms > 5000:
            triggers.append(DegradationTrigger(
                trigger_type='latency',
                threshold=5000,
                current_value=latency_ms,
                severity='critical'
            ))
        elif latency_ms > 2000:
            triggers.append(DegradationTrigger(
                trigger_type='latency',
                threshold=2000,
                current_value=latency_ms,
                severity='high'
            ))
            
        # Check sensori/servizi
        services_healthy = metrics.get('services_healthy', 1.0)
        if services_healthy < 0.5:
            triggers.append(DegradationTrigger(
                trigger_type='sensor_failure',
                threshold=0.5,
                current_value=services_healthy,
                severity='critical'
            ))
        elif services_healthy < 0.7:
            triggers.append(DegradationTrigger(
                trigger_type='sensor_failure',
                threshold=0.7,
                current_value=services_healthy,
                severity='high'
            ))
            
        # Check confidenza media
        avg_confidence = metrics.get('avg_confidence', 1.0)
        if avg_confidence < 0.4:
            triggers.append(DegradationTrigger(
                trigger_type='confidence',
                threshold=0.4,
                current_value=avg_confidence,
                severity='critical'
            ))
        elif avg_confidence < 0.6:
            triggers.append(DegradationTrigger(
                trigger_type='confidence',
                threshold=0.6,
                current_value=avg_confidence,
                severity='high'
            ))
            
        self.triggers = triggers
        
        # Determina livello basato sui trigger
        new_level = self._calculate_level(triggers)
        
        if new_level != self.current_level:
            await self._transition_level(new_level)
            
        return self.current_level
    
    def _calculate_level(self, triggers: List[DegradationTrigger]) -> OperationalLevel:
        """Calcola il livello operativo dai trigger"""
        if not triggers:
            return OperationalLevel.L1_FULL
            
        critical_count = sum(1 for t in triggers if t.severity == 'critical')
        high_count = sum(1 for t in triggers if t.severity == 'high')
        
        if critical_count >= 2:
            return OperationalLevel.L4_MINIMAL
        elif critical_count >= 1:
            return OperationalLevel.L3_ESSENTIAL
        elif high_count >= 2:
            return OperationalLevel.L3_ESSENTIAL
        elif high_count >= 1:
            return OperationalLevel.L2_REDUCED
        else:
            return OperationalLevel.L2_REDUCED
    
    async def _transition_level(self, new_level: OperationalLevel):
        """Transizione a un nuovo livello operativo"""
        old_level = self.current_level
        self.current_level = new_level
        
        # Aggiorna feature status
        for name, feature in self.features.items():
            should_enable = feature.level_required.value >= new_level.value
            if feature.enabled != should_enable:
                feature.enabled = should_enable
                feature.last_check = datetime.now()
                feature.reason_disabled = None if should_enable else f"Level {new_level.name} active"
        
        # Log transizione
        transition_record = {
            'timestamp': datetime.now().isoformat(),
            'from_level': old_level.name,
            'to_level': new_level.name,
            'triggers': [
                {
                    'type': t.trigger_type,
                    'severity': t.severity,
                    'value': t.current_value
                }
                for t in self.triggers
            ],
            'features_affected': [
                name for name, f in self.features.items()
                if f.level_required.value < new_level.value
            ]
        }
        self.history.append(transition_record)
        
        logger.warning(
            f"🔄 DEGRADATION: {old_level.name} → {new_level.name} | "
            f"Triggers: {len(self.triggers)} | "
            f"Features disabled: {len(transition_record['features_affected'])}"
        )
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Verifica se una feature è abilitata al livello corrente"""
        feature = self.features.get(feature_name)
        if not feature:
            return False
        return feature.enabled
    
    def get_available_features(self) -> List[str]:
        """Ritorna la lista delle feature disponibili"""
        return [name for name, f in self.features.items() if f.enabled]
    
    def get_status(self) -> Dict[str, Any]:
        """Ritorna lo stato completo del sistema di degradazione"""
        return {
            'current_level': self.current_level.name,
            'level_value': self.current_level.value,
            'active_triggers': [
                {
                    'type': t.trigger_type,
                    'severity': t.severity,
                    'threshold': t.threshold,
                    'current': t.current_value
                }
                for t in self.triggers
            ],
            'features': {
                name: {
                    'enabled': f.enabled,
                    'required_level': f.level_required.name,
                    'reason_disabled': f.reason_disabled
                }
                for name, f in self.features.items()
            },
            'available_features': self.get_available_features(),
            'recent_transitions': self.history[-5:] if self.history else []
        }
    
    async def force_level(self, level: OperationalLevel, reason: str):
        """Forza un livello operativo specifico (per emergenze)"""
        logger.warning(f"⚠️ FORCED DEGRADATION to {level.name}: {reason}")
        self.triggers.append(DegradationTrigger(
            trigger_type='manual_override',
            threshold=0,
            current_value=0,
            severity='critical'
        ))
        await self._transition_level(level)
    
    async def attempt_recovery(self) -> bool:
        """Tenta di recuperare a un livello superiore"""
        if self.current_level == OperationalLevel.L1_FULL:
            return True
            
        # Rimuovi trigger scaduti (più vecchi di 5 minuti)
        cutoff = datetime.now()
        self.triggers = [
            t for t in self.triggers
            if (cutoff - t.timestamp).total_seconds() < 300
        ]
        
        # Ricalcola livello
        new_level = self._calculate_level(self.triggers)
        
        if new_level.value < self.current_level.value:
            await self._transition_level(new_level)
            logger.info(f"✅ RECOVERY: Upgraded to {new_level.name}")
            return True
            
        return False
