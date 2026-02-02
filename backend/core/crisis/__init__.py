"""
🚨 GIDEON Crisis Management Core
================================
Capacità fondamentali per situazioni critiche

Moduli:
- GracefulDegradation: Riduzione progressiva funzionalità
- DecisionFreeze: Sospensione decisioni automatiche
- ExplainFirst: Protocollo spiegazione obbligatoria
- RoleAware: Output adattati al ruolo
- Counterfactual: Simulazione scenari alternativi
- EthicalTripwire: Sistema allarme etico
- PostCrisis: Ricostruzione post-crisi
- CognitiveLoad: Monitoraggio carico cognitivo umano
- RedundantReasoning: Ragionamento ridondante
- SafeState: Stato di sicurezza default
"""

from .graceful_degradation import GracefulDegradationEngine
from .decision_freeze import DecisionFreezeController
from .explain_first import ExplainFirstProtocol
from .role_aware import RoleAwareOutputAdapter
from .counterfactual import CounterfactualSimulator
from .ethical_tripwire import EthicalTripwireSystem
from .post_crisis import PostCrisisReconstructor
from .cognitive_load import CognitiveLoadMonitor
from .redundant_reasoning import RedundantReasoningEngine
from .safe_state import SafeStateController

__all__ = [
    'GracefulDegradationEngine',
    'DecisionFreezeController', 
    'ExplainFirstProtocol',
    'RoleAwareOutputAdapter',
    'CounterfactualSimulator',
    'EthicalTripwireSystem',
    'PostCrisisReconstructor',
    'CognitiveLoadMonitor',
    'RedundantReasoningEngine',
    'SafeStateController'
]
