"""
🛠️ GIDEON Advanced Tools System
================================

Tool ≠ potere
Tool = capacità di analisi + automazione sicura

Ogni tool ha:
- Input normalizzati
- Motore di analisi
- Set di azioni consentite
- Output spiegabile
- Hook per automazioni
"""

from .security import SecurityTool, get_security_tool
from .cyber import CyberTool, get_cyber_tool
from .science import ScienceTool, get_science_tool
from .archaeology import ArchaeologyTool, get_archaeology_tool
from .core import GideonCore, get_gideon_core

__all__ = [
    'SecurityTool',
    'CyberTool', 
    'ScienceTool',
    'ArchaeologyTool',
    'GideonCore',
    'get_security_tool',
    'get_cyber_tool',
    'get_science_tool',
    'get_archaeology_tool',
    'get_gideon_core',
]


def get_all_tools():
    """Get all tool instances"""
    return {
        "security": get_security_tool(),
        "cyber": get_cyber_tool(),
        "science": get_science_tool(),
        "archaeology": get_archaeology_tool(),
        "core": get_gideon_core(),
    }


def get_tool_capabilities():
    """Get summary of all tool capabilities"""
    return {
        "security": {
            "name": "Security Tool",
            "icon": "🔒",
            "actions": [
                "predictive_risk_mapping",
                "anomaly_narrator",
                "defensive_scenario_simulator"
            ],
            "description": "Physical & Infrastructure Defense"
        },
        "cyber": {
            "name": "Cyber Tool",
            "icon": "🛡️",
            "actions": [
                "behavioral_baseline_builder",
                "incident_explainability_engine",
                "supply_chain_trust_scanner"
            ],
            "description": "Defensive AI-SOC"
        },
        "science": {
            "name": "Science Tool (SAFE)",
            "icon": "🧬",
            "actions": [
                "molecular_pattern_validator",
                "environmental_contamination_scan",
                "scientific_cross_check"
            ],
            "description": "Health/Chemistry Analysis - NO SYNTHESIS",
            "safety_mode": True
        },
        "archaeology": {
            "name": "Archaeology Tool",
            "icon": "🏛️",
            "actions": [
                "predictive_reconstruction",
                "temporal_layer_fusion",
                "authenticity_risk_assessment"
            ],
            "description": "Digital Heritage Analysis"
        },
        "core": {
            "name": "Gideon Core",
            "icon": "🧠",
            "actions": [
                "multi_tool_reasoning",
                "confidence_weighted_output",
                "human_override_gate",
                "audit_trail",
                "bias_and_drift_monitor",
                "failsafe_trigger"
            ],
            "description": "Central Reasoning & Safety Systems"
        }
    }
