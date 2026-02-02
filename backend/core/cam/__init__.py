"""
🚨 GIDEON — CRISIS AUTOMATION MODE (CAM)
=========================================
Massima potenza cognitiva, non offensiva, ultra-robusta.
Progettato per funzionare quando tutto va male.

CAM non rende Gideon più aggressivo
👉 lo rende più lucido, più potente, più umano

🧠 PRINCIPI FONDAMENTALI (non negoziabili):
- Explain > Act
- Slow is Smooth, Smooth is Fast
- Uncertainty is a Signal
- Human-in-the-Loop sempre
- Safe-State > Wrong-State

🧩 STRUTTURA:
├─ Detection Layer      - "Siamo in crisi?"
├─ Control Layer        - Governa la potenza
├─ Reasoning Layer      - Massima potenza cognitiva
├─ Automation Layer     - Automazioni intelligenti
├─ Human Interface      - Anti-panico
└─ Recovery Layer       - Post-crisi e lessons learned

🔐 REGOLE D'ORO (hard-coded):
❌ Nessuna decisione irreversibile automatica
❌ Nessuna azione senza spiegazione
❌ Nessuna escalation senza consenso umano
✅ Sempre audit trail
✅ Sempre possibilità di STOP
"""

# Detection Layer
from .detection import (
    CrisisSignalAggregator,
    CrisisSignal,
    CrisisLevel,
    CrisisAssessment,
    SignalSource,
    SignalType,
    EmotionalInputDetector,
    get_signal_aggregator
)

# Control Layer
from .control import (
    ControlLayer,
    AutonomyClamp,
    TemporalGovernor,
    SafeStateEnforcer,
    ControlState,
    AutonomyDimension,
    get_control_layer
)

# Reasoning Layer
from .reasoning import (
    ReasoningLayer,
    MultiPathReasoningEngine,
    UncertaintyMapper,
    NoActionIntelligence,
    ReasoningPath,
    ReasoningResult,
    UncertaintyMap,
    NoActionAnalysis,
    CertaintyLevel,
    get_reasoning_layer
)

# Automation Layer
from .automation import (
    AutomationLayer,
    CAMAutomationManager,
    PreDecisionProcessor,
    RiskTriggeredActions,
    SelfCorrectionLoop,
    AutomationType,
    ActionReversibility,
    AutomationAction,
    get_automation_layer
)

# Human Interface Layer
from .human_interface import (
    HumanInterfaceLayer,
    CrisisUIMode,
    RoleAwareViews,
    CognitiveLoadMonitor,
    UserRole,
    UIMode,
    CognitiveState,
    InformationPriority,
    get_human_interface_layer
)

# Recovery Layer
from .recovery import (
    RecoveryLayer,
    CrisisTimelineReconstructor,
    LessonsExtractor,
    GradualPowerRestore,
    TimelineEvent,
    Lesson,
    LessonCategory,
    RestorePhase,
    EventType,
    get_recovery_layer
)

# Main Orchestrator
from .orchestrator import (
    CrisisAutomationMode,
    CAMStatus,
    CAMState,
    CAMEvent,
    CAMDecisionRequest,
    get_cam,
    initialize_cam
)

__all__ = [
    # === MAIN ORCHESTRATOR ===
    'CrisisAutomationMode',
    'CAMStatus',
    'CAMState',
    'CAMEvent',
    'CAMDecisionRequest',
    'get_cam',
    'initialize_cam',
    
    # === DETECTION LAYER ===
    'CrisisSignalAggregator',
    'CrisisSignal',
    'CrisisLevel',
    'CrisisAssessment',
    'SignalSource',
    'SignalType',
    'EmotionalInputDetector',
    'get_signal_aggregator',
    
    # === CONTROL LAYER ===
    'ControlLayer',
    'AutonomyClamp',
    'TemporalGovernor',
    'SafeStateEnforcer',
    'ControlState',
    'AutonomyDimension',
    'get_control_layer',
    
    # === REASONING LAYER ===
    'ReasoningLayer',
    'MultiPathReasoningEngine',
    'UncertaintyMapper',
    'NoActionIntelligence',
    'ReasoningPath',
    'ReasoningResult',
    'UncertaintyMap',
    'NoActionAnalysis',
    'CertaintyLevel',
    'get_reasoning_layer',
    
    # === AUTOMATION LAYER ===
    'AutomationLayer',
    'CAMAutomationManager',
    'PreDecisionProcessor',
    'RiskTriggeredActions',
    'SelfCorrectionLoop',
    'AutomationType',
    'ActionReversibility',
    'AutomationAction',
    'get_automation_layer',
    
    # === HUMAN INTERFACE LAYER ===
    'HumanInterfaceLayer',
    'CrisisUIMode',
    'RoleAwareViews',
    'CognitiveLoadMonitor',
    'UserRole',
    'UIMode',
    'CognitiveState',
    'InformationPriority',
    'get_human_interface_layer',
    
    # === RECOVERY LAYER ===
    'RecoveryLayer',
    'CrisisTimelineReconstructor',
    'LessonsExtractor',
    'GradualPowerRestore',
    'TimelineEvent',
    'Lesson',
    'LessonCategory',
    'RestorePhase',
    'EventType',
    'get_recovery_layer'
]
