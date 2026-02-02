"""
🎯 GIDEON 3.0 + JARVIS CORE - Core Module
Moduli fondamentali del sistema cognitivo
"""

from .config import settings, Settings
from .events import EventEmitter

# Nuovi moduli Gideon 3.0 + Jarvis Core
from .mode_manager import ModeManager, OperatingMode
from .action_logger import ActionLogger, RollbackManager, ActionRecord
from .emergency import EmergencySystem, EmergencyLevel
from .plugin_manager import PluginManager, PluginInterface
from .agent_bus import AgentBus, Agent, AgentMessage, GideonAgent, JarvisAgent
from .orchestrator import Orchestrator, ProcessingPhase
from .communication import (
    CommunicationBridge, CommunicationChannel,
    JarvisCommunicator, GideonCommunicator,
    Objective, Constraint, GideonResponse as CommGideonResponse,
    CommunicationMessage, MessageType, MessagePriority, MessageStatus, Sender
)
from .reasoning import (
    MultiStepReasoning, ReasoningCommunicator,
    ReasoningChain, ReasoningStep, VerificationCheckpoint,
    StepType, StepStatus, VerificationResult
)

# Memory Systems
from .memory_system import (
    MemorySystem, ShortTermMemory, LongTermMemory,
    MemoryItem, LearnedPattern, Strategy, MemoryType, MemoryPriority
)
from .episodic_memory import (
    EpisodicMemory, Event, Episode, Scenario,
    EventType, EpisodeStatus, ScenarioType
)
from .decision_memory import (
    DecisionMemory, DecisionRecord, ActionResult, LearningInsight,
    DecisionOutcome, FeedbackType, AdaptationType, Adaptation
)
from .continuous_learning import (
    ContinuousLearningAdapter, UnifiedMemoryCoordinator,
    LearningMode, LearningMetrics, CrossSystemPattern
)

# Avatar System
from .avatar_controller import (
    AvatarController, AvatarState, Expression, GazeTarget,
    HeadPose, EyeState, MouthState, HologramEffect,
    Viseme, AvatarAnimations,
    AvatarFeedbackSystem, OperatingModeColors,
    HUDIndicator, CalculationAnimation
)

__all__ = [
    # Config
    'settings', 'Settings',
    'EventEmitter',
    
    # Mode Management
    'ModeManager', 'OperatingMode',
    
    # Logging & Rollback
    'ActionLogger', 'RollbackManager', 'ActionRecord',
    
    # Emergency
    'EmergencySystem', 'EmergencyLevel',
    
    # Plugins
    'PluginManager', 'PluginInterface',
    
    # Multi-Agent
    'AgentBus', 'Agent', 'AgentMessage', 'GideonAgent', 'JarvisAgent',
    
    # Orchestration
    'Orchestrator', 'ProcessingPhase',
    
    # Communication Jarvis ↔ Gideon
    'CommunicationBridge', 'CommunicationChannel',
    'JarvisCommunicator', 'GideonCommunicator',
    'Objective', 'Constraint', 'CommGideonResponse',
    'CommunicationMessage', 'MessageType', 'MessagePriority', 'MessageStatus', 'Sender',
    
    # Multi-Step Reasoning
    'MultiStepReasoning', 'ReasoningCommunicator',
    'ReasoningChain', 'ReasoningStep', 'VerificationCheckpoint',
    'StepType', 'StepStatus', 'VerificationResult',
    
    # Memory Systems
    'MemorySystem', 'ShortTermMemory', 'LongTermMemory',
    'MemoryItem', 'LearnedPattern', 'Strategy', 'MemoryType', 'MemoryPriority',
    
    # Episodic Memory
    'EpisodicMemory', 'Event', 'Episode', 'Scenario',
    'EventType', 'EpisodeStatus', 'ScenarioType',
    
    # Decision Memory
    'DecisionMemory', 'DecisionRecord', 'ActionResult', 'LearningInsight',
    'DecisionOutcome', 'FeedbackType', 'AdaptationType', 'Adaptation',
    
    # Continuous Learning
    'ContinuousLearningAdapter', 'UnifiedMemoryCoordinator',
    'LearningMode', 'LearningMetrics', 'CrossSystemPattern',
    
    # Avatar System
    'AvatarController', 'AvatarState', 'Expression', 'GazeTarget',
    'HeadPose', 'EyeState', 'MouthState', 'HologramEffect',
    'Viseme', 'AvatarAnimations',
    
    # Avatar Feedback System
    'AvatarFeedbackSystem', 'OperatingModeColors',
    'HUDIndicator', 'CalculationAnimation'
]
