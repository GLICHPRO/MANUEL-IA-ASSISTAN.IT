"""
🗣️ Voice Module - GIDEON 3.0

Natural voice interaction system with:
- Text-to-Speech with natural pauses and intonations
- Response composition after complete reasoning
- Conversation management
"""

from .manager import VoiceManager
from .natural_voice import (
    NaturalVoiceEngine,
    ResponseComposer,
    ConversationManager,
    VoiceEmotion,
    VoiceProfile,
    ReasoningContext,
    ComposedResponse,
    SpeechOutput,
    SpeechSegment,
    PauseType,
    IntonationPattern,
    ConversationTurn
)

__all__ = [
    # Manager
    'VoiceManager',
    
    # Natural Voice Engine
    'NaturalVoiceEngine',
    'VoiceEmotion',
    'VoiceProfile',
    'SpeechOutput',
    'SpeechSegment',
    'PauseType',
    'IntonationPattern',
    
    # Response Composer
    'ResponseComposer',
    'ReasoningContext',
    'ComposedResponse',
    
    # Conversation Manager
    'ConversationManager',
    'ConversationTurn'
]
