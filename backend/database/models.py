"""
Database models for Gideon memory and learning system
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class Interaction(BaseModel):
    """User interaction record"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    timestamp: datetime
    query: str
    response: str
    intent: str
    confidence: float
    mode: str
    context: Optional[Dict[str, Any]] = None
    user_feedback: Optional[str] = None  # 'positive', 'negative', 'neutral'


class AnalysisResult(BaseModel):
    """System analysis result"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    timestamp: datetime
    analysis_type: str
    target: str
    findings: Dict[str, Any]
    recommendations: list[str]
    priority: str  # 'low', 'medium', 'high', 'critical'


class LearnedPattern(BaseModel):
    """Learned user patterns and preferences"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    pattern_type: str  # 'command_preference', 'time_pattern', 'topic_interest'
    pattern_data: Dict[str, Any]
    frequency: int
    last_seen: datetime
    confidence_score: float


class ContextMemory(BaseModel):
    """Short-term contextual memory"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    session_id: str
    timestamp: datetime
    context_type: str
    context_data: Dict[str, Any]
    expiry: datetime

