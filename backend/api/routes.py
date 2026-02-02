"""
API Routes - Main router for Gideon
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
import os
from datetime import datetime

main_router = APIRouter()

@main_router.get("/status")
async def status():
    return {"status": "ok", "version": "2.0.0"}

voice_router = APIRouter()

@voice_router.post("/command")
async def voice_command(text: str):
    return {"received": text}

analysis_router = APIRouter()

@analysis_router.get("/system")
async def analyze_system():
    return {"analysis": "system"}

pilot_router = APIRouter()

# Global level state
current_level = "normal"

class LevelRequest(BaseModel):
    level: str

@pilot_router.post("/activate")
async def activate_pilot():
    global current_level
    current_level = "pilot"
    return {"pilot": "activated", "level": current_level}

@pilot_router.post("/deactivate")
async def deactivate_pilot():
    global current_level
    current_level = "normal"
    return {"pilot": "deactivated", "level": current_level}

@pilot_router.get("/level")
async def get_level():
    """Get current operation level"""
    return {"level": current_level}

@pilot_router.post("/level")
async def set_level(request: LevelRequest):
    """Set operation level: normal, advanced, pilot"""
    global current_level
    if request.level not in ["normal", "advanced", "pilot"]:
        raise HTTPException(status_code=400, detail="Invalid level. Must be: normal, advanced, pilot")
    current_level = request.level
    
    # Sync personality level with assistant instance
    if _assistant_instance:
        _assistant_instance.set_operation_level(request.level)
    
    return {"level": current_level, "updated": True}

@pilot_router.get("/personality")
async def get_personality_info():
    """Get current personality traits and info"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'personality'):
            personality = _assistant_instance.personality
            traits = personality.get_personality_info()
            return {
                "level": personality.current_level.value,
                "traits": traits,
                "sample_greeting": personality.get_greeting(),
                "sample_acknowledgment": personality.get_acknowledgment()
            }
        else:
            return {
                "level": current_level,
                "traits": {},
                "message": "Personality not fully initialized"
            }
    except Exception as e:
        return {"level": current_level, "error": str(e)}

# Logs router
logs_router = APIRouter()

@logs_router.get("/recent")
async def get_recent_logs(lines: int = 100):
    """Get recent conversation logs"""
    try:
        log_dir = "logs"
        log_files = [f for f in os.listdir(log_dir) if f.startswith("conversations_")]
        if not log_files:
            return {"logs": [], "message": "No conversation logs found"}
        
        # Get most recent log file
        log_files.sort(reverse=True)
        latest_log = os.path.join(log_dir, log_files[0])
        
        with open(latest_log, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": [line.strip() for line in recent],
            "file": log_files[0],
            "total_lines": len(all_lines)
        }
    except Exception as e:
        return {"logs": [], "error": str(e)}

@logs_router.get("/stats")
async def get_log_stats():
    """Get logging statistics"""
    try:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            return {"stats": {}, "message": "No logs directory"}
        
        files = os.listdir(log_dir)
        conversation_logs = [f for f in files if f.startswith("conversations_")]
        gideon_logs = [f for f in files if f.startswith("gideon_")]
        
        total_size = sum(
            os.path.getsize(os.path.join(log_dir, f)) 
            for f in files if os.path.isfile(os.path.join(log_dir, f))
        )
        
        return {
            "stats": {
                "conversation_log_files": len(conversation_logs),
                "system_log_files": len(gideon_logs),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "log_directory": log_dir
            }
        }
    except Exception as e:
        return {"stats": {}, "error": str(e)}

# Memory router
memory_router = APIRouter()

# Reference to global assistant instance (set from main.py)
_assistant_instance = None

def set_assistant_instance(assistant):
    """Set the global assistant instance reference"""
    global _assistant_instance
    _assistant_instance = assistant

@memory_router.get("/conversation")
async def get_conversation_history(limit: int = 20):
    """Get recent conversation history"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'memory'):
            memory = _assistant_instance.memory
            history = memory.short_term_memory[-limit:] if memory.short_term_memory else []
            return {
                "history": history,
                "count": len(history),
                "total": len(memory.conversation_history) if hasattr(memory, 'conversation_history') else 0
            }
        else:
            return {"history": [], "count": 0, "message": "Memory not initialized yet"}
    except Exception as e:
        return {"history": [], "count": 0, "error": str(e)}

@memory_router.get("/suggestions")
async def get_suggestions():
    """Get smart suggestions based on learned patterns"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'memory'):
            suggestions = await _assistant_instance.memory.suggest_next_actions()
            return {"suggestions": suggestions}
        else:
            return {"suggestions": ["Check system status", "Ask the time", "Run diagnostics"]}
    except Exception as e:
        return {"suggestions": [], "error": str(e)}

@memory_router.get("/patterns")
async def get_learned_patterns():
    """Get learned user patterns"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'memory'):
            return {
                "patterns": dict(_assistant_instance.memory.learned_patterns),
                "preferences": _assistant_instance.memory.user_preferences
            }
        else:
            return {"patterns": {}, "preferences": {}}
    except Exception as e:
        return {"patterns": {}, "preferences": {}, "error": str(e)}

@memory_router.get("/analysis")
async def analyze_history():
    """Analyze interaction history for improvements"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'memory'):
            analysis = await _assistant_instance.memory.analyze_history_for_improvements()
            return {"analysis": analysis}
        else:
            return {"analysis": {"message": "Memory not initialized yet"}}
    except Exception as e:
        return {"analysis": {}, "error": str(e)}

@memory_router.get("/scenarios")
async def get_scenarios(scenario_type: Optional[str] = None, limit: int = 20):
    """Get past scenarios and outcomes"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'memory'):
            scenarios = await _assistant_instance.memory.get_scenario_history(scenario_type, limit)
            return {"scenarios": scenarios, "count": len(scenarios)}
        else:
            return {"scenarios": [], "count": 0}
    except Exception as e:
        return {"scenarios": [], "count": 0, "error": str(e)}


# ============ ACTIONS ROUTER ============
actions_router = APIRouter()

class ActionRequest(BaseModel):
    action_type: str
    params: Optional[dict] = None
    force: bool = False
    pilot_confirmed: bool = False

class RoutineToggle(BaseModel):
    routine_id: str
    enabled: bool

@actions_router.get("/available")
async def get_available_actions():
    """Get list of available actions"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'action_manager'):
            actions = _assistant_instance.action_manager.get_available_actions()
            return {"actions": actions, "count": len(actions)}
        return {"actions": [], "error": "Action manager not initialized"}
    except Exception as e:
        return {"actions": [], "error": str(e)}

@actions_router.post("/execute")
async def execute_action(request: ActionRequest):
    """Execute an action"""
    try:
        if not _assistant_instance or not hasattr(_assistant_instance, 'action_manager'):
            raise HTTPException(status_code=503, detail="Action manager not initialized")
        
        result = await _assistant_instance.action_manager.execute_action(
            action_type=request.action_type,
            params=request.params,
            force=request.force,
            pilot_confirmed=request.pilot_confirmed
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@actions_router.get("/logs")
async def get_action_logs(limit: int = 50, status: Optional[str] = None, action_type: Optional[str] = None):
    """Get action execution logs"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'action_manager'):
            from brain.action_manager import ActionStatus
            
            status_filter = None
            if status:
                try:
                    status_filter = ActionStatus(status)
                except ValueError:
                    pass
                    
            logs = _assistant_instance.action_manager.get_action_logs(
                limit=limit,
                status=status_filter,
                action_type=action_type
            )
            return {"logs": logs, "count": len(logs)}
        return {"logs": [], "error": "Action manager not initialized"}
    except Exception as e:
        return {"logs": [], "error": str(e)}

@actions_router.get("/routines")
async def get_routines():
    """Get all scheduled routines"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'action_manager'):
            routines = _assistant_instance.action_manager.get_routines()
            return {"routines": routines, "count": len(routines)}
        return {"routines": [], "error": "Action manager not initialized"}
    except Exception as e:
        return {"routines": [], "error": str(e)}

@actions_router.post("/routines/toggle")
async def toggle_routine(request: RoutineToggle):
    """Enable or disable a routine"""
    try:
        if not _assistant_instance or not hasattr(_assistant_instance, 'action_manager'):
            raise HTTPException(status_code=503, detail="Action manager not initialized")
        
        success = _assistant_instance.action_manager.toggle_routine(
            request.routine_id,
            request.enabled
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to toggle routine. May require Pilot Mode for critical routines."
            )
            
        return {"success": True, "routine_id": request.routine_id, "enabled": request.enabled}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@actions_router.get("/rollback")
async def get_rollback_stack():
    """Get actions that can be rolled back"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'action_manager'):
            stack = _assistant_instance.action_manager.get_rollback_stack()
            return {"rollback_stack": stack, "count": len(stack)}
        return {"rollback_stack": [], "error": "Action manager not initialized"}
    except Exception as e:
        return {"rollback_stack": [], "error": str(e)}

@actions_router.post("/rollback")
async def rollback_last_action():
    """Rollback the last rollbackable action"""
    try:
        if not _assistant_instance or not hasattr(_assistant_instance, 'action_manager'):
            raise HTTPException(status_code=503, detail="Action manager not initialized")
        
        result = await _assistant_instance.action_manager.rollback_last()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ AI Providers Router ============
ai_router = APIRouter()

class AIQueryRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None  # openai, anthropic, google, groq, ollama
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: Optional[str] = None
    include_history: bool = True

class AIStreamRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None

@ai_router.get("/providers")
async def get_ai_providers():
    """Get list of available AI providers"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'ai_manager'):
            providers = _assistant_instance.ai_manager.get_available_providers()
            status = _assistant_instance.ai_manager.get_status()
            return {
                "success": True,
                "providers": providers,
                "status": status
            }
        return {"success": False, "providers": [], "error": "AI manager not initialized"}
    except Exception as e:
        return {"success": False, "providers": [], "error": str(e)}

@ai_router.post("/query")
async def ai_query(request: AIQueryRequest):
    """Send a query to an AI provider"""
    try:
        if not _assistant_instance or not hasattr(_assistant_instance, 'ai_manager'):
            raise HTTPException(status_code=503, detail="AI manager not initialized")
        
        # Get conversation history if requested
        conversation_history = None
        if request.include_history and hasattr(_assistant_instance, 'memory'):
            history = await _assistant_instance.memory.get_conversation_history(limit=5)
            conversation_history = [
                {"role": "user" if h.get("role") == "user" else "assistant", "content": h.get("content", "")}
                for h in history
            ]
        
        response = await _assistant_instance.ai_manager.generate(
            prompt=request.prompt,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            conversation_history=conversation_history,
            fallback=True
        )
        
        return {
            "success": response.success,
            "content": response.content,
            "provider": response.provider,
            "model": response.model,
            "tokens_used": response.tokens_used,
            "finish_reason": response.finish_reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@ai_router.get("/models/{provider}")
async def get_provider_models(provider: str):
    """Get available models for a specific provider"""
    try:
        if not _assistant_instance or not hasattr(_assistant_instance, 'ai_manager'):
            raise HTTPException(status_code=503, detail="AI manager not initialized")
        
        ai_provider = _assistant_instance.ai_manager.get_provider(provider)
        if not ai_provider:
            raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")
        
        models = getattr(ai_provider, "MODELS", {})
        default_model = getattr(ai_provider, "default_model", "")
        
        return {
            "provider": provider,
            "name": ai_provider.provider_name,
            "available": ai_provider.is_available if hasattr(ai_provider, 'is_available') else False,
            "models": models,
            "default_model": default_model
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@ai_router.post("/chat")
async def ai_chat(request: AIQueryRequest):
    """Have a conversation with Gideon using AI"""
    try:
        if not _assistant_instance:
            raise HTTPException(status_code=503, detail="Assistant not initialized")
        
        # Process through the main assistant for full Gideon experience
        result = await _assistant_instance.process_command(
            text=f"chiedi all'ai {request.prompt}",
            mode="text"
        )
        
        return {
            "success": result.get("success", False),
            "response": result.get("text", ""),
            "data": result.get("data", {}),
            "intent": result.get("intent", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@ai_router.get("/status")
async def get_ai_status():
    """Get detailed status of all AI providers"""
    try:
        if _assistant_instance and hasattr(_assistant_instance, 'ai_manager'):
            return _assistant_instance.ai_manager.get_status()
        return {"initialized": False, "error": "AI manager not available"}
    except Exception as e:
        return {"initialized": False, "error": str(e)}


# ============================================================================
# SOCIAL MEDIA ANALYZER ROUTES
# ============================================================================

from brain.social_analyzer import social_analyzer

social_router = APIRouter()


class InstagramExportRequest(BaseModel):
    export_path: Optional[str] = None
    followers_json: Optional[str] = None
    following_json: Optional[str] = None


@social_router.get("/status")
async def social_status():
    """Get status of social analyzer module"""
    return social_analyzer.get_status()


@social_router.post("/instagram/analyze")
async def analyze_instagram(request: InstagramExportRequest):
    """
    Analizza i dati Instagram esportati.
    
    Usa il tuo export Instagram (Impostazioni → Privacy → Scarica i tuoi dati).
    Puoi fornire:
    - export_path: percorso alla cartella estratta
    - followers_json + following_json: contenuto JSON diretto
    """
    try:
        result = social_analyzer.analyze_instagram(
            export_path=request.export_path,
            followers_json=request.followers_json,
            following_json=request.following_json
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@social_router.get("/instagram/not-following-back")
async def get_not_following_back():
    """
    Ottieni la lista di chi segui ma NON ti segue.
    Richiede prima di aver caricato i dati con /instagram/analyze
    """
    if not social_analyzer.instagram.data_loaded:
        raise HTTPException(
            status_code=400, 
            detail="Dati non caricati. Prima usa POST /api/social/instagram/analyze"
        )
    return social_analyzer.instagram.get_not_following_back()


@social_router.get("/instagram/summary")
async def get_instagram_summary():
    """
    Riepilogo completo dell'analisi Instagram.
    """
    if not social_analyzer.instagram.data_loaded:
        raise HTTPException(
            status_code=400, 
            detail="Dati non caricati. Prima usa POST /api/social/instagram/analyze"
        )
    return social_analyzer.instagram.get_analysis_summary()


class ManualFollowersRequest(BaseModel):
    followers: list[str]
    following: list[str]


@social_router.post("/instagram/manual")
async def analyze_manual_lists(request: ManualFollowersRequest):
    """
    Analizza liste manuali di followers e following.
    Utile per test o se hai già le liste in formato semplice.
    
    Esempio:
    {
        "followers": ["user1", "user2", "user3"],
        "following": ["user1", "user4", "user5"]
    }
    """
    try:
        social_analyzer.instagram.followers = set(u.lower() for u in request.followers)
        social_analyzer.instagram.following = set(u.lower() for u in request.following)
        social_analyzer.instagram.data_loaded = True
        social_analyzer.instagram.load_timestamp = datetime.now()
        
        return social_analyzer.instagram.get_analysis_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
