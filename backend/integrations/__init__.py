"""
🔗 GIDEON Integrations

Moduli di integrazione con servizi esterni:
- ai_hub: Hub multi-AI (OpenAI, Anthropic, ElevenLabs, HeyGen, etc.)
- github_integration: GitHub API
"""

from .github_integration import GitHubIntegration, github

# AI Hub import (lazy per evitare dipendenze circolari)
def get_ai_hub():
    from .ai_hub import get_ai_hub as _get_ai_hub
    return _get_ai_hub()

async def init_ai_hub():
    from .ai_hub import init_ai_hub as _init_ai_hub
    return await _init_ai_hub()

__all__ = [
    "GitHubIntegration", 
    "github",
    "get_ai_hub",
    "init_ai_hub"
]
