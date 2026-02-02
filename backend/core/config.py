"""
Core Configuration for Gideon 2.0
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path
import os

# Load .env from backend folder
from dotenv import load_dotenv

# Try multiple locations for .env
env_paths = [
    Path(__file__).parent.parent / ".env",  # backend/.env
    Path(__file__).parent.parent.parent / ".env",  # project root/.env
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded .env from: {env_path}")
        break
else:
    print("⚠️ No .env file found!")


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )
    
    # Application
    APP_NAME: str = "Gideon 2.0"
    VERSION: str = "2.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"
    
    # CORS - Allow all local origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000", 
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "*"  # Allow all for development
    ]
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./gideon2.db"
    REDIS_URL: str = "redis://localhost:6379/0"  # Optional for caching
    
    # Voice System
    WAKE_WORD: str = "gideon"
    VOICE_LANGUAGE: str = "it-IT"
    TTS_VOICE: str = "it-IT-GiuseppeNeural"  # VOCE UFFICIALE DI GIDEON
    VOICE_PROVIDER: str = "azure"  # azure, google, local
    
    # Azure Speech (if using Azure)
    AZURE_SPEECH_KEY: str = os.getenv("AZURE_SPEECH_KEY", "")
    AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "westeurope")
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.7
    
    # Anthropic Claude
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
    
    # Google Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_MODEL: str = "gemini-1.5-flash"
    
    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # Ollama (local)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = "llama3.2"
    
    # OpenRouter (multimodal gateway)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_DEFAULT_MODEL: str = os.getenv("OPENROUTER_DEFAULT_MODEL", "openai/gpt-4o-mini")
    OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "https://gideon.ai")
    OPENROUTER_SITE_NAME: str = os.getenv("OPENROUTER_SITE_NAME", "GIDEON 3.0")
    
    # Default AI Provider (openai, anthropic, google, groq, ollama, openrouter)
    DEFAULT_AI_PROVIDER: str = os.getenv("DEFAULT_AI_PROVIDER", "openrouter")
    
    # Security
    PILOT_PHRASE: str = "Autorizzazione Pilot Alfa Zero Uno"
    PILOT_TIMEOUT: int = 300  # seconds
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Avatar
    AVATAR_MODEL: str = "default"
    EXPRESSIONS_ENABLED: bool = True
    LIP_SYNC_ENABLED: bool = True
    
    # Analysis
    ANALYSIS_INTERVAL: int = 60  # seconds
    MAX_MEMORY_ITEMS: int = 100
    OPTIMIZATION_THRESHOLD: float = 0.7


settings = Settings()
