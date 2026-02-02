"""
Gideon AI Providers - Multi-Provider AI Integration
Supporta: OpenAI GPT, Anthropic Claude, Google Gemini, Groq, Ollama (locale), OpenRouter
"""

import asyncio
import httpx
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import os


class AIProvider(Enum):
    """Available AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    LOCAL = "local"


@dataclass
class AIMessage:
    """Standard message format for AI conversations"""
    role: str  # "user", "assistant", "system"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class AIResponse:
    """Standard response from AI providers"""
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    finish_reason: str = "complete"
    raw_response: Optional[Dict] = None
    
    @property
    def success(self) -> bool:
        return bool(self.content)


class BaseAIProvider(ABC):
    """Base class for all AI providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)  # Ridotto da 60s a 30s per risposta più veloce
        
    async def close(self):
        await self.client.aclose()
        
    @abstractmethod
    async def generate(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AIResponse:
        """Generate a response from the AI"""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from the AI"""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name"""
        pass


class OpenAIProvider(BaseAIProvider):
    """OpenAI GPT Provider (GPT-4, GPT-4o, GPT-3.5)"""
    
    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"
    
    MODELS = {
        "gpt-4o": "Modello più avanzato, multimodale",
        "gpt-4o-mini": "Veloce ed economico, ottimo per chat",
        "gpt-4-turbo": "GPT-4 Turbo con contesto 128K",
        "gpt-4": "GPT-4 classico",
        "gpt-3.5-turbo": "Veloce, economico, per task semplici"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"))
        self.default_model = os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    @property
    def provider_name(self) -> str:
        return "OpenAI"
    
    async def generate(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AIResponse:
        if not self.is_available:
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
            
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model or self.default_model,
                    "messages": [m.to_dict() for m in messages],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                return AIResponse(
                    content=choice["message"]["content"],
                    provider=self.provider_name,
                    model=data["model"],
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "complete"),
                    raw_response=data
                )
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return AIResponse(
                    content="",
                    provider=self.provider_name,
                    model=model or self.default_model,
                    finish_reason=f"error_{response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
    
    async def stream(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        if not self.is_available:
            yield ""
            return
            
        try:
            async with self.client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model or self.default_model,
                    "messages": [m.to_dict() for m in messages],
                    "stream": True,
                    **kwargs
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        try:
                            data = json.loads(line[6:])
                            delta = data["choices"][0].get("delta", {})
                            if content := delta.get("content"):
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"OpenAI stream failed: {e}")
            yield ""


class AnthropicProvider(BaseAIProvider):
    """Anthropic Claude Provider (Claude 3.5, Claude 3)"""
    
    BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    API_VERSION = "2023-06-01"
    
    MODELS = {
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet - Più intelligente",
        "claude-3-5-haiku-20241022": "Claude 3.5 Haiku - Veloce",
        "claude-3-opus-20240229": "Claude 3 Opus - Massima capacità",
        "claude-3-sonnet-20240229": "Claude 3 Sonnet - Bilanciato",
        "claude-3-haiku-20240307": "Claude 3 Haiku - Ultra veloce"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.default_model = os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    @property
    def provider_name(self) -> str:
        return "Anthropic Claude"
    
    async def generate(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AIResponse:
        if not self.is_available:
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
        
        # Separate system message from user messages
        system_content = ""
        user_messages = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                user_messages.append(msg.to_dict())
                
        try:
            payload = {
                "model": model or self.default_model,
                "messages": user_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if system_content:
                payload["system"] = system_content
                
            response = await self.client.post(
                f"{self.BASE_URL}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.API_VERSION,
                    "Content-Type": "application/json"
                },
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                        
                return AIResponse(
                    content=content,
                    provider=self.provider_name,
                    model=data.get("model", model or self.default_model),
                    tokens_used=data.get("usage", {}).get("input_tokens", 0) + 
                               data.get("usage", {}).get("output_tokens", 0),
                    finish_reason=data.get("stop_reason", "complete"),
                    raw_response=data
                )
            else:
                logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                return AIResponse(
                    content="",
                    provider=self.provider_name,
                    model=model or self.default_model,
                    finish_reason=f"error_{response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
    
    async def stream(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        if not self.is_available:
            yield ""
            return
            
        system_content = ""
        user_messages = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                user_messages.append(msg.to_dict())
                
        try:
            payload = {
                "model": model or self.default_model,
                "messages": user_messages,
                "max_tokens": 2000,
                "stream": True
            }
            if system_content:
                payload["system"] = system_content
                
            async with self.client.stream(
                "POST",
                f"{self.BASE_URL}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.API_VERSION,
                    "Content-Type": "application/json"
                },
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if text := delta.get("text"):
                                    yield text
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Anthropic stream failed: {e}")
            yield ""


class GoogleProvider(BaseAIProvider):
    """Google Gemini Provider"""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    DEFAULT_MODEL = "gemini-1.5-flash"
    
    MODELS = {
        "gemini-1.5-pro": "Gemini 1.5 Pro - Modello più capace",
        "gemini-1.5-flash": "Gemini 1.5 Flash - Veloce e versatile",
        "gemini-1.5-flash-8b": "Gemini 1.5 Flash 8B - Ultra leggero",
        "gemini-2.0-flash-exp": "Gemini 2.0 Flash - Sperimentale"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"))
        self.default_model = os.getenv("GOOGLE_MODEL", self.DEFAULT_MODEL)
        
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    @property
    def provider_name(self) -> str:
        return "Google Gemini"
    
    async def generate(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AIResponse:
        if not self.is_available:
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
        
        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
                
        try:
            model_name = model or self.default_model
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
                
            response = await self.client.post(
                f"{self.BASE_URL}/models/{model_name}:generateContent?key={self.api_key}",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = ""
                    for part in candidates[0].get("content", {}).get("parts", []):
                        content += part.get("text", "")
                    
                    return AIResponse(
                        content=content,
                        provider=self.provider_name,
                        model=model_name,
                        tokens_used=data.get("usageMetadata", {}).get("totalTokenCount", 0),
                        finish_reason=candidates[0].get("finishReason", "complete"),
                        raw_response=data
                    )
                    
            logger.error(f"Google API error: {response.status_code} - {response.text}")
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason=f"error_{response.status_code}"
            )
            
        except Exception as e:
            logger.error(f"Google request failed: {e}")
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
    
    async def stream(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        # Google Gemini streaming implementation
        if not self.is_available:
            yield ""
            return
            
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
        
        try:
            model_name = model or self.default_model
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2000
                }
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
                
            async with self.client.stream(
                "POST",
                f"{self.BASE_URL}/models/{model_name}:streamGenerateContent?key={self.api_key}&alt=sse",
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            candidates = data.get("candidates", [])
                            if candidates:
                                for part in candidates[0].get("content", {}).get("parts", []):
                                    if text := part.get("text"):
                                        yield text
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Google stream failed: {e}")
            yield ""


class GroqProvider(BaseAIProvider):
    """Groq Provider - Ultra-fast inference"""
    
    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    
    MODELS = {
        "llama-3.3-70b-versatile": "Llama 3.3 70B - Versatile e potente",
        "llama-3.1-70b-versatile": "Llama 3.1 70B - Ottimo per ragionamento",
        "llama-3.1-8b-instant": "Llama 3.1 8B - Ultra veloce",
        "mixtral-8x7b-32768": "Mixtral 8x7B - Contesto lungo",
        "gemma2-9b-it": "Gemma 2 9B - Google's open model"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("GROQ_API_KEY"))
        self.default_model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    @property
    def provider_name(self) -> str:
        return "Groq"
    
    async def generate(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AIResponse:
        if not self.is_available:
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
            
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model or self.default_model,
                    "messages": [m.to_dict() for m in messages],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                return AIResponse(
                    content=choice["message"]["content"],
                    provider=self.provider_name,
                    model=data["model"],
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "complete"),
                    raw_response=data
                )
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return AIResponse(
                    content="",
                    provider=self.provider_name,
                    model=model or self.default_model,
                    finish_reason=f"error_{response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Groq request failed: {e}")
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
    
    async def stream(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        if not self.is_available:
            yield ""
            return
            
        try:
            async with self.client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model or self.default_model,
                    "messages": [m.to_dict() for m in messages],
                    "stream": True,
                    **kwargs
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        try:
                            data = json.loads(line[6:])
                            delta = data["choices"][0].get("delta", {})
                            if content := delta.get("content"):
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Groq stream failed: {e}")
            yield ""


class OllamaProvider(BaseAIProvider):
    """Ollama Provider - Local AI models"""
    
    DEFAULT_MODEL = "llama3.2"
    
    MODELS = {
        "llama3.2": "Llama 3.2 - Ultimo modello Meta",
        "llama3.1": "Llama 3.1 - Molto capace",
        "mistral": "Mistral 7B - Veloce ed efficiente",
        "codellama": "Code Llama - Specializzato in codice",
        "gemma2": "Gemma 2 - Google's model",
        "phi3": "Phi-3 - Microsoft's small model",
        "qwen2.5": "Qwen 2.5 - Alibaba's model"
    }
    
    def __init__(self, base_url: Optional[str] = None):
        super().__init__()
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self._available = None
        
    @property
    def is_available(self) -> bool:
        # Will be checked async
        return True
    
    async def check_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=0.5)
            self._available = response.status_code == 200
            return self._available
        except:
            self._available = False
            return False
    
    @property
    def provider_name(self) -> str:
        return "Ollama (Local)"
    
    async def generate(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AIResponse:
        if not await self.check_available():
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="unavailable"
            )
            
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model or self.default_model,
                    "messages": [m.to_dict() for m in messages],
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False
                },
                timeout=120.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return AIResponse(
                    content=data.get("message", {}).get("content", ""),
                    provider=self.provider_name,
                    model=data.get("model", model or self.default_model),
                    tokens_used=data.get("eval_count", 0),
                    finish_reason="complete",
                    raw_response=data
                )
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return AIResponse(
                    content="",
                    provider=self.provider_name,
                    model=model or self.default_model,
                    finish_reason="error"
                )
                
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
    
    async def stream(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        if not await self.check_available():
            yield ""
            return
            
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": model or self.default_model,
                    "messages": [m.to_dict() for m in messages],
                    "stream": True
                },
                timeout=120.0
            ) as response:
                async for line in response.aiter_lines():
                    try:
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content"):
                            yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            yield ""


class OpenRouterProvider(BaseAIProvider):
    """
    OpenRouter Provider - Gateway multimodale per 100+ modelli AI.
    Supporta: Text, Vision, Video analysis.
    https://openrouter.ai
    
    MODELLI GRATUITI VERIFICATI (Gennaio 2026):
    - deepseek/deepseek-r1-0528:free - DeepSeek R1 ✅ FUNZIONA
    - mistralai/mistral-small-3.1-24b-instruct:free - Mistral ✅ FUNZIONA
    - qwen/qwen3-4b:free - Qwen 3 veloce ✅ FUNZIONA
    - meta-llama/llama-3.2-3b-instruct:free - Llama veloce
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "qwen/qwen3-4b:free"  # ⚡ VELOCISSIMO per ridurre lag
    
    # ======== MODELLI GRATUITI - ORDINATI PER VELOCITÀ ========
    # IMPORTANTE: I primi modelli sono i più VELOCI per minimizzare latenza
    FREE_MODELS = [
        # 🚀 ULTRA-FAST TIER (< 2 secondi)
        "qwen/qwen3-4b:free",                             # ⚡ #1 VELOCISSIMO
        "google/gemma-3-4b-it:free",                      # ⚡ Google 4B veloce
        "meta-llama/llama-3.2-3b-instruct:free",          # ⚡ Llama 3B veloce
        
        # ⚡ FAST TIER (2-4 secondi)
        "mistralai/mistral-small-3.1-24b-instruct:free",  # ✅ VERIFICATO
        "nvidia/nemotron-nano-9b-v2:free",                # NVIDIA veloce
        
        # BALANCED (solo se i veloci falliscono)
        "deepseek/deepseek-r1-0528:free",                 # DeepSeek R1
        "qwen/qwen3-coder:free",                          # Per codice
    ]
    
    # Modelli ordinati per velocità (più veloci prima)
    FAST_MODELS = [
        "qwen/qwen3-4b:free",                             # #1 Ultra-fast
        "google/gemma-3-4b-it:free",                      # #2 Google veloce
        "meta-llama/llama-3.2-3b-instruct:free",          # #3 Llama veloce
    ]
    
    MODELS = {
        # ========== FREE MODELS VERIFICATI (Gennaio 2026) ==========
        "mistralai/mistral-small-3.1-24b-instruct:free": "🏆 Mistral Small 3.1 24B - VERIFICATO",
        "deepseek/deepseek-r1-0528:free": "✅ DeepSeek R1 - Reasoning avanzato",
        "qwen/qwen3-4b:free": "⚡ Qwen 3 4B - Velocissimo",
        "qwen/qwen3-coder:free": "💻 Qwen 3 Coder - Per codice",
        "meta-llama/llama-3.2-3b-instruct:free": "⚡ Llama 3.2 3B - Veloce",
        "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B - Potente",
        "google/gemma-3-4b-it:free": "Google Gemma 3 4B",
        "google/gemma-3-12b-it:free": "Google Gemma 3 12B",
        "google/gemma-3-27b-it:free": "Google Gemma 3 27B",
        "nvidia/nemotron-nano-9b-v2:free": "NVIDIA Nemotron 9B",
        
        # ========== PAID MODELS (Premium) ==========
        "openai/gpt-4o": "GPT-4o - Multimodale avanzato",
        "openai/gpt-4o-mini": "GPT-4o Mini - Veloce ed economico",
        "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet - Eccellente",
        "anthropic/claude-3-haiku": "Claude 3 Haiku - Velocissimo",
        "google/gemini-1.5-pro": "Gemini 1.5 Pro - Multimodale",
        "google/gemini-1.5-flash": "Gemini 1.5 Flash - Veloce",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("OPENROUTER_API_KEY"))
        self.default_model = os.getenv("OPENROUTER_DEFAULT_MODEL", self.DEFAULT_MODEL)
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "https://gideon.ai")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME", "GIDEON 3.0")
        # ⚡ TIMEOUT RIDOTTO per risposte più veloci
        self.client = httpx.AsyncClient(timeout=15.0)
        
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    @property
    def provider_name(self) -> str:
        return "OpenRouter"
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name
        }
    
    async def generate(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        temperature: float = 0.5,   # ⚡ Ridotto per risposte più dirette
        max_tokens: int = 300,       # ⚡ Ridotto per risposte concise e veloci
        **kwargs
    ) -> AIResponse:
        if not self.is_available:
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
            
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": model or self.default_model,
                    "messages": [m.to_dict() for m in messages],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"OpenRouter raw response keys: {data.keys()}")
                
                # Check for API error in response
                if "error" in data:
                    error_msg = data.get("error", {}).get("message", str(data["error"]))
                    logger.error(f"OpenRouter API returned error: {error_msg}")
                    return AIResponse(
                        content="",
                        provider=self.provider_name,
                        model=model or self.default_model,
                        finish_reason=f"api_error: {error_msg}"
                    )
                
                # Check for choices
                if "choices" not in data or not data["choices"]:
                    logger.error(f"OpenRouter response missing 'choices'. Full response: {data}")
                    return AIResponse(
                        content="",
                        provider=self.provider_name,
                        model=model or self.default_model,
                        finish_reason="error_no_choices"
                    )
                    
                choice = data["choices"][0]
                return AIResponse(
                    content=choice["message"]["content"],
                    provider=self.provider_name,
                    model=data.get("model", model or self.default_model),
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "complete"),
                    raw_response=data
                )
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return AIResponse(
                    content="",
                    provider=self.provider_name,
                    model=model or self.default_model,
                    finish_reason=f"error_{response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            return AIResponse(
                content="",
                provider=self.provider_name,
                model=model or self.default_model,
                finish_reason="error"
            )
    
    async def generate_with_images(
        self,
        prompt: str,
        image_urls: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """Generate response with image analysis"""
        content = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
        
        messages = [AIMessage(role="user", content="")]  # Placeholder
        
        # Build custom request for multimodal
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": model or "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": content}],
                    **kwargs
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                return AIResponse(
                    content=choice["message"]["content"],
                    provider=self.provider_name,
                    model=data.get("model", model),
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    finish_reason="complete",
                    raw_response=data
                )
            else:
                return AIResponse(
                    content="",
                    provider=self.provider_name,
                    model=model or self.default_model,
                    finish_reason="error"
                )
        except Exception as e:
            logger.error(f"OpenRouter multimodal failed: {e}")
            return AIResponse(content="", provider=self.provider_name, model=model or "", finish_reason="error")
    
    async def stream(
        self,
        messages: List[AIMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        if not self.is_available:
            yield ""
            return
            
        try:
            async with self.client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": model or self.default_model,
                    "messages": [m.to_dict() for m in messages],
                    "stream": True
                },
                timeout=60.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if delta := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                                yield delta
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"OpenRouter stream failed: {e}")
            yield ""


class AIProviderManager:
    """
    Manager for multiple AI providers with automatic fallback.
    Handles provider selection, fallback logic, and unified API.
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseAIProvider] = {}
        self.priority_order: List[str] = []
        self.default_provider: Optional[str] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize all available providers - FAST VERSION"""
        logger.info("🤖 Initializing AI Providers...")
        
        # Initialize all providers
        provider_classes = {
            "openrouter": (OpenRouterProvider, "OPENROUTER_API_KEY"),
            "openai": (OpenAIProvider, "OPENAI_API_KEY"),
            "anthropic": (AnthropicProvider, "ANTHROPIC_API_KEY"),
            "google": (GoogleProvider, "GOOGLE_API_KEY"),
            "groq": (GroqProvider, "GROQ_API_KEY"),
            "ollama": (OllamaProvider, None)  # Local, no key needed
        }
        
        for name, (cls, env_key) in provider_classes.items():
            # Skip providers without API key (except Ollama)
            if env_key and not os.getenv(env_key):
                logger.info(f"   ⏭️ {name} - Skipped (no API key)")
                continue
                
            try:
                provider = cls()
                self.providers[name] = provider
                
                # Quick check availability
                if name == "ollama":
                    available = await provider.check_available()
                else:
                    available = provider.is_available
                    
                if available:
                    self.priority_order.append(name)
                    logger.info(f"   ✅ {provider.provider_name} - Disponibile")
                else:
                    logger.info(f"   ⚠️ {provider.provider_name} - Non configurato")
                    
            except Exception as e:
                logger.error(f"   ❌ {name} initialization failed: {e}")
        
        # Set default provider (first available)
        if self.priority_order:
            self.default_provider = os.getenv("DEFAULT_AI_PROVIDER", self.priority_order[0])
            if self.default_provider not in self.priority_order:
                self.default_provider = self.priority_order[0]
            logger.info(f"   🎯 Provider predefinito: {self.default_provider}")
        else:
            logger.warning("   ⚠️ Nessun provider AI disponibile!")
            
        self._initialized = True
        logger.info(f"✅ AI Providers ready ({len(self.priority_order)} disponibili)")
        
    async def shutdown(self):
        """Close all provider connections"""
        for provider in self.providers.values():
            await provider.close()
            
    def get_provider(self, name: Optional[str] = None) -> Optional[BaseAIProvider]:
        """Get a specific provider or the default one"""
        name = name or self.default_provider
        return self.providers.get(name)
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available providers with their models"""
        result = []
        for name in self.priority_order:
            provider = self.providers[name]
            result.append({
                "id": name,
                "name": provider.provider_name,
                "models": getattr(provider, "MODELS", {}),
                "default_model": getattr(provider, "default_model", ""),
                "is_default": name == self.default_provider
            })
        return result
    
    def has_available_provider(self) -> bool:
        """Check if at least one AI provider is available"""
        return len(self.priority_order) > 0
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,  # Ridotto da 2000 a 1000 per risposte più veloci
        conversation_history: Optional[List[Dict[str, str]]] = None,
        fallback: bool = True,
        **kwargs
    ) -> AIResponse:
        """
        Generate a response using the specified or default provider.
        Supports automatic fallback to other providers.
        """
        if not self._initialized:
            await self.initialize()
            
        # Build messages
        messages = []
        
        # System prompt
        if system_prompt:
            messages.append(AIMessage(role="system", content=system_prompt))
        else:
            messages.append(AIMessage(
                role="system",
                content=self._get_gideon_system_prompt()
            ))
            
        # Conversation history - Limitato a 5 per velocità
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages (ridotto da 10)
                messages.append(AIMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
                
        # Current prompt
        messages.append(AIMessage(role="user", content=prompt))
        
        # Try providers
        providers_to_try = []
        if provider and provider in self.priority_order:
            providers_to_try.append(provider)
        if fallback:
            for p in self.priority_order:
                if p not in providers_to_try:
                    providers_to_try.append(p)
        elif not providers_to_try and self.default_provider:
            providers_to_try.append(self.default_provider)
            
        for provider_name in providers_to_try:
            ai_provider = self.providers.get(provider_name)
            if not ai_provider:
                continue
            
            # Se è OpenRouter, prova più modelli FREE in caso di fallimento
            if provider_name == "openrouter" and isinstance(ai_provider, OpenRouterProvider):
                models_to_try = [model] if model else []
                models_to_try.extend(ai_provider.FREE_MODELS[:5])  # Aggiungi top 5 FREE
                models_to_try = list(dict.fromkeys(models_to_try))  # Rimuovi duplicati
                
                for try_model in models_to_try:
                    if not try_model:
                        continue
                    logger.debug(f"🔄 Trying OpenRouter model: {try_model}")
                    response = await ai_provider.generate(
                        messages=messages,
                        model=try_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    
                    if response.success:
                        logger.info(f"✅ OpenRouter success with model: {try_model}")
                        return response
                    else:
                        logger.warning(f"⚠️ Model {try_model} failed, trying next...")
                        continue
            else:
                # Altri provider: prova una sola volta
                response = await ai_provider.generate(
                    messages=messages,
                    model=model if provider_name == (provider or self.default_provider) else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                if response.success:
                    logger.debug(f"AI response from {provider_name}: {len(response.content)} chars")
                    return response
                else:
                    logger.warning(f"Provider {provider_name} failed, trying next...")
                
        # All providers failed
        return AIResponse(
            content="Mi dispiace, non sono riuscito a elaborare la tua richiesta. "
                   "Nessun provider AI è disponibile al momento.",
            provider="none",
            model="",
            finish_reason="all_failed"
        )
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a response from an AI provider"""
        if not self._initialized:
            await self.initialize()
            
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append(AIMessage(role="system", content=system_prompt))
        else:
            messages.append(AIMessage(
                role="system",
                content=self._get_gideon_system_prompt()
            ))
            
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append(AIMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
                
        messages.append(AIMessage(role="user", content=prompt))
        
        # Get provider
        provider_name = provider or self.default_provider
        ai_provider = self.providers.get(provider_name)
        
        if ai_provider:
            async for chunk in ai_provider.stream(messages=messages, model=model, **kwargs):
                yield chunk
        else:
            yield "Nessun provider AI disponibile."
    
    def _get_gideon_system_prompt(self) -> str:
        """Get Gideon's default system prompt - Ottimizzato per velocità"""
        from datetime import datetime
        today = datetime.now()
        date_str = today.strftime("%d %B %Y")
        time_str = today.strftime("%H:%M")
        
        return f"""Sei G.I.D.E.O.N., assistente AI personale avanzato.

INFORMAZIONI TEMPORALI ATTUALI:
- Data odierna: {date_str}
- Ora corrente: {time_str}
- Anno corrente: {today.year}

Hai accesso a informazioni aggiornate fino ad oggi. Rispondi in italiano, in modo conciso ed efficiente. Sii amichevole ma professionale. Quando ti chiedono date, ore o eventi attuali, usa le informazioni sopra."""

    def has_any_provider(self) -> bool:
        """Check if any AI provider is available"""
        return len(self.priority_order) > 0
    
    async def generate_free(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        fast_mode: bool = False,
        **kwargs
    ) -> AIResponse:
        """
        Genera risposta usando SOLO modelli gratuiti con fallback automatico.
        
        Args:
            prompt: Il messaggio dell'utente
            system_prompt: Prompt di sistema opzionale
            fast_mode: Se True, usa modelli più piccoli e veloci
            
        Returns:
            AIResponse con la risposta generata
        """
        openrouter = self.providers.get("openrouter")
        if not openrouter or not isinstance(openrouter, OpenRouterProvider):
            return await self.generate(prompt, system_prompt=system_prompt, **kwargs)
        
        # Seleziona modelli in base alla modalità
        if fast_mode:
            models_to_try = [
                "meta-llama/llama-3.2-3b-instruct:free",    # Ultra veloce
                "meta-llama/llama-3.2-1b-instruct:free",    # Istantaneo
                "qwen/qwen-2-7b-instruct:free",
            ]
            max_tokens = kwargs.pop("max_tokens", 300)
        else:
            models_to_try = openrouter.FREE_MODELS[:5]  # Top 5 modelli gratuiti
            max_tokens = kwargs.pop("max_tokens", 800)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append(AIMessage(role="system", content=system_prompt))
        else:
            messages.append(AIMessage(role="system", content=self._get_gideon_system_prompt()))
        messages.append(AIMessage(role="user", content=prompt))
        
        # Prova i modelli gratuiti uno alla volta
        last_error = ""
        for model in models_to_try:
            try:
                logger.debug(f"🆓 Trying free model: {model}")
                response = await openrouter.generate(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    **kwargs
                )
                if response.success:
                    logger.info(f"✅ Free model success: {model}")
                    return response
                else:
                    last_error = f"Model {model} returned empty response"
            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️ Free model {model} failed: {e}")
                continue
        
        # Se tutti i modelli gratuiti falliscono, usa fallback normale
        logger.warning(f"⚠️ All free models failed, using fallback. Last error: {last_error}")
        return await self.generate(prompt, system_prompt=system_prompt, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        return {
            "initialized": self._initialized,
            "default_provider": self.default_provider,
            "available_count": len(self.priority_order),
            "providers": {
                name: {
                    "available": name in self.priority_order,
                    "name": self.providers[name].provider_name if name in self.providers else name
                }
                for name in ["openai", "anthropic", "google", "groq", "ollama"]
            }
        }


# Singleton instance
_ai_manager: Optional[AIProviderManager] = None


def get_ai_manager() -> AIProviderManager:
    """Get the global AI provider manager"""
    global _ai_manager
    if _ai_manager is None:
        _ai_manager = AIProviderManager()
    return _ai_manager


async def quick_ai_response(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> str:
    """Quick helper function to get an AI response"""
    manager = get_ai_manager()
    if not manager._initialized:
        await manager.initialize()
    response = await manager.generate(prompt, provider=provider, model=model, **kwargs)
    return response.content
