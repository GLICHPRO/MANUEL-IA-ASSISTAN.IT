"""
🌐 GIDEON 3.0 - OpenRouter Client

Client per OpenRouter API con supporto multimodale:
- Testo
- Immagini (URL e base64)
- Video (URL)
"""

import os
import base64
import asyncio
import aiohttp
import requests
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============ ENUMS ============

class ContentType(Enum):
    """Tipi di contenuto supportati"""
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"
    VIDEO_URL = "video_url"


class OpenRouterModel(Enum):
    """Modelli OpenRouter disponibili"""
    # Free models
    MOLMO_2_8B_FREE = "allenai/molmo-2-8b:free"
    LLAMA_3_8B_FREE = "meta-llama/llama-3-8b-instruct:free"
    GEMMA_7B_FREE = "google/gemma-7b-it:free"
    
    # Vision models
    GPT_4_VISION = "openai/gpt-4-vision-preview"
    GPT_4O = "openai/gpt-4o"
    GPT_4O_MINI = "openai/gpt-4o-mini"
    CLAUDE_3_SONNET = "anthropic/claude-3-sonnet"
    CLAUDE_3_OPUS = "anthropic/claude-3-opus"
    CLAUDE_3_HAIKU = "anthropic/claude-3-haiku"
    GEMINI_PRO_VISION = "google/gemini-pro-vision"
    GEMINI_1_5_PRO = "google/gemini-1.5-pro"
    LLAVA_13B = "liuhaotian/llava-13b"
    
    # Text models
    GPT_4_TURBO = "openai/gpt-4-turbo"
    GPT_3_5_TURBO = "openai/gpt-3.5-turbo"
    CLAUDE_3_5_SONNET = "anthropic/claude-3.5-sonnet"
    MISTRAL_7B = "mistralai/mistral-7b-instruct"
    MIXTRAL_8X7B = "mistralai/mixtral-8x7b-instruct"


# ============ DATA CLASSES ============

@dataclass
class ContentPart:
    """Parte di contenuto nel messaggio"""
    type: ContentType
    content: str  # Testo, URL, o base64
    
    def to_dict(self) -> dict:
        if self.type == ContentType.TEXT:
            return {
                "type": "text",
                "text": self.content
            }
        elif self.type == ContentType.IMAGE_URL:
            return {
                "type": "image_url",
                "image_url": {"url": self.content}
            }
        elif self.type == ContentType.IMAGE_BASE64:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self.content}"}
            }
        elif self.type == ContentType.VIDEO_URL:
            return {
                "type": "video_url",
                "video_url": {"url": self.content}
            }
        return {"type": "text", "text": self.content}


@dataclass
class Message:
    """Messaggio per la chat"""
    role: str  # "user", "assistant", "system"
    content: Union[str, List[ContentPart]]
    
    def to_dict(self) -> dict:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        else:
            return {
                "role": self.role,
                "content": [part.to_dict() for part in self.content]
            }


@dataclass
class OpenRouterResponse:
    """Risposta da OpenRouter"""
    success: bool
    content: str = ""
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "latency_ms": self.latency_ms
        }


@dataclass
class OpenRouterConfig:
    """Configurazione client"""
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = OpenRouterModel.GPT_4O_MINI.value
    site_url: str = "https://gideon.ai"
    site_name: str = "GIDEON 3.0"
    timeout: int = 60
    max_retries: int = 3
    
    # Defaults
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0


# ============ OPENROUTER CLIENT ============

class OpenRouterClient:
    """
    Client per OpenRouter API.
    
    Supporta:
    - Chat completions con testo
    - Analisi immagini (URL e file locali)
    - Analisi video (URL)
    - Streaming responses
    - Async e sync calls
    """
    
    def __init__(self, config: OpenRouterConfig = None, api_key: str = None):
        self.config = config or OpenRouterConfig()
        
        # Override API key if provided
        if api_key:
            self.config.api_key = api_key
        
        # Try to get from environment
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENROUTER_API_KEY", "")
        
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Stats
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "total_latency_ms": 0.0
        }
    
    @property
    def headers(self) -> dict:
        """Headers per le richieste"""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.site_url,
            "X-Title": self.config.site_name
        }
    
    @property
    def is_configured(self) -> bool:
        """Verifica se il client è configurato"""
        return bool(self.config.api_key)
    
    # === Sync Methods ===
    
    def chat(self, 
             prompt: str,
             model: str = None,
             system_prompt: str = None,
             temperature: float = None,
             max_tokens: int = None) -> OpenRouterResponse:
        """
        Chat semplice con testo.
        
        Args:
            prompt: Testo del prompt
            model: Modello da usare
            system_prompt: Prompt di sistema opzionale
            temperature: Temperatura
            max_tokens: Max tokens risposta
        
        Returns:
            OpenRouterResponse
        """
        messages = []
        
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        
        messages.append(Message(role="user", content=prompt))
        
        return self._complete_sync(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def analyze_image(self,
                      image: str,
                      prompt: str = "Descrivi questa immagine in dettaglio.",
                      model: str = None) -> OpenRouterResponse:
        """
        Analizza un'immagine.
        
        Args:
            image: URL dell'immagine o path locale
            prompt: Domanda/istruzione
            model: Modello vision
        
        Returns:
            OpenRouterResponse
        """
        content_parts = [ContentPart(type=ContentType.TEXT, content=prompt)]
        
        # Determina se è URL o file locale
        if image.startswith(("http://", "https://")):
            content_parts.append(ContentPart(type=ContentType.IMAGE_URL, content=image))
        else:
            # File locale - converti in base64
            base64_image = self._load_image_as_base64(image)
            if base64_image:
                content_parts.append(ContentPart(type=ContentType.IMAGE_BASE64, content=base64_image))
            else:
                return OpenRouterResponse(success=False, error=f"Cannot load image: {image}")
        
        messages = [Message(role="user", content=content_parts)]
        
        # Usa modello vision se non specificato
        model = model or OpenRouterModel.GPT_4O_MINI.value
        
        return self._complete_sync(messages=messages, model=model)
    
    def analyze_video(self,
                      video_url: str,
                      prompt: str = "Descrivi questo video.",
                      model: str = None) -> OpenRouterResponse:
        """
        Analizza un video.
        
        Args:
            video_url: URL del video
            prompt: Domanda/istruzione
            model: Modello con supporto video
        
        Returns:
            OpenRouterResponse
        """
        content_parts = [
            ContentPart(type=ContentType.TEXT, content=prompt),
            ContentPart(type=ContentType.VIDEO_URL, content=video_url)
        ]
        
        messages = [Message(role="user", content=content_parts)]
        
        # Usa modello con supporto video
        model = model or OpenRouterModel.MOLMO_2_8B_FREE.value
        
        return self._complete_sync(messages=messages, model=model)
    
    def analyze_multimodal(self,
                           prompt: str,
                           images: List[str] = None,
                           videos: List[str] = None,
                           model: str = None) -> OpenRouterResponse:
        """
        Analisi multimodale con testo, immagini e video.
        
        Args:
            prompt: Testo/domanda
            images: Lista URL/path immagini
            videos: Lista URL video
            model: Modello da usare
        
        Returns:
            OpenRouterResponse
        """
        content_parts = [ContentPart(type=ContentType.TEXT, content=prompt)]
        
        # Aggiungi immagini
        if images:
            for img in images:
                if img.startswith(("http://", "https://")):
                    content_parts.append(ContentPart(type=ContentType.IMAGE_URL, content=img))
                else:
                    base64_img = self._load_image_as_base64(img)
                    if base64_img:
                        content_parts.append(ContentPart(type=ContentType.IMAGE_BASE64, content=base64_img))
        
        # Aggiungi video
        if videos:
            for vid in videos:
                content_parts.append(ContentPart(type=ContentType.VIDEO_URL, content=vid))
        
        messages = [Message(role="user", content=content_parts)]
        
        return self._complete_sync(messages=messages, model=model)
    
    def _complete_sync(self,
                       messages: List[Message],
                       model: str = None,
                       temperature: float = None,
                       max_tokens: int = None) -> OpenRouterResponse:
        """Esegue richiesta sincrona"""
        start_time = datetime.now()
        self._stats["total_requests"] += 1
        
        try:
            payload = {
                "model": model or self.config.default_model,
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "top_p": self.config.top_p
            }
            
            response = requests.post(
                url=f"{self.config.base_url}/chat/completions",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=self.config.timeout
            )
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self._stats["total_latency_ms"] += latency
            
            if response.status_code == 200:
                data = response.json()
                
                # Estrai risposta
                content = ""
                if data.get("choices") and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                
                usage = data.get("usage", {})
                self._stats["total_tokens_used"] += usage.get("total_tokens", 0)
                self._stats["successful_requests"] += 1
                
                return OpenRouterResponse(
                    success=True,
                    content=content,
                    model=data.get("model", ""),
                    usage=usage,
                    finish_reason=data["choices"][0].get("finish_reason", "") if data.get("choices") else "",
                    raw_response=data,
                    latency_ms=latency
                )
            else:
                self._stats["failed_requests"] += 1
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"OpenRouter error: {error_msg}")
                
                return OpenRouterResponse(
                    success=False,
                    error=error_msg,
                    latency_ms=latency
                )
                
        except Exception as e:
            self._stats["failed_requests"] += 1
            latency = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"OpenRouter exception: {e}")
            
            return OpenRouterResponse(
                success=False,
                error=str(e),
                latency_ms=latency
            )
    
    # === Async Methods ===
    
    async def chat_async(self,
                         prompt: str,
                         model: str = None,
                         system_prompt: str = None,
                         temperature: float = None,
                         max_tokens: int = None) -> OpenRouterResponse:
        """Chat asincrona"""
        messages = []
        
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        
        messages.append(Message(role="user", content=prompt))
        
        return await self._complete_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def analyze_image_async(self,
                                  image: str,
                                  prompt: str = "Descrivi questa immagine in dettaglio.",
                                  model: str = None) -> OpenRouterResponse:
        """Analizza immagine in modo asincrono"""
        content_parts = [ContentPart(type=ContentType.TEXT, content=prompt)]
        
        if image.startswith(("http://", "https://")):
            content_parts.append(ContentPart(type=ContentType.IMAGE_URL, content=image))
        else:
            base64_image = self._load_image_as_base64(image)
            if base64_image:
                content_parts.append(ContentPart(type=ContentType.IMAGE_BASE64, content=base64_image))
            else:
                return OpenRouterResponse(success=False, error=f"Cannot load image: {image}")
        
        messages = [Message(role="user", content=content_parts)]
        model = model or OpenRouterModel.GPT_4O_MINI.value
        
        return await self._complete_async(messages=messages, model=model)
    
    async def analyze_video_async(self,
                                  video_url: str,
                                  prompt: str = "Descrivi questo video.",
                                  model: str = None) -> OpenRouterResponse:
        """Analizza video in modo asincrono"""
        content_parts = [
            ContentPart(type=ContentType.TEXT, content=prompt),
            ContentPart(type=ContentType.VIDEO_URL, content=video_url)
        ]
        
        messages = [Message(role="user", content=content_parts)]
        model = model or OpenRouterModel.MOLMO_2_8B_FREE.value
        
        return await self._complete_async(messages=messages, model=model)
    
    async def _complete_async(self,
                              messages: List[Message],
                              model: str = None,
                              temperature: float = None,
                              max_tokens: int = None) -> OpenRouterResponse:
        """Esegue richiesta asincrona"""
        start_time = datetime.now()
        self._stats["total_requests"] += 1
        
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            payload = {
                "model": model or self.config.default_model,
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "top_p": self.config.top_p
            }
            
            async with self._session.post(
                url=f"{self.config.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self._stats["total_latency_ms"] += latency
                
                if response.status == 200:
                    data = await response.json()
                    
                    content = ""
                    if data.get("choices") and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                    
                    usage = data.get("usage", {})
                    self._stats["total_tokens_used"] += usage.get("total_tokens", 0)
                    self._stats["successful_requests"] += 1
                    
                    return OpenRouterResponse(
                        success=True,
                        content=content,
                        model=data.get("model", ""),
                        usage=usage,
                        finish_reason=data["choices"][0].get("finish_reason", "") if data.get("choices") else "",
                        raw_response=data,
                        latency_ms=latency
                    )
                else:
                    self._stats["failed_requests"] += 1
                    text = await response.text()
                    error_msg = f"HTTP {response.status}: {text}"
                    logger.error(f"OpenRouter error: {error_msg}")
                    
                    return OpenRouterResponse(
                        success=False,
                        error=error_msg,
                        latency_ms=latency
                    )
                    
        except Exception as e:
            self._stats["failed_requests"] += 1
            latency = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"OpenRouter exception: {e}")
            
            return OpenRouterResponse(
                success=False,
                error=str(e),
                latency_ms=latency
            )
    
    async def close(self):
        """Chiude la sessione asincrona"""
        if self._session:
            await self._session.close()
            self._session = None
    
    # === Utilities ===
    
    def _load_image_as_base64(self, path: str) -> Optional[str]:
        """Carica immagine locale come base64"""
        try:
            file_path = Path(path)
            if file_path.exists():
                with open(file_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
        return None
    
    def get_available_models(self) -> List[str]:
        """Lista modelli disponibili"""
        return [m.value for m in OpenRouterModel]
    
    def get_vision_models(self) -> List[str]:
        """Lista modelli con supporto vision"""
        return [
            OpenRouterModel.MOLMO_2_8B_FREE.value,
            OpenRouterModel.GPT_4_VISION.value,
            OpenRouterModel.GPT_4O.value,
            OpenRouterModel.GPT_4O_MINI.value,
            OpenRouterModel.CLAUDE_3_SONNET.value,
            OpenRouterModel.CLAUDE_3_OPUS.value,
            OpenRouterModel.CLAUDE_3_HAIKU.value,
            OpenRouterModel.GEMINI_PRO_VISION.value,
            OpenRouterModel.GEMINI_1_5_PRO.value,
            OpenRouterModel.LLAVA_13B.value
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche utilizzo"""
        return {
            **self._stats,
            "avg_latency_ms": self._stats["total_latency_ms"] / max(1, self._stats["total_requests"]),
            "success_rate": self._stats["successful_requests"] / max(1, self._stats["total_requests"])
        }
    
    def set_api_key(self, api_key: str):
        """Imposta API key"""
        self.config.api_key = api_key
    
    def set_default_model(self, model: str):
        """Imposta modello default"""
        self.config.default_model = model


# ============ FACTORY ============

def create_openrouter_client(api_key: str = None) -> OpenRouterClient:
    """Crea client OpenRouter"""
    return OpenRouterClient(api_key=api_key)


# ============ CONVENIENCE FUNCTIONS ============

# Singleton globale
_default_client: Optional[OpenRouterClient] = None


def get_client() -> OpenRouterClient:
    """Ottiene client singleton"""
    global _default_client
    if _default_client is None:
        _default_client = OpenRouterClient()
    return _default_client


def quick_chat(prompt: str, model: str = None) -> str:
    """Chat veloce - ritorna solo il testo"""
    response = get_client().chat(prompt, model=model)
    return response.content if response.success else f"Error: {response.error}"


def quick_analyze_image(image: str, prompt: str = None) -> str:
    """Analisi immagine veloce"""
    response = get_client().analyze_image(
        image, 
        prompt=prompt or "Descrivi questa immagine in dettaglio."
    )
    return response.content if response.success else f"Error: {response.error}"


async def quick_chat_async(prompt: str, model: str = None) -> str:
    """Chat veloce asincrona"""
    response = await get_client().chat_async(prompt, model=model)
    return response.content if response.success else f"Error: {response.error}"
