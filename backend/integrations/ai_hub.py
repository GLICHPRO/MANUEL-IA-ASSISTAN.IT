"""
🚀 GIDEON AI HUB - Centro di Controllo per tutte le AI

Integra tutti i servizi AI in un unico punto di accesso:
- 🎤 VOICE: ElevenLabs, Edge TTS, OpenAI TTS
- 💬 CHAT: ChatGPT, Claude, Gemini, Qwen, Llama
- 🎨 IMAGES: DALL-E, Midjourney, Stable Diffusion, Flux
- 🎬 VIDEO: HeyGen, Kling AI, Runway, Pika
- 🎵 MUSIC: Suno AI, Udio
- 📊 PRESENTATIONS: Gamma AI
- 🎙️ PODCAST: NotebookLM style
- ⚙️ AUTOMATION: n8n style workflows
- 🤖 AGENTS: Auto-GPT style autonomous agents

Architettura:
┌─────────────────────────────────────────────────────────────────────┐
│                        GIDEON AI HUB                                 │
│              "Un Solo Assistente, Infinite Possibilità"              │
├─────────────────────────────────────────────────────────────────────┤
│  🎤 VOICE      │  🎨 CREATIVE    │  ⚙️ AUTOMATION   │  🤖 AGENTS    │
│  - ElevenLabs  │  - DALL-E 3     │  - Workflows     │  - Task Agent │
│  - Edge TTS    │  - Midjourney   │  - Triggers      │  - Research   │
│  - OpenAI TTS  │  - Kling AI     │  - Schedules     │  - Code Agent │
│  - Clone Voice │  - HeyGen       │  - Webhooks      │  - Web Agent  │
├─────────────────────────────────────────────────────────────────────┤
│  💬 CHAT       │  🎬 VIDEO       │  🎵 AUDIO        │  📊 DOCS      │
│  - GPT-4       │  - HeyGen       │  - Suno AI       │  - Gamma AI   │
│  - Claude      │  - Runway       │  - Podcast Gen   │  - Slides     │
│  - Gemini      │  - Pika Labs    │  - Music Gen     │  - Reports    │
└─────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import aiohttp
import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger


class AIServiceType(Enum):
    """Tipi di servizi AI"""
    VOICE = "voice"
    CHAT = "chat"
    IMAGE = "image"
    VIDEO = "video"
    MUSIC = "music"
    PRESENTATION = "presentation"
    PODCAST = "podcast"
    AUTOMATION = "automation"
    AGENT = "agent"


@dataclass
class AIService:
    """Configurazione di un servizio AI"""
    name: str
    type: AIServiceType
    api_key_env: str
    base_url: str
    enabled: bool = True
    free_tier: bool = False
    rate_limit: int = 60  # requests per minute
    models: List[str] = field(default_factory=list)
    

class GideonAIHub:
    """
    🚀 GIDEON AI HUB
    
    Centro di controllo per tutte le integrazioni AI.
    Un unico punto di accesso a decine di servizi AI.
    """
    
    def __init__(self):
        self.services: Dict[str, AIService] = {}
        self.api_keys: Dict[str, str] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_dir = Path(__file__).parent.parent / "cache" / "ai_hub"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistiche
        self.stats = {
            "total_requests": 0,
            "by_service": {},
            "errors": 0,
            "started_at": datetime.now()
        }
        
        # Registra servizi
        self._register_services()
        self._load_api_keys()
    
    def _register_services(self):
        """Registra tutti i servizi AI supportati"""
        
        # ============ VOICE SERVICES ============
        self.services["elevenlabs"] = AIService(
            name="ElevenLabs",
            type=AIServiceType.VOICE,
            api_key_env="ELEVENLABS_API_KEY",
            base_url="https://api.elevenlabs.io/v1",
            models=["eleven_multilingual_v2", "eleven_turbo_v2"]
        )
        
        self.services["openai_tts"] = AIService(
            name="OpenAI TTS",
            type=AIServiceType.VOICE,
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
            models=["tts-1", "tts-1-hd"]
        )
        
        # ============ CHAT/LLM SERVICES ============
        self.services["openai"] = AIService(
            name="OpenAI (ChatGPT)",
            type=AIServiceType.CHAT,
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
            models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        )
        
        self.services["anthropic"] = AIService(
            name="Anthropic (Claude)",
            type=AIServiceType.CHAT,
            api_key_env="ANTHROPIC_API_KEY",
            base_url="https://api.anthropic.com/v1",
            models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        )
        
        self.services["google"] = AIService(
            name="Google (Gemini)",
            type=AIServiceType.CHAT,
            api_key_env="GOOGLE_API_KEY",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            models=["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro"]
        )
        
        self.services["openrouter"] = AIService(
            name="OpenRouter (Multi-Model)",
            type=AIServiceType.CHAT,
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            free_tier=True,
            models=[
                "qwen/qwen3-4b:free",
                "google/gemini-2.0-flash-exp:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "anthropic/claude-3-haiku"
            ]
        )
        
        # ============ IMAGE SERVICES ============
        self.services["dalle"] = AIService(
            name="DALL-E 3",
            type=AIServiceType.IMAGE,
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
            models=["dall-e-3", "dall-e-2"]
        )
        
        self.services["stability"] = AIService(
            name="Stability AI (Stable Diffusion)",
            type=AIServiceType.IMAGE,
            api_key_env="STABILITY_API_KEY",
            base_url="https://api.stability.ai/v1",
            models=["stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6"]
        )
        
        self.services["replicate"] = AIService(
            name="Replicate (Flux, SDXL)",
            type=AIServiceType.IMAGE,
            api_key_env="REPLICATE_API_KEY",
            base_url="https://api.replicate.com/v1",
            models=["flux-schnell", "flux-dev", "sdxl"]
        )
        
        # ============ VIDEO SERVICES ============
        self.services["heygen"] = AIService(
            name="HeyGen (AI Avatars)",
            type=AIServiceType.VIDEO,
            api_key_env="HEYGEN_API_KEY",
            base_url="https://api.heygen.com/v2",
            models=["avatar_video", "talking_photo"]
        )
        
        self.services["runway"] = AIService(
            name="Runway (Gen-3)",
            type=AIServiceType.VIDEO,
            api_key_env="RUNWAY_API_KEY",
            base_url="https://api.runwayml.com/v1",
            models=["gen-3-alpha", "gen-2"]
        )
        
        self.services["kling"] = AIService(
            name="Kling AI",
            type=AIServiceType.VIDEO,
            api_key_env="KLING_API_KEY",
            base_url="https://api.kling.ai/v1",
            models=["kling-video", "kling-image"]
        )
        
        # Google Veo 3 / Veo 3.1 - Video Generation
        self.services["veo"] = AIService(
            name="Google Veo 3.1 (Video AI)",
            type=AIServiceType.VIDEO,
            api_key_env="GOOGLE_API_KEY",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            models=["veo-3.1-generate-preview", "veo-3.1-fast-preview", "veo-2"]
        )
        
        # OpenAI Sora 2 - Video Generation
        self.services["sora"] = AIService(
            name="OpenAI Sora 2 (Video AI)",
            type=AIServiceType.VIDEO,
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
            models=["sora-2", "sora-turbo"]
        )
        
        # ByteDance Seedream 3.0 - Image Generation (via Replicate)
        self.services["seedream"] = AIService(
            name="Seedream 3.0 (Image AI)",
            type=AIServiceType.IMAGE,
            api_key_env="REPLICATE_API_KEY",
            base_url="https://api.replicate.com/v1",
            models=["bytedance/seedream-3.0", "seedream-3.0-turbo"]
        )
        
        # SeedVR - VR/360 Video Generation (via Replicate)
        self.services["seedvr"] = AIService(
            name="SeedVR (VR/360 Video AI)",
            type=AIServiceType.VIDEO,
            api_key_env="REPLICATE_API_KEY",
            base_url="https://api.replicate.com/v1",
            models=["seedvr-360", "seedvr-vr"]
        )
        
        # Luma AI - Dream Machine for video
        self.services["luma"] = AIService(
            name="Luma AI (Dream Machine)",
            type=AIServiceType.VIDEO,
            api_key_env="LUMA_API_KEY",
            base_url="https://api.lumalabs.ai/v1",
            models=["dream-machine", "luma-video"]
        )
        
        # Pika Labs - Video Generation
        self.services["pika"] = AIService(
            name="Pika Labs (Video AI)",
            type=AIServiceType.VIDEO,
            api_key_env="PIKA_API_KEY",
            base_url="https://api.pika.art/v1",
            models=["pika-1.0", "pika-turbo"]
        )
        
        # ============ MUSIC SERVICES ============
        self.services["suno"] = AIService(
            name="Suno AI (Music)",
            type=AIServiceType.MUSIC,
            api_key_env="SUNO_API_KEY",
            base_url="https://api.suno.ai/v1",
            models=["chirp-v3", "bark"]
        )
        
        # Udio - Music Generation
        self.services["udio"] = AIService(
            name="Udio (Music AI)",
            type=AIServiceType.MUSIC,
            api_key_env="UDIO_API_KEY",
            base_url="https://api.udio.com/v1",
            models=["udio-v1", "udio-extended"]
        )
        
        # ============ PRESENTATION SERVICES ============
        self.services["gamma"] = AIService(
            name="Gamma AI (Presentations)",
            type=AIServiceType.PRESENTATION,
            api_key_env="GAMMA_API_KEY",
            base_url="https://api.gamma.app/v1",
            models=["presentation", "document"]
        )
        
        logger.info(f"📦 Registrati {len(self.services)} servizi AI")
    
    def _load_api_keys(self):
        """Carica le API key dalle variabili d'ambiente"""
        for service_id, service in self.services.items():
            key = os.getenv(service.api_key_env)
            if key:
                self.api_keys[service_id] = key
                logger.info(f"   ✅ {service.name}: API key configurata")
            else:
                logger.debug(f"   ⚠️ {service.name}: API key non configurata")
    
    async def initialize(self):
        """Inizializza la sessione HTTP"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        logger.info("🚀 GIDEON AI Hub inizializzato")
    
    async def close(self):
        """Chiude la sessione"""
        if self.session:
            await self.session.close()
            self.session = None
    
    # ============================================================
    #                      VOICE SERVICES
    # ============================================================
    
    async def text_to_speech(
        self,
        text: str,
        provider: str = "auto",
        voice: str = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        🎤 Text-to-Speech con provider multipli
        
        Providers: elevenlabs, openai_tts, edge_tts
        """
        # Auto-select provider
        if provider == "auto":
            if "elevenlabs" in self.api_keys:
                provider = "elevenlabs"
            elif "openai" in self.api_keys:
                provider = "openai_tts"
            else:
                provider = "edge_tts"
        
        if provider == "elevenlabs":
            return await self._elevenlabs_tts(text, voice, model)
        elif provider == "openai_tts":
            return await self._openai_tts(text, voice, model)
        else:
            return await self._edge_tts(text, voice)
    
    async def _elevenlabs_tts(self, text: str, voice: str = None, model: str = None) -> Dict:
        """ElevenLabs TTS"""
        if "elevenlabs" not in self.api_keys:
            return {"success": False, "error": "ElevenLabs API key non configurata"}
        
        voice_id = voice or "21m00Tcm4TlvDq8ikWAM"  # Rachel default
        model_id = model or "eleven_multilingual_v2"
        
        headers = {
            "xi-api-key": self.api_keys["elevenlabs"],
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        try:
            async with self.session.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    audio_base64 = base64.b64encode(audio_data).decode()
                    return {
                        "success": True,
                        "audio_base64": audio_base64,
                        "provider": "elevenlabs",
                        "format": "mp3"
                    }
                else:
                    error = await response.text()
                    return {"success": False, "error": error}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _openai_tts(self, text: str, voice: str = None, model: str = None) -> Dict:
        """OpenAI TTS"""
        if "openai" not in self.api_keys:
            return {"success": False, "error": "OpenAI API key non configurata"}
        
        voice = voice or "alloy"
        model = model or "tts-1"
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['openai']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "input": text,
            "voice": voice
        }
        
        try:
            async with self.session.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    audio_base64 = base64.b64encode(audio_data).decode()
                    return {
                        "success": True,
                        "audio_base64": audio_base64,
                        "provider": "openai",
                        "format": "mp3"
                    }
                else:
                    error = await response.text()
                    return {"success": False, "error": error}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _edge_tts(self, text: str, voice: str = None) -> Dict:
        """Microsoft Edge TTS (gratuito)"""
        try:
            import edge_tts
            
            voice = voice or "it-IT-GiuseppeNeural"
            communicate = edge_tts.Communicate(text, voice, rate="+20%")
            
            # Genera audio
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if audio_data:
                audio_base64 = base64.b64encode(audio_data).decode()
                return {
                    "success": True,
                    "audio_base64": audio_base64,
                    "provider": "edge_tts",
                    "format": "mp3"
                }
            return {"success": False, "error": "Nessun audio generato"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ============================================================
    #                      CHAT/LLM SERVICES
    # ============================================================
    
    async def chat(
        self,
        message: str,
        provider: str = "auto",
        model: str = None,
        system_prompt: str = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        💬 Chat con LLM multipli
        
        Providers: openai, anthropic, google, openrouter
        """
        if provider == "auto":
            # Priorità: OpenRouter (free) > OpenAI > Anthropic > Google
            if "openrouter" in self.api_keys:
                provider = "openrouter"
            elif "openai" in self.api_keys:
                provider = "openai"
            elif "anthropic" in self.api_keys:
                provider = "anthropic"
            else:
                return {"success": False, "error": "Nessun provider chat configurato"}
        
        self.stats["total_requests"] += 1
        self.stats["by_service"][provider] = self.stats["by_service"].get(provider, 0) + 1
        
        if provider == "openrouter":
            return await self._openrouter_chat(message, model, system_prompt, max_tokens, temperature)
        elif provider == "openai":
            return await self._openai_chat(message, model, system_prompt, max_tokens, temperature)
        elif provider == "anthropic":
            return await self._anthropic_chat(message, model, system_prompt, max_tokens, temperature)
        else:
            return {"success": False, "error": f"Provider {provider} non supportato"}
    
    async def _openrouter_chat(self, message: str, model: str, system: str, max_tokens: int, temp: float) -> Dict:
        """OpenRouter Chat (supporta molti modelli, alcuni gratuiti)"""
        model = model or "qwen/qwen3-4b:free"
        
        headers = {
            "Authorization": f"Bearer {self.api_keys.get('openrouter', '')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://gideon.ai",
            "X-Title": "GIDEON AI"
        }
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp
        }
        
        try:
            async with self.session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                data = await response.json()
                if "choices" in data:
                    return {
                        "success": True,
                        "response": data["choices"][0]["message"]["content"],
                        "model": model,
                        "provider": "openrouter",
                        "usage": data.get("usage", {})
                    }
                return {"success": False, "error": data.get("error", "Unknown error")}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _openai_chat(self, message: str, model: str, system: str, max_tokens: int, temp: float) -> Dict:
        """OpenAI Chat"""
        model = model or "gpt-4o-mini"
        
        headers = {
            "Authorization": f"Bearer {self.api_keys.get('openai', '')}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp
        }
        
        try:
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                if "choices" in data:
                    return {
                        "success": True,
                        "response": data["choices"][0]["message"]["content"],
                        "model": model,
                        "provider": "openai"
                    }
                return {"success": False, "error": data.get("error", {}).get("message", "Error")}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _anthropic_chat(self, message: str, model: str, system: str, max_tokens: int, temp: float) -> Dict:
        """Anthropic Claude Chat"""
        model = model or "claude-3-haiku-20240307"
        
        headers = {
            "x-api-key": self.api_keys.get("anthropic", ""),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": message}]
        }
        if system:
            payload["system"] = system
        
        try:
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                if "content" in data:
                    return {
                        "success": True,
                        "response": data["content"][0]["text"],
                        "model": model,
                        "provider": "anthropic"
                    }
                return {"success": False, "error": data.get("error", {}).get("message", "Error")}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ============================================================
    #                      IMAGE GENERATION
    # ============================================================
    
    async def generate_image(
        self,
        prompt: str,
        provider: str = "auto",
        model: str = None,
        size: str = "1024x1024",
        style: str = None
    ) -> Dict[str, Any]:
        """
        🎨 Genera immagini con AI
        
        Providers: dalle, stability, replicate
        """
        if provider == "auto":
            if "openai" in self.api_keys:
                provider = "dalle"
            elif "stability" in self.api_keys:
                provider = "stability"
            elif "replicate" in self.api_keys:
                provider = "replicate"
            else:
                return {"success": False, "error": "Nessun provider immagini configurato"}
        
        if provider == "dalle":
            return await self._dalle_generate(prompt, model, size, style)
        elif provider == "stability":
            return await self._stability_generate(prompt, model, size)
        elif provider == "replicate":
            return await self._replicate_generate(prompt, model)
        
        return {"success": False, "error": f"Provider {provider} non supportato"}
    
    async def _dalle_generate(self, prompt: str, model: str, size: str, style: str) -> Dict:
        """DALL-E Image Generation"""
        model = model or "dall-e-3"
        
        headers = {
            "Authorization": f"Bearer {self.api_keys.get('openai', '')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json"
        }
        if style:
            payload["style"] = style
        
        try:
            async with self.session.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                if "data" in data:
                    return {
                        "success": True,
                        "image_base64": data["data"][0]["b64_json"],
                        "revised_prompt": data["data"][0].get("revised_prompt"),
                        "provider": "dalle",
                        "model": model
                    }
                return {"success": False, "error": data.get("error", {}).get("message", "Error")}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _stability_generate(self, prompt: str, model: str, size: str) -> Dict:
        """Stability AI Image Generation"""
        # Implementazione Stability AI
        return {"success": False, "error": "Stability AI coming soon"}
    
    async def _replicate_generate(self, prompt: str, model: str) -> Dict:
        """Replicate Image Generation (Flux, SDXL)"""
        # Implementazione Replicate
        return {"success": False, "error": "Replicate coming soon"}
    
    # ============================================================
    #                      VIDEO GENERATION
    # ============================================================
    
    async def generate_video(
        self,
        prompt: str = None,
        image: str = None,
        avatar_id: str = None,
        script: str = None,
        provider: str = "auto"
    ) -> Dict[str, Any]:
        """
        🎬 Genera video con AI
        
        Providers: heygen (avatar), runway (gen-3), kling
        """
        if provider == "auto":
            if avatar_id or script:
                provider = "heygen"
            elif "runway" in self.api_keys:
                provider = "runway"
            else:
                provider = "kling"
        
        if provider == "heygen":
            return await self._heygen_generate(avatar_id, script)
        elif provider == "runway":
            return await self._runway_generate(prompt, image)
        elif provider == "kling":
            return await self._kling_generate(prompt, image)
        
        return {"success": False, "error": f"Provider {provider} non supportato"}
    
    async def _heygen_generate(self, avatar_id: str, script: str) -> Dict:
        """HeyGen Avatar Video"""
        if "heygen" not in self.api_keys:
            return {"success": False, "error": "HeyGen API key non configurata"}
        
        headers = {
            "X-Api-Key": self.api_keys["heygen"],
            "Content-Type": "application/json"
        }
        
        payload = {
            "video_inputs": [{
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id or "default"
                },
                "voice": {
                    "type": "text",
                    "input_text": script
                }
            }],
            "dimension": {"width": 1280, "height": 720}
        }
        
        try:
            async with self.session.post(
                "https://api.heygen.com/v2/video/generate",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                if data.get("data", {}).get("video_id"):
                    return {
                        "success": True,
                        "video_id": data["data"]["video_id"],
                        "status": "processing",
                        "provider": "heygen"
                    }
                return {"success": False, "error": data.get("message", "Error")}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _runway_generate(self, prompt: str, image: str) -> Dict:
        """Runway Gen-3 Video"""
        return {"success": False, "error": "Runway coming soon"}
    
    async def _kling_generate(self, prompt: str, image: str) -> Dict:
        """Kling AI Video"""
        return {"success": False, "error": "Kling AI coming soon"}
    
    # ============================================================
    #                      NEW VIDEO AI SERVICES
    # ============================================================
    
    async def generate_video_veo(
        self,
        prompt: str,
        image: str = None,
        model: str = "veo-3.1-generate-preview",
        aspect_ratio: str = "16:9",
        resolution: str = "720p"
    ) -> Dict[str, Any]:
        """
        📹 Google Veo 3.1 Video Generation
        
        Uses Google's Gemini API for video generation
        Models: veo-3.1-generate-preview, veo-3.1-fast-preview, veo-2
        """
        if "google" not in self.api_keys:
            return {"success": False, "error": "Google API key non configurata (GOOGLE_API_KEY)"}
        
        try:
            # Using Google Generative AI SDK approach
            headers = {
                "Content-Type": "application/json"
            }
            
            # Veo 3.1 API endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateVideo?key={self.api_keys['google']}"
            
            payload = {
                "prompt": prompt,
                "config": {
                    "aspectRatio": aspect_ratio,
                    "resolution": resolution
                }
            }
            
            if image:
                payload["image"] = {"imageUri": image}
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                data = await response.json()
                
                if response.status == 200:
                    # Veo returns an operation that needs polling
                    operation_name = data.get("name")
                    return {
                        "success": True,
                        "operation_id": operation_name,
                        "status": "processing",
                        "provider": "veo",
                        "model": model,
                        "message": "Video generation started. Poll for completion."
                    }
                else:
                    return {
                        "success": False,
                        "error": data.get("error", {}).get("message", "Veo API error"),
                        "status_code": response.status
                    }
                    
        except Exception as e:
            return {"success": False, "error": f"Veo error: {str(e)}"}
    
    async def generate_video_sora(
        self,
        prompt: str,
        duration: int = 5,
        model: str = "sora-2",
        aspect_ratio: str = "16:9"
    ) -> Dict[str, Any]:
        """
        🎥 OpenAI Sora 2 Video Generation
        
        Note: Sora API is currently in limited access
        """
        if "openai" not in self.api_keys:
            return {"success": False, "error": "OpenAI API key non configurata"}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_keys['openai']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio
            }
            
            # Note: This endpoint may not be publicly available yet
            async with self.session.post(
                "https://api.openai.com/v1/videos/generations",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                if response.status == 200:
                    return {
                        "success": True,
                        "video_url": data.get("url"),
                        "status": "completed",
                        "provider": "sora",
                        "model": model
                    }
                else:
                    return {
                        "success": False,
                        "error": data.get("error", {}).get("message", "Sora API not available yet"),
                        "suggestion": "Sora API potrebbe non essere ancora pubblica. Prova Veo 3.1 o Kling."
                    }
                    
        except Exception as e:
            return {"success": False, "error": f"Sora error: {str(e)}"}
    
    async def generate_image_seedream(
        self,
        prompt: str,
        model: str = "bytedance/seedream-3.0",
        aspect_ratio: str = "1:1",
        num_outputs: int = 1
    ) -> Dict[str, Any]:
        """
        🌸 ByteDance Seedream 3.0 Image Generation (via Replicate)
        """
        if "replicate" not in self.api_keys:
            return {"success": False, "error": "Replicate API key non configurata (REPLICATE_API_KEY)"}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_keys['replicate']}",
                "Content-Type": "application/json"
            }
            
            # Get model version first (using Replicate pattern)
            payload = {
                "version": model,
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "num_outputs": num_outputs
                }
            }
            
            async with self.session.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                if response.status in [200, 201]:
                    return {
                        "success": True,
                        "prediction_id": data.get("id"),
                        "status": data.get("status", "starting"),
                        "provider": "seedream",
                        "model": model,
                        "urls": {
                            "get": data.get("urls", {}).get("get"),
                            "cancel": data.get("urls", {}).get("cancel")
                        }
                    }
                else:
                    return {"success": False, "error": data.get("detail", "Replicate API error")}
                    
        except Exception as e:
            return {"success": False, "error": f"Seedream error: {str(e)}"}
    
    async def generate_video_seedvr(
        self,
        prompt: str,
        model: str = "seedvr-360",
        resolution: str = "4k"
    ) -> Dict[str, Any]:
        """
        🌐 SeedVR 360°/VR Video Generation (via Replicate)
        """
        if "replicate" not in self.api_keys:
            return {"success": False, "error": "Replicate API key non configurata"}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_keys['replicate']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "version": model,
                "input": {
                    "prompt": prompt,
                    "resolution": resolution,
                    "format": "360"
                }
            }
            
            async with self.session.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                if response.status in [200, 201]:
                    return {
                        "success": True,
                        "prediction_id": data.get("id"),
                        "status": data.get("status", "starting"),
                        "provider": "seedvr",
                        "model": model
                    }
                else:
                    return {"success": False, "error": data.get("detail", "SeedVR API error")}
                    
        except Exception as e:
            return {"success": False, "error": f"SeedVR error: {str(e)}"}
    
    async def generate_video_luma(
        self,
        prompt: str,
        image: str = None,
        model: str = "dream-machine"
    ) -> Dict[str, Any]:
        """
        ✨ Luma AI Dream Machine Video Generation
        """
        if "luma" not in self.api_keys:
            return {"success": False, "error": "Luma API key non configurata (LUMA_API_KEY)"}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_keys['luma']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "model": model
            }
            
            if image:
                payload["image_url"] = image
            
            async with self.session.post(
                "https://api.lumalabs.ai/dream-machine/v1/generations",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                if response.status in [200, 201]:
                    return {
                        "success": True,
                        "generation_id": data.get("id"),
                        "status": data.get("state", "pending"),
                        "provider": "luma",
                        "model": model
                    }
                else:
                    return {"success": False, "error": data.get("message", "Luma API error")}
                    
        except Exception as e:
            return {"success": False, "error": f"Luma error: {str(e)}"}
    
    async def generate_video_pika(
        self,
        prompt: str,
        image: str = None,
        model: str = "pika-1.0"
    ) -> Dict[str, Any]:
        """
        🎞️ Pika Labs Video Generation
        """
        if "pika" not in self.api_keys:
            return {"success": False, "error": "Pika API key non configurata (PIKA_API_KEY)"}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_keys['pika']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "model": model
            }
            
            if image:
                payload["image"] = image
            
            async with self.session.post(
                "https://api.pika.art/v1/generations",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                if response.status in [200, 201]:
                    return {
                        "success": True,
                        "generation_id": data.get("id"),
                        "status": "processing",
                        "provider": "pika",
                        "model": model
                    }
                else:
                    return {"success": False, "error": data.get("message", "Pika API error")}
                    
        except Exception as e:
            return {"success": False, "error": f"Pika error: {str(e)}"}

    # ============================================================
    #                      MUSIC GENERATION
    # ============================================================
    
    async def generate_music(
        self,
        prompt: str,
        style: str = None,
        duration: int = 30,
        provider: str = "suno"
    ) -> Dict[str, Any]:
        """
        🎵 Genera musica con AI
        
        Providers: suno
        """
        if provider == "suno":
            return await self._suno_generate(prompt, style, duration)
        
        return {"success": False, "error": f"Provider {provider} non supportato"}
    
    async def _suno_generate(self, prompt: str, style: str, duration: int) -> Dict:
        """Suno AI Music Generation"""
        # Suno non ha API pubblica ufficiale, usiamo alternative
        return {
            "success": False,
            "error": "Suno AI richiede integrazione custom",
            "suggestion": "Usa l'interfaccia web di Suno o attendi l'API ufficiale"
        }
    
    # ============================================================
    #                      PRESENTATION GENERATION
    # ============================================================
    
    async def generate_presentation(
        self,
        topic: str,
        slides: int = 10,
        style: str = "professional",
        provider: str = "gamma"
    ) -> Dict[str, Any]:
        """
        📊 Genera presentazioni con AI
        
        Providers: gamma (Gamma.app style)
        """
        # Per ora generiamo con AI chat + template
        prompt = f"""Crea una presentazione su "{topic}" con {slides} slide.
Per ogni slide fornisci:
- Titolo
- Punti chiave (3-5 bullet points)
- Suggerimento per immagine

Stile: {style}
Formato output: JSON array"""

        result = await self.chat(prompt, max_tokens=2000)
        
        if result.get("success"):
            return {
                "success": True,
                "content": result["response"],
                "slides": slides,
                "topic": topic,
                "provider": "gideon_ai"
            }
        return result
    
    # ============================================================
    #                      PODCAST GENERATION
    # ============================================================
    
    async def generate_podcast(
        self,
        content: str,
        title: str = "GIDEON Podcast",
        hosts: int = 2,
        duration_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        🎙️ Genera podcast da contenuti (stile NotebookLM)
        
        1. Analizza il contenuto
        2. Genera script conversazionale
        3. Genera audio con voci diverse
        """
        # Step 1: Genera script
        script_prompt = f"""Crea uno script per un podcast di {duration_minutes} minuti su:

{content[:2000]}

Il podcast ha {hosts} host che discutono in modo naturale e coinvolgente.
Includi:
- Introduzione accattivante
- Discussione dei punti chiave
- Domande interessanti
- Conclusione

Formato: dialogo con [Host1] e [Host2]"""

        script_result = await self.chat(script_prompt, max_tokens=2000)
        
        if not script_result.get("success"):
            return script_result
        
        script = script_result["response"]
        
        # Step 2: Genera audio (per ora solo lo script)
        return {
            "success": True,
            "title": title,
            "script": script,
            "duration_minutes": duration_minutes,
            "hosts": hosts,
            "audio_status": "Script generato. Audio TTS disponibile separatamente.",
            "provider": "gideon_ai"
        }
    
    # ============================================================
    #                      UTILITY METHODS
    # ============================================================
    
    def get_available_services(self) -> Dict[str, List[str]]:
        """Ritorna i servizi disponibili per tipo"""
        available = {}
        for service_id, service in self.services.items():
            service_type = service.type.value
            if service_type not in available:
                available[service_type] = []
            
            status = "✅" if service_id in self.api_keys else "❌"
            available[service_type].append(f"{status} {service.name}")
        
        return available
    
    def get_stats(self) -> Dict:
        """Ritorna statistiche uso"""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["started_at"]).total_seconds()
        }


# Singleton
_ai_hub: Optional[GideonAIHub] = None

def get_ai_hub() -> GideonAIHub:
    """Ottieni istanza AI Hub"""
    global _ai_hub
    if _ai_hub is None:
        _ai_hub = GideonAIHub()
    return _ai_hub

async def init_ai_hub() -> GideonAIHub:
    """Inizializza AI Hub"""
    hub = get_ai_hub()
    await hub.initialize()
    return hub
