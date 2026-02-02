"""
AI Search Service for Gideon 2.0
Handles queries to external AI services (OpenAI, web search, etc.)
"""

import asyncio
import webbrowser
from typing import Dict, Any, Optional
from loguru import logger
import httpx

from core.config import settings


class AISearchService:
    """Service for searching answers via AI providers"""
    
    def __init__(self):
        self.openai_available = bool(settings.OPENAI_API_KEY)
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.temperature = settings.OPENAI_TEMPERATURE
        self.http_client: Optional[httpx.AsyncClient] = None
        
    async def initialize(self):
        """Initialize the AI search service"""
        logger.info("🔍 Initializing AI Search Service...")
        
        if self.openai_available:
            self.http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            logger.info("✅ OpenAI API configured and ready")
        else:
            logger.warning("⚠️ OpenAI API key not configured - will use browser fallback")
            
        logger.info("✅ AI Search Service ready")
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
        logger.info("🔍 AI Search Service stopped")
    
    async def search_answer(self, question: str, use_browser_fallback: bool = True) -> Dict[str, Any]:
        """
        Search for an answer to a question using AI services.
        
        Args:
            question: The question to answer
            use_browser_fallback: If True, open browser if API fails
            
        Returns:
            Dict with answer, source, and metadata
        """
        logger.info(f"🔍 AI Search: {question[:50]}...")
        
        # Try OpenAI API first if available
        if self.openai_available and self.http_client:
            try:
                result = await self._query_openai(question)
                if result["success"]:
                    return result
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
        
        # Fallback to browser
        if use_browser_fallback:
            return await self._open_chatgpt_browser(question)
        
        return {
            "success": False,
            "answer": "Non sono riuscito a cercare la risposta. Configura l'API OpenAI o usa la ricerca browser.",
            "source": "none",
            "error": "No AI service available"
        }
    
    async def _query_openai(self, question: str) -> Dict[str, Any]:
        """Query OpenAI API for an answer"""
        logger.info("🤖 Querying OpenAI API...")
        
        try:
            response = await self.http_client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": """Sei un assistente AI esperto. Rispondi alle domande in modo:
- Conciso ma completo
- In italiano
- Con esempi pratici quando utile
- Citando fonti se possibile"""
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                
                return {
                    "success": True,
                    "answer": answer,
                    "source": "openai",
                    "model": self.model,
                    "usage": data.get("usage", {})
                }
            else:
                error_msg = response.text
                logger.error(f"OpenAI API error {response.status_code}: {error_msg}")
                return {
                    "success": False,
                    "answer": None,
                    "source": "openai",
                    "error": f"API error: {response.status_code}"
                }
                
        except httpx.TimeoutException:
            logger.error("OpenAI API timeout")
            return {
                "success": False,
                "answer": None,
                "source": "openai",
                "error": "Timeout"
            }
        except Exception as e:
            logger.error(f"OpenAI API exception: {e}")
            return {
                "success": False,
                "answer": None,
                "source": "openai",
                "error": str(e)
            }
    
    async def _open_chatgpt_browser(self, question: str) -> Dict[str, Any]:
        """Open ChatGPT in browser with the question"""
        import urllib.parse
        
        logger.info("🌐 Opening ChatGPT in browser...")
        
        # ChatGPT doesn't support direct query URLs, so we'll open it
        # and copy the question to clipboard if possible
        chatgpt_url = "https://chat.openai.com/"
        
        try:
            # Try to copy question to clipboard (Windows)
            import subprocess
            process = subprocess.Popen(
                ['clip'],
                stdin=subprocess.PIPE,
                shell=True
            )
            process.communicate(question.encode('utf-16-le'))
            clipboard_copied = True
        except:
            clipboard_copied = False
        
        # Open browser
        webbrowser.open(chatgpt_url)
        
        message = "Ho aperto ChatGPT nel browser."
        if clipboard_copied:
            message += " La tua domanda è stata copiata negli appunti - incollala (Ctrl+V) nella chat."
        else:
            message += f" Chiedi: '{question[:100]}...'" if len(question) > 100 else f" Chiedi: '{question}'"
        
        return {
            "success": True,
            "answer": message,
            "source": "browser",
            "url": chatgpt_url,
            "clipboard_copied": clipboard_copied,
            "original_question": question
        }
    
    async def search_web(self, query: str, engine: str = "google") -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            engine: Search engine to use (google, bing, duckduckgo)
        """
        import urllib.parse
        
        engines = {
            "google": "https://www.google.com/search?q=",
            "bing": "https://www.bing.com/search?q=",
            "duckduckgo": "https://duckduckgo.com/?q=",
            "perplexity": "https://www.perplexity.ai/search?q="
        }
        
        base_url = engines.get(engine, engines["google"])
        search_url = base_url + urllib.parse.quote(query)
        
        webbrowser.open(search_url)
        
        return {
            "success": True,
            "answer": f"Ho aperto {engine.capitalize()} con la ricerca '{query}'.",
            "source": engine,
            "url": search_url
        }
    
    async def ask_perplexity(self, question: str) -> Dict[str, Any]:
        """
        Open Perplexity AI for a question (has web search built-in)
        """
        import urllib.parse
        
        perplexity_url = f"https://www.perplexity.ai/search?q={urllib.parse.quote(question)}"
        webbrowser.open(perplexity_url)
        
        return {
            "success": True,
            "answer": f"Ho aperto Perplexity AI con la tua domanda. Perplexity cercherà sul web e ti darà una risposta con fonti.",
            "source": "perplexity",
            "url": perplexity_url
        }


# Singleton instance
ai_search = AISearchService()
