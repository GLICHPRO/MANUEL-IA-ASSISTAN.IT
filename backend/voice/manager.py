"""
Voice Manager - Speech Recognition and Text-to-Speech
Handles all voice input/output with wake word detection
"""

import asyncio
from typing import Optional, Callable
from loguru import logger
import speech_recognition as sr
import pyttsx3
from threading import Thread, Event
from queue import Queue

from core.config import settings


class VoiceManager:
    """Manages voice input and output"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone: Optional[sr.Microphone] = None
        self.tts_engine: Optional[pyttsx3.Engine] = None
        self.is_listening = False
        self.wake_word_detected = Event()
        self.command_queue = Queue()
        self.callback: Optional[Callable] = None
        
    async def initialize(self):
        """Initialize voice systems"""
        logger.info("🎤 Initializing voice system...")
        
        try:
            # Initialize TTS
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS
            self._configure_tts()
            
            # Skip microphone initialization in dev mode (PyAudio issues on Windows)
            logger.warning("⚠️ Voice input disabled (PyAudio not available - OK for dev)")
            
            logger.info("✅ Voice system ready (TTS only)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize voice: {e}")
            # Don't raise - allow app to continue with TTS only
            
    def _configure_tts(self):
        """Configure TTS settings"""
        if not self.tts_engine:
            return
            
        # Set voice properties
        voices = self.tts_engine.getProperty('voices')
        
        # Try to find Italian voice
        for voice in voices:
            if 'italian' in voice.name.lower() or 'it-IT' in voice.id:
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        # Set rate and volume
        self.tts_engine.setProperty('rate', 175)  # Speed
        self.tts_engine.setProperty('volume', 0.9)  # Volume
        
    async def start_listening(self, callback: Callable):
        """Start continuous listening for wake word"""
        self.callback = callback
        self.is_listening = True
        
        # Start listening thread
        Thread(target=self._listen_loop, daemon=True).start()
        
        logger.info(f"👂 Listening for wake word: '{settings.WAKE_WORD}'")
        
    def _listen_loop(self):
        """Continuous listening loop"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen for speech
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    try:
                        # Recognize speech
                        text = self.recognizer.recognize_google(
                            audio,
                            language=settings.VOICE_LANGUAGE
                        )
                        
                        logger.debug(f"🎤 Heard: {text}")
                        
                        # Check for wake word
                        if settings.WAKE_WORD.lower() in text.lower():
                            logger.info(f"✅ Wake word detected!")
                            self.wake_word_detected.set()
                            
                            # Process the command
                            if self.callback:
                                asyncio.run(self.callback(text))
                        
                        # Add to queue for processing
                        self.command_queue.put(text)
                        
                    except sr.UnknownValueError:
                        # Could not understand audio
                        pass
                    except sr.RequestError as e:
                        logger.error(f"Speech recognition error: {e}")
                        
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                
    async def listen_for_command(self, timeout: int = 10) -> Optional[str]:
        """Listen for a single command"""
        try:
            with self.microphone as source:
                logger.info("🎤 Listening for command...")
                
                # Play listening sound
                await self.play_sound("listening")
                
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
                # Recognize
                text = self.recognizer.recognize_google(
                    audio,
                    language=settings.VOICE_LANGUAGE
                )
                
                logger.info(f"📝 Recognized: {text}")
                return text
                
        except sr.WaitTimeoutError:
            logger.warning("⏱️ Listening timeout")
            return None
        except sr.UnknownValueError:
            logger.warning("❓ Could not understand audio")
            return None
        except Exception as e:
            logger.error(f"Error listening: {e}")
            return None
            
    async def speak(self, text: str, expression: str = "neutral"):
        """Speak text using TTS"""
        if not self.tts_engine:
            logger.warning("TTS engine not available")
            return
            
        logger.info(f"🗣️ Speaking: {text}")
        
        try:
            # Run TTS in thread to avoid blocking
            def _speak():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            thread = Thread(target=_speak)
            thread.start()
            thread.join(timeout=30)  # Max 30 seconds
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            
    async def play_sound(self, sound_type: str):
        """Play a system sound"""
        # Could use pygame or other library for sounds
        logger.debug(f"🔊 Playing sound: {sound_type}")
        
    async def shutdown(self):
        """Shutdown voice system"""
        self.is_listening = False
        if self.tts_engine:
            self.tts_engine.stop()
        logger.info("🔇 Voice system stopped")
        
    def is_ready(self) -> bool:
        """Check if voice system is ready"""
        return self.microphone is not None and self.tts_engine is not None
