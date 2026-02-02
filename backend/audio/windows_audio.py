"""
Windows Audio System Integration for Gideon 2.0
Handles microphone input and speaker output on Windows
"""

import asyncio
import numpy as np
from loguru import logger
from typing import Optional, Callable
import threading
import queue

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ sounddevice/soundfile not available - audio disabled")
    AUDIO_AVAILABLE = False


class WindowsAudioSystem:
    """Windows-specific audio system for microphone and speaker"""
    
    def __init__(self):
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.callback: Optional[Callable] = None
        self.stream = None
        self.audio_available = AUDIO_AVAILABLE
        
    async def initialize(self):
        """Initialize audio system"""
        if not self.audio_available:
            logger.warning("❌ Audio system not available")
            return
            
        logger.info("🎤 Initializing Windows Audio System...")
        
        try:
            # List available devices
            devices = sd.query_devices()
            logger.debug(f"Available audio devices: {devices}")
            
            logger.info("✅ Windows Audio System ready")
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            self.audio_available = False
            
    async def start_recording(self, callback: Callable):
        """Start recording microphone"""
        if not self.audio_available:
            logger.warning("⚠️ Audio not available - recording skipped")
            return
            
        self.callback = callback
        self.is_recording = True
        
        logger.info("🎤 Started recording...")
        
        try:
            # Start recording in separate thread
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.error(f"Audio status: {status}")
                
                # Add audio data to queue
                self.audio_queue.put(indata.copy())
            
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=audio_callback,
                blocksize=2048
            )
            
            self.stream.start()
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            
    async def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        logger.info("⏹️ Recording stopped")
        
    async def play_audio(self, audio_data: np.ndarray):
        """Play audio through speakers"""
        if not self.audio_available:
            logger.warning("⚠️ Audio not available - playback skipped")
            return
            
        try:
            logger.debug("🔊 Playing audio...")
            sd.play(audio_data, self.sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            
    async def shutdown(self):
        """Shutdown audio system"""
        if self.is_recording:
            await self.stop_recording()
        logger.info("🔇 Audio system shutdown")
