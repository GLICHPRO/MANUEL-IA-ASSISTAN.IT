"""
🎭 Avatar Controller - Controllo Avatar 3D GIDEON

Sistema di controllo per avatar 3D stile Gideon (The Flash):
- Gestione espressioni facciali
- Lip sync sincronizzato con TTS
- Movimenti testa e occhi realistici
- Effetti olografici
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
import math
import random
import asyncio
import json
import logging
from collections import deque


# Logger
avatar_logger = logging.getLogger("avatar_controller")
avatar_logger.setLevel(logging.DEBUG)


# === OPERATING MODE COLORS ===
# Verde = Pilot (controllo totale)
# Giallo = Analisi/Thinking
# Blu = Idle/Copilot
# Arancione = Alert
# Viola = Executive

class OperatingModeColors:
    """Colori per modalità operativa"""
    PILOT = "#00FF7F"        # Verde brillante - controllo totale
    COPILOT = "#00BFFF"      # Blu ciano - assistenza
    PASSIVE = "#4169E1"      # Blu reale - osservazione
    EXECUTIVE = "#9370DB"    # Viola - supervisione
    ANALYZING = "#FFD700"    # Giallo oro - analisi in corso
    PROCESSING = "#FFA500"   # Arancione - elaborazione
    ALERT = "#FF4500"        # Rosso-arancio - allarme
    IDLE = "#00BFFF"         # Blu ciano - standby


# === ENUMS ===

class AvatarState(Enum):
    """Stato dell'avatar"""
    IDLE = "idle"
    SPEAKING = "speaking"
    LISTENING = "listening"
    THINKING = "thinking"
    PROCESSING = "processing"
    ALERT = "alert"
    SLEEPING = "sleeping"


class Expression(Enum):
    """Espressioni facciali"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    THINKING = "thinking"
    FOCUSED = "focused"
    CONCERNED = "concerned"
    CONFIDENT = "confident"
    CURIOUS = "curious"
    SURPRISED = "surprised"
    SERIOUS = "serious"
    FRIENDLY = "friendly"


class Viseme(Enum):
    """Visemi per lip sync (forme della bocca)"""
    SILENCE = "sil"      # Bocca chiusa
    PP = "PP"            # P, B, M
    FF = "FF"            # F, V
    TH = "TH"            # TH
    DD = "DD"            # T, D, N, L
    KK = "kk"            # K, G, NG
    CH = "CH"            # CH, J, SH
    SS = "SS"            # S, Z
    NN = "nn"            # N
    RR = "RR"            # R
    AA = "aa"            # A
    EE = "E"             # E, I
    II = "I"             # I
    OO = "O"             # O
    UU = "U"             # U


class GazeTarget(Enum):
    """Target per lo sguardo"""
    USER = "user"
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    RANDOM = "random"


# === DATA CLASSES ===

@dataclass
class HeadPose:
    """Posizione della testa"""
    pitch: float = 0.0      # Su/giù (-30 to +20 gradi)
    yaw: float = 0.0        # Destra/sinistra (-45 to +45 gradi)
    roll: float = 0.0       # Inclinazione (-15 to +15 gradi)
    
    def to_dict(self) -> dict:
        return {
            "pitch": round(self.pitch, 3),
            "yaw": round(self.yaw, 3),
            "roll": round(self.roll, 3)
        }
    
    def interpolate(self, target: 'HeadPose', t: float) -> 'HeadPose':
        """Interpola verso una posizione target"""
        return HeadPose(
            pitch=self.pitch + (target.pitch - self.pitch) * t,
            yaw=self.yaw + (target.yaw - self.yaw) * t,
            roll=self.roll + (target.roll - self.roll) * t
        )


@dataclass
class EyeState:
    """Stato degli occhi"""
    left_x: float = 0.0     # Posizione orizzontale (-1 to 1)
    left_y: float = 0.0     # Posizione verticale (-1 to 1)
    right_x: float = 0.0
    right_y: float = 0.0
    left_open: float = 1.0  # Apertura (0 = chiuso, 1 = aperto)
    right_open: float = 1.0
    pupil_size: float = 1.0  # Dimensione pupilla (0.5 to 1.5)
    
    def to_dict(self) -> dict:
        return {
            "left": {
                "x": round(self.left_x, 3),
                "y": round(self.left_y, 3),
                "open": round(self.left_open, 3)
            },
            "right": {
                "x": round(self.right_x, 3),
                "y": round(self.right_y, 3),
                "open": round(self.right_open, 3)
            },
            "pupil_size": round(self.pupil_size, 3)
        }


@dataclass
class MouthState:
    """Stato della bocca"""
    open: float = 0.0       # Apertura verticale (0 to 1)
    wide: float = 0.5       # Larghezza (0 to 1)
    smile: float = 0.0      # Sorriso (-1 triste, 0 neutro, 1 sorriso)
    viseme: Viseme = Viseme.SILENCE
    
    def to_dict(self) -> dict:
        return {
            "open": round(self.open, 3),
            "wide": round(self.wide, 3),
            "smile": round(self.smile, 3),
            "viseme": self.viseme.value
        }


@dataclass
class ExpressionState:
    """Stato espressione facciale completo"""
    expression: Expression = Expression.NEUTRAL
    intensity: float = 1.0  # 0 to 1
    
    # Componenti singoli
    eyebrow_left: float = 0.0   # -1 (giù) to 1 (su)
    eyebrow_right: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "expression": self.expression.value,
            "intensity": round(self.intensity, 3),
            "eyebrow_left": round(self.eyebrow_left, 3),
            "eyebrow_right": round(self.eyebrow_right, 3)
        }


@dataclass
class HologramEffect:
    """Effetti olografici stile Gideon"""
    glow_intensity: float = 0.6     # Intensità glow (0 to 1)
    glow_color: str = "#00BFFF"     # Colore primario (Deep Sky Blue)
    scan_line_speed: float = 1.0   # Velocità linee scansione
    flicker_amount: float = 0.02   # Quantità flicker
    distortion: float = 0.0        # Distorsione (0 to 1)
    particle_density: float = 0.5  # Densità particelle
    ring_count: int = 3            # Anelli olografici
    
    def to_dict(self) -> dict:
        return {
            "glow_intensity": round(self.glow_intensity, 3),
            "glow_color": self.glow_color,
            "scan_line_speed": round(self.scan_line_speed, 3),
            "flicker_amount": round(self.flicker_amount, 3),
            "distortion": round(self.distortion, 3),
            "particle_density": round(self.particle_density, 3),
            "ring_count": self.ring_count
        }


@dataclass
class LipSyncFrame:
    """Frame singolo di lip sync"""
    timestamp: float        # Tempo in secondi
    viseme: Viseme
    intensity: float = 1.0  # Intensità del visema
    
    def to_dict(self) -> dict:
        return {
            "t": round(self.timestamp, 3),
            "v": self.viseme.value,
            "i": round(self.intensity, 3)
        }


@dataclass
class AvatarAnimation:
    """Animazione completa dell'avatar"""
    id: str
    name: str
    duration: float  # secondi
    
    # Keyframes
    head_keyframes: List[Tuple[float, HeadPose]] = field(default_factory=list)
    eye_keyframes: List[Tuple[float, EyeState]] = field(default_factory=list)
    expression_keyframes: List[Tuple[float, ExpressionState]] = field(default_factory=list)
    
    loop: bool = False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "duration": self.duration,
            "loop": self.loop,
            "keyframe_count": len(self.head_keyframes) + len(self.eye_keyframes)
        }


# === AVATAR CONTROLLER ===

class AvatarController:
    """
    Controller principale per l'avatar 3D di GIDEON.
    Gestisce tutte le animazioni, espressioni e lip sync.
    """
    
    def __init__(self):
        # State
        self.state = AvatarState.IDLE
        self.expression = ExpressionState()
        self.head = HeadPose()
        self.eyes = EyeState()
        self.mouth = MouthState()
        self.hologram = HologramEffect()
        
        # Animation
        self.current_animation: Optional[AvatarAnimation] = None
        self.animation_time: float = 0.0
        
        # Lip sync
        self.lip_sync_frames: deque = deque(maxlen=1000)
        self.lip_sync_active: bool = False
        self.lip_sync_start_time: float = 0.0
        
        # Blink system
        self.last_blink_time: float = 0.0
        self.blink_interval: float = 3.5  # Secondi tra blink
        self.blink_duration: float = 0.15
        self.is_blinking: bool = False
        
        # Gaze system
        self.gaze_target = GazeTarget.USER
        self.gaze_offset = (0.0, 0.0)
        self.micro_saccade_timer: float = 0.0
        
        # Idle movement
        self.idle_time: float = 0.0
        self.breathing_phase: float = 0.0
        
        # Expression presets
        self.expression_presets = self._init_expression_presets()
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        
        avatar_logger.info("AvatarController initialized")
    
    def _init_expression_presets(self) -> Dict[Expression, Dict]:
        """Inizializza preset espressioni"""
        return {
            Expression.NEUTRAL: {
                "eyebrow_left": 0.0, "eyebrow_right": 0.0,
                "eye_open": 1.0, "smile": 0.0,
                "glow_color": "#00BFFF", "glow_intensity": 0.6
            },
            Expression.HAPPY: {
                "eyebrow_left": 0.2, "eyebrow_right": 0.2,
                "eye_open": 0.85, "smile": 0.6,
                "glow_color": "#00FF7F", "glow_intensity": 0.8
            },
            Expression.THINKING: {
                "eyebrow_left": -0.3, "eyebrow_right": 0.2,
                "eye_open": 0.7, "smile": 0.0,
                "glow_color": "#FFD700", "glow_intensity": 0.7
            },
            Expression.FOCUSED: {
                "eyebrow_left": -0.1, "eyebrow_right": -0.1,
                "eye_open": 1.1, "smile": 0.0,
                "glow_color": "#00BFFF", "glow_intensity": 0.9
            },
            Expression.CONCERNED: {
                "eyebrow_left": 0.3, "eyebrow_right": 0.3,
                "eye_open": 1.1, "smile": -0.3,
                "glow_color": "#FFA500", "glow_intensity": 0.7
            },
            Expression.CONFIDENT: {
                "eyebrow_left": 0.1, "eyebrow_right": 0.1,
                "eye_open": 0.95, "smile": 0.3,
                "glow_color": "#9370DB", "glow_intensity": 0.85
            },
            Expression.CURIOUS: {
                "eyebrow_left": 0.4, "eyebrow_right": 0.4,
                "eye_open": 1.2, "smile": 0.1,
                "glow_color": "#00CED1", "glow_intensity": 0.75
            },
            Expression.SURPRISED: {
                "eyebrow_left": 0.5, "eyebrow_right": 0.5,
                "eye_open": 1.3, "smile": 0.0,
                "glow_color": "#FF69B4", "glow_intensity": 0.9
            },
            Expression.SERIOUS: {
                "eyebrow_left": -0.2, "eyebrow_right": -0.2,
                "eye_open": 0.9, "smile": -0.1,
                "glow_color": "#4169E1", "glow_intensity": 0.65
            },
            Expression.FRIENDLY: {
                "eyebrow_left": 0.15, "eyebrow_right": 0.15,
                "eye_open": 0.95, "smile": 0.4,
                "glow_color": "#00FA9A", "glow_intensity": 0.75
            }
        }
    
    # === State Management ===
    
    def set_state(self, state: AvatarState):
        """Imposta stato avatar"""
        old_state = self.state
        self.state = state
        
        # Applica effetti stato
        if state == AvatarState.SPEAKING:
            self.lip_sync_active = True
            self.hologram.glow_intensity = 0.8
        elif state == AvatarState.LISTENING:
            self.set_expression(Expression.FOCUSED)
            self.hologram.glow_intensity = 0.7
        elif state == AvatarState.THINKING:
            self.set_expression(Expression.THINKING)
            self.hologram.glow_intensity = 0.6
            self.hologram.scan_line_speed = 2.0
        elif state == AvatarState.PROCESSING:
            self.hologram.glow_intensity = 0.9
            self.hologram.scan_line_speed = 3.0
        elif state == AvatarState.ALERT:
            self.set_expression(Expression.CONCERNED)
            self.hologram.glow_color = "#FF4500"
            self.hologram.flicker_amount = 0.1
        elif state == AvatarState.SLEEPING:
            self.eyes.left_open = 0.0
            self.eyes.right_open = 0.0
            self.hologram.glow_intensity = 0.2
        elif state == AvatarState.IDLE:
            self.lip_sync_active = False
            self.set_expression(Expression.NEUTRAL)
            self.hologram = HologramEffect()  # Reset
        
        if self.on_state_change:
            self.on_state_change(old_state, state)
        
        avatar_logger.debug(f"State changed: {old_state.value} -> {state.value}")
    
    def set_expression(self, expression: Expression, intensity: float = 1.0, preserve_color: bool = False):
        """Imposta espressione facciale"""
        preset = self.expression_presets.get(expression, {})
        
        self.expression.expression = expression
        self.expression.intensity = intensity
        self.expression.eyebrow_left = preset.get("eyebrow_left", 0) * intensity
        self.expression.eyebrow_right = preset.get("eyebrow_right", 0) * intensity
        
        # Aggiorna altri componenti
        self.eyes.left_open = preset.get("eye_open", 1.0)
        self.eyes.right_open = preset.get("eye_open", 1.0)
        self.mouth.smile = preset.get("smile", 0) * intensity
        
        # Aggiorna ologramma solo se non preserviamo il colore (gestito da FeedbackSystem)
        if not preserve_color:
            if "glow_color" in preset:
                self.hologram.glow_color = preset["glow_color"]
            if "glow_intensity" in preset:
                self.hologram.glow_intensity = preset["glow_intensity"]
    
    # === Lip Sync ===
    
    def start_lip_sync(self, text: str, duration: float = None):
        """Avvia lip sync per testo"""
        # Genera visemi dal testo
        frames = self._text_to_visemes(text, duration)
        
        self.lip_sync_frames.clear()
        self.lip_sync_frames.extend(frames)
        self.lip_sync_active = True
        self.lip_sync_start_time = 0.0
        
        self.set_state(AvatarState.SPEAKING)
        
        avatar_logger.debug(f"Lip sync started: {len(frames)} frames")
    
    def _text_to_visemes(self, text: str, duration: float = None) -> List[LipSyncFrame]:
        """Converte testo in visemi"""
        # Mappa fonemi -> visemi (semplificata per italiano/inglese)
        phoneme_map = {
            'a': Viseme.AA, 'e': Viseme.EE, 'i': Viseme.II,
            'o': Viseme.OO, 'u': Viseme.UU,
            'p': Viseme.PP, 'b': Viseme.PP, 'm': Viseme.PP,
            'f': Viseme.FF, 'v': Viseme.FF,
            't': Viseme.DD, 'd': Viseme.DD, 'n': Viseme.NN, 'l': Viseme.DD,
            'k': Viseme.KK, 'g': Viseme.KK,
            'c': Viseme.CH, 'j': Viseme.CH,
            's': Viseme.SS, 'z': Viseme.SS,
            'r': Viseme.RR,
            ' ': Viseme.SILENCE, '.': Viseme.SILENCE,
            ',': Viseme.SILENCE, '!': Viseme.SILENCE, '?': Viseme.SILENCE
        }
        
        text_lower = text.lower()
        
        # Calcola durata se non specificata (circa 150ms per carattere)
        if duration is None:
            duration = len(text) * 0.08
        
        frames = []
        time_per_char = duration / max(len(text), 1)
        
        for i, char in enumerate(text_lower):
            viseme = phoneme_map.get(char, Viseme.SILENCE)
            timestamp = i * time_per_char
            
            # Intensità basata sul tipo di visema
            intensity = 0.8 if viseme != Viseme.SILENCE else 0.0
            
            frames.append(LipSyncFrame(
                timestamp=timestamp,
                viseme=viseme,
                intensity=intensity
            ))
        
        # Aggiungi frame finale di chiusura
        frames.append(LipSyncFrame(
            timestamp=duration,
            viseme=Viseme.SILENCE,
            intensity=0.0
        ))
        
        return frames
    
    def load_lip_sync_data(self, frames: List[Dict]):
        """Carica dati lip sync pre-generati (es. da TTS)"""
        self.lip_sync_frames.clear()
        
        for frame_data in frames:
            frame = LipSyncFrame(
                timestamp=frame_data.get("t", 0),
                viseme=Viseme(frame_data.get("v", "sil")),
                intensity=frame_data.get("i", 1.0)
            )
            self.lip_sync_frames.append(frame)
        
        self.lip_sync_active = True
        self.lip_sync_start_time = 0.0
    
    def stop_lip_sync(self):
        """Ferma lip sync"""
        self.lip_sync_active = False
        self.lip_sync_frames.clear()
        self.mouth.open = 0.0
        self.mouth.viseme = Viseme.SILENCE
        
        if self.state == AvatarState.SPEAKING:
            self.set_state(AvatarState.IDLE)
    
    # === Eye/Gaze Control ===
    
    def set_gaze_target(self, target: GazeTarget, offset: Tuple[float, float] = (0, 0)):
        """Imposta target dello sguardo"""
        self.gaze_target = target
        self.gaze_offset = offset
    
    def look_at(self, x: float, y: float):
        """Guarda a coordinate specifiche (-1 to 1)"""
        x = max(-1, min(1, x))
        y = max(-1, min(1, y))
        
        self.eyes.left_x = x
        self.eyes.right_x = x
        self.eyes.left_y = y
        self.eyes.right_y = y
    
    def _update_gaze(self, dt: float):
        """Aggiorna posizione sguardo"""
        target_x, target_y = 0.0, 0.0
        
        if self.gaze_target == GazeTarget.USER:
            target_x, target_y = 0.0, 0.0
        elif self.gaze_target == GazeTarget.LEFT:
            target_x = -0.5
        elif self.gaze_target == GazeTarget.RIGHT:
            target_x = 0.5
        elif self.gaze_target == GazeTarget.UP:
            target_y = 0.3
        elif self.gaze_target == GazeTarget.DOWN:
            target_y = -0.3
        elif self.gaze_target == GazeTarget.RANDOM:
            self.micro_saccade_timer += dt
            if self.micro_saccade_timer > 2.0:
                self.micro_saccade_timer = 0.0
                target_x = random.uniform(-0.3, 0.3)
                target_y = random.uniform(-0.2, 0.2)
        
        # Applica offset
        target_x += self.gaze_offset[0]
        target_y += self.gaze_offset[1]
        
        # Smooth interpolation
        lerp_speed = 8.0 * dt
        self.eyes.left_x += (target_x - self.eyes.left_x) * lerp_speed
        self.eyes.right_x += (target_x - self.eyes.right_x) * lerp_speed
        self.eyes.left_y += (target_y - self.eyes.left_y) * lerp_speed
        self.eyes.right_y += (target_y - self.eyes.right_y) * lerp_speed
        
        # Micro-saccades (piccoli movimenti naturali)
        saccade = math.sin(self.idle_time * 3.0) * 0.02
        self.eyes.left_x += saccade
        self.eyes.right_x += saccade
    
    def _update_blink(self, dt: float, current_time: float):
        """Aggiorna sistema di blink"""
        time_since_blink = current_time - self.last_blink_time
        
        # Trigger blink
        if time_since_blink > self.blink_interval and not self.is_blinking:
            # Variazione naturale nell'intervallo
            if random.random() < 0.3:  # 30% chance per frame
                self.is_blinking = True
                self.last_blink_time = current_time
                # Varia l'intervallo per il prossimo blink
                self.blink_interval = random.uniform(2.5, 5.0)
        
        # Processo blink
        if self.is_blinking:
            blink_progress = (current_time - self.last_blink_time) / self.blink_duration
            
            if blink_progress < 0.5:
                # Chiusura
                openness = 1.0 - (blink_progress * 2)
            elif blink_progress < 1.0:
                # Apertura
                openness = (blink_progress - 0.5) * 2
            else:
                openness = 1.0
                self.is_blinking = False
            
            self.eyes.left_open = openness
            self.eyes.right_open = openness
    
    # === Head Movement ===
    
    def set_head_pose(self, pitch: float = 0, yaw: float = 0, roll: float = 0):
        """Imposta posizione testa"""
        self.head.pitch = max(-30, min(20, pitch))
        self.head.yaw = max(-45, min(45, yaw))
        self.head.roll = max(-15, min(15, roll))
    
    def _update_head_idle(self, dt: float):
        """Movimento idle naturale della testa"""
        # Breathing movement
        self.breathing_phase += dt * 0.5
        breathing = math.sin(self.breathing_phase) * 0.5
        
        # Slow wandering
        wander_pitch = math.sin(self.idle_time * 0.2) * 2
        wander_yaw = math.sin(self.idle_time * 0.15) * 3
        
        # Smooth blend
        target = HeadPose(
            pitch=wander_pitch + breathing,
            yaw=wander_yaw,
            roll=math.sin(self.idle_time * 0.1) * 1
        )
        
        # Interpolate
        lerp_speed = 2.0 * dt
        self.head = self.head.interpolate(target, lerp_speed)
    
    # === Update Loop ===
    
    def update(self, dt: float, current_time: float = None) -> Dict:
        """
        Aggiorna stato avatar.
        Chiamare ogni frame con delta time.
        
        Returns: Stato completo per rendering
        """
        if current_time is None:
            current_time = self.idle_time
        
        self.idle_time += dt
        
        # Update components
        self._update_gaze(dt)
        self._update_blink(dt, current_time)
        
        if self.state == AvatarState.IDLE:
            self._update_head_idle(dt)
        
        # Update lip sync
        if self.lip_sync_active and self.lip_sync_frames:
            self._update_lip_sync(dt)
        
        # Update hologram effects
        self._update_hologram(dt)
        
        return self.get_render_state()
    
    def _update_lip_sync(self, dt: float):
        """Aggiorna lip sync"""
        self.lip_sync_start_time += dt
        
        # Trova frame corrente
        current_frame = None
        for frame in self.lip_sync_frames:
            if frame.timestamp <= self.lip_sync_start_time:
                current_frame = frame
            else:
                break
        
        if current_frame:
            self.mouth.viseme = current_frame.viseme
            
            # Calcola apertura bocca basata su visema
            viseme_openness = {
                Viseme.SILENCE: 0.0,
                Viseme.PP: 0.0,
                Viseme.FF: 0.15,
                Viseme.TH: 0.2,
                Viseme.DD: 0.25,
                Viseme.KK: 0.3,
                Viseme.CH: 0.35,
                Viseme.SS: 0.2,
                Viseme.NN: 0.15,
                Viseme.RR: 0.25,
                Viseme.AA: 0.7,
                Viseme.EE: 0.4,
                Viseme.II: 0.3,
                Viseme.OO: 0.6,
                Viseme.UU: 0.5
            }
            
            target_open = viseme_openness.get(current_frame.viseme, 0.3) * current_frame.intensity
            
            # Smooth mouth movement
            self.mouth.open += (target_open - self.mouth.open) * 0.3
        
        # Check if finished
        if self.lip_sync_frames and self.lip_sync_start_time > self.lip_sync_frames[-1].timestamp:
            self.stop_lip_sync()
    
    def _update_hologram(self, dt: float):
        """Aggiorna effetti olografici"""
        # Flicker effect
        if self.hologram.flicker_amount > 0:
            flicker = random.uniform(-self.hologram.flicker_amount, self.hologram.flicker_amount)
            self.hologram.glow_intensity = max(0, min(1, self.hologram.glow_intensity + flicker))
        
        # Pulse effect quando parla
        if self.state == AvatarState.SPEAKING:
            pulse = math.sin(self.idle_time * 8) * 0.1
            self.hologram.glow_intensity = 0.7 + pulse
    
    # === Render State ===
    
    def get_render_state(self) -> Dict:
        """Ritorna stato completo per il renderer 3D"""
        return {
            "state": self.state.value,
            "head": self.head.to_dict(),
            "eyes": self.eyes.to_dict(),
            "mouth": self.mouth.to_dict(),
            "expression": self.expression.to_dict(),
            "hologram": self.hologram.to_dict(),
            "timestamp": self.idle_time
        }
    
    def get_status(self) -> Dict:
        """Status per debug/API"""
        return {
            "state": self.state.value,
            "expression": self.expression.expression.value,
            "is_speaking": self.lip_sync_active,
            "gaze_target": self.gaze_target.value,
            "hologram_color": self.hologram.glow_color
        }


# === PRESET ANIMATIONS ===

class AvatarAnimations:
    """Collezione di animazioni predefinite"""
    
    @staticmethod
    def nod() -> AvatarAnimation:
        """Annuire"""
        return AvatarAnimation(
            id="nod",
            name="Nod",
            duration=0.8,
            head_keyframes=[
                (0.0, HeadPose(pitch=0, yaw=0, roll=0)),
                (0.2, HeadPose(pitch=-10, yaw=0, roll=0)),
                (0.4, HeadPose(pitch=5, yaw=0, roll=0)),
                (0.6, HeadPose(pitch=-5, yaw=0, roll=0)),
                (0.8, HeadPose(pitch=0, yaw=0, roll=0))
            ]
        )
    
    @staticmethod
    def shake_head() -> AvatarAnimation:
        """Scuotere la testa"""
        return AvatarAnimation(
            id="shake",
            name="Shake Head",
            duration=1.0,
            head_keyframes=[
                (0.0, HeadPose(pitch=0, yaw=0, roll=0)),
                (0.2, HeadPose(pitch=0, yaw=15, roll=0)),
                (0.4, HeadPose(pitch=0, yaw=-15, roll=0)),
                (0.6, HeadPose(pitch=0, yaw=10, roll=0)),
                (0.8, HeadPose(pitch=0, yaw=-5, roll=0)),
                (1.0, HeadPose(pitch=0, yaw=0, roll=0))
            ]
        )
    
    @staticmethod
    def thinking() -> AvatarAnimation:
        """Animazione pensiero"""
        return AvatarAnimation(
            id="thinking",
            name="Thinking",
            duration=2.0,
            head_keyframes=[
                (0.0, HeadPose(pitch=0, yaw=0, roll=0)),
                (0.5, HeadPose(pitch=-5, yaw=10, roll=3)),
                (1.5, HeadPose(pitch=-5, yaw=-5, roll=-2)),
                (2.0, HeadPose(pitch=0, yaw=0, roll=0))
            ],
            loop=True
        )
    
    @staticmethod
    def alert() -> AvatarAnimation:
        """Animazione allerta"""
        return AvatarAnimation(
            id="alert",
            name="Alert",
            duration=0.5,
            head_keyframes=[
                (0.0, HeadPose(pitch=0, yaw=0, roll=0)),
                (0.1, HeadPose(pitch=10, yaw=0, roll=0)),
                (0.3, HeadPose(pitch=5, yaw=0, roll=0)),
                (0.5, HeadPose(pitch=0, yaw=0, roll=0))
            ]
        )


# === AVATAR FEEDBACK SYSTEM ===

@dataclass
class HUDIndicator:
    """Indicatore HUD per feedback visivo"""
    id: str
    label: str
    value: float = 0.0          # 0-1 per barre, valore numerico per contatori
    max_value: float = 1.0
    color: str = "#00BFFF"
    icon: str = ""               # Emoji/icona
    visible: bool = True
    animated: bool = False
    pulse_speed: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "value": round(self.value, 3),
            "max_value": self.max_value,
            "color": self.color,
            "icon": self.icon,
            "visible": self.visible,
            "animated": self.animated,
            "pulse_speed": self.pulse_speed
        }


@dataclass
class CalculationAnimation:
    """Animazione calcolo/elaborazione"""
    id: str
    type: str                   # 'particles', 'ring_pulse', 'scan', 'data_stream'
    active: bool = False
    progress: float = 0.0       # 0-1
    speed: float = 1.0
    color: str = "#FFD700"
    intensity: float = 0.8
    data_points: List[str] = field(default_factory=list)  # Dati visualizzati
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "active": self.active,
            "progress": round(self.progress, 3),
            "speed": self.speed,
            "color": self.color,
            "intensity": self.intensity,
            "data_points": self.data_points[-5:]  # Ultimi 5 punti
        }


class AvatarFeedbackSystem:
    """
    Sistema di feedback visivo integrato con l'avatar.
    
    Gestisce:
    - Colori glow basati su modalità operativa
    - Espressioni coerenti con stato
    - Indicatori HUD
    - Animazioni di calcolo/elaborazione
    - Feedback visivo continuo
    """
    
    def __init__(self, avatar: AvatarController):
        self.avatar = avatar
        
        # Stato corrente
        self.operating_mode: str = "copilot"  # passive, copilot, pilot, executive
        self.activity_state: str = "idle"      # idle, analyzing, processing, executing
        
        # HUD Indicators
        self.hud_indicators: Dict[str, HUDIndicator] = {
            "cpu": HUDIndicator("cpu", "CPU", icon="⚡"),
            "memory": HUDIndicator("memory", "MEM", icon="💾"),
            "confidence": HUDIndicator("confidence", "CONF", icon="🎯"),
            "progress": HUDIndicator("progress", "PROG", icon="📊", visible=False),
            "mode": HUDIndicator("mode", "MODE", icon="🎮", value=0.5),
            "status": HUDIndicator("status", "STATUS", icon="●")
        }
        
        # Calculation Animations
        self.calculations: Dict[str, CalculationAnimation] = {
            "primary": CalculationAnimation("primary", "ring_pulse"),
            "data_stream": CalculationAnimation("data_stream", "data_stream"),
            "particle_burst": CalculationAnimation("particle_burst", "particles"),
            "scan": CalculationAnimation("scan", "scan")
        }
        
        # Mode → Color mapping
        self.mode_colors = {
            "passive": OperatingModeColors.PASSIVE,
            "copilot": OperatingModeColors.COPILOT,
            "pilot": OperatingModeColors.PILOT,
            "executive": OperatingModeColors.EXECUTIVE
        }
        
        # Activity → Color mapping (override)
        self.activity_colors = {
            "idle": None,  # Usa colore mode
            "analyzing": OperatingModeColors.ANALYZING,
            "processing": OperatingModeColors.PROCESSING,
            "executing": None,  # Usa colore mode
            "alert": OperatingModeColors.ALERT
        }
        
        # Mode → Expression mapping
        self.mode_expressions = {
            "passive": Expression.NEUTRAL,
            "copilot": Expression.FRIENDLY,
            "pilot": Expression.FOCUSED,
            "executive": Expression.CONFIDENT
        }
        
        # Activity → Expression mapping (override)
        self.activity_expressions = {
            "idle": None,  # Usa espressione mode
            "analyzing": Expression.THINKING,
            "processing": Expression.FOCUSED,
            "executing": Expression.CONFIDENT,
            "alert": Expression.CONCERNED
        }
        
        avatar_logger.info("AvatarFeedbackSystem initialized")
    
    def set_operating_mode(self, mode: str):
        """
        Imposta modalità operativa e aggiorna avatar.
        
        Args:
            mode: 'passive', 'copilot', 'pilot', 'executive'
        """
        mode = mode.lower()
        if mode not in self.mode_colors:
            avatar_logger.warning(f"Unknown mode: {mode}")
            return
        
        self.operating_mode = mode
        
        # Aggiorna colore (se non in attività speciale)
        if self.activity_state in ["idle", "executing"]:
            self.avatar.hologram.glow_color = self.mode_colors[mode]
        
        # Aggiorna espressione (preserve_color=True per mantenere il colore modalità)
        if self.activity_state == "idle":
            self.avatar.set_expression(self.mode_expressions[mode], preserve_color=True)
        
        # Aggiorna HUD
        mode_values = {"passive": 0.25, "copilot": 0.5, "pilot": 0.75, "executive": 1.0}
        self.hud_indicators["mode"].value = mode_values.get(mode, 0.5)
        self.hud_indicators["mode"].color = self.mode_colors[mode]
        
        # Effetti speciali per Pilot
        if mode == "pilot":
            self.avatar.hologram.glow_intensity = 0.9
            self.avatar.hologram.particle_density = 0.8
            self.avatar.hologram.ring_count = 5
        elif mode == "executive":
            self.avatar.hologram.glow_intensity = 1.0
            self.avatar.hologram.scan_line_speed = 1.5
            self.avatar.hologram.ring_count = 4
        else:
            self.avatar.hologram.glow_intensity = 0.6
            self.avatar.hologram.particle_density = 0.5
            self.avatar.hologram.ring_count = 3
        
        avatar_logger.debug(f"Operating mode set: {mode} → {self.mode_colors[mode]}")
    
    def set_activity(self, activity: str, progress: float = 0.0):
        """
        Imposta stato attività e aggiorna feedback visivo.
        
        Args:
            activity: 'idle', 'analyzing', 'processing', 'executing', 'alert'
            progress: Progresso 0-1 (per analyzing/processing)
        """
        activity = activity.lower()
        self.activity_state = activity
        
        # Aggiorna colore
        activity_color = self.activity_colors.get(activity)
        if activity_color:
            self.avatar.hologram.glow_color = activity_color
        else:
            self.avatar.hologram.glow_color = self.mode_colors[self.operating_mode]
        
        # Aggiorna espressione (preserve_color=True per non sovrascrivere il colore)
        activity_expr = self.activity_expressions.get(activity)
        if activity_expr:
            self.avatar.set_expression(activity_expr, preserve_color=True)
        else:
            self.avatar.set_expression(self.mode_expressions[self.operating_mode], preserve_color=True)
        
        # Configura animazioni per attività
        if activity == "analyzing":
            self._start_analysis_animation()
            self.hud_indicators["progress"].visible = True
            self.hud_indicators["progress"].value = progress
            self.hud_indicators["progress"].color = OperatingModeColors.ANALYZING
        elif activity == "processing":
            self._start_processing_animation()
            self.hud_indicators["progress"].visible = True
            self.hud_indicators["progress"].value = progress
            self.hud_indicators["progress"].color = OperatingModeColors.PROCESSING
        elif activity == "alert":
            self._start_alert_animation()
        else:
            self._stop_all_animations()
            self.hud_indicators["progress"].visible = False
        
        avatar_logger.debug(f"Activity set: {activity} (progress: {progress})")
    
    def _start_analysis_animation(self):
        """Avvia animazione analisi (giallo)"""
        self.calculations["primary"].active = True
        self.calculations["primary"].color = OperatingModeColors.ANALYZING
        self.calculations["primary"].type = "ring_pulse"
        self.calculations["primary"].speed = 1.5
        
        self.calculations["data_stream"].active = True
        self.calculations["data_stream"].color = OperatingModeColors.ANALYZING
        
        self.avatar.hologram.scan_line_speed = 2.0
        self.avatar.hologram.glow_intensity = 0.75
    
    def _start_processing_animation(self):
        """Avvia animazione elaborazione (arancione)"""
        self.calculations["primary"].active = True
        self.calculations["primary"].color = OperatingModeColors.PROCESSING
        self.calculations["primary"].type = "particles"
        self.calculations["primary"].speed = 2.0
        
        self.calculations["scan"].active = True
        self.calculations["scan"].color = OperatingModeColors.PROCESSING
        
        self.avatar.hologram.scan_line_speed = 3.0
        self.avatar.hologram.glow_intensity = 0.85
        self.avatar.hologram.flicker_amount = 0.05
    
    def _start_alert_animation(self):
        """Avvia animazione allerta (rosso)"""
        self.calculations["particle_burst"].active = True
        self.calculations["particle_burst"].color = OperatingModeColors.ALERT
        self.calculations["particle_burst"].intensity = 1.0
        
        self.avatar.hologram.flicker_amount = 0.15
        self.avatar.hologram.glow_intensity = 1.0
    
    def _stop_all_animations(self):
        """Ferma tutte le animazioni"""
        for calc in self.calculations.values():
            calc.active = False
            calc.progress = 0.0
        
        self.avatar.hologram.flicker_amount = 0.02
        self.avatar.hologram.scan_line_speed = 1.0
    
    def update_progress(self, progress: float):
        """Aggiorna progresso attività corrente"""
        if self.activity_state in ["analyzing", "processing"]:
            self.hud_indicators["progress"].value = max(0, min(1, progress))
            self.calculations["primary"].progress = progress
    
    def add_data_point(self, text: str):
        """Aggiunge punto dati per visualizzazione stream"""
        self.calculations["data_stream"].data_points.append(text)
        if len(self.calculations["data_stream"].data_points) > 10:
            self.calculations["data_stream"].data_points.pop(0)
    
    def update_hud(self, indicator_id: str, value: float = None, 
                   color: str = None, visible: bool = None):
        """Aggiorna indicatore HUD specifico"""
        if indicator_id not in self.hud_indicators:
            return
        
        ind = self.hud_indicators[indicator_id]
        if value is not None:
            ind.value = max(0, min(ind.max_value, value))
        if color is not None:
            ind.color = color
        if visible is not None:
            ind.visible = visible
    
    def set_confidence(self, confidence: float):
        """Aggiorna indicatore confidenza"""
        self.hud_indicators["confidence"].value = confidence
        
        # Colore basato su confidenza
        if confidence >= 0.8:
            self.hud_indicators["confidence"].color = "#00FF7F"  # Verde
        elif confidence >= 0.5:
            self.hud_indicators["confidence"].color = "#FFD700"  # Giallo
        else:
            self.hud_indicators["confidence"].color = "#FF6347"  # Rosso
    
    def simulate_cpu_activity(self, load: float = 0.5):
        """Simula attività CPU per effetto visivo"""
        self.hud_indicators["cpu"].value = load
        self.hud_indicators["cpu"].animated = load > 0.3
        
        if load > 0.7:
            self.hud_indicators["cpu"].color = "#FF6347"
        elif load > 0.4:
            self.hud_indicators["cpu"].color = "#FFD700"
        else:
            self.hud_indicators["cpu"].color = "#00FF7F"
    
    def update(self, delta_time: float = 0.016):
        """Aggiorna sistema feedback (chiamare ogni frame)"""
        # Aggiorna animazioni attive
        for calc in self.calculations.values():
            if calc.active:
                calc.progress = (calc.progress + delta_time * calc.speed) % 1.0
        
        # Pulse su indicatori animati
        import math
        pulse = (math.sin(self.avatar.idle_time * 3) + 1) / 2
        
        for ind in self.hud_indicators.values():
            if ind.animated:
                ind.pulse_speed = 1.0 + pulse * 0.5
    
    def get_feedback_state(self) -> Dict:
        """Ritorna stato completo del feedback system"""
        return {
            "operating_mode": self.operating_mode,
            "activity_state": self.activity_state,
            "glow_color": self.avatar.hologram.glow_color,
            "hud_indicators": {k: v.to_dict() for k, v in self.hud_indicators.items()},
            "calculations": {k: v.to_dict() for k, v in self.calculations.items()},
            "avatar_expression": self.avatar.expression.expression.value
        }
    
    def get_full_render_state(self) -> Dict:
        """Ritorna stato completo per rendering (avatar + feedback)"""
        avatar_state = self.avatar.get_render_state()
        feedback_state = self.get_feedback_state()
        
        return {
            **avatar_state,
            "feedback": feedback_state
        }
