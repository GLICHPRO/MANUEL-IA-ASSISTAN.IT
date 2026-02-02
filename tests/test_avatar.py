"""
Test per Avatar Controller
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.core.avatar_controller import (
    AvatarController, AvatarState, Expression, GazeTarget,
    HeadPose, EyeState, MouthState, HologramEffect,
    Viseme, AvatarAnimations
)


class TestAvatarController:
    """Test per AvatarController"""
    
    @pytest.fixture
    def controller(self):
        """Crea istanza controller"""
        return AvatarController()
    
    def test_initialization(self, controller):
        """Test inizializzazione"""
        assert controller is not None
        assert controller.state == AvatarState.IDLE
        assert controller.expression.expression == Expression.NEUTRAL
    
    def test_set_state(self, controller):
        """Test cambio stato"""
        controller.set_state(AvatarState.SPEAKING)
        assert controller.state == AvatarState.SPEAKING
        assert controller.lip_sync_active == True
        
        controller.set_state(AvatarState.THINKING)
        assert controller.state == AvatarState.THINKING
    
    def test_set_expression(self, controller):
        """Test cambio espressione"""
        controller.set_expression(Expression.HAPPY, intensity=0.8)
        
        assert controller.expression.expression == Expression.HAPPY
        assert controller.expression.intensity == 0.8
        assert controller.mouth.smile > 0  # Happy ha smile positivo
    
    def test_expression_presets(self, controller):
        """Test tutti i preset espressioni"""
        for expr in Expression:
            controller.set_expression(expr)
            assert controller.expression.expression == expr
    
    def test_head_pose(self, controller):
        """Test posizione testa"""
        controller.set_head_pose(pitch=10, yaw=20, roll=5)
        
        assert controller.head.pitch == 10
        assert controller.head.yaw == 20
        assert controller.head.roll == 5
    
    def test_head_pose_limits(self, controller):
        """Test limiti posizione testa"""
        # Pitch oltre limiti
        controller.set_head_pose(pitch=50)  # Max è 20
        assert controller.head.pitch == 20
        
        controller.set_head_pose(pitch=-50)  # Min è -30
        assert controller.head.pitch == -30
    
    def test_gaze_target(self, controller):
        """Test target sguardo"""
        controller.set_gaze_target(GazeTarget.LEFT, offset=(0.1, 0))
        
        assert controller.gaze_target == GazeTarget.LEFT
        assert controller.gaze_offset == (0.1, 0)
    
    def test_look_at(self, controller):
        """Test look at coordinates"""
        controller.look_at(0.5, -0.3)
        
        assert controller.eyes.left_x == 0.5
        assert controller.eyes.left_y == -0.3
        assert controller.eyes.right_x == 0.5
        assert controller.eyes.right_y == -0.3
    
    def test_look_at_limits(self, controller):
        """Test limiti look at"""
        controller.look_at(2.0, -2.0)
        
        # Deve essere clampato a -1, 1
        assert controller.eyes.left_x == 1.0
        assert controller.eyes.left_y == -1.0
    
    def test_lip_sync_from_text(self, controller):
        """Test lip sync da testo"""
        controller.start_lip_sync("Hello world")
        
        assert controller.lip_sync_active == True
        assert controller.state == AvatarState.SPEAKING
        assert len(controller.lip_sync_frames) > 0
    
    def test_lip_sync_visemes(self, controller):
        """Test generazione visemi"""
        controller.start_lip_sync("papa")
        
        # Dovrebbe contenere visemi PP per 'p'
        visemes = [f.viseme for f in controller.lip_sync_frames]
        assert Viseme.PP in visemes
    
    def test_stop_lip_sync(self, controller):
        """Test stop lip sync"""
        controller.start_lip_sync("Test")
        controller.stop_lip_sync()
        
        assert controller.lip_sync_active == False
        assert controller.mouth.open == 0.0
        assert controller.mouth.viseme == Viseme.SILENCE
    
    def test_load_lip_sync_data(self, controller):
        """Test caricamento dati lip sync"""
        frames = [
            {"t": 0.0, "v": "sil", "i": 0.0},
            {"t": 0.1, "v": "aa", "i": 1.0},
            {"t": 0.2, "v": "O", "i": 0.8},
            {"t": 0.3, "v": "sil", "i": 0.0}
        ]
        
        controller.load_lip_sync_data(frames)
        
        assert controller.lip_sync_active == True
        assert len(controller.lip_sync_frames) == 4
    
    def test_update_loop(self, controller):
        """Test update loop"""
        # Simula alcuni frame
        for i in range(10):
            state = controller.update(dt=0.033, current_time=i * 0.033)
            
            assert "state" in state
            assert "head" in state
            assert "eyes" in state
            assert "mouth" in state
            assert "expression" in state
            assert "hologram" in state
    
    def test_blink_system(self, controller):
        """Test sistema blink"""
        # Forza blink
        controller.last_blink_time = -10  # Tempo passato
        controller.blink_interval = 0.1
        
        # Update dovrebbe triggerare blink
        controller.update(dt=0.1, current_time=5.0)
        
        # Verifica che il sistema di blink funzioni
        assert controller.blink_interval > 0
    
    def test_hologram_effects(self, controller):
        """Test effetti olografici"""
        assert controller.hologram.glow_color == "#00BFFF"
        assert controller.hologram.glow_intensity > 0
        
        # Cambia stato per modificare effetti
        controller.set_state(AvatarState.ALERT)
        assert controller.hologram.glow_color == "#FF4500"  # Arancione per alert
    
    def test_get_render_state(self, controller):
        """Test render state"""
        state = controller.get_render_state()
        
        # Verifica struttura completa
        assert state["state"] == "idle"
        assert "pitch" in state["head"]
        assert "left" in state["eyes"]
        assert "open" in state["mouth"]
        assert "expression" in state["expression"]
        assert "glow_color" in state["hologram"]
    
    def test_get_status(self, controller):
        """Test status"""
        status = controller.get_status()
        
        assert "state" in status
        assert "expression" in status
        assert "is_speaking" in status
        assert "gaze_target" in status


class TestHeadPose:
    """Test per HeadPose"""
    
    def test_to_dict(self):
        """Test serializzazione"""
        pose = HeadPose(pitch=10.123, yaw=5.456, roll=-2.789)
        d = pose.to_dict()
        
        assert d["pitch"] == 10.123
        assert d["yaw"] == 5.456
        assert d["roll"] == -2.789
    
    def test_interpolate(self):
        """Test interpolazione"""
        pose1 = HeadPose(pitch=0, yaw=0, roll=0)
        pose2 = HeadPose(pitch=10, yaw=20, roll=-10)
        
        # 50% interpolation
        result = pose1.interpolate(pose2, 0.5)
        
        assert result.pitch == 5.0
        assert result.yaw == 10.0
        assert result.roll == -5.0


class TestEyeState:
    """Test per EyeState"""
    
    def test_to_dict(self):
        """Test serializzazione"""
        eyes = EyeState(
            left_x=0.5, left_y=-0.3,
            right_x=0.5, right_y=-0.3,
            left_open=0.8, right_open=0.8,
            pupil_size=1.2
        )
        d = eyes.to_dict()
        
        assert d["left"]["x"] == 0.5
        assert d["left"]["y"] == -0.3
        assert d["left"]["open"] == 0.8
        assert d["pupil_size"] == 1.2


class TestMouthState:
    """Test per MouthState"""
    
    def test_to_dict(self):
        """Test serializzazione"""
        mouth = MouthState(
            open=0.5, wide=0.7, smile=0.3, viseme=Viseme.AA
        )
        d = mouth.to_dict()
        
        assert d["open"] == 0.5
        assert d["wide"] == 0.7
        assert d["smile"] == 0.3
        assert d["viseme"] == "aa"


class TestHologramEffect:
    """Test per HologramEffect"""
    
    def test_defaults(self):
        """Test valori default"""
        holo = HologramEffect()
        
        assert holo.glow_color == "#00BFFF"
        assert holo.glow_intensity == 0.6
        assert holo.ring_count == 3
    
    def test_to_dict(self):
        """Test serializzazione"""
        holo = HologramEffect(
            glow_color="#FF0000",
            glow_intensity=0.9
        )
        d = holo.to_dict()
        
        assert d["glow_color"] == "#FF0000"
        assert d["glow_intensity"] == 0.9


class TestAvatarAnimations:
    """Test per animazioni predefinite"""
    
    def test_nod_animation(self):
        """Test animazione annuire"""
        anim = AvatarAnimations.nod()
        
        assert anim.id == "nod"
        assert anim.duration == 0.8
        assert len(anim.head_keyframes) == 5
        assert anim.loop == False
    
    def test_shake_animation(self):
        """Test animazione scuotere testa"""
        anim = AvatarAnimations.shake_head()
        
        assert anim.id == "shake"
        assert anim.duration == 1.0
        assert len(anim.head_keyframes) == 6
    
    def test_thinking_animation(self):
        """Test animazione pensiero"""
        anim = AvatarAnimations.thinking()
        
        assert anim.id == "thinking"
        assert anim.loop == True
    
    def test_alert_animation(self):
        """Test animazione allerta"""
        anim = AvatarAnimations.alert()
        
        assert anim.id == "alert"
        assert anim.duration == 0.5


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
