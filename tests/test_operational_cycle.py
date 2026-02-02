"""
Test Suite per Operational Cycle Manager

Ciclo Operativo Completo:
1. Attivazione tramite comando vocale
2. Ascolto continuo e monitoraggio contesto
3. Intent detection (Jarvis)
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Configura pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

sys.path.insert(0, os.path.dirname(__file__))

from backend.core.operational_cycle import (
    # Enums
    CycleState,
    ActivationType,
    ListeningMode,
    ContextType,
    # Data classes
    WakeWordConfig,
    ListeningConfig,
    ContextSnapshot,
    CycleEvent,
    IntentResult,
    CycleSession,
    # Classes
    WakeWordDetector,
    ContextMonitor,
    OperationalCycleManager,
    # Factory
    create_operational_cycle
)


# ============ FIXTURES ============

@pytest.fixture
def wake_config():
    """Configurazione wake word"""
    return WakeWordConfig(
        primary="gideon",
        alternatives=["jarvis", "assistente"],
        sensitivity=0.7,
        cooldown_ms=100  # Basso per test
    )


@pytest.fixture
def wake_detector(wake_config):
    """Wake word detector"""
    return WakeWordDetector(wake_config)


@pytest.fixture
def context_monitor():
    """Context monitor"""
    return ContextMonitor()


@pytest.fixture
def cycle_manager():
    """Operational cycle manager"""
    return OperationalCycleManager()


@pytest.fixture
def mock_jarvis():
    """Mock Jarvis core"""
    jarvis = Mock()
    
    # Mock interpreter
    mock_intent = Mock()
    mock_intent.name = "greeting"
    mock_intent.confidence = 0.9
    mock_intent.entities = {}
    mock_intent.category = "general"
    
    jarvis.interpreter = Mock()
    jarvis.interpreter.interpret = Mock(return_value=mock_intent)
    
    jarvis.process = AsyncMock(return_value={"response": "Ciao! Come posso aiutarti?"})
    
    return jarvis


@pytest.fixture
def cycle_with_jarvis(mock_jarvis):
    """Cycle manager con Jarvis mock"""
    return OperationalCycleManager(jarvis_core=mock_jarvis)


# ============ ENUM TESTS ============

class TestEnums:
    """Test enums"""
    
    def test_cycle_state_values(self):
        """Verifica valori CycleState"""
        assert CycleState.DORMANT.value == "dormant"
        assert CycleState.STANDBY.value == "standby"
        assert CycleState.LISTENING.value == "listening"
        assert CycleState.PROCESSING.value == "processing"
        assert CycleState.COOLDOWN.value == "cooldown"
    
    def test_activation_type_values(self):
        """Verifica valori ActivationType"""
        assert ActivationType.WAKE_WORD.value == "wake_word"
        assert ActivationType.BUTTON.value == "button"
        assert ActivationType.CONTINUOUS.value == "continuous"
        assert ActivationType.API.value == "api"
    
    def test_listening_mode_values(self):
        """Verifica valori ListeningMode"""
        assert ListeningMode.WAKE_WORD_ONLY.value == "wake_word_only"
        assert ListeningMode.CONTINUOUS.value == "continuous"
        assert ListeningMode.PUSH_TO_TALK.value == "push_to_talk"
    
    def test_context_type_values(self):
        """Verifica valori ContextType"""
        assert ContextType.AUDIO.value == "audio"
        assert ContextType.VISUAL.value == "visual"
        assert ContextType.TEMPORAL.value == "temporal"


# ============ WAKE WORD DETECTOR TESTS ============

class TestWakeWordDetector:
    """Test WakeWordDetector"""
    
    def test_create_detector(self, wake_detector):
        """Crea detector"""
        assert wake_detector is not None
        assert wake_detector.config.primary == "gideon"
    
    def test_detect_primary_wake_word(self, wake_detector):
        """Rileva wake word primario"""
        detected, wake_word, command = wake_detector.detect("gideon")
        assert detected is True
        assert wake_word == "gideon"
        assert command is None
    
    def test_detect_alternative_wake_word(self, wake_detector):
        """Rileva wake word alternativo"""
        # Cooldown reset
        wake_detector._last_activation = None
        
        detected, wake_word, command = wake_detector.detect("jarvis")
        assert detected is True
        assert "jarvis" in wake_word.lower()
    
    def test_detect_wake_word_with_prefix(self, wake_detector):
        """Rileva wake word con prefisso"""
        wake_detector._last_activation = None
        
        detected, wake_word, command = wake_detector.detect("hey gideon")
        assert detected is True
        assert "gideon" in wake_word.lower()
    
    def test_detect_wake_word_with_command(self, wake_detector):
        """Rileva wake word con comando"""
        wake_detector._last_activation = None
        
        detected, wake_word, command = wake_detector.detect("gideon che ore sono")
        assert detected is True
        assert command == "che ore sono"
    
    def test_no_wake_word(self, wake_detector):
        """Nessun wake word rilevato"""
        wake_detector._last_activation = None
        
        detected, wake_word, command = wake_detector.detect("apri il browser")
        assert detected is False
        assert wake_word is None
        assert command is None
    
    def test_cooldown_blocks_rapid_activation(self, wake_detector):
        """Cooldown blocca attivazioni rapide"""
        # Prima attivazione
        detected1, _, _ = wake_detector.detect("gideon")
        assert detected1 is True
        
        # Seconda attivazione immediata (bloccata)
        detected2, _, _ = wake_detector.detect("gideon")
        assert detected2 is False
    
    def test_add_wake_word(self, wake_detector):
        """Aggiunge wake word"""
        wake_detector.add_wake_word("computer")
        assert "computer" in wake_detector.config.alternatives
        
        # Verifica rilevamento
        wake_detector._last_activation = None
        detected, _, _ = wake_detector.detect("computer")
        assert detected is True
    
    def test_remove_wake_word(self, wake_detector):
        """Rimuove wake word"""
        wake_detector.remove_wake_word("jarvis")
        assert "jarvis" not in [w.lower() for w in wake_detector.config.alternatives]
    
    def test_set_sensitivity(self, wake_detector):
        """Imposta sensibilità"""
        wake_detector.set_sensitivity(0.9)
        assert wake_detector.config.sensitivity == 0.9
        
        # Boundary check
        wake_detector.set_sensitivity(1.5)
        assert wake_detector.config.sensitivity == 1.0
        
        wake_detector.set_sensitivity(-0.5)
        assert wake_detector.config.sensitivity == 0.0
    
    def test_case_insensitive(self, wake_detector):
        """Case insensitive"""
        wake_detector._last_activation = None
        
        detected1, _, _ = wake_detector.detect("GIDEON")
        assert detected1 is True
        
        wake_detector._last_activation = None
        detected2, _, _ = wake_detector.detect("GiDeOn")
        assert detected2 is True
    
    def test_ehi_prefix(self, wake_detector):
        """Prefisso ehi"""
        wake_detector._last_activation = None
        
        detected, wake_word, _ = wake_detector.detect("ehi gideon")
        assert detected is True


# ============ CONTEXT MONITOR TESTS ============

class TestContextMonitor:
    """Test ContextMonitor"""
    
    def test_create_monitor(self, context_monitor):
        """Crea monitor"""
        assert context_monitor is not None
    
    def test_get_context(self, context_monitor):
        """Ottiene contesto"""
        ctx = context_monitor.get_context()
        assert isinstance(ctx, ContextSnapshot)
    
    def test_context_has_timestamp(self, context_monitor):
        """Contesto ha timestamp"""
        ctx = context_monitor.get_context()
        assert ctx.timestamp is not None
    
    def test_record_interaction(self, context_monitor):
        """Registra interazione"""
        before = context_monitor.get_context().last_interaction
        context_monitor.record_interaction()
        after = context_monitor.get_context().last_interaction
        
        assert after is not None
        if before:
            assert after >= before
    
    def test_temporal_monitor(self, context_monitor):
        """Monitor temporale"""
        result = context_monitor._monitor_temporal()
        
        assert "time_of_day" in result
        assert result["time_of_day"] in ["morning", "afternoon", "evening", "night"]
        assert "day_of_week" in result
        assert "hour" in result
    
    def test_user_monitor(self, context_monitor):
        """Monitor utente"""
        result = context_monitor._monitor_user()
        
        assert "activity" in result
        assert result["activity"] in ["active", "idle", "unknown"]
    
    def test_add_custom_monitor(self, context_monitor):
        """Aggiunge monitor custom"""
        def custom_monitor():
            return {"custom_key": "custom_value"}
        
        context_monitor.add_monitor(ContextType.SYSTEM, custom_monitor)
        assert ContextType.SYSTEM in context_monitor._monitors
    
    def test_get_context_summary(self, context_monitor):
        """Ottiene riassunto contesto"""
        summary = context_monitor.get_context_summary()
        
        assert "time_context" in summary
        assert "user_state" in summary
        assert "recent_interaction" in summary
    
    def test_context_to_dict(self, context_monitor):
        """Contesto to dict"""
        ctx = context_monitor.get_context()
        data = ctx.to_dict()
        
        assert "timestamp" in data
        assert "audio" in data
        assert "system" in data
        assert "user" in data
        assert "temporal" in data


# ============ DATA CLASS TESTS ============

class TestDataClasses:
    """Test data classes"""
    
    def test_wake_word_config_defaults(self):
        """WakeWordConfig defaults"""
        config = WakeWordConfig()
        assert config.primary == "gideon"
        assert "jarvis" in config.alternatives
        assert config.sensitivity == 0.7
    
    def test_listening_config_defaults(self):
        """ListeningConfig defaults"""
        config = ListeningConfig()
        assert config.mode == ListeningMode.WAKE_WORD_ONLY
        assert config.timeout_seconds == 10.0
        assert config.language == "it-IT"
    
    def test_intent_result(self):
        """IntentResult"""
        intent = IntentResult(
            name="test_intent",
            confidence=0.9,
            entities={"entity1": "value1"},
            text="test text"
        )
        
        assert intent.name == "test_intent"
        assert intent.confidence == 0.9
        assert intent.is_actionable is True
    
    def test_intent_result_to_dict(self):
        """IntentResult to dict"""
        intent = IntentResult(
            name="test",
            confidence=0.8,
            entities={},
            text="test"
        )
        
        data = intent.to_dict()
        assert data["name"] == "test"
        assert data["confidence"] == 0.8
    
    def test_cycle_event_to_dict(self):
        """CycleEvent to dict"""
        event = CycleEvent(
            id="evt_001",
            timestamp=datetime.now(),
            event_type="state_change",
            state_from=CycleState.STANDBY,
            state_to=CycleState.LISTENING
        )
        
        data = event.to_dict()
        assert data["id"] == "evt_001"
        assert data["state_from"] == "standby"
        assert data["state_to"] == "listening"
    
    def test_cycle_session_duration(self):
        """CycleSession duration"""
        session = CycleSession(
            id="test_session",
            started_at=datetime.now() - timedelta(seconds=30)
        )
        
        assert session.duration.total_seconds() >= 30
    
    def test_cycle_session_is_active(self):
        """CycleSession is_active"""
        session = CycleSession(id="test")
        
        # DORMANT non è attivo
        session.current_state = CycleState.DORMANT
        assert session.is_active is False
        
        # LISTENING è attivo
        session.current_state = CycleState.LISTENING
        assert session.is_active is True


# ============ OPERATIONAL CYCLE MANAGER TESTS ============

class TestOperationalCycleManager:
    """Test OperationalCycleManager"""
    
    def test_create_manager(self, cycle_manager):
        """Crea manager"""
        assert cycle_manager is not None
        assert cycle_manager.state == CycleState.DORMANT
    
    def test_initial_state(self, cycle_manager):
        """Stato iniziale"""
        assert cycle_manager.state == CycleState.DORMANT
        assert cycle_manager.is_active is False
        assert cycle_manager.is_listening is False
    
    @pytest.mark.asyncio
    async def test_start_cycle(self, cycle_manager):
        """Avvia ciclo"""
        await cycle_manager.start()
        assert cycle_manager.state == CycleState.STANDBY
        assert cycle_manager.is_active is True
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_stop_cycle(self, cycle_manager):
        """Ferma ciclo"""
        await cycle_manager.start()
        await cycle_manager.stop()
        
        assert cycle_manager.state == CycleState.DORMANT
        assert cycle_manager.is_active is False
    
    @pytest.mark.asyncio
    async def test_process_input_in_dormant(self, cycle_manager):
        """Processo input in DORMANT"""
        result = await cycle_manager.process_audio_input("gideon")
        assert result["success"] is False  # Non processa in DORMANT
    
    @pytest.mark.asyncio
    async def test_wake_word_activation(self, cycle_manager):
        """Attivazione con wake word"""
        await cycle_manager.start()
        
        result = await cycle_manager.process_audio_input("gideon")
        
        assert result["success"] is True
        assert cycle_manager.state == CycleState.LISTENING
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_wake_word_with_command(self, cycle_with_jarvis):
        """Wake word con comando inline"""
        await cycle_with_jarvis.start()
        
        result = await cycle_with_jarvis.process_audio_input("gideon che ore sono")
        
        # Dovrebbe processare direttamente il comando
        assert result.get("intent") is not None
        
        await cycle_with_jarvis.stop()
    
    @pytest.mark.asyncio
    async def test_no_wake_word_ignored(self, cycle_manager):
        """Senza wake word viene ignorato"""
        await cycle_manager.start()
        
        result = await cycle_manager.process_audio_input("apri il browser")
        
        assert result["success"] is False
        assert cycle_manager.state == CycleState.STANDBY
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_manual_activation(self, cycle_manager):
        """Attivazione manuale"""
        await cycle_manager.start()
        await cycle_manager.activate_manually(ActivationType.BUTTON)
        
        assert cycle_manager.state == CycleState.LISTENING
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_continuous_listening_mode(self, cycle_manager):
        """Modalità ascolto continuo"""
        await cycle_manager.start()
        await cycle_manager.set_continuous_listening(True)
        
        assert cycle_manager.listening_config.mode == ListeningMode.CONTINUOUS
        assert cycle_manager.state == CycleState.LISTENING
        
        await cycle_manager.set_continuous_listening(False)
        assert cycle_manager.listening_config.mode == ListeningMode.WAKE_WORD_ONLY
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_cancel_current(self, cycle_manager):
        """Annulla operazione corrente"""
        await cycle_manager.start()
        await cycle_manager.activate_manually()
        
        assert cycle_manager.state == CycleState.LISTENING
        
        await cycle_manager.cancel_current()
        
        assert cycle_manager.state == CycleState.STANDBY
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_state_transition_callback(self, cycle_manager):
        """Callback cambio stato"""
        state_changes = []
        
        def on_change(old, new, details):
            state_changes.append((old, new))
        
        cycle_manager.on_state_change(on_change)
        await cycle_manager.start()
        
        assert len(state_changes) > 0
        assert state_changes[-1][1] == CycleState.STANDBY
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_wake_detected_callback(self, cycle_manager):
        """Callback wake word rilevato"""
        wake_events = []
        
        def on_wake(wake_word, command):
            wake_events.append((wake_word, command))
        
        cycle_manager.on_wake_detected(on_wake)
        await cycle_manager.start()
        
        await cycle_manager.process_audio_input("gideon")
        
        assert len(wake_events) == 1
        assert "gideon" in wake_events[0][0].lower()
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_intent_detected_callback(self, cycle_with_jarvis):
        """Callback intent rilevato"""
        intents = []
        
        def on_intent(intent):
            intents.append(intent)
        
        cycle_with_jarvis.on_intent_detected(on_intent)
        await cycle_with_jarvis.start()
        
        await cycle_with_jarvis.process_audio_input("gideon che ore sono")
        
        assert len(intents) >= 1
        
        await cycle_with_jarvis.stop()
    
    @pytest.mark.asyncio
    async def test_register_intent_handler(self, cycle_with_jarvis):
        """Registra handler per intent"""
        handler_called = []
        
        async def greeting_handler(intent):
            handler_called.append(intent)
            return {"response": "Ciao!", "action_taken": True}
        
        cycle_with_jarvis.register_intent_handler("greeting", greeting_handler)
        
        await cycle_with_jarvis.start()
        result = await cycle_with_jarvis.process_audio_input("gideon ciao")
        
        # L'handler dovrebbe essere stato chiamato
        assert len(handler_called) >= 1
        
        await cycle_with_jarvis.stop()
    
    def test_configure_wake_word(self, cycle_manager):
        """Configura wake word"""
        cycle_manager.configure_wake_word(
            primary="computer",
            sensitivity=0.9
        )
        
        assert cycle_manager.wake_config.primary == "computer"
        assert cycle_manager.wake_config.sensitivity == 0.9
    
    def test_configure_listening(self, cycle_manager):
        """Configura ascolto"""
        cycle_manager.configure_listening(
            timeout_seconds=15.0,
            language="en-US"
        )
        
        assert cycle_manager.listening_config.timeout_seconds == 15.0
        assert cycle_manager.listening_config.language == "en-US"
    
    def test_set_cooldown_duration(self, cycle_manager):
        """Imposta durata cooldown"""
        cycle_manager.set_cooldown_duration(10.0)
        assert cycle_manager._cooldown_duration == 10.0
    
    def test_get_status(self, cycle_manager):
        """Ottiene status"""
        status = cycle_manager.get_status()
        
        assert "state" in status
        assert "is_active" in status
        assert "is_listening" in status
        assert "listening_mode" in status
        assert "wake_words" in status
    
    def test_get_stats(self, cycle_manager):
        """Ottiene statistiche"""
        stats = cycle_manager.get_stats()
        
        assert "total_activations" in stats
        assert "total_sessions" in stats
        assert "total_intents" in stats
    
    @pytest.mark.asyncio
    async def test_session_creation(self, cycle_manager):
        """Crea sessione"""
        await cycle_manager.start()
        await cycle_manager.process_audio_input("gideon")
        
        assert cycle_manager._current_session is not None
        assert cycle_manager._current_session.activation_type == ActivationType.WAKE_WORD
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_session_history(self, cycle_manager):
        """Storico sessioni"""
        await cycle_manager.start()
        
        # Prima sessione
        await cycle_manager.process_audio_input("gideon")
        await cycle_manager.cancel_current()
        
        # Seconda sessione
        cycle_manager.wake_detector._last_activation = None
        await cycle_manager.process_audio_input("gideon")
        await cycle_manager.cancel_current()
        
        history = cycle_manager.get_session_history()
        assert len(history) >= 2
        
        await cycle_manager.stop()


# ============ INTEGRATION TESTS ============

class TestIntegration:
    """Test integrazione"""
    
    @pytest.mark.asyncio
    async def test_full_cycle_with_jarvis(self, mock_jarvis):
        """Ciclo completo con Jarvis"""
        cycle = OperationalCycleManager(jarvis_core=mock_jarvis)
        
        # Avvia
        await cycle.start()
        assert cycle.state == CycleState.STANDBY
        
        # Wake word + comando
        result = await cycle.process_audio_input("gideon che ore sono")
        
        assert result["success"] is True
        assert result["intent"] is not None
        assert cycle.state == CycleState.COOLDOWN
        
        # Cleanup
        await cycle.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_turns_conversation(self, cycle_with_jarvis):
        """Conversazione multi-turno"""
        await cycle_with_jarvis.start()
        
        # Primo turno
        result1 = await cycle_with_jarvis.process_audio_input("gideon ciao")
        assert result1["success"] is True
        
        # In cooldown, accetta input senza wake word
        if cycle_with_jarvis.state == CycleState.COOLDOWN:
            result2 = await cycle_with_jarvis.process_audio_input("come stai")
            # Il secondo turno incrementa turns
        
        await cycle_with_jarvis.stop()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, cycle_manager):
        """Recupero da errore"""
        await cycle_manager.start()
        
        # Simula errore
        await cycle_manager._transition_to(CycleState.ERROR, {"error": "test error"})
        assert cycle_manager.state == CycleState.ERROR
        
        # Restart
        await cycle_manager.start()
        assert cycle_manager.state == CycleState.STANDBY
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_context_integration(self, cycle_manager):
        """Integrazione contesto"""
        await cycle_manager.start()
        
        # Registra interazione
        cycle_manager.context_monitor.record_interaction()
        
        # Il contesto dovrebbe riflettere l'interazione
        summary = cycle_manager.context_monitor.get_context_summary()
        assert summary["recent_interaction"] is True
        
        await cycle_manager.stop()


# ============ EDGE CASES ============

class TestEdgeCases:
    """Test casi limite"""
    
    def test_empty_text_detection(self, wake_detector):
        """Testo vuoto"""
        detected, _, _ = wake_detector.detect("")
        assert detected is False
    
    def test_whitespace_only(self, wake_detector):
        """Solo whitespace"""
        detected, _, _ = wake_detector.detect("   ")
        assert detected is False
    
    @pytest.mark.asyncio
    async def test_double_start(self, cycle_manager):
        """Doppio start"""
        await cycle_manager.start()
        state1 = cycle_manager.state
        
        await cycle_manager.start()
        state2 = cycle_manager.state
        
        # Dovrebbe rimanere in standby
        assert state1 == state2 == CycleState.STANDBY
        
        await cycle_manager.stop()
    
    @pytest.mark.asyncio
    async def test_double_stop(self, cycle_manager):
        """Doppio stop"""
        await cycle_manager.start()
        await cycle_manager.stop()
        await cycle_manager.stop()  # Non dovrebbe causare errori
        
        assert cycle_manager.state == CycleState.DORMANT
    
    @pytest.mark.asyncio
    async def test_process_without_jarvis(self, cycle_manager):
        """Processo senza Jarvis"""
        await cycle_manager.start()
        
        result = await cycle_manager.process_audio_input("gideon fai qualcosa")
        
        # Dovrebbe gestire gracefully l'assenza di Jarvis
        assert result is not None
        
        await cycle_manager.stop()
    
    def test_special_characters_in_wake_word(self, wake_detector):
        """Caratteri speciali"""
        wake_detector._last_activation = None
        
        detected, _, _ = wake_detector.detect("gideon! come stai?")
        assert detected is True
    
    @pytest.mark.asyncio
    async def test_rapid_commands(self, cycle_manager):
        """Comandi rapidi"""
        await cycle_manager.start()
        
        # Primo comando
        await cycle_manager.process_audio_input("gideon")
        
        # Secondo comando rapido (cooldown attivo su wake detector)
        result = await cycle_manager.process_audio_input("gideon")
        # Il secondo potrebbe essere bloccato da cooldown
        
        await cycle_manager.stop()


# ============ FACTORY TESTS ============

class TestFactory:
    """Test factory function"""
    
    def test_create_operational_cycle(self):
        """Crea ciclo operativo"""
        cycle = create_operational_cycle()
        
        assert cycle is not None
        assert isinstance(cycle, OperationalCycleManager)
    
    def test_create_with_jarvis(self, mock_jarvis):
        """Crea con Jarvis"""
        cycle = create_operational_cycle(jarvis_core=mock_jarvis)
        
        assert cycle.jarvis == mock_jarvis


# ============ MAIN ============

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
