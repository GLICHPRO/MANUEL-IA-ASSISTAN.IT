"""
💬 Test Chat System - GIDEON 3.0

Test per:
- ChatSessionManager
- Multi-turno con contesto
- Storico completo
- Indicatore livello operativo
"""

import pytest
import sys
import os
import asyncio
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.api.chat_routes import (
    ChatSessionManager,
    ResponseGenerator,
    ChatMessage,
    ChatSession,
    MessageRole,
    MessageStatus,
    OperatingLevel
)


def run_async(coro):
    """Helper per eseguire coroutine nei test"""
    return asyncio.run(coro)


class TestChatSessionManager:
    """Test ChatSessionManager"""
    
    def test_create_session(self):
        """Test creazione sessione"""
        manager = ChatSessionManager()
        session = manager.create_session()
        
        assert session is not None
        assert session.id is not None
        assert len(session.messages) == 1  # Welcome message
        assert session.mode == OperatingLevel.COPILOT
    
    def test_create_session_with_id(self):
        """Test creazione sessione con ID specifico"""
        manager = ChatSessionManager()
        session = manager.create_session("my_session_123")
        
        assert session.id == "my_session_123"
    
    def test_get_session_creates_new(self):
        """Test get_session crea sessione se non esiste"""
        manager = ChatSessionManager()
        session = manager.get_session("new_session")
        
        assert session is not None
        assert "new_session" in manager.sessions
    
    def test_add_user_message(self):
        """Test aggiunta messaggio utente"""
        manager = ChatSessionManager()
        session = manager.create_session("test")
        
        msg = manager.add_message("test", MessageRole.USER, "Ciao!")
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Ciao!"
        assert len(session.messages) == 2  # Welcome + user msg
        assert session.turn_count == 1
    
    def test_add_assistant_message(self):
        """Test aggiunta messaggio assistente"""
        manager = ChatSessionManager()
        manager.create_session("test")
        
        msg = manager.add_message(
            "test", 
            MessageRole.ASSISTANT, 
            "Risposta",
            mode="copilot",
            confidence=0.9
        )
        
        assert msg.mode == "copilot"
        assert msg.confidence == 0.9
    
    def test_get_context_multiturno(self):
        """Test contesto multi-turno"""
        manager = ChatSessionManager()
        manager.create_session("test")
        
        # Aggiungi conversazione
        manager.add_message("test", MessageRole.USER, "Domanda 1")
        manager.add_message("test", MessageRole.ASSISTANT, "Risposta 1")
        manager.add_message("test", MessageRole.USER, "Domanda 2")
        manager.add_message("test", MessageRole.ASSISTANT, "Risposta 2")
        
        context = manager.get_context("test", limit=4)
        
        assert len(context) == 4
        assert context[-1]["content"] == "Risposta 2"
        assert context[-2]["content"] == "Domanda 2"
    
    def test_get_history(self):
        """Test storico completo"""
        manager = ChatSessionManager()
        manager.create_session("test")
        
        manager.add_message("test", MessageRole.USER, "Msg 1")
        manager.add_message("test", MessageRole.ASSISTANT, "Resp 1")
        
        history = manager.get_history("test")
        
        assert history["session_id"] == "test"
        assert history["total_count"] == 3  # Welcome + 2
        assert history["turn_count"] == 1
        assert len(history["messages"]) == 3
    
    def test_history_pagination(self):
        """Test paginazione storico"""
        manager = ChatSessionManager()
        manager.create_session("test")
        
        for i in range(10):
            manager.add_message("test", MessageRole.USER, f"Msg {i}")
        
        history1 = manager.get_history("test", limit=5, offset=0)
        history2 = manager.get_history("test", limit=5, offset=5)
        
        assert len(history1["messages"]) == 5
        assert len(history2["messages"]) == 5
    
    def test_set_mode(self):
        """Test cambio modalità"""
        manager = ChatSessionManager()
        manager.create_session("test")
        
        new_mode = manager.set_mode("test", "pilot")
        
        assert new_mode == OperatingLevel.PILOT
        
        session = manager.get_session("test")
        assert session.mode == OperatingLevel.PILOT
    
    def test_set_invalid_mode(self):
        """Test modalità invalida fallback a COPILOT"""
        manager = ChatSessionManager()
        manager.create_session("test")
        
        new_mode = manager.set_mode("test", "invalid_mode")
        
        assert new_mode == OperatingLevel.COPILOT
    
    def test_clear_session(self):
        """Test eliminazione sessione"""
        manager = ChatSessionManager()
        manager.create_session("to_delete")
        
        assert "to_delete" in manager.sessions
        
        result = manager.clear_session("to_delete")
        
        assert result is True
        assert "to_delete" not in manager.sessions
    
    def test_clear_nonexistent_session(self):
        """Test eliminazione sessione inesistente"""
        manager = ChatSessionManager()
        result = manager.clear_session("nonexistent")
        
        assert result is False
    
    def test_stats(self):
        """Test statistiche"""
        manager = ChatSessionManager()
        manager.create_session("s1")
        manager.create_session("s2")
        manager.add_message("s1", MessageRole.USER, "Test")
        manager.add_message("s1", MessageRole.USER, "Test2")
        
        stats = manager.get_stats()
        
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 2
        assert stats["total_turns"] == 2


class TestResponseGenerator:
    """Test ResponseGenerator"""
    
    def test_generate_greeting(self):
        """Test risposta saluto"""
        generator = ResponseGenerator()
        
        result = run_async(generator.generate("Ciao!", [], "copilot"))
        
        assert "Ciao" in result["response"]
        assert result["confidence"] >= 0.9
    
    def test_generate_time(self):
        """Test risposta orario"""
        generator = ResponseGenerator()
        
        result = run_async(generator.generate("Che ore sono?", [], "copilot"))
        
        assert "**" in result["response"]  # Bold time
        assert result["confidence"] == 1.0
    
    def test_generate_action_pilot(self):
        """Test azione in modalità pilot"""
        generator = ResponseGenerator()
        
        result = run_async(generator.generate("Apri Chrome", [], "pilot"))
        
        assert "eseguita" in result["response"].lower() or "aperto" in result["response"].lower()
        assert len(result["actions"]) > 0
    
    def test_generate_action_copilot(self):
        """Test azione in modalità copilot"""
        generator = ResponseGenerator()
        
        result = run_async(generator.generate("Apri Chrome", [], "copilot"))
        
        assert "conferma" in result["response"].lower() or "vuoi" in result["response"].lower()
    
    def test_generate_action_passive(self):
        """Test azione in modalità passive"""
        generator = ResponseGenerator()
        
        result = run_async(generator.generate("Apri Chrome", [], "passive"))
        
        assert "passa" in result["response"].lower() or "modalità" in result["response"].lower()
    
    def test_generate_help(self):
        """Test richiesta aiuto"""
        generator = ResponseGenerator()
        
        result = run_async(generator.generate("Cosa puoi fare?", [], "copilot"))
        
        assert "Informazioni" in result["response"] or "aiutarti" in result["response"].lower()
    
    def test_generate_with_context(self):
        """Test generazione con contesto"""
        generator = ResponseGenerator()
        
        context = [
            {"role": "user", "content": "Come si chiama la capitale d'Italia?"},
            {"role": "assistant", "content": "La capitale d'Italia è Roma."}
        ]
        
        result = run_async(generator.generate("E quella della Francia?", context, "copilot"))
        
        # Dovrebbe riconoscere contesto precedente
        assert result["confidence"] > 0


class TestChatMessage:
    """Test ChatMessage model"""
    
    def test_default_values(self):
        """Test valori default"""
        msg = ChatMessage(
            role=MessageRole.USER,
            content="Test"
        )
        
        assert msg.id is not None
        assert msg.status == MessageStatus.DELIVERED
        assert msg.timestamp is not None
    
    def test_with_metadata(self):
        """Test con metadata"""
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Risposta",
            mode="pilot",
            confidence=0.95,
            metadata={"reasoning": ["step1", "step2"]}
        )
        
        assert msg.mode == "pilot"
        assert msg.confidence == 0.95
        assert "reasoning" in msg.metadata


class TestChatSession:
    """Test ChatSession model"""
    
    def test_default_session(self):
        """Test sessione default"""
        session = ChatSession(id="test_session")
        
        assert session.mode == OperatingLevel.COPILOT
        assert session.turn_count == 0
        assert session.is_active is True
        assert len(session.messages) == 0


class TestOperatingLevel:
    """Test livelli operativi"""
    
    def test_all_levels(self):
        """Test tutti i livelli"""
        assert OperatingLevel.PASSIVE.value == "passive"
        assert OperatingLevel.COPILOT.value == "copilot"
        assert OperatingLevel.PILOT.value == "pilot"
        assert OperatingLevel.ANALYZING.value == "analyzing"
    
    def test_from_string(self):
        """Test conversione da stringa"""
        assert OperatingLevel("copilot") == OperatingLevel.COPILOT
        assert OperatingLevel("pilot") == OperatingLevel.PILOT


class TestMultiTurnConversation:
    """Test conversazione multi-turno"""
    
    def test_context_maintained(self):
        """Test mantenimento contesto"""
        manager = ChatSessionManager()
        manager.create_session("multiturno")
        
        # Turno 1
        manager.add_message("multiturno", MessageRole.USER, "Mi chiamo Marco")
        manager.add_message("multiturno", MessageRole.ASSISTANT, "Piacere Marco!")
        
        # Turno 2
        manager.add_message("multiturno", MessageRole.USER, "Come mi chiamo?")
        
        # Il contesto dovrebbe contenere il nome
        context = manager.get_context("multiturno")
        
        context_text = " ".join([c["content"] for c in context])
        assert "Marco" in context_text
    
    def test_context_window_limit(self):
        """Test limite finestra contesto"""
        manager = ChatSessionManager()
        manager.context_window = 5
        manager.create_session("limited")
        
        # Aggiungi molti messaggi
        for i in range(20):
            manager.add_message("limited", MessageRole.USER, f"Msg {i}")
        
        context = manager.get_context("limited")
        
        assert len(context) == 5  # Solo ultimi 5


class TestIntegration:
    """Test integrazione"""
    
    def test_full_conversation_flow(self):
        """Test flusso conversazione completo"""
        manager = ChatSessionManager()
        generator = ResponseGenerator()
        
        # Crea sessione
        session = manager.create_session("integration_test")
        
        # Utente invia messaggio
        manager.add_message("integration_test", MessageRole.USER, "Ciao!")
        
        # Ottieni contesto
        context = manager.get_context("integration_test")
        
        # Genera risposta
        result = run_async(generator.generate("Ciao!", context, "copilot"))
        
        # Aggiungi risposta
        manager.add_message(
            "integration_test",
            MessageRole.ASSISTANT,
            result["response"],
            mode="copilot",
            confidence=result["confidence"]
        )
        
        # Verifica storico
        history = manager.get_history("integration_test")
        
        assert history["total_count"] == 3  # Welcome + user + assistant
        assert history["turn_count"] == 1
    
    def test_mode_affects_response(self):
        """Test modalità influenza risposta"""
        generator = ResponseGenerator()
        
        # Stessa richiesta, modalità diverse
        pilot_result = run_async(generator.generate("Apri Chrome", [], "pilot"))
        copilot_result = run_async(generator.generate("Apri Chrome", [], "copilot"))
        passive_result = run_async(generator.generate("Apri Chrome", [], "passive"))
        
        # Le risposte devono essere diverse
        assert pilot_result["response"] != copilot_result["response"]
        assert copilot_result["response"] != passive_result["response"]
        
        # Pilot esegue, copilot chiede conferma
        assert "eseguita" in pilot_result["response"].lower() or "aperto" in pilot_result["response"].lower()
        assert "conferma" in copilot_result["response"].lower() or "vuoi" in copilot_result["response"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
