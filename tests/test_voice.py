"""
🗣️ Test Natural Voice System - GIDEON 3.0

Test per:
- NaturalVoiceEngine
- ResponseComposer
- ConversationManager
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.voice import (
    NaturalVoiceEngine,
    ResponseComposer,
    ConversationManager,
    VoiceEmotion,
    VoiceProfile,
    ReasoningContext,
    PauseType,
    IntonationPattern
)


class TestNaturalVoiceEngine:
    """Test NaturalVoiceEngine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = NaturalVoiceEngine()
        assert engine is not None
        assert engine.profile is not None
        assert engine.profile.name == "Gideon"
    
    def test_process_simple_text(self):
        """Test processing simple text"""
        engine = NaturalVoiceEngine()
        output = engine.process_text("Ciao, come stai?")
        
        assert output is not None
        assert len(output.segments) > 0
        assert output.word_count == 3
        assert output.sentence_count == 1
    
    def test_process_with_emotion(self):
        """Test processing with specific emotion"""
        engine = NaturalVoiceEngine()
        
        output = engine.process_text("Questo è urgente!", VoiceEmotion.URGENT)
        
        assert output.emotion == VoiceEmotion.URGENT
        # Urgent should have higher rate
        assert any(s.rate_multiplier > 1.0 for s in output.segments)
    
    def test_emotion_detection(self):
        """Test automatic emotion detection"""
        engine = NaturalVoiceEngine()
        
        # Question = curious
        output1 = engine.process_text("Come funziona?")
        assert output1.emotion == VoiceEmotion.CURIOUS
        
        # Urgent words
        output2 = engine.process_text("Urgente! Devi agire subito!")
        assert output2.emotion == VoiceEmotion.URGENT
    
    def test_pause_insertion(self):
        """Test pause insertion at punctuation"""
        engine = NaturalVoiceEngine()
        output = engine.process_text("Prima frase. Seconda frase.")
        
        # Should have pause after first sentence
        assert output.segments[0].pause_after > 0
    
    def test_intonation_question(self):
        """Test question intonation"""
        engine = NaturalVoiceEngine()
        output = engine.process_text("Sei sicuro?")
        
        assert output.segments[0].intonation == IntonationPattern.QUESTION
    
    def test_intonation_exclamation(self):
        """Test exclamation intonation"""
        engine = NaturalVoiceEngine()
        output = engine.process_text("Fantastico!")
        
        assert output.segments[0].intonation == IntonationPattern.EXCLAMATION
    
    def test_emphasis_detection(self):
        """Test emphasis on important words"""
        engine = NaturalVoiceEngine()
        output = engine.process_text("Questo è IMPORTANTE!")
        
        # Should have emphasis
        assert any(s.emphasis for s in output.segments)
    
    def test_ssml_generation(self):
        """Test SSML output generation"""
        engine = NaturalVoiceEngine()
        output = engine.process_text("Test SSML.")
        
        ssml = output.get_ssml()
        
        assert ssml.startswith('<speak>')
        assert ssml.endswith('</speak>')
        assert '<prosody' in ssml
    
    def test_duration_estimate(self):
        """Test duration estimation"""
        engine = NaturalVoiceEngine()
        
        short = engine.process_text("Ciao.")
        long = engine.process_text("Questa è una frase molto più lunga con molte parole.")
        
        assert long.total_duration_estimate > short.total_duration_estimate
    
    def test_custom_profile(self):
        """Test with custom voice profile"""
        profile = VoiceProfile(
            name="Custom",
            base_rate=200,
            base_pitch=2,
            base_volume=0.8
        )
        
        engine = NaturalVoiceEngine(profile)
        assert engine.profile.base_rate == 200


class TestResponseComposer:
    """Test ResponseComposer"""
    
    def test_initialization(self):
        """Test composer initialization"""
        composer = ResponseComposer()
        assert composer is not None
        assert composer.voice is not None
    
    def test_compose_simple(self):
        """Test simple response composition"""
        composer = ResponseComposer()
        
        response = composer.compose_simple("Ciao!")
        
        assert response.text == "Ciao!"
        assert response.is_complete
        assert response.speech is not None
    
    def test_compose_with_reasoning(self):
        """Test composition with reasoning context"""
        composer = ResponseComposer()
        
        context = ReasoningContext(
            query="Qual è il tempo?",
            reasoning_steps=[
                "Verifico la posizione",
                "Accedo ai dati meteo",
                "Analizzo le previsioni"
            ],
            confidence=0.85
        )
        
        response = composer.compose(context, "Oggi sarà soleggiato.")
        
        assert "soleggiato" in response.text
        assert response.confidence == 0.85
        assert response.is_complete
    
    def test_emotion_from_confidence(self):
        """Test emotion selection based on confidence"""
        composer = ResponseComposer()
        
        # High confidence = confident
        ctx_high = ReasoningContext(query="test", confidence=0.95)
        resp_high = composer.compose(ctx_high, "Risposta sicura.")
        assert resp_high.emotion == VoiceEmotion.CONFIDENT
        
        # Low confidence = concerned
        ctx_low = ReasoningContext(query="test", confidence=0.4)
        resp_low = composer.compose(ctx_low, "Non sono sicuro.")
        assert resp_low.emotion == VoiceEmotion.CONCERNED
    
    def test_follow_up_suggestion(self):
        """Test follow-up suggestion generation"""
        composer = ResponseComposer()
        
        context = ReasoningContext(
            query="test",
            confidence=0.5  # Low confidence should trigger follow-up
        )
        
        response = composer.compose(context, "Risposta.")
        
        # Low confidence should suggest follow-up
        assert response.follow_up_suggested is not None


class TestConversationManager:
    """Test ConversationManager"""
    
    def test_initialization(self):
        """Test manager initialization"""
        manager = ConversationManager()
        
        assert manager is not None
        assert len(manager.history) == 0
        assert not manager.is_processing
    
    def test_add_user_input(self):
        """Test adding user input"""
        manager = ConversationManager()
        
        manager.add_user_input("Ciao Gideon")
        
        assert len(manager.history) == 1
        assert manager.history[0].role == "user"
        assert manager.history[0].text == "Ciao Gideon"
    
    def test_quick_response(self):
        """Test quick response without reasoning"""
        manager = ConversationManager()
        
        response = manager.quick_response("Ciao!")
        
        assert response.text == "Ciao!"
        assert len(manager.history) == 1
        assert manager.history[0].role == "assistant"
    
    def test_reasoning_flow(self):
        """Test complete reasoning flow"""
        manager = ConversationManager()
        
        # Start reasoning
        ctx = manager.start_reasoning("Che ore sono?")
        assert manager.is_processing
        
        # Add steps
        manager.add_reasoning_step("Verifico il fuso orario")
        manager.add_reasoning_step("Ottengo l'ora corrente")
        
        assert len(ctx.reasoning_steps) == 2
        
        # Complete
        response = manager.complete_reasoning(
            "Sono le 15:30.",
            confidence=0.95
        )
        
        assert not manager.is_processing
        assert response.is_complete
        assert "15:30" in response.text
    
    def test_conversation_history(self):
        """Test conversation history tracking"""
        manager = ConversationManager()
        
        manager.add_user_input("Prima domanda")
        manager.quick_response("Prima risposta")
        manager.add_user_input("Seconda domanda")
        manager.quick_response("Seconda risposta")
        
        assert len(manager.history) == 4
    
    def test_history_limit(self):
        """Test history is limited"""
        manager = ConversationManager()
        manager.max_history = 5
        
        for i in range(10):
            manager.add_user_input(f"Messaggio {i}")
        
        assert len(manager.history) <= 5
    
    def test_get_last_user_query(self):
        """Test getting last user query"""
        manager = ConversationManager()
        
        manager.add_user_input("Prima")
        manager.quick_response("Risposta")
        manager.add_user_input("Seconda")
        
        assert manager.get_last_user_query() == "Seconda"
    
    def test_clear_history(self):
        """Test clearing history"""
        manager = ConversationManager()
        
        manager.add_user_input("Test")
        manager.quick_response("Risposta")
        
        manager.clear_history()
        
        assert len(manager.history) == 0
        assert not manager.is_processing
    
    def test_context_summary(self):
        """Test context summary generation"""
        manager = ConversationManager()
        
        manager.add_user_input("Ciao")
        manager.quick_response("Ciao! Come posso aiutarti?")
        
        summary = manager.get_context_summary()
        
        assert "Utente" in summary
        assert "Gideon" in summary
    
    def test_stats(self):
        """Test statistics"""
        manager = ConversationManager()
        
        manager.add_user_input("Test 1")
        manager.quick_response("Risposta 1")
        manager.add_user_input("Test 2")
        
        stats = manager.get_stats()
        
        assert stats["total_turns"] == 3
        assert stats["user_turns"] == 2
        assert stats["assistant_turns"] == 1
        assert not stats["is_processing"]


class TestVoiceProfile:
    """Test VoiceProfile"""
    
    def test_default_profile(self):
        """Test default profile values"""
        profile = VoiceProfile()
        
        assert profile.name == "Gideon"
        assert profile.base_rate == 175
        assert profile.base_volume == 0.9
    
    def test_emotion_settings(self):
        """Test emotion settings are initialized"""
        profile = VoiceProfile()
        
        assert VoiceEmotion.NEUTRAL.value in profile.emotion_settings
        assert VoiceEmotion.URGENT.value in profile.emotion_settings
        
        # Urgent should have higher rate
        urgent = profile.emotion_settings[VoiceEmotion.URGENT.value]
        neutral = profile.emotion_settings[VoiceEmotion.NEUTRAL.value]
        
        assert urgent["rate"] > neutral["rate"]


class TestPauseTypes:
    """Test pause timing"""
    
    def test_pause_ordering(self):
        """Test pauses are in correct order"""
        assert PauseType.NONE.value < PauseType.MICRO.value
        assert PauseType.MICRO.value < PauseType.SHORT.value
        assert PauseType.SHORT.value < PauseType.MEDIUM.value
        assert PauseType.MEDIUM.value < PauseType.LONG.value
        assert PauseType.LONG.value < PauseType.PARAGRAPH.value
        assert PauseType.PARAGRAPH.value < PauseType.DRAMATIC.value


class TestIntegration:
    """Integration tests"""
    
    def test_full_conversation_flow(self):
        """Test complete conversation flow"""
        manager = ConversationManager()
        
        # User asks
        manager.add_user_input("Qual è la capitale della Francia?")
        
        # System starts reasoning
        ctx = manager.start_reasoning("Qual è la capitale della Francia?")
        
        # Add reasoning steps
        manager.add_reasoning_step("Identifico che è una domanda di geografia")
        manager.add_reasoning_step("Cerco informazioni sulla Francia")
        manager.add_reasoning_step("Confermo la capitale")
        
        # Complete with answer
        response = manager.complete_reasoning(
            "La capitale della Francia è Parigi.",
            confidence=0.99
        )
        
        # Verify
        assert "Parigi" in response.text
        assert response.is_complete
        assert response.confidence == 0.99
        assert response.speech is not None
        assert len(response.speech.segments) > 0
        
        # History should have user + assistant
        assert len(manager.history) == 2
    
    def test_voice_output_for_tts(self):
        """Test voice output is suitable for TTS"""
        engine = NaturalVoiceEngine()
        
        text = """
        Ciao! Sono Gideon, il tuo assistente.
        Posso aiutarti con molte cose:
        - Rispondere a domande
        - Eseguire azioni
        - Analizzare dati
        
        Come posso aiutarti oggi?
        """
        
        output = engine.process_text(text)
        
        # Should have multiple segments
        assert len(output.segments) > 3
        
        # Should have pauses
        total_pause = sum(s.pause_before + s.pause_after for s in output.segments)
        assert total_pause > 0
        
        # Should have reasonable duration
        assert output.total_duration_estimate > 0
        assert output.total_duration_estimate < 60  # Less than a minute


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
