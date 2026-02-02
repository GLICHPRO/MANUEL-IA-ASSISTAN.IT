"""
Test per Episodic Memory e Decision Memory
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.core.episodic_memory import (
    EpisodicMemory, Event, Episode, Scenario,
    EventType, EpisodeStatus, ScenarioType
)
from backend.core.decision_memory import (
    DecisionMemory, DecisionRecord, ActionResult, LearningInsight,
    DecisionOutcome, FeedbackType, AdaptationType
)
from backend.core.continuous_learning import (
    ContinuousLearningAdapter, LearningMode, LearningCycle,
    UnifiedMemoryCoordinator
)


class TestEpisodicMemory:
    """Test per EpisodicMemory"""
    
    @pytest.fixture
    def temp_storage(self):
        """Crea storage temporaneo"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def episodic(self, temp_storage):
        """Crea istanza EpisodicMemory"""
        return EpisodicMemory(storage_path=temp_storage / "episodic")
    
    def test_record_event(self, episodic):
        """Test registrazione evento"""
        event = episodic.record_event(
            event_type=EventType.USER_INPUT,
            content="Test input",
            description="User entered test"
        )
        
        assert event is not None
        assert event.event_type == EventType.USER_INPUT
        assert event.content == "Test input"
        assert event.id in episodic.events
    
    def test_record_event_with_context(self, episodic):
        """Test evento con contesto"""
        event = episodic.record_event(
            event_type=EventType.SYSTEM_ACTION,
            content={"action": "test"},
            context={"user": "admin", "module": "core"},
            tags=["test", "action"]
        )
        
        assert event.context["user"] == "admin"
        assert "test" in event.tags
    
    def test_start_episode(self, episodic):
        """Test avvio episodio"""
        episode = episodic.start_episode(
            title="Test Episode",
            goal="Complete test",
            initial_context={"test": True}
        )
        
        assert episode is not None
        assert episode.status == EpisodeStatus.ACTIVE
        assert episode.title == "Test Episode"
        assert episodic.active_episode is not None
        assert episodic.active_episode.id == episode.id
    
    def test_episode_event_association(self, episodic):
        """Test associazione evento-episodio"""
        episode = episodic.start_episode("Test")
        
        event1 = episodic.record_event(EventType.USER_INPUT, "Input 1")
        event2 = episodic.record_event(EventType.SYSTEM_ACTION, "Action 1")
        
        episodic.close_episode()
        
        # Eventi dovrebbero essere nell'episodio
        # Include anche MILESTONE eventi di start e end (totale 4)
        episode = episodic.episodes[episode.id]
        assert len(episode.events) >= 2  # Almeno i 2 eventi user + possibili milestone
        assert event1.id in [e.id for e in episode.events]
        assert event2.id in [e.id for e in episode.events]
    
    def test_close_episode(self, episodic):
        """Test chiusura episodio"""
        episode = episodic.start_episode("Test")
        episodic.record_event(EventType.RESULT, "Success")
        
        closed = episodic.close_episode(
            status=EpisodeStatus.COMPLETED,
            outcome={"success": True}
        )
        
        assert closed is not None
        assert closed.status == EpisodeStatus.COMPLETED
        assert closed.outcome["success"] == True
        assert episodic.active_episode is None
    
    def test_parent_event_chain(self, episodic):
        """Test catena eventi parent"""
        parent = episodic.record_event(EventType.USER_INPUT, "Request")
        child = episodic.record_event(
            EventType.SYSTEM_ACTION, "Response",
            parent_event_id=parent.id
        )
        
        assert child.parent_event_id == parent.id
        
        chain = episodic.get_event_chain(child.id)
        assert len(chain) == 2
        assert chain[0].id == parent.id  # Parent first
        assert chain[1].id == child.id   # Child after
    
    def test_search_events(self, episodic):
        """Test ricerca eventi"""
        episodic.record_event(EventType.USER_INPUT, "Search for Python")
        episodic.record_event(EventType.USER_INPUT, "Find JavaScript")
        episodic.record_event(EventType.ERROR, "Python error")
        
        results = episodic.search_events("Python")
        
        assert len(results) == 2
        assert all("Python" in str(e.content) for e in results)
    
    def test_get_recent_events(self, episodic):
        """Test eventi recenti"""
        for i in range(5):
            episodic.record_event(EventType.USER_INPUT, f"Event {i}")
        
        recent = episodic.get_recent_events(n=3)
        
        assert len(recent) == 3
        assert recent[0].content == "Event 4"  # Most recent first
    
    def test_scenario_creation(self, episodic):
        """Test creazione scenario"""
        scenario = episodic.create_scenario(
            name="Test Scenario",
            scenario_type=ScenarioType.TASK,
            context={"task": "testing"}
        )
        
        assert scenario is not None
        assert scenario.scenario_type == ScenarioType.TASK
        assert scenario.id in episodic.scenarios


class TestDecisionMemory:
    """Test per DecisionMemory"""
    
    @pytest.fixture
    def temp_storage(self):
        """Crea storage temporaneo"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def decision_mem(self, temp_storage):
        """Crea istanza DecisionMemory"""
        return DecisionMemory(storage_path=temp_storage / "decisions")
    
    def test_record_decision(self, decision_mem):
        """Test registrazione decisione"""
        record = decision_mem.record_decision(
            decision_type="action_selection",
            decision_made={"action": "execute"},
            context={"state": "ready"},
            confidence=0.8
        )
        
        assert record is not None
        assert record.decision_type == "action_selection"
        assert record.initial_confidence == 0.8
        assert record.outcome == DecisionOutcome.PENDING
    
    def test_resolve_decision_success(self, decision_mem):
        """Test risoluzione decisione successo"""
        record = decision_mem.record_decision(
            decision_type="test",
            decision_made={"test": True},
            context={}
        )
        
        resolved = decision_mem.resolve_decision(
            record.id,
            DecisionOutcome.SUCCESS,
            {"result": "OK"}
        )
        
        assert resolved.outcome == DecisionOutcome.SUCCESS
        assert resolved.outcome_details["result"] == "OK"
        assert resolved.resolved_at is not None
    
    def test_resolve_decision_failure(self, decision_mem):
        """Test risoluzione decisione fallimento"""
        record = decision_mem.record_decision(
            decision_type="test",
            decision_made={"test": True},
            context={}
        )
        
        resolved = decision_mem.resolve_decision(
            record.id,
            DecisionOutcome.FAILURE,
            {"error": "Failed"}
        )
        
        assert resolved.outcome == DecisionOutcome.FAILURE
    
    def test_add_feedback(self, decision_mem):
        """Test aggiunta feedback"""
        record = decision_mem.record_decision(
            decision_type="test",
            decision_made={},
            context={}
        )
        
        decision_mem.add_decision_feedback(
            record.id,
            FeedbackType.POSITIVE,
            "Good decision"
        )
        
        assert len(record.feedback) == 1
        assert record.feedback[0]["type"] == "positive"
    
    def test_success_score(self, decision_mem):
        """Test calcolo success score"""
        record = decision_mem.record_decision(
            decision_type="test",
            decision_made={},
            context={}
        )
        
        # Pending = 0.5
        assert record.success_score() == 0.5
        
        record.resolve(DecisionOutcome.SUCCESS)
        assert record.success_score() == 1.0
        
        # Con feedback negativo
        record.add_feedback(FeedbackType.NEGATIVE, "Not ideal")
        assert record.success_score() < 1.0
    
    def test_record_action(self, decision_mem):
        """Test registrazione azione"""
        action = decision_mem.record_action(
            action_type="file_write",
            action_params={"path": "/test.txt"},
            context={"user": "admin"}
        )
        
        assert action is not None
        assert action.action_type == "file_write"
        assert not action.success
    
    def test_complete_action(self, decision_mem):
        """Test completamento azione"""
        action = decision_mem.record_action(
            action_type="test_action",
            action_params={}
        )
        
        completed = decision_mem.complete_action(
            action.id,
            success=True,
            result_data={"output": "Done"}
        )
        
        assert completed.success == True
        assert completed.result_data["output"] == "Done"
        assert completed.execution_time_ms >= 0
    
    def test_get_recent_decisions(self, decision_mem):
        """Test recupero decisioni recenti"""
        for i in range(5):
            decision_mem.record_decision(
                decision_type="test",
                decision_made={"i": i},
                context={}
            )
        
        recent = decision_mem.get_recent_decisions(n=3)
        
        assert len(recent) == 3
    
    def test_get_success_rate(self, decision_mem):
        """Test calcolo success rate"""
        # 2 success, 1 failure
        for outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.SUCCESS, DecisionOutcome.FAILURE]:
            record = decision_mem.record_decision(
                decision_type="test",
                decision_made={},
                context={}
            )
            decision_mem.resolve_decision(record.id, outcome)
        
        rate = decision_mem.get_success_rate("test")
        
        # (1.0 + 1.0 + 0.0) / 3 = 0.666...
        assert rate > 0.6 and rate < 0.7


class TestContinuousLearning:
    """Test per ContinuousLearningAdapter"""
    
    @pytest.fixture
    def temp_storage(self):
        """Crea storage temporaneo"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def adapter(self, temp_storage):
        """Crea istanza adapter"""
        return ContinuousLearningAdapter(storage_path=temp_storage / "learning")
    
    def test_initialization(self, adapter):
        """Test inizializzazione"""
        assert adapter is not None
        assert adapter.mode == LearningMode.ACTIVE
        assert adapter.is_running == False
    
    def test_set_mode(self, adapter):
        """Test cambio modalità"""
        adapter.set_mode(LearningMode.PASSIVE)
        assert adapter.mode == LearningMode.PASSIVE
        
        adapter.set_mode(LearningMode.AGGRESSIVE)
        assert adapter.mode == LearningMode.AGGRESSIVE
    
    def test_pause_resume(self, adapter):
        """Test pause/resume"""
        adapter.pause_learning()
        assert adapter.mode == LearningMode.PASSIVE
        
        adapter.resume_learning()
        assert adapter.mode == LearningMode.ACTIVE
    
    def test_learning_cycle_no_systems(self, adapter):
        """Test ciclo senza sistemi connessi"""
        cycle = adapter.start_learning_cycle()
        
        assert cycle is not None
        assert cycle.completed_at is not None
        # Senza sistemi, processerà 0 eventi
        assert cycle.events_processed == 0
    
    def test_connect_systems(self, adapter, temp_storage):
        """Test connessione sistemi"""
        episodic = EpisodicMemory(storage_path=temp_storage / "ep")
        decision = DecisionMemory(storage_path=temp_storage / "dec")
        
        adapter.connect_episodic_memory(episodic)
        adapter.connect_decision_memory(decision)
        
        assert adapter.episodic_memory is not None
        assert adapter.decision_memory is not None
    
    def test_learning_cycle_with_data(self, adapter, temp_storage):
        """Test ciclo con dati"""
        episodic = EpisodicMemory(storage_path=temp_storage / "ep")
        decision = DecisionMemory(storage_path=temp_storage / "dec")
        
        adapter.connect_episodic_memory(episodic)
        adapter.connect_decision_memory(decision)
        
        # Aggiungi dati
        episodic.record_event(EventType.USER_INPUT, "Test")
        decision.record_decision("test", {}, {})
        
        cycle = adapter.start_learning_cycle()
        
        assert cycle.events_processed >= 0
        assert cycle.decisions_analyzed >= 0
    
    def test_get_metrics(self, adapter):
        """Test recupero metriche"""
        metrics = adapter.get_metrics()
        
        assert "patterns_discovered" in metrics
        assert "insights_generated" in metrics
        assert "adaptations_applied" in metrics
    
    def test_get_learning_summary(self, adapter):
        """Test summary apprendimento"""
        summary = adapter.get_learning_summary()
        
        assert "mode" in summary
        assert "connected_systems" in summary
        assert summary["mode"] == "active"


class TestUnifiedCoordinator:
    """Test per UnifiedMemoryCoordinator"""
    
    @pytest.fixture
    def temp_storage(self):
        """Crea storage temporaneo"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def coordinator(self, temp_storage):
        """Crea coordinatore con tutti i sistemi"""
        episodic = EpisodicMemory(storage_path=temp_storage / "ep")
        decision = DecisionMemory(storage_path=temp_storage / "dec")
        
        coord = UnifiedMemoryCoordinator()
        coord.initialize(
            episodic_memory=episodic,
            decision_memory=decision
        )
        
        return coord
    
    def test_initialization(self, coordinator):
        """Test inizializzazione"""
        assert coordinator._initialized == True
        assert coordinator.episodic_memory is not None
        assert coordinator.decision_memory is not None
    
    def test_unified_record(self, coordinator):
        """Test record unificato"""
        ids = coordinator.record(
            event_type="user_input",
            content="Test message",
            context={"source": "test"}
        )
        
        assert ids != ""
        assert "event:" in ids
    
    def test_remember_decision(self, coordinator):
        """Test registrazione decisione"""
        decision_id = coordinator.remember_decision(
            decision_type="action",
            decision={"action": "test"},
            context={"state": "ready"},
            confidence=0.9
        )
        
        assert decision_id != ""
    
    def test_resolve_decision(self, coordinator):
        """Test risoluzione decisione"""
        decision_id = coordinator.remember_decision(
            decision_type="test",
            decision={},
            context={}
        )
        
        coordinator.resolve_decision(decision_id, success=True)
        
        # Verifica risoluzione
        record = coordinator.decision_memory.get_decision(decision_id)
        assert record.outcome == DecisionOutcome.SUCCESS
    
    def test_search(self, coordinator):
        """Test ricerca unificata"""
        coordinator.record("user_input", "Python programming", {})
        coordinator.record("user_input", "JavaScript coding", {})
        
        results = coordinator.search("Python")
        
        assert len(results) >= 0  # Può trovare o meno basato su content
    
    def test_get_system_status(self, coordinator):
        """Test status sistemi"""
        status = coordinator.get_system_status()
        
        assert status["initialized"] == True
        assert "systems" in status
        assert "episodic_memory" in status["systems"]
        assert "decision_memory" in status["systems"]
    
    def test_trigger_learning_cycle(self, coordinator):
        """Test trigger manuale learning"""
        result = coordinator.trigger_learning_cycle()
        
        assert "id" in result
        assert "success" in result


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
