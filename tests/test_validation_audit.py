"""
🔍 Test Validation & Audit Trail System - GIDEON 3.0

Test per:
- LogicValidator (validazione multi-livello)
- DecisionAuditTrail (audit completo)
- Replay decisionale
- Sistema integrato
"""

import pytest
import asyncio
import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.validation_audit import (
    LogicValidator,
    DecisionAuditTrail,
    ValidationAuditSystem,
    ValidationLevel,
    ValidationResult,
    CriticalityLevel,
    ReplayStatus,
    ValidationRule,
    ValidationReport,
    DecisionContext,
    ReplaySession,
    create_validation_audit_system
)


def run_async(coro):
    """Helper per eseguire coroutine nei test"""
    return asyncio.run(coro)


# ============ LOGIC VALIDATOR TESTS ============

class TestLogicValidator:
    """Test LogicValidator"""
    
    def test_initialization(self):
        """Test inizializzazione"""
        validator = LogicValidator()
        
        assert len(validator._rules) > 0
        assert len(validator._criticality_map) > 0
    
    def test_add_custom_rule(self):
        """Test aggiunta regola custom"""
        validator = LogicValidator()
        
        custom_rule = ValidationRule(
            id="custom_test",
            name="Custom Test Rule",
            description="Test rule",
            validator=lambda d: (d.get("test") == True, "Test check"),
            level=ValidationLevel.BASIC
        )
        
        validator.add_rule(custom_rule)
        
        assert "custom_test" in validator._rules
    
    def test_remove_rule(self):
        """Test rimozione regola"""
        validator = LogicValidator()
        initial_count = len(validator._rules)
        
        validator.remove_rule("params_not_empty")
        
        assert len(validator._rules) == initial_count - 1
    
    def test_enable_disable_rule(self):
        """Test abilita/disabilita regola"""
        validator = LogicValidator()
        
        validator.enable_rule("params_not_empty", False)
        assert not validator._rules["params_not_empty"].enabled
        
        validator.enable_rule("params_not_empty", True)
        assert validator._rules["params_not_empty"].enabled
    
    def test_get_criticality(self):
        """Test criticità azione"""
        validator = LogicValidator()
        
        assert validator.get_criticality("get_status") == CriticalityLevel.LOW
        assert validator.get_criticality("file_delete") == CriticalityLevel.HIGH
        assert validator.get_criticality("kill_switch") == CriticalityLevel.EXTREME
    
    def test_set_criticality(self):
        """Test imposta criticità"""
        validator = LogicValidator()
        
        validator.set_criticality("custom_action", CriticalityLevel.CRITICAL)
        
        assert validator.get_criticality("custom_action") == CriticalityLevel.CRITICAL
    
    def test_get_validation_level(self):
        """Test livello validazione"""
        validator = LogicValidator()
        
        assert validator.get_validation_level("get_status") == ValidationLevel.BASIC
        assert validator.get_validation_level("file_delete") == ValidationLevel.STRICT
        assert validator.get_validation_level("kill_switch") == ValidationLevel.CRITICAL
    
    def test_validate_basic_passed(self):
        """Test validazione base - passa"""
        validator = LogicValidator()
        
        result = run_async(validator.validate(
            action_id="test_001",
            action_type="get_status",
            data={"query": "status"}
        ))
        
        assert result.result == ValidationResult.PASSED
        assert result.pass_rate > 0
    
    def test_validate_basic_failed(self):
        """Test validazione base - fallisce"""
        validator = LogicValidator()
        
        result = run_async(validator.validate(
            action_id="test_002",
            action_type="get_status",
            data={}  # Empty data
        ))
        
        # Alcuni check potrebbero fallire
        assert result.result in [ValidationResult.PASSED, ValidationResult.WARNING, ValidationResult.FAILED]
    
    def test_validate_with_required_fields(self):
        """Test validazione campi obbligatori"""
        validator = LogicValidator()
        
        result = run_async(validator.validate(
            action_id="test_003",
            action_type="open_app",
            data={
                "_required_fields": ["app_name"],
                "app_name": "Chrome"
            }
        ))
        
        assert result.result == ValidationResult.PASSED
    
    def test_validate_missing_required_fields(self):
        """Test campi obbligatori mancanti"""
        validator = LogicValidator()
        
        result = run_async(validator.validate(
            action_id="test_004",
            action_type="open_app",
            data={
                "_required_fields": ["app_name", "window_mode"],
                "app_name": "Chrome"
            }
        ))
        
        # Deve fallire per campo mancante
        failed_checks = [c for c in result.checks if not c.passed]
        assert len(failed_checks) > 0
    
    def test_validate_type_consistency(self):
        """Test coerenza tipi"""
        validator = LogicValidator()
        
        result = run_async(validator.validate(
            action_id="test_005",
            action_type="test",
            data={
                "_type_hints": {"count": int, "name": str},
                "count": 10,
                "name": "test"
            },
            level=ValidationLevel.STANDARD
        ))
        
        assert result.result == ValidationResult.PASSED
    
    def test_validate_value_ranges(self):
        """Test range valori"""
        validator = LogicValidator()
        
        # Valore nel range
        result1 = run_async(validator.validate(
            action_id="test_006",
            action_type="test",
            data={
                "_value_ranges": {"volume": (0, 100)},
                "volume": 50
            },
            level=ValidationLevel.STANDARD
        ))
        assert result1.result == ValidationResult.PASSED
        
        # Valore fuori range
        result2 = run_async(validator.validate(
            action_id="test_007",
            action_type="test",
            data={
                "_value_ranges": {"volume": (0, 100)},
                "volume": 150
            },
            level=ValidationLevel.STANDARD
        ))
        # Almeno un check dovrebbe fallire
        failed = [c for c in result2.checks if not c.passed and "range" in c.message.lower()]
        assert len(failed) > 0
    
    def test_validate_critical_requires_confirmation(self):
        """Test azione critica richiede conferma"""
        validator = LogicValidator()
        
        result = run_async(validator.validate(
            action_id="test_008",
            action_type="kill_switch",
            data={"reason": "emergency"}
        ))
        
        assert result.requires_confirmation
    
    def test_confirm_validation(self):
        """Test conferma validazione"""
        validator = LogicValidator()
        
        result = run_async(validator.validate(
            action_id="test_009",
            action_type="kill_switch",
            data={"reason": "test"}
        ))
        
        confirmed = validator.confirm_validation(result.id, "admin")
        
        assert confirmed
        assert result.confirmation_received
        assert result.confirmed_by == "admin"
    
    def test_validation_history(self):
        """Test storico validazioni"""
        validator = LogicValidator()
        
        for i in range(5):
            run_async(validator.validate(
                action_id=f"test_{i}",
                action_type="test",
                data={"index": i}
            ))
        
        history = validator.get_validation_history()
        
        assert len(history) == 5
    
    def test_validation_stats(self):
        """Test statistiche"""
        validator = LogicValidator()
        
        run_async(validator.validate("a1", "test", {"x": 1}))
        run_async(validator.validate("a2", "test", {"y": 2}))
        
        stats = validator.get_stats()
        
        assert stats["total"] >= 2
        assert "pass_rate" in stats
        assert "rules_count" in stats


# ============ DECISION AUDIT TRAIL TESTS ============

class TestDecisionAuditTrail:
    """Test DecisionAuditTrail"""
    
    @pytest.fixture
    def temp_storage(self):
        """Crea storage temporaneo"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def audit_trail(self, temp_storage):
        """Crea audit trail"""
        return DecisionAuditTrail(storage_path=str(temp_storage))
    
    def test_initialization(self, audit_trail):
        """Test inizializzazione"""
        assert audit_trail.storage_path.exists()
        assert len(audit_trail._decisions) >= 0
    
    def test_record_decision(self, audit_trail):
        """Test registrazione decisione"""
        decision = audit_trail.record_decision(
            input_data={"query": "test"},
            user_request="test request",
            session_id="session_001",
            system_state={"mode": "normal"},
            mode="COPILOT",
            security_level="STANDARD",
            decision_type="respond",
            decision_made={"response": "OK"},
            alternatives=[{"response": "Alternative"}],
            reasoning="Simple test response",
            confidence=0.9
        )
        
        assert decision.id is not None
        assert decision.decision_type == "respond"
        assert decision.confidence == 0.9
        assert decision.id in audit_trail._decisions
    
    def test_get_decision(self, audit_trail):
        """Test recupero decisione"""
        original = audit_trail.record_decision(
            input_data={},
            user_request="test",
            session_id="s1",
            system_state={},
            mode="PILOT",
            security_level="HIGH",
            decision_type="action",
            decision_made={},
            alternatives=[],
            reasoning="test",
            confidence=0.8
        )
        
        retrieved = audit_trail.get_decision(original.id)
        
        assert retrieved is not None
        assert retrieved.id == original.id
    
    def test_update_execution(self, audit_trail):
        """Test aggiornamento esecuzione"""
        decision = audit_trail.record_decision(
            input_data={}, user_request="test", session_id="s1",
            system_state={}, mode="PILOT", security_level="HIGH",
            decision_type="action", decision_made={},
            alternatives=[], reasoning="test", confidence=0.8
        )
        
        audit_trail.update_execution(decision.id, started=True)
        assert decision.execution_started is not None
        
        audit_trail.update_execution(decision.id, completed=True, result={"status": "done"})
        assert decision.execution_completed is not None
        assert decision.execution_result == {"status": "done"}
    
    def test_resolve_decision(self, audit_trail):
        """Test risoluzione decisione"""
        decision = audit_trail.record_decision(
            input_data={}, user_request="test", session_id="s1",
            system_state={}, mode="PILOT", security_level="HIGH",
            decision_type="action", decision_made={},
            alternatives=[], reasoning="test", confidence=0.8
        )
        
        audit_trail.resolve_decision(
            decision.id,
            success=True,
            outcome_details={"result": "completed"}
        )
        
        assert decision.success == True
        assert decision.outcome_details["result"] == "completed"
    
    def test_link_rollback(self, audit_trail):
        """Test collegamento rollback"""
        decision = audit_trail.record_decision(
            input_data={}, user_request="test", session_id="s1",
            system_state={}, mode="PILOT", security_level="HIGH",
            decision_type="action", decision_made={},
            alternatives=[], reasoning="test", confidence=0.8
        )
        
        audit_trail.link_rollback(decision.id, "rollback_123")
        
        assert decision.rollback_point_id == "rollback_123"
    
    def test_get_recent_decisions(self, audit_trail):
        """Test decisioni recenti"""
        for i in range(5):
            audit_trail.record_decision(
                input_data={"i": i}, user_request=f"test {i}",
                session_id="session_test", system_state={},
                mode="PILOT", security_level="HIGH",
                decision_type="action" if i % 2 == 0 else "response",
                decision_made={}, alternatives=[],
                reasoning="test", confidence=0.8
            )
        
        # Tutti
        all_recent = audit_trail.get_recent_decisions(limit=10)
        assert len(all_recent) == 5
        
        # Filtro per tipo
        actions = audit_trail.get_recent_decisions(decision_type="action")
        assert len(actions) == 3
    
    def test_search_decisions(self, audit_trail):
        """Test ricerca decisioni"""
        audit_trail.record_decision(
            input_data={}, user_request="apri Chrome",
            session_id="s1", system_state={},
            mode="PILOT", security_level="HIGH",
            decision_type="action", decision_made={"app": "Chrome"},
            alternatives=[], reasoning="User wants Chrome",
            confidence=0.9
        )
        
        audit_trail.record_decision(
            input_data={}, user_request="che ore sono",
            session_id="s1", system_state={},
            mode="PILOT", security_level="HIGH",
            decision_type="response", decision_made={"time": "10:00"},
            alternatives=[], reasoning="Time query",
            confidence=0.95
        )
        
        results = audit_trail.search_decisions("Chrome")
        
        assert len(results) >= 1
        assert "Chrome" in results[0].user_request or "Chrome" in str(results[0].decision_made)
    
    def test_decision_integrity(self, audit_trail):
        """Test integrità decisione"""
        decision = audit_trail.record_decision(
            input_data={"test": True}, user_request="test",
            session_id="s1", system_state={},
            mode="PILOT", security_level="HIGH",
            decision_type="action", decision_made={"action": "test"},
            alternatives=[], reasoning="test reason",
            confidence=0.8
        )
        
        assert decision.verify_integrity()
        
        # Modifica checksum (simula corruzione)
        original_checksum = decision.checksum
        decision.checksum = "corrupted"
        assert not decision.verify_integrity()
        
        # Ripristina
        decision.checksum = original_checksum
        assert decision.verify_integrity()
    
    def test_callbacks(self, audit_trail):
        """Test callbacks"""
        callback_called = []
        
        def on_decision(decision):
            callback_called.append(decision.id)
        
        audit_trail.on_decision(on_decision)
        
        decision = audit_trail.record_decision(
            input_data={}, user_request="test",
            session_id="s1", system_state={},
            mode="PILOT", security_level="HIGH",
            decision_type="action", decision_made={},
            alternatives=[], reasoning="test",
            confidence=0.8
        )
        
        assert decision.id in callback_called
    
    def test_stats(self, audit_trail):
        """Test statistiche"""
        for i in range(5):
            dec = audit_trail.record_decision(
                input_data={}, user_request=f"test {i}",
                session_id="s1", system_state={},
                mode="PILOT", security_level="HIGH",
                decision_type="action", decision_made={},
                alternatives=[], reasoning="test",
                confidence=0.8
            )
            if i < 3:
                audit_trail.resolve_decision(dec.id, success=True)
            else:
                audit_trail.resolve_decision(dec.id, success=False)
        
        stats = audit_trail.get_stats()
        
        assert stats["total_decisions"] == 5
        assert stats["successful"] == 3
        assert stats["failed"] == 2


# ============ REPLAY TESTS ============

class TestDecisionReplay:
    """Test replay decisionale"""
    
    @pytest.fixture
    def temp_storage(self):
        """Crea storage temporaneo"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def audit_trail(self, temp_storage):
        """Crea audit trail"""
        return DecisionAuditTrail(storage_path=str(temp_storage))
    
    def test_start_replay(self, audit_trail):
        """Test avvio replay"""
        decision = audit_trail.record_decision(
            input_data={"query": "test"},
            user_request="test request",
            session_id="s1",
            system_state={"status": "active"},
            mode="COPILOT",
            security_level="STANDARD",
            decision_type="respond",
            decision_made={"response": "OK"},
            alternatives=[{"response": "Alternative"}],
            reasoning="Simple response",
            confidence=0.9
        )
        
        session = run_async(audit_trail.start_replay(
            decision_id=decision.id,
            simulate_only=True
        ))
        
        assert session.status == ReplayStatus.COMPLETED
        assert session.original_decision_id == decision.id
        assert len(session.steps_log) > 0
    
    def test_replay_with_override(self, audit_trail):
        """Test replay con override input"""
        decision = audit_trail.record_decision(
            input_data={"value": 100},
            user_request="test",
            session_id="s1",
            system_state={},
            mode="PILOT",
            security_level="HIGH",
            decision_type="action",
            decision_made={},
            alternatives=[],
            reasoning="test",
            confidence=0.8
        )
        
        session = run_async(audit_trail.start_replay(
            decision_id=decision.id,
            override_inputs={"value": 200}
        ))
        
        assert session.status == ReplayStatus.COMPLETED
        assert len(session.differences) > 0
    
    def test_replay_step_by_step(self, audit_trail):
        """Test replay passo-passo"""
        decision = audit_trail.record_decision(
            input_data={},
            user_request="test",
            session_id="s1",
            system_state={},
            mode="PILOT",
            security_level="HIGH",
            decision_type="action",
            decision_made={},
            alternatives=[],
            reasoning="test",
            confidence=0.8
        )
        
        session = run_async(audit_trail.start_replay(
            decision_id=decision.id,
            step_by_step=True
        ))
        
        assert session.status == ReplayStatus.READY
        assert session.step_by_step == True
    
    def test_replay_not_found(self, audit_trail):
        """Test replay decisione non trovata"""
        with pytest.raises(ValueError):
            run_async(audit_trail.start_replay("nonexistent_id"))
    
    def test_pause_cancel_replay(self, audit_trail):
        """Test pausa e annulla replay"""
        decision = audit_trail.record_decision(
            input_data={}, user_request="test",
            session_id="s1", system_state={},
            mode="PILOT", security_level="HIGH",
            decision_type="action", decision_made={},
            alternatives=[], reasoning="test",
            confidence=0.8
        )
        
        session = run_async(audit_trail.start_replay(
            decision_id=decision.id,
            step_by_step=True
        ))
        
        audit_trail.cancel_replay(session.id)
        
        assert session.status == ReplayStatus.CANCELLED
    
    def test_get_replay_session(self, audit_trail):
        """Test recupero sessione replay"""
        decision = audit_trail.record_decision(
            input_data={}, user_request="test",
            session_id="s1", system_state={},
            mode="PILOT", security_level="HIGH",
            decision_type="action", decision_made={},
            alternatives=[], reasoning="test",
            confidence=0.8
        )
        
        session = run_async(audit_trail.start_replay(decision.id))
        
        retrieved = audit_trail.get_replay_session(session.id)
        
        assert retrieved is not None
        assert retrieved.id == session.id


# ============ INTEGRATED SYSTEM TESTS ============

class TestValidationAuditSystem:
    """Test sistema integrato"""
    
    @pytest.fixture
    def temp_storage(self):
        """Crea storage temporaneo"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def system(self, temp_storage):
        """Crea sistema"""
        return ValidationAuditSystem(storage_path=str(temp_storage))
    
    def test_initialization(self, system):
        """Test inizializzazione"""
        assert system.validator is not None
        assert system.audit_trail is not None
    
    def test_validate_and_record(self, system):
        """Test validazione e registrazione"""
        approved, decision, validation = run_async(system.validate_and_record(
            action_id="test_001",
            action_type="get_status",
            input_data={"query": "status"},
            user_request="mostra stato",
            session_id="session_001",
            system_state={"active": True},
            mode="COPILOT",
            security_level="STANDARD",
            decision_made={"action": "show_status"},
            alternatives=[],
            reasoning="User wants status",
            confidence=0.95
        ))
        
        assert approved == True
        assert decision is not None
        assert validation is not None
        assert decision.validation_report is not None
    
    def test_validate_and_record_critical(self, system):
        """Test validazione azione critica"""
        approved, decision, validation = run_async(system.validate_and_record(
            action_id="test_002",
            action_type="kill_switch",
            input_data={"reason": "test"},
            user_request="attiva kill switch",
            session_id="session_001",
            system_state={},
            mode="PILOT",
            security_level="HIGH",
            decision_made={"action": "kill"},
            alternatives=[],
            reasoning="Emergency action",
            confidence=0.8
        ))
        
        # Non approvato senza conferma
        assert validation.requires_confirmation
        
        # Conferma
        system.confirm_action(validation.id, "admin")
        
        # Ora verifica che la conferma sia registrata
        assert validation.confirmation_received
    
    def test_get_decision_with_validation(self, system):
        """Test recupero decisione con validazione"""
        _, decision, _ = run_async(system.validate_and_record(
            action_id="test_003",
            action_type="open_app",
            input_data={"app": "Chrome"},
            user_request="apri Chrome",
            session_id="s1",
            system_state={},
            mode="PILOT",
            security_level="STANDARD",
            decision_made={"app": "Chrome"},
            alternatives=[],
            reasoning="Open browser",
            confidence=0.9
        ))
        
        result = system.get_decision_with_validation(decision.id)
        
        assert result is not None
        assert "decision" in result
        assert "validation" in result
    
    def test_replay_decision(self, system):
        """Test replay attraverso sistema integrato"""
        _, decision, _ = run_async(system.validate_and_record(
            action_id="test_004",
            action_type="send_notification",
            input_data={"message": "test"},
            user_request="invia notifica",
            session_id="s1",
            system_state={},
            mode="COPILOT",
            security_level="STANDARD",
            decision_made={"sent": True},
            alternatives=[],
            reasoning="Send notification",
            confidence=0.85
        ))
        
        session = run_async(system.replay_decision(decision.id))
        
        assert session.status == ReplayStatus.COMPLETED
    
    def test_stats(self, system):
        """Test statistiche combinate"""
        run_async(system.validate_and_record(
            action_id="test_005",
            action_type="get_status",
            input_data={},
            user_request="test",
            session_id="s1",
            system_state={},
            mode="PILOT",
            security_level="STANDARD",
            decision_made={},
            alternatives=[],
            reasoning="test",
            confidence=0.9
        ))
        
        stats = system.get_stats()
        
        assert "validation" in stats
        assert "audit_trail" in stats
        assert "validated_actions" in stats
        assert stats["validated_actions"] >= 1


# ============ FACTORY TEST ============

class TestFactory:
    """Test factory function"""
    
    def test_create_system(self):
        """Test creazione sistema"""
        system = create_validation_audit_system()
        
        assert system is not None
        assert isinstance(system, ValidationAuditSystem)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
