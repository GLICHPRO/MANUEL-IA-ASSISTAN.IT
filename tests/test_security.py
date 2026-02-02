"""
🔐 Test Security System - GIDEON 3.0

Test per:
- Kill Switch Manager
- Rollback Manager
- Permission Manager
- Security System integrato
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.security import (
    SecuritySystem,
    KillSwitchManager,
    RollbackManager,
    PermissionManager,
    SecurityLevel,
    UserRole,
    ActionCategory,
    RiskLevel,
    KillSwitchState,
    RollbackTrigger,
    Permission,
    User,
    RollbackPoint,
    create_security_system
)


def run_async(coro):
    """Helper per eseguire coroutine nei test"""
    return asyncio.run(coro)


# ============ KILL SWITCH TESTS ============

class TestKillSwitchManager:
    """Test KillSwitchManager"""
    
    def test_initialization(self):
        """Test inizializzazione"""
        ks = KillSwitchManager()
        
        assert ks.state == KillSwitchState.INACTIVE
        assert not ks.is_active
    
    def test_soft_kill(self):
        """Test soft kill"""
        ks = KillSwitchManager()
        
        result = run_async(ks.soft_kill("Test reason"))
        
        assert result["success"]
        assert ks.state == KillSwitchState.SOFT_KILL
        assert ks.is_active
    
    def test_hard_kill(self):
        """Test hard kill"""
        ks = KillSwitchManager()
        
        result = run_async(ks.hard_kill("Emergency"))
        
        assert result["success"]
        assert ks.state == KillSwitchState.HARD_KILL
    
    def test_emergency_kill(self):
        """Test emergency kill"""
        ks = KillSwitchManager()
        
        result = run_async(ks.emergency_kill("PANIC"))
        
        assert result["success"]
        assert result["emergency"]
        assert ks.state == KillSwitchState.EMERGENCY
    
    def test_revive(self):
        """Test revive dopo kill"""
        ks = KillSwitchManager()
        
        run_async(ks.soft_kill("Test"))
        assert ks.is_active
        
        result = run_async(ks.revive())
        
        assert result["success"]
        assert not ks.is_active
        assert ks.state == KillSwitchState.INACTIVE
    
    def test_revive_when_inactive(self):
        """Test revive quando non attivo"""
        ks = KillSwitchManager()
        
        result = run_async(ks.revive())
        
        assert not result["success"]
    
    def test_can_execute_whitelist(self):
        """Test whitelist azioni sempre permesse"""
        ks = KillSwitchManager()
        
        run_async(ks.hard_kill("Test"))
        
        # Azioni whitelist sempre permesse
        assert ks.can_execute("get_status")
        assert ks.can_execute("health_check")
        
        # Altre azioni bloccate
        assert not ks.can_execute("open_app")
    
    def test_history(self):
        """Test storico eventi"""
        ks = KillSwitchManager()
        
        run_async(ks.soft_kill("Test 1"))
        run_async(ks.revive())
        run_async(ks.hard_kill("Test 2"))
        
        history = ks.get_history()
        
        assert len(history) >= 3
    
    def test_status(self):
        """Test stato"""
        ks = KillSwitchManager()
        
        status = ks.get_status()
        
        assert "state" in status
        assert "is_active" in status
        assert "active_tasks" in status


# ============ ROLLBACK TESTS ============

class TestRollbackManager:
    """Test RollbackManager"""
    
    def test_initialization(self):
        """Test inizializzazione"""
        rm = RollbackManager()
        
        assert rm._auto_rollback_enabled
        assert rm._risk_threshold == 0.7
    
    def test_create_rollback_point(self):
        """Test creazione punto rollback"""
        rm = RollbackManager()
        
        point = rm.create_rollback_point(
            action_id="action_123",
            action_type="open_app",
            original_state={"app": "Chrome"},
            rollback_command="close_app",
            rollback_params={"app": "Chrome"}
        )
        
        assert point.id is not None
        assert point.action_id == "action_123"
        assert point.is_reversible
    
    def test_rollback_point(self):
        """Test rollback di un punto"""
        rm = RollbackManager()
        
        point = rm.create_rollback_point(
            action_id="action_456",
            action_type="test_action",
            original_state={},
            rollback_command="undo_test"
        )
        
        result = run_async(rm.rollback(point.id))
        
        assert result["success"]
        assert result["point_id"] == point.id
    
    def test_rollback_nonexistent(self):
        """Test rollback punto inesistente"""
        rm = RollbackManager()
        
        result = run_async(rm.rollback("nonexistent_id"))
        
        assert not result["success"]
    
    def test_rollback_last(self):
        """Test rollback ultime N azioni"""
        rm = RollbackManager()
        
        # Crea 5 punti
        for i in range(5):
            rm.create_rollback_point(
                action_id=f"action_{i}",
                action_type="test",
                original_state={},
                rollback_command="undo"
            )
        
        results = run_async(rm.rollback_last(3))
        
        assert len(results) == 3
    
    def test_auto_rollback_on_error(self):
        """Test rollback automatico su errore"""
        rm = RollbackManager()
        
        point = rm.create_rollback_point(
            action_id="error_action",
            action_type="test",
            original_state={},
            rollback_command="undo"
        )
        
        result = run_async(rm.auto_rollback_on_error(
            "error_action",
            Exception("Test error")
        ))
        
        assert result is not None
        assert result["success"]
    
    def test_auto_rollback_disabled(self):
        """Test auto-rollback disabilitato"""
        rm = RollbackManager()
        rm.enable_auto_rollback(False)
        
        rm.create_rollback_point(
            action_id="action_test",
            action_type="test",
            original_state={},
            rollback_command="undo"
        )
        
        result = run_async(rm.auto_rollback_on_error(
            "action_test",
            Exception("Error")
        ))
        
        assert result is None
    
    def test_risk_threshold_rollback(self):
        """Test rollback su soglia rischio"""
        rm = RollbackManager()
        rm.set_risk_threshold(0.5)
        
        point = rm.create_rollback_point(
            action_id="risky_action",
            action_type="test",
            original_state={},
            rollback_command="undo"
        )
        
        # Rischio sotto soglia
        result_low = run_async(rm.auto_rollback_on_risk("risky_action", 0.3))
        assert result_low is None
        
        # Rischio sopra soglia
        result_high = run_async(rm.auto_rollback_on_risk("risky_action", 0.8))
        assert result_high is not None
    
    def test_stats(self):
        """Test statistiche"""
        rm = RollbackManager()
        
        rm.create_rollback_point("a1", "test", {}, "undo")
        rm.create_rollback_point("a2", "test", {}, "undo")
        
        stats = rm.get_stats()
        
        assert stats["total_points"] == 2
        assert stats["current_points"] == 2


# ============ PERMISSION TESTS ============

class TestPermissionManager:
    """Test PermissionManager"""
    
    def test_initialization(self):
        """Test inizializzazione con permessi default"""
        pm = PermissionManager()
        
        assert len(pm._permissions) > 0
        assert len(pm._role_permissions) > 0
    
    def test_create_user(self):
        """Test creazione utente"""
        pm = PermissionManager()
        
        user = pm.create_user("testuser", UserRole.USER)
        
        assert user.username == "testuser"
        assert user.role == UserRole.USER
        assert user.is_active
    
    def test_check_permission_allowed(self):
        """Test permesso concesso"""
        pm = PermissionManager()
        user = pm.create_user("testuser", UserRole.USER)
        
        result = pm.check_permission(user.id, "get_status")
        
        assert result["allowed"]
    
    def test_check_permission_denied_role(self):
        """Test permesso negato per ruolo"""
        pm = PermissionManager()
        user = pm.create_user("guest", UserRole.GUEST)
        
        result = pm.check_permission(user.id, "execute_command")
        
        assert not result["allowed"]
    
    def test_check_permission_user_not_found(self):
        """Test utente non trovato"""
        pm = PermissionManager()
        
        result = pm.check_permission("nonexistent", "get_status")
        
        assert not result["allowed"]
    
    def test_grant_permission_override(self):
        """Test override permesso"""
        pm = PermissionManager()
        user = pm.create_user("testuser", UserRole.USER)
        
        # Normalmente non permesso
        result1 = pm.check_permission(user.id, "kill_switch")
        assert not result1["allowed"]
        
        # Grant override
        pm.grant_permission(user.id, "kill_switch")
        
        result2 = pm.check_permission(user.id, "kill_switch")
        assert result2["allowed"]
    
    def test_revoke_permission_override(self):
        """Test revoca override"""
        pm = PermissionManager()
        user = pm.create_user("testuser", UserRole.ADMIN)
        
        # Normalmente permesso per admin
        result1 = pm.check_permission(user.id, "manage_config")
        assert result1["allowed"]
        
        # Revoke
        pm.revoke_permission(user.id, "manage_config")
        
        result2 = pm.check_permission(user.id, "manage_config")
        assert not result2["allowed"]
    
    def test_lock_user(self):
        """Test blocco utente"""
        pm = PermissionManager()
        user = pm.create_user("testuser", UserRole.USER)
        
        pm.lock_user(user.id, minutes=30)
        
        assert user.is_locked()
        
        result = pm.check_permission(user.id, "get_status")
        assert not result["allowed"]
    
    def test_unlock_user(self):
        """Test sblocco utente"""
        pm = PermissionManager()
        user = pm.create_user("testuser", UserRole.USER)
        
        pm.lock_user(user.id, minutes=30)
        pm.unlock_user(user.id)
        
        assert not user.is_locked()
    
    def test_inactive_user(self):
        """Test utente inattivo"""
        pm = PermissionManager()
        user = pm.create_user("testuser", UserRole.USER)
        user.is_active = False
        
        result = pm.check_permission(user.id, "get_status")
        
        assert not result["allowed"]
    
    def test_blacklist(self):
        """Test blacklist globale"""
        pm = PermissionManager()
        user = pm.create_user("admin", UserRole.SUPERADMIN)
        
        pm.add_to_blacklist("dangerous_action")
        
        result = pm.check_permission(user.id, "dangerous_action")
        
        assert not result["allowed"]
        assert "blacklist" in result["reason"].lower()
    
    def test_user_permissions_list(self):
        """Test lista permessi utente"""
        pm = PermissionManager()
        user = pm.create_user("testuser", UserRole.USER)
        
        perms = pm.get_user_permissions(user.id)
        
        assert "get_status" in perms
        assert len(perms) > 0
    
    def test_requires_confirmation(self):
        """Test azioni che richiedono conferma"""
        pm = PermissionManager()
        user = pm.create_user("power", UserRole.POWER_USER)
        
        result = pm.check_permission(user.id, "execute_macro")
        
        assert result["allowed"]
        assert result["requires_confirmation"]
    
    def test_requires_2fa(self):
        """Test azioni che richiedono 2FA"""
        pm = PermissionManager()
        user = pm.create_user("super", UserRole.SUPERADMIN)
        
        result = pm.check_permission(user.id, "kill_switch")
        
        assert result["allowed"]
        assert result["requires_2fa"]
    
    def test_security_events(self):
        """Test eventi sicurezza"""
        pm = PermissionManager()
        user = pm.create_user("guest", UserRole.GUEST)
        
        # Tentativo negato genera evento
        pm.check_permission(user.id, "execute_command")
        
        events = pm.get_security_events()
        
        assert len(events) > 0
        assert events[-1]["event_type"] == "permission_denied"


# ============ SECURITY SYSTEM TESTS ============

class TestSecuritySystem:
    """Test SecuritySystem integrato"""
    
    def test_initialization(self):
        """Test inizializzazione"""
        security = create_security_system()
        
        assert security.kill_switch is not None
        assert security.rollback is not None
        assert security.permissions is not None
    
    def test_authorize_action_allowed(self):
        """Test autorizzazione permessa"""
        security = create_security_system()
        user = security.permissions.create_user("testuser", UserRole.USER)
        
        result = run_async(security.authorize_action(user.id, "get_status"))
        
        assert result["authorized"]
    
    def test_authorize_action_kill_switch_active(self):
        """Test autorizzazione con kill switch attivo"""
        security = create_security_system()
        user = security.permissions.create_user("testuser", UserRole.USER)
        
        run_async(security.kill_switch.hard_kill("Test"))
        
        result = run_async(security.authorize_action(user.id, "open_app"))
        
        assert not result["authorized"]
        assert "kill switch" in result["reason"].lower()
    
    def test_authorize_action_lockdown(self):
        """Test autorizzazione in lockdown"""
        security = create_security_system()
        user = security.permissions.create_user("testuser", UserRole.USER)
        
        security.set_security_level(SecurityLevel.LOCKDOWN)
        
        result = run_async(security.authorize_action(user.id, "get_status"))
        
        assert not result["authorized"]
    
    def test_execute_with_rollback_success(self):
        """Test esecuzione con rollback - successo"""
        security = create_security_system()
        
        async def successful_action():
            return {"result": "success"}
        
        result = run_async(security.execute_with_rollback(
            action_id="test_action",
            action_type="test",
            executor=successful_action,
            rollback_command="undo_test"
        ))
        
        assert result["success"]
        assert "rollback_point" in result
    
    def test_execute_with_rollback_error(self):
        """Test esecuzione con rollback - errore"""
        security = create_security_system()
        
        async def failing_action():
            raise Exception("Test error")
        
        result = run_async(security.execute_with_rollback(
            action_id="test_action",
            action_type="test",
            executor=failing_action,
            rollback_command="undo_test"
        ))
        
        assert not result["success"]
        assert result["rollback_executed"]
    
    def test_emergency_stop(self):
        """Test emergency stop"""
        security = create_security_system()
        
        # Crea alcuni punti rollback
        security.rollback.create_rollback_point("a1", "test", {}, "undo")
        security.rollback.create_rollback_point("a2", "test", {}, "undo")
        
        result = run_async(security.emergency_stop("PANIC"))
        
        assert result["success"]
        assert security.kill_switch.is_active
        assert security._security_level == SecurityLevel.LOCKDOWN
    
    def test_set_security_level(self):
        """Test cambio livello sicurezza"""
        security = create_security_system()
        
        result = security.set_security_level(SecurityLevel.HIGH)
        
        assert result["current"] == "HIGH"
        assert security._security_level == SecurityLevel.HIGH
    
    def test_status(self):
        """Test stato completo"""
        security = create_security_system()
        
        status = security.get_status()
        
        assert "security_level" in status
        assert "kill_switch" in status
        assert "rollback" in status
        assert "permissions" in status


# ============ INTEGRATION TESTS ============

class TestSecurityIntegration:
    """Test integrazione sicurezza"""
    
    def test_full_security_flow(self):
        """Test flusso sicurezza completo"""
        security = create_security_system()
        
        # Crea utente
        user = security.permissions.create_user("operator", UserRole.POWER_USER)
        
        # Verifica permesso
        auth = run_async(security.authorize_action(user.id, "execute_workflow"))
        assert auth["authorized"]
        assert auth["requires_confirmation"]
        
        # Esegui azione con rollback
        async def workflow():
            return {"status": "completed"}
        
        result = run_async(security.execute_with_rollback(
            action_id="wf_001",
            action_type="workflow",
            executor=workflow,
            rollback_command="cancel_workflow",
            rollback_params={"workflow_id": "wf_001"}
        ))
        
        assert result["success"]
        
        # Verifica punto rollback creato
        points = security.rollback.get_points()
        assert len(points) > 0
    
    def test_attack_scenario(self):
        """Test scenario attacco"""
        security = create_security_system()
        
        # Guest prova azione admin
        guest = security.permissions.create_user("guest", UserRole.GUEST)
        
        for _ in range(5):
            result = security.permissions.check_permission(guest.id, "kill_switch")
            assert not result["allowed"]
        
        # Verifica eventi sicurezza registrati
        events = security.permissions.get_security_events()
        denied_events = [e for e in events if e["event_type"] == "permission_denied"]
        
        assert len(denied_events) >= 5
    
    def test_recovery_flow(self):
        """Test flusso recovery"""
        security = create_security_system()
        
        # Crea punti rollback
        for i in range(5):
            security.rollback.create_rollback_point(
                f"action_{i}", "test", {"state": i}, "undo"
            )
        
        # Emergency stop
        run_async(security.emergency_stop("Test emergency"))
        
        assert security.kill_switch.is_active
        assert security._security_level == SecurityLevel.LOCKDOWN
        
        # Revive (con codice per emergency)
        security.kill_switch._emergency_codes["TEST123"] = "test"
        revive_result = run_async(security.kill_switch.revive(code="TEST123"))
        
        assert revive_result["success"]
        
        # Reset livello sicurezza
        security.set_security_level(SecurityLevel.STANDARD)
        
        assert not security.kill_switch.is_active
        assert security._security_level == SecurityLevel.STANDARD


# ============ ROLE TESTS ============

class TestUserRoles:
    """Test ruoli utente"""
    
    def test_guest_permissions(self):
        """Test permessi guest"""
        pm = PermissionManager()
        user = pm.create_user("guest", UserRole.GUEST)
        
        # Solo lettura permessa
        assert pm.check_permission(user.id, "get_status")["allowed"]
        assert pm.check_permission(user.id, "get_time")["allowed"]
        
        # Azioni non permesse
        assert not pm.check_permission(user.id, "open_app")["allowed"]
        assert not pm.check_permission(user.id, "execute_command")["allowed"]
    
    def test_user_permissions(self):
        """Test permessi user"""
        pm = PermissionManager()
        user = pm.create_user("user", UserRole.USER)
        
        assert pm.check_permission(user.id, "open_app")["allowed"]
        assert pm.check_permission(user.id, "send_notification")["allowed"]
        
        assert not pm.check_permission(user.id, "execute_macro")["allowed"]
    
    def test_power_user_permissions(self):
        """Test permessi power user"""
        pm = PermissionManager()
        user = pm.create_user("power", UserRole.POWER_USER)
        
        assert pm.check_permission(user.id, "execute_macro")["allowed"]
        assert pm.check_permission(user.id, "file_write")["allowed"]
        
        assert not pm.check_permission(user.id, "manage_users")["allowed"]
    
    def test_admin_permissions(self):
        """Test permessi admin"""
        pm = PermissionManager()
        user = pm.create_user("admin", UserRole.ADMIN)
        
        assert pm.check_permission(user.id, "manage_users")["allowed"]
        assert pm.check_permission(user.id, "execute_command")["allowed"]
        
        assert not pm.check_permission(user.id, "kill_switch")["allowed"]
    
    def test_superadmin_permissions(self):
        """Test permessi superadmin"""
        pm = PermissionManager()
        user = pm.create_user("superadmin", UserRole.SUPERADMIN)
        
        # Tutto permesso
        assert pm.check_permission(user.id, "kill_switch")["allowed"]
        assert pm.check_permission(user.id, "manage_permissions")["allowed"]
        assert pm.check_permission(user.id, "system_shutdown")["allowed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
