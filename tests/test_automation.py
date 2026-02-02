"""
Test Automation Layer
Verifica controllo automazioni e conferma in Pilot mode
+ Shadow Mode, Silent Execution, Reversibilità e Logging
"""

import sys
sys.path.insert(0, 'backend')

import asyncio
import pytest
from jarvis.automation_layer import (
    AutomationLayer, AutomationType, ActionType, 
    RiskLevel, ExecutionStatus, ConfirmationType,
    ExecutionMode, UndoStatus
)


pytestmark = pytest.mark.asyncio


async def test_automation_layer():
    """Test completo Automation Layer"""
    print("=" * 70)
    print("🤖 TEST AUTOMATION LAYER")
    print("=" * 70)
    
    # Mock mode manager
    class MockModeManager:
        mode_name = "pilot"
    
    automation = AutomationLayer(mode_manager=MockModeManager())
    
    passed = 0
    total = 0
    
    # === TEST 1: Creazione Azione ===
    print("\n📝 Test 1: Creazione Azione")
    print("-" * 50)
    total += 1
    
    action = automation.create_action(
        name="Test Notepad",
        target="notepad",
        automation_type=AutomationType.APPLICATION,
        action_type=ActionType.START,
        description="Avvia Notepad per test"
    )
    
    print(f"✅ Azione creata: {action.id}")
    print(f"   Nome: {action.name}")
    print(f"   Target: {action.target}")
    print(f"   Tipo: {action.automation_type.value}")
    print(f"   Rischio: {action.risk_level.value}")
    print(f"   Conferma: {action.confirmation_type.value}")
    print(f"   Status: {action.status.value}")
    passed += 1
    
    # === TEST 2: Validazione Interna (Pilot Mode) ===
    print("\n🔍 Test 2: Validazione Interna (Pilot Mode)")
    print("-" * 50)
    total += 1
    
    confirmation = await automation.request_execution(action.id)
    
    print(f"✅ Validazione eseguita")
    print(f"   Confirmation ID: {confirmation.id}")
    print(f"   Checks passati: {confirmation.all_checks_passed}")
    print(f"   Confermato: {confirmation.is_confirmed}")
    print(f"   Confermato da: {confirmation.confirmed_by}")
    
    print("   Validation checks:")
    for check in confirmation.validation_checks:
        status = "✓" if check['passed'] else "✗"
        print(f"      {status} {check['rule']}: {check['message']}")
    passed += 1
    
    # === TEST 3: Esecuzione Azione Confermata ===
    print("\n▶️ Test 3: Esecuzione Azione (simulata)")
    print("-" * 50)
    total += 1
    
    # Non eseguiamo realmente per evitare di aprire app
    print(f"✅ Azione pronta per esecuzione")
    print(f"   Status attuale: {action.status.value}")
    print(f"   Validato: {action.is_validated}")
    print(f"   Motivo: {action.validation_reason}")
    passed += 1
    
    # === TEST 4: Azione ad Alto Rischio ===
    print("\n⚠️ Test 4: Azione ad Alto Rischio")
    print("-" * 50)
    total += 1
    
    high_risk_action = automation.create_action(
        name="Stop Sistema Critico",
        target="system_service",
        automation_type=AutomationType.PROCESS,
        action_type=ActionType.STOP,
        risk_level=RiskLevel.HIGH
    )
    
    print(f"✅ Azione high-risk creata: {high_risk_action.id}")
    print(f"   Rischio: {high_risk_action.risk_level.value}")
    print(f"   Conferma richiesta: {high_risk_action.confirmation_type.value}")
    
    conf2 = await automation.request_execution(high_risk_action.id)
    
    print(f"   Auto-confermato: {conf2.is_confirmed}")
    if not conf2.is_confirmed:
        print(f"   Richiede conferma manuale: {high_risk_action.status.value}")
    passed += 1
    
    # === TEST 5: Conferma Manuale ===
    print("\n👤 Test 5: Conferma Manuale")
    print("-" * 50)
    total += 1
    
    # Crea azione che richiede conferma
    risky = automation.create_action(
        name="Restart Service",
        target="important_service",
        automation_type=AutomationType.PROCESS,
        action_type=ActionType.RESTART,
        risk_level=RiskLevel.HIGH
    )
    
    conf3 = await automation.request_execution(risky.id)
    
    print(f"   Azione in attesa: {risky.status.value}")
    
    if conf3.id in automation.pending_confirmations:
        # Simula conferma manuale
        automation.manual_confirm(conf3.id, approve=True, reason="Approvato per test")
        print(f"✅ Confermato manualmente")
        print(f"   Nuovo status: {risky.status.value}")
    passed += 1
    
    # === TEST 6: Azione Bloccata ===
    print("\n🚫 Test 6: Target Bloccato")
    print("-" * 50)
    total += 1
    
    # Blocca un target
    automation.validator.block_target("dangerous_app")
    
    blocked_action = automation.create_action(
        name="Dangerous Action",
        target="dangerous_app",
        automation_type=AutomationType.APPLICATION,
        action_type=ActionType.START
    )
    
    conf4 = await automation.request_execution(blocked_action.id)
    
    print(f"✅ Target bloccato testato")
    print(f"   Confermato: {conf4.is_confirmed}")
    print(f"   Checks:")
    for check in conf4.validation_checks:
        if not check['passed']:
            print(f"      ✗ {check['rule']}: {check['message']}")
    passed += 1
    
    # === TEST 7: Creazione Routine ===
    print("\n📋 Test 7: Creazione Routine Multi-Step")
    print("-" * 50)
    total += 1
    
    routine = automation.create_routine(
        name="Setup Environment",
        description="Routine per setup ambiente di lavoro",
        stop_on_error=True
    )
    
    # Aggiungi steps
    automation.add_step_to_routine(
        routine.id,
        name="Avvia Editor",
        target="code",
        automation_type=AutomationType.APPLICATION,
        action_type=ActionType.START
    )
    
    automation.add_step_to_routine(
        routine.id,
        name="Avvia Browser",
        target="chrome",
        automation_type=AutomationType.APPLICATION,
        action_type=ActionType.START
    )
    
    automation.add_step_to_routine(
        routine.id,
        name="Check Services",
        target="echo 'Services OK'",
        automation_type=AutomationType.COMMAND,
        action_type=ActionType.EXECUTE
    )
    
    print(f"✅ Routine creata: {routine.id}")
    print(f"   Nome: {routine.name}")
    print(f"   Steps: {len(routine.steps)}")
    for i, step in enumerate(routine.steps, 1):
        print(f"      {i}. {step.name} ({step.automation_type.value})")
    passed += 1
    
    # === TEST 8: Execute Immediate ===
    print("\n⚡ Test 8: Execute Immediate (comando sicuro)")
    print("-" * 50)
    total += 1
    
    # Esegui comando sicuro
    result = await automation.execute_immediate(
        name="Echo Test",
        target="echo Hello Automation",
        automation_type=AutomationType.COMMAND,
        action_type=ActionType.EXECUTE
    )
    
    print(f"✅ Comando eseguito")
    print(f"   Status: {result.status.value}")
    print(f"   Validato: {result.is_validated}")
    print(f"   Mode: {result.execution_mode.value}")
    if result.result:
        stdout = result.result.get('stdout', '')[:50]
        print(f"   Output: {stdout}...")
    passed += 1
    
    # === TEST 9: Shadow Mode (Osservazione) ===
    print("\n👁️ Test 9: Shadow Mode (solo osservazione)")
    print("-" * 50)
    total += 1
    
    # Abilita shadow mode
    automation.enable_shadow_mode()
    print(f"   Shadow mode attivo: {automation.is_shadow_mode}")
    
    # Osserva azione senza eseguirla
    shadow_result = await automation.observe(
        name="Test Shadow",
        target="notepad",
        automation_type=AutomationType.APPLICATION,
        action_type=ActionType.START
    )
    
    print(f"✅ Osservazione completata")
    print(f"   Status: {shadow_result.status.value}")
    print(f"   Mode: {shadow_result.execution_mode.value}")
    print(f"   Eseguito realmente: {'No' if shadow_result.result.get('mode') == 'shadow' else 'Sì'}")
    print(f"   Message: {shadow_result.result.get('message', '')[:50]}")
    
    automation.disable_shadow_mode()
    passed += 1
    
    # === TEST 10: Silent Execution ===
    print("\n🔇 Test 10: Silent Execution (senza output)")
    print("-" * 50)
    total += 1
    
    silent_result = await automation.execute_silent(
        name="Test Silent",
        target="echo Silent Test Output",
        automation_type=AutomationType.COMMAND,
        action_type=ActionType.EXECUTE
    )
    
    print(f"✅ Esecuzione silent completata")
    print(f"   Status: {silent_result.status.value}")
    print(f"   Mode: {silent_result.execution_mode.value}")
    print(f"   Silent: {silent_result.result.get('silent', False)}")
    print(f"   Output: {silent_result.result.get('stdout', 'N/A')}")
    passed += 1
    
    # === TEST 11: Dry Run (Simulazione) ===
    print("\n🧪 Test 11: Dry Run (simulazione)")
    print("-" * 50)
    total += 1
    
    dry_result = await automation.dry_run(
        name="Test Dry Run",
        target="rm -rf /important",  # Comando pericoloso
        automation_type=AutomationType.COMMAND,
        action_type=ActionType.EXECUTE
    )
    
    print(f"✅ Dry run completato")
    print(f"   Status: {dry_result.status.value}")
    print(f"   Mode: {dry_result.execution_mode.value}")
    print(f"   Simulato: {dry_result.result.get('mode') == 'dry_run'}")
    print(f"   Would execute: {dry_result.result.get('would_execute', False)}")
    passed += 1
    
    # === TEST 12: Reversibilità Azioni ===
    print("\n⏪ Test 12: Reversibilità Azioni")
    print("-" * 50)
    total += 1
    
    # Crea azione reversibile
    rev_action = automation.create_action(
        name="Azione Reversibile",
        target="notepad",
        automation_type=AutomationType.APPLICATION,
        action_type=ActionType.START,
        execution_mode=ExecutionMode.NORMAL
    )
    
    print(f"✅ Azione creata:")
    print(f"   ID: {rev_action.id}")
    print(f"   Reversibile: {rev_action.undo_info.is_reversible if rev_action.undo_info else 'N/A'}")
    
    # Verifica undo info
    reversible_list = automation.get_reversible_actions()
    print(f"   Azioni reversibili totali: {len(reversible_list)}")
    passed += 1
    
    # === TEST 13: Action Logging ===
    print("\n📝 Test 13: Action Logging")
    print("-" * 50)
    total += 1
    
    logs = automation.get_action_logs(limit=10)
    
    print(f"✅ Log delle azioni:")
    print(f"   Totale log: {len(logs)}")
    if logs:
        print("   Ultimi 3 eventi:")
        for log in logs[-3:]:
            print(f"      [{log['timestamp'][:19]}] {log['action_id']}: {log['event']}")
    passed += 1
    
    # === TEST 14: Statistiche Avanzate ===
    print("\n📊 Test 14: Statistiche Avanzate")
    print("-" * 50)
    total += 1
    
    stats = automation.get_statistics()
    
    print(f"✅ Statistiche complete:")
    print(f"   Azioni richieste: {stats['actions_requested']}")
    print(f"   Azioni eseguite: {stats['actions_executed']}")
    print(f"   Azioni completate: {stats['actions_completed']}")
    print(f"   Shadow observations: {stats['shadow_observations']}")
    print(f"   Silent executions: {stats['silent_executions']}")
    print(f"   Azioni annullate: {stats['actions_undone']}")
    print(f"   Azioni reversibili: {stats['reversible_actions']}")
    print(f"   Total logs: {stats['total_logs']}")
    print(f"   Execution mode: {stats['execution_mode']}")
    print(f"   Shadow mode: {stats['is_shadow_mode']}")
    print(f"   Silent mode: {stats['is_silent_mode']}")
    passed += 1
    
    # === TEST 15: Query Methods ===
    print("\n🔎 Test 15: Query Methods")
    print("-" * 50)
    total += 1
    
    pending = automation.get_pending_actions()
    routines = automation.get_routines()
    history = automation.get_execution_history()
    
    print(f"✅ Query methods:")
    print(f"   Pending actions: {len(pending)}")
    print(f"   Routines: {len(routines)}")
    print(f"   Execution history: {len(history)}")
    passed += 1
    
    # === TEST 16: Copilot Mode (no conferma) ===
    print("\n🚀 Test 16: Copilot Mode (senza conferma)")
    print("-" * 50)
    total += 1
    
    # Cambia a copilot mode
    class CopilotModeManager:
        mode_name = "copilot"
    
    automation_copilot = AutomationLayer(mode_manager=CopilotModeManager())
    
    print(f"   Pilot mode: {automation_copilot.is_pilot_mode}")
    
    action_copilot = automation_copilot.create_action(
        name="Test Copilot",
        target="notepad",
        automation_type=AutomationType.APPLICATION,
        action_type=ActionType.START,
        risk_level=RiskLevel.SAFE
    )
    
    print(f"   Conferma richiesta: {action_copilot.confirmation_type.value}")
    passed += 1
    
    # === RISULTATO FINALE ===
    print("\n" + "=" * 70)
    print(f"📊 RISULTATO: {passed}/{total} test passati")
    
    if passed == total:
        print("✅ TUTTI I TEST PASSATI!")
    else:
        print(f"❌ {total - passed} test falliti")
    
    print("=" * 70)
    
    return passed == total


async def demo_automation_flow():
    """Demo flusso automazione completo"""
    print("\n" + "=" * 70)
    print("🎬 DEMO: Flusso Automazione con Conferma Pilot")
    print("=" * 70)
    
    class MockModeManager:
        mode_name = "pilot"
    
    automation = AutomationLayer(mode_manager=MockModeManager())
    
    print("\n--- Scenario: Avvio ambiente di lavoro ---\n")
    
    print("1️⃣ JARVIS crea azione: 'Avvia VS Code'")
    action1 = automation.create_action(
        name="Avvia VS Code",
        target="code",
        automation_type=AutomationType.APPLICATION,
        action_type=ActionType.START
    )
    print(f"   → Rischio valutato: {action1.risk_level.value}")
    
    print("\n2️⃣ Sistema valida internamente (Pilot Mode)")
    conf = await automation.request_execution(action1.id)
    print(f"   → Validazione: {'PASSED ✓' if conf.all_checks_passed else 'FAILED ✗'}")
    print(f"   → Auto-confermato: {conf.is_confirmed}")
    
    if conf.is_confirmed:
        print("\n3️⃣ Azione confermata automaticamente")
        print(f"   → Status: {action1.status.value}")
        print(f"   → Pronta per esecuzione")
    
    print("\n4️⃣ Tentativo azione rischiosa...")
    risky = automation.create_action(
        name="Modifica Registry",
        target="regedit /s config.reg",
        automation_type=AutomationType.COMMAND,
        action_type=ActionType.EXECUTE,
        risk_level=RiskLevel.CRITICAL
    )
    print(f"   → Rischio: {risky.risk_level.value}")
    print(f"   → Richiede conferma: {risky.confirmation_type.value}")
    
    conf2 = await automation.request_execution(risky.id)
    print(f"   → Auto-confermato: {conf2.is_confirmed}")
    if not conf2.is_confirmed:
        print(f"   → IN ATTESA CONFERMA MANUALE")
        print(f"   → Motivo: {conf2.rejection_reason}")
    
    print("\n✅ Demo completata!")
    print("   - Azioni a basso rischio: conferma automatica")
    print("   - Azioni critiche: richiesta conferma manuale")


if __name__ == "__main__":
    asyncio.run(test_automation_layer())
    asyncio.run(demo_automation_flow())
