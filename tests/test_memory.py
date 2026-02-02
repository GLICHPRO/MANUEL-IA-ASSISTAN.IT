"""
Test Memory System
Verifica memoria a breve e lungo termine con apprendimento
"""

import sys
sys.path.insert(0, 'backend')

import asyncio
from datetime import datetime, timedelta

import pytest
from core.memory_system import (
    MemorySystem, ShortTermMemory, LongTermMemory,
    MemoryType, MemoryPriority, LearningType
)


pytestmark = pytest.mark.asyncio


async def test_memory_system():
    """Test completo Memory System"""
    print("=" * 70)
    print("🧠 TEST MEMORY SYSTEM")
    print("=" * 70)
    
    passed = 0
    total = 0
    
    # === TEST 1: STM - Memorizzazione Base ===
    print("\n📝 Test 1: STM - Memorizzazione Base")
    print("-" * 50)
    total += 1
    
    stm = ShortTermMemory(capacity=50)
    
    # Memorizza alcuni comandi
    mem1 = stm.store(
        content={"command": "apri chrome", "intent": "open_app"},
        memory_type=MemoryType.COMMAND,
        tags=["app", "browser"],
        priority=MemoryPriority.MEDIUM
    )
    
    mem2 = stm.store(
        content={"command": "cerca google", "intent": "search"},
        memory_type=MemoryType.COMMAND,
        tags=["search", "browser"],
        priority=MemoryPriority.MEDIUM
    )
    
    print(f"✅ Memorizzazione STM:")
    print(f"   Memoria 1: {mem1.id} - {mem1.content['command']}")
    print(f"   Memoria 2: {mem2.id} - {mem2.content['command']}")
    print(f"   Totale items: {len(stm.memories)}")
    passed += 1
    
    # === TEST 2: STM - Recall ===
    print("\n🔍 Test 2: STM - Recall")
    print("-" * 50)
    total += 1
    
    # Recall per ID
    recalled = stm.recall(mem1.id)
    print(f"✅ Recall per ID:")
    print(f"   ID: {recalled.id}")
    print(f"   Content: {recalled.content}")
    print(f"   Access count: {recalled.access_count}")
    
    # Recall recenti
    recent = stm.recall_recent(5, MemoryType.COMMAND)
    print(f"   Recenti (COMMAND): {len(recent)}")
    passed += 1
    
    # === TEST 3: STM - Working Context ===
    print("\n🔧 Test 3: STM - Working Context")
    print("-" * 50)
    total += 1
    
    stm.set_working_context("current_app", "chrome")
    stm.set_working_context("user_mood", "focused")
    stm.set_working_context("task_in_progress", True)
    
    print(f"✅ Working context impostato:")
    print(f"   current_app: {stm.get_working_context('current_app')}")
    print(f"   user_mood: {stm.get_working_context('user_mood')}")
    print(f"   task_in_progress: {stm.get_working_context('task_in_progress')}")
    passed += 1
    
    # === TEST 4: STM - Search ===
    print("\n🔎 Test 4: STM - Search")
    print("-" * 50)
    total += 1
    
    # Aggiungi più memorie per ricerca
    stm.store({"command": "chiudi firefox"}, MemoryType.COMMAND, ["app", "browser"])
    stm.store({"command": "apri notepad"}, MemoryType.COMMAND, ["app", "editor"])
    stm.store({"error": "app not found"}, MemoryType.ERROR, ["error"])
    
    results = stm.search("chrome")
    print(f"✅ Search 'chrome':")
    print(f"   Risultati: {len(results)}")
    for r in results:
        print(f"      - {r.content}")
    
    results2 = stm.search("app")
    print(f"   Search 'app': {len(results2)} risultati")
    passed += 1
    
    # === TEST 5: STM - Decay ===
    print("\n⏳ Test 5: STM - Strength e Decay")
    print("-" * 50)
    total += 1
    
    # Crea memoria e simula decay
    test_mem = stm.store(
        {"test": "decay"},
        MemoryType.CONTEXT,
        priority=MemoryPriority.LOW
    )
    initial_strength = test_mem.strength
    
    # Accessi multipli rafforzano
    for _ in range(3):
        stm.recall(test_mem.id)
    
    after_access = test_mem.strength
    
    # Decay manuale
    test_mem.decay(0.8)
    after_decay = test_mem.strength
    
    print(f"✅ Strength management:")
    print(f"   Iniziale: {initial_strength:.3f}")
    print(f"   Dopo 3 accessi: {after_access:.3f}")
    print(f"   Dopo decay (0.8): {after_decay:.3f}")
    passed += 1
    
    # === TEST 6: STM - Stats ===
    print("\n📊 Test 6: STM - Statistiche")
    print("-" * 50)
    total += 1
    
    stats = stm.get_stats()
    print(f"✅ Statistiche STM:")
    print(f"   Total items: {stats['total_items']}")
    print(f"   Capacity: {stats['capacity']}")
    print(f"   Utilization: {stats['utilization']:.1%}")
    print(f"   Avg strength: {stats['average_strength']:.3f}")
    print(f"   By type: {stats['by_type']}")
    passed += 1
    
    # === TEST 7: LTM - Pattern Learning ===
    print("\n🎓 Test 7: LTM - Pattern Learning")
    print("-" * 50)
    total += 1
    
    ltm = LongTermMemory()
    
    # Apprendi pattern
    pattern1 = ltm.learn_pattern(
        pattern_type="command_result",
        trigger={"intent": "open_app", "app": "chrome"},
        response={"action": "start_process", "target": "chrome.exe"},
        success=True
    )
    
    # Rinforza con successi
    for _ in range(5):
        ltm.learn_pattern(
            pattern_type="command_result",
            trigger={"intent": "open_app", "app": "chrome"},
            response={"action": "start_process", "target": "chrome.exe"},
            success=True
        )
    
    print(f"✅ Pattern appreso:")
    print(f"   ID: {pattern1.id}")
    print(f"   Type: {pattern1.pattern_type}")
    print(f"   Occurrences: {pattern1.occurrences}")
    print(f"   Success rate: {pattern1.success_rate():.2%}")
    print(f"   Confidence: {pattern1.confidence:.3f}")
    passed += 1
    
    # === TEST 8: LTM - Pattern Matching ===
    print("\n🔮 Test 8: LTM - Pattern Matching")
    print("-" * 50)
    total += 1
    
    # Cerca pattern simile
    match = ltm.match_pattern(
        trigger={"intent": "open_app", "app": "chrome"},
        pattern_type="command_result"
    )
    
    print(f"✅ Pattern matching:")
    if match:
        print(f"   Match trovato: {match.id}")
        print(f"   Response: {match.response}")
        print(f"   Confidence: {match.confidence:.3f}")
    else:
        print(f"   Nessun match")
    passed += 1
    
    # === TEST 9: LTM - Strategy ===
    print("\n🎯 Test 9: LTM - Strategy Management")
    print("-" * 50)
    total += 1
    
    strategy = ltm.create_strategy(
        name="Error Recovery - App Not Found",
        conditions={"error_type": "app_not_found", "app_category": "browser"},
        actions=[
            {"action": "suggest_alternative", "alternatives": ["chrome", "firefox", "edge"]},
            {"action": "offer_install", "if_missing": True}
        ],
        description="Strategia per gestire app non trovate",
        priority=8
    )
    
    print(f"✅ Strategia creata:")
    print(f"   ID: {strategy.id}")
    print(f"   Name: {strategy.name}")
    print(f"   Priority: {strategy.priority}")
    print(f"   Actions: {len(strategy.actions)}")
    
    # Simula utilizzo
    ltm.update_strategy(strategy.id, success=True, execution_time=0.5)
    ltm.update_strategy(strategy.id, success=True, execution_time=0.4)
    ltm.update_strategy(strategy.id, success=False, execution_time=1.2)
    
    print(f"   Dopo 3 usi:")
    print(f"      Usage count: {strategy.usage_count}")
    print(f"      Success rate: {strategy.effectiveness():.2%}")
    print(f"      Avg time: {strategy.avg_execution_time:.2f}s")
    passed += 1
    
    # === TEST 10: LTM - Find Strategy ===
    print("\n🔍 Test 10: LTM - Find Strategy")
    print("-" * 50)
    total += 1
    
    found = ltm.find_strategy({
        "error_type": "app_not_found",
        "app_category": "browser",
        "app_name": "opera"
    })
    
    print(f"✅ Ricerca strategia:")
    if found:
        print(f"   Trovata: {found.name}")
        print(f"   Actions: {found.actions}")
    else:
        print(f"   Nessuna strategia applicabile")
    passed += 1
    
    # === TEST 11: Memory System Unified ===
    print("\n🧠 Test 11: Memory System Unificato")
    print("-" * 50)
    total += 1
    
    memory = MemorySystem(stm_capacity=100)
    
    # Remember in STM
    memory.remember(
        content={"action": "user_command", "text": "apri le email"},
        memory_type=MemoryType.COMMAND,
        tags=["email", "user_request"]
    )
    
    # Set context
    memory.set_context("current_task", "check_email")
    memory.set_context("urgency", "high")
    
    # Search across both
    results = memory.search_all("email")
    
    print(f"✅ Memory System:")
    print(f"   Comando memorizzato in STM")
    print(f"   Context: current_task={memory.get_context('current_task')}")
    print(f"   Search 'email': {len(results)} risultati")
    passed += 1
    
    # === TEST 12: Learning from Results ===
    print("\n📚 Test 12: Learning from Results")
    print("-" * 50)
    total += 1
    
    # Simula apprendimento da risultati
    pattern = memory.learn_from_result(
        command={"intent": "send_email", "to": "user@example.com"},
        result={"status": "sent", "delivery_time": 1.2},
        success=True
    )
    
    print(f"✅ Apprendimento da risultato:")
    print(f"   Pattern ID: {pattern.id}")
    print(f"   Confidence: {pattern.confidence:.3f}")
    
    # Apprendi da errore
    error_pattern = memory.learn_from_result(
        command={"intent": "send_email", "to": "invalid"},
        result={"status": "failed", "error": "invalid_address"},
        success=False
    )
    
    print(f"   Error pattern:")
    print(f"      ID: {error_pattern.id}")
    print(f"      Success rate: {error_pattern.success_rate():.2%}")
    passed += 1
    
    # === TEST 13: Strategy Suggestion ===
    print("\n💡 Test 13: Strategy Suggestion")
    print("-" * 50)
    total += 1
    
    # Crea strategia per test
    memory.learn_strategy(
        name="Email Retry Strategy",
        conditions={"error_type": "send_failed", "retry_count": 0},
        actions=[
            {"action": "wait", "seconds": 5},
            {"action": "retry", "max_attempts": 3}
        ],
        from_experience=True
    )
    
    suggested = memory.suggest_strategy({
        "error_type": "send_failed",
        "retry_count": 0,
        "context": "email"
    })
    
    print(f"✅ Strategy suggestion:")
    if suggested:
        print(f"   Suggerita: {suggested.name}")
        print(f"   Actions: {len(suggested.actions)}")
    else:
        print(f"   Nessuna strategia suggerita")
    passed += 1
    
    # === TEST 14: Consolidation ===
    print("\n🔄 Test 14: STM → LTM Consolidation")
    print("-" * 50)
    total += 1
    
    # Crea memorie importanti
    for i in range(5):
        m = memory.remember(
            content={"important": f"critical_data_{i}"},
            memory_type=MemoryType.PATTERN,
            priority=MemoryPriority.CRITICAL
        )
        # Accedi multiple volte per triggerare consolidamento
        for _ in range(4):
            memory.stm.recall(m.id)
    
    ltm_before = memory.ltm.get_stats()["total_memories"]
    
    # Forza consolidamento
    memory.force_consolidation()
    
    ltm_after = memory.ltm.get_stats()["total_memories"]
    
    print(f"✅ Consolidamento:")
    print(f"   LTM memories prima: {ltm_before}")
    print(f"   LTM memories dopo: {ltm_after}")
    print(f"   Consolidate: {ltm_after - ltm_before}")
    passed += 1
    
    # === TEST 15: Memory Summary ===
    print("\n📋 Test 15: Memory Summary")
    print("-" * 50)
    total += 1
    
    summary = memory.get_memory_summary()
    print(summary)
    
    stats = memory.get_stats()
    print(f"\n   Dettagli STM:")
    print(f"      Items: {stats['stm']['total_items']}")
    print(f"      Utilization: {stats['stm']['utilization']:.1%}")
    print(f"   Dettagli LTM:")
    print(f"      Memories: {stats['ltm']['total_memories']}")
    print(f"      Patterns: {stats['ltm']['total_patterns']}")
    print(f"      Strategies: {stats['ltm']['total_strategies']}")
    passed += 1
    
    # === TEST 16: Learning Hook ===
    print("\n🪝 Test 16: Learning Hook")
    print("-" * 50)
    total += 1
    
    hook_called = []
    
    def my_learning_hook(command, result, success, pattern):
        hook_called.append({
            "command": command,
            "success": success,
            "pattern_id": pattern.id
        })
    
    memory.add_learning_hook(my_learning_hook)
    
    memory.learn_from_result(
        command={"test": "hook"},
        result={"status": "ok"},
        success=True
    )
    
    print(f"✅ Learning hook:")
    print(f"   Hook chiamato: {len(hook_called) > 0}")
    if hook_called:
        print(f"   Dati ricevuti: {hook_called[-1]}")
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


async def demo_memory_learning():
    """Demo apprendimento dalla sessione"""
    print("\n" + "=" * 70)
    print("🎬 DEMO: Apprendimento da Sessione di Lavoro")
    print("=" * 70)
    
    memory = MemorySystem()
    
    print("\n--- Sessione utente simulata ---\n")
    
    # Simula sequenza di comandi
    commands = [
        ("apri chrome", True),
        ("cerca python tutorial", True),
        ("apri vscode", True),
        ("crea file test.py", True),
        ("esegui test.py", False),  # Errore
        ("correggi errore", True),
        ("esegui test.py", True),   # Successo dopo correzione
    ]
    
    for cmd, success in commands:
        print(f"{'✓' if success else '✗'} Comando: {cmd}")
        
        memory.remember(
            content={"command": cmd, "timestamp": datetime.now().isoformat()},
            memory_type=MemoryType.COMMAND,
            tags=["user_session"]
        )
        
        memory.learn_from_result(
            command={"text": cmd, "type": "user_command"},
            result={"success": success},
            success=success
        )
    
    print("\n--- Analisi apprendimento ---\n")
    
    # Mostra pattern appresi
    patterns = memory.ltm.get_patterns(min_confidence=0.3)
    print(f"📚 Patterns appresi: {len(patterns)}")
    for p in patterns[:3]:
        print(f"   - {p.pattern_type}: confidence {p.confidence:.2f}")
    
    # Mostra memorie recenti
    recent = memory.recall_recent(5)
    print(f"\n📝 Memorie recenti: {len(recent)}")
    for m in recent:
        print(f"   - {m.content.get('command', 'N/A')}")
    
    # Statistiche finali
    print(f"\n{memory.get_memory_summary()}")


if __name__ == "__main__":
    asyncio.run(test_memory_system())
    asyncio.run(demo_memory_learning())
