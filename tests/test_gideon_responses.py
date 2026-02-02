"""
Test Gideon Response Generator
Verifica gli output narrativi tipici di Gideon
"""

import sys
sys.path.insert(0, 'backend')

from gideon import GideonCore, GideonResponse, ResponseType


def test_response_generator():
    """Test output narrativi Gideon"""
    print("=" * 60)
    print("🔮 TEST GIDEON RESPONSE GENERATOR")
    print("=" * 60)
    
    gideon = GideonCore()
    
    # Test 1: Scenario ottimale
    print("\n📊 Test 1: Scenario Ottimale")
    print("-" * 40)
    
    response = gideon.respond_scenario_optimal(
        probability=0.91,
        risk_level="minimal",
        scenario_name="Piano A - Lancio graduale",
        details={"factors": ["mercato favorevole", "team pronto", "budget allocato"]}
    )
    
    print(f"✅ {response}")
    print(f"   Tipo: {response.response_type.value}")
    print(f"   Confidenza: {response.confidence}")
    print(f"   Per Jarvis: {response.for_jarvis}")
    for detail in response.details:
        print(f"   → {detail}")
    
    # Test 2: Simulazione completata
    print("\n🔄 Test 2: Simulazione Completata")
    print("-" * 40)
    
    response = gideon.respond_simulation_completed(
        outcome="convergenza raggiunta",
        iterations=1000,
        confidence=0.87,
        suggestions=[
            "Ottimizzare il timing di lancio",
            "Ridurre il budget marketing iniziale"
        ]
    )
    
    print(f"✅ {response}")
    for detail in response.details:
        print(f"   → {detail}")
    
    # Test 3: Classifica pronta
    print("\n📋 Test 3: Classifica Pronta per Jarvis")
    print("-" * 40)
    
    scenarios = [
        {"name": "Strategia conservativa", "probability": 0.92},
        {"name": "Approccio aggressivo", "probability": 0.78},
        {"name": "Piano bilanciato", "probability": 0.85},
        {"name": "Espansione graduale", "probability": 0.88},
        {"name": "Focus nicchia", "probability": 0.94}
    ]
    
    response = gideon.respond_ranking_ready(scenarios, top_n=5)
    
    print(f"✅ {response}")
    for detail in response.details:
        print(f"   → {detail}")
    
    # Test 4: Valutazione rischio
    print("\n⚠️ Test 4: Valutazione Rischio")
    print("-" * 40)
    
    response = gideon.respond_risk_assessment(
        risk_level="medium",
        risk_score=0.35,
        factors=["volatilità mercato", "competitor attivo"],
        mitigations=["hedging", "piano B pronto"]
    )
    
    print(f"✅ {response}")
    for detail in response.details:
        print(f"   → {detail}")
    
    # Test 5: Previsione
    print("\n🔮 Test 5: Previsione")
    print("-" * 40)
    
    response = gideon.respond_prediction(
        metric="ROI Q2",
        value=0.23,
        confidence=0.82,
        trend="positive"
    )
    
    print(f"✅ {response}")
    
    # Test 6: Raccomandazione
    print("\n💡 Test 6: Raccomandazione")
    print("-" * 40)
    
    response = gideon.respond_recommendation(
        action="Procedere con Piano A",
        probability=0.89,
        reasoning="Miglior rapporto rischio/rendimento",
        alternatives=[
            {"action": "Piano B - Attendere Q3", "probability": 0.76},
            {"action": "Piano C - Partnership", "probability": 0.81}
        ]
    )
    
    print(f"✅ {response}")
    for detail in response.details:
        print(f"   → {detail}")
    if response.recommendations:
        print("   Alternative:")
        for alt in response.recommendations:
            print(f"     • {alt}")
    
    # Test 7: Warning
    print("\n🚨 Test 7: Warning")
    print("-" * 40)
    
    response = gideon.respond_warning(
        message="Competitor ha annunciato prodotto simile",
        severity="high",
        action_required=True
    )
    
    print(f"✅ {response}")
    
    # Test 8: Report completo
    print("\n📑 Test 8: Report Analisi Completo")
    print("-" * 40)
    
    response = gideon.respond_full_analysis(
        scenario={"name": "Lancio prodotto", "probability": 0.85},
        simulation_result={"outcome": "successo in 78% delle simulazioni"},
        risk={"level": "low", "score": 0.22},
        recommendations=["Procedere", "Monitorare KPI settimanali"]
    )
    
    print(f"✅ {response}")
    for detail in response.details:
        print(f"   → {detail}")
    
    # Test 9: Confronto scenari
    print("\n⚖️ Test 9: Confronto Scenari")
    print("-" * 40)
    
    response = gideon.respond_compare_scenarios(scenarios)
    
    print(f"✅ {response}")
    for detail in response.details:
        print(f"   → {detail}")
    
    # Test 10: Storia risposte
    print("\n📜 Test 10: Storia Risposte")
    print("-" * 40)
    
    history = gideon.get_response_history(limit=5)
    print(f"✅ Ultime {len(history)} risposte salvate")
    for h in history[-3:]:
        print(f"   → [{h['type']}] {h['message'][:50]}...")
    
    # Statistiche
    print("\n📊 Statistiche Response Generator")
    print("-" * 40)
    stats = gideon.get_statistics()
    rg_stats = stats.get("response_generator", {})
    print(f"   Risposte generate: {rg_stats.get('responses_generated', 0)}")
    print(f"   Tono default: {rg_stats.get('default_tone', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ TUTTI I TEST COMPLETATI!")
    print("=" * 60)
    
    return True


def demo_output_jarvis():
    """Demo output per Jarvis"""
    print("\n" + "=" * 60)
    print("🤖 DEMO: Output Gideon → Jarvis")
    print("=" * 60)
    
    gideon = GideonCore()
    
    print("\n💬 Esempi di output tipici Gideon:\n")
    
    # Output 1
    r = gideon.respond_scenario_optimal(0.91, "minimal")
    print(f"GIDEON: \"{r}\"")
    print()
    
    # Output 2
    r = gideon.respond_simulation_completed(
        "convergenza", 
        suggestions=["modifiche al flusso operativo"]
    )
    print(f"GIDEON: \"{r}\"")
    print()
    
    # Output 3
    scenarios = [{"probability": 0.9} for _ in range(5)]
    r = gideon.respond_ranking_ready(scenarios, top_n=5, criterion="sicurezza")
    print(f"GIDEON: \"{r}\"")
    print()
    
    # Output 4
    r = gideon.respond_prediction("engagement rate", 0.15, 0.88, "positive")
    print(f"GIDEON: \"{r}\"")
    print()
    
    # Output 5
    r = gideon.respond_recommendation("ottimizzare posting schedule", 0.84)
    print(f"GIDEON: \"{r}\"")
    

if __name__ == "__main__":
    test_response_generator()
    demo_output_jarvis()
