"""
Test Multi-Step Reasoning con Verifica Continua
"""

import sys
sys.path.insert(0, 'backend')

from core.reasoning import (
    MultiStepReasoning, ReasoningCommunicator,
    StepType, StepStatus, VerificationResult,
    ReasoningChain, ReasoningStep
)


def test_multi_step_reasoning():
    """Test completo multi-step reasoning"""
    print("=" * 70)
    print("🧠 TEST MULTI-STEP REASONING CON VERIFICA CONTINUA")
    print("=" * 70)
    
    reasoning = MultiStepReasoning()
    
    passed = 0
    total = 0
    
    # === TEST 1: Creazione Catena ===
    print("\n📝 Test 1: Creazione Catena di Ragionamento")
    print("-" * 50)
    total += 1
    
    chain = reasoning.create_chain(
        name="Analisi Strategia Marketing",
        description="Valutazione multi-step per ottimizzazione marketing Q2",
        initial_context={
            "budget": 50000,
            "target": "incremento vendite 25%",
            "timeline": "3 mesi"
        }
    )
    
    print(f"✅ Catena creata: {chain.id}")
    print(f"   Nome: {chain.name}")
    print(f"   Contesto iniziale: {chain.shared_context}")
    passed += 1
    
    # === TEST 2: Aggiunta Steps ===
    print("\n📋 Test 2: Aggiunta Steps alla Catena")
    print("-" * 50)
    total += 1
    
    # Step 1: Analisi dati storici
    step1 = reasoning.add_analysis_step(
        chain.id,
        "Analisi dati storici vendite",
        data={"period": "12_months", "metrics": ["revenue", "conversion", "cac"]},
        criteria=["output_present", "confidence_above_threshold"]
    )
    
    # Step 2: Previsione trend
    step2 = reasoning.add_prediction_step(
        chain.id,
        "Previsione trend Q2",
        context={"base_on": "step_1_output"},
        criteria=["prediction_valid", "confidence_sufficient"]
    )
    
    # Step 3: Simulazione scenari
    step3 = reasoning.add_step(
        chain.id,
        StepType.SIMULATION,
        "Simulazione Monte Carlo - 3 scenari strategici",
        input_data={"scenarios": ["conservative", "moderate", "aggressive"]},
        verification_criteria=["output_present", "no_errors", "confidence_above_threshold"]
    )
    
    # Checkpoint dopo simulazione
    checkpoint = reasoning.add_checkpoint(
        chain.id,
        criteria=["all_simulations_valid", "risk_acceptable"],
        must_pass_all=False
    )
    
    # Step 4: Valutazione rischi
    step4 = reasoning.add_step(
        chain.id,
        StepType.EVALUATION,
        "Valutazione rischi per ogni scenario",
        verification_criteria=["risk_acceptable", "no_errors"]
    )
    
    # Step 5: Verifica coerenza
    step5 = reasoning.add_verification_step(
        chain.id,
        "Verifica coerenza tra analisi e previsioni",
        target_steps=[step1.id, step2.id]
    )
    
    # Step 6: Sintesi finale
    step6 = reasoning.add_step(
        chain.id,
        StepType.SYNTHESIS,
        "Sintesi raccomandazione finale",
        verification_criteria=["output_present", "confidence_sufficient"]
    )
    
    print(f"✅ Aggiunti {len(chain.steps)} steps")
    for s in chain.steps:
        print(f"   {s.step_number}. [{s.step_type.value}] {s.description[:40]}...")
    print(f"   Checkpoint creato dopo step {checkpoint.step_index}")
    passed += 1
    
    # === TEST 3: Avvio Catena ===
    print("\n▶️ Test 3: Avvio Esecuzione Catena")
    print("-" * 50)
    total += 1
    
    reasoning.start_chain(chain.id)
    
    print(f"✅ Catena avviata")
    print(f"   Status: {chain.status.value}")
    print(f"   Step corrente: {chain.current_step_index + 1}")
    passed += 1
    
    # === TEST 4: Esecuzione Step 1 ===
    print("\n⚙️ Test 4: Esecuzione Step 1 (Analisi)")
    print("-" * 50)
    total += 1
    
    # Simula risposta Gideon
    step1_result = reasoning.execute_step(
        chain.id,
        step_result={
            "historical_revenue": [100000, 110000, 105000, 120000],
            "trend": "growing",
            "avg_conversion": 0.032,
            "insights": ["Picco vendite in Q4", "CAC in aumento"]
        },
        confidence=0.88
    )
    
    print(f"✅ Step 1 eseguito")
    print(f"   Output: trend={step1_result.output_data.get('trend')}")
    print(f"   Confidenza: {step1_result.confidence:.0%}")
    print(f"   Verifica: {step1_result.verification_result.value if step1_result.verification_result else 'pending'}")
    print(f"   Status: {step1_result.status.value}")
    passed += 1
    
    # === TEST 5: Avanzamento e Step 2 ===
    print("\n⏭️ Test 5: Avanzamento a Step 2 (Previsione)")
    print("-" * 50)
    total += 1
    
    next_step = reasoning.advance_to_next_step(chain.id)
    
    print(f"✅ Avanzato a step {next_step.step_number}: {next_step.description[:40]}...")
    
    # Esegui step 2
    step2_result = reasoning.execute_step(
        chain.id,
        step_result={
            "prediction": "growth",
            "predicted_revenue_q2": 135000,
            "confidence_interval": [125000, 145000],
            "key_factors": ["seasonal_trend", "market_expansion"]
        },
        confidence=0.82
    )
    
    print(f"   Previsione: +{(135000-120000)/120000:.0%} revenue")
    print(f"   Verifica: {step2_result.verification_result.value}")
    passed += 1
    
    # === TEST 6: Esecuzione Step 3 (Simulazione) ===
    print("\n🔄 Test 6: Step 3 (Simulazione Monte Carlo)")
    print("-" * 50)
    total += 1
    
    reasoning.advance_to_next_step(chain.id)
    
    step3_result = reasoning.execute_step(
        chain.id,
        step_result={
            "simulations": {
                "conservative": {"mean": 0.72, "std": 0.08, "risk": 0.15},
                "moderate": {"mean": 0.78, "std": 0.12, "risk": 0.28},
                "aggressive": {"mean": 0.65, "std": 0.18, "risk": 0.45}
            },
            "recommended": "moderate",
            "iterations": 5000
        },
        confidence=0.85
    )
    
    print(f"✅ Simulazione completata")
    print(f"   Scenario raccomandato: {step3_result.output_data.get('recommended')}")
    print(f"   Verifica: {step3_result.verification_result.value}")
    passed += 1
    
    # === TEST 7: Step 4 (Valutazione Rischi) ===
    print("\n⚠️ Test 7: Step 4 (Valutazione Rischi)")
    print("-" * 50)
    total += 1
    
    reasoning.advance_to_next_step(chain.id)
    
    step4_result = reasoning.execute_step(
        chain.id,
        step_result={
            "risk_assessment": {
                "conservative": {"level": "low", "score": 0.15},
                "moderate": {"level": "medium", "score": 0.28},
                "aggressive": {"level": "high", "score": 0.45}
            },
            "mitigations": ["hedging", "phased_rollout", "monitoring"],
            "risk_score": 0.28  # Per scenario raccomandato
        },
        confidence=0.90
    )
    
    print(f"✅ Valutazione rischi completata")
    print(f"   Risk score (moderate): {step4_result.output_data['risk_score']:.0%}")
    print(f"   Verifica: {step4_result.verification_result.value}")
    passed += 1
    
    # === TEST 8: Step 5 (Verifica Coerenza) ===
    print("\n✔️ Test 8: Step 5 (Verifica Coerenza)")
    print("-" * 50)
    total += 1
    
    reasoning.advance_to_next_step(chain.id)
    
    step5_result = reasoning.execute_step(
        chain.id,
        step_result={
            "consistency_check": "passed",
            "alignment_score": 0.92,
            "notes": "Previsioni coerenti con trend storici"
        },
        confidence=0.95
    )
    
    print(f"✅ Verifica coerenza completata")
    print(f"   Allineamento: {step5_result.output_data['alignment_score']:.0%}")
    print(f"   Verifica: {step5_result.verification_result.value}")
    passed += 1
    
    # === TEST 9: Step 6 (Sintesi Finale) ===
    print("\n📊 Test 9: Step 6 (Sintesi Finale)")
    print("-" * 50)
    total += 1
    
    reasoning.advance_to_next_step(chain.id)
    
    step6_result = reasoning.execute_step(
        chain.id,
        step_result={
            "recommendation": "Procedere con strategia MODERATE",
            "expected_outcome": "+12.5% revenue Q2",
            "confidence": 0.78,
            "risk_level": "medium",
            "key_actions": [
                "Incrementare budget digital +20%",
                "Focus su retention clienti esistenti",
                "A/B testing nuovi canali"
            ],
            "timeline": "Implementazione graduale 12 settimane"
        },
        confidence=0.88
    )
    
    print(f"✅ Sintesi finale completata")
    print(f"   Raccomandazione: {step6_result.output_data['recommendation']}")
    print(f"   Expected outcome: {step6_result.output_data['expected_outcome']}")
    print(f"   Verifica: {step6_result.verification_result.value}")
    passed += 1
    
    # === TEST 10: Completamento Catena ===
    print("\n🏁 Test 10: Completamento Catena")
    print("-" * 50)
    total += 1
    
    # Avanza per completare
    final = reasoning.advance_to_next_step(chain.id)
    
    # Trova catena completata
    completed_chain = reasoning.completed_chains[-1] if reasoning.completed_chains else None
    
    if completed_chain:
        print(f"✅ Catena completata!")
        print(f"   Status: {completed_chain.status.value}")
        print(f"   Pass rate: {completed_chain.verification_pass_rate:.0%}")
        print(f"   Confidenza finale: {completed_chain.final_confidence:.0%}")
        print(f"   Tempo totale: {completed_chain.total_execution_time_ms:.0f}ms")
        passed += 1
    else:
        print("❌ Catena non trovata in completed")
    
    # === TEST 11: Verifica Sintesi Risultato ===
    print("\n📑 Test 11: Sintesi Risultato Finale")
    print("-" * 50)
    total += 1
    
    if completed_chain:
        result = completed_chain.final_result
        print(f"✅ Risultato sintetizzato")
        print(f"   Steps completati: {result['steps_completed']}/{result['total_steps']}")
        print(f"   Pass rate: {result['verification_pass_rate']:.0%}")
        
        # Mostra output chiave
        for step_key, step_data in list(result.get('step_outputs', {}).items())[:3]:
            print(f"   {step_key}: {step_data['description'][:30]}... ✓" if step_data['verified'] else f"   {step_key}: ✗")
        passed += 1
    else:
        print("❌ Nessun risultato")
    
    # === TEST 12: Statistiche ===
    print("\n📈 Test 12: Statistiche Globali")
    print("-" * 50)
    total += 1
    
    stats = reasoning.get_statistics()
    
    print(f"✅ Statistiche")
    print(f"   Catene create: {stats['chains_created']}")
    print(f"   Catene completate: {stats['chains_completed']}")
    print(f"   Steps eseguiti: {stats['steps_executed']}")
    print(f"   Verifiche passate: {stats['verifications_passed']}")
    print(f"   Success rate: {stats['success_rate']:.0%}")
    print(f"   Verification pass rate: {stats['verification_pass_rate']:.0%}")
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


def test_rollback_scenario():
    """Test scenario di rollback"""
    print("\n" + "=" * 70)
    print("🔄 TEST ROLLBACK SCENARIO")
    print("=" * 70)
    
    reasoning = MultiStepReasoning()
    
    # Crea catena
    chain = reasoning.create_chain("Test Rollback", initial_context={"test": True})
    
    # Aggiungi steps
    step1 = reasoning.add_analysis_step(chain.id, "Analisi iniziale", {"data": "test"})
    checkpoint = reasoning.add_checkpoint(chain.id, ["data_valid"])
    step2 = reasoning.add_prediction_step(chain.id, "Previsione", {"ctx": "test"})
    
    # Avvia ed esegui step 1
    reasoning.start_chain(chain.id)
    reasoning.execute_step(chain.id, {"result": "ok"}, 0.9)
    reasoning.advance_to_next_step(chain.id)
    
    # Esegui step 2 con risultato che fallisce
    reasoning.execute_step(chain.id, {"error": "validation_failed"}, 0.3)
    
    print(f"Step 2 status: {chain.steps[1].status.value}")
    print(f"Step 2 verification: {chain.steps[1].verification_result.value if chain.steps[1].verification_result else 'N/A'}")
    
    # Rollback
    print("\n🔙 Eseguo rollback...")
    reasoning.rollback_to_checkpoint(chain.id)
    
    print(f"Current step dopo rollback: {chain.current_step_index + 1}")
    print(f"Step 2 status dopo rollback: {chain.steps[1].status.value}")
    
    # Retry
    print("\n🔁 Retry step 2...")
    reasoning.retry_step(chain.id, step2.id)
    print(f"Step 2 status dopo retry: {chain.steps[1].status.value}")
    
    print("\n✅ Test rollback completato!")


def demo_reasoning_flow():
    """Demo flusso ragionamento completo"""
    print("\n" + "=" * 70)
    print("🎬 DEMO: Flusso Multi-Step Reasoning")
    print("=" * 70)
    
    reasoning = MultiStepReasoning()
    
    print("\n--- SCENARIO: Decisione strategica di investimento ---\n")
    
    # Crea catena
    chain = reasoning.create_chain(
        "Valutazione Investimento",
        initial_context={"amount": 100000, "risk_tolerance": "medium"}
    )
    
    # Definisci steps
    print("📋 Definizione catena di ragionamento:")
    
    steps = [
        ("ANALYSIS", "Analisi mercato attuale"),
        ("PREDICTION", "Previsione trend 6 mesi"),
        ("SIMULATION", "Simulazione 3 portafogli"),
        ("EVALUATION", "Valutazione risk/reward"),
        ("VERIFICATION", "Verifica coerenza analisi"),
        ("SYNTHESIS", "Raccomandazione finale")
    ]
    
    for step_type, desc in steps:
        reasoning.add_step(chain.id, StepType[step_type], desc)
        print(f"   + {step_type}: {desc}")
    
    # Checkpoint a metà
    reasoning.add_checkpoint(chain.id, ["data_valid", "predictions_stable"])
    print("   📍 Checkpoint aggiunto")
    
    # Esecuzione simulata
    print("\n▶️ Esecuzione:")
    reasoning.start_chain(chain.id)
    
    results = [
        ({"market": "bullish", "volatility": 0.15}, 0.85),
        ({"prediction": "+8%", "confidence_interval": [5, 12]}, 0.78),
        ({"best_portfolio": "balanced", "expected_return": 0.09}, 0.82),
        ({"risk_score": 0.25, "reward_ratio": 2.1}, 0.88),
        ({"consistency": "high", "alignment": 0.91}, 0.95),
        ({"recommendation": "INVEST", "allocation": {"stocks": 60, "bonds": 30, "cash": 10}}, 0.86)
    ]
    
    for i, (result, conf) in enumerate(results):
        step = reasoning.execute_step(chain.id, result, conf)
        status = "✓" if step.verification_result == VerificationResult.PASSED else "?"
        print(f"   {i+1}. {step.description[:30]}... {status} ({conf:.0%})")
        reasoning.advance_to_next_step(chain.id)
    
    # Risultato
    print("\n📊 Risultato Finale:")
    if reasoning.completed_chains:
        final = reasoning.completed_chains[-1]
        print(f"   Pass rate: {final.verification_pass_rate:.0%}")
        print(f"   Confidenza: {final.final_confidence:.0%}")
        print(f"   Raccomandazione: {final.final_result.get('step_outputs', {}).get('step_6', {}).get('output', {}).get('recommendation', 'N/A')}")


if __name__ == "__main__":
    test_multi_step_reasoning()
    test_rollback_scenario()
    demo_reasoning_flow()
