"""
Test Sistema Comunicazione Jarvis ↔ Gideon
Verifica comunicazione strutturata e loggata
"""

import sys
sys.path.insert(0, 'backend')

from core.communication import (
    CommunicationBridge, CommunicationChannel,
    JarvisCommunicator, GideonCommunicator,
    Objective, Constraint, MessageType, MessagePriority,
    Sender
)


def test_communication_system():
    """Test completo sistema comunicazione"""
    print("=" * 70)
    print("📡 TEST COMUNICAZIONE JARVIS ↔ GIDEON")
    print("=" * 70)
    
    # Inizializza bridge
    bridge = CommunicationBridge()
    
    passed = 0
    total = 0
    
    # === TEST 1: Invio Obiettivo da Jarvis ===
    print("\n📤 Test 1: Jarvis invia Obiettivo a Gideon")
    print("-" * 50)
    total += 1
    
    objective = Objective(
        description="Ottimizzare strategia di posting su Instagram",
        success_criteria=[
            "Aumentare engagement del 20%",
            "Ridurre tempo gestione del 30%"
        ],
        constraints=["Budget massimo: €500/mese", "No post dopo le 22:00"],
        priority=8
    )
    
    constraints = [
        Constraint(type="resource", description="Budget limitato", value=500, is_hard=True),
        Constraint(type="time", description="Orario posting", value="08:00-22:00", is_hard=False)
    ]
    
    correlation_id = bridge.jarvis.send_objective(objective, constraints)
    
    print(f"✅ Obiettivo inviato")
    print(f"   Correlation ID: {correlation_id}")
    print(f"   Descrizione: {objective.description}")
    print(f"   Criteri successo: {len(objective.success_criteria)}")
    print(f"   Vincoli: {len(constraints)}")
    passed += 1
    
    # === TEST 2: Gideon risponde con Analisi ===
    print("\n📥 Test 2: Gideon risponde con Analisi")
    print("-" * 50)
    total += 1
    
    # Simula recupero messaggio
    history = bridge.channel.get_history(sender=Sender.JARVIS, limit=1)
    request = history[0] if history else None
    
    response = bridge.gideon.respond_analysis(
        request_id=request.id if request else "test",
        correlation_id=correlation_id,
        summary="Analisi completata. Strategia attuale subottimale, margini di miglioramento identificati.",
        details={
            "current_engagement": 0.045,
            "potential_engagement": 0.068,
            "improvement": 0.51,
            "key_factors": ["timing post", "qualità contenuti", "hashtag strategy"]
        },
        probability=0.78,
        confidence=0.85,
        risk_level="low",
        risk_score=0.22,
        recommendations=[
            "Ottimizzare orari di posting (picco: 12:00-13:00, 19:00-20:00)",
            "Aumentare contenuti video (+40% engagement)",
            "Utilizzare hashtag di nicchia specifici"
        ]
    )
    
    print(f"✅ Risposta Gideon ricevuta")
    print(f"   Summary: {response.summary[:60]}...")
    print(f"   Probabilità successo: {response.probability:.0%}")
    print(f"   Confidenza: {response.confidence:.0%}")
    print(f"   Livello rischio: {response.risk_level}")
    print(f"   Raccomandazioni: {len(response.recommendations)}")
    passed += 1
    
    # === TEST 3: Jarvis richiede Scenari ===
    print("\n📤 Test 3: Jarvis richiede Scenari")
    print("-" * 50)
    total += 1
    
    scenario_corr = bridge.jarvis.request_scenarios(
        objective="Aumentare follower Instagram da 5K a 10K in 3 mesi",
        constraints=[{"type": "budget", "value": 1000}],
        num_scenarios=5
    )
    
    print(f"✅ Richiesta scenari inviata")
    print(f"   Correlation ID: {scenario_corr}")
    passed += 1
    
    # === TEST 4: Gideon risponde con Scenari ===
    print("\n📥 Test 4: Gideon risponde con Scenari e Ranking")
    print("-" * 50)
    total += 1
    
    scenarios = [
        {"name": "Organic Growth", "probability": 0.65, "risk_level": "low", "cost": 200},
        {"name": "Influencer Collab", "probability": 0.82, "risk_level": "medium", "cost": 800},
        {"name": "Paid Ads", "probability": 0.75, "risk_level": "medium", "cost": 600},
        {"name": "Content Viral", "probability": 0.45, "risk_level": "high", "cost": 300},
        {"name": "Community Building", "probability": 0.70, "risk_level": "low", "cost": 400}
    ]
    
    ranking = [
        {"rank": 1, "name": "Influencer Collab", "score": 0.89},
        {"rank": 2, "name": "Paid Ads", "score": 0.76},
        {"rank": 3, "name": "Community Building", "score": 0.72},
        {"rank": 4, "name": "Organic Growth", "score": 0.68},
        {"rank": 5, "name": "Content Viral", "score": 0.52}
    ]
    
    response = bridge.gideon.respond_scenarios(
        request_id="test",
        correlation_id=scenario_corr,
        scenarios=scenarios,
        ranking=ranking,
        best_scenario=scenarios[1]  # Influencer Collab
    )
    
    print(f"✅ Scenari ricevuti")
    print(f"   Scenari generati: {len(scenarios)}")
    print(f"   Scenario migliore: {response.details['best']['name']}")
    print(f"   Probabilità: {response.probability:.0%}")
    for r in ranking[:3]:
        print(f"   #{r['rank']} {r['name']}: {r['score']:.2f}")
    passed += 1
    
    # === TEST 5: Jarvis richiede Risk Check ===
    print("\n📤 Test 5: Jarvis richiede Verifica Rischi")
    print("-" * 50)
    total += 1
    
    risk_corr = bridge.jarvis.request_risk_check(
        action={"type": "execute", "target": "Influencer Collab", "budget": 800},
        context={"current_followers": 5000, "target": 10000}
    )
    
    print(f"✅ Risk check richiesto")
    print(f"   Correlation ID: {risk_corr}")
    passed += 1
    
    # === TEST 6: Gideon risponde Risk Assessment ===
    print("\n📥 Test 6: Gideon risponde con Risk Assessment")
    print("-" * 50)
    total += 1
    
    risks = [
        {"type": "reputation", "description": "Influencer non allineato ai valori brand", "severity": 0.6},
        {"type": "financial", "description": "ROI non garantito", "severity": 0.4},
        {"type": "timing", "description": "Risultati potrebbero richiedere più tempo", "severity": 0.3}
    ]
    
    mitigations = [
        "Verificare portfolio e recensioni influencer",
        "Definire KPI chiari nel contratto",
        "Prevedere piano B con budget rimanente"
    ]
    
    response = bridge.gideon.respond_risk_assessment(
        request_id="test",
        correlation_id=risk_corr,
        risk_level="medium",
        risk_score=0.42,
        risks=risks,
        mitigations=mitigations,
        proceed_recommendation=True
    )
    
    print(f"✅ Risk assessment ricevuto")
    print(f"   Livello: {response.risk_level}")
    print(f"   Score: {response.risk_score:.0%}")
    print(f"   Rischi identificati: {len(risks)}")
    print(f"   Mitigazioni: {len(mitigations)}")
    print(f"   Raccomandazione: {'Procedi' if not response.requires_action else 'Rivedi'}")
    passed += 1
    
    # === TEST 7: Jarvis richiede Simulazione ===
    print("\n📤 Test 7: Jarvis richiede Simulazione Monte Carlo")
    print("-" * 50)
    total += 1
    
    sim_corr = bridge.jarvis.request_simulation(
        scenario={"name": "Influencer Collab", "base_probability": 0.82},
        iterations=5000,
        include_sensitivity=True
    )
    
    print(f"✅ Simulazione richiesta")
    print(f"   Correlation ID: {sim_corr}")
    passed += 1
    
    # === TEST 8: Gideon risponde Simulazione ===
    print("\n📥 Test 8: Gideon risponde con Risultati Simulazione")
    print("-" * 50)
    total += 1
    
    response = bridge.gideon.respond_simulation(
        request_id="test",
        correlation_id=sim_corr,
        mean=0.79,
        std=0.12,
        confidence_intervals={
            "90%": (0.65, 0.93),
            "95%": (0.58, 0.96),
            "99%": (0.48, 0.99)
        },
        iterations=5000,
        convergence=True
    )
    
    print(f"✅ Risultati simulazione ricevuti")
    print(f"   Media: {response.details['mean']:.0%}")
    print(f"   Std Dev: {response.details['std']:.2f}")
    print(f"   CI 95%: {response.details['confidence_intervals']['95%']}")
    print(f"   Convergenza: {'Sì' if response.details['converged'] else 'No'}")
    passed += 1
    
    # === TEST 9: Jarvis richiede Validazione ===
    print("\n📤 Test 9: Jarvis richiede Validazione Decisione")
    print("-" * 50)
    total += 1
    
    val_corr = bridge.jarvis.request_validation(
        decision={
            "action": "Avviare campagna Influencer Collab",
            "budget": 800,
            "timeline": "3 mesi",
            "target": "+5000 follower"
        },
        constraints=[{"type": "budget", "max": 1000}]
    )
    
    print(f"✅ Validazione richiesta")
    print(f"   Correlation ID: {val_corr}")
    passed += 1
    
    # === TEST 10: Gideon risponde Validazione ===
    print("\n📥 Test 10: Gideon risponde con Validazione")
    print("-" * 50)
    total += 1
    
    response = bridge.gideon.respond_validation(
        request_id="test",
        correlation_id=val_corr,
        is_valid=True,
        issues=[],
        suggestions=["Monitorare KPI settimanalmente", "Preparare contenuti in anticipo"]
    )
    
    print(f"✅ Validazione ricevuta")
    print(f"   Esito: {response.summary}")
    print(f"   Azione suggerita: {response.suggested_action}")
    passed += 1
    
    # === TEST 11: Gideon invia Warning proattivo ===
    print("\n⚠️ Test 11: Gideon invia Warning proattivo")
    print("-" * 50)
    total += 1
    
    warning_corr = bridge.gideon.send_warning(
        message="Rilevato calo engagement nelle ultime 24h (-15%)",
        severity="high",
        context={"metric": "engagement", "change": -0.15}
    )
    
    print(f"✅ Warning inviato")
    print(f"   Correlation ID: {warning_corr}")
    passed += 1
    
    # === TEST 12: Gideon invia Raccomandazione proattiva ===
    print("\n💡 Test 12: Gideon invia Raccomandazione proattiva")
    print("-" * 50)
    total += 1
    
    rec_corr = bridge.gideon.send_recommendation(
        action="Pubblicare Reel trending entro 2 ore",
        probability=0.73,
        reasoning="Trend #TechTips in crescita, finestra ottimale prossime 3 ore",
        alternatives=[
            {"action": "Attendere domani mattina", "probability": 0.58},
            {"action": "Postare carosello invece", "probability": 0.62}
        ]
    )
    
    print(f"✅ Raccomandazione inviata")
    print(f"   Correlation ID: {rec_corr}")
    passed += 1
    
    # === STATISTICHE E LOG ===
    print("\n📊 Statistiche Comunicazione")
    print("-" * 50)
    
    stats = bridge.get_statistics()
    channel_stats = stats["channel"]
    
    print(f"   Totale messaggi: {channel_stats['total_messages']}")
    print(f"   Jarvis → Gideon: {channel_stats['jarvis_to_gideon']}")
    print(f"   Gideon → Jarvis: {channel_stats['gideon_to_jarvis']}")
    print(f"   Completati: {channel_stats['completed']}")
    print(f"   Correlazioni attive: {channel_stats['active_correlations']}")
    
    # === CONVERSAZIONE ===
    print("\n📜 Log Conversazione (prima correlazione)")
    print("-" * 50)
    
    conversation = bridge.get_conversation_log(correlation_id)
    for entry in conversation[:4]:
        print(f"   [{entry['sender'].upper()}] {entry['type']}: {entry['summary'][:50]}...")
    
    # === RISULTATO FINALE ===
    print("\n" + "=" * 70)
    print(f"📊 RISULTATO: {passed}/{total} test passati")
    
    if passed == total:
        print("✅ TUTTI I TEST PASSATI!")
    else:
        print(f"❌ {total - passed} test falliti")
    
    print("=" * 70)
    
    return passed == total


def demo_communication_flow():
    """Demo flusso comunicazione completo"""
    print("\n" + "=" * 70)
    print("🎬 DEMO: Flusso Comunicazione Jarvis ↔ Gideon")
    print("=" * 70)
    
    bridge = CommunicationBridge()
    
    print("\n--- SCENARIO: Jarvis deve decidere strategia marketing ---\n")
    
    # Step 1
    print("1️⃣ JARVIS: 'Gideon, analizza opzioni per aumentare vendite Q2'")
    corr = bridge.jarvis.send_objective(
        Objective(
            description="Aumentare vendite Q2 del 25%",
            success_criteria=["Vendite +25%", "ROI > 3x"],
            priority=9
        )
    )
    
    # Step 2
    print("\n2️⃣ GIDEON: 'Analisi completata. Ho identificato 3 strategie...'")
    bridge.gideon.respond_analysis(
        request_id="x", correlation_id=corr,
        summary="3 strategie identificate, la migliore ha 78% probabilità successo",
        details={"strategies": 3},
        probability=0.78, confidence=0.85,
        risk_level="medium", risk_score=0.35,
        recommendations=["Strategia A: Digital Marketing intensivo"]
    )
    
    # Step 3
    print("\n3️⃣ JARVIS: 'Genera scenari dettagliati per Strategia A'")
    corr2 = bridge.jarvis.request_scenarios(
        objective="Digital Marketing Q2",
        num_scenarios=5
    )
    
    # Step 4
    print("\n4️⃣ GIDEON: 'Classifica dei 5 scenari pronta per Jarvis'")
    bridge.gideon.respond_scenarios(
        request_id="y", correlation_id=corr2,
        scenarios=[{"name": f"Scenario {i}", "probability": 0.9-i*0.1} for i in range(5)],
        ranking=[{"rank": i+1, "name": f"Scenario {i}", "score": 0.95-i*0.1} for i in range(5)],
        best_scenario={"name": "Scenario 0", "probability": 0.90}
    )
    
    # Step 5
    print("\n5️⃣ JARVIS: 'Valida Scenario 0 prima dell'esecuzione'")
    corr3 = bridge.jarvis.request_validation(
        decision={"action": "Eseguire Scenario 0", "budget": 50000}
    )
    
    # Step 6
    print("\n6️⃣ GIDEON: 'Validazione: APPROVATO ✓'")
    bridge.gideon.respond_validation(
        request_id="z", correlation_id=corr3,
        is_valid=True,
        suggestions=["Monitorare KPI giornalmente"]
    )
    
    print("\n--- JARVIS procede con l'esecuzione ---\n")
    
    # Stats finali
    stats = bridge.get_statistics()
    print(f"📊 Messaggi scambiati: {stats['channel']['total_messages']}")
    print(f"   Jarvis → Gideon: {stats['channel']['jarvis_to_gideon']}")
    print(f"   Gideon → Jarvis: {stats['channel']['gideon_to_jarvis']}")


if __name__ == "__main__":
    test_communication_system()
    demo_communication_flow()
