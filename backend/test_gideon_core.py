# Test Gideon Core - Scenario Builder & Simulation Engine
"""Test dei nuovi componenti Gideon: ScenarioBuilder e SimulationEngine"""

import sys
sys.path.insert(0, r"c:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\backend")

from gideon import (
    GideonCore, ScenarioBuilder, SimulationEngine,
    ScenarioType, ScenarioStatus, VariableType,
    SimulationMode, DistributionType
)

print("=" * 60)
print("🔮 TEST GIDEON CORE - Scenario Builder & Simulation Engine")
print("=" * 60)

# ===== TEST SCENARIO BUILDER =====
print("\n📊 TEST SCENARIO BUILDER")
print("-" * 40)

builder = ScenarioBuilder()

# 1. Crea scenario base da template
print("\n1️⃣ Creazione scenario da template 'system_action':")
scenario1 = builder.create_scenario(
    name="Deploy nuovo modulo",
    description="Deploy del modulo di analisi predittiva",
    scenario_type=ScenarioType.BASELINE,
    template="system_action",
    context={"module": "gideon_predictor", "priority": "high"}
)
print(f"   ID: {scenario1.id}")
print(f"   Nome: {scenario1.name}")
print(f"   Tipo: {scenario1.scenario_type.value}")
print(f"   Variabili: {list(scenario1.variables.keys())}")
print(f"   Outcomes: {[o.name for o in scenario1.outcomes]}")

# 2. Crea varianti ottimistica/pessimistica
print("\n2️⃣ Creazione varianti ottimistica/pessimistica:")
opt_scenario, pess_scenario = builder.create_optimistic_pessimistic(scenario1.id)
print(f"   Ottimistico: {opt_scenario.name}")
print(f"     - success_rate: {opt_scenario.variables['success_rate'].value:.3f}")
print(f"   Pessimistico: {pess_scenario.name}")
print(f"     - success_rate: {pess_scenario.variables['success_rate'].value:.3f}")

# 3. Crea scenario what-if
print("\n3️⃣ Creazione scenario What-If:")
whatif = builder.create_what_if(
    scenario1.id,
    "What if resource usage doubles?",
    {"resource_usage": 60.0}
)
print(f"   What-If: {whatif.name}")
print(f"   Ipotesi: {whatif.assumptions}")
print(f"   Resource Usage: {whatif.variables['resource_usage'].value}")

# 4. Confronta scenari
print("\n4️⃣ Confronto scenari:")
comparison = builder.compare_scenarios(
    [scenario1.id, opt_scenario.id, pess_scenario.id, whatif.id]
)
if comparison:
    print(f"   Scenari confrontati: {len(comparison.scenarios)}")
    print(f"   Best Overall: {comparison.best_overall}")
    print(f"   Worst Overall: {comparison.worst_overall}")
    print(f"   Trade-offs trovati: {len(comparison.trade_offs)}")

# 5. Status builder
print("\n5️⃣ Stato ScenarioBuilder:")
status = builder.get_status()
print(f"   Scenari totali: {status['total_scenarios']}")
print(f"   Per tipo: {status['by_type']}")
print(f"   Templates disponibili: {status['templates']}")


# ===== TEST SIMULATION ENGINE =====
print("\n\n⚡ TEST SIMULATION ENGINE")
print("-" * 40)

engine = SimulationEngine(max_workers=4)
engine.set_seed(42)  # Per riproducibilità

# Prepara scenario per simulazione
sim_scenario = {
    "id": scenario1.id,
    "variables": {var_name: var.to_dict() for var_name, var in scenario1.variables.items()},
    "constraints": [c.to_dict() for c in scenario1.constraints],
    "outcomes": [o.to_dict() for o in scenario1.outcomes]
}

# 1. Singola simulazione
print("\n1️⃣ Simulazione singola:")
result = engine.simulate_single(sim_scenario)
print(f"   Iteration: {result.iteration}")
print(f"   Outcome: {result.outcome}")
print(f"   Success: {result.success}")
print(f"   Values: {result.values}")

# 2. Monte Carlo
print("\n2️⃣ Simulazione Monte Carlo (1000 iterazioni):")
mc_result = engine.simulate_monte_carlo(sim_scenario, iterations=1000)
print(f"   Success Rate: {mc_result.success_rate:.2%}")
print(f"   Expected Value: {mc_result.expected_value:.3f}")
print(f"   Risk Score: {mc_result.risk_score:.3f}")
print(f"   Outcome Distribution: {mc_result.outcome_probabilities}")
print(f"   Execution Time: {mc_result.execution_time_ms:.2f}ms")

# 3. Monte Carlo Parallelo
print("\n3️⃣ Simulazione Monte Carlo PARALLELA (1000 iterazioni):")
mc_parallel = engine.simulate_monte_carlo_parallel(sim_scenario, iterations=1000)
print(f"   Success Rate: {mc_parallel.success_rate:.2%}")
print(f"   Execution Time: {mc_parallel.execution_time_ms:.2f}ms")
print(f"   Confidence Intervals: ")
for var, interval in list(mc_parallel.confidence_intervals.items())[:3]:
    print(f"     - {var}: [{interval[0]:.3f}, {interval[1]:.3f}]")

# 4. Analisi di Sensitività
print("\n4️⃣ Analisi di Sensitività su 'success_rate':")
sensitivity = engine.analyze_sensitivity(
    sim_scenario,
    variable="success_rate",
    values=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    iterations_per_value=50
)
print(f"   Variabile: {sensitivity.variable}")
print(f"   Elasticità: {sensitivity.elasticity:.3f}")
print(f"   Valori testati: {sensitivity.values_tested}")
print(f"   Soglie critiche: {sensitivity.critical_thresholds}")

# 5. Simulazione Predittiva
print("\n5️⃣ Simulazione Predittiva (24h, 12 step):")
from datetime import timedelta
predictive = engine.simulate_predictive(
    sim_scenario,
    time_horizon=timedelta(hours=24),
    time_steps=12
)
print(f"   Scenario: {predictive.scenario_id}")
print(f"   Time Steps: {predictive.time_steps}")
print(f"   Final Predictions:")
for var, pred in list(predictive.final_predictions.items())[:3]:
    print(f"     - {var}: {pred:.3f}")
print(f"   Trends:")
for var, trend in list(predictive.trends.items())[:3]:
    print(f"     - {var}: {trend}")
print(f"   Turning Points trovati: {len(predictive.turning_points)}")

# 6. Status engine
print("\n6️⃣ Stato SimulationEngine:")
eng_status = engine.get_status()
print(f"   Workers: {eng_status['max_workers']}")
print(f"   Total Simulations: {eng_status['total_simulations']}")
print(f"   Total Iterations: {eng_status['total_iterations']}")


# ===== TEST GIDEON CORE INTEGRATION =====
print("\n\n🔮 TEST GIDEON CORE (INTEGRAZIONE)")
print("-" * 40)

gideon = GideonCore(max_workers=4)

# 1. Build scenario tramite GideonCore
print("\n1️⃣ Build scenario via GideonCore:")
scn = gideon.build_scenario(
    name="Automation Workflow Test",
    description="Test workflow automazione",
    scenario_type=ScenarioType.BASELINE,
    template="automation_workflow"
)
print(f"   Scenario: {scn.name}")
print(f"   ID: {scn.id}")

# 2. Run Monte Carlo via GideonCore
print("\n2️⃣ Monte Carlo via GideonCore:")
scn_dict = {
    "id": scn.id,
    "variables": {k: v.to_dict() for k, v in scn.variables.items()},
    "constraints": [c.to_dict() for c in scn.constraints],
    "outcomes": [o.to_dict() for o in scn.outcomes]
}
mc = gideon.run_monte_carlo(scn_dict, iterations=500, parallel=True)
print(f"   Success Rate: {mc.success_rate:.2%}")
print(f"   Risk Score: {mc.risk_score:.3f}")

# 3. Predictive via GideonCore
print("\n3️⃣ Predictive Simulation via GideonCore:")
pred = gideon.run_predictive_simulation(scn_dict, hours=12, steps=6)
print(f"   Final Predictions: {len(pred.final_predictions)} variabili")
print(f"   Trends: {list(pred.trends.values())[:3]}")

# 4. Statistics complete
print("\n4️⃣ Statistiche complete GideonCore:")
stats = gideon.get_statistics()
print(f"   Scenario Builder: {stats['scenario_builder']['total_scenarios']} scenari")
print(f"   Simulation Engine: {stats['simulation_engine']['total_iterations']} iterazioni")
print(f"   Is Active: {stats['is_active']}")

print("\n" + "=" * 60)
print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
print("=" * 60)
