# Test Gideon Core - Probability, Risk, Historical Analyzers
"""Test dei nuovi componenti Gideon: ProbabilityCalculator, RiskAnalyzer, HistoricalAnalyzer"""

import sys
sys.path.insert(0, r"c:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\backend")

from gideon import (
    GideonCore,
    ProbabilityCalculator, ProbabilityMethod, ConfidenceLevel,
    RiskAnalyzer, RiskLevel, RiskCategory, MitigationType,
    HistoricalAnalyzer, TrendDirection, PatternType, LearningType
)

print("=" * 60)
print("🔮 TEST GIDEON - Probability, Risk & Historical Analyzers")
print("=" * 60)

# ===== TEST PROBABILITY CALCULATOR =====
print("\n📊 TEST PROBABILITY CALCULATOR")
print("-" * 40)

prob_calc = ProbabilityCalculator()

# 1. Calcolo probabilità combinato
print("\n1️⃣ Calcolo probabilità (metodo COMBINED):")
context = {
    "action_type": "system_action",
    "system_health": 0.85,
    "resources_available": 0.90,
    "complexity": 0.3,
    "historical_success_rate": 0.78,
    "time_pressure": 0.2,
    "external_reliability": 0.92,
    "history": [
        {"success": True}, {"success": True}, {"success": False},
        {"success": True}, {"success": True}, {"success": True},
        {"success": True}, {"success": False}, {"success": True},
        {"success": True}
    ]
}
result = prob_calc.calculate(context, ProbabilityMethod.COMBINED)
print(f"   Success Probability: {result.success_probability:.2%}")
print(f"   Failure Probability: {result.failure_probability:.2%}")
print(f"   Confidence Level: {result.confidence_level.value}")
print(f"   Confidence Score: {result.confidence_score:.3f}")
print(f"   Method: {result.method.value}")
print(f"   Breakdown: {result.breakdown}")

# 2. Calcolo Bayesiano
print("\n2️⃣ Calcolo probabilità Bayesiano:")
bayes_result = prob_calc.calculate(context, ProbabilityMethod.BAYESIAN)
print(f"   Prior → Posterior: {bayes_result.breakdown.get('prior', 0):.3f} → {bayes_result.breakdown.get('posterior', 0):.3f}")
print(f"   Success Probability: {bayes_result.success_probability:.2%}")

# 3. Analisi scenario completo
print("\n3️⃣ Analisi scenario completa:")
scenario_analysis = prob_calc.analyze_scenario(context)
print(f"   Base case: {scenario_analysis['base_case']['success_probability']:.2%}")
print(f"   Optimistic: {scenario_analysis['optimistic_case']['success_probability']:.2%}")
print(f"   Pessimistic: {scenario_analysis['pessimistic_case']['success_probability']:.2%}")
print(f"   Recommendation: {scenario_analysis['recommendation']}")

# 4. Probabilità catena eventi
print("\n4️⃣ Probabilità catena eventi:")
events = [
    {"probability": 0.9},
    {"probability": 0.85},
    {"probability": 0.95}
]
chain_prob = prob_calc.chain_probability(events)
print(f"   Eventi: {[e['probability'] for e in events]}")
print(f"   Probabilità tutti successo: {chain_prob:.2%}")

# 5. Almeno un successo
print("\n5️⃣ Almeno un successo in 3 tentativi:")
probs = [0.6, 0.6, 0.6]
at_least_one = prob_calc.at_least_one_success(probs)
print(f"   P(singolo) = 60%, tentativi = 3")
print(f"   P(almeno 1 successo) = {at_least_one:.2%}")


# ===== TEST RISK ANALYZER =====
print("\n\n⚠️ TEST RISK ANALYZER")
print("-" * 40)

risk_analyzer = RiskAnalyzer()

# 1. Valutazione rischio scenario
print("\n1️⃣ Valutazione rischio scenario:")
scenario = {
    "id": "deploy_scenario",
    "action_type": "system_deployment",
    "components": ["api", "database", "web_server"],
    "context": {
        "system_health": 0.75,
        "complexity": 0.6,
        "triggers": ["dependency_failure", "high_load"]
    }
}
assessment = risk_analyzer.assess(scenario)
print(f"   Overall Risk Level: {assessment.overall_risk_level.value}")
print(f"   Overall Risk Score: {assessment.overall_risk_score:.3f}")
print(f"   Total Risks: {assessment.total_risks}")
print(f"   Critical Risks: {assessment.critical_risks}")
print(f"   Residual Risk: {assessment.residual_risk:.3f}")
print(f"   Recommendation: {assessment.proceed_recommendation}")

# 2. Dettaglio rischi identificati
print("\n2️⃣ Rischi identificati:")
for risk in assessment.risks[:3]:
    print(f"   - {risk.name} [{risk.level.value}]")
    print(f"     P={risk.probability:.2f}, I={risk.impact:.2f}, Score={risk.risk_score:.3f}")

# 3. Mitigazioni proposte
print("\n3️⃣ Mitigazioni proposte:")
for mit in assessment.mitigations[:4]:
    print(f"   - {mit.name} ({mit.mitigation_type.value})")
    print(f"     Effectiveness: {mit.effectiveness:.2%}, Priority: {mit.priority}")

# 4. Matrice rischio
print("\n4️⃣ Matrice di rischio:")
matrix = risk_analyzer.analyze_risk_matrix(assessment.risks)
print(f"   High-High (critico): {matrix['counts']['high_high']} rischi")
print(f"   High-Low: {matrix['counts']['high_low']} rischi")
print(f"   Low-High: {matrix['counts']['low_high']} rischi")
print(f"   Low-Low: {matrix['counts']['low_low']} rischi")

# 5. Confronto scenari
print("\n5️⃣ Confronto rischi tra scenari:")
scenarios = [
    {"id": "conservative", "action_type": "safe", "components": ["api"], "context": {"system_health": 0.95}},
    {"id": "aggressive", "action_type": "risky", "components": ["api", "database", "core"], "context": {"system_health": 0.7, "triggers": ["overload"]}}
]
comparison = risk_analyzer.compare_scenarios(scenarios)
print(f"   Safest: {comparison['safest_scenario']}")
print(f"   Riskiest: {comparison['riskiest_scenario']}")
print(f"   Risk Scores: {comparison['risk_scores']}")


# ===== TEST HISTORICAL ANALYZER =====
print("\n\n📈 TEST HISTORICAL ANALYZER")
print("-" * 40)

hist_analyzer = HistoricalAnalyzer()

# 1. Registra eventi storici
print("\n1️⃣ Registrazione eventi storici:")
import random
random.seed(42)

for i in range(50):
    success = random.random() > 0.25  # 75% success rate
    duration = random.uniform(100, 500)
    hist_analyzer.record_event(
        event_type="api_call",
        success=success,
        context={"endpoint": random.choice(["users", "data", "auth"])},
        parameters={"timeout": random.uniform(5, 30)},
        duration_ms=duration,
        tags=["production"]
    )

print(f"   Eventi registrati: {len(hist_analyzer.events)}")

# 2. Analisi storica
print("\n2️⃣ Analisi storica 'api_call':")
analysis = hist_analyzer.analyze("api_call")
print(f"   Total Events: {analysis.total_events}")
print(f"   Success Rate: {analysis.success_rate:.2%}")
print(f"   Avg Duration: {analysis.average_duration:.1f}ms")
print(f"   Trend: {analysis.trend.value} (strength: {analysis.trend_strength:.3f})")
print(f"   Patterns trovati: {len(analysis.patterns)}")
print(f"   Learnings: {len(analysis.learnings)}")

# 3. Pattern identificati
print("\n3️⃣ Pattern identificati:")
for pattern in analysis.patterns[:3]:
    print(f"   - {pattern.name} [{pattern.pattern_type.value}]")
    print(f"     Confidence: {pattern.confidence:.2%}, Insight: {pattern.insight[:50]}...")

# 4. Predizione
print("\n4️⃣ Predizione futura:")
print(f"   Predicted Success Rate: {analysis.predicted_success_rate:.2%}")
print(f"   Prediction Confidence: {analysis.prediction_confidence:.2%}")

# 5. Performance nel tempo
print("\n5️⃣ Performance over time:")
perf = hist_analyzer.get_performance_over_time("api_call", "day")
for p in perf[-3:]:
    print(f"   {p['period']}: {p['success_rate']:.2%} ({p['count']} events)")

# 6. Applicazione apprendimenti
print("\n6️⃣ Applicazione apprendimenti:")
suggestions = hist_analyzer.apply_learning("api_call", {"endpoint": "users"})
print(f"   Recommended params: {suggestions['recommended_params']}")
print(f"   Confidence boost: +{suggestions['confidence_boost']:.2%}")


# ===== TEST GIDEON CORE INTEGRATION =====
print("\n\n🔮 TEST GIDEON CORE (INTEGRAZIONE)")
print("-" * 40)

gideon = GideonCore()

# 1. Calcolo probabilità via GideonCore
print("\n1️⃣ Probability via GideonCore:")
prob = gideon.calculate_probability(context)
print(f"   Success: {prob.success_probability:.2%}")
print(f"   Confidence: {prob.confidence_level.value}")

# 2. Risk assessment via GideonCore
print("\n2️⃣ Risk Assessment via GideonCore:")
risk = gideon.assess_risk(scenario)
print(f"   Risk Level: {risk.overall_risk_level.value}")
print(f"   Risks found: {risk.total_risks}")

# 3. Record & analyze history via GideonCore
print("\n3️⃣ Historical Analysis via GideonCore:")
for i in range(20):
    gideon.record_event("workflow", random.random() > 0.2, duration_ms=random.uniform(50, 200))
hist = gideon.analyze_history("workflow")
print(f"   Events: {hist.total_events}")
print(f"   Success Rate: {hist.success_rate:.2%}")

# 4. Statistics complete
print("\n4️⃣ Statistiche complete GideonCore:")
stats = gideon.get_statistics()
print(f"   Probability Calculator: {stats['probability_calculator']['default_priors']} priors")
print(f"   Risk Analyzer: {stats['risk_analyzer']['assessments_performed']} assessments")
print(f"   Historical Analyzer: {stats['historical_analyzer']['total_events']} events")

print("\n" + "=" * 60)
print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
print("=" * 60)
