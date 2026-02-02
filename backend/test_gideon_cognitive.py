# Test per Gideon Advanced AI Components
# Temporal Reasoning, Goal Management, Meta-Cognition

import sys
sys.path.insert(0, r'c:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\backend')

from datetime import datetime, timedelta

print("=" * 60)
print("🔮 TEST GIDEON ADVANCED AI COMPONENTS")
print("=" * 60)

# ============================================
# 1. TEST TEMPORAL REASONING
# ============================================
print("\n📅 TEST 1: TEMPORAL REASONING")
print("-" * 40)

from gideon import (
    TemporalReasoning, TimeHorizon, ImpactType,
    TemporalEvent
)

temporal = TemporalReasoning()

# Create events
now = datetime.now()
event1 = temporal.create_event(
    name="Backup Database",
    start_time=now,
    duration=timedelta(minutes=30),
    priority=8,
    impact_type="persistent"
)
print(f"  Evento 1: {event1.name} ({event1.id})")

event2 = temporal.create_event(
    name="Deploy Update",
    start_time=now + timedelta(hours=1),
    duration=timedelta(minutes=15),
    depends_on=[event1.id],
    priority=9
)
print(f"  Evento 2: {event2.name} (dipende da {event1.id})")

# Analyze impact
action = {
    "id": "deploy_001",
    "type": "system_update",
    "impact": 0.7,
    "urgency": 0.8
}

impact = temporal.analyze_impact(action, {"system_health": 0.85})
print(f"\n  📊 Impatto Temporale:")
print(f"     Tipo: {impact.impact_type.value}")
print(f"     Immediato: {impact.immediate_impact:.2f}")
print(f"     Breve termine: {impact.short_term_impact:.2f}")
print(f"     Medio termine: {impact.medium_term_impact:.2f}")
print(f"     Lungo termine: {impact.long_term_impact:.2f}")
print(f"     Totale: {impact.total_impact():.2f}")
print(f"     Reversibilità: {impact.reversal_possibility:.0%}")

# Suggest optimal timing
timing = temporal.suggest_optimal_timing(action)
print(f"\n  ⏰ Timing Ottimale: {timing['suggested_timing']}")
print(f"     Motivo: {timing['reason']}")

# Find conflicts
conflicts = temporal.find_conflicts()
print(f"\n  ⚠️ Conflitti trovati: {len(conflicts)}")

print("\n✅ Temporal Reasoning OK")

# ============================================
# 2. TEST GOAL MANAGEMENT
# ============================================
print("\n🎯 TEST 2: GOAL MANAGEMENT")
print("-" * 40)

from gideon import (
    GoalManagement, GoalStatus, GoalPriority, GoalType
)

goals = GoalManagement()

# Create main goal
main_goal = goals.create_goal(
    name="Ottimizzazione Sistema",
    description="Migliorare performance del sistema del 20%",
    goal_type=GoalType.OUTCOME,
    priority=GoalPriority.HIGH,
    deadline=datetime.now() + timedelta(days=30),
    template="system_optimization"
)
print(f"  Goal principale: {main_goal.name} ({main_goal.id})")
print(f"     Priorità: {main_goal.priority.value}")
print(f"     Tipo: {main_goal.goal_type.value}")
print(f"     Metriche: {len(main_goal.metrics)}")

# Create sub-goals
sub1 = goals.create_sub_goal(
    parent_id=main_goal.id,
    name="Ottimizzare Database",
    description="Query optimization e indexing"
)
print(f"  Sub-goal 1: {sub1.name}")

sub2 = goals.create_sub_goal(
    parent_id=main_goal.id,
    name="Cache Layer",
    description="Implementare caching Redis"
)
print(f"  Sub-goal 2: {sub2.name}")

# Update progress
goals.update_status(main_goal.id, GoalStatus.IN_PROGRESS)
goals.update_metric(main_goal.id, "performance_score", 75)
print(f"\n  📈 Progresso dopo aggiornamento: {main_goal.progress:.1f}%")

# Analyze health
health = goals.analyze_goal_health(main_goal.id)
print(f"\n  🏥 Analisi Salute Goal:")
print(f"     Overall Health: {health['overall_health']:.0%}")
print(f"     At Risk: {health['is_at_risk']}")
for indicator, value in health['indicators'].items():
    print(f"     {indicator}: {value:.2f}")

# Prioritize
active = goals.get_active_goals()
print(f"\n  📋 Goal attivi prioritizzati: {len(active)}")

# Get statistics
stats = goals.get_statistics()
print(f"\n  📊 Statistiche:")
print(f"     Totale goal: {stats['total_goals']}")
print(f"     Progresso medio: {stats['average_progress']:.1f}%")

print("\n✅ Goal Management OK")

# ============================================
# 3. TEST META-COGNITION
# ============================================
print("\n🧠 TEST 3: META-COGNITION")
print("-" * 40)

from gideon import (
    MetaCognition, UncertaintyType, ReflectionType
)

meta = MetaCognition()

# Assess uncertainty
uncertainty = meta.assess_uncertainty(
    domain="user_behavior",
    context={"data_quality": 0.6, "completeness": 0.7}
)
print(f"  📊 Valutazione Incertezza (user_behavior):")
print(f"     Tipo: {uncertainty.uncertainty_type.value}")
print(f"     Livello: {uncertainty.level:.2f}")
print(f"     Riducibile: {uncertainty.is_reducible}")
print(f"     Fonti: {', '.join(uncertainty.sources) if uncertainty.sources else 'N/A'}")
print(f"     Strategia: {uncertainty.reduction_strategy}")

# Check limitations
task = {
    "type": "real_time_prediction",
    "domains": ["real_time_events", "live_data"]
}
limitations = meta.check_limitations(task)
print(f"\n  ⚠️ Limitazioni rilevate: {len(limitations)}")
for limit in limitations[:3]:
    print(f"     - {limit.description} (severità: {limit.severity:.1f})")

# Acknowledge limitations
ack = meta.acknowledge_limitation(task)
print(f"\n  📝 Acknowledgement:")
print(f"     Ha limitazioni: {ack['has_limitations']}")
print(f"     Può procedere: {ack['can_proceed']}")
print(f"     Modificatore confidenza: {ack['confidence_modifier']:.2f}")
if ack.get('disclaimer'):
    print(f"     Disclaimer: {ack['disclaimer']}")

# Self-reflection
reflection = meta.reflect(
    subject="prediction_accuracy_test",
    reflection_type=ReflectionType.PREDICTION,
    predicted_confidence=0.85,
    actual_outcome=0.72,
    context={"domain": "performance"}
)
print(f"\n  🔍 Auto-Riflessione:")
print(f"     Soggetto: {reflection.subject}")
print(f"     Calibrazione: {reflection.calibration.value}")
print(f"     Predetto: {reflection.predicted_confidence:.2f}")
print(f"     Effettivo: {reflection.actual_outcome:.2f}")
print(f"     Lezioni: {len(reflection.lessons_learned)}")
if reflection.lessons_learned:
    print(f"       → {reflection.lessons_learned[0]}")

# Meta-cognitive state
state = meta.get_meta_state()
print(f"\n  🧠 Stato Meta-Cognitivo:")
print(f"     Calibrazione: {state.overall_calibration.value}")
print(f"     Self-awareness: {state.self_awareness_score:.2f}")
print(f"     Incertezza totale: {state.total_uncertainty:.2f}")
print(f"     Limitazioni attive: {len(state.active_limitations)}")

# What I know / don't know
knows = meta.what_i_know()
print(f"\n  📚 Conoscenza:")
print(f"     Domini tracciati: {knows['total_domains']}")

dont_know = meta.what_i_dont_know()
print(f"     Limitazioni note: {len(dont_know['limitations'])}")

print("\n✅ Meta-Cognition OK")

# ============================================
# 4. TEST GIDEON CORE INTEGRATION
# ============================================
print("\n🔮 TEST 4: GIDEON CORE INTEGRATION")
print("-" * 40)

from gideon import GideonCore

gideon = GideonCore()

# Test temporal through core
event = gideon.create_temporal_event(
    name="Test Event",
    start_time=datetime.now(),
    duration=timedelta(minutes=10)
)
print(f"  TemporalReasoning: Evento '{event.name}' creato")

# Test goal through core
goal = gideon.create_goal(
    name="Test Goal",
    description="Obiettivo di test"
)
print(f"  GoalManagement: Goal '{goal.name}' creato")

# Test meta-cognition through core
unc = gideon.assess_uncertainty("test_domain")
print(f"  MetaCognition: Incertezza '{unc.domain}' = {unc.level:.2f}")

# Test what I know
knows = gideon.what_i_know()
print(f"  MetaCognition: {knows['total_domains']} domini conosciuti")

# Get full statistics
stats = gideon.get_statistics()
print(f"\n  📊 Statistiche Gideon Complete:")
print(f"     Temporal: {stats.get('temporal_reasoning', {}).get('total_events', 0)} eventi")
print(f"     Goals: {stats.get('goal_management', {}).get('total_goals', 0)} obiettivi")
print(f"     MetaCog: {stats.get('meta_cognition', {}).get('reflections_count', 0)} riflessioni")

print("\n✅ GideonCore Integration OK")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("📊 RIEPILOGO MODULI GIDEON 3.0")
print("=" * 60)
print("""
  MODULI COMPLETI (14 totali):
  
  Core:
  ├── Predictor - Previsioni
  ├── Analyzer - Analisi dati
  ├── Simulator - Simulazione Monte Carlo/Bayesian
  ├── DecisionRanker - Ranking TOPSIS/Pareto
  └── SuggestionEngine - Suggerimenti intelligenti
  
  Advanced Simulation:
  ├── ScenarioBuilder - Costruzione scenari
  └── SimulationEngine - Simulazioni parallele
  
  Analytics:
  ├── ProbabilityCalculator - Calcolo probabilità
  ├── RiskAnalyzer - Analisi rischi
  └── HistoricalAnalyzer - Apprendimento storico
  
  Cognitive (NEW):
  ├── TemporalReasoning - Ragionamento temporale
  ├── GoalManagement - Gestione obiettivi  
  └── MetaCognition - Auto-valutazione
""")
print("=" * 60)
print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
print("=" * 60)
