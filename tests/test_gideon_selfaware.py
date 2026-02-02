"""
Test dei 3 nuovi componenti Gideon: Self-Awareness & Correction
- SelfCorrectionEngine: corregge previsioni errate
- IdentityCore: mantiene coerenza e memoria contestuale
- IntentOutcomeResolver: confronta obiettivi con risultati
"""

import sys
sys.path.insert(0, 'backend')

from gideon import GideonCore
from gideon.self_correction import SelfCorrectionEngine
from gideon.identity_core import IdentityCore, MemoryType, ContextType
from gideon.intent_outcome_resolver import IntentOutcomeResolver


def test_self_correction():
    """Test SelfCorrectionEngine"""
    print("\n" + "="*60)
    print("🔧 TEST SELF-CORRECTION ENGINE")
    print("="*60)
    
    engine = SelfCorrectionEngine()
    
    # 1. Track predictions
    print("\n📊 1. Tracking Predictions...")
    
    pred1 = engine.record_prediction(
        domain="follower_growth",
        predicted_value=150.0,
        confidence=0.8,
        context={"day": "monday"},
        factors=["engagement"]
    )
    print(f"   Prediction 1: {pred1.predicted_value} (ID: {pred1.id})")
    
    pred2 = engine.record_prediction(
        domain="follower_growth",
        predicted_value=120.0,
        confidence=0.7
    )
    print(f"   Prediction 2: {pred2.predicted_value} (ID: {pred2.id})")
    
    # 2. Resolve with actuals
    print("\n📈 2. Resolving Predictions...")
    
    res1 = engine.resolve_prediction(pred1.id, actual_value=130.0)
    if res1:
        pred_dict = res1['prediction']
        err_analysis = res1['error_analysis']
        print(f"   Pred 1: predicted={pred_dict['predicted']}, actual={pred_dict['actual']}")
        print(f"   Error: {err_analysis['error_value']:.2f}, Type: {err_analysis['error_type']}")
    
    res2 = engine.resolve_prediction(pred2.id, actual_value=140.0)
    if res2:
        print(f"   Pred 2: error={res2['error_analysis']['error_value']:.2f}")
    
    # 3. Get accuracy metrics
    print("\n📉 3. Accuracy Metrics...")
    
    metrics = engine.get_accuracy_metrics()
    print(f"   Total predictions: {metrics['total_predictions']}")
    print(f"   Resolved: {metrics['resolved_predictions']}")
    print(f"   MAE: {metrics['mae']:.3f}")
    print(f"   Accuracy rate: {metrics['accuracy_rate']:.2%}")
    
    # 4. Calibration
    print("\n🔄 4. Calibrating Domain...")
    
    # Add more predictions for calibration
    for i in range(5):
        p = engine.record_prediction("follower_growth", 100 + i*10, 0.6)
        engine.resolve_prediction(p.id, 90 + i*12)
    
    cal = engine.calibrate_domain("follower_growth", force=True)
    print(f"   Calibration success: {cal['success']}")
    print(f"   Samples used: {cal.get('samples_used', 'N/A')}")
    
    # 5. Correct new prediction
    print("\n🎯 5. Applying Correction...")
    
    corrected = engine.correct_prediction(
        predicted_value=200.0,
        domain="follower_growth"
    )
    print(f"   Original: {corrected['original']}")
    print(f"   Corrected: {corrected['corrected']:.2f}")
    
    # 6. Status
    print("\n📊 6. Engine Status...")
    status = engine.get_status()
    print(f"   Total predictions: {status['total_predictions']}")
    print(f"   Calibrated domains: {status['calibrated_domains']}")
    
    print("\n✅ Self-Correction Engine: PASSED")
    return True


def test_identity_core():
    """Test IdentityCore"""
    print("\n" + "="*60)
    print("🧠 TEST IDENTITY CORE")
    print("="*60)
    
    identity = IdentityCore()
    
    # 1. Identity
    print("\n👤 1. System Identity...")
    
    id_info = identity.get_identity()
    profile = id_info['profile']
    print(f"   Name: {profile['name']}")
    print(f"   Version: {profile['version']}")
    print(f"   Role: {profile['role']}")
    print(f"   Personality: {', '.join(profile['personality'][:3])}...")
    
    # 2. Introduction
    print("\n🎤 2. Self-Introduction...")
    intro = identity.introduce()
    print(f"   {intro[:80]}...")
    
    # 3. Memory operations
    print("\n💾 3. Memory Operations...")
    
    mem1 = identity.remember(
        {"topic": "user_preference", "value": "detailed_analysis"},
        memory_type=MemoryType.LONG_TERM,
        tags=["user", "preference"],
        persistent=True
    )
    print(f"   Stored long-term: {mem1.id}")
    
    mem2 = identity.remember(
        {"topic": "recent_query", "value": "follower prediction"},
        memory_type=MemoryType.SHORT_TERM,
        tags=["query"]
    )
    print(f"   Stored short-term: {mem2.id}")
    
    # 4. Recall
    print("\n🔍 4. Memory Recall...")
    
    recalled = identity.recall(memory_type=MemoryType.LONG_TERM)
    print(f"   Found {len(recalled)} long-term memories")
    
    recalled_tags = identity.recall(tags=["user"])
    print(f"   Memories with 'user' tag: {len(recalled_tags)}")
    
    # 5. Context management (use valid ContextTypes)
    print("\n📚 5. Context Stack...")
    
    identity.push_context(ContextType.TASK, {"domain": "social_media"})
    print("   Pushed: TASK context")
    
    identity.push_context(ContextType.DOMAIN, {"target": "followers"})
    print("   Pushed: DOMAIN context")
    
    full_ctx = identity.get_full_context()
    print(f"   Context data keys: {list(full_ctx.keys())}")
    
    identity.pop_context()
    print("   Popped context")
    
    # 6. Consistency check
    print("\n✓ 6. Consistency Check...")
    
    check = identity.check_consistency({
        "type": "statement",
        "claim": "GIDEON executes actions"
    })
    print(f"   Consistent: {check.is_consistent}")
    
    # 7. Status
    print("\n📊 7. Identity Status...")
    status = identity.get_status()
    print(f"   Working memory: {status['working_memory_size']}")
    print(f"   Short-term: {status['short_term_memories']}")
    print(f"   Long-term: {status['long_term_memories']}")
    print(f"   Context depth: {status['context_depth']}")
    
    print("\n✅ Identity Core: PASSED")
    return True


def test_intent_outcome_resolver():
    """Test IntentOutcomeResolver"""
    print("\n" + "="*60)
    print("🎯 TEST INTENT-OUTCOME RESOLVER")
    print("="*60)
    
    resolver = IntentOutcomeResolver()
    
    # 1. Register intents
    print("\n📝 1. Registering Intents...")
    
    intent1 = resolver.register_intent(
        description="Aumentare follower di 500",
        expected_outcomes=["Crescita follower +500"],
        success_criteria=["follower_delta >= 500"],
        priority=8,
        context={"campaign": "growth"}
    )
    print(f"   Intent 1: {intent1.description[:35]}... (ID: {intent1.id})")
    
    intent2 = resolver.register_intent(
        description="Migliorare engagement al 5%",
        expected_outcomes=["Engagement 5%"],
        priority=7
    )
    print(f"   Intent 2: {intent2.description[:35]}...")
    
    # 2. Record outcomes
    print("\n📊 2. Recording Outcomes...")
    
    outcome1 = resolver.record_outcome(
        intent_id=intent1.id,
        description="Campagna completata",
        achieved_results=["Crescita follower +420"],
        metrics={"follower_delta": 420},
        side_effects=["Aumento DM spam"]
    )
    print(f"   Outcome 1: {outcome1.id}")
    
    outcome2 = resolver.record_outcome(
        intent_id=intent2.id,
        description="Engagement migliorato",
        achieved_results=["Engagement 4.2%"],
        metrics={"engagement": 4.2}
    )
    print(f"   Outcome 2: {outcome2.id}")
    
    # 3. Resolve intent-outcome
    print("\n🔍 3. Resolving Intent-Outcome...")
    
    resolution1 = resolver.resolve(intent1.id, outcome1.id)
    print(f"   Intent 1 Resolution:")
    print(f"     Match quality: {resolution1.match_quality.value}")
    print(f"     Match score: {resolution1.match_score:.2%}")
    print(f"     Gaps found: {len(resolution1.gaps)}")
    print(f"     Recommended: {resolution1.recommended_action.value}")
    
    # 4. Full reconciliation
    print("\n🔄 4. Full Reconciliation...")
    
    recon = resolver.reconcile(intent1.id, outcome1.id)
    print(f"   Satisfied: {recon.is_satisfied}")
    print(f"   Satisfaction level: {recon.satisfaction_level:.2%}")
    print(f"   Recommended action: {recon.resolution.recommended_action.value}")
    
    if recon.learnings:
        print(f"   Learnings: {recon.learnings[0][:50]}...")
    
    # 5. Pending intents
    print("\n⏳ 5. Pending Intents...")
    
    resolver.register_intent(description="Test pending", priority=5)
    pending = resolver.get_pending_intents()
    print(f"   Pending intents: {len(pending)}")
    
    # 6. Status (use get_status instead of get_gap_statistics)
    print("\n📊 6. Resolver Status...")
    status = resolver.get_status()
    print(f"   Total intents: {status['total_intents']}")
    print(f"   Pending: {status['pending_intents']}")
    print(f"   Resolved: {status['resolved_intents']}")
    print(f"   Reconciliations: {status['reconciliations']}")
    print(f"   Satisfaction rate: {status['satisfaction_rate']:.2%}")
    
    print("\n✅ Intent-Outcome Resolver: PASSED")
    return True


def test_gideon_integration():
    """Test integrazione in GideonCore"""
    print("\n" + "="*60)
    print("🤖 TEST GIDEON CORE INTEGRATION")
    print("="*60)
    
    gideon = GideonCore()
    
    # 1. Self-Correction via GideonCore
    print("\n🔧 1. Self-Correction Integration...")
    
    pred = gideon.track_prediction(
        domain="test_domain",
        predicted_value=100,
        confidence=0.8
    )
    print(f"   Tracked: {pred.id}")
    
    resolved = gideon.resolve_prediction(pred.id, 95)
    if resolved:
        print(f"   Resolved: error={resolved['error_analysis']['error_value']:.2f}")
    
    metrics = gideon.get_accuracy_metrics()
    print(f"   Accuracy: {metrics['accuracy_rate']:.2%}")
    
    # 2. Identity via GideonCore
    print("\n🧠 2. Identity Integration...")
    
    id_info = gideon.get_identity()
    print(f"   Identity: {id_info['profile']['name']} v{id_info['profile']['version']}")
    
    intro = gideon.introduce()
    print(f"   Intro: {intro[:50]}...")
    
    gideon.remember({"key": "test", "value": "integration"}, tags=["test"])
    print("   Memory stored")
    
    recalled = gideon.recall(tags=["test"])
    print(f"   Recalled: {len(recalled)} memories")
    
    # 3. Intent-Outcome via GideonCore
    print("\n🎯 3. Intent-Outcome Integration...")
    
    intent = gideon.register_intent(
        description="Test intent via core",
        expected_outcomes=["Test outcome"],
        priority=5
    )
    print(f"   Intent: {intent.id}")
    
    outcome = gideon.record_outcome(
        intent.id,
        description="Test completed",
        achieved_results=["Test outcome achieved"]
    )
    print(f"   Outcome: {outcome.id}")
    
    resolution = gideon.resolve_intent_outcome(intent.id, outcome.id)
    print(f"   Resolution: {resolution.match_quality.value}, score={resolution.match_score:.2%}")
    
    # 4. Full statistics
    print("\n📊 4. Full Statistics...")
    
    stats = gideon.get_statistics()
    print(f"   Active: {stats['is_active']}")
    print(f"   Self-Correction: {stats['self_correction']['total_predictions']} predictions")
    print(f"   Identity: {stats['identity']['working_memory_size']} working mem")
    print(f"   Intent Resolver: {stats['intent_resolver']['total_intents']} intents")
    
    # Show all modules
    print("\n📦 5. All 17 Modules Active:")
    module_keys = [k for k in stats.keys() if k != 'is_active']
    for key in module_keys:
        print(f"   ✓ {key}")
    
    print("\n✅ GideonCore Integration: PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("       🧠 GIDEON 3.0 - SELF-AWARENESS COMPONENTS TEST")
    print("="*70)
    print("Testing: SelfCorrectionEngine, IdentityCore, IntentOutcomeResolver")
    
    results = []
    
    try:
        results.append(("Self-Correction Engine", test_self_correction()))
    except Exception as e:
        print(f"\n❌ Self-Correction Engine FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Self-Correction Engine", False))
    
    try:
        results.append(("Identity Core", test_identity_core()))
    except Exception as e:
        print(f"\n❌ Identity Core FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Identity Core", False))
    
    try:
        results.append(("Intent-Outcome Resolver", test_intent_outcome_resolver()))
    except Exception as e:
        print(f"\n❌ Intent-Outcome Resolver FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Intent-Outcome Resolver", False))
    
    try:
        results.append(("GideonCore Integration", test_gideon_integration()))
    except Exception as e:
        print(f"\n❌ GideonCore Integration FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("GideonCore Integration", False))
    
    # Summary
    print("\n" + "="*70)
    print("                         📋 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status} - {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        print("\n  GIDEON 3.0 Self-Awareness Components:")
        print("    ✓ SelfCorrectionEngine - Corregge previsioni errate")
        print("    ✓ IdentityCore - Mantiene coerenza e memoria contestuale")  
        print("    ✓ IntentOutcomeResolver - Confronta obiettivi con risultati")
        print("\n  Total Gideon Modules: 17")
    else:
        print("\n  ⚠️  Some tests failed. Check errors above.")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
