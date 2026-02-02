#!/usr/bin/env python3
"""
Test Suite for Gideon 2.0
"""

import asyncio
import sys

import pytest
sys.path.insert(0, "C:\\OneDrive\\OneDrive - Technetpro\\Desktop\\gideon2.0\\backend")

from brain.assistant import GideonAssistant
from brain.optimizer import OptimizerEngine
import time


pytestmark = pytest.mark.asyncio

async def test_assistant():
    """Test the Gideon Assistant"""
    
    print("\n" + "="*60)
    print("🧪 GIDEON 2.0 TEST SUITE")
    print("="*60)
    
    # Initialize assistant
    assistant = GideonAssistant()
    await assistant.initialize()
    
    print("\n✅ Assistant initialized successfully\n")
    
    # Test 1: Time Query
    print("📝 TEST 1: Time Query")
    print("-" * 40)
    response = await assistant.process_command("Che ora è?", mode="text")
    print(f"Command: 'Che ora è?'")
    print(f"Response: {response['text']}")
    print(f"Intent: {response['intent']}")
    print(f"Data: {response.get('data', {})}")
    
    # Test 2: System Status
    print("\n📝 TEST 2: System Status")
    print("-" * 40)
    response = await assistant.process_command("Qual è lo stato del sistema?", mode="text")
    print(f"Command: 'Qual è lo stato del sistema?'")
    print(f"Response: {response['text']}")
    print(f"Intent: {response['intent']}")
    print(f"System Metrics: {response.get('data', {})}")
    
    # Test 3: Analysis Request
    print("\n📝 TEST 3: System Analysis")
    print("-" * 40)
    response = await assistant.process_command("Analizza il sistema", mode="text")
    print(f"Command: 'Analizza il sistema'")
    print(f"Response: {response['text']}")
    print(f"Intent: {response['intent']}")
    
    # Test 4: Optimization Suggestions
    print("\n📝 TEST 4: Optimization Suggestions")
    print("-" * 40)
    response = await assistant.process_command("Suggerisci ottimizzazioni", mode="text")
    print(f"Command: 'Suggerisci ottimizzazioni'")
    print(f"Response: {response['text']}")
    print(f"Intent: {response['intent']}")
    data = response.get('data')
    if data and isinstance(data, list):
        print(f"\n🎯 Top Optimizations:")
        for i, opt in enumerate(data[:3], 1):
            if isinstance(opt, dict):
                print(f"  {i}. {opt.get('description', 'N/A')}")
                print(f"     Impact: +{opt.get('impact_percent', 0):.1f}%")
    
    # Test 5: Avatar Expression
    print("\n📝 TEST 5: Avatar Expressions")
    print("-" * 40)
    intents = ["time", "status", "analysis", "optimization", "information"]
    for intent in intents:
        response = await assistant.process_command(f"test {intent}", mode="text")
        print(f"Intent: {response['intent']:15} → Avatar: {response['avatar_expression']}")
    
    # Test 6: Optimizer Engine Deep Dive
    print("\n📝 TEST 6: Optimizer Deep Analysis")
    print("-" * 40)
    optimizer = assistant.optimizer
    analysis = await optimizer.analyze("system")
    print(f"Target: {analysis['target']}")
    print(f"Efficiency Score: {analysis['score']:.1%}")
    print(f"Total Issues: {len(analysis.get('issues', []))}")
    print(f"Available Optimizations: {len(analysis.get('optimizations', []))}")
    
    if analysis.get('issues'):
        print(f"\n⚠️ Issues Found:")
        for issue in analysis['issues'][:3]:
            print(f"  • [{issue['severity'].upper()}] {issue['component']}: {issue['message']}")
    
    if analysis.get('optimizations'):
        print(f"\n💡 Recommended Optimizations:")
        for opt in analysis['optimizations'][:3]:
            print(f"  • {opt['description']}")
            print(f"    Impact: +{opt['impact_percent']:.1f}% | Priority: {opt['priority']}")
    
    # Test 7: Comprehensive Analysis
    print("\n📝 TEST 7: Comprehensive System Analysis")
    print("-" * 40)
    comprehensive = await optimizer.comprehensive_analysis("system")
    print(f"Overall System Score: {comprehensive['overall_score']:.1%}")
    print(f"Total Issues Detected: {comprehensive['total_issues']}")
    print(f"Top Optimizations Available: {len(comprehensive['top_optimizations'])}")
    
    # Test 8: NLP Intent Recognition
    print("\n📝 TEST 8: NLP Intent Recognition")
    print("-" * 40)
    test_queries = [
        "Che ore sono?",
        "Come sta il sistema?",
        "Analizza le performance",
        "Quali sono i suggerimenti?",
        "Come posso migliorare?",
    ]
    
    for query in test_queries:
        intent_result = await assistant.nlp.extract_intent(query)
        print(f"Query: '{query}'")
        print(f"  → Intent: {intent_result['intent']} ({intent_result['confidence']:.0%} confidence)")
    
    # Test 9: Sentiment Analysis
    print("\n📝 TEST 9: Sentiment Analysis")
    print("-" * 40)
    sentiments = [
        "Gideon funziona perfettamente!",
        "C'è un problema critico",
        "Non so cosa pensare",
        "Ottimo lavoro, fantastico!",
        "Pessimo, non funziona",
    ]
    
    for text in sentiments:
        sentiment = await assistant.nlp.analyze_sentiment(text)
        print(f"Text: '{text}'")
        print(f"  → Positive: {sentiment['positive']:.0%}, Negative: {sentiment['negative']:.0%}")
    
    # Shutdown
    await assistant.shutdown()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_assistant())
