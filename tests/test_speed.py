"""Test velocità risposta GIDEON 3.0"""
import asyncio
import time
import sys
import os

# Carica .env
from dotenv import load_dotenv
load_dotenv()

# Aggiungi backend al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def main():
    print("=" * 50)
    print("  TEST VELOCITA' GIDEON 3.0 OTTIMIZZATO")
    print("=" * 50)
    print()
    
    # Import diretto dei componenti
    from brain.ai_providers import get_ai_manager
    
    print("Inizializzazione AI Manager...")
    ai_manager = get_ai_manager()
    await ai_manager.initialize()
    print(f"Provider disponibili: {len(ai_manager.priority_order)}")
    print(f"Provider default: {ai_manager.default_provider}")
    print()
    
    # Test 1: Query semplice
    print("TEST 1: Query semplice")
    q1 = "Ciao, come stai?"
    start = time.time()
    r1 = await ai_manager.generate(prompt=q1)
    t1 = (time.time() - start) * 1000
    print(f"  Tempo: {int(t1)} ms")
    print(f"  Risposta: {r1.content[:80]}...")
    print()
    
    # Test 2: Stessa query (no cache a questo livello)
    print("TEST 2: Seconda query")
    q2 = "Quanto fa 10+5?"
    start = time.time()
    r2 = await ai_manager.generate(prompt=q2)
    t2 = (time.time() - start) * 1000
    print(f"  Tempo: {int(t2)} ms")
    print(f"  Risposta: {r2.content[:80]}...")
    print()
    
    # Test 3: Query breve
    print("TEST 3: Query breve")
    q3 = "Dimmi si o no"
    start = time.time()
    r3 = await ai_manager.generate(prompt=q3)
    t3 = (time.time() - start) * 1000
    print(f"  Tempo: {int(t3)} ms")
    print(f"  Risposta: {r3.content[:80]}...")
    print()
    
    # Risultati
    print("=" * 50)
    print("  RISULTATI")
    print("=" * 50)
    avg = (t1 + t2 + t3) / 3
    print(f"  Media: {int(avg)} ms")
    print(f"  Min: {int(min(t1, t2, t3))} ms")
    print(f"  Max: {int(max(t1, t2, t3))} ms")
    print()
    
    if avg < 3000:
        print("  ✅ VELOCITA' BUONA!")
    elif avg < 5000:
        print("  ⚠️ VELOCITA' MEDIA")
    else:
        print("  ❌ VELOCITA' LENTA - dipende dalla latenza API")
    
    # Cleanup
    await ai_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
