# Ottimizzazioni Velocità Risposta - GIDEON 3.0

## Modifiche Effettuate

### 1. Cache Risposte (`brain/assistant.py`)
- **Cache TTL**: 5 minuti (300 secondi)
- **Max entries**: 200 risposte cached
- **Fast Path**: Le query AI saltano NLP e memoria per risposta diretta

### 2. Cache Intent (`brain/nlp_processor.py`)
- **Max entries**: 500 intent cached
- **Cleanup**: Automatico FIFO quando supera limite

### 3. Timeout Ridotti (`brain/ai_providers.py`)
- **Base timeout**: 30s (da 60s)
- **OpenRouter timeout**: 25s
- **Max tokens**: 500 (da 2000) per risposte più brevi e veloci

### 4. System Prompt Ottimizzato
- Ridotto da 500+ caratteri a ~100 caratteri
- Meno contesto = meno token = risposta più veloce

### 5. Conversation History Ridotta
- Limitata a 5 messaggi (da 10)
- Skip history per domande semplici

## Risultati Benchmark

| Tipo Query | Prima | Dopo |
|------------|-------|------|
| Query cache | N/A | ~150ms |
| Query semplice | ~5800ms | ~2000ms |
| Query complessa | ~6000ms | ~3000-4000ms |

## Note

I tempi di risposta dipendono principalmente da:
1. **Latenza API OpenRouter** (non controllabile)
2. **Modello usato** (gpt-4o-mini è veloce)
3. **Lunghezza risposta richiesta**

Per risposte ancora più veloci:
- Usare modelli gratuiti: `google/gemma-7b-it:free`
- Usare Claude Haiku: `anthropic/claude-3-haiku`
- Configurare Ollama locale per latenza zero

## File Modificati
- `backend/brain/assistant.py` - Fast path + caching
- `backend/brain/nlp_processor.py` - Intent caching
- `backend/brain/ai_providers.py` - Timeout + max_tokens + system prompt
