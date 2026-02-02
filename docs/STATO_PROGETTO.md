# 🤖 GIDEON 3.0 + JARVIS CORE - Stato del Progetto

**Ultimo aggiornamento:** 24 Gennaio 2026

---

## 🎯 Architettura Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      UTENTE                                  │
│                   (Voce/Testo)                               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                               │
│            (Coordinatore Pipeline)                           │
└─────────────────────────────────────────────────────────────┘
          │                                    │
          ▼                                    ▼
┌──────────────────────┐          ┌──────────────────────────┐
│    GIDEON 3.0        │          │      JARVIS CORE         │
│   (Analitico)        │          │     (Esecutivo)          │
│                      │          │                          │
│ • Analyzer           │◄────────►│ • IntentInterpreter      │
│ • Predictor          │          │ • DecisionMaker          │
│ • Simulator          │          │ • Executor               │
│ • Ranker             │          │ • SecurityManager        │
│                      │          │ • Automator              │
│ NON ESEGUE MAI       │          │ DECIDE ED ESEGUE         │
└──────────────────────┘          └──────────────────────────┘
```

---

## 📁 Posizione Progetto
```
C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0
```

---

## 🚀 Come Avviare

### Backend (API Server - Porta 8001)
```powershell
cd "C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\backend"
& "C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

### Frontend (Web Server - Porta 3000)
```powershell
cd "C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\frontend"
& "C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\.venv\Scripts\python.exe" -m http.server 3000
```

---

## 🔗 Link di Accesso

| Servizio | URL |
|----------|-----|
| 🌐 Frontend | http://127.0.0.1:3000 |
| ⚙️ Backend API | http://127.0.0.1:8001 |
| 📚 API Docs | http://127.0.0.1:8001/api/docs |
| ❤️ Health Check | http://127.0.0.1:8001/health |
| 🔌 WebSocket | ws://127.0.0.1:8001/ws |

---

## 📂 Struttura Progetto

```
gideon2.0/
├── .venv/                    # Ambiente virtuale Python
├── backend/
│   ├── main.py               # Entry point FastAPI
│   ├── requirements.txt      # Dipendenze Python
│   │
│   ├── gideon/               # 🧠 GIDEON 3.0 - Modulo Analitico
│   │   ├── __init__.py       # GideonCore coordinator
│   │   ├── analyzer.py       # Analisi richieste
│   │   ├── predictor.py      # Previsioni conseguenze
│   │   ├── simulator.py      # Simulazione scenari
│   │   └── ranker.py         # Classificazione opzioni
│   │
│   ├── jarvis/               # ⚡ JARVIS CORE - Modulo Esecutivo
│   │   ├── __init__.py       # JarvisCore + pipeline cognitiva
│   │   ├── intent_interpreter.py  # Interpretazione intent
│   │   ├── decision_maker.py      # Valutazione e decisioni
│   │   ├── executor.py       # Esecuzione azioni
│   │   ├── security.py       # Sicurezza e permessi
│   │   ├── automator.py      # Automazioni e routine
│   │   └── controller.py     # Controllo sistema
│   │
│   ├── core/                 # 🔧 Core Infrastructure
│   │   ├── mode_manager.py   # Modalità (PASSIVE/COPILOT/PILOT/EXECUTIVE)
│   │   ├── orchestrator.py   # Coordinatore pipeline
│   │   ├── action_logger.py  # Log azioni + rollback
│   │   ├── emergency.py      # Kill switch + emergenze
│   │   ├── plugin_manager.py # Sistema plugin
│   │   ├── agent_bus.py      # Comunicazione multi-agente
│   │   ├── voice_activation.py # Attivazione vocale
│   │   └── config.py         # Configurazione
│   │
│   ├── plugins/              # 🔌 Plugin System
│   │   └── example_plugin.py # Plugin esempio
│   │
│   ├── api/
│   │   └── routes.py         # Endpoint API REST
│   ├── brain/
│   │   ├── assistant.py      # Cervello legacy (164KB)
│   │   ├── ai_providers.py   # Provider AI multipli
│   │   └── ...
│   ├── database/
│   │   └── ...
│   └── voice/
│       └── ...
│
├── frontend/
│   ├── index.html            # Interfaccia web
│   └── ...
│
└── *.bat                     # Script avvio
```

---

## 🎛️ Modalità Operative

| Modalità | Autonomia | Comportamento |
|----------|-----------|---------------|
| **PASSIVE** | 0% | Solo analisi e suggerimenti |
| **COPILOT** | 50% | Chiede conferma per azioni |
| **PILOT** | 100% | Esegue autonomamente (hands-free) |
| **EXECUTIVE** | 100%+ | Jarvis Mode - orchestrazione completa |

### Livelli Risposta
| Livello | Stile |
|---------|-------|
| **NORMAL** | Amichevole, breve, emoji |
| **ADVANCED** | Tecnico, dettagliato |

---

## 🔊 Comandi Vocali

### Cambio Modalità
- "Assistente modalità passiva/copilota/pilota"
- "Jarvis prendi il controllo" → EXECUTIVE
- "Gideon analizza solo" → PASSIVE

### Cambio Livello
- "Modalità avanzata/tecnica"
- "Modalità normale/semplice"

### Emergenza
- "EMERGENZA STOP" → Kill switch
- "Blocca tutto"

---

## ✅ Componenti Implementati

### GIDEON 3.0 (Analitico) ✅
- [x] Analyzer - Analisi semantica
- [x] Predictor - Previsioni conseguenze
- [x] Simulator - Scenari what-if
- [x] Ranker - Classificazione opzioni
- [x] GideonCore - Coordinatore

### JARVIS CORE (Esecutivo) ✅
- [x] IntentInterpreter - NLP italiano
- [x] DecisionMaker - Valutazione alternative
- [x] Executor - Esecuzione 15+ azioni
- [x] SecurityManager - Permessi e PIN
- [x] Automator - Task schedulati
- [x] SystemController - Controllo OS

### Core Infrastructure ✅
- [x] ModeManager - 4 modalità + 2 livelli
- [x] Orchestrator - Pipeline cognitiva
- [x] ActionLogger - Log + rollback
- [x] EmergencySystem - Kill switch
- [x] PluginManager - Estensibilità
- [x] AgentBus - Multi-agente
- [x] VoiceActivation - Trigger vocali

---

## 🔄 Pipeline Cognitiva

```
INPUT → Intent → [Gideon Analysis] → Decision → Execute → RESPONSE
         │              │                │          │
         │         (se complesso)        │          │
         │              │                │          │
         └──────────────┴────────────────┴──────────┘
                     Cognitive Trace
```

---

## ✅ Funzionalità Legacy (da Gideon 2.0)

### 🧮 Calcoli
- `quanto fa 25 più 17` → `42`
- `calcola 100 diviso 4` → `25`
- Operazioni matematiche complete

### 🌤️ Meteo
- `che tempo fa a Roma`
- Open-Meteo API gratuita

### 📚 Informazioni
- Wikipedia, traduzioni, definizioni
- Conversioni valuta, notizie

### ⏰ Ora e Data
- `che ore sono`
- `che giorno è oggi`

### 🌐 Apertura Siti/App
- `apri YouTube/Google/calcolatrice`

### 📊 Sistema
- `stato del sistema`
- CPU, memoria, disco

---

## 🧪 Test

```powershell
cd "C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0"
& ".venv\Scripts\python.exe" -m pytest test_*.py -v
```

---

## 📝 TODO

- [ ] Integrazione API routes con nuova pipeline
- [ ] Frontend aggiornato per modalità
- [ ] Voice recognition continuo
- [ ] Plugin community
- [ ] Memory persistence
- [ ] Learning module

---

## 🔄 Stato Server

```powershell
Get-NetTCPConnection -LocalPort 8001,3000 -ErrorAction SilentlyContinue | Where-Object State -eq 'Listen'
```

---

*GIDEON 3.0 + JARVIS CORE - Sistema Cognitivo Autonomo*
