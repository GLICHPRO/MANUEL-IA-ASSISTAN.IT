# 🎭 GIDEON 2.0 - APPLICAZIONE DESKTOP WINDOWS

## 📋 SPECIFICHE COMPLETE

### Piattaforma
- **OS**: Windows 10/11
- **Architettura**: Desktop Application + Web Backend
- **Linguaggio**: Python (Backend) + React + Electron (Frontend)

### Input
- 🎤 **Microfono**: Riconoscimento vocale continuo con wake word "Gideon"
- ⌨️ **Tastiera**: Input testuale tramite chat interface

### Output
- 🔊 **Speaker/Casse**: Sintesi vocale (TTS) con voce italiana
- 🖥️ **Interfaccia Grafica**: 
  - Avatar 3D animato con espressioni
  - Chat interface moderna
  - Dashboard con metriche percentuali
  - Analisi dati in tempo reale

---

## 🚀 INSTALLAZIONE RAPIDA

### Fase 1: Setup Backend (Python)

```bash
cd C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\backend

# Attiva l'ambiente virtuale
C:\OneDrive\OneDrive - Technetpro\Desktop\gideon\.venv\Scripts\activate.ps1

# Installa dipendenze
pip install fastapi uvicorn pydantic loguru pydantic-settings aiofiles

# (Opzionale) Installa dipendenze audio su Windows
pip install sounddevice soundfile
```

### Fase 2: Setup Frontend (Node.js)

```bash
cd C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\frontend

# Installa dipendenze
npm install

# (Per desktop app con Electron)
npm install electron electron-builder electron-is-dev wait-on concurrently --save-dev
```

---

## 🎯 AVVIO DELL'APPLICAZIONE

### Opzione 1: Modalità Sviluppo (Dev)

**Terminale 1 - Backend**
```bash
cd backend
Activate-VirtualEnv  # Attiva venv Python
python main.py
```
Server in ascolto su: `http://localhost:8001`

**Terminale 2 - Frontend (React)**
```bash
cd frontend
npm start
```
App in ascolto su: `http://localhost:3000`

### Opzione 2: Desktop App Electron (Windows)

```bash
cd frontend
npm run dev  # Avvia sia React che Electron in parallelo
```

---

## 📊 DASHBOARD E METRICHE

### Visualizzazione Real-Time
- **CPU**: Percentuale di utilizzo con barra di progresso
- **Memoria**: Consumo RAM in tempo reale
- **Disco**: Spazio disponibile e utilizzato
- **Response Time**: Latenza media dei comandi

### Suggerimenti Intelligenti
- Calcoli percentuali di impatto
- Priorità (High, Medium, Low)
- Descriptions dettagliate

Esempio:
```
💡 Ottimizzazione: Ridurre il carico CPU
   Impact: +25.0% improvement
   Priority: HIGH
```

---

## 🎤 COMANDI VOCALI SUPPORTATI

### Attivazione Base
```
"Gideon, che ore sono?"
"Gideon, mostra lo stato del sistema"
"Gideon, analizza il sistema"
"Gideon, suggerisci ottimizzazioni"
```

### Modalità Pilot (Controllo Avanzato)
```
1. "Gideon, attiva modalità Pilot"
2. (Autenticazione) "Autorizzazione Pilot Alfa Zero Uno"
3. Comando: "Deploy applicazione" / "Riavvia servizio"
```

---

## 🎭 AVATAR 3D

### Espressioni Facciali
| Espressione | Quando | Colore |
|-------------|--------|--------|
| 😊 **Happy** | Risposte positive | Verde |
| 🤔 **Thinking** | Elaborazione | Arancione |
| 😐 **Neutral** | Standard | Blu |
| 👁️ **Focused** | Analisi intensa | Azzurro |
| 😟 **Concerned** | Problemi rilevati | Rosso |
| 💪 **Confident** | Controllo eseguito | Viola |

### Animazioni
- ✅ Lip-sync sincronizzato con voce
- ✅ Respirazione naturale (idle)
- ✅ Movimento oculare
- ✅ Espressioni microgesture

---

## 🔧 CONFIGURAZIONE AUDIO (WINDOWS)

### Microphone Setup
1. **Impostazioni Windows** → Sound Settings
2. Seleziona microfono predefinito
3. Testa il volume (dovrebbe essere ~80%)

### Speaker Output
1. **Impostazioni Windows** → Sound → Volume
2. Seleziona speaker predefinito
3. Gideon userà questa configurazione per il TTS

### Modifica Voce TTS
```python
# In backend/core/config.py
TTS_VOICE: str = "it-IT-ElsaNeural"  # Voce italiana
# Opzioni: it-IT-Liv, it-IT-ElsaNeural, etc.
```

---

## 📈 METRICHE E PERCENTUALI

### Sistema Operativo
```
CPU:     46.8%  [████████░░] GOOD
Memory:  89.2%  [██████████] WARNING ⚠️
Disk:    77.6%  [█████████░] GOOD
```

### Analisi Dettagliata
- Cache Hit Rate: **87.3%** ✅
- Query Response: **124ms** ✅
- Network Latency: **12ms** ✅
- Error Rate: **0.02%** ✅

### Suggerimenti Ottimizzazione
```
🎯 TOP 3 OPTIMIZATIONS:

1. Ottimizzare la memoria con caching
   Impact: +30.0% improvement
   Time to implement: 15 min

2. Ridurre il carico CPU
   Impact: +25.0% improvement
   Time to implement: 10 min

3. Migliorare query database
   Impact: +40.0% improvement
   Time to implement: 20 min
```

---

## 🔐 MODALITÀ PILOT (Controllo Avanzato)

### Attivazione
1. Pronuncia: "Gideon, attiva modalità Pilot"
2. Gideon chiede autenticazione
3. Pronuncia frase: "Autorizzazione Pilot Alfa Zero Uno"
4. ✅ Modalità Pilot attivata

### Comandi Disponibili
- Deploy applicazione
- Riavvia servizio
- Rollback deployment
- Stop processo
- Restart sistema

### Log di Audit
- Tutti i comandi Pilot vengono registrati
- Timestamp e utente
- Azione eseguita e risultato

---

## 📱 INTERFACCIA UTENTE

### Layout Desktop
```
┌─ HEADER ─────────────────────────────────────┐
│ GIDEON 2.0 | Status | Audio | Pilot         │
├─────────────────────────────────────────────┤
│          │                                  │
│ AVATAR   │    CHAT INTERFACE               │
│ 3D       │                                  │
│          │                                  │
│ METRICS  │    MESSAGE HISTORY              │
│          │                                  │
│ ACTIONS  │   INPUT + SEND                  │
│          │                                  │
└─────────────────────────────────────────────┘
```

### Chat Area
- Messaggio utente: Blu ← Right align
- Messaggio Gideon: Viola → Left align
- Timestamp per ogni messaggio
- Scroll automatico all'ultimo messaggio

### Sidebar Sinistro
- Avatar 3D (120x120px)
- Stato attuale
- Azioni rapide (bottoni)
- Metriche sistema con barre di progresso

---

## 🛠️ TROUBLESHOOTING

### Backend non si avvia
```bash
# Verifica che la porta 8001 sia libera
netstat -ano | findstr :8001

# Se occupata, cambia porta in config.py
PORT: int = 8002
```

### Frontend non si connette
```bash
# Verifica che il backend sia in ascolto
curl http://localhost:8001/health

# Controlla CORS in .env
CORS_ORIGINS=["http://localhost:3000"]
```

### Microfono non funziona
```bash
# Installa librerie audio su Windows
pip install sounddevice soundfile

# Verifica periferica
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Avatar non si anima
```bash
# Verifica Three.js
npm ls three @react-three/fiber

# Reinstalla se necessario
npm install three@latest @react-three/fiber@latest
```

---

## 📊 COMANDI DI TEST

```bash
# Test backend
curl http://localhost:8001/
curl http://localhost:8001/health
curl http://localhost:8001/api/docs

# Test dal Python
cd gideon2.0
python test_gideon.py

# Output atteso:
# ✅ Assistant initialized successfully
# 📝 TEST 1: Time Query ✅
# 📝 TEST 2: System Status ✅
# ... (tutti i test passano)
```

---

## 🎁 Bonus Features

- ✅ Dark Mode (default)
- ✅ Light Mode (opzionale)
- ✅ Registrazione conversazioni
- ✅ Export metriche (CSV/JSON)
- ✅ Notifiche desktop
- ✅ Sistema di plugin

---

## 📞 Supporto

**Documentazione**: [README.md](README.md)  
**Quick Start**: [QUICKSTART.md](QUICKSTART.md)  
**Issues**: https://github.com/technetpro/gideon2.0/issues  
**Email**: info@technetpro.com

---

**Versione**: 2.0.0  
**Ultima Update**: 14 Gennaio 2026  
**Status**: 🟢 Production Ready
