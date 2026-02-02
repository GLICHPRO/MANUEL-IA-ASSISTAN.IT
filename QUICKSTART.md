# Gideon 2.0 - Guida di Avvio Rapido

## Installazione

### 1. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configurazione

Copia `.env.example` in `.env` e personalizza le configurazioni:

```bash
cp .env.example .env
```

Modifica `.env` con le tue chiavi API se necessario.

### 3. Frontend Setup

```bash
cd frontend
npm install
```

## Avvio

### Avvia Backend

```bash
cd backend
venv\Scripts\activate
python main.py
```

Il server sarà disponibile su http://localhost:8000

### Avvia Frontend

In un nuovo terminale:

```bash
cd frontend
npm run dev
```

L'interfaccia sarà disponibile su http://localhost:3000

## Test Comandi Vocali

1. Assicurati che il microfono sia attivo
2. Clicca sull'icona del microfono nell'interfaccia
3. Pronuncia: "Gideon, che ore sono?"
4. L'avatar risponderà con animazioni

## Comandi Disponibili

### Comandi Base
- "Gideon, che ore sono?"
- "Gideon, mostra stato sistema"
- "Gideon, analizza il sistema"

### Comandi Avanzati
- "Gideon, suggerisci ottimizzazioni"
- "Gideon, calcola l'efficienza"

### Modalità Pilot (Controllo Avanzato)
1. "Gideon, attiva modalità Pilot"
2. Pronuncia: "Autorizzazione Pilot Alfa Zero Uno"
3. Ora puoi eseguire comandi di controllo

## Troubleshooting

### Backend non si avvia
- Verifica che Python 3.11+ sia installato
- Controlla che tutte le dipendenze siano installate
- Verifica i file di configurazione

### Frontend non si connette
- Verifica che il backend sia in esecuzione su porta 8000
- Controlla le impostazioni CORS in `.env`
- Controlla la console del browser per errori

### Voce non funziona
- Verifica permessi microfono nel browser
- Controlla che PyAudio sia installato correttamente
- Su Windows potrebbe essere necessario installare dipendenze audio

## Architettura

```
Backend (FastAPI) ──WebSocket──> Frontend (React)
        │
        ├── Voice Manager (STT/TTS)
        ├── Brain (NLP + AI)
        ├── Optimizer (Analysis)
        └── Database (PostgreSQL)
```

## Documentazione Completa

Vedi [README.md](README.md) per documentazione completa.

## Supporto

Per problemi o domande: https://github.com/technetpro/gideon2.0/issues
