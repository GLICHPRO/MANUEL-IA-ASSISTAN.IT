# GIDEON TTS - Sistema Vocale

Sistema Text-to-Speech per GIDEON usando Google Cloud TTS.

## 📁 Struttura

```
gideon-tts/
├── package.json        # Dipendenze npm
├── credentials.json    # ← Credenziali Google Cloud (da aggiungere)
├── tts.js              # Core sintesi vocale
├── audio.js            # Riproduzione audio
├── gideon.js           # Collegamento al backend
└── README.md           # Questa guida
```

## 🚀 Setup

### 1. Installa dipendenze

```bash
cd gideon-tts
npm install
```

### 2. Configura Google Cloud TTS

1. Vai su [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuovo progetto o seleziona uno esistente
3. Abilita l'API "Cloud Text-to-Speech"
4. Crea una Service Account Key:
   - Vai su "IAM & Admin" > "Service Accounts"
   - Crea nuovo account o usa esistente
   - Crea una nuova chiave JSON
5. Scarica il file JSON e salvalo come `credentials.json` in questa cartella

### 3. Avvia

```bash
npm start
```

## 📋 Comandi Interattivi

| Comando | Descrizione |
|---------|-------------|
| `/stop` | Ferma la riproduzione |
| `/voice` | Mostra voci disponibili |
| `/speed 1.2` | Cambia velocità (0.25-4.0) |
| `/test` | Test della voce |
| `/quit` | Esci |

## 🎤 Voci Italiane Disponibili

| Nome | Tipo | Genere |
|------|------|--------|
| it-IT-Wavenet-A | Wavenet (naturale) | Femminile |
| it-IT-Wavenet-B | Wavenet (naturale) | Femminile |
| it-IT-Wavenet-C | Wavenet (naturale) | Maschile |
| it-IT-Wavenet-D | Wavenet (naturale) | Maschile |
| it-IT-Standard-A | Standard | Femminile |
| it-IT-Standard-B | Standard | Femminile |
| it-IT-Standard-C | Standard | Maschile |
| it-IT-Standard-D | Standard | Maschile |

**Default**: `it-IT-Wavenet-C` (voce maschile naturale)

## 💰 Costi Google Cloud TTS

- **Standard voices**: $4 per 1 milione di caratteri
- **WaveNet voices**: $16 per 1 milione di caratteri
- **Free tier**: 1 milione di caratteri/mese (Standard), 1 milione di caratteri/mese (WaveNet)

## 🔧 Uso Programmatico

```javascript
const tts = require('./tts');
const audio = require('./audio');

// Sintesi e riproduzione
const buffer = await tts.synthesize("Ciao, sono GIDEON!");
await audio.playBuffer(buffer);

// Cambia voce
tts.setVoice('it-IT-Wavenet-A', 'FEMALE');

// Cambia velocità
tts.setSpeed(1.2);
```

## ⚠️ Troubleshooting

### "credentials.json non trovato"
Assicurati di aver scaricato le credenziali da Google Cloud e salvato il file come `credentials.json` nella cartella `gideon-tts/`.

### Audio non funziona su Windows
Il sistema usa PowerShell per riprodurre MP3. Assicurati che PowerShell possa accedere a `System.Windows.Media.MediaPlayer`.

### Errore di autenticazione Google
Verifica che:
1. L'API Text-to-Speech sia abilitata nel progetto
2. Le credenziali siano valide e non scadute
3. Il service account abbia i permessi corretti
