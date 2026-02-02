/**
 * GIDEON TTS - Collegamento al Backend GIDEON
 * 
 * Si connette al backend via WebSocket e riproduce le risposte vocalmente
 */

import WebSocket from 'ws';
import axios from 'axios';
import * as tts from './tts.js';
import * as audio from './audio.js';
import readline from 'readline';

// Configurazione
const CONFIG = {
    backendUrl: 'http://127.0.0.1:8001',
    wsUrl: 'ws://127.0.0.1:8001/ws',
    autoSpeak: true,           // Parla automaticamente le risposte
    speakWelcome: true,        // Messaggio di benvenuto
    reconnectInterval: 5000    // Riconnessione automatica
};

// Stato
const state = {
    ws: null,
    connected: false,
    reconnecting: false
};

/**
 * Connette al backend GIDEON via WebSocket
 */
function connect() {
    console.log('\n🔌 Connessione a GIDEON...\n');
    
    state.ws = new WebSocket(CONFIG.wsUrl);
    
    state.ws.on('open', async () => {
        state.connected = true;
        state.reconnecting = false;
        console.log('✅ Connesso a GIDEON Backend');
        
        if (CONFIG.speakWelcome) {
            await speak('Connessione stabilita. Gideon TTS attivo.');
        }
    });
    
    state.ws.on('message', async (data) => {
        try {
            const message = JSON.parse(data.toString());
            await handleMessage(message);
        } catch (error) {
            console.error('❌ Errore parsing messaggio:', error.message);
        }
    });
    
    state.ws.on('close', () => {
        state.connected = false;
        console.log('🔌 Disconnesso da GIDEON');
        
        if (!state.reconnecting) {
            scheduleReconnect();
        }
    });
    
    state.ws.on('error', (error) => {
        console.error('❌ Errore WebSocket:', error.message);
    });
}

/**
 * Riconnessione automatica
 */
function scheduleReconnect() {
    state.reconnecting = true;
    console.log(`🔄 Riconnessione in ${CONFIG.reconnectInterval / 1000}s...`);
    
    setTimeout(() => {
        if (!state.connected) {
            connect();
        }
    }, CONFIG.reconnectInterval);
}

/**
 * Gestisce i messaggi dal backend
 */
async function handleMessage(message) {
    console.log('📨 Ricevuto:', message.type);
    
    switch (message.type) {
        case 'message_result':
        case 'response':
            const text = message.payload?.text || 
                        message.payload?.response || 
                        message.text || 
                        message.response;
            
            if (text && CONFIG.autoSpeak) {
                await speak(text);
            }
            break;
            
        case 'status':
            console.log('📊 Status:', message.payload);
            break;
            
        case 'error':
            console.error('❌ Errore dal backend:', message.payload);
            await speak('Si è verificato un errore');
            break;
            
        default:
            console.log('📦 Messaggio non gestito:', message);
    }
}

/**
 * Invia messaggio al backend
 */
function send(text) {
    if (!state.connected) {
        console.error('❌ Non connesso al backend');
        return false;
    }
    
    const message = {
        type: 'text_message',
        payload: { text }
    };
    
    state.ws.send(JSON.stringify(message));
    console.log('📤 Inviato:', text);
    return true;
}

/**
 * Sintetizza e riproduce testo
 */
async function speak(text) {
    try {
        console.log(`\n🎤 GIDEON dice: "${text.substring(0, 60)}..."\n`);
        
        const audioBuffer = await tts.synthesize(text);
        await audio.playBuffer(audioBuffer);
        
    } catch (error) {
        console.error('❌ Errore TTS:', error.message);
        
        // Fallback: usa TTS di sistema Windows
        if (process.platform === 'win32') {
            const { exec } = await import('child_process');
            const cleanText = text.replace(/"/g, "'").replace(/'/g, "''");
            exec(`powershell -c "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('${cleanText}')"`);
            console.log('🔊 (Usando voce Windows di fallback)');
        }
    }
}

/**
 * Ferma la riproduzione
 */
function stopSpeaking() {
    audio.stop();
}

/**
 * Chat diretta via API REST
 */
async function chat(message) {
    try {
        console.log(`\n💬 Tu: ${message}`);
        
        const response = await axios.post(`${CONFIG.backendUrl}/api/chat/send`, {
            message
        }, {
            timeout: 60000
        });
        
        const text = response.data.response || response.data.text;
        console.log(`🤖 GIDEON: ${text}`);
        
        if (CONFIG.autoSpeak) {
            await speak(text);
        }
        
        return text;
    } catch (error) {
        console.error('❌ Errore chat:', error.message);
        return null;
    }
}

/**
 * Modalità interattiva da terminale
 */
function startInteractive() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    console.log('\n╔════════════════════════════════════════╗');
    console.log('║     GIDEON TTS - Modalità Interattiva  ║');
    console.log('╠════════════════════════════════════════╣');
    console.log('║ Comandi:                               ║');
    console.log('║   /stop   - Ferma riproduzione         ║');
    console.log('║   /voice  - Cambia voce                ║');
    console.log('║   /speed  - Cambia velocità            ║');
    console.log('║   /test   - Test voce                  ║');
    console.log('║   /quit   - Esci                       ║');
    console.log('╚════════════════════════════════════════╝\n');
    
    const prompt = () => {
        rl.question('Tu > ', async (input) => {
            const cmd = input.trim().toLowerCase();
            
            if (cmd === '/quit' || cmd === '/exit') {
                console.log('\n👋 Arrivederci!');
                await speak('Arrivederci!');
                process.exit(0);
            }
            
            if (cmd === '/stop') {
                stopSpeaking();
                prompt();
                return;
            }
            
            if (cmd === '/test') {
                await tts.testVoice();
                prompt();
                return;
            }
            
            if (cmd.startsWith('/voice')) {
                await tts.listVoices();
                prompt();
                return;
            }
            
            if (cmd.startsWith('/speed')) {
                const speed = parseFloat(cmd.split(' ')[1]) || 1.0;
                tts.setSpeed(speed);
                prompt();
                return;
            }
            
            if (input.trim()) {
                await chat(input);
            }
            
            prompt();
        });
    };
    
    prompt();
}

/**
 * Avvio principale
 */
async function main() {
    console.log('\n');
    console.log('╔═══════════════════════════════════════════════╗');
    console.log('║                                               ║');
    console.log('║   ██████╗ ██╗██████╗ ███████╗ ██████╗ ███╗   ██╗');
    console.log('║  ██╔════╝ ██║██╔══██╗██╔════╝██╔═══██╗████╗  ██║');
    console.log('║  ██║  ███╗██║██║  ██║█████╗  ██║   ██║██╔██╗ ██║');
    console.log('║  ██║   ██║██║██║  ██║██╔══╝  ██║   ██║██║╚██╗██║');
    console.log('║  ╚██████╔╝██║██████╔╝███████╗╚██████╔╝██║ ╚████║');
    console.log('║   ╚═════╝ ╚═╝╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝');
    console.log('║                                               ║');
    console.log('║              TTS Voice System                 ║');
    console.log('║                                               ║');
    console.log('╚═══════════════════════════════════════════════╝');
    console.log('\n');
    
    // Verifica backend
    try {
        const health = await axios.get(`${CONFIG.backendUrl}/health`, { timeout: 3000 });
        console.log('✅ Backend GIDEON attivo');
    } catch {
        console.log('⚠️ Backend GIDEON non raggiungibile');
        console.log('   Avvia il backend con: uvicorn main:app --port 8001\n');
    }
    
    // Connetti WebSocket
    connect();
    
    // Avvia modalità interattiva
    startInteractive();
}

// Avvia
main().catch(console.error);

export {
    connect,
    send,
    speak,
    stopSpeaking,
    chat,
    CONFIG,
    state
};
