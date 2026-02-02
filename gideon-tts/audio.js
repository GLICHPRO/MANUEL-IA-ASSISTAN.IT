/**
 * GIDEON Audio - Riproduzione Audio
 * 
 * Gestisce la riproduzione dell'audio generato dal TTS
 */

import { exec, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

// ES Module dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Stato riproduzione
const state = {
    isPlaying: false,
    currentProcess: null,
    queue: [],
    volume: 100
};

/**
 * Rileva il player disponibile nel sistema
 */
function detectPlayer() {
    const platform = os.platform();
    
    if (platform === 'win32') {
        // Windows: usa PowerShell con Windows Media Player
        return {
            name: 'powershell',
            command: (file) => `powershell -c "(New-Object Media.SoundPlayer '${file}').PlaySync()"`,
            // Alternativa con ffplay se disponibile
            altCommand: (file) => `ffplay -nodisp -autoexit "${file}"`,
            // O usa VLC se installato
            vlcCommand: (file) => `"C:\\Program Files\\VideoLAN\\VLC\\vlc.exe" --play-and-exit --intf dummy "${file}"`
        };
    } else if (platform === 'darwin') {
        // macOS
        return {
            name: 'afplay',
            command: (file) => `afplay "${file}"`
        };
    } else {
        // Linux
        return {
            name: 'aplay/mpv',
            command: (file) => `mpv --no-video "${file}" || aplay "${file}"`
        };
    }
}

const player = detectPlayer();

/**
 * Riproduce un file audio
 * @param {string} filePath - Percorso del file audio
 * @returns {Promise<void>}
 */
function playFile(filePath) {
    return new Promise((resolve, reject) => {
        if (!fs.existsSync(filePath)) {
            return reject(new Error(`File non trovato: ${filePath}`));
        }
        
        state.isPlaying = true;
        console.log(`🔊 Riproducendo: ${path.basename(filePath)}`);
        
        // Su Windows, usa un approccio diverso per MP3
        if (os.platform() === 'win32') {
            // Usa Windows Media Player tramite PowerShell
            const psScript = `
                Add-Type -AssemblyName presentationCore
                $mediaPlayer = New-Object System.Windows.Media.MediaPlayer
                $mediaPlayer.Open([System.Uri]"${filePath.replace(/\\/g, '/')}")
                $mediaPlayer.Volume = ${state.volume / 100}
                $mediaPlayer.Play()
                Start-Sleep -Milliseconds 500
                while ($mediaPlayer.Position -lt $mediaPlayer.NaturalDuration.TimeSpan) {
                    Start-Sleep -Milliseconds 100
                }
                $mediaPlayer.Close()
            `;
            
            state.currentProcess = exec(`powershell -Command "${psScript.replace(/"/g, '\\"').replace(/\n/g, '; ')}"`, (error) => {
                state.isPlaying = false;
                state.currentProcess = null;
                
                if (error && !error.killed) {
                    console.error('❌ Errore riproduzione:', error.message);
                    reject(error);
                } else {
                    console.log('✅ Riproduzione completata');
                    resolve();
                }
            });
        } else {
            // Linux/macOS
            state.currentProcess = exec(player.command(filePath), (error) => {
                state.isPlaying = false;
                state.currentProcess = null;
                
                if (error && !error.killed) {
                    reject(error);
                } else {
                    resolve();
                }
            });
        }
    });
}

/**
 * Riproduce audio da buffer
 * @param {Buffer} audioBuffer - Buffer audio (MP3)
 * @returns {Promise<void>}
 */
async function playBuffer(audioBuffer) {
    // Salva temporaneamente il buffer
    const tempFile = path.join(os.tmpdir(), `gideon_audio_${Date.now()}.mp3`);
    
    try {
        fs.writeFileSync(tempFile, audioBuffer);
        await playFile(tempFile);
    } finally {
        // Pulisci file temporaneo
        if (fs.existsSync(tempFile)) {
            fs.unlinkSync(tempFile);
        }
    }
}

/**
 * Ferma la riproduzione corrente
 */
function stop() {
    if (state.currentProcess) {
        state.currentProcess.kill();
        state.currentProcess = null;
        state.isPlaying = false;
        console.log('⏹️ Riproduzione fermata');
    }
    
    // Svuota la coda
    state.queue = [];
}

/**
 * Aggiunge audio alla coda
 * @param {Buffer|string} audio - Buffer o path del file
 */
function enqueue(audio) {
    state.queue.push(audio);
    
    if (!state.isPlaying) {
        processQueue();
    }
}

/**
 * Processa la coda di riproduzione
 */
async function processQueue() {
    while (state.queue.length > 0) {
        const audio = state.queue.shift();
        
        try {
            if (Buffer.isBuffer(audio)) {
                await playBuffer(audio);
            } else {
                await playFile(audio);
            }
        } catch (error) {
            console.error('❌ Errore nella coda:', error.message);
        }
    }
}

/**
 * Imposta il volume
 * @param {number} vol - Volume (0-100)
 */
function setVolume(vol) {
    state.volume = Math.max(0, Math.min(100, vol));
    console.log(`🔊 Volume: ${state.volume}%`);
}

/**
 * Verifica se sta riproducendo
 */
function isPlaying() {
    return state.isPlaying;
}

/**
 * Test audio con beep
 */
async function testAudio() {
    console.log('\n🧪 Test Audio\n');
    
    if (os.platform() === 'win32') {
        // Windows beep
        exec('powershell -c "[console]::beep(800,500)"', (err) => {
            if (err) {
                console.log('❌ Test audio fallito');
            } else {
                console.log('✅ Audio funzionante!');
            }
        });
    } else {
        console.log('Test disponibile solo su Windows');
    }
}

export {
    playFile,
    playBuffer,
    stop,
    enqueue,
    setVolume,
    isPlaying,
    testAudio,
    state
};
