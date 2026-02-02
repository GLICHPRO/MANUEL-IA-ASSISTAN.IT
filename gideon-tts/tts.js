/**
 * GIDEON TTS - Text-to-Speech con Microsoft Edge (gratuito)
 * 
 * Usa edge-tts per voce naturale italiana senza costi
 */

import { MsEdgeTTS, OUTPUT_FORMAT } from 'edge-tts';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';

// ES Module dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configurazione VOCE DI GIDEON
const CONFIG = {
    // Voce di Gideon - Microsoft Edge Neural TTS
    edgeVoice: {
        name: 'it-IT-GiuseppeNeural',   // VOCE UFFICIALE DI GIDEON
        // Alternative (non usare):
        // 'it-IT-DiegoNeural'           // Troppo robotica
        // 'it-IT-ElsaNeural'            // Femminile
        rate: '+0%',                      // Velocità naturale
        pitch: '+0Hz',                    // Tono naturale
        volume: '+0%'                     // Volume
    },
    
    // Directory per cache audio
    cacheDir: path.join(__dirname, 'audio_cache')
};

// Crea directory cache se non esiste
if (!fs.existsSync(CONFIG.cacheDir)) {
    fs.mkdirSync(CONFIG.cacheDir, { recursive: true });
}

// Client Edge TTS
let edgeClient = null;

/**
 * Inizializza il client Edge TTS
 */
async function initEdgeClient() {
    if (!edgeClient) {
        edgeClient = new MsEdgeTTS();
        await edgeClient.setMetadata(
            CONFIG.edgeVoice.name,
            OUTPUT_FORMAT.AUDIO_24KHZ_96KBITRATE_MONO_MP3
        );
        console.log('✅ Microsoft Edge TTS inizializzato');
        console.log(`   Voce: ${CONFIG.edgeVoice.name}`);
    }
    return edgeClient;
}

/**
 * Genera hash per cache
 */
function generateCacheKey(text, voiceName) {
    const key = `${text}_${voiceName}_${CONFIG.edgeVoice.rate}`;
    return crypto.createHash('md5').update(key).digest('hex');
}

/**
 * Sintetizza testo in audio
 * @param {string} text - Testo da sintetizzare
 * @param {object} options - Opzioni personalizzate
 * @returns {Promise<Buffer>} Audio MP3 come buffer
 */
async function synthesize(text, options = {}) {
    const client = await initEdgeClient();
    
    // Pulisci il testo
    const cleanText = text
        .replace(/[*_`#]/g, '')
        .replace(/<[^>]*>/g, '')
        .replace(/\s+/g, ' ')
        .trim();
    
    if (!cleanText) {
        throw new Error('Testo vuoto');
    }
    
    // Controlla cache
    const cacheKey = generateCacheKey(cleanText, CONFIG.edgeVoice.name);
    const cachePath = path.join(CONFIG.cacheDir, `${cacheKey}.mp3`);
    
    if (fs.existsSync(cachePath)) {
        console.log('📦 Audio dalla cache');
        return fs.readFileSync(cachePath);
    }
    
    console.log(`🎤 Sintetizzando: "${cleanText.substring(0, 50)}${cleanText.length > 50 ? '...' : ''}"`);
    
    try {
        // Usa SSML per controllo voce
        const ssml = `
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="it-IT">
                <voice name="${CONFIG.edgeVoice.name}">
                    <prosody rate="${options.rate || CONFIG.edgeVoice.rate}" 
                             pitch="${options.pitch || CONFIG.edgeVoice.pitch}"
                             volume="${options.volume || CONFIG.edgeVoice.volume}">
                        ${cleanText}
                    </prosody>
                </voice>
            </speak>
        `;
        
        // Genera audio
        const { audioBuffer } = await client.toBuffer(ssml);
        
        // Salva in cache
        fs.writeFileSync(cachePath, audioBuffer);
        console.log('✅ Audio generato e salvato in cache');
        
        return audioBuffer;
    } catch (error) {
        console.error('❌ Errore sintesi:', error.message);
        throw error;
    }
}

/**
 * Sintetizza con SSML per controllo avanzato
 * @param {string} ssml - Markup SSML
 * @returns {Promise<Buffer>}
 */
async function synthesizeSSML(ssml) {
    const client = await initEdgeClient();
    const { audioBuffer } = await client.toBuffer(ssml);
    return audioBuffer;
}

/**
 * Crea SSML con pause ed enfasi
 * @param {string} text - Testo base
 * @returns {string} SSML formattato
 */
function createSSML(text) {
    let ssml = text
        .replace(/\. /g, '.<break time="300ms"/> ')
        .replace(/\? /g, '?<break time="400ms"/> ')
        .replace(/! /g, '!<break time="350ms"/> ')
        .replace(/, /g, ',<break time="150ms"/> ');
    
    return `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="it-IT">
        <voice name="${CONFIG.edgeVoice.name}">
            <prosody rate="${CONFIG.edgeVoice.rate}" pitch="${CONFIG.edgeVoice.pitch}">
                ${ssml}
            </prosody>
        </voice>
    </speak>`;
}

/**
 * Lista voci Edge disponibili
 */
async function listVoices() {
    console.log('\n🎤 Voci italiane Microsoft Edge disponibili:\n');
    
    const voices = [
        { name: 'it-IT-DiegoNeural', gender: 'Maschile', desc: 'Voce profonda e autorevole' },
        { name: 'it-IT-GiuseppeNeural', gender: 'Maschile', desc: 'Voce maschile standard' },
        { name: 'it-IT-BenignoNeural', gender: 'Maschile', desc: 'Voce maschile matura' },
        { name: 'it-IT-ElsaNeural', gender: 'Femminile', desc: 'Voce femminile standard' },
        { name: 'it-IT-IsabellaNeural', gender: 'Femminile', desc: 'Voce femminile alternativa' },
        { name: 'it-IT-CalimeroNeural', gender: 'Maschile', desc: 'Voce giovane' },
        { name: 'it-IT-CataldoNeural', gender: 'Maschile', desc: 'Voce maschile calma' },
        { name: 'it-IT-FabiolaNeural', gender: 'Femminile', desc: 'Voce femminile energica' },
        { name: 'it-IT-FiammaNeural', gender: 'Femminile', desc: 'Voce femminile vivace' },
        { name: 'it-IT-GianniNeural', gender: 'Maschile', desc: 'Voce maschile cordiale' },
        { name: 'it-IT-ImeldaNeural', gender: 'Femminile', desc: 'Voce femminile professionale' },
        { name: 'it-IT-IrmaNeural', gender: 'Femminile', desc: 'Voce femminile matura' },
        { name: 'it-IT-LisandroNeural', gender: 'Maschile', desc: 'Voce maschile seria' },
        { name: 'it-IT-PalmiraNeural', gender: 'Femminile', desc: 'Voce femminile gentile' },
        { name: 'it-IT-PierinaNeural', gender: 'Femminile', desc: 'Voce femminile anziana' },
        { name: 'it-IT-RinaldoNeural', gender: 'Maschile', desc: 'Voce maschile formale' }
    ];
    
    voices.forEach(v => {
        const current = v.name === CONFIG.edgeVoice.name ? ' ← ATTUALE' : '';
        console.log(`  ${v.name}${current}`);
        console.log(`    Genere: ${v.gender}`);
        console.log(`    ${v.desc}`);
        console.log('');
    });
    
    return voices;
}

/**
 * Cambia voce
 * @param {string} voiceName - Nome della voce
 */
async function setVoice(voiceName) {
    CONFIG.edgeVoice.name = voiceName;
    edgeClient = null; // Reset client per applicare nuova voce
    await initEdgeClient();
    console.log(`🎤 Voce cambiata: ${voiceName}`);
}

/**
 * Cambia velocità
 * @param {number} percent - Percentuale (-50 a +100)
 */
function setSpeed(percent) {
    const val = Math.max(-50, Math.min(100, percent));
    CONFIG.edgeVoice.rate = `${val >= 0 ? '+' : ''}${val}%`;
    console.log(`⚡ Velocità: ${CONFIG.edgeVoice.rate}`);
}

/**
 * Cambia tono
 * @param {number} hz - Hertz (-50 a +50)
 */
function setPitch(hz) {
    const val = Math.max(-50, Math.min(50, hz));
    CONFIG.edgeVoice.pitch = `${val >= 0 ? '+' : ''}${val}Hz`;
    console.log(`🎵 Tono: ${CONFIG.edgeVoice.pitch}`);
}

/**
 * Pulisci cache
 */
function clearCache() {
    const files = fs.readdirSync(CONFIG.cacheDir);
    files.forEach(file => {
        fs.unlinkSync(path.join(CONFIG.cacheDir, file));
    });
    console.log(`🗑️ Cache pulita (${files.length} file rimossi)`);
}

/**
 * Test della voce
 */
async function testVoice() {
    try {
        console.log('\n🧪 Test GIDEON TTS (Microsoft Edge)\n');
        
        const testText = "Ciao! Sono Gideon, il tuo assistente personale. Come posso aiutarti oggi?";
        const audio = await synthesize(testText);
        
        const testFile = path.join(__dirname, 'test_output.mp3');
        fs.writeFileSync(testFile, audio);
        
        console.log(`\n✅ Test completato!`);
        console.log(`   Audio salvato in: ${testFile}`);
        console.log(`   Dimensione: ${(audio.length / 1024).toFixed(2)} KB`);
        console.log(`   Voce: ${CONFIG.edgeVoice.name}`);
        
        return true;
    } catch (error) {
        console.error('\n❌ Test fallito:', error.message);
        return false;
    }
}

export {
    synthesize,
    synthesizeSSML,
    createSSML,
    listVoices,
    setVoice,
    setSpeed,
    setPitch,
    clearCache,
    testVoice,
    CONFIG
};
