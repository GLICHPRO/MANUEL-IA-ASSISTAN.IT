/**
 * GIDEON SPEAK - Interfaccia Semplice TTS
 * 
 * Uso:
 *   import { gideonSpeak } from "./speak.js";
 *   gideonSpeak("Ciao, sono Gideon!");
 */

import { textToVoice } from "./tts.js";
import { playAudio } from "./audio.js";

// QUESTA È LA FUNZIONE CHE GIDEON DEVE CHIAMARE
export async function gideonSpeak(text) {
  console.log("🤖 Gideon:", text);   // risposta testuale
  const audio = await textToVoice(text);
  await playAudio(audio);            // risposta vocale
}

// Esporta anche le funzioni base per controllo avanzato
export { textToVoice } from "./tts.js";
export { playAudio, stopAudio } from "./audio.js";

// ============================================
// TEST - Esegui con: node speak.js
// ============================================
const isMainModule = import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`;

if (isMainModule) {
  console.log("╔════════════════════════════════════════╗");
  console.log("║       🎙️  GIDEON TTS - TEST           ║");
  console.log("╚════════════════════════════════════════╝");
  console.log();
  
  gideonSpeak(
    "Sono Gideon. Ora la mia voce è naturale, chiara e completamente gratuita."
  );
}
