/**
 * 🎤 useVoiceRecognition - Hook per riconoscimento vocale avanzato
 * 
 * Features:
 * - Voice Activity Detection (VAD) - rileva quando l'utente parla
 * - Ascolto continuo o singolo
 * - Auto-stop dopo silenzio
 * - Feedback visuale del livello audio
 * - Wake word detection opzionale
 */

import { useState, useCallback, useRef, useEffect } from 'react';

interface VoiceRecognitionOptions {
  language?: string;
  continuous?: boolean;
  interimResults?: boolean;
  maxSilenceMs?: number;  // Tempo massimo di silenzio prima di fermare
  wakeWord?: string;      // Parola chiave per attivare (opzionale)
  onResult?: (text: string, isFinal: boolean) => void;
  onError?: (error: string) => void;
  onStateChange?: (state: VoiceState) => void;
  onVolumeChange?: (volume: number) => void;
}

export type VoiceState = 
  | 'idle'           // Non attivo
  | 'listening'      // In ascolto attivo
  | 'processing'     // Elaborando
  | 'waitingWakeWord'// In attesa della wake word
  | 'speaking'       // Gideon sta parlando
  | 'error';         // Errore

interface VoiceRecognitionReturn {
  state: VoiceState;
  isListening: boolean;
  transcript: string;
  interimTranscript: string;
  volume: number;
  error: string | null;
  startListening: () => void;
  stopListening: () => void;
  toggleListening: () => void;
  startContinuousListening: () => void;
}

// Verifica supporto browser
const isSpeechRecognitionSupported = () => {
  return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
};

export function useVoiceRecognition(options: VoiceRecognitionOptions = {}): VoiceRecognitionReturn {
  const {
    language = 'it-IT',
    continuous = false,
    interimResults = true,
    maxSilenceMs = 2000,  // 2 secondi di silenzio = stop
    wakeWord,
    onResult,
    onError,
    onStateChange,
    onVolumeChange,
  } = options;

  const [state, setState] = useState<VoiceState>('idle');
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [volume, setVolume] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<any>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Aggiorna stato e notifica
  const updateState = useCallback((newState: VoiceState) => {
    setState(newState);
    onStateChange?.(newState);
  }, [onStateChange]);

  // Cleanup audio analysis
  const cleanupAudio = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current?.state !== 'closed') {
      audioContextRef.current?.close();
      audioContextRef.current = null;
    }
    setVolume(0);
  }, []);

  // Inizia analisi audio per VAD
  const startAudioAnalysis = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const updateVolume = () => {
        if (!analyserRef.current) return;
        
        analyserRef.current.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        const normalizedVolume = Math.min(100, (avg / 128) * 100);
        
        setVolume(normalizedVolume);
        onVolumeChange?.(normalizedVolume);

        // Reset silence timer se c'è attività vocale
        if (normalizedVolume > 10) {
          resetSilenceTimer();
        }

        animationFrameRef.current = requestAnimationFrame(updateVolume);
      };

      updateVolume();
    } catch (err) {
      console.error('Error starting audio analysis:', err);
    }
  }, [onVolumeChange]);

  // Timer silenzio
  const resetSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
    }
    
    silenceTimerRef.current = setTimeout(() => {
      // Dopo maxSilenceMs di silenzio, ferma l'ascolto
      if (state === 'listening' && !continuous) {
        console.log('🔇 Silenzio rilevato, stopping...');
        stopListening();
      }
    }, maxSilenceMs);
  }, [maxSilenceMs, state, continuous]);

  // Stop listening
  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }

    cleanupAudio();
    updateState('idle');
  }, [cleanupAudio, updateState]);

  // Start listening
  const startListening = useCallback(() => {
    if (!isSpeechRecognitionSupported()) {
      setError('Speech recognition not supported in this browser');
      onError?.('Speech recognition not supported');
      return;
    }

    setError(null);
    setTranscript('');
    setInterimTranscript('');

    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.lang = language;
    recognition.continuous = continuous;
    recognition.interimResults = interimResults;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      console.log('🎤 Voice recognition started');
      updateState('listening');
      startAudioAnalysis();
      resetSilenceTimer();
    };

    recognition.onresult = (event: any) => {
      let finalTranscript = '';
      let interim = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          finalTranscript += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }

      if (finalTranscript) {
        // Wake word check
        if (wakeWord && !finalTranscript.toLowerCase().includes(wakeWord.toLowerCase())) {
          // Non contiene wake word, ignora
          console.log('⏭️ No wake word detected');
          return;
        }

        setTranscript(finalTranscript);
        onResult?.(finalTranscript, true);
        
        // Se non continuous, ferma dopo risultato finale
        if (!continuous) {
          stopListening();
        }
      } else if (interim) {
        setInterimTranscript(interim);
        onResult?.(interim, false);
        resetSilenceTimer(); // Reset timer quando c'è input
      }
    };

    recognition.onerror = (event: any) => {
      console.error('🎤 Voice recognition error:', event.error);
      
      let errorMessage = 'Voice recognition error';
      switch (event.error) {
        case 'no-speech':
          errorMessage = 'Nessun audio rilevato. Riprova a parlare.';
          break;
        case 'audio-capture':
          errorMessage = 'Microfono non disponibile';
          break;
        case 'not-allowed':
          errorMessage = 'Permesso microfono negato';
          break;
        case 'network':
          errorMessage = 'Errore di rete';
          break;
        case 'aborted':
          // User stopped, non è un errore
          return;
      }
      
      setError(errorMessage);
      onError?.(errorMessage);
      updateState('error');
      cleanupAudio();
    };

    recognition.onend = () => {
      console.log('🎤 Voice recognition ended');
      
      // Se continuous e non è stato fermato manualmente, riavvia
      if (continuous && state === 'listening') {
        setTimeout(() => {
          if (recognitionRef.current === recognition) {
            recognition.start();
          }
        }, 100);
      } else {
        cleanupAudio();
        updateState('idle');
      }
    };

    recognition.onspeechend = () => {
      console.log('🔇 Speech ended');
      // L'utente ha smesso di parlare
      if (!continuous) {
        // Aspetta un momento per eventuali ultime parole
        setTimeout(() => {
          if (recognitionRef.current === recognition) {
            stopListening();
          }
        }, 500);
      }
    };

    recognitionRef.current = recognition;
    
    try {
      recognition.start();
    } catch (err) {
      console.error('Failed to start recognition:', err);
      setError('Failed to start voice recognition');
      updateState('error');
    }
  }, [
    language, 
    continuous, 
    interimResults, 
    wakeWord,
    onResult, 
    onError, 
    updateState, 
    startAudioAnalysis, 
    resetSilenceTimer,
    stopListening,
    cleanupAudio,
    state
  ]);

  // Continuous listening mode
  const startContinuousListening = useCallback(() => {
    if (wakeWord) {
      updateState('waitingWakeWord');
    }
    startListening();
  }, [startListening, wakeWord, updateState]);

  // Toggle
  const toggleListening = useCallback(() => {
    if (state === 'listening' || state === 'waitingWakeWord') {
      stopListening();
    } else {
      startListening();
    }
  }, [state, startListening, stopListening]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening();
    };
  }, [stopListening]);

  return {
    state,
    isListening: state === 'listening' || state === 'waitingWakeWord',
    transcript,
    interimTranscript,
    volume,
    error,
    startListening,
    stopListening,
    toggleListening,
    startContinuousListening,
  };
}

export default useVoiceRecognition;
