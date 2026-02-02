import React, { useState, useEffect, useRef, useCallback } from 'react';
import Avatar3D from './components/Avatar3D';
import VoiceIndicator from './components/VoiceIndicator';
import { useVoiceRecognition, VoiceState } from './hooks/useVoiceRecognition';
import { Mic, MicOff, Send, Activity, Brain, Shield, BarChart3, Volume2, VolumeX, Wand2 } from 'lucide-react';
import './App.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'gideon';
  timestamp: Date;
  intent?: string;
}

interface SystemMetrics {
  cpu: number;
  memory: number;
  disk: number;
  responseTime: number;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [expression, setExpression] = useState<'neutral' | 'happy' | 'thinking' | 'focused' | 'concerned' | 'confident'>('neutral');
  const [pilotMode, setPilotMode] = useState(false);
  const [metrics, setMetrics] = useState<SystemMetrics>({ cpu: 0, memory: 0, disk: 0, responseTime: 0 });
  const [wsConnected, setWsConnected] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [voiceMode, setVoiceMode] = useState<'push-to-talk' | 'continuous'>('push-to-talk');
  
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Wrapper per addMessage (necessario prima dell'hook)
  const addMessageRef = useRef<(text: string, sender: 'user' | 'gideon', intent?: string) => void>(() => {});

  // 🎤 Voice Recognition Hook con VAD avanzato
  const {
    state: voiceState,
    isListening,
    transcript,
    interimTranscript,
    volume,
    error: voiceError,
    startListening,
    stopListening,
    toggleListening,
  } = useVoiceRecognition({
    language: 'it-IT',
    continuous: voiceMode === 'continuous',
    interimResults: true,
    maxSilenceMs: 2000, // 2 secondi di silenzio = auto-stop
    onResult: (text, isFinal) => {
      if (isFinal && text.trim()) {
        // Testo finale riconosciuto - invia automaticamente
        handleVoiceResult(text.trim());
      } else {
        // Testo provvisorio - mostra in input
        setInputText(text);
      }
    },
    onStateChange: (state) => {
      // Aggiorna espressione avatar in base allo stato voice
      switch (state) {
        case 'listening':
          setExpression('focused');
          break;
        case 'processing':
          setExpression('thinking');
          break;
        case 'error':
          setExpression('concerned');
          break;
      }
    },
    onError: (error) => {
      console.error('Voice error:', error);
    },
  });

  // Funzione addMessage
  const addMessage = useCallback((text: string, sender: 'user' | 'gideon', intent?: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      text,
      sender,
      timestamp: new Date(),
      intent
    };
    setMessages(prev => [...prev, newMessage]);
  }, []);

  // Aggiorna il ref
  useEffect(() => {
    addMessageRef.current = addMessage;
  }, [addMessage]);

  // Gestisce risultato vocale finale
  const handleVoiceResult = useCallback((text: string) => {
    setInputText(text);
    
    // Invia automaticamente dopo riconoscimento vocale
    addMessageRef.current(text, 'user');
    
    wsRef.current?.send(JSON.stringify({
      type: 'voice_command',
      payload: { text }
    }));
    
    setExpression('thinking');
    setInputText(''); // Pulisci dopo invio
  }, []);

  // Initialize WebSocket
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket('ws://localhost:8001/ws');
      
      ws.onopen = () => {
        console.log('✅ Connected to Gideon');
        setWsConnected(true);
        addMessage('Ciao! Sono Gideon, il tuo assistente IA avanzato. Puoi scrivere o usare il microfono per parlare con me. 🎤', 'gideon');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      };
      
      ws.onerror = () => setWsConnected(false);
      ws.onclose = () => {
        setWsConnected(false);
        // Riconnessione automatica dopo 3 secondi
        setTimeout(connectWebSocket, 3000);
      };
      
      wsRef.current = ws;
    };

    connectWebSocket();
    
    return () => wsRef.current?.close();
  }, [addMessage]);

  // Update metrics every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics({
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        disk: Math.random() * 100,
        responseTime: Math.random() * 500
      });
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleWebSocketMessage = useCallback((data: any) => {
    const { type, payload } = data;
    
    if (type === 'command_result' || type === 'message_result') {
      const response = payload.text;
      addMessage(response, 'gideon', payload.intent);
      
      if (payload.avatar_expression) {
        setExpression(payload.avatar_expression);
      }
      
      if (soundEnabled) {
        speakText(response);
      }
    }
  }, [addMessage, soundEnabled]);

  const sendMessage = useCallback(() => {
    if (!inputText.trim() || !wsConnected) return;
    
    addMessage(inputText, 'user');
    
    wsRef.current?.send(JSON.stringify({
      type: 'text_message',
      payload: { text: inputText }
    }));
    
    setInputText('');
    setExpression('thinking');
  }, [inputText, wsConnected, addMessage]);

  const speakText = useCallback((text: string) => {
    // Ferma voice recognition mentre Gideon parla
    if (isListening) {
      stopListening();
    }
    
    setIsSpeaking(true);
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'it-IT';
    utterance.rate = 1.0;
    
    utterance.onend = () => {
      setIsSpeaking(false);
      setExpression('neutral');
      
      // Se in modalità continua, riprendi ascolto dopo che Gideon ha finito
      if (voiceMode === 'continuous') {
        setTimeout(() => startListening(), 500);
      }
    };
    
    window.speechSynthesis.speak(utterance);
  }, [isListening, stopListening, voiceMode, startListening]);

  const toggleVoiceMode = useCallback(() => {
    const newMode = voiceMode === 'push-to-talk' ? 'continuous' : 'push-to-talk';
    setVoiceMode(newMode);
    
    if (isListening) {
      stopListening();
    }
    
    addMessage(
      newMode === 'continuous' 
        ? '🎤 Modalità ascolto continuo attivata. Parla quando vuoi!' 
        : '🎤 Modalità push-to-talk attivata. Premi il microfono per parlare.',
      'gideon'
    );
  }, [voiceMode, isListening, stopListening, addMessage]);

  // Stato del pulsante voice in base allo stato
  const getVoiceButtonClass = () => {
    if (voiceState === 'error') return 'bg-red-500 hover:bg-red-600';
    if (isListening) return 'bg-green-500 hover:bg-green-600 animate-pulse';
    return 'bg-white/10 hover:bg-white/20';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white">
      {/* Header */}
      <header className="bg-black/50 backdrop-blur-md border-b border-white/10 p-4 sticky top-0 z-50">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-blue-400 animate-pulse" />
            <h1 className="text-3xl font-bold">GIDEON 2.0</h1>
            <span className="text-sm text-gray-400">Advanced AI Desktop Assistant</span>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Voice Mode Toggle */}
            <button
              onClick={toggleVoiceMode}
              className={`px-3 py-2 rounded-lg transition flex items-center gap-2 ${
                voiceMode === 'continuous' ? 'bg-green-500/30 text-green-400' : 'bg-gray-500/30 text-gray-400'
              }`}
              title={`Modalità: ${voiceMode === 'continuous' ? 'Ascolto continuo' : 'Push-to-talk'}`}
            >
              <Wand2 className="w-4 h-4" />
              <span className="text-sm hidden md:inline">
                {voiceMode === 'continuous' ? 'Continuo' : 'Push-to-talk'}
              </span>
            </button>

            {/* Sound Toggle */}
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`p-2 rounded-lg transition ${
                soundEnabled ? 'bg-blue-500/30 text-blue-400' : 'bg-gray-500/30 text-gray-400'
              }`}
              title={soundEnabled ? 'Audio attivo' : 'Audio disattivato'}
            >
              {soundEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
            </button>
            
            {/* Connection Status */}
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${wsConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
              <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`} />
              <span className="text-sm">{wsConnected ? 'Connesso' : 'Disconnesso'}</span>
            </div>
            
            {pilotMode && (
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-red-500/20 text-red-400">
                <Shield className="w-4 h-4" />
                <span className="text-sm font-semibold">PILOT</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="container mx-auto p-6 grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-100px)]">
        {/* Left Sidebar - Avatar & Voice */}
        <div className="lg:col-span-1 space-y-4 overflow-y-auto">
          {/* Avatar */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 sticky top-0">
            <div className="aspect-square rounded-xl overflow-hidden bg-gradient-to-b from-blue-500/20 to-purple-500/20">
              <Avatar3D expression={expression} isSpeaking={isSpeaking} />
            </div>
            
            <div className="mt-4 text-center">
              <p className="text-lg font-semibold capitalize">{expression}</p>
              {isSpeaking && (
                <p className="text-sm text-blue-400 flex items-center justify-center gap-2 mt-2">
                  <Activity className="w-4 h-4 animate-pulse" />
                  Parlando...
                </p>
              )}
            </div>
          </div>

          {/* Voice Indicator - Nuovo componente */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/20">
            <h3 className="text-sm font-semibold mb-3 text-gray-300 text-center">🎤 Riconoscimento Vocale</h3>
            <VoiceIndicator
              state={voiceState}
              volume={volume}
              interimTranscript={interimTranscript}
              error={voiceError}
              onToggle={toggleListening}
            />
          </div>

          {/* Quick Actions */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/20">
            <h3 className="text-sm font-semibold mb-3 text-gray-300">Azioni Rapide</h3>
            <div className="space-y-2">
              <button 
                onClick={() => {
                  addMessage('Analizza il sistema', 'user');
                  wsRef.current?.send(JSON.stringify({
                    type: 'text_message',
                    payload: { text: 'Analizza il sistema' }
                  }));
                }}
                className="w-full px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 rounded-lg flex items-center gap-2 transition text-sm"
              >
                <Activity className="w-4 h-4" />
                Analizza Sistema
              </button>
              <button 
                onClick={() => {
                  addMessage('Mostra performance', 'user');
                  wsRef.current?.send(JSON.stringify({
                    type: 'text_message',
                    payload: { text: 'Mostra performance' }
                  }));
                }}
                className="w-full px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30 rounded-lg flex items-center gap-2 transition text-sm"
              >
                <BarChart3 className="w-4 h-4" />
                Performance
              </button>
            </div>
          </div>

          {/* System Metrics */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/20">
            <h3 className="text-sm font-semibold mb-3 text-gray-300">Metriche Sistema</h3>
            <div className="space-y-3 text-sm">
              <div>
                <div className="flex justify-between mb-1">
                  <span>CPU</span>
                  <span className="text-blue-400 font-semibold">{metrics.cpu.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-blue-400 h-2 rounded-full transition-all"
                    style={{ width: `${metrics.cpu}%` }}
                  />
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span>Memoria</span>
                  <span className="text-green-400 font-semibold">{metrics.memory.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-green-500 to-green-400 h-2 rounded-full transition-all"
                    style={{ width: `${metrics.memory}%` }}
                  />
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span>Disco</span>
                  <span className="text-yellow-400 font-semibold">{metrics.disk.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-yellow-500 to-yellow-400 h-2 rounded-full transition-all"
                    style={{ width: `${metrics.disk}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="lg:col-span-3 bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 flex flex-col overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-gray-400">
                <div className="text-center">
                  <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Nessun messaggio ancora. Inizia a parlare con Gideon!</p>
                  <p className="text-sm mt-2">🎤 Premi il microfono o scrivi un messaggio</p>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[70%] p-4 rounded-2xl ${
                      message.sender === 'user'
                        ? 'bg-blue-500/30 border border-blue-400/50'
                        : 'bg-purple-500/20 border border-purple-400/30'
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                    <span className="text-xs text-gray-400 mt-2 block">
                      {message.timestamp.toLocaleTimeString('it-IT')}
                    </span>
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Voice Status Bar - Mostra stato ascolto */}
          {isListening && (
            <div className="px-4 py-2 bg-green-500/20 border-t border-green-500/30 flex items-center justify-center gap-3">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
              <span className="text-green-400 text-sm font-medium">
                🎤 Sto ascoltando... parla pure!
              </span>
              <div className="flex gap-1">
                {[...Array(5)].map((_, i) => (
                  <div 
                    key={i}
                    className="w-1 bg-green-500 rounded-full transition-all"
                    style={{ height: `${Math.min(20, 4 + (volume / 100) * 16 * Math.random())}px` }}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Interim Transcript - Mostra cosa sta sentendo */}
          {interimTranscript && (
            <div className="px-4 py-2 bg-blue-500/10 border-t border-blue-500/30">
              <span className="text-blue-400 text-sm italic">
                Sento: "{interimTranscript}"
              </span>
            </div>
          )}

          {/* Input Area */}
          <div className="p-4 border-t border-white/10">
            <div className="flex gap-2">
              {/* Voice Button con stato migliorato */}
              <button
                onClick={toggleListening}
                disabled={isSpeaking}
                className={`p-3 rounded-xl transition font-semibold relative ${getVoiceButtonClass()} ${isSpeaking ? 'opacity-50 cursor-not-allowed' : ''}`}
                title={
                  isSpeaking 
                    ? 'Gideon sta parlando...' 
                    : isListening 
                      ? 'Clicca per fermare' 
                      : 'Clicca per parlare'
                }
              >
                {isListening ? (
                  <MicOff className="w-5 h-5 text-white" />
                ) : (
                  <Mic className="w-5 h-5" />
                )}
                
                {/* Volume indicator circle */}
                {isListening && (
                  <div 
                    className="absolute inset-0 rounded-xl border-2 border-green-400 animate-ping"
                    style={{ opacity: volume / 200 }}
                  />
                )}
              </button>
              
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                placeholder={isListening ? "Sto ascoltando..." : "Scrivi un messaggio o premi il microfono..."}
                className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:border-blue-400/50 transition text-white placeholder-gray-400"
                disabled={isListening}
              />
              
              <button
                onClick={sendMessage}
                disabled={!inputText.trim() || !wsConnected || isListening}
                className="p-3 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-500 disabled:cursor-not-allowed rounded-xl transition font-semibold"
                title="Invia"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>

            {/* Voice hints */}
            <div className="mt-2 text-xs text-gray-500 text-center">
              {voiceMode === 'continuous' 
                ? '🔄 Modalità continua: Gideon ascolta sempre dopo aver risposto' 
                : '👆 Modalità push-to-talk: Premi il microfono per parlare'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
