/**
 * 🎤 VoiceIndicator - Indicatore visuale per lo stato del riconoscimento vocale
 * 
 * Mostra:
 * - Stato corrente (idle, listening, processing)
 * - Livello volume in tempo reale
 * - Transcript provvisorio
 * - Animazioni di feedback
 */

import React from 'react';
import { VoiceState } from '../hooks/useVoiceRecognition';

interface VoiceIndicatorProps {
  state: VoiceState;
  volume: number;
  interimTranscript?: string;
  error?: string | null;
  onToggle: () => void;
}

const VoiceIndicator: React.FC<VoiceIndicatorProps> = ({
  state,
  volume,
  interimTranscript,
  error,
  onToggle,
}) => {
  const isActive = state === 'listening' || state === 'waitingWakeWord';
  
  // Colori basati sullo stato
  const getStateColor = () => {
    switch (state) {
      case 'listening':
        return '#4CAF50'; // Verde - attivo
      case 'waitingWakeWord':
        return '#FF9800'; // Arancione - in attesa wake word
      case 'processing':
        return '#2196F3'; // Blu - elaborando
      case 'error':
        return '#f44336'; // Rosso - errore
      case 'speaking':
        return '#9C27B0'; // Viola - Gideon sta parlando
      default:
        return '#666';    // Grigio - idle
    }
  };

  // Testo stato
  const getStateText = () => {
    switch (state) {
      case 'listening':
        return '🎤 Sto ascoltando... parla pure!';
      case 'waitingWakeWord':
        return '👂 Di "Hey Gideon" per iniziare';
      case 'processing':
        return '🤔 Sto elaborando...';
      case 'error':
        return `❌ ${error || 'Errore'}`;
      case 'speaking':
        return '🗣️ Gideon sta parlando...';
      default:
        return '🎙️ Premi per parlare';
    }
  };

  return (
    <div className="voice-indicator-container">
      {/* Pulsante microfono principale */}
      <button
        className={`voice-button ${isActive ? 'active' : ''} ${state === 'error' ? 'error' : ''}`}
        onClick={onToggle}
        style={{
          '--state-color': getStateColor(),
          '--volume': volume / 100,
        } as React.CSSProperties}
        aria-label={isActive ? 'Stop listening' : 'Start listening'}
      >
        <div className="mic-icon">
          <MicrophoneIcon active={isActive} />
        </div>
        
        {/* Cerchi animati per volume */}
        {isActive && (
          <div className="volume-rings">
            <div className="ring ring-1" style={{ transform: `scale(${1 + volume / 100 * 0.3})` }} />
            <div className="ring ring-2" style={{ transform: `scale(${1 + volume / 100 * 0.5})` }} />
            <div className="ring ring-3" style={{ transform: `scale(${1 + volume / 100 * 0.7})` }} />
          </div>
        )}
      </button>

      {/* Stato testuale */}
      <div className="voice-status" style={{ color: getStateColor() }}>
        {getStateText()}
      </div>

      {/* Volume meter */}
      {isActive && (
        <div className="volume-meter">
          <div 
            className="volume-bar" 
            style={{ 
              width: `${Math.min(100, volume)}%`,
              backgroundColor: volume > 70 ? '#4CAF50' : volume > 30 ? '#FF9800' : '#666'
            }} 
          />
        </div>
      )}

      {/* Transcript provvisorio */}
      {interimTranscript && (
        <div className="interim-transcript">
          <span className="interim-label">Sento:</span>
          <span className="interim-text">"{interimTranscript}"</span>
        </div>
      )}

      {/* Stili inline per semplicità */}
      <style>{`
        .voice-indicator-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 10px;
          padding: 15px;
        }

        .voice-button {
          position: relative;
          width: 70px;
          height: 70px;
          border-radius: 50%;
          border: 3px solid var(--state-color);
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          cursor: pointer;
          transition: all 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: visible;
        }

        .voice-button:hover {
          transform: scale(1.05);
          box-shadow: 0 0 20px var(--state-color);
        }

        .voice-button.active {
          animation: pulse-glow 1.5s infinite;
          box-shadow: 0 0 30px var(--state-color);
        }

        .voice-button.error {
          animation: shake 0.5s ease-in-out;
        }

        @keyframes pulse-glow {
          0%, 100% {
            box-shadow: 0 0 20px var(--state-color);
          }
          50% {
            box-shadow: 0 0 40px var(--state-color), 0 0 60px var(--state-color);
          }
        }

        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-5px); }
          75% { transform: translateX(5px); }
        }

        .mic-icon {
          width: 30px;
          height: 30px;
          z-index: 2;
        }

        .mic-icon svg {
          width: 100%;
          height: 100%;
          fill: var(--state-color);
          transition: fill 0.3s ease;
        }

        .volume-rings {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          pointer-events: none;
        }

        .ring {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 70px;
          height: 70px;
          border-radius: 50%;
          border: 2px solid var(--state-color);
          opacity: 0.3;
          transition: transform 0.1s ease-out;
        }

        .ring-1 { opacity: 0.4; }
        .ring-2 { opacity: 0.25; }
        .ring-3 { opacity: 0.15; }

        .voice-status {
          font-size: 14px;
          font-weight: 500;
          text-align: center;
          min-height: 20px;
        }

        .volume-meter {
          width: 150px;
          height: 6px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 3px;
          overflow: hidden;
        }

        .volume-bar {
          height: 100%;
          border-radius: 3px;
          transition: width 0.05s ease-out, background-color 0.3s ease;
        }

        .interim-transcript {
          background: rgba(255, 255, 255, 0.05);
          padding: 8px 15px;
          border-radius: 20px;
          max-width: 300px;
          text-align: center;
        }

        .interim-label {
          color: #888;
          font-size: 12px;
          margin-right: 5px;
        }

        .interim-text {
          color: #fff;
          font-style: italic;
        }
      `}</style>
    </div>
  );
};

// Icona microfono SVG
const MicrophoneIcon: React.FC<{ active: boolean }> = ({ active }) => (
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    {active ? (
      // Microfono attivo con onde
      <>
        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
        {/* Onde sonore animate */}
        <path 
          d="M19 11a7 7 0 0 1-7 7" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="1.5" 
          strokeLinecap="round"
          style={{ opacity: 0.5 }}
        >
          <animate attributeName="opacity" values="0.5;1;0.5" dur="1s" repeatCount="indefinite" />
        </path>
      </>
    ) : (
      // Microfono normale
      <>
        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
      </>
    )}
  </svg>
);

export default VoiceIndicator;
