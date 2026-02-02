import React, { Suspense, useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Environment, 
  Float,
  MeshDistortMaterial,
  Sparkles,
  Trail,
  Billboard,
  Text
} from '@react-three/drei';
import * as THREE from 'three';

/**
 * 🎭 GIDEON 3D Avatar Component
 * Replica fedele di Gideon da The Flash
 * - Aspetto olografico blu brillante
 * - Lip sync sincronizzato
 * - Movimenti testa e occhi realistici
 * - Effetti particellari e glow
 * - HUD con feedback visivo
 * - Colori basati su modalità operativa
 */

// === OPERATING MODE COLORS ===
const OperatingModeColors = {
  PILOT: '#00FF7F',        // Verde brillante - controllo totale
  COPILOT: '#00BFFF',      // Blu ciano - assistenza
  PASSIVE: '#4169E1',      // Blu reale - osservazione
  EXECUTIVE: '#9370DB',    // Viola - supervisione
  ANALYZING: '#FFD700',    // Giallo oro - analisi in corso
  PROCESSING: '#FFA500',   // Arancione - elaborazione
  ALERT: '#FF4500',        // Rosso-arancio - allarme
  IDLE: '#00BFFF'          // Blu ciano - standby
};

// === TYPES ===

interface AvatarState {
  state: 'idle' | 'speaking' | 'listening' | 'thinking' | 'processing' | 'alert' | 'sleeping';
  operatingMode?: 'passive' | 'copilot' | 'pilot' | 'executive';
  activityState?: 'idle' | 'analyzing' | 'processing' | 'executing' | 'alert';
  head: {
    pitch: number;
    yaw: number;
    roll: number;
  };
  eyes: {
    left: { x: number; y: number; open: number };
    right: { x: number; y: number; open: number };
    pupil_size: number;
  };
  mouth: {
    open: number;
    wide: number;
    smile: number;
    viseme: string;
  };
  expression: {
    expression: string;
    intensity: number;
    eyebrow_left: number;
    eyebrow_right: number;
  };
  hologram: {
    glow_intensity: number;
    glow_color: string;
    scan_line_speed: number;
    flicker_amount: number;
    distortion: number;
    particle_density: number;
    ring_count: number;
  };
  feedback?: {
    hud_indicators?: Record<string, HUDIndicator>;
    calculations?: Record<string, CalculationAnimation>;
  };
  timestamp: number;
}

interface HUDIndicator {
  id: string;
  label: string;
  value: number;
  max_value: number;
  color: string;
  icon: string;
  visible: boolean;
  animated: boolean;
  pulse_speed: number;
}

interface CalculationAnimation {
  id: string;
  type: string;
  active: boolean;
  progress: number;
  speed: number;
  color: string;
  intensity: number;
  data_points?: string[];
}

interface GideonAvatarProps {
  avatarState?: Partial<AvatarState>;
  isSpeaking?: boolean;
  expression?: string;
  operatingMode?: 'passive' | 'copilot' | 'pilot' | 'executive';
  activityState?: 'idle' | 'analyzing' | 'processing' | 'executing' | 'alert';
  scale?: number;
  onReady?: () => void;
  showParticles?: boolean;
  showRings?: boolean;
  showHUD?: boolean;
  debugMode?: boolean;
}

// === SHADERS ===

const hologramVertexShader = `
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  uniform float time;
  uniform float distortion;
  
  void main() {
    vUv = uv;
    vPosition = position;
    vNormal = normal;
    
    vec3 pos = position;
    
    // Subtle distortion effect
    pos.x += sin(pos.y * 10.0 + time * 2.0) * distortion * 0.02;
    pos.z += cos(pos.y * 8.0 + time * 1.5) * distortion * 0.02;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const hologramFragmentShader = `
  uniform vec3 glowColor;
  uniform float glowIntensity;
  uniform float time;
  uniform float scanLineSpeed;
  uniform float flickerAmount;
  
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  
  void main() {
    // Base color with fresnel effect
    float fresnel = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 2.0);
    vec3 color = glowColor * (0.5 + fresnel * 0.5);
    
    // Scan lines
    float scanLine = sin(vUv.y * 100.0 + time * scanLineSpeed) * 0.5 + 0.5;
    scanLine = smoothstep(0.4, 0.6, scanLine);
    color *= 0.8 + scanLine * 0.2;
    
    // Horizontal scan band
    float scanBand = sin(time * 0.5) * 0.5 + 0.5;
    float band = smoothstep(scanBand - 0.1, scanBand, vUv.y) * 
                 smoothstep(scanBand + 0.1, scanBand, vUv.y);
    color += glowColor * band * 0.3;
    
    // Flicker
    float flicker = 1.0 - flickerAmount * (fract(sin(time * 100.0) * 43758.5453) * 0.5);
    color *= flicker;
    
    // Edge glow
    float edge = pow(fresnel, 3.0);
    color += glowColor * edge * 0.5;
    
    // Alpha with fresnel
    float alpha = 0.7 + fresnel * 0.3;
    alpha *= glowIntensity;
    
    gl_FragColor = vec4(color, alpha);
  }
`;

// === SUB-COMPONENTS ===

// Holographic Material
function HologramMaterial({ 
  color = '#00BFFF', 
  intensity = 0.6, 
  scanSpeed = 1.0,
  flicker = 0.02,
  distortion = 0.0
}: {
  color?: string;
  intensity?: number;
  scanSpeed?: number;
  flicker?: number;
  distortion?: number;
}) {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  const uniforms = useMemo(() => ({
    glowColor: { value: new THREE.Color(color) },
    glowIntensity: { value: intensity },
    time: { value: 0 },
    scanLineSpeed: { value: scanSpeed },
    flickerAmount: { value: flicker },
    distortion: { value: distortion }
  }), []);
  
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime;
      materialRef.current.uniforms.glowColor.value.set(color);
      materialRef.current.uniforms.glowIntensity.value = intensity;
      materialRef.current.uniforms.scanLineSpeed.value = scanSpeed;
      materialRef.current.uniforms.flickerAmount.value = flicker;
    }
  });
  
  return (
    <shaderMaterial
      ref={materialRef}
      vertexShader={hologramVertexShader}
      fragmentShader={hologramFragmentShader}
      uniforms={uniforms}
      transparent
      side={THREE.DoubleSide}
      depthWrite={false}
    />
  );
}

// Holographic Rings
function HologramRings({ 
  count = 3, 
  color = '#00BFFF',
  radius = 1.8 
}: { 
  count?: number; 
  color?: string;
  radius?: number;
}) {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.2;
    }
  });
  
  return (
    <group ref={groupRef}>
      {Array.from({ length: count }).map((_, i) => (
        <mesh key={i} rotation={[Math.PI / 2 + i * 0.2, 0, 0]} position={[0, -0.5 + i * 0.3, 0]}>
          <torusGeometry args={[radius - i * 0.2, 0.02, 16, 64]} />
          <meshBasicMaterial 
            color={color} 
            transparent 
            opacity={0.3 - i * 0.08}
          />
        </mesh>
      ))}
    </group>
  );
}

// Eye Component
function Eye({ 
  position, 
  lookX = 0, 
  lookY = 0, 
  openness = 1,
  pupilSize = 1,
  color = '#00BFFF'
}: { 
  position: [number, number, number];
  lookX?: number;
  lookY?: number;
  openness?: number;
  pupilSize?: number;
  color?: string;
}) {
  const eyeRef = useRef<THREE.Group>(null);
  const irisRef = useRef<THREE.Mesh>(null);
  
  useFrame(() => {
    if (irisRef.current) {
      // Eye movement
      irisRef.current.position.x = lookX * 0.03;
      irisRef.current.position.y = lookY * 0.03;
    }
  });
  
  return (
    <group ref={eyeRef} position={position} scale={[1, openness, 1]}>
      {/* Eye socket glow */}
      <mesh>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} />
      </mesh>
      
      {/* Iris */}
      <mesh ref={irisRef} position={[0, 0, 0.08]}>
        <circleGeometry args={[0.06 * pupilSize, 16]} />
        <meshBasicMaterial color="#FFFFFF" />
      </mesh>
      
      {/* Pupil */}
      <mesh position={[lookX * 0.03, lookY * 0.03, 0.09]}>
        <circleGeometry args={[0.03 * pupilSize, 16]} />
        <meshBasicMaterial color="#000033" />
      </mesh>
    </group>
  );
}

// Mouth Component with Visemes
function Mouth({ 
  open = 0, 
  wide = 0.5, 
  smile = 0,
  viseme = 'sil',
  color = '#00BFFF'
}: { 
  open?: number;
  wide?: number;
  smile?: number;
  viseme?: string;
  color?: string;
}) {
  const mouthRef = useRef<THREE.Mesh>(null);
  
  // Viseme shapes
  const visemeShapes: Record<string, { width: number; height: number; curve: number }> = {
    'sil': { width: 0.3, height: 0.02, curve: 0 },
    'PP': { width: 0.15, height: 0.02, curve: 0 },
    'FF': { width: 0.25, height: 0.05, curve: 0 },
    'TH': { width: 0.28, height: 0.08, curve: 0 },
    'DD': { width: 0.3, height: 0.1, curve: 0 },
    'kk': { width: 0.25, height: 0.15, curve: 0 },
    'CH': { width: 0.2, height: 0.12, curve: 0 },
    'SS': { width: 0.25, height: 0.08, curve: 0 },
    'nn': { width: 0.3, height: 0.06, curve: 0 },
    'RR': { width: 0.22, height: 0.1, curve: 0 },
    'aa': { width: 0.35, height: 0.25, curve: 0 },
    'E': { width: 0.32, height: 0.15, curve: 0.1 },
    'I': { width: 0.28, height: 0.12, curve: 0.15 },
    'O': { width: 0.2, height: 0.22, curve: 0 },
    'U': { width: 0.15, height: 0.18, curve: 0 }
  };
  
  const shape = visemeShapes[viseme] || visemeShapes['sil'];
  const actualHeight = Math.max(shape.height, open * 0.25);
  const actualWidth = shape.width * (0.8 + wide * 0.4);
  
  return (
    <group position={[0, -0.35, 0.95]}>
      {/* Outer lip glow */}
      <mesh scale={[actualWidth * 1.2, actualHeight * 1.5 + 0.02, 1]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial color={color} transparent opacity={0.2} />
      </mesh>
      
      {/* Inner mouth */}
      <mesh ref={mouthRef} scale={[actualWidth, actualHeight, 1]} position={[0, smile * 0.05, 0.01]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial color="#001a33" />
      </mesh>
      
      {/* Lip highlight */}
      <mesh scale={[actualWidth * 0.8, 0.02, 1]} position={[0, actualHeight / 2 + smile * 0.05, 0.02]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial color={color} transparent opacity={0.5} />
      </mesh>
    </group>
  );
}

// Eyebrow Component
function Eyebrow({ 
  position, 
  angle = 0,
  side = 'left',
  color = '#00BFFF'
}: { 
  position: [number, number, number];
  angle?: number;
  side?: 'left' | 'right';
  color?: string;
}) {
  const rotationZ = side === 'left' ? angle * 0.3 : -angle * 0.3;
  
  return (
    <mesh 
      position={position} 
      rotation={[0, 0, rotationZ]}
    >
      <boxGeometry args={[0.2, 0.03, 0.02]} />
      <meshBasicMaterial color={color} transparent opacity={0.7} />
    </mesh>
  );
}

// Main Head Component
function GideonHead({
  avatarState,
  color = '#00BFFF',
  intensity = 0.6
}: {
  avatarState: AvatarState;
  color?: string;
  intensity?: number;
}) {
  const headRef = useRef<THREE.Group>(null);
  const targetRotation = useRef({ x: 0, y: 0, z: 0 });
  
  useFrame((state, delta) => {
    if (headRef.current) {
      // Smooth head rotation
      const { pitch, yaw, roll } = avatarState.head;
      targetRotation.current.x = THREE.MathUtils.degToRad(pitch);
      targetRotation.current.y = THREE.MathUtils.degToRad(yaw);
      targetRotation.current.z = THREE.MathUtils.degToRad(roll);
      
      headRef.current.rotation.x = THREE.MathUtils.lerp(
        headRef.current.rotation.x,
        targetRotation.current.x,
        delta * 5
      );
      headRef.current.rotation.y = THREE.MathUtils.lerp(
        headRef.current.rotation.y,
        targetRotation.current.y,
        delta * 5
      );
      headRef.current.rotation.z = THREE.MathUtils.lerp(
        headRef.current.rotation.z,
        targetRotation.current.z,
        delta * 5
      );
    }
  });
  
  return (
    <group ref={headRef}>
      {/* Main head shape - elongated sphere like Gideon */}
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <HologramMaterial 
          color={color}
          intensity={intensity}
          scanSpeed={avatarState.hologram.scan_line_speed}
          flicker={avatarState.hologram.flicker_amount}
          distortion={avatarState.hologram.distortion}
        />
      </mesh>
      
      {/* Inner glow core */}
      <mesh scale={[0.9, 0.9, 0.9]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshBasicMaterial 
          color={color} 
          transparent 
          opacity={0.1}
        />
      </mesh>
      
      {/* Face features */}
      <group position={[0, 0.1, 0]}>
        {/* Eyes */}
        <Eye 
          position={[-0.3, 0.15, 0.85]}
          lookX={avatarState.eyes.left.x}
          lookY={avatarState.eyes.left.y}
          openness={avatarState.eyes.left.open}
          pupilSize={avatarState.eyes.pupil_size}
          color={color}
        />
        <Eye 
          position={[0.3, 0.15, 0.85]}
          lookX={avatarState.eyes.right.x}
          lookY={avatarState.eyes.right.y}
          openness={avatarState.eyes.right.open}
          pupilSize={avatarState.eyes.pupil_size}
          color={color}
        />
        
        {/* Eyebrows */}
        <Eyebrow 
          position={[-0.3, 0.35, 0.9]}
          angle={avatarState.expression.eyebrow_left}
          side="left"
          color={color}
        />
        <Eyebrow 
          position={[0.3, 0.35, 0.9]}
          angle={avatarState.expression.eyebrow_right}
          side="right"
          color={color}
        />
        
        {/* Mouth */}
        <Mouth 
          open={avatarState.mouth.open}
          wide={avatarState.mouth.wide}
          smile={avatarState.mouth.smile}
          viseme={avatarState.mouth.viseme}
          color={color}
        />
      </group>
      
      {/* Cheekbones/structure lines */}
      <mesh position={[-0.6, 0, 0.6]} rotation={[0, 0.5, 0]}>
        <boxGeometry args={[0.4, 0.02, 0.02]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} />
      </mesh>
      <mesh position={[0.6, 0, 0.6]} rotation={[0, -0.5, 0]}>
        <boxGeometry args={[0.4, 0.02, 0.02]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} />
      </mesh>
    </group>
  );
}

// Particle System
function HologramParticles({ 
  color = '#00BFFF',
  density = 0.5,
  radius = 2
}: {
  color?: string;
  density?: number;
  radius?: number;
}) {
  return (
    <Sparkles
      count={Math.floor(100 * density)}
      scale={radius * 2}
      size={2}
      speed={0.3}
      color={color}
      opacity={0.6}
    />
  );
}

// === HUD COMPONENTS ===

// HUD Indicator Bar
function HUDIndicatorBar({
  position,
  indicator,
  width = 0.4
}: {
  position: [number, number, number];
  indicator: HUDIndicator;
  width?: number;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const fillRef = useRef<THREE.Mesh>(null);
  const [pulse, setPulse] = useState(0);
  
  useFrame((state) => {
    if (fillRef.current && indicator.animated) {
      const p = Math.sin(state.clock.elapsedTime * indicator.pulse_speed * 3) * 0.1;
      fillRef.current.scale.x = Math.max(0.01, indicator.value / indicator.max_value + p);
    } else if (fillRef.current) {
      fillRef.current.scale.x = Math.max(0.01, indicator.value / indicator.max_value);
    }
  });
  
  if (!indicator.visible) return null;
  
  const fillWidth = width * (indicator.value / indicator.max_value);
  
  return (
    <Billboard position={position}>
      <group ref={groupRef}>
        {/* Label */}
        <Text 
          position={[-width/2 - 0.1, 0, 0]} 
          fontSize={0.06} 
          color={indicator.color}
          anchorX="right"
        >
          {indicator.icon} {indicator.label}
        </Text>
        
        {/* Background bar */}
        <mesh position={[0, 0, -0.01]}>
          <planeGeometry args={[width, 0.05]} />
          <meshBasicMaterial color="#001133" transparent opacity={0.5} />
        </mesh>
        
        {/* Fill bar */}
        <mesh 
          ref={fillRef}
          position={[-width/2 * (1 - indicator.value / indicator.max_value), 0, 0]}
        >
          <planeGeometry args={[width, 0.04]} />
          <meshBasicMaterial color={indicator.color} transparent opacity={0.8} />
        </mesh>
        
        {/* Value text */}
        <Text 
          position={[width/2 + 0.08, 0, 0]} 
          fontSize={0.05} 
          color="#FFFFFF"
          anchorX="left"
        >
          {Math.round(indicator.value * 100)}%
        </Text>
      </group>
    </Billboard>
  );
}

// Mode Indicator Badge
function ModeIndicatorBadge({
  position,
  mode,
  color
}: {
  position: [number, number, number];
  mode: string;
  color: string;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.z = state.clock.elapsedTime * 0.5;
    }
  });
  
  const modeLabels: Record<string, string> = {
    passive: '👁️ PASSIVE',
    copilot: '🤝 COPILOT',
    pilot: '🚀 PILOT',
    executive: '👑 EXECUTIVE'
  };
  
  return (
    <Billboard position={position}>
      <group>
        {/* Glow ring */}
        <mesh ref={meshRef}>
          <ringGeometry args={[0.18, 0.22, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.6} />
        </mesh>
        
        {/* Inner circle */}
        <mesh>
          <circleGeometry args={[0.15, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.3} />
        </mesh>
        
        {/* Mode text */}
        <Text 
          position={[0, -0.35, 0]} 
          fontSize={0.08} 
          color={color}
          fontWeight="bold"
        >
          {modeLabels[mode] || mode.toUpperCase()}
        </Text>
      </group>
    </Billboard>
  );
}

// Calculation Animation Ring
function CalculationRing({
  active,
  progress,
  color,
  radius = 1.5
}: {
  active: boolean;
  progress: number;
  color: string;
  radius?: number;
}) {
  const ringRef = useRef<THREE.Mesh>(null);
  const [currentProgress, setCurrentProgress] = useState(0);
  
  useFrame((state, delta) => {
    if (!active) return;
    
    if (ringRef.current) {
      ringRef.current.rotation.z = state.clock.elapsedTime * 2;
      setCurrentProgress(prev => prev + delta * 0.5);
    }
  });
  
  if (!active) return null;
  
  return (
    <group>
      {/* Rotating progress ring */}
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]} position={[0, -0.8, 0]}>
        <torusGeometry args={[radius, 0.03, 8, 64, Math.PI * 2 * (progress || currentProgress % 1)]} />
        <meshBasicMaterial color={color} transparent opacity={0.8} />
      </mesh>
      
      {/* Static outer ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -0.8, 0]}>
        <torusGeometry args={[radius + 0.05, 0.01, 8, 64]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} />
      </mesh>
    </group>
  );
}

// Data Stream Animation
function DataStreamEffect({
  active,
  color,
  dataPoints = []
}: {
  active: boolean;
  color: string;
  dataPoints?: string[];
}) {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current && active) {
      groupRef.current.children.forEach((child, i) => {
        child.position.y = ((state.clock.elapsedTime * 0.5 + i * 0.2) % 2) - 1;
        const mesh = child as THREE.Mesh;
        if (mesh.material && !Array.isArray(mesh.material)) {
          (mesh.material as THREE.MeshBasicMaterial).opacity = 
            1 - Math.abs(child.position.y);
        }
      });
    }
  });
  
  if (!active) return null;
  
  return (
    <group ref={groupRef} position={[1.8, 0, 0]}>
      {dataPoints.slice(-5).map((text, i) => (
        <Billboard key={i} position={[0, -1 + i * 0.3, 0]}>
          <Text 
            fontSize={0.06} 
            color={color}
            anchorX="left"
          >
            {text.slice(0, 20)}
          </Text>
        </Billboard>
      ))}
    </group>
  );
}

// Activity Status Display
function ActivityStatusDisplay({
  position,
  activity,
  color
}: {
  position: [number, number, number];
  activity: string;
  color: string;
}) {
  const [dots, setDots] = useState('');
  
  useEffect(() => {
    if (activity === 'idle') return;
    
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 300);
    
    return () => clearInterval(interval);
  }, [activity]);
  
  const activityLabels: Record<string, string> = {
    idle: '',
    analyzing: '🔍 ANALYZING',
    processing: '⚡ PROCESSING',
    executing: '🎯 EXECUTING',
    alert: '⚠️ ALERT'
  };
  
  if (activity === 'idle') return null;
  
  return (
    <Billboard position={position}>
      <Text 
        fontSize={0.1} 
        color={color}
        fontWeight="bold"
      >
        {activityLabels[activity]}{dots}
      </Text>
    </Billboard>
  );
}

// Complete HUD Overlay
function HUDOverlay({
  avatarState,
  showIndicators = true
}: {
  avatarState: AvatarState;
  showIndicators?: boolean;
}) {
  const mode = avatarState.operatingMode || 'copilot';
  const activity = avatarState.activityState || 'idle';
  const feedback = avatarState.feedback;
  
  // Get color based on mode/activity
  const getActiveColor = () => {
    if (activity === 'analyzing') return OperatingModeColors.ANALYZING;
    if (activity === 'processing') return OperatingModeColors.PROCESSING;
    if (activity === 'alert') return OperatingModeColors.ALERT;
    return OperatingModeColors[mode.toUpperCase() as keyof typeof OperatingModeColors] || OperatingModeColors.COPILOT;
  };
  
  const activeColor = getActiveColor();
  
  // Default indicators if not provided
  const defaultIndicators: Record<string, HUDIndicator> = {
    confidence: {
      id: 'confidence',
      label: 'CONF',
      value: 0.85,
      max_value: 1,
      color: '#00FF7F',
      icon: '🎯',
      visible: true,
      animated: false,
      pulse_speed: 1
    },
    progress: {
      id: 'progress',
      label: 'PROG',
      value: 0,
      max_value: 1,
      color: activeColor,
      icon: '📊',
      visible: activity !== 'idle',
      animated: true,
      pulse_speed: 2
    }
  };
  
  const indicators = feedback?.hud_indicators || defaultIndicators;
  const calculations = feedback?.calculations || {};
  
  return (
    <group>
      {/* Mode Badge - Top */}
      <ModeIndicatorBadge
        position={[0, 1.8, 0]}
        mode={mode}
        color={activeColor}
      />
      
      {/* Activity Status - Below mode */}
      <ActivityStatusDisplay
        position={[0, 1.4, 0]}
        activity={activity}
        color={activeColor}
      />
      
      {/* HUD Indicators - Left side */}
      {showIndicators && Object.values(indicators).filter(i => i.visible).map((ind, i) => (
        <HUDIndicatorBar
          key={ind.id}
          position={[-2, 0.5 - i * 0.2, 0]}
          indicator={ind}
        />
      ))}
      
      {/* Calculation Ring */}
      <CalculationRing
        active={activity === 'analyzing' || activity === 'processing'}
        progress={calculations.primary?.progress || 0}
        color={activeColor}
      />
      
      {/* Data Stream */}
      <DataStreamEffect
        active={calculations.data_stream?.active || false}
        color={activeColor}
        dataPoints={calculations.data_stream?.data_points || []}
      />
    </group>
  );
}

// === MAIN AVATAR COMPONENT ===

function AvatarScene({
  avatarState,
  showParticles = true,
  showRings = true,
  showHUD = true,
  scale = 1,
  debugMode = false
}: {
  avatarState: AvatarState;
  showParticles?: boolean;
  showRings?: boolean;
  showHUD?: boolean;
  scale?: number;
  debugMode?: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);
  
  // Idle floating animation
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.05;
    }
  });
  
  // Determine glow color based on operating mode and activity
  const getEffectiveGlowColor = (): string => {
    const activity = avatarState.activityState || 'idle';
    const mode = avatarState.operatingMode || 'copilot';
    
    // Activity overrides mode color
    if (activity === 'analyzing') return OperatingModeColors.ANALYZING;
    if (activity === 'processing') return OperatingModeColors.PROCESSING;
    if (activity === 'alert') return OperatingModeColors.ALERT;
    
    // Mode-based color
    const modeColorKey = mode.toUpperCase() as keyof typeof OperatingModeColors;
    return OperatingModeColors[modeColorKey] || avatarState.hologram.glow_color;
  };
  
  const glowColor = getEffectiveGlowColor();
  const glowIntensity = avatarState.hologram.glow_intensity;
  
  return (
    <group ref={groupRef} scale={scale}>
      {/* Main head */}
      <Float 
        speed={1} 
        rotationIntensity={0.1} 
        floatIntensity={0.2}
        floatingRange={[-0.02, 0.02]}
      >
        <GideonHead 
          avatarState={avatarState}
          color={glowColor}
          intensity={glowIntensity}
        />
      </Float>
      
      {/* Holographic rings */}
      {showRings && (
        <HologramRings 
          count={avatarState.hologram.ring_count}
          color={glowColor}
        />
      )}
      
      {/* Particles */}
      {showParticles && (
        <HologramParticles 
          color={glowColor}
          density={avatarState.hologram.particle_density}
        />
      )}
      
      {/* Point lights for glow effect */}
      <pointLight 
        position={[0, 0, 2]} 
        intensity={glowIntensity * 0.5} 
        distance={5} 
        color={glowColor}
      />
      <pointLight 
        position={[0, 0, -1]} 
        intensity={glowIntensity * 0.3} 
        distance={3} 
        color={glowColor}
      />
      
      {/* Ambient glow sphere */}
      <mesh scale={[2.5, 2.5, 2.5]}>
        <sphereGeometry args={[1, 16, 16]} />
        <meshBasicMaterial 
          color={glowColor} 
          transparent 
          opacity={0.03}
          side={THREE.BackSide}
        />
      </mesh>
      
      {/* Debug info */}
      {debugMode && (
        <Billboard position={[0, -2, 0]}>
          <Text fontSize={0.1} color="white">
            {`State: ${avatarState.state} | Mode: ${avatarState.operatingMode} | Activity: ${avatarState.activityState}`}
          </Text>
        </Billboard>
      )}
      
      {/* HUD Overlay */}
      {showHUD && (
        <HUDOverlay avatarState={avatarState} />
      )}
    </group>
  );
}

// === EXPORTED COMPONENT ===

export default function GideonAvatar3D({
  avatarState: externalState,
  isSpeaking = false,
  expression = 'neutral',
  operatingMode = 'copilot',
  activityState = 'idle',
  scale = 1,
  onReady,
  showParticles = true,
  showRings = true,
  showHUD = true,
  debugMode = false
}: GideonAvatarProps) {
  
  // Get color based on mode
  const getModeColor = (mode: string): string => {
    const modeKey = mode.toUpperCase() as keyof typeof OperatingModeColors;
    return OperatingModeColors[modeKey] || OperatingModeColors.COPILOT;
  };
  
  // Default avatar state
  const defaultState: AvatarState = {
    state: isSpeaking ? 'speaking' : 'idle',
    operatingMode: operatingMode,
    activityState: activityState,
    head: { pitch: 0, yaw: 0, roll: 0 },
    eyes: {
      left: { x: 0, y: 0, open: 1 },
      right: { x: 0, y: 0, open: 1 },
      pupil_size: 1
    },
    mouth: {
      open: isSpeaking ? 0.3 : 0,
      wide: 0.5,
      smile: 0,
      viseme: 'sil'
    },
    expression: {
      expression: expression,
      intensity: 1,
      eyebrow_left: 0,
      eyebrow_right: 0
    },
    hologram: {
      glow_intensity: operatingMode === 'pilot' ? 0.9 : 0.6,
      glow_color: getModeColor(operatingMode),
      scan_line_speed: activityState === 'processing' ? 3 : 1,
      flicker_amount: activityState === 'alert' ? 0.1 : 0.02,
      distortion: 0,
      particle_density: operatingMode === 'pilot' ? 0.8 : 0.5,
      ring_count: operatingMode === 'pilot' ? 5 : 3
    },
    timestamp: 0
  };
  
  // Merge external state with defaults
  const [avatarState, setAvatarState] = useState<AvatarState>(() => ({
    ...defaultState,
    ...externalState
  }));
  
  // Update state when props change
  useEffect(() => {
    setAvatarState(prev => ({
      ...prev,
      ...externalState,
      state: isSpeaking ? 'speaking' : prev.state,
      operatingMode: operatingMode,
      activityState: activityState,
      expression: {
        ...prev.expression,
        expression: expression
      },
      hologram: {
        ...prev.hologram,
        glow_color: getModeColor(operatingMode),
        glow_intensity: operatingMode === 'pilot' ? 0.9 : 0.6,
        scan_line_speed: activityState === 'processing' ? 3 : 1,
        particle_density: operatingMode === 'pilot' ? 0.8 : 0.5
      }
    }));
  }, [externalState, isSpeaking, expression, operatingMode, activityState]);
  
  // Simulate lip sync when speaking
  useEffect(() => {
    if (!isSpeaking) return;
    
    const interval = setInterval(() => {
      setAvatarState(prev => ({
        ...prev,
        mouth: {
          ...prev.mouth,
          open: 0.1 + Math.random() * 0.4,
          viseme: ['aa', 'E', 'O', 'sil'][Math.floor(Math.random() * 4)]
        }
      }));
    }, 100);
    
    return () => clearInterval(interval);
  }, [isSpeaking]);
  
  // Blink simulation
  useEffect(() => {
    const blink = () => {
      setAvatarState(prev => ({
        ...prev,
        eyes: {
          ...prev.eyes,
          left: { ...prev.eyes.left, open: 0 },
          right: { ...prev.eyes.right, open: 0 }
        }
      }));
      
      setTimeout(() => {
        setAvatarState(prev => ({
          ...prev,
          eyes: {
            ...prev.eyes,
            left: { ...prev.eyes.left, open: 1 },
            right: { ...prev.eyes.right, open: 1 }
          }
        }));
      }, 150);
    };
    
    const interval = setInterval(blink, 3000 + Math.random() * 2000);
    return () => clearInterval(interval);
  }, []);
  
  // Notify when ready
  useEffect(() => {
    onReady?.();
  }, [onReady]);
  
  return (
    <div className="w-full h-full" style={{ background: 'transparent' }}>
      <Canvas
        camera={{ position: [0, 0, 3.5], fov: 45 }}
        gl={{ 
          alpha: true, 
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.2
        }}
        dpr={[1, 2]}
      >
        <Suspense fallback={null}>
          {/* Minimal ambient light */}
          <ambientLight intensity={0.1} />
          
          {/* Main scene */}
          <AvatarScene 
            avatarState={avatarState}
            showParticles={showParticles}
            showRings={showRings}
            showHUD={showHUD}
            scale={scale}
            debugMode={debugMode}
          />
          
          {/* Camera controls - limited */}
          <OrbitControls
            enableZoom={false}
            enablePan={false}
            minPolarAngle={Math.PI / 2.5}
            maxPolarAngle={Math.PI / 1.5}
            minAzimuthAngle={-Math.PI / 4}
            maxAzimuthAngle={Math.PI / 4}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}

// Export types and constants
export { OperatingModeColors };
export type { AvatarState, GideonAvatarProps, HUDIndicator, CalculationAnimation };
