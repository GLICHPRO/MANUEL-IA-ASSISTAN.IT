import React, { Suspense, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Gideon 3D Avatar Component
 * Displays an animated 3D avatar with lip sync and expressions
 */

interface AvatarProps {
  expression?: 'neutral' | 'happy' | 'thinking' | 'focused' | 'concerned' | 'confident';
  isSpeaking?: boolean;
  lipSyncData?: number[];
}

function AvatarModel({ expression = 'neutral', isSpeaking = false }: AvatarProps) {
  const groupRef = useRef<THREE.Group>(null);
  const headRef = useRef<THREE.Mesh>(null);
  const mouthRef = useRef<THREE.Mesh>(null);
  const leftEyeRef = useRef<THREE.Mesh>(null);
  const rightEyeRef = useRef<THREE.Mesh>(null);

  // Idle animation
  useFrame((state) => {
    if (groupRef.current) {
      // Gentle floating animation
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.05;
    }

    if (headRef.current) {
      // Gentle head rotation
      headRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.1;
    }

    // Lip sync animation when speaking
    if (isSpeaking && mouthRef.current) {
      const mouthOpen = Math.abs(Math.sin(state.clock.elapsedTime * 10)) * 0.5;
      mouthRef.current.scale.y = 1 + mouthOpen;
    }
  });

  // Expression configurations
  const expressions = {
    neutral: { eyeSize: 1.0, mouthCurve: 0, eyebrowAngle: 0, color: '#4A90E2' },
    happy: { eyeSize: 0.9, mouthCurve: 0.3, eyebrowAngle: 0.1, color: '#52C41A' },
    thinking: { eyeSize: 0.8, mouthCurve: -0.1, eyebrowAngle: -0.15, color: '#FA8C16' },
    focused: { eyeSize: 1.1, mouthCurve: 0, eyebrowAngle: -0.05, color: '#1890FF' },
    concerned: { eyeSize: 1.2, mouthCurve: -0.2, eyebrowAngle: -0.2, color: '#F5222D' },
    confident: { eyeSize: 1.0, mouthCurve: 0.2, eyebrowAngle: 0.05, color: '#722ED1' }
  };

  const currentExpression = expressions[expression];

  return (
    <group ref={groupRef}>
      {/* Head */}
      <mesh ref={headRef} position={[0, 0, 0]}>
        <sphereGeometry args={[1.2, 32, 32]} />
        <meshStandardMaterial color={currentExpression.color} metalness={0.3} roughness={0.4} />
      </mesh>

      {/* Eyes */}
      <mesh ref={leftEyeRef} position={[-0.4, 0.3, 1.0]}>
        <sphereGeometry args={[0.15 * currentExpression.eyeSize, 16, 16]} />
        <meshStandardMaterial color="#000000" />
      </mesh>
      
      <mesh ref={rightEyeRef} position={[0.4, 0.3, 1.0]}>
        <sphereGeometry args={[0.15 * currentExpression.eyeSize, 16, 16]} />
        <meshStandardMaterial color="#000000" />
      </mesh>

      {/* Eyebrows */}
      <mesh position={[-0.4, 0.5 + currentExpression.eyebrowAngle, 1.05]} rotation={[0, 0, currentExpression.eyebrowAngle]}>
        <boxGeometry args={[0.4, 0.08, 0.08]} />
        <meshStandardMaterial color="#333333" />
      </mesh>
      
      <mesh position={[0.4, 0.5 + currentExpression.eyebrowAngle, 1.05]} rotation={[0, 0, -currentExpression.eyebrowAngle]}>
        <boxGeometry args={[0.4, 0.08, 0.08]} />
        <meshStandardMaterial color="#333333" />
      </mesh>

      {/* Mouth */}
      <mesh ref={mouthRef} position={[0, -0.3 + currentExpression.mouthCurve * 0.3, 1.0]}>
        <capsuleGeometry args={[0.05, 0.5, 4, 8]} />
        <meshStandardMaterial color="#222222" />
      </mesh>

      {/* Glow effect when speaking */}
      {isSpeaking && (
        <pointLight position={[0, 0, 0]} intensity={1} distance={5} color={currentExpression.color} />
      )}
    </group>
  );
}

export default function Avatar3D({ expression, isSpeaking }: AvatarProps) {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [0, 0, 4], fov: 50 }}
        gl={{ alpha: true, antialias: true }}
      >
        <Suspense fallback={null}>
          {/* Lighting */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 5, 5]} intensity={1} />
          <directionalLight position={[-5, -5, -5]} intensity={0.3} />
          
          {/* Avatar */}
          <AvatarModel expression={expression} isSpeaking={isSpeaking} />
          
          {/* Controls */}
          <OrbitControls
            enableZoom={false}
            enablePan={false}
            minPolarAngle={Math.PI / 2.5}
            maxPolarAngle={Math.PI / 1.5}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}
