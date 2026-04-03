import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@r128/build/three.module.js';

const NeuralViz3D = () => {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const wsRef = useRef(null);
  const particleRef = useRef(null);
  const layersRef = useRef([]);
  const attentionBeamsRef = useRef([]);
  const trailRef = useRef(null);
  const orbitStateRef = useRef({ theta: Math.PI / 4, phi: Math.PI / 3, radius: 80 });

  const [connected, setConnected] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentToken, setCurrentToken] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [topLogits, setTopLogits] = useState([]);
  const [currentLayer, setCurrentLayer] = useState(-1);
  const [speed, setSpeed] = useState(1);
  const [isPaused, setIsPaused] = useState(false);
  const [layerData, setLayerData] = useState({});
  const [errorMsg, setErrorMsg] = useState('');

  // Initialize Three.js scene
  useEffect(() => {
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0f);
    scene.fog = new THREE.Fog(0x0a0a0f, 200, 500);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(60, 40, 60);
    camera.lookAt(0, 20, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0x00ffff, 1, 200);
    pointLight1.position.set(50, 50, 50);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xff00ff, 0.8, 200);
    pointLight2.position.set(-50, 50, -50);
    scene.add(pointLight2);

    // Create 36 transformer layers
    const layerHeight = 2;
    const layerSpacing = 1.5;
    const layerGroup = new THREE.Group();

    for (let i = 0; i < 36; i++) {
      const z = i * layerSpacing;

      // Layer geometry: hexagonal plane
      const geometry = new THREE.CylinderGeometry(8, 8, 0.3, 6);
      const material = new THREE.MeshStandardMaterial({
        color: 0x1a1a2e,
        emissive: 0x0a0a0f,
        metalness: 0.5,
        roughness: 0.5,
        transparent: true,
        opacity: 0.7,
      });

      const layer = new THREE.Mesh(geometry, material);
      layer.position.z = z;
      layer.rotation.x = Math.PI / 2;
      layer.userData = { layerIdx: i, originalMaterial: material.clone() };

      layerGroup.add(layer);
      layersRef.current.push(layer);

      // Add some random initial "hot neurons"
      for (let j = 0; j < 3; j++) {
        const neuronGeom = new THREE.SphereGeometry(0.3, 8, 8);
        const neuronMat = new THREE.MeshBasicMaterial({ color: 0x00ffff });
        const neuron = new THREE.Mesh(neuronGeom, neuronMat);

        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * 6;
        neuron.position.set(Math.cos(angle) * radius, Math.sin(angle) * radius, z);
        layerGroup.add(neuron);
      }
    }

    scene.add(layerGroup);

    // Token particle (cyan glowing sphere)
    const particleGeom = new THREE.SphereGeometry(0.6, 16, 16);
    const particleMat = new THREE.MeshBasicMaterial({
      color: 0x00ffff,
      emissive: 0x00ffff,
    });
    const particle = new THREE.Mesh(particleGeom, particleMat);
    particle.position.z = -5;
    scene.add(particle);
    particleRef.current = particle;

    // Trail geometry for particle
    const trailGeom = new THREE.BufferGeometry();
    const trailPositions = [];
    const trailMat = new THREE.LineBasicMaterial({
      color: 0x00ffff,
      linewidth: 2,
      transparent: true,
      opacity: 0.6,
    });
    trailGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(trailPositions), 3));
    const trail = new THREE.Line(trailGeom, trailMat);
    scene.add(trail);
    trailRef.current = { line: trail, positions: trailPositions };

    // Keyboard + mouse controls
    const onMouseMove = (e) => {
      if (e.buttons === 1) {
        // Left mouse drag
        const deltaX = e.movementX * 0.01;
        const deltaY = e.movementY * 0.01;
        orbitStateRef.current.theta -= deltaX;
        orbitStateRef.current.phi -= deltaY;
        orbitStateRef.current.phi = Math.max(0.1, Math.min(Math.PI - 0.1, orbitStateRef.current.phi));
      }
    };

    const onWheel = (e) => {
      e.preventDefault();
      orbitStateRef.current.radius *= 1 + e.deltaY * 0.001;
      orbitStateRef.current.radius = Math.max(30, Math.min(200, orbitStateRef.current.radius));
    };

    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false });

    // Animation loop
    let trailCounter = 0;
    const animate = () => {
      requestAnimationFrame(animate);

      // Update orbit camera
      const { theta, phi, radius } = orbitStateRef.current;
      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.cos(phi);
      const z = radius * Math.sin(phi) * Math.sin(theta);
      camera.position.set(x, y + 20, z);
      camera.lookAt(0, 20, 0);

      // Animate layers (pulsing glow)
      layersRef.current.forEach((layer, idx) => {
        const data = layerData[idx];
        if (data) {
          const intensity = 0.3 + (data.residual_norm / 10) * 0.5;
          const hue = data.mlp_energy > data.attention_energy ? 0xff6600 : 0x00ffff;
          layer.material.emissive.setHex(hue);
          layer.material.emissive.multiplyScalar(intensity);
        }

        if (idx === currentLayer) {
          layer.material.opacity = 0.9;
          layer.scale.x = 1.2;
          layer.scale.y = 1.2;
        } else {
          layer.material.opacity = 0.5;
          layer.scale.x = 1;
          layer.scale.y = 1;
        }
      });

      // Update particle trail
      if (trailCounter % 5 === 0) {
        trailRef.current.positions.push(particle.position.x, particle.position.y, particle.position.z);
        if (trailRef.current.positions.length > 300) {
          trailRef.current.positions.shift();
          trailRef.current.positions.shift();
          trailRef.current.positions.shift();
        }
        trailRef.current.line.geometry.attributes.position.needsUpdate = true;
      }
      trailCounter++;

      // Particle glow animation
      particleMat.emissive.multiplyScalar(0.9 + Math.sin(Date.now() * 0.005) * 0.1);

      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      renderer.domElement.removeEventListener('mousemove', onMouseMove);
      renderer.domElement.removeEventListener('wheel', onWheel);
      containerRef.current?.removeChild(renderer.domElement);
    };
  }, [currentLayer, layerData]);

  // WebSocket connection
  useEffect(() => {
    const connectWS = () => {
      try {
        const ws = new WebSocket('ws://localhost:8080');

        ws.onopen = () => {
          console.log('WebSocket connected');
          setConnected(true);
          setErrorMsg('');
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);

            if (msg.type === 'pipeline_layer') {
              setCurrentLayer(msg.layer_idx);
              setLayerData((prev) => ({
                ...prev,
                [msg.layer_idx]: {
                  attention_energy: msg.attention_energy,
                  mlp_energy: msg.mlp_energy,
                  residual_norm: msg.residual_norm,
                  max_activation: msg.max_activation,
                  hot_neurons: msg.hot_neurons || [],
                },
              }));

              // Animate particle moving up
              if (particleRef.current) {
                const targetZ = msg.layer_idx * 1.5;
                const current = particleRef.current.position.z;
                particleRef.current.position.z += (targetZ - current) * 0.1 * speed;
              }
            } else if (msg.type === 'pipeline_result') {
              setCurrentToken(msg.token_str);
              setGeneratedText(msg.generated_so_far);
              if (msg.top_logits) {
                setTopLogits(msg.top_logits.slice(0, 5));
              }

              // Move particle to top when token generated
              if (particleRef.current && msg.step > 0) {
                particleRef.current.position.z = 36 * 1.5 + 5;
              }
            } else if (msg.type === 'status') {
              if (msg.message.includes('completa')) {
                setIsGenerating(false);
              }
            }
          } catch (e) {
            console.error('Failed to parse WS message:', e);
          }
        };

        ws.onerror = () => {
          setConnected(false);
          setErrorMsg('Failed to connect to IARA Engine');
        };

        ws.onclose = () => {
          setConnected(false);
          setTimeout(connectWS, 3000);
        };

        wsRef.current = ws;
      } catch (e) {
        setErrorMsg(`Connection error: ${e.message}`);
      }
    };

    connectWS();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [speed]);

  const handleGenerate = useCallback(() => {
    if (!prompt.trim()) return;

    if (connected && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stimulus', text: prompt }));
      setIsGenerating(true);
      setGeneratedText('');
      setCurrentToken('');
      setTopLogits([]);
      setCurrentLayer(-1);

      // Reset particle to start
      if (particleRef.current) {
        particleRef.current.position.z = -5;
      }
      trailRef.current.positions = [];
    } else {
      // Demo mode: simulate generation
      simulateGeneration();
    }
  }, [prompt, connected]);

  const simulateGeneration = () => {
    setIsGenerating(true);
    setGeneratedText('');
    let text = '';
    const tokens = prompt.split(' ').slice(0, 3);
    let tokenIdx = 0;

    const interval = setInterval(() => {
      if (tokenIdx < tokens.length) {
        text += (text ? ' ' : '') + tokens[tokenIdx];
        setGeneratedText(text);
        setCurrentToken(tokens[tokenIdx]);
        setCurrentLayer(Math.min(35, tokenIdx * 12));

        // Simulate top logits
        setTopLogits([
          { token: tokens[tokenIdx], logit: 8.5 - tokenIdx * 0.5 },
          { token: 'response', logit: 7.2 },
          { token: 'question', logit: 6.8 },
          { token: 'answer', logit: 6.1 },
          { token: 'here', logit: 5.5 },
        ]);

        tokenIdx++;
      } else {
        setIsGenerating(false);
        clearInterval(interval);
      }
    }, 800 / speed);
  };

  const handleReset = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset' }));
    }
    setGeneratedText('');
    setCurrentToken('');
    setCurrentLayer(-1);
    setTopLogits([]);
    if (particleRef.current) {
      particleRef.current.position.z = -5;
    }
    trailRef.current.positions = [];
  };

  return (
    <div className="w-full h-screen bg-gray-950 text-white flex">
      {/* 3D Visualization */}
      <div
        ref={containerRef}
        className="flex-1 bg-gradient-to-br from-slate-900 to-slate-950 relative"
        style={{ flex: 3 }}
      />

      {/* Side Panel */}
      <div className="bg-gray-900 border-l border-cyan-500/30 p-6 overflow-y-auto" style={{ width: '380px' }}>
        <h1 className="text-2xl font-bold text-cyan-400 mb-6">Neural Viz</h1>

        {/* Connection Status */}
        <div className="mb-6 p-3 rounded border-2" style={{ borderColor: connected ? '#00ff88' : '#ff4444' }}>
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: connected ? '#00ff88' : '#ff4444' }}
            />
            <span className="text-sm">{connected ? 'Connected to IARA' : 'Disconnected (Demo Mode)'}</span>
          </div>
        </div>

        {errorMsg && <div className="mb-4 p-3 bg-red-900/30 border border-red-500 rounded text-sm">{errorMsg}</div>}

        {/* Prompt Input */}
        <div className="mb-6">
          <label className="block text-cyan-300 text-sm font-mono mb-2">Prompt:</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt here..."
            className="w-full h-20 bg-gray-800 border border-cyan-500/50 rounded p-2 text-white text-sm focus:outline-none focus:border-cyan-400"
          />
        </div>

        {/* Controls */}
        <div className="mb-6 flex gap-2">
          <button
            onClick={handleGenerate}
            disabled={isGenerating || !prompt.trim()}
            className="flex-1 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-600 text-white font-bold py-2 rounded transition"
          >
            {isGenerating ? 'Generating...' : 'Generate'}
          </button>
          <button onClick={handleReset} className="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-bold py-2 rounded transition">
            Reset
          </button>
        </div>

        {/* Speed Control */}
        <div className="mb-6">
          <label className="block text-cyan-300 text-sm font-mono mb-2">Speed: {speed.toFixed(1)}x</label>
          <input
            type="range"
            min="0.1"
            max="3"
            step="0.1"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Current Generation */}
        <div className="mb-6 p-4 bg-gray-800/50 border border-cyan-500/20 rounded">
          <h3 className="text-cyan-300 font-mono text-sm mb-2">Generated Text:</h3>
          <div className="text-white text-sm min-h-12 font-mono">{generatedText || '(awaiting generation)'}</div>
        </div>

        {/* Current Token */}
        <div className="mb-6 p-3 bg-gray-800/50 border border-purple-500/30 rounded">
          <h3 className="text-purple-300 font-mono text-xs mb-2">Current Token:</h3>
          <div className="text-lg font-bold text-purple-400">{currentToken || '—'}</div>
        </div>

        {/* Current Layer */}
        <div className="mb-6 p-3 bg-gray-800/50 border border-orange-500/30 rounded">
          <h3 className="text-orange-300 font-mono text-xs mb-2">Layer:</h3>
          <div className="text-lg font-bold text-orange-400">{currentLayer >= 0 ? `${currentLayer} / 36` : '—'}</div>
        </div>

        {/* Top Logits */}
        <div className="mb-6">
          <h3 className="text-cyan-300 font-mono text-sm mb-3">Top Logits:</h3>
          <div className="space-y-2">
            {topLogits.length > 0 ? (
              topLogits.map((item, idx) => (
                <div key={idx} className="flex justify-between items-center bg-gray-800/50 p-2 rounded border border-cyan-500/20">
                  <span className="text-xs font-mono text-cyan-200">{item.token}</span>
                  <span className="text-xs font-bold text-orange-400">{item.logit.toFixed(2)}</span>
                </div>
              ))
            ) : (
              <div className="text-xs text-gray-500">(no logits yet)</div>
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="text-xs text-gray-400 border-t border-gray-700 pt-4 space-y-1">
          <p>
            <span className="inline-block w-2 h-2 bg-cyan-400 rounded-full mr-2" />
            Token particle path
          </p>
          <p>
            <span className="inline-block w-2 h-2 bg-blue-400 rounded-full mr-2" />
            Attention energy
          </p>
          <p>
            <span className="inline-block w-2 h-2 bg-orange-400 rounded-full mr-2" />
            MLP energy
          </p>
          <p className="mt-3">Drag to rotate • Scroll to zoom</p>
        </div>
      </div>
    </div>
  );
};

export default NeuralViz3D;
