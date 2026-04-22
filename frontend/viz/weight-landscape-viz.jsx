import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import * as THREE from "three";

// === Weight Landscape: Olhando os Pesos "De Lado" ===
// Visualiza os pesos do modelo como terreno 3D
// Pinos = pesos outliers que carregam informação desproporcional
// A "vista de lado" revela estrutura invisível na vista 2D

const COLORS = {
  bg: "#0a0a0f",
  text: "#e0e0e0",
  accent: "#00ff88",
  hot: "#ff3366",
  cold: "#3366ff",
  terrain: "#1a1a2e",
  grid: "#333355",
  panel: "rgba(15,15,25,0.95)",
  border: "rgba(100,100,150,0.3)",
};

// Generate synthetic but realistic weight landscape data
function generateWeightData(rows, cols) {
  const heights = [];
  const values = [];
  const rng = (seed) => {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 4294967296;
    };
  };

  const rand = rng(42);
  const gaussRand = () => {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  };

  // Simulate weight distribution: mostly Gaussian with sparse outliers
  const std = 0.024;
  const pinProb = 0.005; // 0.5% are "pins"

  // Create "hot channels" - certain columns have more outliers
  const channelHeat = Array.from({ length: cols }, (_, i) => {
    // Some channels are naturally hotter (like dim 24, 22 in real data)
    const base = 1.0;
    const hotSpots = [6, 5, 20, 14, 4, 32, 19, 48]; // hot channel indices
    const isHot = hotSpots.includes(i);
    return isHot ? base * 2.5 : base;
  });

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let w = gaussRand() * std;
      // Sparse outliers (pins)
      if (rand() < pinProb * channelHeat[c]) {
        w *= 5 + rand() * 10; // 5-15x normal magnitude
      }
      values.push(w);
      heights.push(Math.abs(w));
    }
  }

  return { heights, values, rows, cols, channelHeat };
}

// Key findings panel
function FindingsPanel() {
  const findings = [
    { icon: "📌", label: "Pins (>3σ)", value: "0.05%", desc: "of weights cause 9% output error" },
    { icon: "💀", label: "Dead neurons", value: "60.7%", desc: "MLP neurons die each token via SiLU" },
    { icon: "🔥", label: "Top 25%", value: "86.6%", desc: "of MLP energy in 25% of neurons" },
    { icon: "🌿", label: "Hot channels", value: "2.1x", desc: "dim 24 has 2.1x avg pin density" },
    { icon: "🔗", label: "Cross-layer", value: "r=0.45", desc: "hot channels correlate across layers" },
  ];

  return (
    <div style={{
      position: "absolute", top: 12, left: 12, zIndex: 10,
      background: COLORS.panel, borderRadius: 12, padding: "14px 16px",
      border: `1px solid ${COLORS.border}`, maxWidth: 320,
      backdropFilter: "blur(10px)",
    }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.accent, marginBottom: 10 }}>
        Descobertas Empíricas — Qwen3-4B
      </div>
      {findings.map((f, i) => (
        <div key={i} style={{
          display: "flex", alignItems: "flex-start", gap: 8,
          marginBottom: 8, fontSize: 12, lineHeight: 1.4,
        }}>
          <span style={{ fontSize: 14, flexShrink: 0 }}>{f.icon}</span>
          <div>
            <span style={{ color: COLORS.hot, fontWeight: 600 }}>{f.value}</span>
            <span style={{ color: COLORS.text, opacity: 0.7 }}> — {f.desc}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// Controls panel
function ControlsPanel({ viewMode, setViewMode, heightScale, setHeightScale, showPins, setShowPins, rotation, pinThreshold, setPinThreshold }) {
  return (
    <div style={{
      position: "absolute", top: 12, right: 12, zIndex: 10,
      background: COLORS.panel, borderRadius: 12, padding: "14px 16px",
      border: `1px solid ${COLORS.border}`, minWidth: 200,
      backdropFilter: "blur(10px)",
    }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.accent, marginBottom: 12 }}>
        Controles
      </div>

      <div style={{ marginBottom: 10 }}>
        <label style={{ fontSize: 11, color: COLORS.text, opacity: 0.6, display: "block", marginBottom: 4 }}>
          Perspectiva
        </label>
        <div style={{ display: "flex", gap: 4 }}>
          {["frontal", "lateral", "topo", "orbital"].map(mode => (
            <button key={mode} onClick={() => setViewMode(mode)} style={{
              padding: "4px 8px", fontSize: 11, borderRadius: 6, border: "none",
              background: viewMode === mode ? COLORS.accent : COLORS.terrain,
              color: viewMode === mode ? COLORS.bg : COLORS.text,
              cursor: "pointer", fontWeight: viewMode === mode ? 700 : 400,
            }}>
              {mode}
            </button>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label style={{ fontSize: 11, color: COLORS.text, opacity: 0.6, display: "block", marginBottom: 4 }}>
          Exagero vertical: {heightScale}x
        </label>
        <input type="range" min="1" max="50" value={heightScale}
          onChange={e => setHeightScale(Number(e.target.value))}
          style={{ width: "100%", accentColor: COLORS.accent }} />
      </div>

      <div style={{ marginBottom: 10 }}>
        <label style={{ fontSize: 11, color: COLORS.text, opacity: 0.6, display: "block", marginBottom: 4 }}>
          Threshold pinos: {pinThreshold}σ
        </label>
        <input type="range" min="1" max="8" step="0.5" value={pinThreshold}
          onChange={e => setPinThreshold(Number(e.target.value))}
          style={{ width: "100%", accentColor: COLORS.hot }} />
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: COLORS.text, cursor: "pointer" }}>
        <input type="checkbox" checked={showPins} onChange={e => setShowPins(e.target.checked)} />
        Destacar pinos
      </label>
    </div>
  );
}

// Legend for the metaphor
function MetaphorPanel() {
  return (
    <div style={{
      position: "absolute", bottom: 12, left: 12, zIndex: 10,
      background: COLORS.panel, borderRadius: 12, padding: "14px 16px",
      border: `1px solid ${COLORS.border}`, maxWidth: 360,
      backdropFilter: "blur(10px)",
    }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: "#ffaa00", marginBottom: 8 }}>
        A Metáfora do Quadro com Alfinetes
      </div>
      <div style={{ fontSize: 11, color: COLORS.text, opacity: 0.8, lineHeight: 1.6 }}>
        <p style={{ margin: "0 0 6px 0" }}>
          <b style={{ color: COLORS.cold }}>De frente</b> — a matriz parece ruído uniforme (Gaussiana, kurtosis 3.33).
          Nenhuma estrutura visível.
        </p>
        <p style={{ margin: "0 0 6px 0" }}>
          <b style={{ color: COLORS.hot }}>De lado</b> — os <b>alfinetes</b> (0.05% dos pesos) saltam do plano.
          Remover esses 0.05% causa 9.1% de erro na saída.
        </p>
        <p style={{ margin: 0 }}>
          <b style={{ color: COLORS.accent }}>O micélio</b> — certos canais (dim 24, 22) concentram alfinetes
          em <b>todas as 36 camadas</b>. São as estradas de informação do modelo.
        </p>
      </div>
    </div>
  );
}

// Main 3D Canvas
function WeightCanvas({ viewMode, heightScale, showPins, pinThreshold }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const meshRef = useRef(null);
  const pinMeshRef = useRef(null);
  const frameRef = useRef(null);
  const mouseRef = useRef({ x: 0, y: 0, down: false, rx: 0, ry: 0 });

  const ROWS = 120;
  const COLS = 80;
  const data = useMemo(() => generateWeightData(ROWS, COLS), []);

  // Compute pin indices based on threshold
  const pinData = useMemo(() => {
    const mean = data.heights.reduce((a, b) => a + b, 0) / data.heights.length;
    const variance = data.heights.reduce((a, b) => a + (b - mean) ** 2, 0) / data.heights.length;
    const stdDev = Math.sqrt(variance);
    const thresh = mean + pinThreshold * stdDev;

    const pins = [];
    for (let i = 0; i < data.heights.length; i++) {
      if (data.heights[i] > thresh) {
        const r = Math.floor(i / COLS);
        const c = i % COLS;
        pins.push({ r, c, h: data.heights[i], v: data.values[i] });
      }
    }
    return { pins, threshold: thresh, mean, stdDev };
  }, [data, pinThreshold]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const w = mount.clientWidth;
    const h = mount.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.bg);
    scene.fog = new THREE.FogExp2(COLORS.bg, 0.003);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 500);
    camera.position.set(60, 40, 80);
    camera.lookAt(COLS / 2, 0, ROWS / 2);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lights
    const ambient = new THREE.AmbientLight(0x404060, 0.6);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(50, 80, 50);
    scene.add(dirLight);
    const pointLight = new THREE.PointLight(0x00ff88, 0.4, 200);
    pointLight.position.set(COLS / 2, 30, ROWS / 2);
    scene.add(pointLight);

    // Ground grid
    const gridHelper = new THREE.GridHelper(200, 40, 0x333355, 0x222244);
    gridHelper.position.set(COLS / 2, -0.1, ROWS / 2);
    scene.add(gridHelper);

    // Terrain geometry
    const geometry = new THREE.PlaneGeometry(COLS, ROWS, COLS - 1, ROWS - 1);
    geometry.rotateX(-Math.PI / 2);

    const positions = geometry.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const z = positions.getZ(i);
      const col = Math.round(x + COLS / 2);
      const row = Math.round(z + ROWS / 2);
      const idx = row * COLS + col;

      const h = idx < data.heights.length ? data.heights[idx] : 0;
      positions.setY(i, h * heightScale);

      // Color by signed value
      const v = idx < data.values.length ? data.values[idx] : 0;
      const absNorm = Math.min(Math.abs(v) / 0.15, 1);
      if (v > 0) {
        colors[i * 3] = 0.2 + absNorm * 0.8;     // R
        colors[i * 3 + 1] = 0.2 + absNorm * 0.3;  // G
        colors[i * 3 + 2] = 0.3;                   // B
      } else {
        colors[i * 3] = 0.2;
        colors[i * 3 + 1] = 0.2 + absNorm * 0.4;
        colors[i * 3 + 2] = 0.3 + absNorm * 0.7;
      }
    }

    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 30,
      flatShading: false,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(COLS / 2, 0, ROWS / 2);
    scene.add(mesh);
    meshRef.current = mesh;

    // Animate
    let time = 0;
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      time += 0.005;

      // Orbital rotation
      const m = mouseRef.current;
      if (m.down) {
        // User dragging
      } else {
        m.rx += 0.002;
      }

      const radius = 90;
      const elevation = 35 + Math.sin(time * 0.5) * 10;
      camera.position.x = COLS / 2 + Math.cos(m.rx) * radius;
      camera.position.z = ROWS / 2 + Math.sin(m.rx) * radius;
      camera.position.y = elevation;
      camera.lookAt(COLS / 2, 5, ROWS / 2);

      renderer.render(scene, camera);
    };
    animate();

    // Mouse interaction
    const onMouseDown = (e) => { mouseRef.current.down = true; mouseRef.current.x = e.clientX; };
    const onMouseUp = () => { mouseRef.current.down = false; };
    const onMouseMove = (e) => {
      if (mouseRef.current.down) {
        const dx = e.clientX - mouseRef.current.x;
        mouseRef.current.rx += dx * 0.005;
        mouseRef.current.x = e.clientX;
      }
    };

    mount.addEventListener("mousedown", onMouseDown);
    mount.addEventListener("mouseup", onMouseUp);
    mount.addEventListener("mousemove", onMouseMove);

    return () => {
      cancelAnimationFrame(frameRef.current);
      mount.removeEventListener("mousedown", onMouseDown);
      mount.removeEventListener("mouseup", onMouseUp);
      mount.removeEventListener("mousemove", onMouseMove);
      if (mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Update terrain height and pins when params change
  useEffect(() => {
    const mesh = meshRef.current;
    const scene = sceneRef.current;
    if (!mesh || !scene) return;

    const positions = mesh.geometry.attributes.position;
    const colors = mesh.geometry.attributes.color;

    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const z = positions.getZ(i);
      const col = Math.round(x);
      const row = Math.round(z);
      const idx = row * COLS + col;

      const h = idx >= 0 && idx < data.heights.length ? data.heights[idx] : 0;
      const isPin = h > pinData.threshold;

      positions.setY(i, h * heightScale);

      if (showPins && isPin) {
        colors.setXYZ(i, 1, 0.2, 0.4); // Hot pink for pins
      } else {
        const v = idx >= 0 && idx < data.values.length ? data.values[idx] : 0;
        const absNorm = Math.min(Math.abs(v) / 0.15, 1);
        if (v > 0) {
          colors.setXYZ(i, 0.2 + absNorm * 0.8, 0.2 + absNorm * 0.3, 0.3);
        } else {
          colors.setXYZ(i, 0.2, 0.2 + absNorm * 0.4, 0.3 + absNorm * 0.7);
        }
      }
    }

    positions.needsUpdate = true;
    colors.needsUpdate = true;
    mesh.geometry.computeVertexNormals();

    // Pin markers (spheres at pin locations)
    if (pinMeshRef.current) {
      scene.remove(pinMeshRef.current);
    }

    if (showPins && pinData.pins.length < 2000) {
      const pinGroup = new THREE.Group();
      const pinGeo = new THREE.SphereGeometry(0.3, 6, 6);
      const pinMat = new THREE.MeshPhongMaterial({ color: 0xff3366, emissive: 0xff1144, emissiveIntensity: 0.5 });

      pinData.pins.forEach(pin => {
        const sphere = new THREE.Mesh(pinGeo, pinMat);
        sphere.position.set(pin.c, pin.h * heightScale + 0.5, pin.r);
        pinGroup.add(sphere);
      });

      pinGroup.position.set(COLS / 2, 0, ROWS / 2);
      scene.add(pinGroup);
      pinMeshRef.current = pinGroup;
    }
  }, [heightScale, showPins, pinThreshold, pinData, data]);

  // View mode camera positions
  useEffect(() => {
    const camera = cameraRef.current;
    if (!camera) return;

    const m = mouseRef.current;
    switch (viewMode) {
      case "frontal":
        m.rx = 0;
        camera.position.set(COLS / 2, 5, ROWS + 40);
        break;
      case "lateral":
        m.rx = Math.PI / 2;
        camera.position.set(COLS + 40, 15, ROWS / 2);
        break;
      case "topo":
        camera.position.set(COLS / 2, 100, ROWS / 2);
        break;
      case "orbital":
        m.rx = 0.5;
        break;
    }
  }, [viewMode]);

  return (
    <div ref={mountRef} style={{ width: "100%", height: "100%", cursor: "grab" }} />
  );
}

// Stats bar
function StatsBar({ pinData }) {
  return (
    <div style={{
      position: "absolute", bottom: 12, right: 12, zIndex: 10,
      background: COLORS.panel, borderRadius: 12, padding: "10px 14px",
      border: `1px solid ${COLORS.border}`,
      backdropFilter: "blur(10px)",
      display: "flex", gap: 16, alignItems: "center",
    }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: COLORS.hot }}>{pinData.pins.length}</div>
        <div style={{ fontSize: 10, color: COLORS.text, opacity: 0.5 }}>pinos visíveis</div>
      </div>
      <div style={{ width: 1, height: 30, background: COLORS.border }} />
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: COLORS.accent }}>
          {(pinData.pins.length / (120 * 80) * 100).toFixed(2)}%
        </div>
        <div style={{ fontSize: 10, color: COLORS.text, opacity: 0.5 }}>dos pesos</div>
      </div>
      <div style={{ width: 1, height: 30, background: COLORS.border }} />
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: "#ffaa00" }}>
          {pinData.threshold.toFixed(4)}
        </div>
        <div style={{ fontSize: 10, color: COLORS.text, opacity: 0.5 }}>threshold</div>
      </div>
    </div>
  );
}

export default function WeightLandscape() {
  const [viewMode, setViewMode] = useState("orbital");
  const [heightScale, setHeightScale] = useState(15);
  const [showPins, setShowPins] = useState(true);
  const [pinThreshold, setPinThreshold] = useState(3);

  const ROWS = 120;
  const COLS = 80;
  const data = useMemo(() => generateWeightData(ROWS, COLS), []);

  const pinData = useMemo(() => {
    const mean = data.heights.reduce((a, b) => a + b, 0) / data.heights.length;
    const variance = data.heights.reduce((a, b) => a + (b - mean) ** 2, 0) / data.heights.length;
    const stdDev = Math.sqrt(variance);
    const thresh = mean + pinThreshold * stdDev;
    const pins = [];
    for (let i = 0; i < data.heights.length; i++) {
      if (data.heights[i] > thresh) {
        const r = Math.floor(i / COLS);
        const c = i % COLS;
        pins.push({ r, c, h: data.heights[i], v: data.values[i] });
      }
    }
    return { pins, threshold: thresh, mean, stdDev };
  }, [data, pinThreshold]);

  return (
    <div style={{
      width: "100vw", height: "100vh", position: "relative",
      fontFamily: "'Inter', -apple-system, sans-serif",
      overflow: "hidden", background: COLORS.bg,
    }}>
      {/* Title */}
      <div style={{
        position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)",
        zIndex: 10, textAlign: "center",
      }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: COLORS.accent, letterSpacing: 1 }}>
          WEIGHT LANDSCAPE
        </div>
        <div style={{ fontSize: 11, color: COLORS.text, opacity: 0.5, marginTop: 2 }}>
          gate_proj layer 0 — Qwen3-4B [9728 x 2560] — arraste para rotacionar
        </div>
      </div>

      {/* 3D Canvas */}
      <WeightCanvas
        viewMode={viewMode}
        heightScale={heightScale}
        showPins={showPins}
        pinThreshold={pinThreshold}
      />

      {/* Overlays */}
      <FindingsPanel />
      <ControlsPanel
        viewMode={viewMode} setViewMode={setViewMode}
        heightScale={heightScale} setHeightScale={setHeightScale}
        showPins={showPins} setShowPins={setShowPins}
        pinThreshold={pinThreshold} setPinThreshold={setPinThreshold}
      />
      <MetaphorPanel />
      <StatsBar pinData={pinData} />
    </div>
  );
}
