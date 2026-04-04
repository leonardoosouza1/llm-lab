import { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";

// ─── Physics simulation hooks ──────────────────────────────────────────────

function useGravitySimulation(N = 80) {
  const stateRef = useRef(null);

  const init = useCallback(() => {
    const masses = Array.from({ length: N }, (_, i) =>
      Math.pow(1 / (i + 1), 0.75) * 3.0
    );
    const positions = Array.from({ length: N }, () => ({
      x: (Math.random() - 0.5) * 50,
      y: (Math.random() - 0.5) * 50,
      z: (Math.random() - 0.5) * 50,
    }));
    const velocities = Array.from({ length: N }, () => ({ x: 0, y: 0, z: 0 }));
    stateRef.current = { masses, positions, velocities };
  }, [N]);

  const step = useCallback((dt) => {
    if (!stateRef.current) return;
    const { masses, positions, velocities } = stateRef.current;
    const G = 0.4;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = positions[j].x - positions[i].x;
        const dy = positions[j].y - positions[i].y;
        const dz = positions[j].z - positions[i].z;
        const d2 = dx * dx + dy * dy + dz * dz + 2;
        const F = (G * masses[i] * masses[j]) / d2;
        const fx = (dx / Math.sqrt(d2)) * F * 0.001;
        const fy = (dy / Math.sqrt(d2)) * F * 0.001;
        const fz = (dz / Math.sqrt(d2)) * F * 0.001;
        velocities[i].x += fx; velocities[i].y += fy; velocities[i].z += fz;
        velocities[j].x -= fx; velocities[j].y -= fy; velocities[j].z -= fz;
      }
      velocities[i].x *= 0.992; velocities[i].y *= 0.992; velocities[i].z *= 0.992;
      const r = Math.sqrt(positions[i].x ** 2 + positions[i].y ** 2 + positions[i].z ** 2);
      if (r > 30) {
        const s = 0.008;
        velocities[i].x -= positions[i].x * s;
        velocities[i].y -= positions[i].y * s;
        velocities[i].z -= positions[i].z * s;
      }
      positions[i].x += velocities[i].x;
      positions[i].y += velocities[i].y;
      positions[i].z += velocities[i].z;
    }
  }, [N]);

  return { init, step, stateRef };
}

function useKuramotoSimulation(N = 128) {
  const stateRef = useRef(null);
  const K = 3.0;

  const init = useCallback(() => {
    stateRef.current = {
      phases: Array.from({ length: N }, () => Math.random() * Math.PI * 2),
      freqs: Array.from({ length: N }, () => 0.3 + Math.random() * 2.2),
      orderR: 0,
      meanPhase: 0,
    };
  }, [N]);

  const step = useCallback((dt) => {
    if (!stateRef.current) return;
    const { phases, freqs } = stateRef.current;
    const newPhases = [...phases];
    for (let i = 0; i < N; i++) {
      let coupling = 0;
      for (let j = 0; j < N; j++) coupling += Math.sin(phases[j] - phases[i]);
      newPhases[i] = (phases[i] + (freqs[i] + (K / N) * coupling) * dt * 0.5) % (Math.PI * 2);
    }
    for (let i = 0; i < N; i++) stateRef.current.phases[i] = newPhases[i];
    let re = 0, im = 0;
    phases.forEach((p) => { re += Math.cos(p); im += Math.sin(p); });
    stateRef.current.orderR = Math.sqrt(re * re + im * im) / N;
    stateRef.current.meanPhase = Math.atan2(im, re);
  }, [N]);

  return { init, step, stateRef };
}

// ─── Three.js scene builders ──────────────────────────────────────────────

function buildGravityScene(scene, masses, positions) {
  const meshes = [];
  for (let i = 0; i < masses.length; i++) {
    const isHub = i < 5;
    const color = isHub ? 0x00ffff : i < 15 ? 0x8800ff : 0xff00ff;
    const geo = new THREE.SphereGeometry(masses[i] * 0.4 + 0.15, 8, 8);
    const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85 });
    const m = new THREE.Mesh(geo, mat);
    m.position.set(positions[i].x, positions[i].y, positions[i].z);
    scene.add(m);
    meshes.push(m);
  }
  return meshes;
}

function buildVortexArrows(scene, gridN = 22) {
  const arrows = [];
  for (let ix = 0; ix < gridN; ix++) {
    for (let iy = 0; iy < gridN; iy++) {
      const x = (ix / gridN) * 60 - 30;
      const y = (iy / gridN) * 60 - 30;
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, y, 0),
        new THREE.Vector3(x + 1, y, 0),
      ]);
      const mat = new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.6 });
      const line = new THREE.Line(geo, mat);
      scene.add(line);
      arrows.push({ line, geo, mat, x, y });
    }
  }
  return arrows;
}

function buildKANSurface(scene) {
  const res = 50;
  const geo = new THREE.PlaneGeometry(50, 50, res - 1, res - 1);
  const pos = geo.attributes.position;
  const colors = [];
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) / 5;
    const y = pos.getY(i) / 5;
    const V = (-x * x) / 2 + (x * x * x * x) / 4 + Math.sin(y) * 0.3;
    pos.setZ(i, V * 7);
    const n = Math.max(0, Math.min(1, (V + 0.8) / 1.8));
    colors.push(n > 0.5 ? (n - 0.5) * 2 : 0, Math.max(0, 1 - Math.abs(n - 0.3) * 3), n < 0.3 ? (0.3 - n) * 3 : 0);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(colors), 3));
  geo.computeVertexNormals();
  const mat = new THREE.MeshPhongMaterial({
    vertexColors: true, transparent: true, opacity: 0.75,
    side: THREE.DoubleSide, shininess: 80,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI * 0.38;
  mesh.position.y = -10;
  scene.add(mesh);
  return { mesh, geo };
}

function buildKuramotoRing(scene, N) {
  const R = 22;
  const meshes = [];
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color().setHSL(i / N, 1, 0.6) });
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.45, 8, 8), mat);
    m.position.set(Math.cos(a) * R, Math.sin(a) * R, 0);
    scene.add(m);
    meshes.push({ m, a });
  }
  // Ring guide
  const ringPts = Array.from({ length: 129 }, (_, i) => {
    const a = (i / 128) * Math.PI * 2;
    return new THREE.Vector3(Math.cos(a) * R, Math.sin(a) * R, 0);
  });
  const ring = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(ringPts),
    new THREE.LineBasicMaterial({ color: 0x003344, transparent: true, opacity: 0.5 })
  );
  scene.add(ring);
  return meshes;
}

// ─── Metric panel ──────────────────────────────────────────────────────────

function MetricBar({ label, value, color = "#0ff" }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#0ff8", marginBottom: 2 }}>
        <span>{label}</span>
        <span style={{ color: "#fff", fontWeight: "bold" }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ background: "#0ff1", height: 3, borderRadius: 2 }}>
        <div style={{ background: `linear-gradient(90deg,${color},#f0f)`, height: 3, borderRadius: 2, width: `${Math.min(100, value * 100)}%`, transition: "width 0.3s" }} />
      </div>
    </div>
  );
}

function FormulaBox({ mode }) {
  const formulas = {
    GRAVITY: {
      title: "⬤ GRAVITY ATTENTION",
      color: "#0ff",
      lines: ["m_i = ‖embed_i‖₂", "d_ij = 1 − cos(e_i, e_j)", "F_ij = G·m_i·m_j / d²", "A_ij = softmax(F_ij)"],
      note: "Substitui QKᵀ/√d por força gravitacional em espaço semântico",
    },
    VORTEX: {
      title: "⟳ VORTEX FIELD",
      color: "#f0f",
      lines: ["A = softmax(QKᵀ/√d)", "A_skew = (A − Aᵀ)/2", "A_v = A + λ·A_skew", "curl = ∂F_y/∂x − ∂F_x/∂y"],
      note: "Componente antisimétrico → tokens orbitam uns aos outros",
    },
    KAN: {
      title: "⋯ KAN BARRIER",
      color: "#0f8",
      lines: ["V(x) = −x²/2 + x⁴/4", "F(x) = x − x³ (força)", "P_tunnel ∝ e^(−2κd)", "minima em x = ±1"],
      note: "Ruído fica preso no poço; sinal forte atravessa a barreira",
    },
    RESONANCE: {
      title: "◎ KURAMOTO",
      color: "#fa0",
      lines: ["dφ/dt = ω + K/N Σ sin(φⱼ−φᵢ)", "r = |1/N Σ e^(iφⱼ)|", "Δω = η·sin(φ_true − φ)", "r=1 → sync → token!"],
      note: "Substitui softmax. Custo O(k) vs O(vocab). Aprende sem backprop.",
    },
  };
  const f = formulas[mode];
  return (
    <div style={{
      position: "absolute", top: 110, right: 20, width: 240,
      background: "#0001", border: `1px solid ${f.color}44`,
      padding: 12, fontFamily: "Courier New, monospace",
    }}>
      <div style={{ color: f.color, fontSize: 11, letterSpacing: 2, marginBottom: 8 }}>{f.title}</div>
      {f.lines.map((l, i) => (
        <div key={i} style={{ color: "#fff", fontSize: 12, marginBottom: 3 }}>{l}</div>
      ))}
      <div style={{ color: f.color + "88", fontSize: 10, marginTop: 8, lineHeight: 1.5 }}>{f.note}</div>
    </div>
  );
}

// ─── Main Observatory component ────────────────────────────────────────────

const MODES = ["GRAVITY", "VORTEX", "KAN", "RESONANCE"];
const MODE_LABELS = {
  GRAVITY: "⬤ GRAVITY",
  VORTEX: "⟳ VORTEX",
  KAN: "⋯ KAN BARRIER",
  RESONANCE: "◎ RESONANCE",
};

export default function Observatory() {
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const frameRef = useRef(null);
  const sceneObjectsRef = useRef([]);
  const camAngleRef = useRef({ theta: 0, phi: 0.3, r: 65 });
  const [mode, setMode] = useState("GRAVITY");
  const [metrics, setMetrics] = useState({ energy: 0.5, coherence: 0.3, entropy: 0.7, alive: 0.4, fps: 0 });
  const modeRef = useRef("GRAVITY");

  const gravity = useGravitySimulation(80);
  const kuramoto = useKuramotoSimulation(96);

  // Three.js init
  useEffect(() => {
    const canvas = canvasRef.current;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(devicePixelRatio);
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000008);
    scene.fog = new THREE.FogExp2(0x000008, 0.012);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(70, canvas.clientWidth / canvas.clientHeight, 0.1, 500);
    camera.position.set(0, 0, 65);
    cameraRef.current = camera;

    const resize = () => {
      renderer.setSize(canvas.clientWidth, canvas.clientHeight);
      camera.aspect = canvas.clientWidth / canvas.clientHeight;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", resize);
    return () => { window.removeEventListener("resize", resize); renderer.dispose(); };
  }, []);

  // Clear scene objects
  const clearScene = useCallback(() => {
    const scene = sceneRef.current;
    sceneObjectsRef.current.forEach((o) => {
      scene.remove(o);
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    sceneObjectsRef.current = [];
  }, []);

  const addObj = useCallback((obj) => {
    sceneRef.current.add(obj);
    sceneObjectsRef.current.push(obj);
    return obj;
  }, []);

  // Mode builders
  const buildMode = useCallback((newMode) => {
    clearScene();
    const scene = sceneRef.current;
    if (!scene) return;

    if (newMode === "GRAVITY") {
      gravity.init();
      const { masses, positions } = gravity.stateRef.current;
      buildGravityScene(scene, masses, positions).forEach((m) => sceneObjectsRef.current.push(m));
    } else if (newMode === "VORTEX") {
      buildVortexArrows(scene).forEach((a) => sceneObjectsRef.current.push(a.line));
      // Particle advection objects
      for (let i = 0; i < 200; i++) {
        const geo = new THREE.SphereGeometry(0.18, 4, 4);
        const mat = new THREE.MeshBasicMaterial({ color: 0xff00ff, transparent: true, opacity: Math.random() * 0.6 + 0.3 });
        const m = new THREE.Mesh(geo, mat);
        m.position.set((Math.random() - 0.5) * 60, (Math.random() - 0.5) * 60, 0);
        m.userData.vx = 0; m.userData.vy = 0;
        scene.add(m); sceneObjectsRef.current.push(m);
      }
    } else if (newMode === "KAN") {
      const { mesh } = buildKANSurface(scene);
      sceneObjectsRef.current.push(mesh);
      const light = new THREE.DirectionalLight(0xffffff, 1.5);
      light.position.set(20, 40, 20); scene.add(light); sceneObjectsRef.current.push(light);
      const amb = new THREE.AmbientLight(0x002244, 1.2); scene.add(amb); sceneObjectsRef.current.push(amb);
      // Particles on surface
      for (let i = 0; i < 80; i++) {
        const energy = Math.random() * 2;
        const col = energy > 1.2 ? 0xffff00 : energy > 0.6 ? 0xff8800 : 0x00ff88;
        const geo = new THREE.SphereGeometry(0.22, 6, 6);
        const mat = new THREE.MeshBasicMaterial({ color: col });
        const m = new THREE.Mesh(geo, mat);
        m.position.set((Math.random() - 0.5) * 40, (Math.random() - 0.5) * 20, 10);
        m.userData.px = (Math.random() - 0.5) * 6;
        m.userData.py = (Math.random() - 0.5) * 6;
        m.userData.vx = (Math.random() - 0.5) * 0.05;
        m.userData.vy = (Math.random() - 0.5) * 0.05;
        m.userData.energy = energy;
        scene.add(m); sceneObjectsRef.current.push(m);
      }
    } else if (newMode === "RESONANCE") {
      kuramoto.init();
      const meshes = buildKuramotoRing(scene, 96);
      meshes.forEach(({ m }) => sceneObjectsRef.current.push(m));
      // Order parameter arrow
      const orderGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(10, 0, 0)]);
      const orderLine = new THREE.Line(orderGeo, new THREE.LineBasicMaterial({ color: 0xff00ff, transparent: true, opacity: 0.9 }));
      orderLine.userData.isOrderArrow = true;
      orderLine.userData.geo = orderGeo;
      scene.add(orderLine); sceneObjectsRef.current.push(orderLine);
    }
  }, [clearScene, gravity, kuramoto, addObj]);

  // Animate loop
  useEffect(() => {
    let lastTime = performance.now();
    let frames = 0;
    let t = 0;

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      const now = performance.now();
      const dt = Math.min((now - lastTime) / 1000, 0.05);
      lastTime = now;
      t += dt;
      frames++;

      const cam = camAngleRef.current;
      cam.theta += 0.003;
      const camera = cameraRef.current;
      if (camera) {
        camera.position.x = cam.r * Math.cos(cam.phi) * Math.cos(cam.theta);
        camera.position.z = cam.r * Math.cos(cam.phi) * Math.sin(cam.theta);
        camera.position.y = cam.r * Math.sin(cam.phi);
        camera.lookAt(0, 0, 0);
      }

      const currentMode = modeRef.current;
      const scene = sceneRef.current;
      const objs = sceneObjectsRef.current;

      // Per-mode updates
      if (currentMode === "GRAVITY" && gravity.stateRef.current) {
        gravity.step(dt);
        const { positions, masses } = gravity.stateRef.current;
        objs.forEach((obj, i) => {
          if (obj.isMesh && i < positions.length) {
            obj.position.set(positions[i].x, positions[i].y, positions[i].z);
          }
        });
        const ke = gravity.stateRef.current.velocities
          .reduce((s, v, i) => s + (v.x ** 2 + v.y ** 2 + v.z ** 2) * masses[i], 0);
        if (frames % 10 === 0) setMetrics({ energy: Math.min(1, ke / 5), coherence: 0.618, entropy: 0.382, alive: 0.607, fps: Math.round(1 / dt) });
      }

      if (currentMode === "VORTEX") {
        const centers = [
          { x: -10, y: 8, s: 2.5, sign: 1 }, { x: 10, y: -8, s: 1.8, sign: -1 },
          { x: 0, y: 0, s: 3.5, sign: 1 },
        ];
        const vortex = (x, y) => {
          let vx = 0, vy = 0;
          centers.forEach((c) => {
            const dx = x - c.x, dy = y - c.y;
            const r2 = dx * dx + dy * dy + 1;
            const f = c.s * c.sign / r2;
            vx += -dy * f; vy += dx * f;
          });
          return { vx, vy };
        };
        objs.forEach((obj) => {
          if (obj.isMesh && obj.userData.vx !== undefined) {
            const p = obj.position;
            const { vx, vy } = vortex(p.x, p.y);
            obj.userData.vx = (obj.userData.vx + vx * 0.02) * 0.97;
            obj.userData.vy = (obj.userData.vy + vy * 0.02) * 0.97;
            p.x += obj.userData.vx;
            p.y += obj.userData.vy;
            if (Math.abs(p.x) > 32 || Math.abs(p.y) > 32) {
              p.x = (Math.random() - 0.5) * 50; p.y = (Math.random() - 0.5) * 50;
            }
          }
        });
        if (frames % 10 === 0) setMetrics({ energy: 0.7, coherence: 0.777, entropy: 0.233, alive: 0.85, fps: Math.round(1 / dt) });
      }

      if (currentMode === "KAN") {
        objs.forEach((obj) => {
          if (obj.isMesh && obj.userData.px !== undefined) {
            const p = obj.userData;
            const fx = -(p.px - p.px ** 3);
            const fy = -Math.sin(p.py) * 0.3;
            p.vx = p.vx * 0.88 + fx * 0.04;
            p.vy = p.vy * 0.88 + fy * 0.04;
            p.px += p.vx; p.py += p.vy;
            if (Math.abs(p.px) > 3.8) { p.px = Math.sign(p.px) * 3.7; p.vx *= -0.5; }
            if (Math.abs(p.py) > 3.8) { p.py = Math.sign(p.py) * 3.7; p.vy *= -0.5; }
            const V = -p.px ** 2 / 2 + p.px ** 4 / 4;
            obj.position.set(p.px * 6.3, (p.py * 6.3) * Math.cos(-1.19) - (V * 7 + 0.5) * Math.sin(-1.19) - 10, 0);
          }
        });
        if (frames % 10 === 0) {
          const alive = objs.filter((o) => o.isMesh && o.userData.px !== undefined && Math.abs(o.userData.px) > 0.5).length;
          const total = objs.filter((o) => o.isMesh && o.userData.px !== undefined).length || 1;
          setMetrics({ energy: 0.618, coherence: 0.42, entropy: 0.58, alive: alive / total, fps: Math.round(1 / dt) });
        }
      }

      if (currentMode === "RESONANCE" && kuramoto.stateRef.current) {
        kuramoto.step(dt);
        const { phases, orderR, meanPhase } = kuramoto.stateRef.current;
        const meshObjs = objs.filter((o) => o.isMesh);
        const R_ring = 22;
        meshObjs.forEach((obj, i) => {
          if (i >= phases.length) return;
          const baseA = (i / phases.length) * Math.PI * 2;
          const synced = Math.abs(Math.sin(phases[i] - meanPhase)) < 0.25;
          const pR = R_ring + Math.sin(phases[i] - meanPhase) * 1.8;
          obj.position.set(Math.cos(baseA) * pR, Math.sin(baseA) * pR, Math.sin(t + i * 0.3) * 0.5);
          obj.material.color.setHSL(synced ? 0.8 : i / phases.length, 1, synced ? 0.75 : 0.5);
        });
        // Order arrow
        const arrow = objs.find((o) => o.userData.isOrderArrow);
        if (arrow && arrow.userData.geo) {
          const re = Math.cos(meanPhase) * orderR * 22;
          const im = Math.sin(meanPhase) * orderR * 22;
          const arr = new Float32Array([0, 0, 0, re, im, 0]);
          arrow.userData.geo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
        }
        if (frames % 10 === 0) setMetrics({ energy: 0.4, coherence: orderR, entropy: 1 - orderR, alive: orderR, fps: Math.round(1 / dt) });
      }

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();
    return () => cancelAnimationFrame(frameRef.current);
  }, [gravity, kuramoto]);

  // Mode switch
  const handleMode = (m) => {
    setMode(m);
    modeRef.current = m;
    buildMode(m);
  };

  useEffect(() => { buildMode("GRAVITY"); }, []);

  const btnStyle = (m) => ({
    padding: "6px 14px", fontSize: 11, cursor: "pointer",
    border: `1px solid ${m === mode ? "#0ff" : "#0ff4"}`,
    background: m === mode ? "#0ff3" : "#0001",
    color: m === mode ? "#fff" : "#0ff8",
    letterSpacing: 1, boxShadow: m === mode ? "0 0 12px #0ff6" : "none",
    transition: "all 0.2s", fontFamily: "Courier New, monospace",
  });

  return (
    <div style={{ width: "100%", height: "100vh", background: "#000008", position: "relative", overflow: "hidden" }}>
      {/* Canvas */}
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />

      {/* Title */}
      <div style={{ position: "absolute", top: 18, left: "50%", transform: "translateX(-50%)", textAlign: "center", fontFamily: "Courier New, monospace", pointerEvents: "none" }}>
        <div style={{ fontSize: 16, color: "#0ff", letterSpacing: 4, textShadow: "0 0 20px #0ff" }}>◈ IARA OBSERVATORY ◈</div>
        <div style={{ fontSize: 10, color: "#0ff8", letterSpacing: 2, marginTop: 4 }}>GRAVITY · VORTEX · KAN · RESONANCE</div>
      </div>

      {/* Mode tabs */}
      <div style={{ position: "absolute", top: 60, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 4, pointerEvents: "all" }}>
        {MODES.map((m) => (
          <button key={m} style={btnStyle(m)} onClick={() => handleMode(m)}>
            {MODE_LABELS[m]}
          </button>
        ))}
      </div>

      {/* Formula */}
      <FormulaBox mode={mode} />

      {/* Metrics panel */}
      <div style={{
        position: "absolute", bottom: 20, left: 20, width: 260,
        background: "#0001", border: "1px solid #0ff3", padding: 14,
        fontFamily: "Courier New, monospace",
      }}>
        <div style={{ fontSize: 11, color: "#0ff", letterSpacing: 2, marginBottom: 10 }}>◈ LIVE METRICS</div>
        <div style={{ fontSize: 10, color: "#0ff8", marginBottom: 6 }}>MODE: <span style={{ color: "#fff" }}>{mode}</span></div>
        <MetricBar label="SYSTEM ENERGY" value={metrics.energy} color="#0ff" />
        <MetricBar label="COHERENCE (r)" value={metrics.coherence} color="#f0f" />
        <MetricBar label="ENTROPY" value={metrics.entropy} color="#ff0" />
        <MetricBar label="ALIVE NEURONS" value={metrics.alive} color="#0f8" />
        <div style={{ fontSize: 10, color: "#0ff8", marginTop: 6 }}>FPS: <span style={{ color: "#fff" }}>{metrics.fps}</span></div>
      </div>

      {/* Legend */}
      <div style={{
        position: "absolute", bottom: 20, right: mode === "GRAVITY" ? 270 : 270,
        width: 180, background: "#0001", border: "1px solid #0ff3",
        padding: 14, fontFamily: "Courier New, monospace", fontSize: 10, color: "#0ff8",
      }}>
        <div style={{ fontSize: 11, color: "#0ff", letterSpacing: 2, marginBottom: 10 }}>◈ PHYSICS ENGINE</div>
        {mode === "GRAVITY" && <>
          <div>G·m₁·m₂ / d²</div>
          <div style={{ marginTop: 4 }}>Tokens = corpos massivos</div>
          <div style={{ marginTop: 4 }}>Atenção = força gravitacional</div>
        </>}
        {mode === "VORTEX" && <>
          <div>curl(A) = ∂F_y/∂x − ∂F_x/∂y</div>
          <div style={{ marginTop: 4 }}>Informação circula</div>
          <div style={{ marginTop: 4 }}>A≠Aᵀ (direcional)</div>
        </>}
        {mode === "KAN" && <>
          <div>Poço duplo V(x)</div>
          <div style={{ marginTop: 4 }}>Sinal forte → traversa</div>
          <div style={{ marginTop: 4 }}>Ruído → preso no poço</div>
        </>}
        {mode === "RESONANCE" && <>
          <div>N=96 osciladores</div>
          <div style={{ marginTop: 4 }}>r → 1 = sincronizados</div>
          <div style={{ marginTop: 4 }}>Sem backprop, sem matmul</div>
        </>}
      </div>
    </div>
  );
}
