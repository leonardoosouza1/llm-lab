// ============================================================================
// IARA Resonance Engine — Frontend Dashboard (app.js)
// ============================================================================
// Connects to WebSocket, receives telemetry frames, renders heatmap on Canvas
// ============================================================================

(function () {
    'use strict';

    // ── Configuration ──
    const WS_URL = 'ws://localhost:8080';
    const RECONNECT_DELAY_MS = 3000;
    const HEATMAP_DISPLAY_SIZE = 400; // Size for heatmap canvas
    const BRAIN_DISPLAY_SIZE = 600;   // Size for brain map canvas

    // ── DOM Elements ──
    const canvas = document.getElementById('heatmapCanvas');
    const ctx = canvas.getContext('2d');
    const brainCanvas = document.getElementById('brainCanvas');
    const bctx = brainCanvas.getContext('2d');
    const overlay = document.getElementById('canvasOverlay');

    const barFocus = document.getElementById('barFocus');
    const barNovelty = document.getElementById('barNovelty');
    const barDrive = document.getElementById('barDrive');

    const connectionIndicator = document.getElementById('connectionIndicator');
    const connectionText = document.getElementById('connectionText');
    const promptInput = document.getElementById('promptInput');
    const btnSend = document.getElementById('btnSend');
    const btnReset = document.getElementById('btnReset');
    const btnResonate = document.getElementById('btnResonate');
    const btnListLayers = document.getElementById('btnListLayers');
    const layerList = document.getElementById('layerList');
    const currentLayer = document.getElementById('currentLayer');
    const logContainer = document.getElementById('logContainer');

    // Pipeline UI
    const pipelinePanel = document.getElementById('pipelinePanel');
    const pipelinePhase = document.getElementById('pipelinePhase');
    const pipelineProgress = document.getElementById('pipelineProgress');
    const responsePanel = document.getElementById('responsePanel');
    const responseTime = document.getElementById('responseTime');
    const responseTokens = document.getElementById('responseTokens');

    // Metrics
    const metricTick = document.getElementById('metricTick');
    const metricFps = document.getElementById('metricFps');
    const metricEnergy = document.getElementById('metricEnergy');
    const metricActive = document.getElementById('metricActive');
    const metricTotal = document.getElementById('metricTotal');
    const metricGrid = document.getElementById('metricGrid');

    // ── State ──
    let ws = null;
    let gridSize = 128;
    let frameCount = 0;
    let lastFpsTime = performance.now();
    let currentFps = 0;
    let imageData = null;

    // ── Color Mapping ──
    // Maps a float value to RGB color
    // Negative → cold (blue/cyan), Zero → dark, Positive → warm (red/yellow)
    function valueToColor(value) {
        // Clamp to [-1, 1] range with soft normalization
        const maxVal = 5.0; // Expected max activation
        const normalized = Math.max(-1, Math.min(1, value / maxVal));

        let r, g, b;

        if (normalized >= 0) {
            // Warm: black → red → orange → yellow
            const t = normalized;
            if (t < 0.5) {
                const s = t * 2;
                r = Math.floor(s * 255);
                g = Math.floor(s * 50);
                b = 0;
            } else {
                const s = (t - 0.5) * 2;
                r = 255;
                g = Math.floor(50 + s * 205);
                b = Math.floor(s * 40);
            }
        } else {
            // Cold: black → blue → cyan
            const t = -normalized;
            if (t < 0.5) {
                const s = t * 2;
                r = 0;
                g = Math.floor(s * 80);
                b = Math.floor(s * 255);
            } else {
                const s = (t - 0.5) * 2;
                r = 0;
                g = Math.floor(80 + s * 175);
                b = 255;
            }
        }

        return [r, g, b];
    }

    // ── Canvas Setup ──
    function setupCanvas(size) {
        gridSize = size;
        canvas.width = size;
        canvas.height = size;
        canvas.style.width = HEATMAP_DISPLAY_SIZE + 'px';
        canvas.style.height = HEATMAP_DISPLAY_SIZE + 'px';
        
        brainCanvas.width = 800; // Fixed high res for brain map
        brainCanvas.height = 700;
        
        imageData = ctx.createImageData(size, size);
        metricGrid.textContent = size + '×' + size;
    }

    setupCanvas(128);

    // ── Brain Map Rendering ──
    function renderBrainMap(snap) {
        const w = brainCanvas.width;
        const h = brainCanvas.height;
        bctx.clearRect(0, 0, w, h);

        // Update bars
        if (snap.bands) {
            barFocus.style.setProperty('--progress', (snap.bands[0].energy * 100) + '%');
            barNovelty.style.setProperty('--progress', (snap.bands[1].energy * 100) + '%');
            barDrive.style.setProperty('--progress', (snap.bands[2].energy * 100) + '%');
        }

        // Draw Synapses
        if (snap.synapses) {
            snap.synapses.forEach(syn => {
                const from = snap.nodes[syn.from];
                const to = snap.nodes[syn.to];
                if (!from || !to) return;

                const fx = from.x * w;
                const fy = from.y * h;
                const tx = to.x * w;
                const ty = to.y * h;

                bctx.beginPath();
                bctx.moveTo(fx, fy);
                bctx.lineTo(tx, ty);
                
                const alpha = syn.strength * 0.4;
                bctx.strokeStyle = syn.excitatory 
                    ? `rgba(0, 255, 136, ${alpha})` 
                    : `rgba(255, 68, 102, ${alpha})`;
                bctx.lineWidth = syn.strength * 2;
                bctx.stroke();
            });
        }

        // Draw Nodes
        snap.nodes.forEach(nd => {
            const nx = nd.x * w;
            const ny = nd.y * h;
            
            // Pulse based on activation and amplitude
            const size = 3 + nd.amplitude * 4;
            const glow = 5 + Math.abs(nd.activation) * 10;
            
            bctx.beginPath();
            bctx.arc(nx, ny, size, 0, Math.PI * 2);
            
            // Color by band
            let color;
            switch(nd.band) {
                case 0: color = '#00d4ff'; break; // Focus
                case 1: color = '#00ff88'; break; // Novelty
                default: color = '#ff4466'; break; // Drive
            }
            
            bctx.fillStyle = color;
            bctx.shadowBlur = glow;
            bctx.shadowColor = color;
            bctx.fill();
            
            // Highlight if highly active
            if (Math.abs(nd.activation) > 0.8) {
                bctx.beginPath();
                bctx.arc(nx, ny, size + 2, 0, Math.PI * 2);
                bctx.strokeStyle = 'white';
                bctx.lineWidth = 1;
                bctx.stroke();
            }
        });
        
        bctx.shadowBlur = 0; // reset
    }

    // ── Heatmap Rendering ──
    function renderHeatmap(data) {
        if (!imageData || data.length !== gridSize * gridSize) {
            if (data.length > 0) {
                const newSize = Math.floor(Math.sqrt(data.length));
                if (newSize * newSize === data.length) {
                    setupCanvas(newSize);
                }
            }
            if (!imageData) return;
        }

        const pixels = imageData.data;

        for (let i = 0; i < data.length; i++) {
            const [r, g, b] = valueToColor(data[i]);
            const idx = i * 4;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = 255; // Alpha
        }

        ctx.putImageData(imageData, 0, 0);

        // FPS tracking
        frameCount++;
        const now = performance.now();
        const elapsed = now - lastFpsTime;
        if (elapsed >= 1000) {
            currentFps = Math.round(frameCount * 1000 / elapsed);
            metricFps.textContent = currentFps;
            frameCount = 0;
            lastFpsTime = now;
        }
    }

    // ── Logging ──
    function addLog(message, level = 'info') {
        const entry = document.createElement('div');
        entry.className = 'log-entry ' + level;
        const time = new Date().toLocaleTimeString('pt-BR', { hour12: false });
        entry.textContent = '[' + time + '] ' + message;
        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;

        // Keep only last 50 entries
        while (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.firstChild);
        }
    }

    // ── WebSocket Connection ──
    function connect() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        addLog('Conectando a ' + WS_URL + '...', 'info');

        ws = new WebSocket(WS_URL);

        ws.onopen = function () {
            connectionIndicator.className = 'status-dot connected';
            connectionText.textContent = 'Conectado';
            overlay.classList.add('hidden');
            addLog('Conectado ao Motor IARA', 'success');

            // Request layer list on connect
            ws.send(JSON.stringify({ type: 'list_layers' }));
        };

        ws.onclose = function () {
            connectionIndicator.className = 'status-dot disconnected';
            connectionText.textContent = 'Desconectado';
            overlay.classList.remove('hidden');
            overlay.querySelector('.overlay-text').textContent = 'Conexão perdida. Reconectando...';
            addLog('Desconectado — reconectando em ' + (RECONNECT_DELAY_MS / 1000) + 's', 'warning');

            setTimeout(connect, RECONNECT_DELAY_MS);
        };

        ws.onerror = function (err) {
            addLog('Erro de conexão WebSocket', 'error');
        };

        ws.onmessage = function (event) {
            try {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            } catch (e) {
                addLog('Erro ao parsear mensagem: ' + e.message, 'error');
            }
        };
    }

    // ── Message Handler ──
    function handleMessage(msg) {
        switch (msg.type) {
            case 'frame':
                overlay.classList.add('hidden');
                renderHeatmap(msg.data);
                metricTick.textContent = msg.tick.toLocaleString();
                metricEnergy.textContent = msg.energy < 1000
                    ? msg.energy.toFixed(2)
                    : msg.energy.toExponential(2);
                metricActive.textContent = (msg.spikes !== undefined) ? msg.spikes.toLocaleString() : msg.active_nodes.toLocaleString();
                if (msg.spikes > 100) {
                    canvas.style.filter = 'drop-shadow(0 0 15px rgba(255, 255, 255, 0.4))';
                } else {
                    canvas.style.filter = '';
                }
                metricTotal.textContent = msg.total_nodes.toLocaleString();
                if (msg.grid_size && msg.grid_size !== gridSize) {
                    setupCanvas(msg.grid_size);
                }
                if (msg.layer) {
                    currentLayer.textContent = msg.layer;
                }
                break;

            case 'brain_map':
                renderBrainMap(msg);
                break;

            case 'generation_update':
                // Atualiza o log com a resposta em tempo real
                addLog('IARA: ' + msg.data.full_text, 'info');
                // Se você tiver um elemento dedicado para a resposta, pode atualizar aqui
                break;

            case 'layers':
                renderLayerList(msg.names);
                addLog(msg.names.length + ' tensores disponíveis', 'info');
                break;

            case 'status':
                overlay.classList.add('hidden');
                addLog(msg.message, 'success');
                break;

            case 'loading':
                addLog('Carregando: ' + msg.layer, 'info');
                overlay.classList.remove('hidden');
                overlay.querySelector('.overlay-text').textContent =
                    '⏳ ' + (msg.progress || 'Carregando...');
                break;

            case 'pipeline_layer':
                overlay.classList.add('hidden');
                renderHeatmap(msg.data);
                pipelinePanel.style.display = '';
                pipelinePhase.textContent = msg.phase || ('Layer ' + (msg.layer_idx + 1) + '/' + msg.total_layers);
                var pct = ((msg.layer_idx + 1) / (msg.total_layers || 36) * 100).toFixed(0);
                pipelineProgress.style.width = pct + '%';
                if (msg.spikes !== undefined) {
                    metricActive.textContent = msg.spikes.toLocaleString();
                    if (msg.spikes > 50) {
                        canvas.style.filter = 'brightness(1.2) drop-shadow(0 0 10px white)';
                    } else {
                        canvas.style.filter = '';
                    }
                }
                break;

            case 'pipeline_result':
                pipelinePanel.style.display = 'none';
                pipelineProgress.style.width = '0%';
                responsePanel.style.display = '';
                responseTime.textContent = '⏱ ' + msg.total_time_ms + 'ms | ' + msg.layers_processed + ' camadas | Prompt: "' + msg.prompt + '"';
                renderResponseTokens(msg.top_tokens);
                addLog('Pipeline completo: ' + msg.total_time_ms + 'ms — top: ' + (msg.top_tokens[0] ? msg.top_tokens[0].text + ' (' + msg.top_tokens[0].score.toFixed(4) + ')' : '?'), 'success');
                break;

            default:
                addLog('Mensagem desconhecida: ' + msg.type, 'warning');
        }
    }

    // ── Layer List ──
    function renderLayerList(names) {
        layerList.innerHTML = '';

        // Group by layer number for easier navigation
        const grouped = {};
        names.forEach(function (name) {
            const match = name.match(/layers\.(\d+)\./);
            const key = match ? 'Layer ' + match[1] : 'Other';
            if (!grouped[key]) grouped[key] = [];
            grouped[key].push(name);
        });

        Object.keys(grouped).sort(function (a, b) {
            const na = parseInt(a.replace(/\D/g, '')) || 999;
            const nb = parseInt(b.replace(/\D/g, '')) || 999;
            return na - nb;
        }).forEach(function (group) {
            const groupHeader = document.createElement('div');
            groupHeader.className = 'layer-item';
            groupHeader.style.color = 'var(--accent-purple)';
            groupHeader.style.fontWeight = '600';
            groupHeader.style.fontSize = '11px';
            groupHeader.style.marginTop = '6px';
            groupHeader.style.cursor = 'default';
            groupHeader.textContent = '▸ ' + group;
            layerList.appendChild(groupHeader);

            grouped[group].forEach(function (name) {
                const item = document.createElement('div');
                item.className = 'layer-item';
                // Show just the last part of the name for readability
                const shortName = name.split('.').slice(-2).join('.');
                item.textContent = '  ' + shortName;
                item.title = name;
                item.addEventListener('click', function () {
                    selectLayer(name);
                });
                if (name === currentLayer.textContent) {
                    item.classList.add('active');
                }
                layerList.appendChild(item);
            });
        });

        layerList.classList.remove('hidden');
    }

    function selectLayer(name) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'select_layer', name: name }));
            addLog('Selecionando: ' + name, 'info');
        }
    }

    // ── Event Handlers ──
    function sendStimulus() {
        const text = promptInput.value.trim();
        if (!text) return;

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'stimulus', text: text }));
            addLog('Estímulo enviado: "' + text + '"', 'success');
            promptInput.value = '';
        } else {
            addLog('Não conectado ao servidor', 'error');
        }
    }

    btnSend.addEventListener('click', sendStimulus);

    promptInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
            if (e.shiftKey) {
                sendResonate();
            } else {
                sendStimulus();
            }
        }
    });

    // ── Resonate (Full Pipeline) ──
    function sendResonate() {
        var text = promptInput.value.trim();
        if (!text) return;

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'resonate', text: text }));
            addLog('🧬 Pipeline iniciado: "' + text + '"', 'success');
            promptInput.value = '';
            pipelinePanel.style.display = '';
            pipelinePhase.textContent = 'Iniciando...';
            pipelineProgress.style.width = '0%';
            responsePanel.style.display = 'none';
        } else {
            addLog('Não conectado ao servidor', 'error');
        }
    }

    btnResonate.addEventListener('click', sendResonate);

    // ── Render Response Tokens ──
    function renderResponseTokens(tokens) {
        responseTokens.innerHTML = '';
        if (!tokens || tokens.length === 0) return;

        var maxScore = Math.abs(tokens[0].score);

        tokens.forEach(function (tok, i) {
            var row = document.createElement('div');
            row.className = 'token-row';

            var pctWidth = (Math.abs(tok.score) / maxScore * 100).toFixed(0);

            row.innerHTML =
                '<span class="token-rank">#' + (i + 1) + '</span>' +
                '<span class="token-text">' + escapeHtml(tok.text) + '</span>' +
                '<span class="token-id">ID:' + tok.token_id + '</span>' +
                '<div class="token-score-bar"><div class="token-score-fill" style="width:' + pctWidth + '%"></div></div>' +
                '<span class="token-score-value">' + tok.score.toFixed(4) + '</span>';

            responseTokens.appendChild(row);
        });
    }

    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    btnReset.addEventListener('click', function () {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'reset' }));
            addLog('Reset solicitado', 'info');
        }
    });

    btnListLayers.addEventListener('click', function () {
        if (layerList.classList.contains('hidden')) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'list_layers' }));
            }
        } else {
            layerList.classList.add('hidden');
        }
    });

    // ── Init ──
    addLog('IARA Resonance Engine — Dashboard inicializado', 'info');
    connect();

})();
