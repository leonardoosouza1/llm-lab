// ============================================================================
// main.rs — Orquestração do Motor de Ressonância IARA
// ============================================================================
// Pipeline:
//   1. Lê index.json e lista tensores disponíveis
//   2. Carrega tensor padrão via mmap (zero-copy)
//   3. Instancia Simulation com NoLiquido para cada peso
//   4. Spawna WebSocket server (tokio)
//   5. Roda loop de simulação, enviando snapshots via broadcast
// ============================================================================

mod actuator;
mod decoder;
mod embedding;
mod generate;
mod homeostasis;
mod liquid_brain;
mod liquid_network;
mod model;
mod nlm_orchestrator;
mod physics;
mod physics_gpu;
mod pipeline;
mod proprioception;
mod prompt;
mod safetensors_reader;
mod simulation;
mod tokenizer;
mod websocket;

/// Configuração do modelo (vinda do config.json)
#[derive(Debug, serde::Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub model_type: String,
}

use actuator::{ActuatorBus, FocusActuator, InhibitionActuator, ResearchActuator};
use homeostasis::HomeostaticRegulator;
use liquid_network::{LiquidNetwork, LlmTelemetry};
use simulation::Simulation;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{broadcast, mpsc, Mutex};
use candle_core::Device;

/// Configuração do motor
struct EngineConfig {
    model_dir: PathBuf,
    ws_host: String,
    ws_port: u16,
    tick_rate_hz: f32,
    grid_size: usize,
    default_layer: Option<String>,
    max_nodes: Option<usize>,
    model_config: Option<ModelConfig>,
    device: Device,
}

impl EngineConfig {
    fn new() -> Self {
        // ── Override de GPU AMD (ROCm) ──
        // Para GPUs Navi 2x (RX 6000), o ROCm as vezes precisa de um override 
        // para funcionar corretamente no Linux se o driver for genérico.
        if std::env::var("HSA_OVERRIDE_GFX_VERSION").is_err() {
            std::env::set_var("HSA_OVERRIDE_GFX_VERSION", "10.3.0");
        }

        // Tentar usar GPU (Candle tentará carregar HIP/ROCm em runtime)
        let device = match Device::new_cuda(0) {
            Ok(d) => {
                log::info!("🎯 Hardware Selecionado: GPU/HIP ({:?})", d);
                d
            }
            Err(e) => {
                log::warn!("⚠️ GPU não detectada ou erro no driver: {:?}. Usando CPU.", e);
                Device::Cpu
            }
        };

        EngineConfig {
            model_dir: PathBuf::from("."),
            ws_host: "0.0.0.0".to_string(),
            ws_port: 8080,
            tick_rate_hz: 60.0,
            grid_size: 128,
            default_layer: None,
            max_nodes: Some(1_000_000),
            model_config: None,
            device,
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_args() -> EngineConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut config = EngineConfig::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                if i < args.len() {
                    config.model_dir = PathBuf::from(&args[i]);
                }
            }
            "--port" => {
                i += 1;
                if i < args.len() {
                    config.ws_port = args[i].parse().unwrap_or(8080);
                }
            }
            "--tick-rate" => {
                i += 1;
                if i < args.len() {
                    config.tick_rate_hz = args[i].parse().unwrap_or(60.0);
                }
            }
            "--grid-size" => {
                i += 1;
                if i < args.len() {
                    config.grid_size = args[i].parse().unwrap_or(128);
                }
            }
            "--layer" => {
                i += 1;
                if i < args.len() {
                    config.default_layer = Some(args[i].clone());
                }
            }
            "--max-nodes" => {
                i += 1;
                if i < args.len() {
                    config.max_nodes = Some(args[i].parse().unwrap_or(1_000_000));
                }
            }
            "--help" => {
                println!("🧬 IARA Resonance Engine — Motor de Ressonância");
                println!();
                println!("USO: iara-engine [OPÇÕES]");
                println!();
                println!("OPÇÕES:");
                println!("  --model-dir <PATH>    Diretório com os arquivos safetensors (default: .)");
                println!("  --port <PORT>         Porta do WebSocket server (default: 8080)");
                println!("  --tick-rate <HZ>      Taxa de simulação em Hz (default: 60)");
                println!("  --grid-size <N>       Tamanho do grid NxN para heatmap (default: 128)");
                println!("  --layer <NAME>        Nome do tensor inicial (default: auto)");
                println!("  --max-nodes <N>       Máximo de nós líquidos (default: 1000000)");
                println!("  --help                Mostra esta ajuda");
                std::process::exit(0);
            }
            _ => {
                eprintln!("⚠️  Argumento desconhecido: {}", args[i]);
            }
        }
        i += 1;
    }

    config
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── PASSO 0: Override de Hardware (CRÍTICO) ──
    // Deve ser a primeira linha do binário.
    std::env::set_var("HSA_OVERRIDE_GFX_VERSION", "10.3.0");

    // Inicializar logging
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_millis()
        .init();

    log::info!("🚀 IARA Engine Iniciando...");

    // Validar se o driver ROCm está vendo a RX 6750 XT
    if candle_core::utils::cuda_is_available() {
        log::info!("🔥 GPU HEARTBEAT: AMD ROCm/HIP detectado e pronto para o Direct Point!");
    } else {
        log::warn!("⚠️ GPU HEARTBEAT: Placa AMD não detectada. Verifique o container ROCm.");
    }

    let config = parse_args();
    let mut config = config; // Torna mutável para injetar ModelConfig

    log::info!("╔══════════════════════════════════════════════════════════╗");
    log::info!("║     🧬 IARA RESONANCE ENGINE — Motor de Ressonância    ║");
    log::info!("║     Liquid Neural Network Physics Simulator             ║");
    log::info!("╚══════════════════════════════════════════════════════════╝");
    log::info!("");

    // ── 0. Carregar config.json ──
    let config_path = config.model_dir.join("config.json");
    if config_path.exists() {
        log::info!("⚙️  Lendo config.json: {}", config_path.display());
        let config_str = std::fs::read_to_string(&config_path)?;
        match serde_json::from_str::<ModelConfig>(&config_str) {
            Ok(mcfg) => {
                log::info!("✅ Config carregada: {} ({} layers, {} hidden, {} vocab)", 
                    mcfg.model_type, mcfg.num_hidden_layers, mcfg.hidden_size, mcfg.vocab_size);
                config.model_config = Some(mcfg);
            }
            Err(e) => log::warn!("⚠️  Erro ao parsear config.json: {}", e),
        }
    } else {
        log::warn!("⚠️  config.json não encontrado em {}", config.model_dir.display());
    }

    // ── 1. Carregar index.json ──
    let index_path = config.model_dir.join("model.safetensors.index.json");
    log::info!("📄 Carregando index: {}", index_path.display());
    let index = safetensors_reader::load_index(&index_path)?;

    let tensor_names = safetensors_reader::list_tensor_names(&index);
    log::info!("📋 {} tensores disponíveis no modelo", tensor_names.len());

    // ── 1b. Carregar tokenizer ──
    match tokenizer::load_tokenizer(&config.model_dir) {
        Ok(()) => log::info!("✅ Tokenizer BPE carregado com sucesso"),
        Err(e) => log::warn!("⚠️  Tokenizer não carregado (fallback byte-level): {}", e),
    }

    // ── 2. Selecionar tensor inicial ──
    let initial_layer = config.default_layer.clone().unwrap_or_else(|| {
        // Escolher um tensor MLP de tamanho razoável por padrão
        // Preferência: layernorm (pequeno, bom para teste rápido)
        tensor_names
            .iter()
            .find(|n| n.contains("layers.0.input_layernorm"))
            .cloned()
            .unwrap_or_else(|| tensor_names[0].clone())
    });

    log::info!("🎯 Tensor inicial: {}", initial_layer);

    // ── 3. Carregar tensor via mmap ──
    let tensor_data = safetensors_reader::read_tensor(&config.model_dir, &index, &initial_layer)?;

    // Limitar número de nós se necessário
    let weights = if let Some(max) = config.max_nodes {
        if tensor_data.data.len() > max {
            log::warn!(
                "⚠️  Tensor tem {} floats, limitando a {} nós",
                tensor_data.data.len(),
                max
            );
            tensor_data.data[..max].to_vec()
        } else {
            tensor_data.data
        }
    } else {
        tensor_data.data
    };

    // ── 4. Instanciar simulação ──
    let mut sim = Simulation::from_weights(&weights, &initial_layer, config.device.clone());
    sim.grid_size = config.grid_size;

    // ── 4b. LiquidNetwork neuromórfica (STDP + EWC + Homeostase) ──
    let num_layers = config.model_config.as_ref().map(|c| c.num_hidden_layers).unwrap_or(32);
    // Usamos 256 osciladores distribuídos pelas camadas.
    let liquid_net = Arc::new(StdMutex::new(LiquidNetwork::new(256, num_layers, config.device.clone())));
    
    // ── 4c. Inicializar GPU Physics se disponível ──
    if !config.device.is_cpu() {
        if let Err(e) = liquid_net.lock().unwrap().init_gpu() {
            log::warn!("⚠️ Falha ao inicializar kernels GPU (usando path CPU): {}", e);
        }
    }

    // Restaurar estado consolidado da sessão anterior
    {
        let mut regulator = HomeostaticRegulator::new();
        regulator.try_restore(&liquid_net);
    }

    log::info!("🧬 LiquidNetwork: 256 osciladores CfC distribuídos em {} camadas", num_layers);

    // ── 4c. QwenModel Funcional (Direct Point / Tensor Math) ──
    // Este é o motor que realmente faz a matemática para responder "O que é um carro"
    // Carregamos os metadados e os tensores reais para a GPU (Axiomatic KAN Mesh).
    log::info!("⚡ Inicializando Motor Funcional (Carregando Qwen 8B na VRAM)...");
    let model_config = config.model_config.as_ref()
        .expect("config.json é necessário para carregar o modelo funcional");

    let qwen_model = Arc::new(Mutex::new(
        crate::model::QwenModel::load(&config.model_dir, &config.device, model_config)
            .unwrap_or_else(|e| panic!("Falha fatal ao carregar o modelo na VRAM: {}", e))
    ));


    // ── 5. Canais de comunicação ──
    // Comandos do WS handler → loop principal
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<websocket::EngineCommand>(64);

    // Estímulos de Pensamento (Pipeline → Simulação Global + LiquidNetwork)
    // Agora recebe (layer_idx, activation)
    let (_think_tx, mut think_rx) = mpsc::channel::<(usize, Vec<f32>)>(128);

    // Telemetria do pipeline → LiquidNetwork (canal dedicado)
    let (lnn_tel_tx, _) = broadcast::channel::<LlmTelemetry>(256);

    // Frames do loop → todos os clientes WS (broadcast)
    let (frame_tx, frame_rx) = broadcast::channel::<String>(16);

    // Compartilhar lista de tensores
    let tensor_names_arc = Arc::new(tensor_names);
    let index_arc = Arc::new(index);
    let model_dir_arc = Arc::new(config.model_dir.clone());

    // ── 6. Spawnar WebSocket server ──
    let ws_addr = format!("{}:{}", config.ws_host, config.ws_port)
        .parse()
        .expect("Endereço WS inválido");

    let cmd_tx_ws = cmd_tx.clone();
    tokio::spawn(async move {
        websocket::start_ws_server(ws_addr, cmd_tx_ws, frame_rx).await;
    });

    // ── 6b. Task: Física da LiquidNetwork (100 Hz) ──────────────────────────
    {
        let lnn = liquid_net.clone();
        let mut lnn_rx = lnn_tel_tx.subscribe();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(10));
            let mut slow_tick = 0u64;
            loop {
                interval.tick().await;
                let mut net = lnn.lock().unwrap();
                while let Ok(tel) = lnn_rx.try_recv() {
                    net.inject_telemetry(&tel);
                }
                if net.device.is_cpu() {
                    let _ = net.step();
                } else if net.gpu_module.is_some() {
                    let _ = net.step_gpu(0.01);
                } else {
                    let _ = net.step(); // Fallback CPU
                }
                slow_tick += 1;
                if slow_tick % 100 == 0 {
                    let energy = net.mean_energy();
                    net.slow_update(energy.sqrt().min(1.0) as f64);
                }
            }
        });
    }

    // ── 6c. Task: Homeostase Circadiana (1 Hz) ───────────────────────────────
    {
        let lnn = liquid_net.clone();
        let mut lnn_rx2 = lnn_tel_tx.subscribe();
        tokio::spawn(async move {
            let mut regulator = HomeostaticRegulator::new();
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            let mut last_tel: Option<LlmTelemetry> = None;
            loop {
                interval.tick().await;
                while let Ok(tel) = lnn_rx2.try_recv() {
                    last_tel = Some(tel);
                }
                regulator.tick(&lnn, last_tel.as_ref());
                last_tel = None;
            }
        });
    }

    // ── 6d. Task: ActuatorBus (1 Hz) ────────────────────────────────────────
    {
        let lnn = liquid_net.clone();
        let ftx = frame_tx.clone();
        tokio::spawn(async move {
            let mut bus = ActuatorBus::new();
            bus.register(ResearchActuator::new(60)); // Cooldown 60s
            bus.register(InhibitionActuator::new());
            bus.register(FocusActuator::new());
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                let commands = bus.tick(&lnn);
                for (name, action) in &commands {
                    log::info!("[ACTUATOR:{}] → {}", name, action);
                    
                    // Se o ResearchActuator disparar, fazemos uma consulta real ao NLM
                    if name == "ResearchActuator" {
                        let query = "O que há de novo sobre este tópico?"; // Poderia ser derivado do estado
                        tokio::spawn(async move {
                            match nlm_orchestrator::NlmOrchestrator::extract_knowledge("iara-main", query).await {
                                Ok(res) => log::info!("🧬 NLM DISCOVERY: {}", res),
                                Err(e) => log::warn!("⚠️ NLM Research failed: {}", e),
                            }
                        });
                    }

                    // Enviar ações ao frontend via WebSocket
                    let msg = serde_json::json!({
                        "type": "actuator",
                        "actuator": name,
                        "action": action,
                    });
                    let _ = ftx.send(msg.to_string());
                }
            }
        });
    }

    // ── 6e. Task: Brain Map Visualization (20 Hz) ───────────────────────────
    {
        let lnn = liquid_net.clone();
        let ftx = frame_tx.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(50));
            let mut tick = 0u64;
            loop {
                interval.tick().await;
                let net = lnn.lock().unwrap();
                let snap = crate::liquid_brain::snapshot(&net, tick);
                if let Ok(json) = serde_json::to_string(&snap) {
                    let _ = ftx.send(json);
                }
                tick += 1;
            }
        });
    }

    // ── 7. Loop principal de simulação ──
    let dt = 1.0 / config.tick_rate_hz;
    let tick_duration = tokio::time::Duration::from_secs_f32(dt);
    let frames_per_send = 2; // Enviar a cada 2 ticks (~30 FPS de telemetria)

    log::info!(
        "🔄 Simulação iniciada: dt={:.4}s ({} Hz), grid={}x{}, {} nós",
        dt,
        config.tick_rate_hz,
        config.grid_size,
        config.grid_size,
        sim.nodes.len()
    );
    log::info!("🌐 WebSocket: ws://{}", ws_addr);
    log::info!("──────────────────────────────────────────────────────────");

    let sim = Arc::new(Mutex::new(sim));

    loop {
        let loop_start = tokio::time::Instant::now();

        // Processar comandos do frontend (non-blocking)
        while let Ok(cmd) = cmd_rx.try_recv() {
            let mut sim_lock = sim.lock().await;
            match cmd {
                websocket::EngineCommand::Stimulus(text) => {
                    log::info!("💬 Estímulo recebido: '{}'", text);
                    let energy = prompt::text_to_energy(&text);
                    sim_lock.apply_stimulus(&energy);
                }
                websocket::EngineCommand::Reset => {
                    sim_lock.reset_all();
                    let status = serde_json::json!({
                        "type": "status",
                        "message": "🔄 Simulação resetada"
                    });
                    let _ = frame_tx.send(status.to_string());
                }
                websocket::EngineCommand::ListLayers => {
                    let layers_msg = serde_json::json!({
                        "type": "layers",
                        "names": *tensor_names_arc
                    });
                    let _ = frame_tx.send(layers_msg.to_string());
                }
                websocket::EngineCommand::Resonate(text) => {
                    log::info!("🧬 MODO FUNCIONAL: Gerando resposta para '{}'", text);
                    drop(sim_lock); // Liberar lock da simulação

                    let model_arc = qwen_model.clone();
                    let ftx = frame_tx.clone();

                    tokio::spawn(async move {
                        // Travar o modelo para o loop de geração
                        let mut model = model_arc.lock().await;
                        
                        match crate::generate::generate_response(&text, &mut model, &ftx, 100).await {
                            Ok(_full_response) => {
                                log::info!("✅ Resposta completa gerada");
                                let _ = ftx.send(serde_json::json!({
                                    "type": "status",
                                    "message": "✅ Geração finalizada"
                                }).to_string());
                            }
                            Err(e) => {
                                log::error!("❌ Geração falhou: {}", e);
                            }
                        }
                    });
                    continue;
                }
                websocket::EngineCommand::SelectLayer(name) => {
                    log::info!("🔀 Trocando para tensor: {}", name);
                    let loading_msg = serde_json::json!({
                        "type": "loading",
                        "layer": name,
                        "progress": "Carregando tensor via mmap..."
                    });
                    let _ = frame_tx.send(loading_msg.to_string());

                    match safetensors_reader::read_tensor(&model_dir_arc, &index_arc, &name) {
                        Ok(tensor_data) => {
                            let w = if let Some(max) = config.max_nodes {
                                if tensor_data.data.len() > max {
                                    tensor_data.data[..max].to_vec()
                                } else {
                                    tensor_data.data
                                }
                            } else {
                                tensor_data.data
                            };
                            *sim_lock = Simulation::from_weights(&w, &name, config.device.clone());
                            sim_lock.grid_size = config.grid_size;

                            let status = serde_json::json!({
                                "type": "status",
                                "message": format!("✅ Tensor '{}' carregado: {} nós", name, sim_lock.nodes.len())
                            });
                            let _ = frame_tx.send(status.to_string());
                        }
                        Err(e) => {
                            log::error!("❌ Erro ao carregar tensor '{}': {}", name, e);
                            let status = serde_json::json!({
                                "type": "status",
                                "message": format!("❌ Erro: {}", e)
                            });
                            let _ = frame_tx.send(status.to_string());
                        }
                    }
                }
            }
        }
        // Processar estímulos vindos do pensamento (Pipeline por Camada)
        while let Ok((layer_idx, energy)) = think_rx.try_recv() {
            // 1. Injetar na simulação global (Fauna) se estivermos visualizando a mesma camada
            {
                let mut sim_lock = sim.lock().await;
                if sim_lock.layer_name.contains(&format!(".{}.", layer_idx)) || sim_lock.layer_name == "initial" {
                    sim_lock.apply_stimulus(&energy);
                }
            }
            // 2. Injetar no Cérebro Líquido (Coluna Cortical)
            {
                let mut net = liquid_net.lock().unwrap();
                net.inject_layer_stimulus(layer_idx, &energy);
            }
        }

        // ── Tick da simulação ──
        {
            let mut sim_lock = sim.lock().await;
            sim_lock.tick(dt);

            // Enviar frame de telemetria a cada N ticks
            if sim_lock.tick_count % frames_per_send as u64 == 0 {
                let grid = sim_lock.snapshot_grid();
                let frame = websocket::TelemetryFrame {
                    msg_type: "frame".to_string(),
                    grid_size: sim_lock.grid_size,
                    data: grid,
                    tick: sim_lock.tick_count,
                    layer: sim_lock.layer_name.clone(),
                    energy: sim_lock.total_energy(),
                    active_nodes: sim_lock.active_count(0.01),
                    total_nodes: sim_lock.nodes.len(),
                };

                if let Ok(json) = serde_json::to_string(&frame) {
                    let _ = frame_tx.send(json);
                }
            }
        }

        // Manter tick rate
        let elapsed = loop_start.elapsed();
        if elapsed < tick_duration {
            tokio::time::sleep(tick_duration - elapsed).await;
        }
    }
}
