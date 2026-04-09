// ============================================================================
// main.rs — IARA Server: HTTP (port 3000) + WebSocket (port 8080)
// ============================================================================

// ── Core inference pipeline ───────────────────────────────────────────────────
mod acp;        // Agent Communication Protocol (ACP) module
mod api;
mod debug;
mod fire_test;  // Performance stress test ("fire test")
mod generation;
mod hardware;
mod inference;
mod lnn;
mod model;
mod speculative; // Speculative decoding framework
mod storage;
mod tokenizer;
mod transport;
mod telemetry; // Adiciona o módulo de telemetria


// ── Módulos exploratórios (pesquisa, não integrados ao pipeline) ──────────────
// Compilados apenas em modo experimental para não poluir build de produção.
#[cfg(feature = "experimental")]
mod experimental;

use crate::api::handlers::{AppState, generate_handler, debug_trace_handler, deep_trace_handler,
                            models_handler, load_model_handler, unload_model_handler,
                            train_lnn_handler, lnn_status_handler};
use crate::hardware::HardwareManager;
use crate::lnn::EsnObserver;
use crate::lnn::train::LnnTrainer;
use crate::model::CausalLM;
use crate::inference::{forward_traced, forward_traced_buffered, SamplingParams, SamplingBuffer};
use crate::transport::{start_ws_server, EngineCommand};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast, mpsc};
use axum::{routing::{get, post}, Router};
use tower_http::cors::{CorsLayer, Any};
use tower_http::services::ServeDir;
use sysinfo::System; // Importa System


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Check for special flags first
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && (args[1] == "--fire-test" || args[1] == "-f") {
        fire_test::run_fire_tests();
        return Ok(());
    }

    // Convert safetensors → .iara v2
    // Usage: iara-engine --convert <model_dir> [--recipe fp8|q4|aggressive] [--output file.iara]
    if args.len() > 2 && args[1] == "--convert" {
        let model_dir = std::path::PathBuf::from(&args[2]);
        let recipe_name = args.iter().position(|a| a == "--recipe")
            .map(|i| args.get(i + 1).map(|s| s.as_str()))
            .flatten()
            .unwrap_or("fp8");
        let output = args.iter().position(|a| a == "--output")
            .map(|i| args.get(i + 1).cloned())
            .flatten()
            .unwrap_or_else(|| {
                let stem = model_dir.file_name()
                    .unwrap_or_default().to_string_lossy();
                format!("{}.iara", stem)
            });

        // Read config.json for model metadata (needed for pulse recipe)
        let config_path = model_dir.join("config.json");
        let config: serde_json::Value = if config_path.exists() {
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?
        } else {
            eprintln!("⚠️  config.json not found at {}", config_path.display());
            std::process::exit(1);
        };

        let n_layers = config.get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let recipe = match recipe_name {
            "fp8" => storage::iara_format::QuantRecipe::fp8_standard(),
            "q4" => storage::iara_format::QuantRecipe::q4_standard(),
            "aggressive" => storage::iara_format::QuantRecipe::aggressive(),
            "pulse" => storage::iara_format::QuantRecipe::pulse(n_layers),
            "ptt" => storage::iara_format::QuantRecipe::ptt_standard(),
            _ => {
                eprintln!("Unknown recipe: {}. Use: fp8, q4, aggressive, pulse, ptt", recipe_name);
                std::process::exit(1);
            }
        };

        // Build metadata
        let metadata = storage::iara_format::IaraV2Metadata {
            architecture: config.get("model_type")
                .and_then(|v| v.as_str()).unwrap_or("unknown").into(),
            hidden_dim: config.get("hidden_size")
                .and_then(|v| v.as_u64()).unwrap_or(4096) as usize,
            n_layers: n_layers,
            vocab_size: config.get("vocab_size")
                .and_then(|v| v.as_u64()).unwrap_or(32000) as usize,
            n_heads: config.get("num_attention_heads")
                .and_then(|v| v.as_u64()).unwrap_or(32) as usize,
            n_kv_heads: config.get("num_key_value_heads")
                .and_then(|v| v.as_u64()).unwrap_or(8) as usize,
            intermediate_dim: config.get("intermediate_size")
                .and_then(|v| v.as_u64()).unwrap_or(11008) as usize,
            rope_theta: config.get("rope_theta")
                .and_then(|v| v.as_f64()).unwrap_or(10000.0),
            source_model: model_dir.file_name()
                .unwrap_or_default().to_string_lossy().into(),
            converted_at: chrono::Utc::now().to_rfc3339(),
            iara_version: "2.0".into(),
            quant_recipe: {
                let mut m = std::collections::HashMap::new();
                m.insert("default".into(), recipe_name.into());
                m
            },
            original_size_bytes: 0,
            converted_size_bytes: 0,
            compression_ratio: 1.0,
            epigenome: None,
        };

        println!("🔄 Converting: {} → {}", model_dir.display(), output);
        println!("   Recipe: {}", recipe_name);

        let output_path = std::path::PathBuf::from(&output);
        match storage::iara_format::convert_safetensors_to_iara(
            &model_dir, &output_path, &recipe, metadata
        ) {
            Ok(size) => {
                println!("✅ Conversion complete: {:.1} MB", size as f64 / 1e6);

                // Print summary
                if let Ok(reader) = storage::iara_format::IaraV2Reader::open(&output_path) {
                    println!("\n{}", reader.summary());
                }
            }
            Err(e) => {
                eprintln!("❌ Conversion failed: {}", e);
                std::process::exit(1);
            }
        }

        return Ok(());
    }

    // Analyze tensors in an .iara file
    // Usage: iara-engine --analyze-importance <model.iara>
    if args.len() > 2 && args[1] == "--analyze-importance" {
        let iara_path = std::path::PathBuf::from(&args[2]);
        match storage::iara_format::IaraV2Reader::open(&iara_path) {
            Ok(reader) => {
                println!("📊 Tensor Analysis: {}", iara_path.display());
                println!("{}", reader.summary());
                println!("\n✅ Analysis complete");
            }
            Err(e) => {
                eprintln!("❌ Failed to analyze {}: {}", iara_path.display(), e);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    // Prune dead tensors from an .iara file
    // Usage: iara-engine --prune <model.iara> --tensors tensor1,tensor2 [--output pruned.iara]
    if args.len() > 2 && args[1] == "--prune" {
        let iara_path = std::path::PathBuf::from(&args[2]);
        let tensors_str = args.iter().position(|a| a == "--tensors")
            .map(|i| args.get(i + 1).map(|s| s.as_str()))
            .flatten()
            .unwrap_or("");
        let output = args.iter().position(|a| a == "--output")
            .map(|i| args.get(i + 1).cloned())
            .flatten()
            .unwrap_or_else(|| {
                let stem = iara_path.file_stem()
                    .unwrap_or_default().to_string_lossy();
                format!("{}_pruned.iara", stem)
            });

        let tensors_to_prune: Vec<String> = tensors_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if tensors_to_prune.is_empty() {
            eprintln!("⚠️  No tensors specified. Use --tensors tensor1,tensor2,...");
            std::process::exit(1);
        }

        match storage::surgery::TensorSurgeon::from_iara(&iara_path) {
            Ok(mut surgeon) => {
                surgeon.prune(&tensors_to_prune);
                let report = surgeon.report();
                println!("{}", report.display());

                match surgeon.export(std::path::Path::new(&output)) {
                    Ok(size) => {
                        println!("\n✅ Pruned model exported: {} ({:.1} MB)", output, size as f64 / 1e6);
                    }
                    Err(e) => {
                        eprintln!("❌ Export failed: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Failed to open {}: {}", iara_path.display(), e);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    // 1. Hardware init
    let hw = HardwareManager::new()?;
    log::info!("Hardware: {}", hw.gpu_name);

    // 2. Models directory (default: workspace_root/models)
    let models_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
    });

    if !models_dir.exists() {
        log::warn!("⚠️  models_dir não existe: {:?}", models_dir);
    } else {
        log::info!("📂 models_dir: {:?}", models_dir.canonicalize().unwrap_or(models_dir.clone()));
    }

    // 2b. Opções de carregamento de modelo (variáveis de ambiente)
    let quantize_on_load = std::env::var("IARA_QUANTIZE")
        .map(|v| v == "1" || v == "q4" || v == "true")
        .unwrap_or(false);
    let enable_cfc = std::env::var("IARA_CFC")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);
    let enable_speculative = std::env::var("IARA_SPECULATIVE")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);

    if quantize_on_load {
        log::info!("🔧 Quantização Q4 habilitada ao carregar modelos (IARA_QUANTIZE=1)");
    }
    if enable_cfc {
        log::info!("🔧 Cascade Feedback Control (CfC) habilitado (IARA_CFC=1)");
    }
    if enable_speculative {
        log::info!("🔧 Speculative decoding habilitado (IARA_SPECULATIVE=1)");
    }

    // 3. Shared state — no model loaded at startup
    let qwen_model:    Arc<Mutex<Option<CausalLM>>> = Arc::new(Mutex::new(None));
    let draft_model:   Arc<Mutex<Option<CausalLM>>> = Arc::new(Mutex::new(None));
    let model_ready    = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let model_loading  = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let last_trace     = Arc::new(Mutex::new(None));

    // 4. WS channels
    let (frame_tx, _) = broadcast::channel::<String>(512);
    let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>(64);

    // 5. Start WebSocket server (port 8080)
    let ws_addr: std::net::SocketAddr = "127.0.0.1:8080".parse()?;
    let frame_tx_ws = frame_tx.clone();
    tokio::spawn(async move {
        start_ws_server(ws_addr, cmd_tx, frame_tx_ws.subscribe()).await;
    });
    log::info!("WebSocket em ws://127.0.0.1:8080");

    // 6. Engine command processor (bridges WS commands -> model)
    let engine_model    = qwen_model.clone();
    let engine_frame_tx = frame_tx.clone();
    tokio::spawn(run_engine_loop(engine_model, cmd_rx, engine_frame_tx));

    // Initialize sysinfo and create telemetry router
    // Usamos new_with_specifics() em vez de new_all() para evitar varreduras lentas (ou deadlocks) em discos (mounts fuse) e rede
    let mut sys = System::new_with_specifics(
        sysinfo::RefreshKind::new()
            .with_memory(sysinfo::MemoryRefreshKind::everything())
            .with_cpu(sysinfo::CpuRefreshKind::everything())
    );
    sys.refresh_memory();
    sys.refresh_cpu(); // Initial refresh
    let shared_sys_info = Arc::new(Mutex::new(sys));
    let telemetry_routes = telemetry::telemetry_router();

    // Cria armazenador de runs ACP
    let acp_store = acp::create_run_store();

    // 7. Axum HTTP server (port 3000) — starts immediately, no model loaded yet
    let app_state = AppState {
        qwen_model:    qwen_model.clone(),
        draft_model:   draft_model.clone(),
        last_trace:    last_trace.clone(),
        model_ready:   model_ready.clone(),
        model_loading: model_loading.clone(),
        models_dir:    Arc::new(models_dir),
        frame_tx:      frame_tx.clone(),
        backend:       hw.backend.clone(),
        shared_sys_info: shared_sys_info.clone(),
        acp_store:     Arc::new(acp_store),
        quantize_on_load,
        enable_cfc,
        enable_speculative,
    };
    let frontend_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../frontend");
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .merge(telemetry_routes) // Merge telemetry routes (inclui /health)
        .route("/api/models",             get(models_handler))
        .route("/api/load-model",         post(load_model_handler))
        .route("/api/unload-model",       post(unload_model_handler))
        .route("/api/generate",           post(generate_handler))
        .route("/api/trace",              post(deep_trace_handler))
        .route("/api/train-lnn",          post(train_lnn_handler))
        .route("/api/lnn/status",         get(lnn_status_handler))
        .route("/api/debug/last-trace",   get(debug_trace_handler))
        .nest("/acp/v1", acp::acp_router())
        .fallback_service(ServeDir::new(&frontend_dir))
        .layer(cors)
        .with_state(app_state);

    let http_addr: std::net::SocketAddr = std::env::var("IARA_HTTP_PORT")
        .ok()
        .and_then(|p| format!("0.0.0.0:{}", p).parse().ok())
        .unwrap_or_else(|| "0.0.0.0:3000".parse().unwrap());
    log::info!("🌐 HTTP em http://localhost:3000");
    log::info!("📦 Nenhum modelo carregado. Selecione um no painel do frontend.");

    let listener = tokio::net::TcpListener::bind(http_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Engine loop: receives EngineCommand from WS, calls forward_traced, broadcasts results.
async fn run_engine_loop(
    model: Arc<Mutex<Option<CausalLM>>>,
    mut cmd_rx: mpsc::Receiver<EngineCommand>,
    frame_tx: broadcast::Sender<String>,
) {
    #[cfg(feature = "experimental")]
    use crate::experimental::orchestrator::PulseOrchestrator;

    const MAX_STEPS: usize = 256;
    const ESN_SEED: u64 = 0xDEAD_BEEF_1337_4242;

    // ── LNN Trainer: persiste entre gerações ─────────────────────────
    // Inicializado lazy na primeira geração com CfC habilitado.
    let mut lnn_trainer: Option<LnnTrainer> = None;

    // ── Experimental Orchestrator ──────────────────────────────────────
    // Created lazily when the model is loaded (num_layers known).
    #[cfg(feature = "experimental")]
    let mut orchestrator: Option<PulseOrchestrator> = None;

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            EngineCommand::Resonate(text) | EngineCommand::Stimulus(text) => {
                log::info!("Engine: '{}'", &text[..text.len().min(50)]);

                // Check model is ready
                {
                    let mut guard = model.lock().await;
                    if guard.is_none() {
                        let _ = frame_tx.send(r#"{"type":"status","message":"Nenhum modelo carregado. Selecione um no painel."}"#.to_string());
                        continue;
                    }

                    // Initialize orchestrator lazily on first generation
                    #[cfg(feature = "experimental")]
                    {
                        if orchestrator.is_none() {
                            if let Some(m) = guard.as_ref() {
                                let num_layers = m.cfg.num_layers;
                                log::info!("Initializing PulseOrchestrator with {} layers", num_layers);
                                let (stim_tx, stim_rx) = tokio::sync::mpsc::channel(256);
                                orchestrator = Some(PulseOrchestrator::new(num_layers, stim_rx));

                                // Attach stimulus_tx to model for this and subsequent generations
                                if let Some(m) = guard.as_mut() {
                                    m.stimulus_tx = Some(stim_tx);
                                }
                            }
                        }
                    }
                }

                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("Processando: '{}'", &text[..text.len().min(30)])
                }).to_string());

                // Formato Q/A para modelo BASE: "Q: {pergunta}\nA:" para perguntas,
                // passthrough para completions diretas (ex: "1+1=").
                let tokens = crate::tokenizer::tokenize_qa(&text);
                let prompt_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();

                if prompt_ids.is_empty() { continue; }

                // Reset KV cache before generation
                {
                    let mut guard = model.lock().await;
                    if let Some(m) = guard.as_mut() { m.reset_cache(); }
                }

                let mut generated_text = String::new();
                let mut generated_token_ids: Vec<u32> = Vec::new();
                let mut sampling = SamplingParams::default(); // temperature=0.7, top_k=40, top_p=0.9

                // Layer budget tracking: [Fast, Medium, Slow]
                let mut budget_counts = [0u32; 3];

                // Extract EOS token IDs from model architecture
                let eos_ids = {
                    let guard = model.lock().await;
                    match guard.as_ref() {
                        Some(m) => m.arch.eos_token_ids.clone(),
                        None => continue,
                    }
                };

                // ── ESN Fase 1: observador do stream de tokens ────────────
                let mut esn = EsnObserver::new(ESN_SEED);
                esn.reset();

                // Prefill em batch: tokens 0..n-2 em um único forward pass para popular KV cache.
                let mut prefill_ok = true;
                if prompt_ids.len() > 1 {
                    let mut guard = model.lock().await;
                    if let Some(m) = guard.as_mut() {
                        if let Err(e) = m.forward(&prompt_ids[..prompt_ids.len() - 1], None) {
                            log::error!("Prefill error: {}", e);
                            let _ = frame_tx.send(serde_json::json!({
                                "type": "status",
                                "message": format!("❌ Erro no prefill: {}", e)
                            }).to_string());
                            prefill_ok = false;
                        }
                    }
                }
                if !prefill_ok { continue; }

                // Decode: começa com o último token do prompt, gera até MAX_STEPS tokens novos.
                let mut decode_token = match prompt_ids.last() {
                    Some(&t) => t,
                    None => continue,
                };

                // Pre-allocate sampling buffer for zero-alloc decode
                let vocab_size = {
                    let guard = model.lock().await;
                    guard.as_ref().map(|m| m.cfg.vocab_size).unwrap_or(152_000)
                };
                let mut sampling_buf = SamplingBuffer::with_vocab_size(vocab_size);

                for step in 0..MAX_STEPS {
                    let (gen_step, layer_budget) = {
                        let mut guard = model.lock().await;
                        let m = match guard.as_mut() {
                            Some(m) => m,
                            None => break,
                        };
                        let budget = esn.layer_budget();
                        match forward_traced_buffered(m, &[decode_token], &sampling, Some(budget), &mut sampling_buf) {
                            Ok(s) => (s, budget),
                            Err(e) => {
                                log::error!("Forward error step {}: {}", step, e);
                                let _ = frame_tx.send(serde_json::json!({
                                    "type": "status",
                                    "message": format!("❌ Erro no forward pass (step {}): {}", step, e)
                                }).to_string());
                                break;
                            }
                        }
                    };

                    // Track layer budget usage
                    match layer_budget {
                        crate::lnn::esn::LayerBudget::Fast => budget_counts[0] += 1,
                        crate::lnn::esn::LayerBudget::Medium => budget_counts[1] += 1,
                        crate::lnn::esn::LayerBudget::Slow => budget_counts[2] += 1,
                    }

                    decode_token = gen_step.token_id;

                    // ── Orchestrator tick (experimental) ────────────────────
                    #[cfg(feature = "experimental")]
                    {
                        // Tick the liquid network physics and homeostasis.
                        if let Some(ref mut orch) = orchestrator {
                            orch.tick();
                            let orch_state = orch.state();
                            let _ = frame_tx.send(serde_json::json!({
                                "type": "orchestrator_state",
                                "step": step,
                                "is_sleeping": orch_state.is_sleeping,
                                "dominant_mode": orch_state.dominant_mode.to_string(),
                                "global_coherence": orch_state.global_coherence,
                                "stress": orch_state.stress,
                            }).to_string());
                        }
                    }

                    // Não acumula tokens especiais/EOS no texto visível
                    if eos_ids.contains(&decode_token) {
                        let _ = frame_tx.send(serde_json::json!({
                            "type": "status",
                            "message": format!("Geração completa: {} tokens", generated_text.split_whitespace().count())
                        }).to_string());
                        break;
                    }

                    generated_token_ids.push(decode_token);

                    // ── ESN: observa logits e modula sampling ─────────────
                    // top_logits são os 40 maiores scores (pre-softmax)
                    // A ESN computa entropia e top1_prob a partir deles (aproximação)
                    let top_scores: Vec<f32> = gen_step.top_logits.iter().map(|(_, s)| *s).collect();
                    esn.step(&top_scores, step, MAX_STEPS);
                    sampling = esn.modulate(&sampling);
                    let esn_tel = esn.telemetry();

                    // Broadcast telemetria ESN
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "esn_state",
                        "step": step,
                        "entropy": esn_tel.entropy,
                        "top1_prob": esn_tel.top1_prob,
                        "state_norm": esn_tel.state_norm,
                        "temp_delta": esn_tel.temp_delta,
                        "temperature": sampling.temperature,
                        "readout_trained": esn_tel.readout_trained,
                        "budget_score": esn_tel.budget_score,
                        "layer_budget": format!("{:?}", layer_budget),
                    }).to_string());

                    // ── Gravity + CfC + Q4 telemetry (Fase 2) ────────────
                    {
                        let guard = model.lock().await;
                        if let Some(m) = guard.as_ref() {
                            // Gravity telemetry com mean/max da última observação
                            if let Some(ref grav) = m.gravity {
                                let hist = &grav.history;
                                let (mean_g, max_g) = if let Some(&last) = hist.last() {
                                    (last, hist.iter().cloned().fold(0f32, f32::max))
                                } else {
                                    (0.0, 0.0)
                                };
                                let _ = frame_tx.send(serde_json::json!({
                                    "type": "gravity_state",
                                    "step": step,
                                    "enabled": true,
                                    "history_len": grav.history_len(),
                                    "threshold": grav.current_threshold(),
                                    "mean_gravity": mean_g,
                                    "max_gravity": max_g,
                                }).to_string());
                            }

                            // CfC telemetry com state_norm e correction_norm
                            if let Some(ref cfc) = m.cfc {
                                // Download state h e correction para calcular normas
                                let h_data = m.backend.download_f32(&cfc.state.h);
                                let state_norm = h_data.iter()
                                    .map(|x| x * x).sum::<f32>().sqrt();
                                let corr_data = m.backend.download_f32(&cfc.correction_buf);
                                let correction_norm = corr_data.iter()
                                    .map(|x| x * x).sum::<f32>().sqrt();

                                // Record CfC activations para treinamento
                                // (f_gate e g_candidate extraídos dos buffers internos)
                                if let Some(ref mut trainer) = lnn_trainer {
                                    let f_data = m.backend.download_f32(&cfc.f_proj_buf);
                                    let g_data = m.backend.download_f32(&cfc.g_proj_buf);
                                    // f_proj → sigmoid = f_gate
                                    let f_gate: Vec<f32> = f_data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
                                    // g_proj → tanh = g_candidate
                                    let g_cand: Vec<f32> = g_data.iter().map(|x| x.tanh()).collect();
                                    // x (hidden state que entrou no CfC) — usar h_prev como proxy
                                    // h_prev foi o estado antes do forward
                                    let reward_est = esn_tel.top1_prob; // reward instantâneo
                                    trainer.record_cfc_activation(
                                        corr_data.clone(),  // x placeholder (will be normed hidden)
                                        h_data.clone(),     // h_new (post-update)
                                        h_data.clone(),     // h_new
                                        corr_data.clone(),  // correction
                                        f_gate,
                                        g_cand,
                                        reward_est,
                                    );
                                }

                                let _ = frame_tx.send(serde_json::json!({
                                    "type": "cfc_state",
                                    "step": step,
                                    "enabled": true,
                                    "cfc_dim": cfc.config.cfc_dim,
                                    "param_count": cfc.param_count(),
                                    "vram_mb": cfc.vram_bytes() as f64 / (1024.0 * 1024.0),
                                    "state_norm": state_norm,
                                    "correction_norm": correction_norm,
                                }).to_string());
                            }

                            // Q4 status
                            if m.quantized {
                                let _ = frame_tx.send(serde_json::json!({
                                    "type": "q4_status",
                                    "step": step,
                                    "active": true,
                                    "quantized": true,
                                    "vram_mb": m.vram_bytes as f64 / (1024.0 * 1024.0),
                                }).to_string());
                            }
                        }
                    }

                    for trace in &gen_step.layer_traces {
                        let msg = serde_json::json!({
                            "type": "pipeline_layer",
                            "layer_idx": trace.layer_idx,
                            "total_layers": gen_step.layer_traces.len(),
                            "phase": format!("Layer {}/{}", trace.layer_idx + 1, gen_step.layer_traces.len()),
                            "attention_energy": trace.attention_energy,
                            "mlp_energy": trace.mlp_energy,
                            "residual_norm": trace.residual_norm,
                            "max_activation": trace.max_activation,
                            "hot_neurons": trace.hot_neurons,
                            "time_us": trace.time_us,
                            "spikes": trace.hot_neurons.len()
                        });
                        let _ = frame_tx.send(msg.to_string());
                    }

                    let top_tokens: Vec<_> = gen_step.top_logits.iter().take(40).map(|(id, score)| {
                        serde_json::json!({
                            "token_id": id,
                            "text": crate::tokenizer::token_id_to_text(*id),
                            "score": score
                        })
                    }).collect();

                    generated_text.push_str(&gen_step.token_str);

                    let result = serde_json::json!({
                        "type": "pipeline_result",
                        "step": step,
                        "total_time_ms": gen_step.time_ms,
                        "layers_processed": gen_step.layer_traces.len(),
                        "prompt": text,
                        "generated_so_far": generated_text,
                        "top_tokens": top_tokens,
                        "token_str": gen_step.token_str,
                        "active_layers": gen_step.active_layers,
                    });
                    let _ = frame_tx.send(result.to_string());

                    if eos_ids.contains(&decode_token) { break; }
                }

                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("Geração completa: {} tokens", generated_text.split_whitespace().count())
                }).to_string());

                // Log layer budget statistics
                log::info!(
                    "Layer budget stats: Fast={}, Medium={}, Slow={}",
                    budget_counts[0], budget_counts[1], budget_counts[2]
                );
                let _ = frame_tx.send(serde_json::json!({
                    "type": "budget_stats",
                    "fast": budget_counts[0],
                    "medium": budget_counts[1],
                    "slow": budget_counts[2],
                }).to_string());

                // ── LNN Training: treina após cada geração ───────────────
                // Inicializa trainer lazy se necessário
                {
                    let mut guard = model.lock().await;
                    if let Some(m) = guard.as_mut() {
                        // Inicializa trainer na primeira geração
                        if lnn_trainer.is_none() {
                            let cfc_cfg = m.cfc.as_ref().map(|c| &c.config);
                            lnn_trainer = Some(LnnTrainer::new(cfc_cfg));
                            log::info!("LNN Trainer inicializado (CfC: {})", cfc_cfg.is_some());
                        }

                        if let Some(ref mut trainer) = lnn_trainer {
                            let train_result = trainer.train_after_generation(
                                &mut esn,
                                m.cfc.as_mut(),
                                &m.backend,
                                &generated_token_ids,
                            );

                            // Broadcast training telemetry
                            let _ = frame_tx.send(serde_json::json!({
                                "type": "lnn_train",
                                "generation": train_result.generation,
                                "reward": {
                                    "mean": train_result.reward.mean_reward,
                                    "entropy": train_result.reward.mean_entropy,
                                    "diversity": train_result.reward.lexical_diversity,
                                    "repetition": train_result.reward.repetition_score,
                                },
                                "esn_trained": train_result.esn_trained,
                                "esn_readout_active": esn.telemetry().readout_trained,
                                "cfc_trained": train_result.cfc_trained.as_ref().map(|r| serde_json::json!({
                                    "epoch": r.epoch,
                                    "loss": r.loss,
                                    "grad_norm": r.grad_norm_post_clip,
                                    "lr": r.lr,
                                })),
                            }).to_string());

                            // Broadcast training telemetry geral
                            let tel = trainer.telemetry();
                            let _ = frame_tx.send(serde_json::json!({
                                "type": "lnn_train_status",
                                "enabled": tel.enabled,
                                "generations": tel.generation_count,
                                "mean_reward": tel.mean_reward,
                                "recent_reward": tel.recent_reward,
                                "reward_trend": tel.reward_trend,
                                "esn_auto": tel.esn_auto,
                                "cfc_auto": tel.cfc_auto,
                                "cfc": tel.cfc_telemetry,
                            }).to_string());

                            log::info!(
                                "LNN Training completo: gen #{}, reward={:.4}, ESN trained={}, CfC trained={}",
                                train_result.generation,
                                train_result.reward.mean_reward,
                                train_result.esn_trained,
                                train_result.cfc_trained.is_some()
                            );
                        }
                    }
                }
            }

            EngineCommand::ListLayers => {
                // Model may not be loaded; return empty if so
                let guard = model.lock().await;
                let num_layers = guard.as_ref().map(|m| m.cfg.num_layers).unwrap_or(0);
                drop(guard);

                let names: Vec<String> = (0..num_layers).flat_map(|i: usize| {
                    vec![
                        format!("layers.{}.attention.wq", i),
                        format!("layers.{}.attention.wk", i),
                        format!("layers.{}.attention.wv", i),
                        format!("layers.{}.attention.wo", i),
                        format!("layers.{}.mlp.w_gate", i),
                        format!("layers.{}.mlp.w_up", i),
                        format!("layers.{}.mlp.w_down", i),
                    ]
                }).collect();
                let _ = frame_tx.send(serde_json::json!({ "type": "layers", "names": names }).to_string());
            }

            EngineCommand::SelectLayer(name) => {
                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("Layer selecionada: {}", name)
                }).to_string());
            }

            EngineCommand::Reset => {
                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": "Reset completo"
                }).to_string());
            }
        }
    }
}
