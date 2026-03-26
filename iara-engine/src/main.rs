// ============================================================================
// main.rs — IARA Server: HTTP (port 3000) + WebSocket (port 8080)
// ============================================================================

mod actuator;
mod decoder;
mod embedding;
mod generate;
mod homeostasis;
mod hardware;
mod inference;
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
mod api;
mod debug;
mod qwen;

use crate::api::handlers::{AppState, generate_handler, health_handler, debug_trace_handler};
use crate::hardware::HardwareManager;
use crate::model::{QwenModel, ModelCfg};
use crate::websocket::{start_ws_server, EngineCommand};
use crate::qwen::forward_traced;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast, mpsc};
use axum::{routing::{get, post}, Router};
use tower_http::cors::{CorsLayer, Any};
use tower_http::services::ServeDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // 1. Hardware init
    let hw = HardwareManager::new()?;
    log::info!("Hardware: {}", hw.gpu_name);

    // 2. Model path
    let model_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/qwen3-8b")
    });
    log::info!("Modelo: {:?}", model_dir.canonicalize().unwrap_or(model_dir.clone()));

    if !model_dir.join("config.json").exists() {
        return Err(format!("config.json nao encontrado em {:?}", model_dir).into());
    }

    // 3. Load tokenizer (fast, synchronous)
    tokenizer::load_tokenizer(&model_dir)?;
    log::info!("Tokenizer carregado");

    // 4. Shared state — model starts as None, loads in background
    let qwen_model: Arc<Mutex<Option<QwenModel>>> = Arc::new(Mutex::new(None));
    let model_ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let last_trace = Arc::new(Mutex::new(None));

    // 5. WS channels
    let (frame_tx, _) = broadcast::channel::<String>(512);
    let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>(64);

    // 6. Start WebSocket server (port 8080)
    let ws_addr: std::net::SocketAddr = "127.0.0.1:8080".parse()?;
    let frame_tx_ws = frame_tx.clone();
    tokio::spawn(async move {
        start_ws_server(ws_addr, cmd_tx, frame_tx_ws.subscribe()).await;
    });
    log::info!("WebSocket em ws://127.0.0.1:8080");

    // 7. Engine command processor (bridges WS commands -> model)
    let engine_model = qwen_model.clone();
    let engine_frame_tx = frame_tx.clone();
    tokio::spawn(run_engine_loop(engine_model, cmd_rx, engine_frame_tx));

    // 8. Axum HTTP server (port 3000) — starts BEFORE model loads
    let app_state = AppState {
        qwen_model: qwen_model.clone(),
        last_trace: last_trace.clone(),
        model_ready: model_ready.clone(),
    };

    let frontend_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../frontend");
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/debug/last-trace", get(debug_trace_handler))
        .fallback_service(ServeDir::new(&frontend_dir))
        .layer(cors)
        .with_state(app_state);

    let http_addr: std::net::SocketAddr = "0.0.0.0:3000".parse()?;
    log::info!("HTTP em http://localhost:3000 (frontend disponível imediatamente)");

    // 9. Load model in background — server is already running
    let model_arc_bg = qwen_model.clone();
    let model_ready_bg = model_ready.clone();
    let frame_tx_bg = frame_tx.clone();
    let model_dir_bg = model_dir.clone();
    tokio::spawn(async move {
        log::info!("⏳ Carregando Qwen3-8B em background (pode demorar 30-120s dependendo do disco)...");

        // Progress ticker: log elapsed time every 5s while loading
        let model_ready_tick = model_ready_bg.clone();
        tokio::spawn(async move {
            let start = std::time::Instant::now();
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                if model_ready_tick.load(std::sync::atomic::Ordering::Relaxed) { break; }
                log::info!("⏳ Carregando modelo... {}s decorridos", start.elapsed().as_secs());
            }
        });

        match QwenModel::load(&model_dir_bg, ModelCfg::qwen_8b()) {
            Ok(model) => {
                *model_arc_bg.lock().await = Some(model);
                model_ready_bg.store(true, std::sync::atomic::Ordering::Relaxed);
                log::info!("✅ Modelo Qwen3-8B carregado e pronto!");
                let _ = frame_tx_bg.send(r#"{"type":"status","message":"✅ Modelo Qwen3-8B carregado e pronto!"}"#.to_string());
            }
            Err(e) => {
                log::error!("❌ Falha ao carregar modelo: {}", e);
                let msg = serde_json::json!({
                    "type": "status",
                    "message": format!("ERRO ao carregar modelo: {}", e)
                }).to_string();
                let _ = frame_tx_bg.send(msg);
            }
        }
    });

    let listener = tokio::net::TcpListener::bind(http_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Engine loop: receives EngineCommand from WS, calls forward_traced, broadcasts results.
async fn run_engine_loop(
    model: Arc<Mutex<Option<QwenModel>>>,
    mut cmd_rx: mpsc::Receiver<EngineCommand>,
    frame_tx: broadcast::Sender<String>,
) {
    const EOS: u32 = 151645;
    const MAX_STEPS: usize = 256;

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            EngineCommand::Resonate(text) | EngineCommand::Stimulus(text) => {
                log::info!("Engine: '{}'", &text[..text.len().min(50)]);

                // Check model is ready
                {
                    let guard = model.lock().await;
                    if guard.is_none() {
                        let _ = frame_tx.send(r#"{"type":"status","message":"Modelo ainda carregando, aguarde..."}"#.to_string());
                        continue;
                    }
                }

                // Broadcast loading status
                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("Processando: '{}'", &text[..text.len().min(30)])
                }).to_string());

                // Tokenize
                let tokens = crate::tokenizer::tokenize(&text);
                let mut current_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();

                if current_ids.is_empty() {
                    continue;
                }

                let mut generated_text = String::new();

                for step in 0..MAX_STEPS {
                    let gen_step = {
                        let mut guard = model.lock().await;
                        let m = match guard.as_mut() {
                            Some(m) => m,
                            None => break,
                        };
                        match forward_traced(m, &current_ids) {
                            Ok(s) => s,
                            Err(e) => {
                                log::error!("Forward error: {}", e);
                                break;
                            }
                        }
                    };

                    // Broadcast per-layer telemetry
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

                    // Top-40 tokens for result
                    let top_tokens: Vec<_> = gen_step.top_logits.iter().take(40).map(|(id, score)| {
                        serde_json::json!({
                            "token_id": id,
                            "text": crate::tokenizer::token_id_to_text(*id),
                            "score": score
                        })
                    }).collect();

                    generated_text.push_str(&gen_step.token_str);

                    // Broadcast pipeline_result for this step
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

                    let next = gen_step.token_id;
                    current_ids.push(next);

                    if next == EOS || next == 151643 {
                        break;
                    }
                }

                // Final done message
                let done = serde_json::json!({
                    "type": "status",
                    "message": format!("Geracao completa: {} tokens", generated_text.split_whitespace().count())
                });
                let _ = frame_tx.send(done.to_string());
            }

            EngineCommand::ListLayers => {
                let names: Vec<String> = (0..36).flat_map(|i: usize| {
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
                let _ = frame_tx.send(serde_json::json!({
                    "type": "layers",
                    "names": names
                }).to_string());
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
