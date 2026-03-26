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

use crate::api::handlers::{AppState, generate_handler, health_handler, debug_trace_handler,
                            models_handler, load_model_handler};
use crate::hardware::HardwareManager;
use crate::model::QwenModel;
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

    // 2. Models directory (default: workspace_root/models)
    let models_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
    });

    if !models_dir.exists() {
        log::warn!("⚠️  models_dir não existe: {:?}", models_dir);
    } else {
        log::info!("📂 models_dir: {:?}", models_dir.canonicalize().unwrap_or(models_dir.clone()));
    }

    // 3. Shared state — no model loaded at startup
    let qwen_model:    Arc<Mutex<Option<QwenModel>>> = Arc::new(Mutex::new(None));
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

    // 7. Axum HTTP server (port 3000) — starts immediately, no model loaded yet
    let app_state = AppState {
        qwen_model:    qwen_model.clone(),
        last_trace:    last_trace.clone(),
        model_ready:   model_ready.clone(),
        model_loading: model_loading.clone(),
        models_dir:    Arc::new(models_dir),
        frame_tx:      frame_tx.clone(),
    };

    let frontend_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../frontend");
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health",                 get(health_handler))
        .route("/api/models",             get(models_handler))
        .route("/api/load-model",         post(load_model_handler))
        .route("/api/generate",           post(generate_handler))
        .route("/api/debug/last-trace",   get(debug_trace_handler))
        .fallback_service(ServeDir::new(&frontend_dir))
        .layer(cors)
        .with_state(app_state);

    let http_addr: std::net::SocketAddr = "0.0.0.0:3000".parse()?;
    log::info!("🌐 HTTP em http://localhost:3000");
    log::info!("📦 Nenhum modelo carregado. Selecione um no painel do frontend.");

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
                        let _ = frame_tx.send(r#"{"type":"status","message":"Nenhum modelo carregado. Selecione um no painel."}"#.to_string());
                        continue;
                    }
                }

                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("Processando: '{}'", &text[..text.len().min(30)])
                }).to_string());

                let tokens = crate::tokenizer::tokenize(&text);
                let mut current_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();

                if current_ids.is_empty() { continue; }

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
                            Err(e) => { log::error!("Forward error: {}", e); break; }
                        }
                    };

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

                    let next = gen_step.token_id;
                    current_ids.push(next);
                    if next == EOS || next == 151643 { break; }
                }

                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("Geração completa: {} tokens", generated_text.split_whitespace().count())
                }).to_string());
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
