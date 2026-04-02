// ============================================================================
// api/handlers.rs — HTTP Handlers for IARA API
// ============================================================================
// GET  /api/models            — lista modelos disponíveis em models_dir
// POST /api/load-model        — carrega um modelo selecionado
// POST /api/generate          — geração (não-streaming)
// GET  /api/debug/last-trace  — último brain trace
// GET  /health                — health check
// ============================================================================

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::debug::brain::BrainTrace;
use crate::model::{QwenModel, ModelCfg};
use crate::inference::forward_traced;
use crate::inference::SamplingParams;
use compute_core::ComputeBackend;

// ─────────────────────────────────────────────────────────────────────────────
// App State
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub qwen_model:    Arc<Mutex<Option<QwenModel>>>,
    pub last_trace:    Arc<Mutex<Option<BrainTrace>>>,
    pub model_ready:   Arc<std::sync::atomic::AtomicBool>,
    pub model_loading: Arc<std::sync::atomic::AtomicBool>,
    pub models_dir:    Arc<PathBuf>,
    pub frame_tx:      tokio::sync::broadcast::Sender<String>,
    /// Backend de compute ativo — passado para QwenModel::load() ao carregar.
    pub backend:       Arc<dyn ComputeBackend>,
    pub shared_sys_info: crate::telemetry::SharedSystemInfo,
    /// ACP Run Store — armazena estado das execuções do protocolo ACP
    pub acp_store:     Arc<crate::acp::state::AcpRunStore>,
    /// Habilita quantização Q4 ao carregar modelos (env: IARA_QUANTIZE)
    pub quantize_on_load: bool,
    /// Habilita Cascade Feedback Control (CfC) para modularização de camadas
    pub enable_cfc: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Request / Response types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

#[derive(Debug, Serialize, Clone)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<u32>>,
    pub total_duration: u128,
    pub prompt_eval_count: u32,
    pub eval_count: u32,
    pub eval_duration: u128,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub hidden_size: Option<u64>,
    pub num_layers: Option<u64>,
    pub vocab_size: Option<u64>,
    pub num_heads: Option<u64>,
    pub num_kv_heads: Option<u64>,
    pub head_dim: Option<u64>,
    pub intermediate_size: Option<u64>,
    pub architecture: Option<String>,
    pub torch_dtype: Option<String>,
    pub model_kind: String,   // "causal_lm" | "vlm" | "unknown"
    pub has_vision: bool,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub version: String,
    pub timestamp: String,
    pub model_loading: bool,
    /// Pesos do modelo carregados em VRAM (GB, F32). 0.0 se nenhum modelo ativo.
    pub model_vram_gb: f64,
    /// VRAM disponível no momento da consulta (GB). None se HIP indisponível.
    pub vram_free_gb: Option<f64>,
    /// VRAM total da GPU (GB). None se HIP indisponível.
    pub vram_total_gb: Option<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub async fn health_handler(
    State(state): State<AppState>,
) -> Json<HealthResponse> {
    let ready   = state.model_ready.load(std::sync::atomic::Ordering::Relaxed);
    let loading = state.model_loading.load(std::sync::atomic::Ordering::Relaxed);
    let status  = if ready { "ok" } else if loading { "loading" } else { "idle" };

    let (model_name, model_vram_gb) = {
        let guard = state.qwen_model.lock().await;
        match guard.as_ref() {
            Some(m) => (
                format!("{}L-{}h", m.cfg.num_layers, m.cfg.hidden_size),
                m.vram_bytes as f64 / 1e9,
            ),
            None => ("none".to_string(), 0.0),
        }
    };

    // VRAM livre/total via backend (retorna None se não disponível)
    let (vram_free_gb, vram_total_gb) = match state.backend.mem_info() {
        Some((free, total)) => (Some(free as f64 / 1e9), Some(total as f64 / 1e9)),
        None => (None, None),
    };

    Json(HealthResponse {
        status: status.to_string(),
        model: model_name,
        version: "2.0.0".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        model_loading: loading,
        model_vram_gb,
        vram_free_gb,
        vram_total_gb,
    })
}

/// GET /api/models — scan models_dir and return info for each
pub async fn models_handler(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let models_dir = state.models_dir.as_ref();
    let mut models: Vec<ModelInfo> = Vec::new();

    let entries = match std::fs::read_dir(models_dir) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Não foi possível ler models_dir: {}", e) })),
            ).into_response();
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() { continue; }

        let config_path = path.join("config.json");
        if !config_path.exists() { continue; }

        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let info = match std::fs::read_to_string(&config_path) {
            Ok(content) => {
                let v: serde_json::Value = serde_json::from_str(&content).unwrap_or_default();

                // Params may be at root (causal LM) or nested in text_config (VLM)
                let p = if v["hidden_size"].is_number() { &v }
                        else if v["text_config"]["hidden_size"].is_number() { &v["text_config"] }
                        else { &v };

                let uint_field = |key: &str| -> Option<u64> {
                    p[key].as_u64().or_else(|| p[key].as_f64().map(|f| f as u64))
                };

                let arch = v["architectures"].as_array()
                    .and_then(|a| a.first())
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string());

                let model_kind = match arch.as_deref().unwrap_or("") {
                    a if a.ends_with("ForCausalLM") => "causal_lm".to_string(),
                    a if a.ends_with("ForConditionalGeneration") => "vlm".to_string(),
                    _ => "unknown".to_string(),
                };
                let has_vision = v["vision_config"].is_object();

                let display_arch = arch.as_deref().map(|a| {
                    if model_kind == "causal_lm" { a.to_string() }
                    else if model_kind == "vlm" { format!("{} [VLM — suporte parcial]", a) }
                    else { format!("{} ⚠ não suportado", a) }
                });

                let dtype = v["torch_dtype"].as_str()
                    .or_else(|| p["dtype"].as_str())
                    .map(|s| s.to_string());

                ModelInfo {
                    name: name.clone(),
                    path: path.to_string_lossy().to_string(),
                    hidden_size:       uint_field("hidden_size"),
                    num_layers:        uint_field("num_hidden_layers"),
                    vocab_size:        uint_field("vocab_size"),
                    num_heads:         uint_field("num_attention_heads"),
                    num_kv_heads:      uint_field("num_key_value_heads"),
                    head_dim:          uint_field("head_dim"),
                    intermediate_size: uint_field("intermediate_size"),
                    architecture:      display_arch,
                    torch_dtype:       dtype,
                    model_kind,
                    has_vision,
                }
            }
            Err(_) => ModelInfo {
                name: name.clone(),
                path: path.to_string_lossy().to_string(),
                hidden_size: None, num_layers: None, vocab_size: None,
                num_heads: None, num_kv_heads: None, head_dim: None,
                intermediate_size: None, architecture: None, torch_dtype: None,
                model_kind: "unknown".to_string(),
                has_vision: false,
            },
        };
        models.push(info);
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    (StatusCode::OK, Json(serde_json::json!({ "models": models }))).into_response()
}

/// POST /api/load-model — carrega o modelo selecionado em background
pub async fn load_model_handler(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> impl IntoResponse {
    // Prevent double-loading
    if state.model_loading.load(std::sync::atomic::Ordering::Relaxed) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "Já existe um modelo sendo carregado" })),
        ).into_response();
    }

    let model_path = state.models_dir.join(&req.name);
    if !model_path.join("config.json").exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("Modelo '{}' não encontrado", req.name) })),
        ).into_response();
    }

    // Detect VLM before spawning load
    let arch_check = std::fs::read_to_string(model_path.join("config.json"))
        .ok()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
        .and_then(|v| v["architectures"].as_array()
            .and_then(|a| a.first())
            .and_then(|s| s.as_str())
            .map(|s| s.to_string()));

    let is_vlm = arch_check.as_deref()
        .map(|a| a.ends_with("ForConditionalGeneration"))
        .unwrap_or(false);

    if is_vlm {
        log::warn!("⚠️  Modelo VLM '{}' — apenas o decoder de texto será carregado (vision encoder na Fase 2)", req.name);
        let _ = state.frame_tx.send(serde_json::json!({
            "type": "status",
            "message": format!("⚠️ VLM '{}': carregando apenas decoder de texto (vision = Fase 2)", req.name)
        }).to_string());
    }

    // Mark as loading
    state.model_loading.store(true, std::sync::atomic::Ordering::Relaxed);
    state.model_ready.store(false, std::sync::atomic::Ordering::Relaxed);

    // Unload current model immediately
    *state.qwen_model.lock().await = None;

    // Broadcast loading started
    let _ = state.frame_tx.send(serde_json::json!({
        "type": "status",
        "message": format!("Carregando modelo '{}'...", req.name)
    }).to_string());

    // Spawn background load
    let model_arc     = state.qwen_model.clone();
    let model_ready   = state.model_ready.clone();
    let model_loading = state.model_loading.clone();
    let frame_tx      = state.frame_tx.clone();
    let model_name    = req.name.clone();
    let backend       = state.backend.clone();
    let quantize_on_load = state.quantize_on_load;
    let enable_cfc    = state.enable_cfc;

    tokio::spawn(async move {
        let start = std::time::Instant::now();

        // Progress ticker
        let loading_tick = model_loading.clone();
        let tx_tick = frame_tx.clone();
        let name_tick = model_name.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                if !loading_tick.load(std::sync::atomic::Ordering::Relaxed) { break; }
                let elapsed = start.elapsed().as_secs();
                log::info!("⏳ Carregando '{}'... {}s", name_tick, elapsed);
                let _ = tx_tick.send(serde_json::json!({
                    "type": "status",
                    "message": format!("Carregando '{}'... {}s decorridos", name_tick, elapsed)
                }).to_string());
            }
        });

        // Load tokenizer
        log::info!("🔤 Carregando tokenizer de '{}'...", model_name);
        if let Err(e) = crate::tokenizer::load_tokenizer(&model_path) {
            log::error!("❌ Falha no tokenizer: {}", e);
            let _ = frame_tx.send(serde_json::json!({
                "type": "status",
                "message": format!("ERRO tokenizer '{}': {}", model_name, e)
            }).to_string());
            model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        // Load model weights
        log::info!("⏳ Carregando pesos de '{}'...", model_name);
        let cfg = match ModelCfg::from_config_json(&model_path.join("config.json")) {
            Ok(c) => c,
            Err(e) => {
                log::error!("❌ Falha ao ler config.json: {}", e);
                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("ERRO config '{}': {}", model_name, e)
                }).to_string());
                model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        };

        // Tenta carregar com quantização se habilitado
        let load_result = if quantize_on_load {
            log::info!("🔧 Carregando com quantização Q4 (IARA_QUANTIZE=1)");
            let _ = frame_tx.send(serde_json::json!({
                "type": "status",
                "message": format!("Carregando '{}' com quantização Q4...", model_name)
            }).to_string());
            QwenModel::load_quantized(&model_path, cfg, backend, enable_cfc)
        } else {
            QwenModel::load(&model_path, cfg, backend)
        };

        match load_result {
            Ok(model) => {
                *model_arc.lock().await = Some(model);
                model_ready.store(true, std::sync::atomic::Ordering::Relaxed);
                model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                log::info!("✅ Modelo '{}' carregado em {:.1}s", model_name, start.elapsed().as_secs_f32());
                let backend_name = {
                    let g = model_arc.lock().await;
                    g.as_ref().map(|m| m.backend.name().to_string()).unwrap_or_default()
                };
                let quant_str = if quantize_on_load { "q4" } else { "bf16" };
                let _ = frame_tx.send(serde_json::json!({
                    "type": "model_loaded",
                    "name": model_name,
                    "message": format!("✅ '{}' pronto! ({:.1}s)", model_name, start.elapsed().as_secs_f32()),
                    "backend": backend_name,
                    "quantization": quant_str,
                }).to_string());
            }
            Err(e) => {
                log::error!("❌ Falha ao carregar '{}': {}", model_name, e);
                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("ERRO ao carregar '{}': {}", model_name, e)
                }).to_string());
                model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });

    (StatusCode::ACCEPTED, Json(serde_json::json!({
        "status": "loading",
        "model": req.name
    }))).into_response()
}

pub async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    log::info!("📝 Generate: prompt={}, stream={}",
        &req.prompt[..req.prompt.len().min(50)],
        req.stream);

    let start = std::time::Instant::now();

    match generate_internal(&state, &req).await {
        Ok(response) => {
            log::info!("✅ Geração: {:.2}ms", start.elapsed().as_secs_f32() * 1000.0);
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            log::error!("❌ Erro na geração: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response()
        }
    }
}

pub async fn debug_trace_handler(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let trace = state.last_trace.lock().await.clone();
    match trace {
        Some(t) => (StatusCode::OK, Json(t.analyze())).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "No trace available yet" }))).into_response(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal
// ─────────────────────────────────────────────────────────────────────────────

async fn generate_internal(
    state: &AppState,
    req: &GenerateRequest,
) -> Result<GenerateResponse, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();

    if !state.model_ready.load(std::sync::atomic::Ordering::Relaxed) {
        if state.model_loading.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("Modelo ainda carregando, aguarde...".into());
        }
        return Err("Nenhum modelo carregado. Selecione um modelo no painel.".into());
    }

    let sampling = SamplingParams {
        temperature: req.temperature,
        top_k: req.top_k as usize,
        top_p: req.top_p,
    };

    // Formato Q/A para modelo BASE: wrapa perguntas em "Q: ...\nA:".
    let tokens = crate::tokenizer::tokenize_qa(&req.prompt);
    let prompt_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    let prompt_eval_count = prompt_ids.len() as u32;
    let mut generated_text = String::new();

    let mut brain_trace = BrainTrace::new(req.prompt.clone());
    let mut eval_count = 0u32;
    let mut last_token: u32;

    const EOS_TOKEN: u32 = 151645;

    // Reset KV cache before generation — prevents stale state from previous calls.
    {
        let mut guard = state.qwen_model.lock().await;
        let model = guard.as_mut().ok_or("Model not loaded")?;
        model.reset_cache();
    }

    // Prefill sequencial: processa tokens 0..n-2 apenas para popular o KV cache.
    // Isso garante atenção causal (cada token só vê os anteriores), como no treinamento.
    // O último token do prompt é processado pela primeira iteração do decode loop,
    // que retorna os logits para amostrar o primeiro token gerado.
    if prompt_ids.len() > 1 {
        let mut model_guard = state.qwen_model.lock().await;
        let model = model_guard.as_mut().ok_or("Model not loaded")?;
        for &tok in &prompt_ids[..prompt_ids.len() - 1] {
            let _ = model.forward(&[tok])
                .map_err(|e| format!("Prefill error: {}", e))?;
            // Discard logits and traces from prefill
        }
    }

    // Decode: a primeira iteração processa o último token do prompt → logits para 1º token gerado.
    // Iterações seguintes processam cada token gerado → logits para o próximo.
    let mut decode_token = *prompt_ids.last().ok_or("Prompt vazio")?;

    for step in 0..req.max_tokens {
        let gen_step = {
            let mut model_guard = state.qwen_model.lock().await;
            let model = model_guard.as_mut().ok_or("Model not loaded")?;
            forward_traced(model, &[decode_token], &sampling)
                .map_err(|e| format!("Forward pass error: {}", e))?
        };

        decode_token = gen_step.token_id;
        last_token   = decode_token;
        generated_text.push_str(&gen_step.token_str);
        brain_trace.add_step(gen_step);
        eval_count += 1;

        if last_token == EOS_TOKEN || last_token == 151643 { break; }

        let _ = step; // evita warning de variável não usada
    }

    brain_trace.finalize();
    *state.last_trace.lock().await = Some(brain_trace);

    let total_duration = start_time.elapsed();
    Ok(GenerateResponse {
        model: req.model.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        response: generated_text,
        done: true,
        context: None,
        total_duration: total_duration.as_millis(),
        prompt_eval_count,
        eval_count,
        eval_duration: total_duration.as_millis(),
    })
}

fn default_temperature() -> f32  { 0.7 }
fn default_top_k() -> u32        { 40 }
fn default_top_p() -> f32        { 0.9 }
fn default_max_tokens() -> usize { 100 }
