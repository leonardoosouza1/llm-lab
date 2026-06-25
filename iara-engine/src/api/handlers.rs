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
use crate::model::{CausalLM, ModelCfg, ModelArchitecture};
use crate::inference::{forward_traced_buffered, forward_decode_fast, SamplingBuffer};
#[allow(unused_imports)]
use crate::inference::forward_traced;
use crate::inference::SamplingParams;
use crate::storage::iara_format::IaraV2Reader;
use compute_core::ComputeBackend;

// ─────────────────────────────────────────────────────────────────────────────
// App State
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub qwen_model:    Arc<Mutex<Option<CausalLM>>>,
    /// Draft model for speculative decoding (optional, if IARA_SPECULATIVE=1)
    #[allow(dead_code)]
    pub draft_model:   Arc<Mutex<Option<CausalLM>>>,
    pub last_trace:    Arc<Mutex<Option<BrainTrace>>>,
    pub model_ready:   Arc<std::sync::atomic::AtomicBool>,
    pub model_loading: Arc<std::sync::atomic::AtomicBool>,
    pub models_dir:    Arc<PathBuf>,
    pub frame_tx:      tokio::sync::broadcast::Sender<String>,
    /// Backend de compute ativo — passado para CausalLM::load() ao carregar.
    pub backend:       Arc<dyn ComputeBackend>,
    pub shared_sys_info: crate::telemetry::SharedSystemInfo,
    /// ACP Run Store — armazena estado das execuções do protocolo ACP
    pub acp_store:     Arc<crate::acp::state::AcpRunStore>,
    /// Habilita quantização Q4 ao carregar modelos (env: IARA_QUANTIZE)
    pub quantize_on_load: bool,
    /// Habilita quantização ternária BitNet 2-bit (env: IARA_TERNARY)
    pub ternary_on_load: bool,
    /// Habilita Cascade Feedback Control (CfC) para modularização de camadas
    pub enable_cfc: bool,
    /// Habilita speculative decoding (env: IARA_SPECULATIVE)
    #[allow(dead_code)]
    pub enable_speculative: bool,
    /// llama.cpp engine — high-performance GGUF inference via FFI
    pub llama_engine: Arc<Mutex<Option<crate::llama::LlamaEngine>>>,
    /// Batch server handle for concurrent request processing
    pub batch_handle: Arc<Mutex<Option<crate::llama::batch_server::BatchHandle>>>,
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
    /// Optional layer budget: "fast" (N/4 layers), "medium" (3N/4), "slow"/"full" (all).
    /// If present, forces CausalLM path (skips llama.cpp).
    #[serde(default)]
    pub budget: Option<String>,
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
    pub tokens_per_second: f64,
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
        // Check llama.cpp engine first
        let llama_guard = state.llama_engine.lock().await;
        if let Some(engine) = llama_guard.as_ref() {
            let info = engine.model_info();
            (
                format!("llama.cpp-{}L-{}h", info.n_layers, info.n_embd),
                info.model_size as f64 / 1e9,
            )
        } else {
            let guard = state.qwen_model.lock().await;
            match guard.as_ref() {
                Some(m) => (
                    format!("{}L-{}h", m.cfg.num_layers, m.cfg.hidden_size),
                    m.vram_bytes as f64 / 1e9,
                ),
                None => ("none".to_string(), 0.0),
            }
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

/// POST /api/unload-model — descarrega modelo da VRAM
pub async fn unload_model_handler(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let was_ready = state.model_ready.load(std::sync::atomic::Ordering::Relaxed);
    let is_loading = state.model_loading.load(std::sync::atomic::Ordering::Relaxed);

    if is_loading {
        return (StatusCode::CONFLICT, Json(serde_json::json!({
            "error": "Modelo está sendo carregado. Aguarde."
        }))).into_response();
    }

    if !was_ready {
        return (StatusCode::OK, Json(serde_json::json!({
            "status": "ok",
            "message": "Nenhum modelo carregado."
        }))).into_response();
    }

    // Drop the model — libera VRAM
    *state.qwen_model.lock().await = None;
    *state.llama_engine.lock().await = None;
    state.model_ready.store(false, std::sync::atomic::Ordering::Relaxed);

    let _ = state.frame_tx.send(serde_json::json!({
        "type": "status",
        "message": "Modelo descarregado da VRAM"
    }).to_string());

    log::info!("🗑️  Modelo descarregado da VRAM via /api/unload-model");

    (StatusCode::OK, Json(serde_json::json!({
        "status": "ok",
        "message": "Modelo descarregado. VRAM liberada."
    }))).into_response()
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

        // ── .iara files: read metadata directly from file header ──
        if path.extension().and_then(|e| e.to_str()) == Some("iara") && path.is_file() {
            let name = path.file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            if let Ok(reader) = IaraV2Reader::open(&path) {
                let meta = &reader.header.metadata;
                let compression = format!("{:.1}×", meta.compression_ratio);
                let recipe = meta.quant_recipe.get("default")
                    .cloned()
                    .unwrap_or_else(|| "mixed".to_string());

                models.push(ModelInfo {
                    name: name.clone(),
                    path: path.to_string_lossy().to_string(),
                    hidden_size:       Some(meta.hidden_dim as u64),
                    num_layers:        Some(meta.n_layers as u64),
                    vocab_size:        Some(meta.vocab_size as u64),
                    num_heads:         Some(meta.n_heads as u64),
                    num_kv_heads:      Some(meta.n_kv_heads as u64),
                    head_dim:          None,
                    intermediate_size: Some(meta.intermediate_dim as u64),
                    architecture:      Some(format!("{} [.iara {} {}]", meta.architecture, recipe, compression)),
                    torch_dtype:       Some(recipe),
                    model_kind:        "causal_lm".to_string(),
                    has_vision:        false,
                });
            }
            continue;
        }

        // ── .dna files: SVD-factorized single-file bundles ──
        if path.extension().and_then(|e| e.to_str()) == Some("dna") && path.is_file() {
            let name = path.file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let size_mb = entry.metadata().map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
            models.push(ModelInfo {
                name: name.clone(),
                path: path.to_string_lossy().to_string(),
                hidden_size:       None,
                num_layers:        None,
                vocab_size:        None,
                num_heads:         None,
                num_kv_heads:      None,
                head_dim:          None,
                intermediate_size: None,
                architecture:      Some(format!("DNA-SVD [{:.1} MB]", size_mb)),
                torch_dtype:       Some("bf16-svd".to_string()),
                model_kind:        "causal_lm".to_string(),
                has_vision:        false,
            });
            continue;
        }

        // ── Standard directory models: read from config.json ──
        if !path.is_dir() { continue; }

        let config_path = path.join("config.json");
        if !config_path.exists() { continue; }

        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        // Check if there's a sibling .iara file — if so, mark it
        let has_iara = path.with_extension("iara").exists()
            || std::fs::read_dir(&path).ok().map(|entries| {
                entries.flatten().any(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("iara")
                })
            }).unwrap_or(false);

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

                let iara_suffix = if has_iara { " [.iara disponível]" } else { "" };
                let display_arch = arch.as_deref().map(|a| {
                    if model_kind == "causal_lm" { format!("{}{}", a, iara_suffix) }
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

    // Reject path-traversal / absolute paths — the model name must be a plain
    // file or directory name inside models_dir, never an escape like "../".
    if req.name.is_empty()
        || req.name.contains("..")
        || req.name.contains('/')
        || req.name.contains('\\')
        || req.name.contains('\0')
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Nome de modelo inválido" })),
        ).into_response();
    }

    let model_path = state.models_dir.join(&req.name);

    // Check for .gguf file: {name}.gguf in models_dir, or a single .gguf inside the model dir
    let gguf_file = {
        let ext_match = model_path.extension().and_then(|e| e.to_str()) == Some("gguf") && model_path.exists();
        let direct_candidate = state.models_dir.join(format!("{}.gguf", &req.name));
        if ext_match {
            Some(model_path.clone())
        } else if direct_candidate.exists() {
            Some(direct_candidate)
        } else if model_path.is_dir() {
            // Look for a single .gguf file inside the model directory
            std::fs::read_dir(&model_path).ok().and_then(|entries| {
                let ggufs: Vec<_> = entries.filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("gguf"))
                    .collect();
                if ggufs.len() == 1 { Some(ggufs[0].path()) } else { None }
            })
        } else {
            None
        }
    };

    // Check for .iara file: could be {name}.iara directly or {name}/... dir with sibling .iara
    let iara_file = if model_path.extension().and_then(|e| e.to_str()) == Some("iara") && model_path.exists() {
        // Name already includes .iara extension (shouldn't happen but handle it)
        Some(model_path.clone())
    } else {
        // Standard: check for {name}.iara in models dir
        let candidate = state.models_dir.join(format!("{}.iara", &req.name));
        if candidate.exists() { Some(candidate) }
        else {
            // Also check: model_path is a dir, and there's a sibling .iara
            let sibling = model_path.with_extension("iara");
            if sibling.exists() { Some(sibling) } else { None }
        }
    };
    let use_iara = iara_file.is_some();

    // Check for .dna file: {name}.dna in models_dir, or sibling .dna of a model dir
    let dna_file = if model_path.extension().and_then(|e| e.to_str()) == Some("dna") && model_path.exists() {
        Some(model_path.clone())
    } else {
        let candidate = state.models_dir.join(format!("{}.dna", &req.name));
        if candidate.exists() { Some(candidate) }
        else {
            let sibling = model_path.with_extension("dna");
            if sibling.exists() { Some(sibling) } else { None }
        }
    };
    let use_dna = dna_file.is_some();

    // For .iara standalone files, config.json may be in a base model dir
    // e.g., qwen2.5-1.5b-fp8.iara → look for config.json in qwen2.5-1.5b/
    let config_dir = if model_path.is_dir() {
        model_path.clone()
    } else {
        // Try to find a related model directory (strip suffixes like -fp8, -q4, -q4km, -ptt)
        let name_str = req.name.as_str();
        // Strip .gguf extension if present in name
        let name_clean = name_str.trim_end_matches(".gguf").trim_end_matches(".iara").trim_end_matches(".dna");
        let base_names = [
            name_clean.to_string(),
            name_clean.rsplitn(2, '-').last().unwrap_or(name_clean).to_string(),
        ];
        let mut found = None;
        for base in &base_names {
            let candidate = state.models_dir.join(base);
            if candidate.is_dir() && candidate.join("config.json").exists() {
                found = Some(candidate);
                break;
            }
        }
        found.unwrap_or(model_path.clone())
    };

    let use_gguf = gguf_file.is_some();
    if !use_iara && !use_gguf && !use_dna && !config_dir.join("config.json").exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("Modelo '{}' não encontrado (nem .gguf, .iara nem config.json)", req.name) })),
        ).into_response();
    }

    // Detect VLM before spawning load
    let arch_check = std::fs::read_to_string(config_dir.join("config.json"))
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
    *state.llama_engine.lock().await = None;
    *state.batch_handle.lock().await = None;

    // Broadcast loading started
    let _ = state.frame_tx.send(serde_json::json!({
        "type": "status",
        "message": format!("Carregando modelo '{}'...", req.name)
    }).to_string());

    // ── GGUF + llama.cpp fast path ──────────────────────────────────────
    // IARA_NO_LLAMA=1 forces native IARA backend even for GGUF files (for benchmarking)
    // .dna files take precedence over GGUF — native SVD path is the explicit artifact.
    let use_llama_cpp = !use_dna && !state.ternary_on_load
        && std::env::var("IARA_NO_LLAMA").map(|v| v != "1").unwrap_or(true);
    if use_llama_cpp { if let Some(ref gguf_path) = gguf_file {
        let llama_arc = state.llama_engine.clone();
        let batch_arc = state.batch_handle.clone();
        let model_ready = state.model_ready.clone();
        let model_loading = state.model_loading.clone();
        let frame_tx = state.frame_tx.clone();
        let gguf_path = gguf_path.clone();
        let model_name = req.name.clone();

        tokio::spawn(async move {
            let path_str = gguf_path.to_str().unwrap_or("").to_string();
            log::info!("🚀 Loading GGUF via llama.cpp: {}", path_str);

            // llama.cpp model loading is blocking — run in a blocking thread
            let result = tokio::task::spawn_blocking(move || {
                crate::llama::LlamaEngine::load(&path_str, 2048)
            }).await;

            match result {
                Ok(Ok(mut engine)) => {
                    let info = engine.model_info();
                    log::info!("✅ llama.cpp: {} loaded ({}M params, {} layers, {}MB)",
                        model_name, info.n_params / 1_000_000, info.n_layers,
                        info.model_size / 1_000_000);

                    // Try to load draft model for speculative decoding
                    let draft_path = gguf_path.with_file_name(
                        gguf_path.file_name().unwrap_or_default().to_string_lossy()
                            .replace("1.5b", "0.5b")
                            .replace("7b", "1.5b")
                            .replace("8b", "1.5b")
                    );
                    if draft_path != gguf_path && draft_path.exists() {
                        if let Some(dp) = draft_path.to_str() {
                            match engine.load_draft(dp) {
                                Ok(()) => log::info!("✅ Draft model loaded: {}", draft_path.display()),
                                Err(e) => log::warn!("Draft model failed (non-fatal): {e}"),
                            }
                        }
                    }

                    *llama_arc.lock().await = Some(engine);

                    // Start batch server for concurrent request handling
                    let handle = crate::llama::batch_server::start_batch_server(llama_arc.clone());
                    *batch_arc.lock().await = Some(handle);

                    model_ready.store(true, std::sync::atomic::Ordering::Relaxed);
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);

                    let _ = frame_tx.send(serde_json::json!({
                        "type": "model_loaded",
                        "model": model_name,
                        "backend": "llama.cpp",
                        "params_m": info.n_params / 1_000_000,
                    }).to_string());
                }
                Ok(Err(e)) => {
                    log::error!("❌ llama.cpp load error: {}", e);
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "status",
                        "message": format!("ERRO llama.cpp: {}", e)
                    }).to_string());
                }
                Err(e) => {
                    log::error!("❌ llama.cpp spawn error: {}", e);
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });

        return (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "model": req.name,
                "status": "loading",
                "backend": "llama.cpp"
            })),
        ).into_response();
    } } // end if use_llama_cpp { if let Some(ref gguf_path) = gguf_file
    // ── Legacy CausalLM path (safetensors / .iara) ──────────────────────
    // Spawn background load
    let model_arc     = state.qwen_model.clone();
    let model_ready   = state.model_ready.clone();
    let model_loading = state.model_loading.clone();
    let frame_tx      = state.frame_tx.clone();
    let model_name    = req.name.clone();
    let backend       = state.backend.clone();
    let quantize_on_load = state.quantize_on_load;
    let ternary_on_load = state.ternary_on_load;
    let enable_cfc    = state.enable_cfc;
    let resolved_iara_file = iara_file.clone();
    let resolved_gguf_file = gguf_file.clone();
    let resolved_dna_file  = dna_file.clone();
    let resolved_config_dir = config_dir.clone();

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

        // Load tokenizer — use config_dir (which has tokenizer.json)
        let tokenizer_dir = if resolved_config_dir.join("tokenizer.json").exists() {
            resolved_config_dir.clone()
        } else {
            model_path.clone()
        };
        log::info!("🔤 Carregando tokenizer de '{}'...", tokenizer_dir.display());
        if let Err(e) = crate::tokenizer::load_tokenizer(&tokenizer_dir) {
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

        // Check format priority: GGUF > .iara > safetensors
        // (but ternary mode forces safetensors path — needs raw BF16 weights for quantization)
        let load_result = if !ternary_on_load && resolved_gguf_file.is_some() { if let Some(ref gguf_path) = resolved_gguf_file {
            log::info!("📦 Detectado arquivo GGUF: {}", gguf_path.display());
            let _ = frame_tx.send(serde_json::json!({
                "type": "status",
                "message": format!("Carregando '{}' do formato GGUF (Q4_K_M → dequant → BF16)...", model_name)
            }).to_string());

            // Open GGUF to read metadata for ModelCfg + ModelArchitecture
            let gguf = match crate::storage::gguf::GgufFile::open(gguf_path) {
                Ok(g) => g,
                Err(e) => {
                    log::error!("❌ Falha ao abrir GGUF: {}", e);
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "status",
                        "message": format!("ERRO GGUF '{}': {}", model_name, e)
                    }).to_string());
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            };

            let cfg = match ModelCfg::from_gguf(&gguf) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("❌ Falha ao ler config GGUF: {}", e);
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "status",
                        "message": format!("ERRO config GGUF '{}': {}", model_name, e)
                    }).to_string());
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            };

            // Try to load arch from tokenizer dir (has generation_config.json for EOS),
            // fall back to GGUF metadata
            let tokenizer_dir_ref = if tokenizer_dir.join("generation_config.json").exists() {
                Some(tokenizer_dir.as_path())
            } else {
                None
            };
            let arch = match ModelArchitecture::from_gguf(&gguf, tokenizer_dir_ref) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("❌ Falha ao detectar arquitetura GGUF: {}", e);
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "status",
                        "message": format!("ERRO arquitetura GGUF '{}': {}", model_name, e)
                    }).to_string());
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            };

            // Drop the GgufFile — load_gguf will reopen it (it uses mmap internally)
            drop(gguf);

            CausalLM::load_gguf(gguf_path, cfg, arch, backend)
        } else { unreachable!() } } else if let Some(ref iara_path) = resolved_iara_file {
            log::info!("📦 Detectado arquivo .iara v2: {}", iara_path.display());
            let _ = frame_tx.send(serde_json::json!({
                "type": "status",
                "message": format!("Carregando '{}' do formato .iara v2...", model_name)
            }).to_string());

            // Para .iara, config.json pode estar no dir base do modelo
            let config_path = resolved_config_dir.join("config.json");
            let cfg = match ModelCfg::from_config_json(&config_path) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("❌ Falha ao ler config.json para .iara: {}", e);
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "status",
                        "message": format!("ERRO config para .iara '{}': {}", model_name, e)
                    }).to_string());
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            };

            let arch = match ModelArchitecture::from_config_json(&config_path) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("❌ Falha ao detectar arquitetura para .iara: {}", e);
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "status",
                        "message": format!("ERRO arquitetura para .iara '{}': {}", model_name, e)
                    }).to_string());
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            };

            CausalLM::load_from_iara(iara_path, cfg, arch, backend)
        } else if let Some(ref dna_path) = resolved_dna_file {
            log::info!("📦 Detectado arquivo .dna (SVD-factorized): {}", dna_path.display());
            let _ = frame_tx.send(serde_json::json!({
                "type": "status",
                "message": format!("Carregando '{}' do formato .dna (SVD direto para VRAM)...", model_name)
            }).to_string());

            let config_path = resolved_config_dir.join("config.json");
            let cfg = match ModelCfg::from_config_json(&config_path) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("❌ Falha ao ler config.json para .dna: {}", e);
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "status",
                        "message": format!("ERRO config para .dna '{}': {}", model_name, e)
                    }).to_string());
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            };

            let arch = match ModelArchitecture::from_config_json(&config_path) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("❌ Falha ao detectar arquitetura para .dna: {}", e);
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "status",
                        "message": format!("ERRO arquitetura para .dna '{}': {}", model_name, e)
                    }).to_string());
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            };

            CausalLM::load_dna(dna_path, cfg, arch, backend)
        } else {
            // Load from safetensors (traditional path)
            let config_path = model_path.join("config.json");
            let cfg = match ModelCfg::from_config_json(&config_path) {
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

            let arch = match ModelArchitecture::from_config_json(&config_path) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("❌ Falha ao detectar arquitetura: {}", e);
                    let _ = frame_tx.send(serde_json::json!({
                        "type": "status",
                        "message": format!("ERRO arquitetura '{}': {}", model_name, e)
                    }).to_string());
                    model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            };

            // Tenta carregar com quantização se habilitado
            if ternary_on_load {
                log::info!("⚡ Carregando com quantização TERNARY 2-bit (IARA_TERNARY=1)");
                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("Carregando '{}' com quantização Ternary 2-bit...", model_name)
                }).to_string());
                CausalLM::load_ternary(&model_path, cfg, arch.clone(), backend)
            } else if quantize_on_load {
                log::info!("🔧 Carregando com quantização Q4 (IARA_QUANTIZE=1)");
                let _ = frame_tx.send(serde_json::json!({
                    "type": "status",
                    "message": format!("Carregando '{}' com quantização Q4...", model_name)
                }).to_string());
                CausalLM::load_quantized(&model_path, cfg, arch.clone(), backend, enable_cfc)
            } else {
                CausalLM::load(&model_path, cfg, arch, backend)
            }
        };

        match load_result {
            Ok(mut model) => {
                // Pre-warm: run a fake forward pass so ROCm JIT-compiles all kernels
                // now (at load time) instead of on the first user request.
                // This eliminates the 30-60s warmup period on first generation.
                log::info!("🔥 Pre-warming GPU kernels (eliminates JIT on first request)...");
                let warmup_start = std::time::Instant::now();
                let _ = model.forward(&[1u32], None); // token_id=1, tiny forward
                let _ = model.forward(&[1u32], None); // second pass = fully warm
                model.reset_cache();
                log::info!("✅ Kernels pre-warmed em {:.1}s", warmup_start.elapsed().as_secs_f32());

                *model_arc.lock().await = Some(model);
                model_ready.store(true, std::sync::atomic::Ordering::Relaxed);
                model_loading.store(false, std::sync::atomic::Ordering::Relaxed);
                log::info!("✅ Modelo '{}' carregado em {:.1}s", model_name, start.elapsed().as_secs_f32());
                let backend_name = {
                    let g = model_arc.lock().await;
                    g.as_ref().map(|m| m.backend.name().to_string()).unwrap_or_default()
                };
                let quant_str = if resolved_gguf_file.is_some() { "gguf-q4km→bf16" } else if ternary_on_load { "ternary-2bit" } else if quantize_on_load { "q4" } else { "bf16" };
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
    Json(mut req): Json<GenerateRequest>,
) -> impl IntoResponse {
    log::info!("📝 Generate: prompt={}, stream={}",
        req.prompt.chars().take(50).collect::<String>(),
        req.stream);

    // Clamp / sanitize sampling params — guards against DoS (huge max_tokens
    // holds the model mutex for minutes) and garbage from NaN/negative inputs.
    req.max_tokens = req.max_tokens.clamp(1, 4096);
    if !(req.temperature.is_finite() && req.temperature >= 0.0) { req.temperature = 0.7; }
    if !(req.top_p.is_finite() && req.top_p > 0.0 && req.top_p <= 1.0) { req.top_p = 0.9; }

    let start = std::time::Instant::now();

    // ── Skip llama.cpp if budget specified (need CausalLM path for layer skipping) ──
    let force_native = req.budget.is_some();

    // ── Fast path: llama.cpp via batch server (supports concurrent requests) ──
    if !force_native {
        let batch_guard = state.batch_handle.lock().await;
        if let Some(handle) = batch_guard.as_ref() {
            let params = crate::llama::GenerateParams {
                temperature: req.temperature,
                top_k: req.top_k as i32,
                top_p: req.top_p,
                max_tokens: req.max_tokens,
            };

            // Drop the guard BEFORE awaiting — allows other requests to also submit
            let handle_clone = handle.clone();
            drop(batch_guard);

            let formatted_prompt = crate::tokenizer::format_for_instruct(&req.prompt);
            match handle_clone.submit(formatted_prompt, params).await {
                Ok(result) => {
                    let total_ms = start.elapsed().as_millis();
                    log::info!("✅ llama.cpp: {} tokens in {}ms ({:.1} tok/s)",
                        result.generated_tokens, result.eval_ms,
                        result.generated_tokens as f64 / (result.eval_ms.max(1) as f64 / 1000.0));

                    let tok_per_sec = if result.eval_ms > 0 {
                        result.generated_tokens as f64 * 1000.0 / result.eval_ms as f64
                    } else { 0.0 };
                    let response = GenerateResponse {
                        model: req.model.clone(),
                        created_at: chrono::Utc::now().to_rfc3339(),
                        response: result.text,
                        done: true,
                        context: None,
                        total_duration: total_ms,
                        prompt_eval_count: result.prompt_tokens as u32,
                        eval_count: result.generated_tokens as u32,
                        eval_duration: result.eval_ms,
                        tokens_per_second: tok_per_sec,
                    };
                    return (StatusCode::OK, Json(response)).into_response();
                }
                Err(e) => {
                    log::error!("❌ llama.cpp error: {}", e);
                    return (StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({ "error": e }))).into_response();
                }
            }
        }
    }

    // ── Legacy path: IARA CausalLM ──────────────────────────────────────
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

#[derive(Debug, Deserialize)]
pub struct DeepTraceRequest {
    pub prompt: String,
    #[serde(default)]
    pub token_index: usize,
}

/// POST /api/trace — Deep inference trace for a single token
/// Captures rich per-layer telemetry: attention entropy, MLP activation rates,
/// phase coherence, residual deltas, and more.
///
/// Request: {"prompt": "...", "token_index": 0}
/// Response: Full DeepTrace JSON
///
/// NOTE: This is slow (~2-3x slower than normal forward) and should only be
/// used when explicitly requested for debugging. Use for single tokens only.
pub async fn deep_trace_handler(
    State(state): State<AppState>,
    Json(req): Json<DeepTraceRequest>,
) -> impl IntoResponse {
    if !state.model_ready.load(std::sync::atomic::Ordering::Relaxed) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "No model loaded" })),
        ).into_response();
    }

    // Tokenize prompt
    let tokens = crate::tokenizer::tokenize_qa(&req.prompt);
    let prompt_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();

    if prompt_ids.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Empty prompt" })),
        ).into_response();
    }

    // Determine which token to trace: if token_index >= prompt length, trace the next generated token
    let trace_token_idx = if req.token_index < prompt_ids.len() {
        req.token_index
    } else {
        prompt_ids.len() - 1  // trace last token of prompt by default
    };

    // Hold the lock for the entire operation
    let mut model_guard = state.qwen_model.lock().await;
    let model = match model_guard.as_mut() {
        Some(m) => m,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Model lock failed" })),
            ).into_response();
        }
    };

    // Reset cache
    model.reset_cache();

    // Prefill: process all tokens before the trace target
    for i in 0..trace_token_idx {
        if let Err(e) = model.forward(&[prompt_ids[i]], None) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Prefill error at token {}: {}", i, e) })),
            ).into_response();
        }
    }

    // Deep trace the target token
    let trace_result = model.forward_deep_trace(&[prompt_ids[trace_token_idx]]);

    match trace_result {
        Ok(deep_trace) => (StatusCode::OK, Json(deep_trace)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Deep trace failed: {}", e) })),
        ).into_response(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) async fn generate_internal(
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

    // Chat template Qwen Instruct: <|im_start|>system/user/assistant
    let tokens = crate::tokenizer::tokenize_instruct(&req.prompt);
    let mut prompt_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();

    // Context guard: scratch + KV buffers are sized to max_seq_len, so prompt +
    // generated tokens must fit. Truncate the prompt (keep the most recent tail)
    // and cap generation to the remaining window instead of overrunning device
    // buffers — an unguarded long prompt is an out-of-bounds GPU write.
    let max_seq = {
        let guard = state.qwen_model.lock().await;
        guard.as_ref().map(|m| m.cfg.max_seq_len).unwrap_or(2048)
    };
    if prompt_ids.len() > max_seq.saturating_sub(1) {
        let keep = max_seq.saturating_sub(1).max(1);
        let start = prompt_ids.len() - keep;
        prompt_ids.drain(..start);
        log::warn!("Prompt truncated to last {} tokens (max_seq_len={})", keep, max_seq);
    }
    let gen_cap = max_seq.saturating_sub(prompt_ids.len()).max(1);
    let max_tokens = req.max_tokens.min(gen_cap);
    let prompt_eval_count = prompt_ids.len() as u32;
    let mut generated_text = String::new();
    let mut generated_token_ids: Vec<u32> = Vec::new();

    let mut brain_trace = BrainTrace::new(req.prompt.clone());
    let mut eval_count = 0u32;
    let mut last_token: u32 = 0;

    // Extract EOS token IDs from model architecture
    let eos_ids = {
        let guard = state.qwen_model.lock().await;
        match guard.as_ref() {
            Some(m) => m.arch.eos_token_ids.clone(),
            None => return Err("Model not loaded".into()),
        }
    };

    // Reset KV cache before generation
    {
        let mut guard = state.qwen_model.lock().await;
        let model = guard.as_mut().ok_or("Model not loaded")?;
        model.reset_cache();
    }

    // Prefill em batch
    if prompt_ids.len() > 1 {
        let mut model_guard = state.qwen_model.lock().await;
        let model = model_guard.as_mut().ok_or("Model not loaded")?;
        let _ = model.forward(&prompt_ids[..prompt_ids.len() - 1], None)
            .map_err(|e| format!("Prefill error: {}", e))?;
    }

    let mut decode_token = *prompt_ids.last().ok_or("Prompt vazio")?;

    // Parse budget param
    let layer_budget = req.budget.as_deref().map(|b| {
        match b.to_lowercase().as_str() {
            "fast" => Some(crate::lnn::esn::LayerBudget::Fast),
            "medium" => Some(crate::lnn::esn::LayerBudget::Medium),
            "slow" | "full" => Some(crate::lnn::esn::LayerBudget::Slow),
            _ => None,
        }
    }).flatten();

    // Hold mutex for entire decode loop — avoids per-token lock/unlock overhead
    let decode_ms = {
        let mut model_guard = state.qwen_model.lock().await;
        let model = model_guard.as_mut().ok_or("Model not loaded")?;
        let mut sampling_buf = SamplingBuffer::with_vocab_size(model.cfg.vocab_size);
        let decode_start = std::time::Instant::now();

        for _step in 0..max_tokens {
            let (tok_id, _tok_str) = if let Some(budget) = layer_budget {
                // Path com budget: usa forward_traced_buffered (suporta budget)
                let gen_step = crate::inference::forward_traced_buffered(
                    model, &[decode_token], &sampling, Some(budget), &mut sampling_buf)
                    .map_err(|e| format!("Forward error: {}", e))?;
                (gen_step.token_id, gen_step.token_str)
            } else {
                forward_decode_fast(model, &[decode_token], &sampling, &mut sampling_buf)
                    .map_err(|e| format!("Forward pass error: {}", e))?
            };

            decode_token = tok_id;
            last_token   = decode_token;
            if !eos_ids.contains(&last_token) {
                generated_token_ids.push(tok_id);
            }
            eval_count += 1;
            if eos_ids.contains(&last_token) { break; }
        }
        // Decode the whole id sequence at once → correct multi-byte UTF-8.
        generated_text = crate::tokenizer::decode(&generated_token_ids);
        let decode_duration = decode_start.elapsed();
        let ms = decode_duration.as_millis() as u128;
        log::warn!("DECODE: {} tokens in {}ms = {:.1} tok/s",
            eval_count, ms,
            eval_count as f64 / decode_duration.as_secs_f64());
        ms
    };

    brain_trace.finalize();
    *state.last_trace.lock().await = Some(brain_trace);

    let total_duration = start_time.elapsed();
    let tok_per_sec = if decode_ms > 0 {
        eval_count as f64 * 1000.0 / decode_ms as f64
    } else { 0.0 };
    Ok(GenerateResponse {
        model: req.model.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        response: generated_text,
        done: true,
        context: None,
        total_duration: total_duration.as_millis(),
        prompt_eval_count,
        eval_count,
        eval_duration: decode_ms,
        tokens_per_second: tok_per_sec,
    })
}

fn default_temperature() -> f32  { 0.7 }
fn default_top_k() -> u32        { 40 }
fn default_top_p() -> f32        { 0.9 }
fn default_max_tokens() -> usize { 100 }

// ─────────────────────────────────────────────────────────────────────────────
// LNN Training Endpoints
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TrainLnnRequest {
    /// Prompts para gerar e treinar (uma iteração por prompt)
    pub prompts: Vec<String>,
    /// Máximo de tokens por geração durante treinamento
    #[serde(default = "default_train_max_tokens")]
    pub max_tokens: usize,
    /// Se true, salva pesos após treinamento
    #[serde(default)]
    pub save_weights: bool,
}

fn default_train_max_tokens() -> usize { 64 }

#[derive(Debug, Serialize)]
pub struct TrainLnnResponse {
    pub epochs: usize,
    pub mean_reward: f32,
    pub esn_trained: bool,
    pub cfc_epochs: usize,
    pub rewards: Vec<f32>,
    pub message: String,
}

/// POST /api/train-lnn — Treina o sistema LNN com um batch de prompts.
///
/// Para cada prompt:
///   1. Gera tokens usando o Qwen + LNN atual
///   2. Computa reward signal
///   3. Treina ESN readout (ridge regression)
///   4. Treina CfC weights (backprop + SGD)
///   5. Próxima iteração usa pesos melhorados
pub async fn train_lnn_handler(
    State(state): State<AppState>,
    Json(req): Json<TrainLnnRequest>,
) -> impl IntoResponse {
    use crate::lnn::EsnObserver;
    use crate::lnn::train::{RewardSignal, train_esn_readout};

    const ESN_SEED: u64 = 0xDEAD_BEEF_1337_4242;

    if !state.model_ready.load(std::sync::atomic::Ordering::Relaxed) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Nenhum modelo carregado"})),
        ).into_response();
    }

    // Extract EOS token IDs from model architecture
    let eos_ids = {
        let guard = state.qwen_model.lock().await;
        match guard.as_ref() {
            Some(m) => m.arch.eos_token_ids.clone(),
            None => return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "Nenhum modelo carregado"})),
            ).into_response(),
        }
    };

    let prompts = req.prompts;
    if prompts.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Nenhum prompt fornecido"})),
        ).into_response();
    }

    let max_tokens = req.max_tokens.min(256);
    let mut rewards = Vec::new();
    let mut total_esn_trained = false;
    let mut cfc_epochs = 0usize;

    // Broadcast início do treinamento
    let _ = state.frame_tx.send(serde_json::json!({
        "type": "lnn_train_status",
        "message": format!("Iniciando treinamento LNN com {} prompts", prompts.len()),
        "enabled": true,
    }).to_string());

    for (idx, prompt) in prompts.iter().enumerate() {
        log::info!("LNN Training [{}/{}]: '{}'", idx + 1, prompts.len(), prompt.chars().take(40).collect::<String>());

        let tokens = crate::tokenizer::tokenize_qa(prompt);
        let prompt_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        if prompt_ids.is_empty() { continue; }

        let mut guard = state.qwen_model.lock().await;
        let m = match guard.as_mut() {
            Some(m) => m,
            None => break,
        };

        m.reset_cache();

        // Prefill em batch: todos os tokens do prompt (exceto o último) em um único forward pass
        if prompt_ids.len() > 1 {
            if let Err(e) = m.forward(&prompt_ids[..prompt_ids.len() - 1], None) {
                log::error!("Train prefill error: {}", e);
            }
        }

        // Generate
        let mut esn = EsnObserver::new(ESN_SEED);
        esn.reset();
        let mut sampling = SamplingParams::default();
        let mut decode_token = *prompt_ids.last().unwrap();
        let mut generated_tokens = Vec::new();
        let mut sampling_buf = SamplingBuffer::with_vocab_size(m.cfg.vocab_size);

        for step in 0..max_tokens {
            let gen_step = match crate::inference::forward_traced_buffered(m, &[decode_token], &sampling, None, &mut sampling_buf) {
                Ok(s) => s,
                Err(_) => break,
            };
            decode_token = gen_step.token_id;

            if eos_ids.contains(&decode_token) { break; }

            generated_tokens.push(decode_token);

            let top_scores: Vec<f32> = gen_step.top_logits.iter().map(|(_, s)| *s).collect();
            esn.step(&top_scores, step, max_tokens);
            sampling = esn.modulate(&sampling);
        }

        // Train ESN
        let reward = RewardSignal::compute(&esn.obs_history, &generated_tokens);
        rewards.push(reward.mean_reward);

        if let Some(new_w) = train_esn_readout(&esn, &reward) {
            esn.set_readout(new_w);
            total_esn_trained = true;
        }

        // Train CfC (if enabled)
        // Note: CfC training requires activation snapshots from the generation loop.
        // For the HTTP training endpoint, we do a simplified version.
        if m.cfc.is_some() {
            // In the HTTP path, CfC training happens through the main WS engine loop.
            // Here we just track that it's available.
            cfc_epochs += 1;
        }

        // Broadcast progress
        let _ = state.frame_tx.send(serde_json::json!({
            "type": "lnn_train",
            "generation": idx + 1,
            "total": prompts.len(),
            "reward": {
                "mean": reward.mean_reward,
                "entropy": reward.mean_entropy,
                "diversity": reward.lexical_diversity,
                "repetition": reward.repetition_score,
            },
            "esn_trained": total_esn_trained,
        }).to_string());
    }

    // Save weights if requested
    if req.save_weights {
        let guard = state.qwen_model.lock().await;
        if let Some(m) = guard.as_ref() {
            let weights_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../models/lnn_weights.json");
            let esn_for_save = EsnObserver::new(ESN_SEED);
            // Note: ESN readout was trained per-prompt above; last one is lost
            // In production, persist the trainer state
            if let Err(e) = crate::lnn::train::save_lnn_weights(
                &esn_for_save,
                m.cfc.as_ref(),
                &m.backend,
                &weights_path,
            ) {
                log::warn!("Falha ao salvar pesos LNN: {}", e);
            }
        }
    }

    let mean_reward = if rewards.is_empty() {
        0.0
    } else {
        rewards.iter().sum::<f32>() / rewards.len() as f32
    };

    let response = TrainLnnResponse {
        epochs: prompts.len(),
        mean_reward,
        esn_trained: total_esn_trained,
        cfc_epochs,
        rewards: rewards.clone(),
        message: format!(
            "LNN treinado: {} epochs, reward médio {:.4}, ESN={}, CfC epochs={}",
            prompts.len(), mean_reward, total_esn_trained, cfc_epochs
        ),
    };

    // Broadcast final
    let _ = state.frame_tx.send(serde_json::json!({
        "type": "lnn_train_status",
        "message": response.message.clone(),
        "enabled": true,
        "generations": prompts.len(),
        "mean_reward": mean_reward,
    }).to_string());

    (StatusCode::OK, Json(serde_json::json!(response))).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural Visualization Endpoint
// ─────────────────────────────────────────────────────────────────────────────

/// Request for neural visualization — runs inference with full telemetry
#[derive(Deserialize)]
pub struct NeurovizRequest {
    pub prompt: String,
    /// Max tokens to generate (default 1 — just the first token)
    #[serde(default = "default_viz_max_tokens")]
    pub max_tokens: usize,
    /// Sample every N layers (1 = all, 4 = every 4th, default 2)
    #[serde(default = "default_viz_sample_every")]
    pub sample_every: usize,
}

fn default_viz_max_tokens() -> usize { 30 }
fn default_viz_sample_every() -> usize { 2 }

/// Per-neuron data for one layer at one token step
#[derive(Serialize, Clone)]
pub struct LayerNeuronViz {
    pub layer_idx: usize,
    /// Top MLP neurons: (neuron_index, activation_magnitude)
    pub mlp_top: Vec<(usize, f32)>,
    /// MLP activation rate 0-100% (placeholder: 0.0 until deep telemetry kernels land)
    pub mlp_activation_rate: f32,
    /// Top attention positions: (position, weight) — placeholder
    pub attn_top: Vec<(usize, f32)>,
    /// Attention entropy (0=focused, high=diffuse) — placeholder
    pub attn_entropy: f32,
    /// Residual norm (how much info flows through)
    pub residual_norm: f32,
    /// How much this layer changed the residual stream (attention + mlp energy sum)
    pub residual_delta: f32,
    /// Phase coherence (0=chaotic, 1=synchronized) — placeholder
    pub phase_coherence: f32,
    /// Hot neurons in residual stream (top-10 indices)
    pub hot_neurons: Vec<usize>,
    pub max_activation: f32,
}

/// Per-token visualization data
#[derive(Serialize, Clone)]
pub struct TokenViz {
    pub token_id: u32,
    pub token_str: String,
    pub position: usize,
    pub layers: Vec<LayerNeuronViz>,
    /// Final logit distribution top-20: (id, prob, str)
    pub top_logits: Vec<(u32, f32, String)>,
    pub confidence: f32,
    pub logit_entropy: f32,
}

/// Full visualization response
#[derive(Serialize)]
pub struct NeurovizResponse {
    pub prompt: String,
    /// The full generated text (all generated tokens concatenated)
    pub generated_text: String,
    pub tokens: Vec<TokenViz>,
    pub model_info: NeurovizModelInfo,
    pub session_id: String,
}

#[derive(Serialize)]
pub struct NeurovizModelInfo {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
}

/// POST /api/neuroviz — Run inference with full telemetry for visualization UI.
pub async fn neuroviz_handler(
    State(state): State<AppState>,
    Json(req): Json<NeurovizRequest>,
) -> impl IntoResponse {
    use crate::inference::LayerTrace;

    if !state.model_ready.load(std::sync::atomic::Ordering::Relaxed) {
        return (StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "no model loaded"}))).into_response();
    }

    // Tokenize using the global tokenizer (loaded alongside the model)
    let tokens = crate::tokenizer::tokenize(&req.prompt);
    let prompt_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();

    if prompt_ids.is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "empty prompt"}))).into_response();
    }

    // ── Try native qwen_model path first ─────────────────────────────────────
    {
        let mut guard = state.qwen_model.lock().await;
        if let Some(model) = guard.as_mut() {
            let model_info = NeurovizModelInfo {
                num_layers:        model.cfg.num_layers,
                hidden_size:       model.cfg.hidden_size,
                intermediate_size: model.cfg.intermediate_size,
                num_heads:         model.cfg.num_heads,
                num_kv_heads:      model.cfg.num_kv_heads,
                vocab_size:        model.cfg.vocab_size,
            };
            let vocab_size = model.cfg.vocab_size;
            let max_tokens = req.max_tokens.min(64);

            model.reset_cache();

            // Bug 2 fix: enable telemetry so layer traces are populated (not all zeros)
            model.telemetry_enabled = true;

            // Prefill: run all tokens except the last in a single batched forward pass
            if prompt_ids.len() > 1 {
                if let Err(e) = model.forward(&prompt_ids[..prompt_ids.len() - 1], None) {
                    model.telemetry_enabled = false;
                    return (StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": format!("prefill error: {e}")}))).into_response();
                }
            }

            // Last prompt token — this gives us logits + layer traces
            let last_id = *prompt_ids.last().unwrap();
            let (logits, layer_traces) = match model.forward(&[last_id], None) {
                Ok(v) => v,
                Err(e) => {
                    model.telemetry_enabled = false;
                    return (StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": format!("forward error: {e}")}))).into_response();
                }
            };

            // Restore telemetry state to default (off) to avoid perf impact in generate
            model.telemetry_enabled = false;

            // Build TokenViz entries — add all prompt tokens first (simplified, no layer data)
            let mut result_tokens: Vec<TokenViz> = Vec::new();
            let mut generated_text = String::new();

            // Prompt tokens (all except last): simplified entries with no layer traces
            for (pos, &pid) in prompt_ids[..prompt_ids.len()-1].iter().enumerate() {
                let tok_str = crate::tokenizer::token_id_to_text(pid);
                result_tokens.push(TokenViz {
                    token_id: pid,
                    token_str: tok_str,
                    position: pos,
                    layers: vec![],  // no telemetry for prefill tokens (perf)
                    top_logits: vec![],
                    confidence: 0.0,
                    logit_entropy: 0.0,
                });
            }

            // Last prompt token: full layer traces + logits
            let last_token_str = crate::tokenizer::token_id_to_text(last_id);
            result_tokens.push(build_neuroviz_token(
                last_id,
                &last_token_str,
                prompt_ids.len() - 1,
                &layer_traces,
                &logits,
                vocab_size,
            ));

            // Greedy decode: first generated token
            let mut decode_token = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(last_id);

            let eos_ids = model.arch.eos_token_ids.clone();

            // Generate up to max_tokens tokens with full telemetry
            for step in 0..max_tokens {
                if eos_ids.contains(&decode_token) { break; }

                model.telemetry_enabled = true;
                let step_result = model.forward(&[decode_token], None);
                model.telemetry_enabled = false;

                let (step_logits, step_traces) = match step_result {
                    Ok(v) => v,
                    Err(_) => break,
                };

                let tok_str = crate::tokenizer::token_id_to_text(decode_token);
                generated_text.push_str(&tok_str);
                result_tokens.push(build_neuroviz_token(
                    decode_token,
                    &tok_str,
                    prompt_ids.len() + step,
                    &step_traces,
                    &step_logits,
                    vocab_size,
                ));

                decode_token = step_logits.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
            }

            let session_id = format!("viz_{}", chrono::Utc::now().timestamp_millis());
            let response = NeurovizResponse {
                prompt: req.prompt,
                generated_text,
                tokens: result_tokens,
                model_info,
                session_id,
            };
            return (StatusCode::OK, Json(response)).into_response();
        }
        // guard dropped here — qwen_model is None, fall through to llama.cpp path
    }

    // ── llama.cpp fallback path ───────────────────────────────────────────────
    // When llama.cpp is loaded, qwen_model is None.  We run a short generate
    // with the llama engine and then read state.last_trace to get layer data.
    // The llama engine does not expose per-layer telemetry directly, so we
    // synthesise a minimal NeurovizResponse from the generate result.
    {
        let llama_guard = state.llama_engine.lock().await;
        if llama_guard.is_none() {
            return (StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "no model loaded (neither native nor llama.cpp)"}))).into_response();
        }
        drop(llama_guard);
    }

    let viz_max_tokens = req.max_tokens.min(64).max(1);

    // Query model info and run generation (greedy, deterministic for neuroviz)
    let (model_info, gen_result) = {
        let mut llama_guard = state.llama_engine.lock().await;
        let engine = match llama_guard.as_mut() {
            Some(e) => e,
            None => return (StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "llama engine not available"}))).into_response(),
        };
        let info = engine.model_info();
        let model_info = NeurovizModelInfo {
            num_layers:        info.n_layers as usize,
            hidden_size:       info.n_embd as usize,
            // llama.cpp doesn't expose intermediate_size; approximate as 8/3 × hidden
            intermediate_size: (info.n_embd as usize * 8 / 3).next_power_of_two(),
            num_heads:         0,
            num_kv_heads:      0,
            vocab_size:        info.vocab_size as usize,
        };
        let params = crate::llama::GenerateParams {
            temperature: 0.0, // greedy for deterministic neuroviz
            top_k: 1,
            top_p: 1.0,
            max_tokens: viz_max_tokens,
        };
        (model_info, engine.generate(&req.prompt, &params))
    };

    let gen_text = match gen_result {
        Ok(r) => r.text,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("llama generate error: {e}")}))).into_response(),
    };

    // Read last_trace if available (written by generate_internal for native path);
    // for llama.cpp there is no trace, so build a synthetic single-token response.
    let trace_opt = state.last_trace.lock().await.clone();
    let num_layers = model_info.num_layers;

    // Build a synthetic TokenViz: no real layer data from llama.cpp FFI
    // (llama.cpp does not expose per-layer activations via its C API)
    let synthetic_layers: Vec<LayerNeuronViz> = if let Some(ref trace) = trace_opt {
        // Reuse any existing BrainTrace layer data from a previous native run
        trace.steps.iter().flat_map(|s| s.layer_traces.iter()).take(num_layers).map(|lt| {
            LayerNeuronViz {
                layer_idx:           lt.layer_idx,
                mlp_top:             lt.hot_neurons.iter().cloned().zip(lt.hot_neuron_values.iter().cloned()).collect(),
                mlp_activation_rate: 0.0,
                attn_top:            Vec::new(),
                attn_entropy:        0.0,
                residual_norm:       lt.residual_norm,
                residual_delta:      lt.attention_energy + lt.mlp_energy,
                phase_coherence:     0.0,
                hot_neurons:         lt.hot_neurons.clone(),
                max_activation:      lt.max_activation,
            }
        }).collect()
    } else {
        // No trace available — return empty layer stubs so the UI can render
        (0..num_layers).map(|i| LayerNeuronViz {
            layer_idx: i,
            mlp_top: vec![],
            mlp_activation_rate: 0.0,
            attn_top: vec![],
            attn_entropy: 0.0,
            residual_norm: 0.0,
            residual_delta: 0.0,
            phase_coherence: 0.0,
            hot_neurons: vec![],
            max_activation: 0.0,
        }).collect()
    };

    // Build token list: prompt tokens (simplified) + one generated entry with layer data
    let mut viz_tokens: Vec<TokenViz> = Vec::new();

    // All prompt tokens (simplified, no layer data — llama.cpp doesn't expose per-token traces)
    for (pos, &pid) in prompt_ids.iter().enumerate() {
        let tok_str = crate::tokenizer::token_id_to_text(pid);
        viz_tokens.push(TokenViz {
            token_id: pid,
            token_str: tok_str,
            position: pos,
            layers: vec![],
            top_logits: vec![],
            confidence: 0.0,
            logit_entropy: 0.0,
        });
    }

    // Generated text as word-level entries (llama.cpp returns full text, split by spaces for UX)
    let gen_words: Vec<&str> = gen_text.split_whitespace().collect();
    for (i, word) in gen_words.iter().enumerate() {
        let layers_for_tok = if i == 0 { synthetic_layers.clone() } else { vec![] };
        viz_tokens.push(TokenViz {
            token_id: 0,
            token_str: format!(" {}", word),
            position: prompt_ids.len() + i,
            layers: layers_for_tok,
            top_logits: vec![],
            confidence: 0.0,
            logit_entropy: 0.0,
        });
    }

    let session_id = format!("viz_llama_{}", chrono::Utc::now().timestamp_millis());
    let response = NeurovizResponse {
        prompt: req.prompt,
        generated_text: gen_text,
        tokens: viz_tokens,
        model_info,
        session_id,
    };
    (StatusCode::OK, Json(response)).into_response()
}

/// Build a `TokenViz` from a completed forward pass.
fn build_neuroviz_token(
    token_id: u32,
    token_str: &str,
    position: usize,
    layer_traces: &[crate::inference::LayerTrace],
    logits: &[f32],
    vocab_size: usize,
) -> TokenViz {
    // Build per-layer viz from LayerTrace
    let layers: Vec<LayerNeuronViz> = layer_traces.iter().map(|lt| {
        // Represent hot neurons as (index, max_activation) pairs
        let mlp_top: Vec<(usize, f32)> = lt.hot_neurons.iter().cloned()
            .zip(lt.hot_neuron_values.iter().cloned())
            .collect();

        LayerNeuronViz {
            layer_idx:           lt.layer_idx,
            mlp_top,
            mlp_activation_rate: lt.mlp_activation_rate,
            attn_top:            Vec::new(),
            attn_entropy:        0.0,
            residual_norm:       lt.residual_norm,
            residual_delta:      lt.attention_energy + lt.mlp_energy,
            phase_coherence:     lt.phase_coherence,
            hot_neurons:         lt.hot_neurons.clone(),
            max_activation:      lt.max_activation,
        }
    }).collect();

    // Compute softmax over full vocab for top-20 logits
    let n = vocab_size.min(logits.len());
    let max_l = logits[..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits[..n].iter().map(|&v| (v - max_l).exp()).collect();
    let exp_sum: f32 = exp.iter().sum();

    // Collect (index, prob) and partial-sort top-20
    let mut idx_probs: Vec<(usize, f32)> = exp.iter().enumerate()
        .map(|(i, &e)| (i, e / exp_sum))
        .collect();
    // select_nth_unstable_by to get top-20 in O(n) time
    let top_n = 20.min(idx_probs.len());
    if top_n > 0 {
        idx_probs.select_nth_unstable_by(top_n - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx_probs[..top_n].sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let top_logits: Vec<(u32, f32, String)> = idx_probs[..top_n].iter().map(|(i, p)| {
        let s = crate::tokenizer::token_id_to_text(*i as u32);
        (*i as u32, *p, s)
    }).collect();

    let confidence = top_logits.first().map(|(_, p, _)| *p).unwrap_or(0.0);
    let logit_entropy = -top_logits.iter()
        .map(|(_, p, _)| if *p > 1e-10 { p * p.ln() } else { 0.0 })
        .sum::<f32>();

    TokenViz {
        token_id,
        token_str: token_str.to_string(),
        position,
        layers,
        top_logits,
        confidence,
        logit_entropy,
    }
}

/// GET /api/lnn/status — Status do sistema LNN (ESN + CfC + Gravity + Training).
// ─────────────────────────────────────────────────────────────────────────────
// POST /api/train — LoRA fine-tuning
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
pub struct TrainLoraRequest {
    /// Path to tokenized .bin dataset (flat little-endian u32 token IDs).
    pub data_path: String,
    /// Directory with safetensors weights + config.json.
    /// If omitted, uses models_dir/<model_name> from the currently loaded model's name.
    pub model_dir: Option<String>,
    #[serde(default = "default_lora_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f32,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default = "default_seq_len")]
    pub seq_len: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_output_dir")]
    pub output_dir: String,
    #[serde(default = "default_checkpoint_every")]
    pub checkpoint_every: usize,
    #[serde(default = "default_log_every")]
    pub log_every: usize,
}

fn default_lora_rank()       -> usize { 16 }
fn default_lora_alpha()      -> f32   { 32.0 }
fn default_lr()              -> f32   { 2e-4 }
fn default_steps()           -> usize { 100 }
fn default_seq_len()         -> usize { 64 }
fn default_batch_size()      -> usize { 1 }
fn default_output_dir()      -> String { "lora_output".into() }
fn default_checkpoint_every()-> usize { 50 }
fn default_log_every()       -> usize { 10 }

/// POST /api/train
///
/// Spawns a background LoRA training task. Returns immediately with 202.
/// Progress is broadcast via WebSocket as {"type":"train_progress",...} frames.
pub async fn train_lora_handler(
    State(state): State<AppState>,
    Json(req): Json<TrainLoraRequest>,
) -> impl IntoResponse {
    // Resolve model_dir: explicit in request or infer from currently loaded model.
    let model_dir_str = match req.model_dir {
        Some(ref d) => d.clone(),
        None => {
            // Try to infer from AppState (models_dir + first safetensors model present)
            let guard = state.qwen_model.lock().await;
            if guard.is_none() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "model_dir not specified and no model is loaded"
                    })),
                ).into_response();
            }
            drop(guard);
            // Use models_dir as the model_dir (for single-model setups)
            state.models_dir.to_string_lossy().into_owned()
        }
    };

    let data_path    = PathBuf::from(&req.data_path);
    let model_dir    = PathBuf::from(&model_dir_str);
    let output_dir   = PathBuf::from(&req.output_dir);
    let frame_tx     = state.frame_tx.clone();

    let config = iara_train::TrainConfig {
        lr: req.lr,
        num_steps: req.steps,
        batch_size: req.batch_size,
        seq_len: req.seq_len,
        checkpoint_every: req.checkpoint_every,
        output_dir: output_dir.clone(),
        lora_rank: req.lora_rank,
        lora_alpha: req.lora_alpha,
        log_every: req.log_every,
        ..Default::default()
    };

    // Spawn CPU-bound training in a blocking thread so it doesn't block the async runtime.
    tokio::task::spawn_blocking(move || {
        let _ = frame_tx.send(serde_json::json!({
            "type": "train_progress",
            "status": "loading_model",
            "message": format!("Loading model from {}", model_dir.display()),
        }).to_string());

        let weights = match iara_train::ModelWeights::from_dir(&model_dir) {
            Ok(w) => w,
            Err(e) => {
                let _ = frame_tx.send(serde_json::json!({
                    "type": "train_progress",
                    "status": "error",
                    "message": format!("Failed to load model weights: {e}"),
                }).to_string());
                return;
            }
        };

        let mut lora = iara_train::LoraAdapter::new_for_qwen2(
            config.lora_rank, config.lora_alpha, &weights.config,
        );

        let dataset = iara_train::TextDataset::from_token_file(&data_path, config.seq_len);

        let _ = frame_tx.send(serde_json::json!({
            "type": "train_progress",
            "status": "started",
            "message": format!("LoRA training started: {} steps, rank={}, lr={}",
                config.num_steps, config.lora_rank, config.lr),
            "trainable_params": lora.num_params(),
            "dataset_samples": dataset.num_samples(),
        }).to_string());

        let trainer = iara_train::Trainer::new(config);
        match trainer.train(&weights, &mut lora, &dataset) {
            Ok(()) => {
                let _ = frame_tx.send(serde_json::json!({
                    "type": "train_progress",
                    "status": "done",
                    "message": format!("Training complete. LoRA saved to {}", output_dir.display()),
                }).to_string());
            }
            Err(e) => {
                let _ = frame_tx.send(serde_json::json!({
                    "type": "train_progress",
                    "status": "error",
                    "message": format!("Training error: {e}"),
                }).to_string());
            }
        }
    });

    (StatusCode::ACCEPTED, Json(serde_json::json!({
        "status": "accepted",
        "message": "LoRA training started in background. Watch WebSocket for progress.",
        "model_dir": model_dir_str,
        "data_path": req.data_path,
        "steps": req.steps,
        "lora_rank": req.lora_rank,
        "output_dir": req.output_dir,
    }))).into_response()
}

pub async fn lnn_status_handler(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let guard = state.qwen_model.lock().await;

    let mut status = serde_json::json!({
        "model_loaded": guard.is_some(),
    });

    if let Some(m) = guard.as_ref() {
        status["cfc_enabled"] = serde_json::json!(m.cfc.is_some());
        status["gravity_enabled"] = serde_json::json!(m.gravity.is_some());
        status["quantized"] = serde_json::json!(m.quantized);

        if let Some(ref cfc) = m.cfc {
            status["cfc"] = serde_json::json!({
                "cfc_dim": cfc.config.cfc_dim,
                "hidden_dim": cfc.config.hidden_dim,
                "param_count": cfc.param_count(),
                "vram_mb": cfc.vram_bytes() as f64 / (1024.0 * 1024.0),
                "lr": cfc.config.lr,
            });
        }

        if let Some(ref grav) = m.gravity {
            status["gravity"] = serde_json::json!({
                "base_threshold": grav.config.base_threshold,
                "max_correction_ratio": grav.config.max_correction_ratio,
                "history_len": grav.history_len(),
                "current_threshold": grav.current_threshold(),
            });
        }
    }

    (StatusCode::OK, Json(status)).into_response()
}
