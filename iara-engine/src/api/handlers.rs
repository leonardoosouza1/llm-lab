// ============================================================================
// api/handlers.rs — HTTP Handlers for IARA API
// ============================================================================
// POST /api/generate — generation endpoint (non-streaming)
// GET /api/debug/last-trace — retrieve last brain trace
// GET /health — health check
// ============================================================================

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::debug::brain::BrainTrace;
use crate::model::QwenModel;
use crate::qwen::forward_traced;

// ─────────────────────────────────────────────────────────────────────────────
// Request/Response Types
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

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub version: String,
    pub timestamp: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// App State
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    /// None enquanto o modelo ainda está carregando
    pub qwen_model: Arc<Mutex<Option<QwenModel>>>,
    pub last_trace: Arc<Mutex<Option<BrainTrace>>>,
    pub model_ready: Arc<std::sync::atomic::AtomicBool>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

pub async fn health_handler(
    State(state): State<AppState>,
) -> Json<HealthResponse> {
    let ready = state.model_ready.load(std::sync::atomic::Ordering::Relaxed);
    Json(HealthResponse {
        status: if ready { "ok".to_string() } else { "loading".to_string() },
        model: "qwen3-8b".to_string(),
        version: "1.0.0".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

pub async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    log::info!("📝 Generate request: prompt={}, stream={}",
        &req.prompt[..req.prompt.len().min(50)],
        req.stream);

    let start = std::time::Instant::now();

    match generate_internal(&state, &req).await {
        Ok(response) => {
            let duration = start.elapsed();
            log::info!("✅ Generation complete: {:.2}ms", duration.as_secs_f32() * 1000.0);
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            log::error!("❌ Generation error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

pub async fn debug_trace_handler(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let trace = state.last_trace.lock().await.clone();

    match trace {
        Some(t) => {
            let analysis = t.analyze();
            (StatusCode::OK, Json(analysis)).into_response()
        }
        None => {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "No trace available yet" })),
            )
                .into_response()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal Implementation
// ─────────────────────────────────────────────────────────────────────────────

async fn generate_internal(
    state: &AppState,
    req: &GenerateRequest,
) -> Result<GenerateResponse, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();

    // Check model is ready
    if !state.model_ready.load(std::sync::atomic::Ordering::Relaxed) {
        return Err("Modelo ainda carregando, aguarde...".into());
    }

    // 1. Tokenize input
    let tokens = crate::tokenizer::tokenize(&req.prompt);
    let token_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    let prompt_eval_count = token_ids.len() as u32;
    let mut current_tokens = token_ids.clone();
    let mut generated_text = String::new();

    let mut brain_trace = BrainTrace::new(req.prompt.clone());
    let mut eval_count = 0u32;

    // Qwen3-8B EOS token id = 151645
    const EOS_TOKEN: u32 = 151645;

    // 2. Generation loop
    for _ in 0..req.max_tokens {
        if current_tokens.is_empty() {
            break;
        }

        let gen_step = {
            let mut model_guard = state.qwen_model.lock().await;
            let model = model_guard.as_mut().ok_or("Model not loaded")?;
            forward_traced(model, &current_tokens)
                .map_err(|e| format!("Forward pass error: {}", e))?
        };

        let next_token = gen_step.token_id;
        generated_text.push_str(&gen_step.token_str);
        brain_trace.add_step(gen_step);

        current_tokens.push(next_token);
        eval_count += 1;

        if next_token == EOS_TOKEN {
            break;
        }
    }

    // 3. Save trace
    brain_trace.finalize();
    *state.last_trace.lock().await = Some(brain_trace);

    let total_duration = start_time.elapsed();

    Ok(GenerateResponse {
        model: req.model.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        response: generated_text,
        done: true,
        context: Some(current_tokens),
        total_duration: total_duration.as_millis(),
        prompt_eval_count,
        eval_count,
        eval_duration: total_duration.as_millis(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Default Values
// ─────────────────────────────────────────────────────────────────────────────

fn default_temperature() -> f32 { 0.7 }
fn default_top_k() -> u32 { 40 }
fn default_max_tokens() -> usize { 100 }
