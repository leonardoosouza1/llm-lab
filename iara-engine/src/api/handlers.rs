// ============================================================================
// api/handlers.rs — HTTP Handlers for IARA Ollama API
// ============================================================================
// POST /api/generate — Main generation endpoint (streaming + non-streaming)
// GET /api/debug/last-trace — Retrieve last brain trace analysis
// GET /health — Health check
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

use crate::debug::BrainTrace;
use crate::qwen::GenerationStep;

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
    // IARA-specific
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_activations: Option<Vec<String>>, // JSON string for streaming
}

#[derive(Debug, Serialize)]
pub struct StreamChunk {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_activations: Option<Vec<String>>,
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
    pub qwen_model: Arc<Mutex<crate::qwen::QwenModel>>,
    pub tokenizer: Arc<crate::tokenizer::Tokenizer>,
    pub iara: Arc<crate::iara::IARAContext>,
    pub last_trace: Arc<Mutex<Option<BrainTrace>>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

pub async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
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

    // 1. Tokenize input
    let token_ids = state.tokenizer.encode(&req.prompt)?;
    let prompt_eval_count = token_ids.len() as u32;
    let mut current_tokens = token_ids.clone();
    let mut generated_tokens = String::new();

    let mut brain_trace = BrainTrace::new(req.prompt.clone());
    let mut eval_count = 0u32;

    // 2. Generation loop
    for step in 0..req.max_tokens {
        if current_tokens.is_empty() {
            break;
        }

        let last_token = *current_tokens.last().unwrap();

        // Forward pass with tracing
        let mut model = state.qwen_model.lock().await;
        let gen_step = crate::qwen::forward_and_trace(
            &*model,
            last_token,
            current_tokens.len(),
        )
        .await?;

        drop(model);

        // Sample next token
        let next_token = sample_token(&gen_step.logits, req.temperature, req.top_k)?;

        // Accumulate response
        generated_tokens.push_str(&gen_step.token_str);
        brain_trace.add_step(gen_step.clone());

        // Trigger IARA simulation (async, non-blocking)
        let iara = state.iara.clone();
        let traces = gen_step.layer_traces.clone();
        tokio::spawn(async move {
            if let Ok(snapshot) = crate::iara::inject_and_simulate(&iara, &traces).await {
                log::debug!(
                    "🧠 IARA: {} active nodes, energy={:.2}",
                    snapshot.nodes_active,
                    snapshot.mean_energy
                );
            }
        });

        current_tokens.push(next_token);
        eval_count += 1;

        // Check stopping condition
        if next_token == state.tokenizer.eos_token() {
            break;
        }
    }

    // 3. Save trace for debugging
    brain_trace.finalize();
    *state.last_trace.lock().await = Some(brain_trace.clone());

    let total_duration = start_time.elapsed();
    let eval_duration = start_time.elapsed(); // Simplified; could track precisely

    Ok(GenerateResponse {
        model: req.model.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        response: generated_tokens,
        done: true,
        context: Some(current_tokens),
        total_duration: total_duration.as_millis(),
        prompt_eval_count,
        eval_count,
        eval_duration: eval_duration.as_millis(),
        layer_activations: None,
    })
}

fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: u32,
) -> Result<u32, Box<dyn std::error::Error>> {
    // Simple greedy sampling (can be improved with top-k, top-p)
    let idx = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .ok_or("Empty logits")?;

    Ok(idx)
}

// ─────────────────────────────────────────────────────────────────────────────
// Default Values
// ─────────────────────────────────────────────────────────────────────────────

fn default_temperature() -> f32 {
    0.7
}

fn default_top_k() -> u32 {
    40
}

fn default_max_tokens() -> usize {
    100
}
