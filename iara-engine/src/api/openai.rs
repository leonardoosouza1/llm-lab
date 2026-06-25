// ============================================================================
// api/openai.rs — OpenAI-compatible API surface
// ============================================================================
// Drop-in subset of the OpenAI API so IARA can be used by any OpenAI client
// (openai-python, llm_bridge, LangChain, vLLM-style tooling, etc.):
//
//   POST /v1/chat/completions   — chat completion (non-streaming)
//   GET  /v1/models             — list available models
//
// Internally this reuses the exact same generation paths as /api/generate:
//   1. llama.cpp batch server (GGUF models) — concurrent
//   2. native IARA CausalLM   (safetensors / .iara / .dna)
//
// The chat messages are rendered to a full Qwen ChatML prompt here. Because the
// rendered prompt already contains `<|im_start|>`, the downstream instruct
// formatter (`tokenizer::format_for_instruct`) detects it and does NOT wrap it
// a second time — so multi-turn context is preserved verbatim.
// ============================================================================

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};

use super::handlers::{generate_internal, AppState, GenerateRequest, GenerateResponse};

// ─────────────────────────────────────────────────────────────────────────────
// Request / Response types (OpenAI schema subset)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Non-standard extension: nucleus sampling top-k (IARA-specific).
    pub top_k: Option<u32>,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    /// Accepted for compatibility; SSE streaming is not implemented yet, so the
    /// full completion is always returned in a single response.
    #[serde(default)]
    pub stream: bool,
}

fn default_model() -> String {
    "iara".to_string()
}

/// Server-side cap on generated tokens — bounds a single request's hold on the
/// model mutex (DoS guard) and keeps generation within the KV-cache window.
const MAX_TOKENS_CAP: usize = 4096;

fn sanitize_temperature(t: Option<f32>) -> f32 {
    match t {
        Some(v) if v.is_finite() && v >= 0.0 => v,
        _ => 0.7,
    }
}

fn sanitize_top_p(p: Option<f32>) -> f32 {
    match p {
        Some(v) if v.is_finite() && v > 0.0 && v <= 1.0 => v,
        _ => 0.9,
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

/// POST /v1/chat/completions — OpenAI-compatible chat completion.
pub async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if req.messages.is_empty() {
        return openai_error("`messages` must not be empty", StatusCode::BAD_REQUEST)
            .into_response();
    }

    if !state.model_ready.load(std::sync::atomic::Ordering::Relaxed) {
        let msg = if state.model_loading.load(std::sync::atomic::Ordering::Relaxed) {
            "Model is still loading, try again shortly"
        } else {
            "No model loaded — POST /api/load-model first"
        };
        return openai_error(msg, StatusCode::SERVICE_UNAVAILABLE).into_response();
    }

    // SSE streaming isn't implemented — fail loudly instead of returning a
    // single JSON blob that an OpenAI streaming client would choke on.
    if req.stream {
        return openai_error(
            "streaming is not supported yet — set \"stream\": false",
            StatusCode::BAD_REQUEST,
        )
        .into_response();
    }

    let prompt = messages_to_chatml(&req.messages);
    let temperature = sanitize_temperature(req.temperature);
    let top_p = sanitize_top_p(req.top_p);
    let top_k = req.top_k.unwrap_or(40);
    let max_tokens = req.max_tokens.unwrap_or(256).clamp(1, MAX_TOKENS_CAP);

    match run_chat(&state, &req.model, prompt, temperature, top_k, top_p, max_tokens).await {
        Ok(gen) => {
            let created = chrono::Utc::now().timestamp();
            let resp = ChatCompletionResponse {
                id: format!("chatcmpl-{}", created),
                object: "chat.completion",
                created,
                model: req.model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: gen.response,
                    },
                    finish_reason: if (gen.eval_count as usize) >= max_tokens {
                        "length"
                    } else {
                        "stop"
                    },
                }],
                usage: Usage {
                    prompt_tokens: gen.prompt_eval_count,
                    completion_tokens: gen.eval_count,
                    total_tokens: gen.prompt_eval_count + gen.eval_count,
                },
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => openai_error(&e, StatusCode::INTERNAL_SERVER_ERROR).into_response(),
    }
}

/// GET /v1/models — list models discoverable in models_dir (OpenAI schema).
pub async fn list_models_handler(State(state): State<AppState>) -> impl IntoResponse {
    let mut ids: Vec<String> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(state.models_dir.as_ref()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.join("config.json").exists() {
                    if let Some(n) = path.file_name().and_then(|n| n.to_str()) {
                        ids.push(n.to_string());
                    }
                }
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if matches!(ext, "gguf" | "iara" | "dna") {
                    if let Some(n) = path.file_stem().and_then(|n| n.to_str()) {
                        ids.push(n.to_string());
                    }
                }
            }
        }
    }

    ids.sort();
    ids.dedup();

    let created = chrono::Utc::now().timestamp();
    let data: Vec<serde_json::Value> = ids
        .into_iter()
        .map(|id| {
            serde_json::json!({
                "id": id,
                "object": "model",
                "created": created,
                "owned_by": "iara"
            })
        })
        .collect();

    Json(serde_json::json!({ "object": "list", "data": data }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal
// ─────────────────────────────────────────────────────────────────────────────

/// Render OpenAI `messages` into a Qwen ChatML prompt. A default system prompt
/// is injected when the caller did not provide one. The result intentionally
/// contains `<|im_start|>` so the downstream formatter does not double-wrap it.
fn messages_to_chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();

    if !messages.iter().any(|m| m.role == "system") {
        out.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    }

    for m in messages {
        let role = match m.role.as_str() {
            "system" | "user" | "assistant" => m.role.as_str(),
            _ => "user",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(m.content.trim());
        out.push_str("<|im_end|>\n");
    }

    out.push_str("<|im_start|>assistant\n");
    out
}

/// Run a generation through the same dispatch as `/api/generate`:
/// llama.cpp batch server first (GGUF), native CausalLM otherwise.
async fn run_chat(
    state: &AppState,
    model_name: &str,
    prompt: String,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    max_tokens: usize,
) -> Result<GenerateResponse, String> {
    let start = std::time::Instant::now();

    // ── Fast path: llama.cpp batch server (GGUF models) ──
    {
        let batch_guard = state.batch_handle.lock().await;
        if let Some(handle) = batch_guard.as_ref() {
            let params = crate::llama::GenerateParams {
                temperature,
                top_k: top_k as i32,
                top_p,
                max_tokens,
            };
            // Release the guard before awaiting so other requests can submit too.
            let handle_clone = handle.clone();
            drop(batch_guard);

            let formatted = crate::tokenizer::format_for_instruct(&prompt);
            match handle_clone.submit(formatted, params).await {
                Ok(result) => {
                    let total_ms = start.elapsed().as_millis();
                    let tok_per_sec = if result.eval_ms > 0 {
                        result.generated_tokens as f64 * 1000.0 / result.eval_ms as f64
                    } else {
                        0.0
                    };
                    return Ok(GenerateResponse {
                        model: model_name.to_string(),
                        created_at: chrono::Utc::now().to_rfc3339(),
                        response: result.text,
                        done: true,
                        context: None,
                        total_duration: total_ms,
                        prompt_eval_count: result.prompt_tokens as u32,
                        eval_count: result.generated_tokens as u32,
                        eval_duration: result.eval_ms,
                        tokens_per_second: tok_per_sec,
                    });
                }
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    // ── Native CausalLM path (safetensors / .iara / .dna) ──
    let req = GenerateRequest {
        model: model_name.to_string(),
        prompt,
        stream: false,
        temperature,
        top_k,
        top_p,
        max_tokens,
        budget: None,
    };
    generate_internal(state, &req).await.map_err(|e| e.to_string())
}

fn openai_error(message: &str, code: StatusCode) -> (StatusCode, Json<serde_json::Value>) {
    (
        code,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": serde_json::Value::Null,
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.to_string(), content: content.to_string() }
    }

    #[test]
    fn injects_default_system_when_absent() {
        let p = messages_to_chatml(&[msg("user", "hi")]);
        assert!(p.starts_with("<|im_start|>system\nYou are a helpful assistant."));
        assert!(p.ends_with("<|im_start|>assistant\n"));
        assert!(p.contains("<|im_start|>user\nhi<|im_end|>"));
    }

    #[test]
    fn keeps_caller_system_no_double_inject() {
        let p = messages_to_chatml(&[msg("system", "S"), msg("user", "hi")]);
        assert_eq!(p.matches("<|im_start|>system").count(), 1);
    }

    #[test]
    fn unknown_role_maps_to_user() {
        let p = messages_to_chatml(&[msg("tool", "x")]);
        assert!(p.contains("<|im_start|>user\nx<|im_end|>"));
    }

    #[test]
    fn multi_turn_order_preserved() {
        let p = messages_to_chatml(&[msg("user", "AAA"), msg("assistant", "BBB"), msg("user", "CCC")]);
        let ia = p.find("AAA").unwrap();
        let ib = p.find("BBB").unwrap();
        let ic = p.find("CCC").unwrap();
        assert!(ia < ib && ib < ic);
        assert!(p.contains("<|im_start|>assistant\nBBB<|im_end|>"));
    }

    #[test]
    fn chatml_passes_through_format_for_instruct_without_double_wrap() {
        // The whole multi-turn design relies on format_for_instruct detecting
        // the existing <|im_start|> and NOT re-wrapping. Lock that contract.
        let p = messages_to_chatml(&[msg("user", "hi")]);
        let wrapped = crate::tokenizer::format_for_instruct(&p);
        assert_eq!(p, wrapped, "format_for_instruct must not re-wrap ChatML");
    }

    #[test]
    fn sanitizers_reject_bad_values() {
        assert_eq!(sanitize_temperature(Some(f32::NAN)), 0.7);
        assert_eq!(sanitize_temperature(Some(-1.0)), 0.7);
        assert_eq!(sanitize_temperature(Some(0.0)), 0.0);
        assert_eq!(sanitize_top_p(Some(0.0)), 0.9);
        assert_eq!(sanitize_top_p(Some(1.5)), 0.9);
        assert_eq!(sanitize_top_p(Some(0.5)), 0.5);
    }
}
