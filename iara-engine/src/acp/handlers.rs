// ============================================================================
// acp/handlers.rs — Handlers para ACP Protocol v1
// ============================================================================
// Implementa endpoints HTTP/REST para o protocolo ACP:
// - GET  /acp/v1/ping
// - GET  /acp/v1/agents
// - GET  /acp/v1/agents/:name
// - POST /acp/v1/runs                  (create & start)
// - GET  /acp/v1/runs/:run_id          (status)
// - POST /acp/v1/runs/:run_id          (resume)
// - POST /acp/v1/runs/:run_id/cancel   (cancel)
// - GET  /acp/v1/runs/:run_id/events   (list events)
// ============================================================================

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde_json::json;

use crate::acp::types::*;
use crate::acp::AcpAppState;
use crate::inference::forward_traced;
use crate::inference::SamplingParams;

/// Resposta de health check
#[derive(serde::Serialize)]
pub struct PingResponse {
    pub status: String,
    pub timestamp: String,
    pub version: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers ACP
// ─────────────────────────────────────────────────────────────────────────────

/// GET /acp/v1/ping — health check
pub async fn ping() -> Json<PingResponse> {
    Json(PingResponse {
        status: "ok".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: "1".to_string(),
    })
}

/// GET /acp/v1/agents — lista agentes disponíveis (um por modelo carregado)
pub async fn list_agents(
    State(state): State<AcpAppState>,
) -> impl IntoResponse {
    let guard = state.app.qwen_model.lock().await;
    let mut agents = vec![];

    if let Some(m) = guard.as_ref() {
        let name = format!(
            "qwen-{}-{}b",
            m.cfg.num_layers,
            m.cfg.hidden_size / 128 // rough approximation
        );
        let desc = format!(
            "Qwen LLM com {} camadas, {} hidden size, vocab {}",
            m.cfg.num_layers, m.cfg.hidden_size, m.cfg.vocab_size
        );

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("num_layers".to_string(), json!(m.cfg.num_layers));
        metadata.insert("hidden_size".to_string(), json!(m.cfg.hidden_size));
        metadata.insert("vocab_size".to_string(), json!(m.cfg.vocab_size));
        metadata.insert("num_heads".to_string(), json!(m.cfg.num_heads));
        metadata.insert("max_seq_len".to_string(), json!(m.cfg.max_seq_len));

        agents.push(AgentManifest {
            name,
            description: desc,
            supported_content_types: Some(vec!["text/plain".to_string(), "text/json".to_string()]),
            metadata: Some(metadata),
        });
    }

    drop(guard);

    if agents.is_empty() {
        return (StatusCode::OK, Json(json!({ "agents": [] }))).into_response();
    }

    (StatusCode::OK, Json(json!({ "agents": agents }))).into_response()
}

/// GET /acp/v1/agents/:name — retorna manifesto de um agente específico
pub async fn get_agent(
    State(state): State<AcpAppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let guard = state.app.qwen_model.lock().await;

    if let Some(m) = guard.as_ref() {
        let agent_name = format!(
            "qwen-{}-{}b",
            m.cfg.num_layers,
            m.cfg.hidden_size / 128
        );

        if agent_name == name {
            let desc = format!(
                "Qwen LLM com {} camadas, {} hidden size, vocab {}",
                m.cfg.num_layers, m.cfg.hidden_size, m.cfg.vocab_size
            );

            let mut metadata = std::collections::HashMap::new();
            metadata.insert("num_layers".to_string(), json!(m.cfg.num_layers));
            metadata.insert("hidden_size".to_string(), json!(m.cfg.hidden_size));
            metadata.insert("vocab_size".to_string(), json!(m.cfg.vocab_size));

            let manifest = AgentManifest {
                name: agent_name,
                description: desc,
                supported_content_types: Some(vec!["text/plain".to_string()]),
                metadata: Some(metadata),
            };

            return (StatusCode::OK, Json(manifest)).into_response();
        }
    }

    drop(guard);
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "error": format!("Agente '{}' não encontrado", name) })),
    )
        .into_response()
}

/// POST /acp/v1/runs — cria e executa uma run (modo sync)
pub async fn create_run(
    State(state): State<AcpAppState>,
    Json(req): Json<RunRequest>,
) -> impl IntoResponse {
    let app_state = &state.app;
    let acp_store = &state.acp_store;
    // Gera ID único para a run
    let run_id = uuid::Uuid::new_v4().to_string();

    // Valida se o agente existe
    {
        let guard = app_state.qwen_model.lock().await;
        if guard.is_none() {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": "Nenhum modelo carregado. Carregue um modelo primeiro."
                })),
            )
                .into_response();
        }
    }

    // Extrai texto do primeiro input
    let prompt_text = req
        .input
        .first()
        .and_then(|msg| msg.parts.first())
        .and_then(|part| {
            if part.content_type.starts_with("text/") {
                part.content.clone()
            } else {
                None
            }
        })
        .unwrap_or_default();

    if prompt_text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Prompt vazio no input" })),
        )
            .into_response();
    }

    // Extrai parâmetros de sampling da metadata
    let temperature = req
        .metadata
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;

    let top_k = req
        .metadata
        .get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(40) as usize;

    let top_p = req
        .metadata
        .get("top_p")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.9) as f32;

    let max_tokens = req
        .metadata
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as usize;

    // Cria resposta inicial
    let now = chrono::Utc::now();
    let mut response = RunResponse {
        run_id: run_id.clone(),
        agent_name: req.agent_name.clone(),
        status: RunStatus::Created,
        output: None,
        created_at: now.to_rfc3339(),
        started_at: None,
        completed_at: None,
        error: None,
    };

    // Valida modo
    if !["sync", "async", "stream"].contains(&req.mode.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("Modo inválido: {}. Use 'sync', 'async' ou 'stream'", req.mode)
            })),
        )
            .into_response();
    }

    // Para modo "sync" — executa imediatamente
    if req.mode == "sync" {
        match execute_inference(
            &app_state,
            &prompt_text,
            temperature,
            top_k,
            top_p,
            max_tokens,
        )
        .await
        {
            Ok(output_text) => {
                response.status = RunStatus::Completed;
                response.started_at = Some(now.to_rfc3339());
                response.completed_at = Some(chrono::Utc::now().to_rfc3339());
                response.output = Some(vec![AcpMessage {
                    role: "agent".to_string(),
                    parts: vec![AcpMessagePart {
                        name: None,
                        content_type: "text/plain".to_string(),
                        content: Some(output_text),
                        content_url: None,
                        content_encoding: None,
                    }],
                }]);

                // Armazena run
                acp_store.store_run(
                    run_id.clone(),
                    crate::acp::state::RunStore {
                        response: response.clone(),
                        events: vec![],
                    },
                ).await;

                return (StatusCode::OK, Json(response)).into_response();
            }
            Err(e) => {
                response.status = RunStatus::Failed;
                response.error = Some(e.to_string());

                acp_store.store_run(
                    run_id.clone(),
                    crate::acp::state::RunStore {
                        response: response.clone(),
                        events: vec![],
                    },
                ).await;

                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(response),
                )
                    .into_response();
            }
        }
    }

    // Para modo "async" — retorna imediatamente com status InProgress
    response.status = RunStatus::InProgress;
    response.started_at = Some(chrono::Utc::now().to_rfc3339());

    acp_store.store_run(
        run_id.clone(),
        crate::acp::state::RunStore {
            response: response.clone(),
            events: vec![],
        },
    ).await;

    // Spawn background task para execução async
    let acp_store_clone = acp_store.clone();
    let app_state_clone = app_state.clone();
    let run_id_clone = run_id.clone();

    tokio::spawn(async move {
        let result_text = execute_inference(&app_state_clone, &prompt_text, temperature, top_k, top_p, max_tokens)
            .await
            .map_err(|e| e.to_string()); // Converte erro para String (Send-safe)

        match result_text {
            Ok(output_text) => {
                let _ = acp_store_clone.set_status(&run_id_clone, RunStatus::Completed).await;

                let mut store = acp_store_clone
                    .get_run(&run_id_clone)
                    .await
                    .unwrap_or_default();

                store.response.output = Some(vec![AcpMessage {
                    role: "agent".to_string(),
                    parts: vec![AcpMessagePart {
                        name: None,
                        content_type: "text/plain".to_string(),
                        content: Some(output_text),
                        content_url: None,
                        content_encoding: None,
                    }],
                }]);
                store.response.completed_at = Some(chrono::Utc::now().to_rfc3339());

                let _ = acp_store_clone.store_run(run_id_clone, store).await;
            }
            Err(error_msg) => {
                let _ = acp_store_clone.set_status(&run_id_clone, RunStatus::Failed).await;

                let mut store = acp_store_clone
                    .get_run(&run_id_clone)
                    .await
                    .unwrap_or_default();

                store.response.error = Some(error_msg);
                store.response.completed_at = Some(chrono::Utc::now().to_rfc3339());

                let _ = acp_store_clone.store_run(run_id_clone, store).await;
            }
        }
    });

    (StatusCode::ACCEPTED, Json(response)).into_response()
}

/// GET /acp/v1/runs/:run_id — retorna status e output de uma run
pub async fn get_run(
    State(state): State<AcpAppState>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    let acp_store = &state.acp_store;
    match acp_store.get_run(&run_id).await {
        Some(store) => (StatusCode::OK, Json(store.response)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("Run '{}' não encontrada", run_id) })),
        )
            .into_response(),
    }
}

/// POST /acp/v1/runs/:run_id — resume uma run pausada (não implementado ainda)
pub async fn resume_run(
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "Resume não implementado ainda",
            "run_id": run_id
        })),
    )
        .into_response()
}

/// POST /acp/v1/runs/:run_id/cancel — cancela uma run
pub async fn cancel_run(
    State(state): State<AcpAppState>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    let acp_store = &state.acp_store;
    match acp_store.get_run(&run_id).await {
        Some(mut store) => {
            if store.response.status == RunStatus::InProgress {
                store.response.status = RunStatus::Cancelled;
                store.response.completed_at = Some(chrono::Utc::now().to_rfc3339());

                let _ = acp_store.store_run(run_id, store.clone()).await;

                return (StatusCode::OK, Json(store.response)).into_response();
            }

            (
                StatusCode::CONFLICT,
                Json(json!({
                    "error": "Apenas runs em progresso podem ser canceladas"
                })),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("Run '{}' não encontrada", run_id) })),
        )
            .into_response(),
    }
}

/// GET /acp/v1/runs/:run_id/events — lista eventos de uma run
pub async fn get_run_events(
    State(state): State<AcpAppState>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    let acp_store = &state.acp_store;
    let events = acp_store.get_events(&run_id).await;
    (StatusCode::OK, Json(json!({ "events": events }))).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper function — Executa inferência replicando generate_internal
// ─────────────────────────────────────────────────────────────────────────────

async fn execute_inference(
    app_state: &crate::api::handlers::AppState,
    prompt: &str,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    max_tokens: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    // Validação de modelo carregado
    {
        let guard = app_state.qwen_model.lock().await;
        if guard.is_none() {
            return Err("Modelo não carregado".into());
        }
    }

    let sampling = SamplingParams {
        temperature,
        top_k,
        top_p,
    };

    // Tokeniza prompt
    let tokens = crate::tokenizer::tokenize(prompt);
    let prompt_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();

    if prompt_ids.is_empty() {
        return Err("Prompt vazio após tokenização".into());
    }

    let mut generated_text = String::new();

    const EOS_TOKEN: u32 = 151645;

    // Reset KV cache — evita estado antigo
    {
        let mut guard = app_state.qwen_model.lock().await;
        let model = guard.as_mut().ok_or("Model not loaded")?;
        model.reset_cache();
    }

    // Prefill: processa tokens 0..n-2 apenas para popular KV cache
    if prompt_ids.len() > 1 {
        let mut model_guard = app_state.qwen_model.lock().await;
        let model = model_guard.as_mut().ok_or("Model not loaded")?;
        for &tok in &prompt_ids[..prompt_ids.len() - 1] {
            model
                .forward(&[tok])
                .map_err(|e| format!("Prefill error: {}", e))?;
        }
    }

    // Decode: começa com último token do prompt, gera até max_tokens novos
    let mut decode_token = *prompt_ids.last().ok_or("Prompt vazio")?;

    for _step in 0..max_tokens {
        let gen_step = {
            let mut model_guard = app_state.qwen_model.lock().await;
            let model = model_guard.as_mut().ok_or("Model not loaded")?;
            forward_traced(model, &[decode_token], &sampling)
                .map_err(|e| format!("Forward pass error: {}", e))?
        };

        decode_token = gen_step.token_id;
        generated_text.push_str(&gen_step.token_str);

        // Stop conditions: EOS ou token especial
        if decode_token == EOS_TOKEN || decode_token == 151643 {
            break;
        }
    }

    Ok(generated_text)
}
