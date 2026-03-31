// ============================================================================
// acp/mod.rs — ACP Protocol Module
// ============================================================================
// Protocolo de Comunicação entre Agentes (Agent Communication Protocol) v1.
// Expõe endpoints HTTP REST para criar runs, controlar agentes e coletar eventos.
//
// NOTE: Requer `uuid` dependency em Cargo.toml:
//   uuid = { version = "1", features = ["v4", "serde"] }
// ============================================================================

pub mod handlers;
pub mod state;
pub mod types;

use axum::{
    routing::{get, post},
    Router,
};

use crate::api::handlers::AppState;

/// Cria router ACP com todos os endpoints v1
/// Usa AppState que inclui o acp_store
pub fn acp_router() -> Router<AppState> {
    Router::new()
        // Health check
        .route("/ping", get(handlers::ping))
        // Agent management
        .route("/agents", get(handlers::list_agents))
        .route("/agents/:name", get(handlers::get_agent))
        // Run management
        // NOTE: Handlers with State + Json extractors são comentados por enquanto (Axum 0.7 trait issues)
        // Fase 2 irá fazer refactor para usar wrapper structs ou boxing
        // .route("/runs", post(handlers::create_run))
        // .route("/runs/:run_id", get(handlers::get_run).post(handlers::resume_run))
        // .route("/runs/:run_id/cancel", post(handlers::cancel_run))
        // .route("/runs/:run_id/events", get(handlers::get_run_events))
}

/// Helper para criar AcpRunStore vazio
pub fn create_run_store() -> state::AcpRunStore {
    state::AcpRunStore::new()
}
