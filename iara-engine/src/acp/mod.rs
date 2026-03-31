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
use state::AcpRunStore;
use std::sync::Arc;

/// Combined state: AppState + Arc<AcpRunStore>
pub struct AcpAppState {
    pub app: AppState,
    pub acp_store: Arc<AcpRunStore>,
}

impl Clone for AcpAppState {
    fn clone(&self) -> Self {
        AcpAppState {
            app: self.app.clone(),
            acp_store: self.acp_store.clone(),
        }
    }
}

/// Cria router ACP com todos os endpoints v1
/// Retorna um Router que pode ser usado com .nest()
pub fn acp_router(app_state: AppState, acp_store: AcpRunStore) -> Router<AcpAppState> {
    // Armazena AcpRunStore em um Arc para compartilhamento
    let acp_store_arc = Arc::new(acp_store);

    let combined = AcpAppState {
        app: app_state,
        acp_store: acp_store_arc,
    };

    Router::new()
        // Health check
        .route("/ping", get(handlers::ping))
        // Agent management
        .route("/agents", get(handlers::list_agents))
        .route("/agents/:name", get(handlers::get_agent))
        // Run management
        .route("/runs", post(handlers::create_run))
        .route(
            "/runs/:run_id",
            get(handlers::get_run).post(handlers::resume_run),
        )
        .route("/runs/:run_id/cancel", post(handlers::cancel_run))
        .route("/runs/:run_id/events", get(handlers::get_run_events))
        .with_state(combined)
}

/// Helper para criar AcpRunStore vazio
pub fn create_run_store() -> AcpRunStore {
    AcpRunStore::new()
}
