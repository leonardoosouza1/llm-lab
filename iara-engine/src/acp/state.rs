// ============================================================================
// acp/state.rs — ACP Run State Management
// ============================================================================
// Armazena e gerencia o estado de execução de runs ACP.
// Usa Arc<Mutex<HashMap>> para acesso thread-safe.
// ============================================================================

use crate::acp::types::{RunResponse, RunEvent, RunStatus};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Armazena dados de uma run (resposta + eventos)
#[derive(Debug, Clone)]
pub struct RunStore {
    /// Resposta principal da run
    pub response: RunResponse,

    /// Eventos gerados durante execução
    pub events: Vec<RunEvent>,
}

impl Default for RunStore {
    fn default() -> Self {
        RunStore {
            response: RunResponse {
                run_id: String::new(),
                agent_name: String::new(),
                status: crate::acp::types::RunStatus::Created,
                output: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                started_at: None,
                completed_at: None,
                error: None,
            },
            events: vec![],
        }
    }
}

/// Gerenciador global de runs — dicionário de run_id → RunStore
#[derive(Clone)]
pub struct AcpRunStore {
    /// HashMap thread-safe: run_id → RunStore
    pub runs: Arc<Mutex<HashMap<String, RunStore>>>,
}

impl AcpRunStore {
    /// Cria um novo armazenador de runs vazio
    pub fn new() -> Self {
        AcpRunStore {
            runs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Armazena uma nova run
    pub async fn store_run(&self, run_id: String, store: RunStore) {
        let mut runs = self.runs.lock().await;
        runs.insert(run_id, store);
    }

    /// Recupera uma run pelo ID
    pub async fn get_run(&self, run_id: &str) -> Option<RunStore> {
        let runs = self.runs.lock().await;
        runs.get(run_id).cloned()
    }

    /// Atualiza o status de uma run
    pub async fn set_status(&self, run_id: &str, status: RunStatus) {
        let mut runs = self.runs.lock().await;
        if let Some(store) = runs.get_mut(run_id) {
            store.response.status = status;
        }
    }

    /// Lista todos os eventos de uma run
    pub async fn get_events(&self, run_id: &str) -> Vec<RunEvent> {
        let runs = self.runs.lock().await;
        runs.get(run_id).map(|s| s.events.clone()).unwrap_or_default()
    }

}

impl Default for AcpRunStore {
    fn default() -> Self {
        Self::new()
    }
}
