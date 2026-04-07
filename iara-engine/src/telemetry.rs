use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    Router,
};
use serde::Serialize;
use sysinfo::System;
use std::sync::Arc;
use tokio::sync::Mutex;
use log::info;
use crate::api::handlers::AppState;

// Shared state for sysinfo, wrapped in Arc<Mutex> for thread-safe access
pub type SharedSystemInfo = Arc<Mutex<System>>;

#[derive(Serialize)]
pub struct HealthResponse {
    // ── Model status ─────────────────────────────────────────────
    pub status: String,
    pub model: String,
    pub version: String,
    pub timestamp: String,
    pub model_loading: bool,
    pub model_vram_gb: f64,
    pub vram_free_gb: Option<f64>,
    pub vram_total_gb: Option<f64>,
    // ── System info ──────────────────────────────────────────────
    pub memory_total_bytes: u64,
    pub memory_available_bytes: u64,
    pub memory_used_bytes: u64,
    pub cpu_usage_percent: f32,
}

pub async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    // ── System info ──────────────────────────────────────────────
    let mut sys = state.shared_sys_info.lock().await;
    sys.refresh_memory();
    sys.refresh_cpu();
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
    sys.refresh_cpu();

    let memory_total_bytes = sys.total_memory();
    let memory_available_bytes = sys.available_memory();
    let memory_used_bytes = sys.used_memory();
    let cpu_usage_percent = sys.global_cpu_info().cpu_usage();
    drop(sys); // libera lock do sysinfo antes de pegar lock do modelo

    info!(
        "Health Check: Mem Total: {} MB, Mem Available: {} MB, CPU Usage: {:.2}%",
        memory_total_bytes / 1024 / 1024,
        memory_available_bytes / 1024 / 1024,
        cpu_usage_percent
    );

    // ── Model status ─────────────────────────────────────────────
    let ready = state.model_ready.load(std::sync::atomic::Ordering::Relaxed);
    let loading = state.model_loading.load(std::sync::atomic::Ordering::Relaxed);
    let status = if ready { "ok" } else if loading { "loading" } else { "idle" };

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

    let (vram_free_gb, vram_total_gb) = match state.backend.mem_info() {
        Some((free, total)) => (Some(free as f64 / 1e9), Some(total as f64 / 1e9)),
        None => (None, None),
    };

    let response = HealthResponse {
        status: status.to_string(),
        model: model_name,
        version: "2.0.0".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        model_loading: loading,
        model_vram_gb,
        vram_free_gb,
        vram_total_gb,
        memory_total_bytes,
        memory_available_bytes,
        memory_used_bytes,
        cpu_usage_percent,
    };

    (StatusCode::OK, Json(response)).into_response()
}

pub fn telemetry_router() -> Router<AppState> {
    Router::new()
        .route("/health", axum::routing::get(health_check))
}
