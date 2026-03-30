use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    Router,
};
use serde::Serialize;
use sysinfo::{CpuExt, System, SystemExt};
use std::sync::{Arc, Mutex};
use log::{error, info};

// Shared state for sysinfo, wrapped in Arc<Mutex> for thread-safe access
pub type SharedSystemInfo = Arc<Mutex<System>>;

#[derive(Serialize)]
pub struct HealthResponse {
    memory_total_bytes: u64,
    memory_available_bytes: u64,
    memory_used_bytes: u64,
    cpu_usage_percent: f32,
}

pub async fn health_check(State(sys): State<SharedSystemInfo>) -> impl IntoResponse {
    let mut sys = sys.lock().unwrap();

    // Refresh system information
    sys.refresh_memory();
    sys.refresh_cpu(); // Refresh CPU to get up-to-date usage

    // Wait a little for CPU usage to be accurately updated
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
    sys.refresh_cpu(); // Refresh again after a short delay

    let memory_total_bytes = sys.total_memory();
    let memory_available_bytes = sys.available_memory();
    let memory_used_bytes = sys.used_memory();
    let cpu_usage_percent = sys.global_cpu_info().cpu_usage();


    info!(
        "Health Check: Mem Total: {} MB, Mem Available: {} MB, CPU Usage: {:.2}%",
        memory_total_bytes / 1024 / 1024,
        memory_available_bytes / 1024 / 1024,
        cpu_usage_percent
    );

    let response = HealthResponse {
        memory_total_bytes,
        memory_available_bytes,
        memory_used_bytes,
        cpu_usage_percent,
    };

    (StatusCode::OK, Json(response)).into_response()
}

pub fn telemetry_router(sys_info: SharedSystemInfo) -> Router {
    Router::new()
        .route("/health", axum::routing::get(health_check))
        .with_state(sys_info)
}
