use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowEvent {
    pub event_type: FlowEventType,
    pub step: u64,
    pub timestamp_us: u64,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowEventType {
    // File I/O
    WeightsLoaded { path: String, format: String, size_bytes: u64 },
    TokenizerLoaded { path: String, vocab_size: usize },
    CheckpointSaved { path: String, step: u64 },

    // Memory
    VramAllocated { tensor_name: String, size_bytes: u64, dtype: String },
    VramFreed { tensor_name: String, size_bytes: u64 },
    VramPeak { used_bytes: u64, total_bytes: u64 },

    // Compute
    KernelLaunched { name: String, grid: [u32; 3], block: [u32; 3], shared_mem: u32 },
    KernelCompleted { name: String, elapsed_us: u64 },

    // Model
    LayerStarted { layer_idx: usize },
    LayerCompleted { layer_idx: usize, elapsed_us: u64 },
    TokenGenerated { token_id: u32, token_str: String, confidence: f32, rank: u32 },
    GenerationStarted { prompt_tokens: usize },
    GenerationCompleted { total_tokens: usize, elapsed_ms: u64, tokens_per_sec: f64 },

    // Training
    ForwardPassStarted { batch_size: usize, seq_len: usize },
    ForwardPassCompleted { loss: f32, elapsed_ms: u64 },
    BackwardPassStarted,
    BackwardPassCompleted { grad_norm: f32, elapsed_ms: u64 },
    OptimizerStep { lr: f64, step: u64 },

    // Anomaly
    NanDetected { location: String, tensor_name: String },
    InfDetected { location: String, tensor_name: String },
    GradientExplosion { norm: f32, threshold: f32 },
}

impl FlowEvent {
    pub fn now(event_type: FlowEventType, step: u64) -> Self {
        Self {
            event_type,
            step,
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            details: String::new(),
        }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = details.into();
        self
    }
}
