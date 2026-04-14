use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSnapshot {
    pub op_name: String,           // "gemm_bf16", "rms_norm", "rope", etc.
    pub layer_idx: Option<usize>,
    pub step: u64,                 // generation step / training step
    pub timestamp_us: u64,

    // Shape info
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,

    // Stats (computed via GPU reduction kernels, NOT cpu readback)
    pub stats: TensorStats,

    // Performance
    pub elapsed_us: u64,
    pub gflops: Option<f64>,
    pub bandwidth_gb_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TensorStats {
    pub mean: f32,
    pub variance: f32,
    pub min: f32,
    pub max: f32,
    pub l2_norm: f32,
    pub nan_count: u32,
    pub inf_count: u32,
    pub zero_count: u32,
    pub sparsity: f32,            // fraction of zeros
    pub histogram: Option<Vec<u32>>, // 64-bin histogram (optional, expensive)
}

impl TensorSnapshot {
    pub fn new(op_name: impl Into<String>, layer_idx: Option<usize>, step: u64) -> Self {
        Self {
            op_name: op_name.into(),
            layer_idx,
            step,
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            input_shapes: Vec::new(),
            output_shape: Vec::new(),
            stats: TensorStats::default(),
            elapsed_us: 0,
            gflops: None,
            bandwidth_gb_s: None,
        }
    }

    pub fn with_shapes(mut self, inputs: Vec<Vec<usize>>, output: Vec<usize>) -> Self {
        self.input_shapes = inputs;
        self.output_shape = output;
        self
    }

    pub fn with_timing(mut self, elapsed_us: u64) -> Self {
        self.elapsed_us = elapsed_us;
        self
    }

    pub fn with_stats(mut self, stats: TensorStats) -> Self {
        self.stats = stats;
        self
    }

    pub fn with_gflops(mut self, flops: f64) -> Self {
        self.gflops = Some(flops / 1e9);
        self
    }

    pub fn with_bandwidth(mut self, bytes: f64) -> Self {
        let secs = self.elapsed_us as f64 / 1e6;
        if secs > 0.0 {
            self.bandwidth_gb_s = Some(bytes / secs / 1e9);
        }
        self
    }

    pub fn has_anomaly(&self) -> bool {
        self.stats.nan_count > 0 || self.stats.inf_count > 0
    }
}
