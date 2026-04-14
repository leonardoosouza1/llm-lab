use parking_lot::RwLock;
use std::sync::Arc;
use crate::store::ObserveStore;
use crate::{TensorSnapshot, NeuronSnapshot, FlowEvent};

pub static GLOBAL_COLLECTOR: std::sync::LazyLock<Collector> =
    std::sync::LazyLock::new(|| Collector::new());

pub struct Collector {
    enabled: std::sync::atomic::AtomicBool,
    store: RwLock<Option<Arc<ObserveStore>>>,

    // In-memory ring buffers for live dashboard
    recent_tensors: RwLock<Vec<TensorSnapshot>>,
    recent_neurons: RwLock<Vec<NeuronSnapshot>>,
    recent_flow: RwLock<Vec<FlowEvent>>,

    // Counters
    tensor_count: std::sync::atomic::AtomicU64,
    neuron_count: std::sync::atomic::AtomicU64,
    flow_count: std::sync::atomic::AtomicU64,
    anomaly_count: std::sync::atomic::AtomicU64,
}

const RING_BUFFER_SIZE: usize = 1024;

impl Collector {
    pub fn new() -> Self {
        Self {
            enabled: std::sync::atomic::AtomicBool::new(false),
            store: RwLock::new(None),
            recent_tensors: RwLock::new(Vec::with_capacity(RING_BUFFER_SIZE)),
            recent_neurons: RwLock::new(Vec::with_capacity(RING_BUFFER_SIZE)),
            recent_flow: RwLock::new(Vec::with_capacity(RING_BUFFER_SIZE)),
            tensor_count: std::sync::atomic::AtomicU64::new(0),
            neuron_count: std::sync::atomic::AtomicU64::new(0),
            flow_count: std::sync::atomic::AtomicU64::new(0),
            anomaly_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Enable observation and start recording to disk
    pub fn enable(&self, output_dir: impl Into<std::path::PathBuf>) {
        let store = Arc::new(ObserveStore::new(output_dir));
        *self.store.write() = Some(store);
        self.enabled.store(true, std::sync::atomic::Ordering::SeqCst);
        log::info!("[observe] Collector enabled");
    }

    /// Disable observation
    pub fn disable(&self) {
        self.enabled.store(false, std::sync::atomic::Ordering::SeqCst);
        if let Some(ref store) = *self.store.read() {
            store.flush();
        }
        log::info!("[observe] Collector disabled");
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn record_tensor(&self, snap: TensorSnapshot) {
        if !self.is_enabled() { return; }

        if snap.has_anomaly() {
            self.anomaly_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            log::warn!("[observe] ANOMALY in {}: {} NaN, {} Inf",
                snap.op_name, snap.stats.nan_count, snap.stats.inf_count);
        }

        self.tensor_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Write to disk
        if let Some(ref store) = *self.store.read() {
            store.record_tensor(&snap);
        }

        // Ring buffer for live dashboard
        let mut buf = self.recent_tensors.write();
        if buf.len() >= RING_BUFFER_SIZE {
            buf.remove(0);
        }
        buf.push(snap);
    }

    pub fn record_neuron(&self, snap: NeuronSnapshot) {
        if !self.is_enabled() { return; }

        self.neuron_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if let Some(ref store) = *self.store.read() {
            store.record_neuron(&snap);
        }

        let mut buf = self.recent_neurons.write();
        if buf.len() >= RING_BUFFER_SIZE {
            buf.remove(0);
        }
        buf.push(snap);
    }

    pub fn record_flow(&self, event: FlowEvent) {
        if !self.is_enabled() { return; }

        self.flow_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if let Some(ref store) = *self.store.read() {
            store.record_flow(&event);
        }

        let mut buf = self.recent_flow.write();
        if buf.len() >= RING_BUFFER_SIZE {
            buf.remove(0);
        }
        buf.push(event);
    }

    // Live dashboard queries
    pub fn recent_tensors(&self) -> Vec<TensorSnapshot> {
        self.recent_tensors.read().clone()
    }

    pub fn recent_neurons(&self) -> Vec<NeuronSnapshot> {
        self.recent_neurons.read().clone()
    }

    pub fn recent_flow(&self) -> Vec<FlowEvent> {
        self.recent_flow.read().clone()
    }

    pub fn stats(&self) -> CollectorStats {
        CollectorStats {
            enabled: self.is_enabled(),
            tensor_count: self.tensor_count.load(std::sync::atomic::Ordering::Relaxed),
            neuron_count: self.neuron_count.load(std::sync::atomic::Ordering::Relaxed),
            flow_count: self.flow_count.load(std::sync::atomic::Ordering::Relaxed),
            anomaly_count: self.anomaly_count.load(std::sync::atomic::Ordering::Relaxed),
            session_id: self.store.read().as_ref().map(|s| s.session_id().to_string()),
        }
    }
}

impl Default for Collector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectorStats {
    pub enabled: bool,
    pub tensor_count: u64,
    pub neuron_count: u64,
    pub flow_count: u64,
    pub anomaly_count: u64,
    pub session_id: Option<String>,
}

/// Convenience macros for zero-cost observation when disabled
#[macro_export]
macro_rules! observe_tensor {
    ($snap:expr) => {
        $crate::GLOBAL_COLLECTOR.record_tensor($snap)
    };
}

#[macro_export]
macro_rules! observe_neuron {
    ($snap:expr) => {
        $crate::GLOBAL_COLLECTOR.record_neuron($snap)
    };
}

#[macro_export]
macro_rules! observe_flow {
    ($event:expr) => {
        $crate::GLOBAL_COLLECTOR.record_flow($event)
    };
}
