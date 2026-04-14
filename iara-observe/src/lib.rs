pub mod tensor_obs;
pub mod neuron_obs;
pub mod flow_obs;
pub mod store;
pub mod collector;

pub use collector::{Collector, GLOBAL_COLLECTOR};
pub use tensor_obs::TensorSnapshot;
pub use neuron_obs::NeuronSnapshot;
pub use flow_obs::{FlowEvent, FlowEventType};

/// Compute Kuramoto phase coherence from a hidden state vector.
/// Interprets pairs (h[2k], h[2k+1]) as complex numbers, returns R in [0,1].
pub fn phase_coherence_from(hidden: &[f32]) -> f32 {
    let n = hidden.len() / 2;
    if n == 0 {
        return 0.0;
    }
    let mut sum_cos = 0.0f32;
    let mut sum_sin = 0.0f32;
    for i in 0..n {
        let re = hidden[2 * i];
        let im = hidden[2 * i + 1];
        let mag = (re * re + im * im).sqrt();
        if mag > 1e-8 {
            sum_cos += re / mag;
            sum_sin += im / mag;
        }
    }
    ((sum_cos / n as f32).powi(2) + (sum_sin / n as f32).powi(2)).sqrt()
}
