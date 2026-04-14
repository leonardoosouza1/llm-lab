pub mod model;
pub mod train;

// Re-export commonly used types for convenience
pub use model::weights::ModelWeights;
pub use model::lora::LoraAdapter;
pub use model::forward::{forward, ForwardResult};
pub use train::trainer::{Trainer, TrainConfig};
pub use train::dataset::TextDataset;
pub use train::loss::cross_entropy_loss;
