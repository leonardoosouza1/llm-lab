// ============================================================================
// lib.rs — IARA Engine Public API
// ============================================================================
// Exposes internal modules for integration testing and external use.

pub mod inference;
pub mod lnn;
pub mod model;
pub mod storage;
pub mod tokenizer;
pub mod trace;

#[cfg(feature = "experimental")]
pub mod experimental;
