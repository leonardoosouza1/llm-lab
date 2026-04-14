//! IARA Ternary Navigator — módulo de estruturas reutilizáveis
//!
//! Contém os blocos fundamentais do pipeline de navegação:
//!   - `bi`: Block Influence calculator
//!   - `lsh`: SimHash tables para lookup sub-linear
//!   - `ternary`: quantização e operações ternárias
//!   - `router`: grafo de super layers + UCS/A*

pub mod bi;
pub mod lsh;
pub mod ternary;
pub mod router;
pub mod route_cache;
pub mod ttt;
pub mod gpu_kernel;
