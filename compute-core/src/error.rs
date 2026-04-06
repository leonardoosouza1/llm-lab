#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("Backend não disponível: {0}")]
    NotAvailable(String),
    #[error("Falha de alocação: {0}")]
    Alloc(String),
    #[error("Erro de kernel: {0}")]
    Kernel(String),
    #[error("Erro de GEMM: {0}")]
    Gemm(String),
    #[error("Erro de transferência: {0}")]
    Transfer(String),
    #[error("Tipo de tensor incompatível: esperado {expected}, recebido {got}")]
    WrongTensorType { expected: String, got: String },
    #[error("Operação não implementada: {0}")]
    NotImplemented(String),
}
