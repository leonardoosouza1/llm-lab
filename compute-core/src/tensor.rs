use std::any::Any;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16, // armazenado como u16 (bits de BF16)
}

/// Tensor type-erased. Cada backend cria tensores com seu próprio buffer interno,
/// envolvido em Arc<dyn Any + Send + Sync>. O backend faz downcast quando precisa.
pub struct AnyTensor {
    pub inner: Arc<dyn Any + Send + Sync>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl AnyTensor {
    pub fn new(
        inner: Arc<dyn Any + Send + Sync>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        Self { inner, shape, dtype }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Downcast para o tipo de buffer específico do backend.
    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}

// AnyTensor não é Clone por padrão — recursos GPU não são clone-áveis.
// Backends que precisam de cópia devem implementar operações de cópia explícitas.
