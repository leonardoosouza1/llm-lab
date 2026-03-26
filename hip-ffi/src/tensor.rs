//! GpuTensor — tensor nativo na GPU sem dependência de framework.
//! Suporta f32 e bf16 (via crate `half`).
//! Shape arbitrário, dados contíguos row-major na GPU.

use crate::safe::{DeviceBuffer, HipError};

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,
    BF16,
    F16,
}

impl DType {
    pub fn bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 | DType::F16 => 2,
        }
    }
}

/// Tensor tipado em f32 residente na GPU.
/// Para bf16/f16 use GpuTensorBF16 ou converta explicitamente.
pub struct GpuTensor {
    pub buf: DeviceBuffer<f32>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl GpuTensor {
    /// Aloca tensor zerado na GPU com o shape dado.
    pub fn zeros(shape: &[usize]) -> Result<Self, HipError> {
        let n = shape.iter().product::<usize>().max(1);
        let buf = DeviceBuffer::<f32>::alloc(n)?;
        Ok(Self {
            buf,
            shape: shape.to_vec(),
            dtype: DType::F32,
        })
    }

    /// Cria tensor a partir de slice host.
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Result<Self, HipError> {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data len não bate com shape"
        );
        let buf = DeviceBuffer::from_slice(data)?;
        Ok(Self {
            buf,
            shape: shape.to_vec(),
            dtype: DType::F32,
        })
    }

    /// Copia tensor de volta para o host.
    pub fn to_host(&self) -> Result<Vec<f32>, HipError> {
        self.buf.to_host()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Interpreta shape como [rows, cols] para operações de matmul.
    pub fn rows(&self) -> usize {
        assert!(self.ndim() >= 2);
        self.shape[self.ndim() - 2]
    }

    pub fn cols(&self) -> usize {
        assert!(self.ndim() >= 1);
        self.shape[self.ndim() - 1]
    }

    /// Retorna ponteiro raw device para uso em kernels ou FFI rocBLAS.
    pub fn dev_ptr(&self) -> *const f32 {
        self.buf.as_ptr()
    }

    pub fn dev_ptr_mut(&mut self) -> *mut f32 {
        self.buf.as_mut_ptr()
    }
}
