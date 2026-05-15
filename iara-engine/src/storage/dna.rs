//! DNA V4 — Universal Tensor Format (SVD low-rank factorization).
//!
//! Layout: sequence of TLV chunks. Each chunk:
//!   magic: [u8; 4]  "DNA4" (file header) | "TSVD" | "TRAW" (tensor chunks)
//!   name_len: u16 LE
//!   name:     [u8; name_len]
//!   payload_len: u64 LE
//!   payload:  [u8; payload_len]
//!
//! File header:
//!   "DNA4" | 4B version_u32 | 4B tensor_count_u32
//!
//! TSVD payload:
//!   rows: u32 LE
//!   cols: u32 LE
//!   rank: u32 LE
//!   dtype: u32 LE (0=BF16, 1=F16, 2=INT8 — only BF16 supported in v0)
//!   A: [rows * rank] dtype  (row-major)
//!   B: [rank * cols] dtype  (row-major)
//!
//! TRAW payload (non-factorized tensors — norm, bias, embed, lm_head):
//!   rank_dim_count: u32 LE  (typically 1 or 2)
//!   dims: [u32 LE; rank_dim_count]
//!   dtype: u32 LE
//!   data: raw bytes
//!
//! Design goals:
//!   - Never reconstruct W in RAM/VRAM — only A, B upload to device
//!   - Per-tensor rank — embed/lm_head use higher rank than ffn_down

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use compute_core::{AnyTensor, ComputeBackend};
use compute_core::dna_format::{
    DNA_MAGIC, CHUNK_TSVD, CHUNK_TRAW, CHUNK_TBLK,
    DTYPE_BF16, DTYPE_Q4_0, DTYPE_Q4_K, DTYPE_Q6_K, DTYPE_Q8_0,
    Q4_0_BLOCK, Q4_0_BYTES_PER_BLOCK,
    dequantize_q4_0,
    read_magic, read_u32, read_u64, read_name, read_u16_vec, read_f32_vec,
};
use std::io::Read;

// Re-export writer so callers can build `.dna` files without reaching into compute-core.
pub use compute_core::dna_format::DnaWriter;

#[derive(Debug, thiserror::Error)]
pub enum DnaError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("bad magic: expected {expected:?}, got {got:?}")]
    BadMagic { expected: [u8; 4], got: [u8; 4] },
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(u32),
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("truncated payload: name={name} expected={expected} got={got}")]
    Truncated { name: String, expected: usize, got: usize },
    #[error("unknown chunk magic: {0:?}")]
    UnknownChunk([u8; 4]),
}

pub type DnaResult<T> = Result<T, DnaError>;

/// Device-resident tensor variant: full or SVD-factorized.
pub enum DnaTensor {
    /// W stored as-is (norm, bias, small weights where SVD hurts).
    Full {
        tensor: AnyTensor,
        shape: Vec<usize>,
    },
    /// W ≈ A·B never materialized. Forward uses gemv_svd_bf16.
    Svd {
        a: AnyTensor,
        b: AnyTensor,
        rows: usize,
        cols: usize,
        rank: usize,
    },
    /// DNA-V5: ggml block-quantized tensor (Q4_K, Q6_K, Q8_0). Raw bytes uploaded
    /// to device; forward uses the existing gemv_q4k / gemv_q6k kernels (same
    /// path as the GGUF loader).
    Block {
        data: AnyTensor,
        shape: Vec<usize>,
        dtype: u32, // DTYPE_Q4_K | DTYPE_Q6_K | DTYPE_Q8_0
    },
}

impl DnaTensor {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            DnaTensor::Full { shape, .. } => shape.clone(),
            DnaTensor::Svd { rows, cols, .. } => vec![*rows, *cols],
            DnaTensor::Block { shape, .. } => shape.clone(),
        }
    }

    /// Max rank across SVD tensors (for scratch buffer sizing).
    pub fn rank(&self) -> Option<usize> {
        match self {
            DnaTensor::Svd { rank, .. } => Some(*rank),
            _ => None,
        }
    }
}

/// Loaded DNA V4 model — tensor registry keyed by name.
pub struct DnaModel {
    pub tensors: HashMap<String, DnaTensor>,
    pub max_rank: usize,
    pub version: u32,
}

impl DnaModel {
    /// Parse `.dna` file and upload every tensor to `backend`.
    pub fn load(path: impl AsRef<Path>, backend: &Arc<dyn ComputeBackend>) -> DnaResult<Self> {
        let mut f = File::open(path)?;

        // File header: DNA4 | version | tensor_count
        let magic = read_magic(&mut f)?;
        if &magic != DNA_MAGIC {
            return Err(DnaError::BadMagic { expected: *DNA_MAGIC, got: magic });
        }
        let version = read_u32(&mut f)?;
        if version != 1 && version != 5 {
            return Err(DnaError::UnsupportedVersion(version));
        }
        let tensor_count = read_u32(&mut f)? as usize;

        let mut tensors = HashMap::with_capacity(tensor_count);
        let mut max_rank = 0usize;

        for _ in 0..tensor_count {
            let chunk_magic = read_magic(&mut f)?;
            let name = read_name(&mut f)?;
            let payload_len = read_u64(&mut f)? as usize;
            let payload_start = f.stream_position()?;

            let tensor = match &chunk_magic {
                CHUNK_TSVD => {
                    let t = parse_tsvd(&mut f, &name, payload_len, backend)?;
                    if let DnaTensor::Svd { rank, .. } = &t {
                        max_rank = max_rank.max(*rank);
                    }
                    t
                }
                CHUNK_TRAW => parse_traw(&mut f, &name, payload_len, backend)?,
                CHUNK_TBLK => parse_tblk(&mut f, &name, payload_len, backend)?,
                _ => return Err(DnaError::UnknownChunk(chunk_magic)),
            };

            // Ensure we advanced exactly payload_len — catches parser bugs.
            let consumed = (f.stream_position()? - payload_start) as usize;
            if consumed != payload_len {
                return Err(DnaError::Truncated {
                    name,
                    expected: payload_len,
                    got: consumed,
                });
            }

            tensors.insert(name, tensor);
        }

        Ok(DnaModel { tensors, max_rank, version })
    }
}

fn parse_tsvd(
    f: &mut File,
    name: &str,
    payload_len: usize,
    backend: &Arc<dyn ComputeBackend>,
) -> DnaResult<DnaTensor> {
    let rows = read_u32(f)? as usize;
    let cols = read_u32(f)? as usize;
    let rank = read_u32(f)? as usize;
    let dtype = read_u32(f)?;
    let header_bytes = 16;

    match dtype {
        DTYPE_BF16 => {
            let a_bytes = rows * rank * 2;
            let b_bytes = rank * cols * 2;
            let expected = header_bytes + a_bytes + b_bytes;
            if expected != payload_len {
                return Err(DnaError::Truncated {
                    name: name.into(), expected: payload_len, got: expected,
                });
            }
            let a_u16 = read_u16_vec(f, rows * rank)?;
            let b_u16 = read_u16_vec(f, rank * cols)?;
            let a = backend.upload_bf16(&a_u16, &[rows, rank]);
            let b = backend.upload_bf16(&b_u16, &[rank, cols]);
            Ok(DnaTensor::Svd { a, b, rows, cols, rank })
        }
        DTYPE_Q4_0 => {
            // Dequant at load → bf16 → upload. Reuses existing gemv_bf16 path.
            // File is ~3.5× smaller; VRAM matches bf16 SVD.
            let a_elems = rows * rank;
            let b_elems = rank * cols;
            if a_elems % Q4_0_BLOCK != 0 || b_elems % Q4_0_BLOCK != 0 {
                return Err(DnaError::Truncated {
                    name: name.into(), expected: payload_len,
                    got: 0,
                });
            }
            let a_q_bytes = a_elems / Q4_0_BLOCK * Q4_0_BYTES_PER_BLOCK;
            let b_q_bytes = b_elems / Q4_0_BLOCK * Q4_0_BYTES_PER_BLOCK;
            let expected = header_bytes + a_q_bytes + b_q_bytes;
            if expected != payload_len {
                return Err(DnaError::Truncated {
                    name: name.into(), expected: payload_len, got: expected,
                });
            }
            let mut a_buf = vec![0u8; a_q_bytes];
            f.read_exact(&mut a_buf)?;
            let mut b_buf = vec![0u8; b_q_bytes];
            f.read_exact(&mut b_buf)?;
            let a_f32 = dequantize_q4_0(&a_buf, a_elems);
            let b_f32 = dequantize_q4_0(&b_buf, b_elems);
            // f32 → bf16 (top 16 bits of ieee754 binary32 with round-to-nearest-even).
            let a_u16: Vec<u16> = a_f32.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect();
            let b_u16: Vec<u16> = b_f32.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect();
            let a = backend.upload_bf16(&a_u16, &[rows, rank]);
            let b = backend.upload_bf16(&b_u16, &[rank, cols]);
            Ok(DnaTensor::Svd { a, b, rows, cols, rank })
        }
        _ => Err(DnaError::UnsupportedDtype(dtype)),
    }
}

fn parse_traw(
    f: &mut File,
    name: &str,
    payload_len: usize,
    backend: &Arc<dyn ComputeBackend>,
) -> DnaResult<DnaTensor> {
    let dim_count = read_u32(f)? as usize;
    let mut shape: Vec<usize> = Vec::with_capacity(dim_count);
    for _ in 0..dim_count {
        shape.push(read_u32(f)? as usize);
    }
    let dtype = read_u32(f)?;
    let header_bytes = 4 + dim_count * 4 + 4;

    let numel: usize = shape.iter().copied().product();

    let tensor = match dtype {
        DTYPE_BF16 => {
            let expect = header_bytes + numel * 2;
            if expect != payload_len {
                return Err(DnaError::Truncated {
                    name: name.into(),
                    expected: payload_len,
                    got: expect,
                });
            }
            let data = read_u16_vec(f, numel)?;
            backend.upload_bf16(&data, &shape)
        }
        _ => {
            // F32 fallback: stored as 4 bytes/elem, used for norms/biases.
            let expect = header_bytes + numel * 4;
            if expect != payload_len {
                return Err(DnaError::Truncated {
                    name: name.into(),
                    expected: payload_len,
                    got: expect,
                });
            }
            let data = read_f32_vec(f, numel)?;
            backend.upload_f32(&data, &shape)
        }
    };

    Ok(DnaTensor::Full { tensor, shape })
}

/// Parse a DNA-V5 TBLK chunk — ggml block-quantized tensor (Q4_K/Q6_K/Q8_0).
/// Layout: ndim u32 + dims[ndim] u32 + dtype u32 + raw bytes.
fn parse_tblk(
    f: &mut File,
    name: &str,
    payload_len: usize,
    backend: &Arc<dyn ComputeBackend>,
) -> DnaResult<DnaTensor> {
    let dim_count = read_u32(f)? as usize;
    let mut shape: Vec<usize> = Vec::with_capacity(dim_count);
    for _ in 0..dim_count {
        shape.push(read_u32(f)? as usize);
    }
    let dtype = read_u32(f)?;
    let header_bytes = 4 + dim_count * 4 + 4;
    let blocks_bytes_len = payload_len.saturating_sub(header_bytes);
    let mut blocks = vec![0u8; blocks_bytes_len];
    f.read_exact(&mut blocks).map_err(DnaError::from)?;

    // Validate dtype + sizing.
    let (block_size, block_bytes) = match dtype {
        DTYPE_Q4_K => (256usize, 144usize), // d + dmin + scales(12) + qs(128) = 144
        DTYPE_Q6_K => (256, 210),           // ql(128) + qh(64) + scales(16) + d(2) = 210
        DTYPE_Q8_0 => (32, 34),             // d(2) + qs(32) = 34
        _ => return Err(DnaError::UnsupportedDtype(dtype)),
    };
    let numel: usize = shape.iter().product();
    let n_blocks = (numel + block_size - 1) / block_size;
    let expected = n_blocks * block_bytes;
    if blocks_bytes_len != expected {
        return Err(DnaError::Truncated {
            name: format!("TBLK {} dtype={}", name, dtype),
            expected,
            got: blocks_bytes_len,
        });
    }

    let data = backend.upload_raw(&blocks, &shape);
    Ok(DnaTensor::Block { data, shape, dtype })
}

// ── Tests (binary-level roundtrip; backend integration tested separately) ──

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    /// Parse headers without uploading tensors. Returns (version, count, entries).
    fn parse_headers(path: &Path) -> DnaResult<(u32, u32, Vec<([u8; 4], String, usize)>)> {
        let mut f = File::open(path)?;
        let magic = read_magic(&mut f)?;
        assert_eq!(&magic, DNA_MAGIC);
        let version = read_u32(&mut f)?;
        let count = read_u32(&mut f)?;
        let mut entries = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let m = read_magic(&mut f)?;
            let name = read_name(&mut f)?;
            let plen = read_u64(&mut f)? as usize;
            entries.push((m, name, plen));
            f.seek(SeekFrom::Current(plen as i64))?;
        }
        Ok((version, count, entries))
    }

    #[test]
    fn tsvd_header_roundtrip() {
        let tmp = NamedTempFile::new().unwrap();
        let (rows, cols, rank) = (8u32, 16u32, 4u32);
        let a: Vec<u16> = (0..rows * rank).map(|i| i as u16).collect();
        let b: Vec<u16> = (0..rank * cols).map(|i| (i + 1000) as u16).collect();

        {
            let mut w = DnaWriter::create(tmp.path()).unwrap();
            w.write_tsvd("layers.0.q_proj.weight", rows, cols, rank, &a, &b).unwrap();
            w.finalize().unwrap();
        }

        let (version, count, entries) = parse_headers(tmp.path()).unwrap();
        assert_eq!(version, 1);
        assert_eq!(count, 1);
        assert_eq!(&entries[0].0, CHUNK_TSVD);
        assert_eq!(entries[0].1, "layers.0.q_proj.weight");
        let expected = 16 + (rows * rank * 2) as usize + (rank * cols * 2) as usize;
        assert_eq!(entries[0].2, expected);
    }

    #[test]
    fn traw_header_roundtrip() {
        let tmp = NamedTempFile::new().unwrap();
        let shape = [128u32];
        let data: Vec<u16> = (0..128).collect();

        {
            let mut w = DnaWriter::create(tmp.path()).unwrap();
            w.write_traw_bf16("norm.weight", &shape, &data).unwrap();
            w.finalize().unwrap();
        }

        let (_v, count, entries) = parse_headers(tmp.path()).unwrap();
        assert_eq!(count, 1);
        assert_eq!(&entries[0].0, CHUNK_TRAW);
        assert_eq!(entries[0].1, "norm.weight");
        assert_eq!(entries[0].2, 4 + 4 + 4 + 128 * 2);
    }

    #[test]
    fn multi_tensor_file() {
        let tmp = NamedTempFile::new().unwrap();
        let a: Vec<u16> = vec![1, 2, 3, 4];
        let b: Vec<u16> = vec![5, 6, 7, 8];
        {
            let mut w = DnaWriter::create(tmp.path()).unwrap();
            w.write_tsvd("w1", 2, 2, 2, &a, &b).unwrap();
            w.write_traw_bf16("norm", &[4], &a).unwrap();
            w.write_tsvd("w2", 2, 2, 2, &b, &a).unwrap();
            w.finalize().unwrap();
        }
        let (_, count, entries) = parse_headers(tmp.path()).unwrap();
        assert_eq!(count, 3);
        assert_eq!(entries[0].1, "w1");
        assert_eq!(entries[1].1, "norm");
        assert_eq!(entries[2].1, "w2");
    }
}
