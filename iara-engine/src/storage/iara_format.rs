// ============================================================================
// storage/iara_format.rs — IARA v2 Model Format
// ============================================================================
//
// Formato nativo otimizado para IARA Engine. Substitui safetensors com:
//   - Multi-quant: BF16, Q4, FP8, Ternary (BitNet) por tensor
//   - Zero-copy mmap loading (memmap2)
//   - Metadata rica: architecture, tokenizer config, quant recipe
//   - Alignment a 64 bytes para DMA/GPU direto
//   - Conversão de safetensors → .iara como pré-processamento
//
// Layout do arquivo .iara v2:
// ┌─────────────────────────────────────────────────┐
// │ [4 bytes]   Magic: "IARA"                       │
// │ [4 bytes]   Version: 2 (u32 LE)                 │
// │ [8 bytes]   Header length (u64 LE)              │
// │ [header_len bytes] JSON header (UTF-8)           │
// │ [padding to 64-byte alignment]                   │
// │ [... tensor data blocks, each 64-byte aligned]   │
// └─────────────────────────────────────────────────┘
//
// JSON Header:
// {
//   "metadata": {
//     "architecture": "qwen3",
//     "hidden_dim": 4096,
//     "n_layers": 36,
//     "vocab_size": 151936,
//     "source_model": "Qwen/Qwen3-8B",
//     "converted_at": "2026-04-06T...",
//     "iara_version": "2.0",
//     "quant_recipe": { "default": "fp8", "embed_tokens": "bf16", ... }
//   },
//   "tensors": {
//     "model.layers.0.self_attn.q_proj.weight": {
//       "dtype": "fp8_e4m3",
//       "shape": [4096, 4096],
//       "data_offset": 4096,
//       "data_length": 16777216,
//       "scales_offset": ...,
//       "scales_length": ...,
//       "group_size": 0,
//       "quant_params": { "scale_type": "per_row" }
//     },
//     ...
//   }
// }
// ============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

// ── Constants ───────────────────────────────────────────────────────────────

const IARA_V2_MAGIC: &[u8; 4] = b"IARA";
const IARA_V2_VERSION: u32 = 2;
const ALIGNMENT: usize = 64; // 64-byte alignment for GPU DMA

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum IaraFormatError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
}

pub type IaraResult<T> = Result<T, IaraFormatError>;

// ── Dtype enum ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IaraDtype {
    /// Standard BF16 (2 bytes per element)
    Bf16,
    /// Standard F32 (4 bytes per element)
    F32,
    /// INT4 group quantized (0.5 bytes per element + scales + zeros)
    Q4,
    /// FP8 E4M3 with per-row scaling (1 byte per element + 4 bytes per row scale)
    Fp8E4m3,
    /// BitNet ternary {-1, 0, +1} packed 2-bit (0.25 bytes per element + 4 bytes scale)
    Ternary,
    /// Pin + Terrain + Ternary compression (top 1-5% BF16 pins + ternary terrain)
    Ptt,
}

impl IaraDtype {
    /// Bytes per element (approximate for packed formats)
    pub fn bytes_per_element(&self) -> f32 {
        match self {
            IaraDtype::Bf16 => 2.0,
            IaraDtype::F32 => 4.0,
            IaraDtype::Q4 => 0.5,
            IaraDtype::Fp8E4m3 => 1.0,
            IaraDtype::Ternary => 0.25,
            IaraDtype::Ptt => 0.29, // ~2 bits terrain + ~1-5% BF16 pins + overhead
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            IaraDtype::Bf16 => "BF16",
            IaraDtype::F32 => "F32",
            IaraDtype::Q4 => "Q4",
            IaraDtype::Fp8E4m3 => "FP8-E4M3",
            IaraDtype::Ternary => "Ternary-1.58b",
            IaraDtype::Ptt => "P+T+T-6.9x",
        }
    }
}

// ── Tensor metadata ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IaraV2TensorMeta {
    /// Quantization dtype
    pub dtype: IaraDtype,

    /// Original shape (e.g. [4096, 4096])
    pub shape: Vec<usize>,

    /// Offset of packed/quantized data in file (absolute from file start)
    pub data_offset: u64,

    /// Length of packed/quantized data in bytes
    pub data_length: u64,

    /// Offset of scales array (for Q4, FP8, Ternary, P+T+T terrain scales)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scales_offset: Option<u64>,

    /// Length of scales array in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scales_length: Option<u64>,

    /// Offset of zero-points array (for Q4)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zeros_offset: Option<u64>,

    /// Length of zero-points array in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zeros_length: Option<u64>,

    /// Offset of pin indices array (for P+T+T)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pin_indices_offset: Option<u64>,

    /// Length of pin indices array in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pin_indices_length: Option<u64>,

    /// Offset of pin values array (for P+T+T)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pin_values_offset: Option<u64>,

    /// Length of pin values array in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pin_values_length: Option<u64>,

    /// Group size for Q4 quantization (0 = not applicable)
    #[serde(default)]
    pub group_size: usize,

    /// Additional quantization parameters
    #[serde(default)]
    pub quant_params: HashMap<String, String>,
}

// ── Epigenome metadata ─────────────────────────────────────────────────────

/// Layer execution modes for adaptive inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerMode {
    /// Fast: ternary only, skip complex operations
    Fast,
    /// Medium: P+T+T, normal compute
    Medium,
    /// Slow: full precision for critical paths
    Slow,
}

/// Epigenome metadata for calibrated model behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigenomeMetadata {
    /// Per-layer importance score (0-1, from TensorImportance)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_importance: Option<Vec<f32>>,

    /// Per-layer recommended compute mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_mode: Option<Vec<LayerMode>>,

    /// Hot channels (mycelium) — which hidden dimensions carry most pins
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hot_channels: Option<Vec<usize>>,

    /// Dead neuron ratio per layer (from calibration)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dead_neuron_ratios: Option<Vec<f32>>,

    /// Calibration timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated_at: Option<String>,
}

// ── File-level metadata ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IaraV2Metadata {
    /// Model architecture (e.g. "qwen3", "llama", "mistral")
    pub architecture: String,

    /// Hidden dimension
    pub hidden_dim: usize,

    /// Number of transformer layers
    pub n_layers: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Number of attention heads
    pub n_heads: usize,

    /// Number of KV heads (for GQA)
    pub n_kv_heads: usize,

    /// Intermediate dimension (MLP)
    pub intermediate_dim: usize,

    /// RoPE theta
    pub rope_theta: f64,

    /// Source model identifier
    pub source_model: String,

    /// Conversion timestamp
    pub converted_at: String,

    /// IARA format version
    pub iara_version: String,

    /// Quantization recipe: maps tensor name patterns to dtype
    /// e.g. {"default": "fp8_e4m3", "embed_tokens": "bf16", "lm_head": "bf16"}
    pub quant_recipe: HashMap<String, String>,

    /// Original model size in bytes (before conversion)
    pub original_size_bytes: u64,

    /// Converted model size in bytes
    pub converted_size_bytes: u64,

    /// Compression ratio
    pub compression_ratio: f64,

    /// Epigenome metadata (initially empty, filled by calibration)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epigenome: Option<EpigenomeMetadata>,
}

// ── Complete header ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IaraV2Header {
    pub metadata: IaraV2Metadata,
    pub tensors: HashMap<String, IaraV2TensorMeta>,
}

// ── Writer ──────────────────────────────────────────────────────────────────

/// Builds and writes an .iara v2 file incrementally.
pub struct IaraV2Writer {
    file: File,
    tensors: HashMap<String, IaraV2TensorMeta>,
    metadata: IaraV2Metadata,
    data_offset: u64, // current write position for tensor data
    header_reserved: u64, // space reserved for header at the start
}

impl IaraV2Writer {
    /// Create a new .iara v2 writer. Reserves space for header at the start.
    pub fn new(path: &Path, metadata: IaraV2Metadata) -> IaraResult<Self> {
        let mut file = File::create(path)?;

        // Write magic + version + placeholder header_len
        file.write_all(IARA_V2_MAGIC)?;
        file.write_all(&IARA_V2_VERSION.to_le_bytes())?;
        file.write_all(&0u64.to_le_bytes())?; // placeholder

        // Reserve generous space for header (will rewrite at end)
        // Estimate: ~500 bytes per tensor, typical model has ~200-300 tensors
        let header_reserved = 256 * 1024u64; // 256KB for header
        let padding = vec![0u8; header_reserved as usize];
        file.write_all(&padding)?;

        // Align data start to ALIGNMENT
        let data_start = align_up(16 + header_reserved as usize, ALIGNMENT) as u64;
        file.seek(SeekFrom::Start(data_start))?;

        Ok(Self {
            file,
            tensors: HashMap::new(),
            metadata,
            data_offset: data_start,
            header_reserved,
        })
    }

    /// Write a BF16 tensor (raw bytes, zero-copy from safetensors mmap)
    pub fn write_bf16_tensor(&mut self, name: &str, shape: &[usize], bf16_bytes: &[u8]) -> IaraResult<()> {
        let offset = self.write_aligned_data(bf16_bytes)?;

        self.tensors.insert(name.to_string(), IaraV2TensorMeta {
            dtype: IaraDtype::Bf16,
            shape: shape.to_vec(),
            data_offset: offset,
            data_length: bf16_bytes.len() as u64,
            scales_offset: None,
            scales_length: None,
            zeros_offset: None,
            zeros_length: None,
            pin_indices_offset: None,
            pin_indices_length: None,
            pin_values_offset: None,
            pin_values_length: None,
            group_size: 0,
            quant_params: HashMap::new(),
        });

        Ok(())
    }

    /// Write an FP8 quantized tensor (data + per-row scales)
    pub fn write_fp8_tensor(
        &mut self, name: &str, shape: &[usize],
        fp8_data: &[u8], scales: &[f32],
    ) -> IaraResult<()> {
        let data_offset = self.write_aligned_data(fp8_data)?;

        // Write scales as f32 LE
        let scales_bytes: Vec<u8> = scales.iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let scales_offset = self.write_aligned_data(&scales_bytes)?;

        let mut params = HashMap::new();
        params.insert("scale_type".to_string(), "per_row".to_string());

        self.tensors.insert(name.to_string(), IaraV2TensorMeta {
            dtype: IaraDtype::Fp8E4m3,
            shape: shape.to_vec(),
            data_offset,
            data_length: fp8_data.len() as u64,
            scales_offset: Some(scales_offset),
            scales_length: Some(scales_bytes.len() as u64),
            zeros_offset: None,
            zeros_length: None,
            pin_indices_offset: None,
            pin_indices_length: None,
            pin_values_offset: None,
            pin_values_length: None,
            group_size: 0,
            quant_params: params,
        });

        Ok(())
    }

    /// Write a Q4 quantized tensor (packed + scales + zeros)
    pub fn write_q4_tensor(
        &mut self, name: &str, shape: &[usize],
        packed: &[u8], scales: &[f32], zeros: &[f32], group_size: usize,
    ) -> IaraResult<()> {
        let data_offset = self.write_aligned_data(packed)?;

        let scales_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let scales_offset = self.write_aligned_data(&scales_bytes)?;

        let zeros_bytes: Vec<u8> = zeros.iter().flat_map(|z| z.to_le_bytes()).collect();
        let zeros_offset = self.write_aligned_data(&zeros_bytes)?;

        self.tensors.insert(name.to_string(), IaraV2TensorMeta {
            dtype: IaraDtype::Q4,
            shape: shape.to_vec(),
            data_offset,
            data_length: packed.len() as u64,
            scales_offset: Some(scales_offset),
            scales_length: Some(scales_bytes.len() as u64),
            zeros_offset: Some(zeros_offset),
            zeros_length: Some(zeros_bytes.len() as u64),
            pin_indices_offset: None,
            pin_indices_length: None,
            pin_values_offset: None,
            pin_values_length: None,
            group_size,
            quant_params: HashMap::new(),
        });

        Ok(())
    }

    /// Write a ternary (BitNet) quantized tensor (packed 2-bit + scale)
    pub fn write_ternary_tensor(
        &mut self, name: &str, shape: &[usize],
        packed: &[u8], scale: f32, sparsity: f32,
    ) -> IaraResult<()> {
        let data_offset = self.write_aligned_data(packed)?;

        let scale_bytes = scale.to_le_bytes();
        let scales_offset = self.write_aligned_data(&scale_bytes)?;

        let mut params = HashMap::new();
        params.insert("sparsity".to_string(), format!("{:.4}", sparsity));

        self.tensors.insert(name.to_string(), IaraV2TensorMeta {
            dtype: IaraDtype::Ternary,
            shape: shape.to_vec(),
            data_offset,
            data_length: packed.len() as u64,
            scales_offset: Some(scales_offset),
            scales_length: Some(4),
            zeros_offset: None,
            zeros_length: None,
            pin_indices_offset: None,
            pin_indices_length: None,
            pin_values_offset: None,
            pin_values_length: None,
            group_size: 0,
            quant_params: params,
        });

        Ok(())
    }

    /// Write a P+T+T (Pin + Terrain + Ternary) encoded tensor
    pub fn write_ptt_tensor(
        &mut self, name: &str, shape: &[usize],
        terrain_packed: &[u8],
        terrain_scales: &[f32],
        pin_indices: &[u32],
        pin_values: &[u16],
        pin_threshold_sigma: f32,
        pin_ratio: f32,
        terrain_sparsity: f32,
    ) -> IaraResult<()> {
        // Write terrain packed data
        let terrain_data_offset = self.write_aligned_data(terrain_packed)?;

        // Write terrain scales
        let scales_bytes: Vec<u8> = terrain_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let scales_offset = self.write_aligned_data(&scales_bytes)?;

        // Write pin indices as u32 LE
        let pin_indices_bytes: Vec<u8> = pin_indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let pin_indices_offset = self.write_aligned_data(&pin_indices_bytes)?;

        // Write pin values as u16 LE (BF16)
        let pin_values_bytes: Vec<u8> = pin_values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let pin_values_offset = self.write_aligned_data(&pin_values_bytes)?;

        // Build metadata
        let mut params = HashMap::new();
        params.insert("pin_ratio".to_string(), format!("{:.4}", pin_ratio));
        params.insert("terrain_sparsity".to_string(), format!("{:.4}", terrain_sparsity));
        params.insert("pin_threshold_sigma".to_string(), format!("{:.2}", pin_threshold_sigma));

        self.tensors.insert(name.to_string(), IaraV2TensorMeta {
            dtype: IaraDtype::Ptt,
            shape: shape.to_vec(),
            data_offset: terrain_data_offset,
            data_length: terrain_packed.len() as u64,
            scales_offset: Some(scales_offset),
            scales_length: Some(scales_bytes.len() as u64),
            zeros_offset: None,
            zeros_length: None,
            pin_indices_offset: Some(pin_indices_offset),
            pin_indices_length: Some(pin_indices_bytes.len() as u64),
            pin_values_offset: Some(pin_values_offset),
            pin_values_length: Some(pin_values_bytes.len() as u64),
            group_size: 0,
            quant_params: params,
        });

        Ok(())
    }

    /// Finalize: write the JSON header and fix up the header_len field
    pub fn finalize(mut self) -> IaraResult<u64> {
        // Build the complete header
        self.metadata.converted_size_bytes = self.data_offset;
        self.metadata.compression_ratio = if self.metadata.original_size_bytes > 0 {
            self.metadata.original_size_bytes as f64 / self.data_offset as f64
        } else {
            1.0
        };

        let header = IaraV2Header {
            metadata: self.metadata,
            tensors: self.tensors,
        };

        let header_json = serde_json::to_vec_pretty(&header)?;

        // Check header fits in reserved space
        if header_json.len() as u64 > self.header_reserved {
            return Err(IaraFormatError::InvalidFormat(format!(
                "Header too large ({} bytes) for reserved space ({} bytes)",
                header_json.len(), self.header_reserved
            )));
        }

        // Write header at the start of file
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(IARA_V2_MAGIC)?;
        self.file.write_all(&IARA_V2_VERSION.to_le_bytes())?;
        self.file.write_all(&(header_json.len() as u64).to_le_bytes())?;
        self.file.write_all(&header_json)?;

        self.file.flush()?;

        let total_size = self.data_offset;
        log::info!("📦 IARA v2 file written: {} bytes ({:.1} MB)",
            total_size, total_size as f64 / 1e6);

        Ok(total_size)
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Write data at current position with 64-byte alignment, return the offset
    fn write_aligned_data(&mut self, data: &[u8]) -> IaraResult<u64> {
        // Align current position
        let aligned = align_up(self.data_offset as usize, ALIGNMENT) as u64;
        if aligned > self.data_offset {
            let pad = vec![0u8; (aligned - self.data_offset) as usize];
            self.file.write_all(&pad)?;
            self.data_offset = aligned;
        }

        let offset = self.data_offset;
        self.file.write_all(data)?;
        self.data_offset += data.len() as u64;

        Ok(offset)
    }
}

// ── Reader ──────────────────────────────────────────────────────────────────

/// Reads an .iara v2 file. Supports mmap for zero-copy tensor access.
pub struct IaraV2Reader {
    pub header: IaraV2Header,
    mmap: memmap2::Mmap,
}

impl IaraV2Reader {
    /// Open an .iara v2 file for reading
    pub fn open(path: &Path) -> IaraResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        if mmap.len() < 16 {
            return Err(IaraFormatError::InvalidFormat("File too small".into()));
        }

        // Validate magic
        if &mmap[0..4] != IARA_V2_MAGIC {
            return Err(IaraFormatError::InvalidFormat(
                format!("Invalid magic: {:?}", &mmap[0..4])));
        }

        // Version
        let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
        if version != IARA_V2_VERSION {
            return Err(IaraFormatError::InvalidFormat(
                format!("Version mismatch: got {}, expected {}", version, IARA_V2_VERSION)));
        }

        // Header length
        let header_len = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        if 16 + header_len > mmap.len() {
            return Err(IaraFormatError::InvalidFormat("Header exceeds file size".into()));
        }

        // Parse JSON header
        let header_json = std::str::from_utf8(&mmap[16..16 + header_len])
            .map_err(|e| IaraFormatError::InvalidFormat(format!("Invalid UTF-8 in header: {}", e)))?;
        let header: IaraV2Header = serde_json::from_str(header_json)?;

        log::info!("📖 IARA v2 opened: {} tensors, arch={}, {} layers",
            header.tensors.len(),
            header.metadata.architecture,
            header.metadata.n_layers);

        Ok(Self { header, mmap })
    }

    /// Get tensor metadata by name
    pub fn tensor_meta(&self, name: &str) -> IaraResult<&IaraV2TensorMeta> {
        self.header.tensors.get(name)
            .ok_or_else(|| IaraFormatError::TensorNotFound(name.to_string()))
    }

    /// Get raw bytes for a tensor's packed data (zero-copy from mmap)
    pub fn tensor_data(&self, name: &str) -> IaraResult<&[u8]> {
        let meta = self.tensor_meta(name)?;
        let start = meta.data_offset as usize;
        let end = start + meta.data_length as usize;

        if end > self.mmap.len() {
            return Err(IaraFormatError::InvalidFormat(
                format!("Tensor '{}' data exceeds file bounds", name)));
        }

        Ok(&self.mmap[start..end])
    }

    /// Get scales for a quantized tensor (zero-copy from mmap)
    pub fn tensor_scales(&self, name: &str) -> IaraResult<&[u8]> {
        let meta = self.tensor_meta(name)?;
        let offset = meta.scales_offset.ok_or_else(||
            IaraFormatError::InvalidFormat(format!("Tensor '{}' has no scales", name)))?;
        let length = meta.scales_length.ok_or_else(||
            IaraFormatError::InvalidFormat(format!("Tensor '{}' has no scales_length", name)))?;

        let start = offset as usize;
        let end = start + length as usize;

        if end > self.mmap.len() {
            return Err(IaraFormatError::InvalidFormat(
                format!("Tensor '{}' scales exceed file bounds", name)));
        }

        Ok(&self.mmap[start..end])
    }

    /// Get zeros for Q4 tensor (zero-copy from mmap)
    pub fn tensor_zeros(&self, name: &str) -> IaraResult<&[u8]> {
        let meta = self.tensor_meta(name)?;
        let offset = meta.zeros_offset.ok_or_else(||
            IaraFormatError::InvalidFormat(format!("Tensor '{}' has no zeros", name)))?;
        let length = meta.zeros_length.ok_or_else(||
            IaraFormatError::InvalidFormat(format!("Tensor '{}' has no zeros_length", name)))?;

        let start = offset as usize;
        let end = start + length as usize;

        if end > self.mmap.len() {
            return Err(IaraFormatError::InvalidFormat(
                format!("Tensor '{}' zeros exceed file bounds", name)));
        }

        Ok(&self.mmap[start..end])
    }

    /// Read FP8 scales as f32 slice
    pub fn tensor_scales_f32(&self, name: &str) -> IaraResult<Vec<f32>> {
        let bytes = self.tensor_scales(name)?;
        Ok(bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Get raw bytes for P+T+T pin indices (zero-copy from mmap)
    pub fn tensor_pin_indices(&self, name: &str) -> IaraResult<&[u8]> {
        let meta = self.tensor_meta(name)?;
        let offset = meta.pin_indices_offset.ok_or_else(||
            IaraFormatError::InvalidFormat(format!("Tensor '{}' has no pin_indices", name)))?;
        let length = meta.pin_indices_length.ok_or_else(||
            IaraFormatError::InvalidFormat(format!("Tensor '{}' has no pin_indices_length", name)))?;

        let start = offset as usize;
        let end = start + length as usize;

        if end > self.mmap.len() {
            return Err(IaraFormatError::InvalidFormat(
                format!("Tensor '{}' pin_indices exceed file bounds", name)));
        }

        Ok(&self.mmap[start..end])
    }

    /// Get raw bytes for P+T+T pin values (zero-copy from mmap)
    pub fn tensor_pin_values(&self, name: &str) -> IaraResult<&[u8]> {
        let meta = self.tensor_meta(name)?;
        let offset = meta.pin_values_offset.ok_or_else(||
            IaraFormatError::InvalidFormat(format!("Tensor '{}' has no pin_values", name)))?;
        let length = meta.pin_values_length.ok_or_else(||
            IaraFormatError::InvalidFormat(format!("Tensor '{}' has no pin_values_length", name)))?;

        let start = offset as usize;
        let end = start + length as usize;

        if end > self.mmap.len() {
            return Err(IaraFormatError::InvalidFormat(
                format!("Tensor '{}' pin_values exceed file bounds", name)));
        }

        Ok(&self.mmap[start..end])
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.header.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Total file size
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Summary of the model
    pub fn summary(&self) -> String {
        let meta = &self.header.metadata;
        let mut dtype_counts: HashMap<IaraDtype, usize> = HashMap::new();
        let mut total_params = 0u64;

        for tensor in self.header.tensors.values() {
            *dtype_counts.entry(tensor.dtype).or_insert(0) += 1;
            let n_params: u64 = tensor.shape.iter().product::<usize>() as u64;
            total_params += n_params;
        }

        let mut lines = vec![
            format!("IARA v2 Model: {}", meta.source_model),
            format!("Architecture: {} | {} layers | hidden={} | vocab={}",
                meta.architecture, meta.n_layers, meta.hidden_dim, meta.vocab_size),
            format!("Total parameters: {:.1}B", total_params as f64 / 1e9),
            format!("File size: {:.1} MB", self.mmap.len() as f64 / 1e6),
            format!("Compression: {:.1}×", meta.compression_ratio),
            format!("Tensors: {}", self.header.tensors.len()),
            "Dtype breakdown:".to_string(),
        ];

        for (dtype, count) in &dtype_counts {
            lines.push(format!("  {}: {} tensors", dtype.name(), count));
        }

        lines.join("\n")
    }
}

// ── Safetensors → IARA Converter ────────────────────────────────────────────

/// Quantization recipe: decides what dtype each tensor gets
#[derive(Debug, Clone)]
pub struct QuantRecipe {
    /// Default dtype for weight tensors
    pub default_dtype: IaraDtype,

    /// Override for specific tensor patterns (checked first)
    /// Patterns are checked as substring matches
    pub overrides: Vec<(String, IaraDtype)>,

    /// For pulse recipe: number of layers (used for depth-aware quantization)
    /// Not serialized, computed at recipe creation time
    #[doc(hidden)]
    pub _pulse_n_layers: Option<usize>,
}

impl QuantRecipe {
    /// Standard FP8 recipe: most weights FP8, embeddings BF16
    pub fn fp8_standard() -> Self {
        Self {
            default_dtype: IaraDtype::Fp8E4m3,
            overrides: vec![
                ("embed_tokens".into(), IaraDtype::Bf16),
                ("lm_head".into(), IaraDtype::Bf16),
                ("layernorm".into(), IaraDtype::Bf16),
                ("norm.weight".into(), IaraDtype::Bf16),
                ("rotary_emb".into(), IaraDtype::Bf16),
            ],
            _pulse_n_layers: None,
        }
    }

    /// Aggressive recipe: ternary for MLP, FP8 for attention, BF16 for norms
    pub fn aggressive() -> Self {
        Self {
            default_dtype: IaraDtype::Fp8E4m3,
            overrides: vec![
                ("embed_tokens".into(), IaraDtype::Bf16),
                ("lm_head".into(), IaraDtype::Bf16),
                ("layernorm".into(), IaraDtype::Bf16),
                ("norm.weight".into(), IaraDtype::Bf16),
                ("gate_proj".into(), IaraDtype::Ternary),
                ("up_proj".into(), IaraDtype::Ternary),
                ("down_proj".into(), IaraDtype::Ternary),
            ],
            _pulse_n_layers: None,
        }
    }

    /// Q4 recipe (backward compatible with v1)
    pub fn q4_standard() -> Self {
        Self {
            default_dtype: IaraDtype::Q4,
            overrides: vec![
                ("embed_tokens".into(), IaraDtype::Bf16),
                ("lm_head".into(), IaraDtype::Bf16),
                ("layernorm".into(), IaraDtype::Bf16),
                ("norm.weight".into(), IaraDtype::Bf16),
            ],
            _pulse_n_layers: None,
        }
    }

    /// Pulse recipe: depth-aware quantization by layer depth
    /// For a model with N layers:
    ///   - Layers 0 to N/4: Ternary (2-bit, fastest)
    ///   - Layers N/4 to 3N/4: FP8 (good balance)
    ///   - Layers 3N/4 to N: BF16 (full precision)
    ///   - Embeddings, LM head, norms: always BF16 (critical)
    pub fn pulse(n_layers: usize) -> Self {
        Self {
            default_dtype: IaraDtype::Fp8E4m3,
            overrides: vec![
                ("embed_tokens".into(), IaraDtype::Bf16),
                ("lm_head".into(), IaraDtype::Bf16),
                ("layernorm".into(), IaraDtype::Bf16),
                ("norm.weight".into(), IaraDtype::Bf16),
                ("rotary_emb".into(), IaraDtype::Bf16),
            ],
            _pulse_n_layers: Some(n_layers),
        }
    }

    /// P+T+T (Pin + Terrain + Ternary) recipe: 6.9x compression
    /// Uses P+T+T for weight matrices, BF16 for embeddings and norms
    pub fn ptt_standard() -> Self {
        Self {
            default_dtype: IaraDtype::Ptt,
            overrides: vec![
                ("embed_tokens".into(), IaraDtype::Bf16),
                ("lm_head".into(), IaraDtype::Bf16),
                ("layernorm".into(), IaraDtype::Bf16),
                ("norm.weight".into(), IaraDtype::Bf16),
                ("rotary_emb".into(), IaraDtype::Bf16),
            ],
            _pulse_n_layers: None,
        }
    }

    /// Determine dtype for a given tensor name
    pub fn dtype_for(&self, tensor_name: &str) -> IaraDtype {
        // First check explicit overrides
        for (pattern, dtype) in &self.overrides {
            if tensor_name.contains(pattern.as_str()) {
                return *dtype;
            }
        }

        // Apply depth-aware logic if pulse recipe
        if let Some(n_layers) = self._pulse_n_layers {
            if let Some(layer_idx) = Self::extract_layer_index(tensor_name) {
                let quarter = n_layers / 4;
                let three_quarter = (3 * n_layers) / 4;

                return if layer_idx < quarter {
                    IaraDtype::Ternary
                } else if layer_idx < three_quarter {
                    IaraDtype::Fp8E4m3
                } else {
                    IaraDtype::Bf16
                };
            }
        }

        self.default_dtype
    }

    /// Extract layer index from tensor name (e.g., "model.layers.5.self_attn..." → 5)
    fn extract_layer_index(tensor_name: &str) -> Option<usize> {
        if !tensor_name.contains(".layers.") {
            return None;
        }

        let parts: Vec<&str> = tensor_name.split('.').collect();
        for i in 0..parts.len() - 1 {
            if parts[i] == "layers" {
                return parts[i + 1].parse::<usize>().ok();
            }
        }
        None
    }
}

/// Convert a safetensors model directory to .iara v2 format
///
/// Reads safetensors files from `model_dir`, applies the quant recipe,
/// and writes a single optimized .iara file.
pub fn convert_safetensors_to_iara(
    model_dir: &Path,
    output_path: &Path,
    recipe: &QuantRecipe,
    model_metadata: IaraV2Metadata,
) -> IaraResult<u64> {
    use crate::storage::{SafetensorsIndex, TensorMeta};

    // 1. Read safetensors index
    let index_path = model_dir.join("model.safetensors.index.json");
    let (weight_map, _all_meta) = if index_path.exists() {
        let index_json = std::fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_json)?;
        let weight_map = index.weight_map;
        (weight_map, HashMap::<String, TensorMeta>::new())
    } else {
        // Single file model
        let mut map = HashMap::new();
        let single_file = "model.safetensors";
        if model_dir.join(single_file).exists() {
            // Parse the single file to get tensor names
            let file = File::open(model_dir.join(single_file))?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
                .map_err(|e| IaraFormatError::InvalidFormat(format!("UTF-8 error: {}", e)))?;
            let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)?;
            for key in raw.keys() {
                if key != "__metadata__" {
                    map.insert(key.clone(), single_file.to_string());
                }
            }
        }
        (map, HashMap::new())
    };

    // 2. Group tensors by shard file
    let mut shard_tensors: HashMap<String, Vec<String>> = HashMap::new();
    for (tensor_name, shard) in &weight_map {
        shard_tensors.entry(shard.clone()).or_default().push(tensor_name.clone());
    }

    // Compute original size
    let mut original_size = 0u64;

    // 3. Create writer
    let mut writer = IaraV2Writer::new(output_path, model_metadata)?;

    // 4. Process each shard
    for (shard_name, tensor_names) in &shard_tensors {
        let shard_path = model_dir.join(shard_name);
        if !shard_path.exists() {
            log::warn!("⚠️  Shard not found: {}", shard_path.display());
            continue;
        }

        let file = File::open(&shard_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Parse shard header (skip __metadata__ key)
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
            .map_err(|e| IaraFormatError::InvalidFormat(format!("UTF-8: {}", e)))?;
        let raw_value: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)?;
        let mut raw: HashMap<String, TensorMeta> = HashMap::new();
        for (k, v) in &raw_value {
            if k == "__metadata__" { continue; }
            match serde_json::from_value::<TensorMeta>(v.clone()) {
                Ok(meta) => { raw.insert(k.clone(), meta); }
                Err(_) => { log::warn!("⚠️  Skipping non-tensor key: {}", k); }
            }
        }
        let data_start = 8 + header_len;

        for tensor_name in tensor_names {
            let meta = match raw.get(tensor_name) {
                Some(m) => m,
                None => {
                    log::warn!("⚠️  Tensor '{}' not found in shard header", tensor_name);
                    continue;
                }
            };

            let n_elements: usize = meta.shape.iter().product();
            let bytes_per_elem = match meta.dtype.as_str() {
                "BF16" | "F16" => 2,
                "F32" => 4,
                _ => 2, // default assume BF16
            };

            let tensor_start = data_start + meta.data_offsets[0];
            let tensor_end = data_start + meta.data_offsets[1];
            let tensor_bytes = &mmap[tensor_start..tensor_end];
            original_size += tensor_bytes.len() as u64;

            let target_dtype = recipe.dtype_for(tensor_name);

            match target_dtype {
                IaraDtype::Bf16 => {
                    // Store as-is (BF16 raw bytes)
                    writer.write_bf16_tensor(tensor_name, &meta.shape, tensor_bytes)?;
                }
                IaraDtype::F32 => {
                    // Convert BF16 to F32
                    let f32_data: Vec<f32> = tensor_bytes.chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect();
                    let f32_bytes: Vec<u8> = f32_data.iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    writer.write_bf16_tensor(tensor_name, &meta.shape, &f32_bytes)?;
                }
                IaraDtype::Fp8E4m3 => {
                    // Convert BF16 → F32 → FP8 with per-row scaling
                    let rows = if meta.shape.len() >= 2 { meta.shape[0] } else { 1 };
                    let cols = n_elements / rows;

                    let bf16_u16: Vec<u16> = tensor_bytes.chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();

                    let quantized = crate::storage::fp8::quantize_bf16_to_fp8(
                        &bf16_u16, [rows, cols]);

                    writer.write_fp8_tensor(
                        tensor_name, &meta.shape,
                        &quantized.data, &quantized.scales,
                    )?;
                }
                IaraDtype::Q4 => {
                    // Convert BF16 → Q4
                    let bf16_u16: Vec<u16> = tensor_bytes.chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();

                    let group_size = 128;
                    match crate::storage::quantized::quantize_bf16_to_q4(
                        &bf16_u16, &meta.shape, group_size) {
                        Ok(q4_result) => {
                            writer.write_q4_tensor(
                                tensor_name, &meta.shape,
                                &q4_result.packed, &q4_result.scales, &q4_result.zeros,
                                group_size,
                            )?;
                        }
                        Err(e) => {
                            log::warn!("⚠️  Q4 quant failed for '{}': {}, storing as BF16", tensor_name, e);
                            writer.write_bf16_tensor(tensor_name, &meta.shape, tensor_bytes)?;
                        }
                    }
                }
                IaraDtype::Ternary => {
                    // Convert BF16 → Ternary
                    let rows = if meta.shape.len() >= 2 { meta.shape[0] } else { 1 };
                    let cols = n_elements / rows;

                    let bf16_u16: Vec<u16> = tensor_bytes.chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();

                    let ternary = crate::storage::bitnet::quantize_to_ternary(
                        &bf16_u16, [rows, cols]);

                    writer.write_ternary_tensor(
                        tensor_name, &meta.shape,
                        &ternary.packed, ternary.scale, ternary.sparsity,
                    )?;
                }
                IaraDtype::Ptt => {
                    // Convert BF16 → P+T+T (Pin + Terrain + Ternary)
                    let rows = if meta.shape.len() >= 2 { meta.shape[0] } else { 1 };
                    let cols = n_elements / rows;

                    let bf16_u16: Vec<u16> = tensor_bytes.chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();

                    // Convert BF16 to F32 for P+T+T encoding
                    let f32_data: Vec<f32> = bf16_u16.iter()
                        .map(|&b| f32::from_bits((b as u32) << 16))
                        .collect();

                    // Encode with P+T+T (sigma_threshold = 2.0 standard deviations)
                    let sigma_threshold = 2.0f32;
                    let ptt = crate::storage::ptt::encode(&f32_data, rows, cols, sigma_threshold);
                    let ptt_stats = crate::storage::ptt::stats(&ptt, None);

                    writer.write_ptt_tensor(
                        tensor_name, &meta.shape,
                        &ptt.terrain_packed, &ptt.terrain_scales,
                        &ptt.pin_indices, &ptt.pin_values,
                        ptt.pin_threshold_sigma,
                        ptt_stats.pin_ratio,
                        ptt_stats.terrain_sparsity,
                    )?;
                }
            }

            log::debug!("  ✅ {} [{:?}] → {}", tensor_name,
                meta.shape, target_dtype.name());
        }
    }

    // Update original size and finalize
    writer.metadata.original_size_bytes = original_size;
    let total = writer.finalize()?;

    Ok(total)
}

// ── Utility ─────────────────────────────────────────────────────────────────

fn align_up(val: usize, alignment: usize) -> usize {
    (val + alignment - 1) & !(alignment - 1)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata() -> IaraV2Metadata {
        IaraV2Metadata {
            architecture: "test".into(),
            hidden_dim: 64,
            n_layers: 2,
            vocab_size: 1000,
            n_heads: 8,
            n_kv_heads: 2,
            intermediate_dim: 128,
            rope_theta: 10000.0,
            source_model: "test-model".into(),
            converted_at: "2026-04-06".into(),
            iara_version: "2.0".into(),
            quant_recipe: HashMap::new(),
            original_size_bytes: 0,
            converted_size_bytes: 0,
            compression_ratio: 1.0,
            epigenome: None,
        }
    }

    #[test]
    fn test_write_read_bf16_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_iara_bf16.iara");

        // Write
        let mut writer = IaraV2Writer::new(&path, test_metadata()).unwrap();
        let data = vec![0x3C, 0x00, 0x40, 0x00]; // Two BF16 values
        writer.write_bf16_tensor("test.weight", &[1, 2], &data).unwrap();
        writer.finalize().unwrap();

        // Read
        let reader = IaraV2Reader::open(&path).unwrap();
        assert_eq!(reader.header.tensors.len(), 1);
        let meta = reader.tensor_meta("test.weight").unwrap();
        assert_eq!(meta.dtype, IaraDtype::Bf16);
        assert_eq!(meta.shape, vec![1, 2]);
        let bytes = reader.tensor_data("test.weight").unwrap();
        assert_eq!(bytes, &data);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_write_read_fp8_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_iara_fp8.iara");

        let mut writer = IaraV2Writer::new(&path, test_metadata()).unwrap();
        let fp8_data = vec![0x3E, 0x42, 0x38, 0x45]; // 4 FP8 values
        let scales = vec![1.5f32];
        writer.write_fp8_tensor("attn.weight", &[1, 4], &fp8_data, &scales).unwrap();
        writer.finalize().unwrap();

        let reader = IaraV2Reader::open(&path).unwrap();
        let meta = reader.tensor_meta("attn.weight").unwrap();
        assert_eq!(meta.dtype, IaraDtype::Fp8E4m3);
        assert!(meta.scales_offset.is_some());

        let data = reader.tensor_data("attn.weight").unwrap();
        assert_eq!(data, &fp8_data);

        let scales_back = reader.tensor_scales_f32("attn.weight").unwrap();
        assert_eq!(scales_back.len(), 1);
        assert!((scales_back[0] - 1.5).abs() < 1e-6);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_write_read_ternary_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_iara_ternary.iara");

        let mut writer = IaraV2Writer::new(&path, test_metadata()).unwrap();
        let packed = vec![0b10_01_00_10, 0b01_10_10_00]; // 8 ternary values
        writer.write_ternary_tensor("mlp.weight", &[2, 4], &packed, 0.5, 0.25).unwrap();
        writer.finalize().unwrap();

        let reader = IaraV2Reader::open(&path).unwrap();
        let meta = reader.tensor_meta("mlp.weight").unwrap();
        assert_eq!(meta.dtype, IaraDtype::Ternary);
        assert_eq!(meta.quant_params.get("sparsity").unwrap(), "0.2500");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_multi_tensor_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_iara_multi.iara");

        let mut writer = IaraV2Writer::new(&path, test_metadata()).unwrap();

        // Write 3 tensors with different dtypes
        writer.write_bf16_tensor("embed", &[1000, 64], &vec![0u8; 128000]).unwrap();
        writer.write_fp8_tensor("attn.q", &[64, 64],
            &vec![0x40u8; 4096], &vec![1.0f32; 64]).unwrap();
        writer.write_ternary_tensor("mlp.gate", &[128, 64],
            &vec![0b10_01_00_10; 2048], 0.3, 0.3).unwrap();

        writer.finalize().unwrap();

        let reader = IaraV2Reader::open(&path).unwrap();
        assert_eq!(reader.header.tensors.len(), 3);
        assert_eq!(reader.tensor_meta("embed").unwrap().dtype, IaraDtype::Bf16);
        assert_eq!(reader.tensor_meta("attn.q").unwrap().dtype, IaraDtype::Fp8E4m3);
        assert_eq!(reader.tensor_meta("mlp.gate").unwrap().dtype, IaraDtype::Ternary);

        let summary = reader.summary();
        assert!(summary.contains("test-model"));
        assert!(summary.contains("Tensors: 3"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_quant_recipe_fp8() {
        let recipe = QuantRecipe::fp8_standard();
        assert_eq!(recipe.dtype_for("model.layers.0.self_attn.q_proj.weight"), IaraDtype::Fp8E4m3);
        assert_eq!(recipe.dtype_for("model.embed_tokens.weight"), IaraDtype::Bf16);
        assert_eq!(recipe.dtype_for("lm_head.weight"), IaraDtype::Bf16);
        assert_eq!(recipe.dtype_for("model.layers.0.input_layernorm.weight"), IaraDtype::Bf16);
    }

    #[test]
    fn test_quant_recipe_aggressive() {
        let recipe = QuantRecipe::aggressive();
        assert_eq!(recipe.dtype_for("model.layers.0.mlp.gate_proj.weight"), IaraDtype::Ternary);
        assert_eq!(recipe.dtype_for("model.layers.0.self_attn.q_proj.weight"), IaraDtype::Fp8E4m3);
        assert_eq!(recipe.dtype_for("model.embed_tokens.weight"), IaraDtype::Bf16);
    }

    #[test]
    fn test_dtype_info() {
        assert_eq!(IaraDtype::Bf16.bytes_per_element(), 2.0);
        assert_eq!(IaraDtype::Fp8E4m3.bytes_per_element(), 1.0);
        assert_eq!(IaraDtype::Ternary.bytes_per_element(), 0.25);
        assert_eq!(IaraDtype::Q4.bytes_per_element(), 0.5);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    #[test]
    fn test_quant_recipe_pulse() {
        // 32-layer model: N/4 = 8, 3N/4 = 24
        let recipe = QuantRecipe::pulse(32);

        // Layers 0-7: Ternary
        assert_eq!(recipe.dtype_for("model.layers.0.mlp.weight"), IaraDtype::Ternary);
        assert_eq!(recipe.dtype_for("model.layers.7.mlp.weight"), IaraDtype::Ternary);

        // Layers 8-23: FP8
        assert_eq!(recipe.dtype_for("model.layers.8.self_attn.weight"), IaraDtype::Fp8E4m3);
        assert_eq!(recipe.dtype_for("model.layers.15.mlp.weight"), IaraDtype::Fp8E4m3);
        assert_eq!(recipe.dtype_for("model.layers.23.self_attn.weight"), IaraDtype::Fp8E4m3);

        // Layers 24-31: BF16
        assert_eq!(recipe.dtype_for("model.layers.24.mlp.weight"), IaraDtype::Bf16);
        assert_eq!(recipe.dtype_for("model.layers.31.self_attn.weight"), IaraDtype::Bf16);

        // Critical layers always BF16
        assert_eq!(recipe.dtype_for("model.embed_tokens.weight"), IaraDtype::Bf16);
        assert_eq!(recipe.dtype_for("lm_head.weight"), IaraDtype::Bf16);
        assert_eq!(recipe.dtype_for("model.norm.weight"), IaraDtype::Bf16);
    }

    #[test]
    fn test_layer_index_extraction() {
        assert_eq!(QuantRecipe::extract_layer_index("model.layers.5.self_attn.q_proj.weight"), Some(5));
        assert_eq!(QuantRecipe::extract_layer_index("model.layers.0.mlp.gate_proj.weight"), Some(0));
        assert_eq!(QuantRecipe::extract_layer_index("model.layers.31.attention.weight"), Some(31));
        assert_eq!(QuantRecipe::extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(QuantRecipe::extract_layer_index("lm_head.weight"), None);
    }
}
