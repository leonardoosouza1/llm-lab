// ============================================================================
// safetensors.rs — Safetensors header parsing and tensor I/O
// ============================================================================
// Consolidated from iara-engine's storage/mod.rs and iara-train's
// export/safetensors.rs. Provides the shared TensorMeta struct and
// header parsing used by both inference and training.
// ============================================================================

use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Data type string (e.g. "BF16", "F32", "F16").
    pub dtype: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Byte offsets [start, end) relative to the beginning of the data section.
    pub data_offsets: [usize; 2],
}

impl TensorMeta {
    /// Number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size in bytes of a single element for the given dtype.
    pub fn element_size(&self) -> usize {
        match self.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            "F64" => 8,
            "I32" => 4,
            "I64" => 8,
            "I16" => 2,
            "I8" | "U8" | "BOOL" => 1,
            _ => 0,
        }
    }
}

/// Raw JSON representation matching the safetensors header format.
#[derive(Deserialize)]
struct RawTensorEntry {
    dtype: Option<String>,
    shape: Option<Vec<usize>>,
    data_offsets: Option<[usize; 2]>,
}

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum SafetensorsError {
    Io(io::Error),
    FileTooSmall,
    InvalidHeader(String),
    TensorNotFound(String),
    UnsupportedDtype(String),
    OffsetOutOfBounds {
        tensor: String,
        offset_end: usize,
        data_len: usize,
    },
}

impl std::fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::FileTooSmall => write!(f, "file too small for safetensors header"),
            Self::InvalidHeader(msg) => write!(f, "invalid safetensors header: {msg}"),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            Self::UnsupportedDtype(dtype) => write!(f, "unsupported dtype: {dtype}"),
            Self::OffsetOutOfBounds { tensor, offset_end, data_len } => write!(
                f, "tensor {tensor}: offset end {offset_end} exceeds data length {data_len}"
            ),
        }
    }
}

impl std::error::Error for SafetensorsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for SafetensorsError {
    fn from(e: io::Error) -> Self { Self::Io(e) }
}

pub type Result<T> = std::result::Result<T, SafetensorsError>;

// ── Header parsing ───────────────────────────────────────────────────────────

/// Parse the safetensors JSON header and memory-map the file.
///
/// Returns (tensor_name -> TensorMeta, Mmap). The Mmap keeps the file mapped
/// so callers can read tensor data at the offsets specified in TensorMeta.
/// Data offsets in TensorMeta are relative to the data section (after header).
/// To get absolute mmap offsets, add `8 + header_len`.
pub fn parse_header(path: &Path) -> Result<(HashMap<String, TensorMeta>, Mmap)> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    if mmap.len() < 8 {
        return Err(SafetensorsError::FileTooSmall);
    }

    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;

    if mmap.len() < header_end {
        return Err(SafetensorsError::FileTooSmall);
    }

    let raw: HashMap<String, RawTensorEntry> = serde_json::from_slice(&mmap[8..header_end])
        .map_err(|e| SafetensorsError::InvalidHeader(e.to_string()))?;

    let mut tensors = HashMap::with_capacity(raw.len());
    for (name, entry) in raw {
        if name == "__metadata__" {
            continue;
        }

        let dtype = entry.dtype
            .ok_or_else(|| SafetensorsError::InvalidHeader(format!("missing dtype for {name}")))?;
        let shape = entry.shape
            .ok_or_else(|| SafetensorsError::InvalidHeader(format!("missing shape for {name}")))?;
        let data_offsets = entry.data_offsets
            .ok_or_else(|| SafetensorsError::InvalidHeader(format!("missing data_offsets for {name}")))?;

        tensors.insert(name, TensorMeta { dtype, shape, data_offsets });
    }

    Ok((tensors, mmap))
}

/// Parse only the header (without keeping the mmap).
pub fn load_header(path: &Path) -> Result<HashMap<String, TensorMeta>> {
    let (tensors, _mmap) = parse_header(path)?;
    Ok(tensors)
}

/// Compute the absolute byte offset of the data section start in a safetensors mmap.
///
/// The data section begins at offset `8 + header_len`.
pub fn data_offset(mmap: &Mmap) -> Result<usize> {
    if mmap.len() < 8 {
        return Err(SafetensorsError::FileTooSmall);
    }
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    Ok(8 + header_len)
}

// ── Writing ──────────────────────────────────────────────────────────────────

/// Write tensors to a safetensors file.
///
/// Each tensor is (name, f32_data, shape). All tensors are written as F32.
pub fn write_safetensors(
    path: &Path,
    tensors: &[(String, Vec<f32>, Vec<usize>)],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;

    for (name, data, shape) in tensors {
        let byte_len = data.len() * 4;
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::String("F32".into()));
        entry.insert("shape".into(), serde_json::Value::Array(
            shape.iter().map(|&s| serde_json::Value::Number(s.into())).collect(),
        ));
        entry.insert("data_offsets".into(), serde_json::Value::Array(vec![
            serde_json::Value::Number(offset.into()),
            serde_json::Value::Number((offset + byte_len).into()),
        ]));
        header.insert(name.clone(), serde_json::Value::Object(entry));
        offset += byte_len;
    }

    let header_json = serde_json::to_vec(&header)?;
    let header_len = header_json.len() as u64;

    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(&header_json)?;

    for (_, data, _) in tensors {
        for &val in data {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    file.flush()?;
    Ok(())
}

// ── Dtype conversion helpers ─────────────────────────────────────────────────

/// Convert a byte slice of BF16 values to f32.
pub fn convert_bf16_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(numel);
    for i in 0..numel {
        let offset = i * 2;
        let bits16 = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let bits32 = (bits16 as u32) << 16;
        output.push(f32::from_bits(bits32));
    }
    output
}

/// Convert a byte slice of F16 values to f32.
pub fn convert_f16_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(numel);
    for i in 0..numel {
        let offset = i * 2;
        let bits16 = u16::from_le_bytes([data[offset], data[offset + 1]]);
        output.push(f16_to_f32(bits16));
    }
    output
}

/// Convert a single F16 u16 to f32. Handles normals, denormals, inf, NaN.
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x3FF) as u32;

    let f32_bits = if exponent == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            let mut m = mantissa;
            let mut e: i32 = -1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let mantissa_shifted = (m & 0x3FF) << 13;
            let exponent_adjusted = (127 - 15 - e as u32) & 0xFF;
            (sign << 31) | (exponent_adjusted << 23) | mantissa_shifted
        }
    } else if exponent == 31 {
        (sign << 31) | (0xFF << 23) | (mantissa << 13)
    } else {
        let exponent_adjusted = exponent - 15 + 127;
        (sign << 31) | (exponent_adjusted << 23) | (mantissa << 13)
    };

    f32::from_bits(f32_bits)
}

/// Read f32 values directly from a byte slice (little-endian).
pub fn convert_f32_direct(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(numel);
    for i in 0..numel {
        let offset = i * 4;
        let bits = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        ]);
        output.push(f32::from_bits(bits));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_parse_roundtrip() {
        let dir = std::env::temp_dir().join("compute_core_test_st");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");

        let tensors = vec![
            ("weight".to_string(), vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]),
            ("bias".to_string(), vec![0.1f32, 0.2], vec![2]),
        ];

        write_safetensors(&path, &tensors).unwrap();

        let (header, mmap) = parse_header(&path).unwrap();
        assert!(header.contains_key("weight"));
        assert!(header.contains_key("bias"));

        let w = &header["weight"];
        assert_eq!(w.dtype, "F32");
        assert_eq!(w.shape, vec![2, 2]);
        assert_eq!(w.numel(), 4);

        // Verify data can be read back
        let data_start = data_offset(&mmap).unwrap();
        let abs_start = data_start + w.data_offsets[0];
        let abs_end = data_start + w.data_offsets[1];
        let loaded = convert_f32_direct(&mmap[abs_start..abs_end], w.numel());
        assert_eq!(loaded, vec![1.0, 2.0, 3.0, 4.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_bf16_conversion() {
        // BF16 1.0 = 0x3F80
        let data = 0x3F80u16.to_le_bytes();
        let result = convert_bf16_to_f32(&data, 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_conversion() {
        // F16 1.0 = 0x3C00
        let data = 0x3C00u16.to_le_bytes();
        let result = convert_f16_to_f32(&data, 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }
}
