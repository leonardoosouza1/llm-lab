//! Safetensors weight loading for burn models.
//!
//! Reads the safetensors binary format (header + tensor data) and produces
//! `Vec<f32>` buffers that can be used to construct burn tensors. Supports
//! BF16 -> F32 and F32 -> F32 conversion.
//!
//! ## Format
//!
//! ```text
//! [8 bytes: header_len as u64 LE]
//! [header_len bytes: JSON header]
//! [tensor data bytes...]
//! ```
//!
//! The JSON header maps tensor names to metadata:
//! ```json
//! {
//!   "model.layers.0.self_attn.q_proj.weight": {
//!     "dtype": "BF16",
//!     "shape": [4096, 4096],
//!     "data_offsets": [0, 33554432]
//!   },
//!   "__metadata__": { "format": "pt" }
//! }
//! ```

use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// The data type string (e.g. "BF16", "F32", "F16").
    pub dtype: String,
    /// The shape dimensions.
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

/// Errors that can occur during safetensors loading.
#[derive(Debug)]
pub enum SafetensorsError {
    /// IO error while reading the file.
    Io(io::Error),
    /// The file is too small to contain a valid header.
    FileTooSmall,
    /// JSON header could not be parsed.
    InvalidHeader(String),
    /// Requested tensor name was not found in the file.
    TensorNotFound(String),
    /// The tensor dtype is not supported for conversion.
    UnsupportedDtype(String),
    /// Data offsets are out of bounds for the file.
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
            Self::OffsetOutOfBounds {
                tensor,
                offset_end,
                data_len,
            } => write!(
                f,
                "tensor {tensor}: offset end {offset_end} exceeds data length {data_len}"
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
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, SafetensorsError>;

/// Parse the safetensors JSON header from a file path.
///
/// Returns a map of tensor names to their metadata (dtype, shape, offsets).
/// The special `__metadata__` key is skipped.
pub fn load_safetensors_header(path: &Path) -> Result<HashMap<String, TensorMeta>> {
    let mut file = File::open(path)?;

    // Read the 8-byte header length
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    // Read the JSON header
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;

    let raw: HashMap<String, RawTensorEntry> = serde_json::from_slice(&header_buf)
        .map_err(|e| SafetensorsError::InvalidHeader(e.to_string()))?;

    let mut tensors = HashMap::with_capacity(raw.len());
    for (name, entry) in raw {
        // Skip the metadata entry
        if name == "__metadata__" {
            continue;
        }

        let dtype = entry
            .dtype
            .ok_or_else(|| SafetensorsError::InvalidHeader(format!("missing dtype for {name}")))?;
        let shape = entry
            .shape
            .ok_or_else(|| SafetensorsError::InvalidHeader(format!("missing shape for {name}")))?;
        let data_offsets = entry.data_offsets.ok_or_else(|| {
            SafetensorsError::InvalidHeader(format!("missing data_offsets for {name}"))
        })?;

        tensors.insert(
            name,
            TensorMeta {
                dtype,
                shape,
                data_offsets,
            },
        );
    }

    Ok(tensors)
}

/// Load a single tensor from a safetensors file as a Vec<f32>.
///
/// Supported source dtypes: BF16, F16, F32.
/// BF16 and F16 values are converted to F32 on the fly.
///
/// # Arguments
///
/// * `path` - Path to the `.safetensors` file.
/// * `name` - The tensor name as it appears in the header.
pub fn load_safetensors_tensor(path: &Path, name: &str) -> Result<Vec<f32>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    if mmap.len() < 8 {
        return Err(SafetensorsError::FileTooSmall);
    }

    // Parse header length
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_len;

    if mmap.len() < data_start {
        return Err(SafetensorsError::FileTooSmall);
    }

    // Parse the header to find our tensor
    let raw: HashMap<String, RawTensorEntry> =
        serde_json::from_slice(&mmap[8..data_start])
            .map_err(|e| SafetensorsError::InvalidHeader(e.to_string()))?;

    let entry = raw
        .get(name)
        .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

    let dtype = entry
        .dtype
        .as_deref()
        .ok_or_else(|| SafetensorsError::InvalidHeader(format!("missing dtype for {name}")))?;
    let shape = entry
        .shape
        .as_ref()
        .ok_or_else(|| SafetensorsError::InvalidHeader(format!("missing shape for {name}")))?;
    let offsets = entry.data_offsets.ok_or_else(|| {
        SafetensorsError::InvalidHeader(format!("missing data_offsets for {name}"))
    })?;

    let numel: usize = shape.iter().product();
    let abs_start = data_start + offsets[0];
    let abs_end = data_start + offsets[1];

    if abs_end > mmap.len() {
        return Err(SafetensorsError::OffsetOutOfBounds {
            tensor: name.to_string(),
            offset_end: abs_end,
            data_len: mmap.len(),
        });
    }

    let data_slice = &mmap[abs_start..abs_end];

    match dtype {
        "BF16" => Ok(convert_bf16_to_f32(data_slice, numel)),
        "F16" => Ok(convert_f16_to_f32(data_slice, numel)),
        "F32" => Ok(convert_f32_direct(data_slice, numel)),
        other => Err(SafetensorsError::UnsupportedDtype(other.to_string())),
    }
}

/// Convert a byte slice of BF16 values to f32.
///
/// BF16 is the upper 16 bits of an IEEE 754 float32. To convert:
/// interpret the u16 as the high 16 bits of a u32, then reinterpret as f32.
fn convert_bf16_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(numel);

    for i in 0..numel {
        let offset = i * 2;
        let bits16 = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let bits32 = (bits16 as u32) << 16;
        output.push(f32::from_bits(bits32));
    }

    output
}

/// Convert a byte slice of IEEE 754 half-precision (F16) values to f32.
///
/// F16 layout: 1 sign bit, 5 exponent bits, 10 mantissa bits.
/// Handles normals, denormals, infinities, and NaNs.
fn convert_f16_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(numel);

    for i in 0..numel {
        let offset = i * 2;
        let bits16 = u16::from_le_bytes([data[offset], data[offset + 1]]);
        output.push(f16_to_f32(bits16));
    }

    output
}

/// Convert a single F16 u16 representation to f32.
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x3FF) as u32;

    let f32_bits = if exponent == 0 {
        if mantissa == 0 {
            // Signed zero
            sign << 31
        } else {
            // Denormalized: convert to normalized f32
            // Find the leading 1 in the mantissa
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
        // Inf or NaN
        (sign << 31) | (0xFF << 23) | (mantissa << 13)
    } else {
        // Normalized
        let exponent_adjusted = exponent - 15 + 127;
        (sign << 31) | (exponent_adjusted << 23) | (mantissa << 13)
    };

    f32::from_bits(f32_bits)
}

/// Read f32 values directly from a byte slice (little-endian).
fn convert_f32_direct(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(numel);

    for i in 0..numel {
        let offset = i * 4;
        let bits = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        output.push(f32::from_bits(bits));
    }

    output
}

/// Load all tensors from a safetensors file into a map of name -> Vec<f32>.
///
/// This is a convenience function that loads every tensor in the file.
/// For large models, prefer `load_safetensors_tensor` to load one at a time.
pub fn load_all_tensors(path: &Path) -> Result<HashMap<String, (TensorMeta, Vec<f32>)>> {
    let header = load_safetensors_header(path)?;
    let mut result = HashMap::with_capacity(header.len());

    for (name, meta) in &header {
        let data = load_safetensors_tensor(path, name)?;
        result.insert(name.clone(), (meta.clone(), data));
    }

    Ok(result)
}

/// Map a safetensors tensor name (e.g. "model.layers.0.self_attn.q_proj.weight")
/// to a burn module parameter path.
///
/// This function provides a default mapping convention. Projects should customize
/// the mapping based on their model architecture.
///
/// # Examples
///
/// ```text
/// "model.embed_tokens.weight"               -> "embed_tokens.weight"
/// "model.layers.0.self_attn.q_proj.weight"  -> "layers.0.self_attn.q_proj.weight"
/// "lm_head.weight"                          -> "lm_head.weight"
/// ```
pub fn map_safetensors_name(name: &str) -> String {
    // Strip the common "model." prefix used by HuggingFace models
    if let Some(rest) = name.strip_prefix("model.") {
        rest.to_string()
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Create a minimal safetensors file in a temp directory for testing.
    /// Returns the path to the file. Caller is responsible for cleanup.
    fn create_test_safetensors(tensors: &[(&str, &str, &[usize], &[u8])]) -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("iara_test_{id}_{}.safetensors", std::process::id()));

        // Build the header JSON
        let mut header_map = serde_json::Map::new();
        let mut current_offset = 0usize;

        for &(name, dtype, shape, data) in tensors {
            let mut entry = serde_json::Map::new();
            entry.insert("dtype".to_string(), serde_json::Value::String(dtype.to_string()));
            entry.insert(
                "shape".to_string(),
                serde_json::Value::Array(
                    shape.iter().map(|&s| serde_json::Value::Number(s.into())).collect(),
                ),
            );
            let end_offset = current_offset + data.len();
            entry.insert(
                "data_offsets".to_string(),
                serde_json::Value::Array(vec![
                    serde_json::Value::Number(current_offset.into()),
                    serde_json::Value::Number(end_offset.into()),
                ]),
            );
            header_map.insert(name.to_string(), serde_json::Value::Object(entry));
            current_offset = end_offset;
        }

        let header_json = serde_json::to_vec(&header_map).unwrap();
        let header_len = header_json.len() as u64;

        let mut file = File::create(&path).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(&header_json).unwrap();
        for &(_, _, _, data) in tensors {
            file.write_all(data).unwrap();
        }
        file.flush().unwrap();

        path
    }

    #[test]
    fn test_load_f32_tensor() {
        // A 2x2 F32 tensor: [[1.0, 2.0], [3.0, 4.0]]
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let file = create_test_safetensors(&[("test_tensor", "F32", &[2, 2], &data)]);

        let header = load_safetensors_header(&file).unwrap();
        assert!(header.contains_key("test_tensor"));
        let meta = &header["test_tensor"];
        assert_eq!(meta.dtype, "F32");
        assert_eq!(meta.shape, vec![2, 2]);
        assert_eq!(meta.numel(), 4);

        let loaded = load_safetensors_tensor(&file, "test_tensor").unwrap();
        assert_eq!(loaded, values.to_vec());
    }

    #[test]
    fn test_load_bf16_tensor() {
        // BF16 representation of some known values
        // 1.0 in BF16 = 0x3F80, 2.0 = 0x4000, -1.0 = 0xBF80, 0.0 = 0x0000
        let bf16_values: Vec<u16> = vec![0x3F80, 0x4000, 0xBF80, 0x0000];
        let expected_f32: Vec<f32> = vec![1.0, 2.0, -1.0, 0.0];

        let mut data = Vec::new();
        for &v in &bf16_values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let file = create_test_safetensors(&[("bf16_tensor", "BF16", &[4], &data)]);

        let loaded = load_safetensors_tensor(&file, "bf16_tensor").unwrap();
        assert_eq!(loaded.len(), 4);
        for (got, expected) in loaded.iter().zip(expected_f32.iter()) {
            assert!(
                (got - expected).abs() < 1e-6,
                "BF16 conversion: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_bf16_conversion_edge_cases() {
        // Test specific BF16 -> F32 conversions
        // BF16 0x7F80 = +Inf
        assert!(convert_bf16_to_f32(&0x7F80u16.to_le_bytes(), 1)[0].is_infinite());
        // BF16 0xFF80 = -Inf
        assert!(convert_bf16_to_f32(&0xFF80u16.to_le_bytes(), 1)[0].is_infinite());
        // BF16 0x7FC0 = NaN
        assert!(convert_bf16_to_f32(&0x7FC0u16.to_le_bytes(), 1)[0].is_nan());
        // BF16 0x0000 = 0.0
        assert_eq!(convert_bf16_to_f32(&0x0000u16.to_le_bytes(), 1)[0], 0.0);
    }

    #[test]
    fn test_tensor_not_found() {
        let file =
            create_test_safetensors(&[("existing", "F32", &[1], &1.0f32.to_le_bytes())]);

        let result = load_safetensors_tensor(&file, "nonexistent");
        assert!(matches!(result, Err(SafetensorsError::TensorNotFound(_))));
    }

    #[test]
    fn test_name_mapping() {
        assert_eq!(
            map_safetensors_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            map_safetensors_name("model.embed_tokens.weight"),
            "embed_tokens.weight"
        );
        assert_eq!(
            map_safetensors_name("lm_head.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn test_metadata_entry_skipped() {
        // Header with a __metadata__ entry should not appear in the result
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir()
            .join(format!("iara_test_meta_{id}_{}.safetensors", std::process::id()));

        let header = r#"{"__metadata__":{"format":"pt"},"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file = File::create(&path).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(header_bytes).unwrap();
        file.write_all(&1.0f32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let result = load_safetensors_header(&path).unwrap();
        assert!(!result.contains_key("__metadata__"));
        assert!(result.contains_key("t"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_multiple_tensors() {
        let data_a: Vec<u8> = [1.0f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let data_b: Vec<u8> = [3.0f32, 4.0, 5.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let file = create_test_safetensors(&[
            ("tensor_a", "F32", &[2], &data_a),
            ("tensor_b", "F32", &[3], &data_b),
        ]);

        let a = load_safetensors_tensor(&file, "tensor_a").unwrap();
        let b = load_safetensors_tensor(&file, "tensor_b").unwrap();

        assert_eq!(a, vec![1.0, 2.0]);
        assert_eq!(b, vec![3.0, 4.0, 5.0]);
    }
}
