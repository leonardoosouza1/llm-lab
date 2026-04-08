// ============================================================================
// storage/gguf.rs — Leitor de arquivos GGUF (formato do llama.cpp / Ollama)
// ============================================================================
//
// Suporta GGUF v3 com dequantização Q4_K, Q5_K, Q6_K, Q8_0, F16, F32, BF16.
// Usado para carregar modelos pré-quantizados do Ollama diretamente.
//
// Referência: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
// ============================================================================

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use half::bf16;

// ── GGUF Constants ──────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in LE

/// GGML tensor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 14,
    Q6_K    = 15,
    Q8_K    = 16,
    IQ2_XXS = 17,
    IQ2_XS  = 18,
    IQ3_XXS = 19,
    IQ1_S   = 20,
    IQ4_NL  = 21,
    IQ3_S   = 22,
    IQ2_S   = 23,
    IQ4_XS  = 24,
    BF16    = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            14 => Some(Self::Q5_K),
            15 => Some(Self::Q6_K),
            16 => Some(Self::Q8_K),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Block size (number of values per quantization block)
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            _ => 256,
        }
    }

    /// Bytes per block
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 2 + 16,   // f16 scale + 16 bytes (32 nibbles)
            Self::Q4_1 => 4 + 16,   // f16 scale + f16 min + 16 bytes
            Self::Q5_0 => 2 + 4 + 16, // f16 scale + 4 hi-bits + 16 bytes
            Self::Q5_1 => 4 + 4 + 16,
            Self::Q8_0 => 2 + 32,   // f16 scale + 32 bytes
            Self::Q8_1 => 8 + 32,   // f16 scale + f16 sum + 32 bytes (but sum not always present)
            Self::Q4_K => 2 + 2 + 12 + 128, // d(f16) + dmin(f16) + scales(12) + qs(128) = 144
            Self::Q5_K => 2 + 2 + 12 + 32 + 128, // d + dmin + scales + qh + qs = 176
            Self::Q6_K => 128 + 64 + 16 + 2, // ql(128) + qh(64) + scales(16) + d(f16) = 210
            Self::Q8_K => 4 + 256 + 16, // d(f32) + qs(256) + bsums(16) = 276
            _ => 144, // fallback
        }
    }
}

// ── GGUF Value Types ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

// ── Tensor Info ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    pub offset: u64, // offset from start of tensor data section
}

impl GgufTensorInfo {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn n_bytes(&self) -> usize {
        let n_elem = self.n_elements();
        let bs = self.dtype.block_size();
        let n_blocks = (n_elem + bs - 1) / bs;
        n_blocks * self.dtype.block_bytes()
    }
}

// ── GGUF File Reader ────────────────────────────────────────────────────────

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub tensor_data_offset: u64, // absolute offset in file where tensor data starts
    data: memmap2::Mmap,
}

impl GgufFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let data = unsafe { memmap2::Mmap::map(&file)? };

        let mut cursor = io::Cursor::new(&data[..]);

        // ── Header ──
        let magic = read_u32(&mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a GGUF file"));
        }
        let version = read_u32(&mut cursor)?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version: {}", version)));
        }
        let n_tensors = read_u64(&mut cursor)? as usize;
        let n_kv = read_u64(&mut cursor)? as usize;

        // ── KV Pairs ──
        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = read_gguf_string(&mut cursor)?;
            let val = read_gguf_value(&mut cursor)?;
            metadata.insert(key, val);
        }

        // ── Tensor Infos ──
        let mut tensors = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = read_u32(&mut cursor)? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u64(&mut cursor)? as usize);
            }
            let dtype_raw = read_u32(&mut cursor)?;
            let dtype = GgmlType::from_u32(dtype_raw)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData,
                    format!("Unknown GGML type: {} for tensor '{}'", dtype_raw, name)))?;
            let offset = read_u64(&mut cursor)?;
            tensors.insert(name.clone(), GgufTensorInfo { name, shape, dtype, offset });
        }

        // ── Alignment ──
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;

        let cur_pos = cursor.position();
        let tensor_data_offset = ((cur_pos + alignment - 1) / alignment) * alignment;

        log::info!("📦 GGUF v{}: {} tensors, {} KV pairs, data@{:#x}",
            version, tensors.len(), metadata.len(), tensor_data_offset);

        Ok(Self { version, metadata, tensors, tensor_data_offset, data })
    }

    /// Get raw bytes for a tensor
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let start = self.tensor_data_offset as usize + info.offset as usize;
        let end = start + info.n_bytes();
        if end > self.data.len() {
            log::error!("Tensor '{}' extends beyond file: {}+{} > {}", name, start, info.n_bytes(), self.data.len());
            return None;
        }
        Some(&self.data[start..end])
    }

    /// Dequantize a tensor to F32
    pub fn dequantize_f32(&self, name: &str) -> Option<Vec<f32>> {
        let info = self.tensors.get(name)?;
        let bytes = self.tensor_bytes(name)?;
        let n_elem = info.n_elements();

        match info.dtype {
            GgmlType::F32 => {
                let floats = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n_elem)
                };
                Some(floats.to_vec())
            }
            GgmlType::F16 => {
                let halfs = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const u16, n_elem)
                };
                Some(halfs.iter().map(|&bits| half::f16::from_bits(bits).to_f32()).collect())
            }
            GgmlType::BF16 => {
                let halfs = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const u16, n_elem)
                };
                Some(halfs.iter().map(|&bits| bf16::from_bits(bits).to_f32()).collect())
            }
            GgmlType::Q8_0 => Some(dequantize_q8_0(bytes, n_elem)),
            GgmlType::Q4_K => Some(dequantize_q4_k(bytes, n_elem)),
            GgmlType::Q5_K => Some(dequantize_q5_k(bytes, n_elem)),
            GgmlType::Q6_K => Some(dequantize_q6_k(bytes, n_elem)),
            other => {
                log::warn!("Unsupported GGML type for dequant: {:?}", other);
                None
            }
        }
    }

    /// Dequantize to BF16 bits (for direct GPU upload)
    pub fn dequantize_bf16(&self, name: &str) -> Option<Vec<u16>> {
        let f32_data = self.dequantize_f32(name)?;
        Some(f32_data.iter().map(|&f| bf16::from_f32(f).to_bits()).collect())
    }

    /// Get a metadata string value
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.as_str()
    }

    /// Get a metadata u32 value
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key)?.as_u32()
    }

    /// Get a metadata f32 value
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key)?.as_f32()
    }
}

// ── Q4_K Dequantization (ported from GGML) ──────────────────────────────────
//
// Block structure (144 bytes per 256 values):
//   d:      FP16 super-block scale for quantized scales
//   dmin:   FP16 super-block scale for quantized mins
//   scales: 12 bytes, 8 pairs of 6-bit (scale, min)
//   qs:     128 bytes, 256 4-bit values (2 per byte)
//
// Dequant: value = d * sub_scale * q4_value - dmin * sub_min

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

fn dequantize_q4_k(data: &[u8], n_elem: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 144; // 2+2+12+128

    let n_blocks = n_elem / QK_K;
    let mut output = Vec::with_capacity(n_elem);

    for i in 0..n_blocks {
        let block = &data[i * BLOCK_SIZE..];

        // Read d and dmin as FP16
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
        let d = half::f16::from_bits(d_bits).to_f32();
        let dmin = half::f16::from_bits(dmin_bits).to_f32();

        let scales = &block[4..16]; // 12 bytes
        let qs = &block[16..144];   // 128 bytes

        let mut is = 0usize;
        let mut q_idx = 0usize;

        // Process 4 groups of 64 values each (= 256 total)
        for _j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;

            // First 32 values: low nibbles
            for l in 0..32 {
                let q = (qs[q_idx + l] & 0x0F) as f32;
                output.push(d1 * q - m1f);
            }
            // Next 32 values: high nibbles
            for l in 0..32 {
                let q = (qs[q_idx + l] >> 4) as f32;
                output.push(d2 * q - m2f);
            }

            q_idx += 32;
            is += 2;
        }
    }

    output
}

// ── Q5_K Dequantization ─────────────────────────────────────────────────────
//
// Block structure (176 bytes per 256 values):
//   d:      FP16 super-block scale
//   dmin:   FP16 super-block scale for mins
//   scales: 12 bytes, packed 6-bit scale/min pairs
//   qh:     32 bytes, high bits (1 bit per value)
//   qs:     128 bytes, low 4 bits (2 per byte)

fn dequantize_q5_k(data: &[u8], n_elem: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 176; // 2+2+12+32+128

    let n_blocks = n_elem / QK_K;
    let mut output = Vec::with_capacity(n_elem);

    for i in 0..n_blocks {
        let block = &data[i * BLOCK_SIZE..];

        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
        let d = half::f16::from_bits(d_bits).to_f32();
        let dmin = half::f16::from_bits(dmin_bits).to_f32();

        let scales = &block[4..16];   // 12 bytes
        let qh = &block[16..48];       // 32 bytes (256 bits, 1 per value)
        let qs = &block[48..176];      // 128 bytes

        let mut is = 0usize;
        let mut q_idx = 0usize;
        let mut qh_idx = 0usize;

        for _j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;

            // First 32 values (low nibble + high bit)
            for l in 0..32 {
                let lo = (qs[q_idx + l] & 0x0F) as u8;
                let hi = ((qh[qh_idx + l / 8] >> (l % 8)) & 1) as u8;
                let q = (lo | (hi << 4)) as f32;
                output.push(d1 * q - m1f);
            }
            // Next 32 values (high nibble + high bit)
            for l in 0..32 {
                let lo = (qs[q_idx + l] >> 4) as u8;
                let hi = ((qh[qh_idx + (l + 32) / 8] >> ((l + 32) % 8)) & 1) as u8;
                let q = (lo | (hi << 4)) as f32;
                output.push(d2 * q - m2f);
            }

            q_idx += 32;
            qh_idx += 8; // 64 bits per group of 64 values
            is += 2;
        }
    }

    output
}

// ── Q6_K Dequantization ─────────────────────────────────────────────────────
//
// Block structure (210 bytes per 256 values):
//   ql: 128 bytes (low 4 bits, 2 per byte)
//   qh:  64 bytes (high 2 bits, 4 per byte)
//   scales: 16 bytes (int8 scales, one per 16 values)
//   d: FP16 super-block scale

fn dequantize_q6_k(data: &[u8], n_elem: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 210;

    let n_blocks = n_elem / QK_K;
    let mut output = Vec::with_capacity(n_elem);

    for i in 0..n_blocks {
        let block = &data[i * BLOCK_SIZE..];

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208]; // 16 int8 scales
        let d_bits = u16::from_le_bytes([block[208], block[209]]);
        let d = half::f16::from_bits(d_bits).to_f32();

        for n in 0..QK_K {
            // Extract 6-bit value from ql (4 low bits) and qh (2 high bits)
            let ql_idx = n / 2;
            let ql_val = if n % 2 == 0 {
                ql[ql_idx] & 0x0F
            } else {
                ql[ql_idx] >> 4
            };

            let qh_idx = n / 4;
            let qh_shift = (n % 4) * 2;
            let qh_val = (qh[qh_idx] >> qh_shift) & 0x03;

            let q = ((ql_val | (qh_val << 4)) as i8 - 32) as f32;
            let sc = scales[n / 16] as i8 as f32;

            output.push(d * sc * q);
        }
    }

    output
}

// ── Q8_0 Dequantization ─────────────────────────────────────────────────────
//
// Block structure (34 bytes per 32 values):
//   d: FP16 scale
//   qs: 32 int8 values

fn dequantize_q8_0(data: &[u8], n_elem: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 34; // 2 + 32
    const QK: usize = 32;

    let n_blocks = n_elem / QK;
    let mut output = Vec::with_capacity(n_elem);

    for i in 0..n_blocks {
        let block = &data[i * BLOCK_SIZE..];
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let d = half::f16::from_bits(d_bits).to_f32();

        for j in 0..QK {
            let q = block[2 + j] as i8 as f32;
            output.push(d * q);
        }
    }

    output
}

// ── IO Helpers ──────────────────────────────────────────────────────────────

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_gguf_value<R: Read>(r: &mut R) -> io::Result<GgufValue> {
    let vtype = read_u32(r)?;
    match vtype {
        0 => { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(GgufValue::U8(b[0])) }
        1 => { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(GgufValue::I8(b[0] as i8)) }
        2 => { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(GgufValue::U16(u16::from_le_bytes(b))) }
        3 => { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(GgufValue::I16(i16::from_le_bytes(b))) }
        4 => Ok(GgufValue::U32(read_u32(r)?)),
        5 => Ok(GgufValue::I32(read_i32(r)?)),
        6 => Ok(GgufValue::F32(read_f32(r)?)),
        7 => { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(GgufValue::Bool(b[0] != 0)) }
        8 => Ok(GgufValue::Str(read_gguf_string(r)?)),
        9 => {
            // Array: elem_type(u32) + n(u64) + values
            let elem_type = read_u32(r)?;
            let n = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(n.min(1024)); // cap initial alloc
            for _ in 0..n {
                let val = read_gguf_array_element(r, elem_type)?;
                arr.push(val);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => { let v = read_u64(r)?; Ok(GgufValue::I64(v as i64)) }
        12 => { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(GgufValue::F64(f64::from_le_bytes(b))) }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Unknown GGUF value type: {}", vtype))),
    }
}

fn read_gguf_array_element<R: Read>(r: &mut R, elem_type: u32) -> io::Result<GgufValue> {
    match elem_type {
        0 => { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(GgufValue::U8(b[0])) }
        1 => { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(GgufValue::I8(b[0] as i8)) }
        2 => { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(GgufValue::U16(u16::from_le_bytes(b))) }
        3 => { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(GgufValue::I16(i16::from_le_bytes(b))) }
        4 => Ok(GgufValue::U32(read_u32(r)?)),
        5 => Ok(GgufValue::I32(read_i32(r)?)),
        6 => Ok(GgufValue::F32(read_f32(r)?)),
        7 => { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(GgufValue::Bool(b[0] != 0)) }
        8 => Ok(GgufValue::Str(read_gguf_string(r)?)),
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => { let v = read_u64(r)?; Ok(GgufValue::I64(v as i64)) }
        12 => { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(GgufValue::F64(f64::from_le_bytes(b))) }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Unknown array element type: {}", elem_type))),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_k_roundtrip() {
        // Create a simple Q4_K block with known values
        let mut block = [0u8; 144];

        // d = 1.0 (as FP16)
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        block[0] = d_bits[0];
        block[1] = d_bits[1];

        // dmin = 0.0 (as FP16)
        block[2] = 0;
        block[3] = 0;

        // scales: all 1 (first 4 bytes of scales[12])
        block[4] = 1; // scale[0] = 1, lower 6 bits
        block[5] = 1;
        block[6] = 1;
        block[7] = 1;
        // mins (bytes 8-11): all 0
        block[8] = 0;
        block[9] = 0;
        block[10] = 0;
        block[11] = 0;
        // high bits (bytes 12-15): all 0
        block[12] = 0;
        block[13] = 0;
        block[14] = 0;
        block[15] = 0;

        // qs: first byte = 0x53 → low nibble=3, high nibble=5
        block[16] = 0x53;

        let result = dequantize_q4_k(&block, 256);
        assert_eq!(result.len(), 256);

        // First value: d=1.0, scale=1, q=3, min=0 → 1.0*1*3 - 0 = 3.0
        assert!((result[0] - 3.0).abs() < 0.01, "got {}", result[0]);
        // Value at index 32 (high nibble of first group's second half):
        // d=1.0, scale=1, q=5, min=0 → 5.0
        assert!((result[32] - 5.0).abs() < 0.01, "got {}", result[32]);
    }
}
