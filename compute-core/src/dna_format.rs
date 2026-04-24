//! DNA V4 binary format primitives.
//!
//! Pure host-side serialization: no GPU, no backend dependency. Both the
//! inference engine (reader) and the training/compression crate (writer)
//! consume this module to stay in sync on the on-disk layout.
//!
//! File layout:
//!   "DNA4" | version_u32 | tensor_count_u32
//!   [chunk*]
//!
//! Chunk:
//!   magic[4] | name_len_u16 | name_utf8 | payload_len_u64 | payload
//!
//! TSVD payload:
//!   rows_u32 | cols_u32 | rank_u32 | dtype_u32 | A[rows*rank] dtype | B[rank*cols] dtype
//!
//! TRAW payload:
//!   dim_count_u32 | dims[u32; dim_count] | dtype_u32 | data

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const DNA_MAGIC: &[u8; 4] = b"DNA4";
pub const CHUNK_TSVD: &[u8; 4] = b"TSVD";
pub const CHUNK_TRAW: &[u8; 4] = b"TRAW";

pub const DTYPE_BF16: u32 = 0;
pub const DTYPE_F16: u32 = 1;
pub const DTYPE_I8: u32 = 2;
pub const DTYPE_F32: u32 = 3;
pub const DTYPE_Q4_0: u32 = 4;

pub const VERSION: u32 = 1;

pub const Q4_0_BLOCK: usize = 32;
pub const Q4_0_BYTES_PER_BLOCK: usize = 18; // fp16 scale (2) + 16 packed nibbles

// ── Writer ──────────────────────────────────────────────────────────────────

pub struct DnaWriter {
    f: BufWriter<File>,
    tensor_count: u32,
    count_pos: u64,
}

impl DnaWriter {
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut f = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);
        f.write_all(DNA_MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        let count_pos = f.stream_position()?;
        f.write_all(&0u32.to_le_bytes())?; // placeholder, rewritten in finalize()
        Ok(Self { f, tensor_count: 0, count_pos })
    }

    pub fn write_tsvd(
        &mut self,
        name: &str,
        rows: u32,
        cols: u32,
        rank: u32,
        a_bf16: &[u16],
        b_bf16: &[u16],
    ) -> io::Result<()> {
        debug_assert_eq!(a_bf16.len(), (rows * rank) as usize);
        debug_assert_eq!(b_bf16.len(), (rank * cols) as usize);
        let payload_len = 16u64 + (a_bf16.len() as u64 + b_bf16.len() as u64) * 2;
        self.write_chunk_header(CHUNK_TSVD, name, payload_len)?;
        self.f.write_all(&rows.to_le_bytes())?;
        self.f.write_all(&cols.to_le_bytes())?;
        self.f.write_all(&rank.to_le_bytes())?;
        self.f.write_all(&DTYPE_BF16.to_le_bytes())?;
        write_u16_slice(&mut self.f, a_bf16)?;
        write_u16_slice(&mut self.f, b_bf16)?;
        self.tensor_count += 1;
        Ok(())
    }

    /// Q4_0-quantized TSVD. Block size 32, fp16 scale + 16 packed nibbles per block.
    /// Both A (rows*rank) and B (rank*cols) must have element counts that are multiples of 32.
    pub fn write_tsvd_q4(
        &mut self,
        name: &str,
        rows: u32,
        cols: u32,
        rank: u32,
        a_q4: &[u8],
        b_q4: &[u8],
    ) -> io::Result<()> {
        let a_elems = (rows * rank) as usize;
        let b_elems = (rank * cols) as usize;
        debug_assert_eq!(a_elems % Q4_0_BLOCK, 0);
        debug_assert_eq!(b_elems % Q4_0_BLOCK, 0);
        debug_assert_eq!(a_q4.len(), a_elems / Q4_0_BLOCK * Q4_0_BYTES_PER_BLOCK);
        debug_assert_eq!(b_q4.len(), b_elems / Q4_0_BLOCK * Q4_0_BYTES_PER_BLOCK);
        let payload_len = 16u64 + a_q4.len() as u64 + b_q4.len() as u64;
        self.write_chunk_header(CHUNK_TSVD, name, payload_len)?;
        self.f.write_all(&rows.to_le_bytes())?;
        self.f.write_all(&cols.to_le_bytes())?;
        self.f.write_all(&rank.to_le_bytes())?;
        self.f.write_all(&DTYPE_Q4_0.to_le_bytes())?;
        self.f.write_all(a_q4)?;
        self.f.write_all(b_q4)?;
        self.tensor_count += 1;
        Ok(())
    }

    pub fn write_traw_bf16(&mut self, name: &str, shape: &[u32], data: &[u16]) -> io::Result<()> {
        let numel: u32 = shape.iter().product();
        debug_assert_eq!(numel as usize, data.len());
        let hdr = 4 + shape.len() as u64 * 4 + 4;
        let payload_len = hdr + data.len() as u64 * 2;
        self.write_chunk_header(CHUNK_TRAW, name, payload_len)?;
        self.f.write_all(&(shape.len() as u32).to_le_bytes())?;
        for d in shape { self.f.write_all(&d.to_le_bytes())?; }
        self.f.write_all(&DTYPE_BF16.to_le_bytes())?;
        write_u16_slice(&mut self.f, data)?;
        self.tensor_count += 1;
        Ok(())
    }

    pub fn write_traw_f32(&mut self, name: &str, shape: &[u32], data: &[f32]) -> io::Result<()> {
        let numel: u32 = shape.iter().product();
        debug_assert_eq!(numel as usize, data.len());
        let hdr = 4 + shape.len() as u64 * 4 + 4;
        let payload_len = hdr + data.len() as u64 * 4;
        self.write_chunk_header(CHUNK_TRAW, name, payload_len)?;
        self.f.write_all(&(shape.len() as u32).to_le_bytes())?;
        for d in shape { self.f.write_all(&d.to_le_bytes())?; }
        self.f.write_all(&DTYPE_F32.to_le_bytes())?;
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.f.write_all(bytes)?;
        self.tensor_count += 1;
        Ok(())
    }

    pub fn finalize(mut self) -> io::Result<()> {
        self.f.seek(SeekFrom::Start(self.count_pos))?;
        self.f.write_all(&self.tensor_count.to_le_bytes())?;
        self.f.flush()?;
        // Drop the BufWriter, which flushes and closes the File on drop.
        Ok(())
    }

    fn write_chunk_header(&mut self, magic: &[u8; 4], name: &str, payload_len: u64) -> io::Result<()> {
        self.f.write_all(magic)?;
        self.f.write_all(&(name.len() as u16).to_le_bytes())?;
        self.f.write_all(name.as_bytes())?;
        self.f.write_all(&payload_len.to_le_bytes())?;
        Ok(())
    }
}

fn write_u16_slice(f: &mut BufWriter<File>, data: &[u16]) -> io::Result<()> {
    // LE target: byte-cast is the on-disk layout. One write_all per slice
    // instead of one per element (∼200× speedup for large tensors).
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
    };
    f.write_all(bytes)
}

// ── Low-level readers (used by loaders) ────────────────────────────────────

pub fn read_magic(f: &mut File) -> io::Result<[u8; 4]> {
    let mut b = [0u8; 4]; f.read_exact(&mut b)?; Ok(b)
}
pub fn read_u16(f: &mut File) -> io::Result<u16> {
    let mut b = [0u8; 2]; f.read_exact(&mut b)?; Ok(u16::from_le_bytes(b))
}
pub fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut b = [0u8; 4]; f.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
pub fn read_u64(f: &mut File) -> io::Result<u64> {
    let mut b = [0u8; 8]; f.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
pub fn read_name(f: &mut File) -> io::Result<String> {
    let len = read_u16(f)? as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
pub fn read_u16_vec(f: &mut File, n: usize) -> io::Result<Vec<u16>> {
    let mut bytes = vec![0u8; n * 2];
    f.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}
pub fn read_f32_vec(f: &mut File, n: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    f.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

// ── Q4_0 quantize/dequantize (host) ────────────────────────────────────────
//
// Layout per 32-elem block:
//   2 B: fp16 scale d
//  16 B: 32 nibbles, packed as byte[i] = (q_{i+16} << 4) | q_i  for i in 0..16
//
// Quant: d = max / -8  where max = argmax(|x|, signed);  q = round(x/d) + 8 clamped [0,15]
// Dequant: x = (q - 8) * d

/// Quantize f32 slice to Q4_0. `data.len()` must be a multiple of 32.
pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % Q4_0_BLOCK, 0, "q4_0: len must be multiple of 32");
    let nblocks = data.len() / Q4_0_BLOCK;
    let mut out = Vec::with_capacity(nblocks * Q4_0_BYTES_PER_BLOCK);
    for b in 0..nblocks {
        let blk = &data[b * Q4_0_BLOCK..(b + 1) * Q4_0_BLOCK];
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in blk {
            let a = v.abs();
            if a > amax { amax = a; max = v; }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let scale_f16 = half::f16::from_f32(d).to_bits();
        out.extend_from_slice(&scale_f16.to_le_bytes());
        for i in 0..16 {
            let x0 = blk[i] * id;
            let x1 = blk[i + 16] * id;
            let q0 = ((x0 + 8.5).floor() as i32).clamp(0, 15) as u8;
            let q1 = ((x1 + 8.5).floor() as i32).clamp(0, 15) as u8;
            out.push((q1 << 4) | q0);
        }
    }
    out
}

/// Dequantize Q4_0 bytes back to f32. `n` = element count; `bytes.len()` must be
/// `n/32 * 18`.
pub fn dequantize_q4_0(bytes: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(n % Q4_0_BLOCK, 0);
    let nblocks = n / Q4_0_BLOCK;
    assert_eq!(bytes.len(), nblocks * Q4_0_BYTES_PER_BLOCK);
    let mut out = vec![0f32; n];
    for b in 0..nblocks {
        let off = b * Q4_0_BYTES_PER_BLOCK;
        let scale_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        let d = half::f16::from_bits(scale_bits).to_f32();
        for i in 0..16 {
            let byte = bytes[off + 2 + i];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = (byte >> 4) as i32 - 8;
            out[b * Q4_0_BLOCK + i] = q0 as f32 * d;
            out[b * Q4_0_BLOCK + i + 16] = q1 as f32 * d;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};

    #[test]
    fn writer_header_layout() {
        let tmp = std::env::temp_dir().join("dna_format_test.dna");
        {
            let mut w = DnaWriter::create(&tmp).unwrap();
            w.write_tsvd("w1", 2, 2, 1, &[1, 2], &[3, 4]).unwrap();
            w.finalize().unwrap();
        }
        let mut f = File::open(&tmp).unwrap();
        let magic = read_magic(&mut f).unwrap();
        assert_eq!(&magic, DNA_MAGIC);
        assert_eq!(read_u32(&mut f).unwrap(), VERSION);
        assert_eq!(read_u32(&mut f).unwrap(), 1); // count
        let chunk = read_magic(&mut f).unwrap();
        assert_eq!(&chunk, CHUNK_TSVD);
        let name = read_name(&mut f).unwrap();
        assert_eq!(name, "w1");
        let plen = read_u64(&mut f).unwrap();
        assert_eq!(plen, 16 + 2 * 2 + 1 * 2 * 2); // hdr + A + B
        f.seek(SeekFrom::Current(plen as i64)).unwrap();
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn q4_0_roundtrip_uniform() {
        // Smooth gradient: quant error should be small relative to range.
        let n = 256;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 16.0).collect();
        let q = quantize_q4_0(&data);
        assert_eq!(q.len(), (n / Q4_0_BLOCK) * Q4_0_BYTES_PER_BLOCK);
        let back = dequantize_q4_0(&q, n);
        let max_abs: f32 = data.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let mut max_err = 0f32;
        for (a, b) in data.iter().zip(back.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        // Q4 worst-case quant error ≈ max_abs / 7 per block. Loose bound.
        assert!(max_err < max_abs / 5.0, "q4 err {} too large (range {})", max_err, max_abs);
    }

    #[test]
    fn q4_0_roundtrip_zero_block() {
        let data = vec![0f32; 32];
        let q = quantize_q4_0(&data);
        let back = dequantize_q4_0(&q, 32);
        assert!(back.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn write_tsvd_q4_layout() {
        let tmp = std::env::temp_dir().join("dna_format_q4.dna");
        let rows = 2u32; let cols = 32u32; let rank = 32u32;
        let a = vec![0f32; (rows * rank) as usize];
        let b = vec![0f32; (rank * cols) as usize];
        let aq = quantize_q4_0(&a);
        let bq = quantize_q4_0(&b);
        {
            let mut w = DnaWriter::create(&tmp).unwrap();
            w.write_tsvd_q4("q", rows, cols, rank, &aq, &bq).unwrap();
            w.finalize().unwrap();
        }
        let mut f = File::open(&tmp).unwrap();
        let _magic = read_magic(&mut f).unwrap();
        let _version = read_u32(&mut f).unwrap();
        assert_eq!(read_u32(&mut f).unwrap(), 1);
        let chunk = read_magic(&mut f).unwrap();
        assert_eq!(&chunk, CHUNK_TSVD);
        let name = read_name(&mut f).unwrap();
        assert_eq!(name, "q");
        let plen = read_u64(&mut f).unwrap();
        assert_eq!(plen, 16 + aq.len() as u64 + bq.len() as u64);
        assert_eq!(read_u32(&mut f).unwrap(), rows);
        assert_eq!(read_u32(&mut f).unwrap(), cols);
        assert_eq!(read_u32(&mut f).unwrap(), rank);
        assert_eq!(read_u32(&mut f).unwrap(), DTYPE_Q4_0);
        std::fs::remove_file(&tmp).ok();
    }
}
