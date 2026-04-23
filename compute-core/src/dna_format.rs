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

pub const VERSION: u32 = 1;

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
}
