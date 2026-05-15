//! Inspect a `.dna` file: list all tensors, their shape, dtype, and chunk kind.
//!
//! Usage:
//!   dna_inspect --input model.dna [--filter substring]
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use clap::Parser;

const DNA_MAGIC: &[u8; 4] = b"DNA4";
const CHUNK_TSVD: [u8; 4] = *b"TSVD";
const CHUNK_TRAW: [u8; 4] = *b"TRAW";
const CHUNK_TBLK: [u8; 4] = *b"TBLK";

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    input: PathBuf,
    #[arg(long, default_value = "")]
    filter: String,
}

fn read_u32(f: &mut File) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(f: &mut File) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_magic(f: &mut File) -> std::io::Result<[u8; 4]> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(b)
}
fn read_u16(f: &mut File) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    f.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_name(f: &mut File) -> std::io::Result<String> {
    let nlen = read_u16(f)? as usize;
    let mut buf = vec![0u8; nlen];
    f.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let mut f = File::open(&cli.input)?;

    let magic = read_magic(&mut f)?;
    assert_eq!(&magic, DNA_MAGIC, "not a DNA file");
    let version = read_u32(&mut f)?;
    let tensor_count = read_u32(&mut f)?;
    println!("DNA v{} — {} tensors", version, tensor_count);
    println!("{:<3} {:<7} {:<6} {:<60} {:<25} {:<4}",
        "#", "kind", "dtype", "name", "shape", "rank");
    println!("{}", "─".repeat(110));

    let mut shown = 0;
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for i in 0..tensor_count {
        let chunk_magic = read_magic(&mut f)?;
        let name = read_name(&mut f)?;
        let payload_len = read_u64(&mut f)?;
        let payload_start = f.stream_position()?;

        let (kind, dtype_str, shape, rank) = match &chunk_magic {
            m if m == &CHUNK_TSVD => {
                let rows = read_u32(&mut f)?;
                let cols = read_u32(&mut f)?;
                let rank = read_u32(&mut f)?;
                let dt = read_u32(&mut f)?;
                let dtype = match dt { 0 => "BF16", 4 => "Q4_0", _ => "?" };
                ("TSVD", dtype.to_string(), format!("[{}, {}]", rows, cols), Some(rank))
            }
            m if m == &CHUNK_TRAW => {
                let dim_count = read_u32(&mut f)? as usize;
                let mut shape: Vec<u32> = Vec::with_capacity(dim_count);
                for _ in 0..dim_count {
                    shape.push(read_u32(&mut f)?);
                }
                let dt = read_u32(&mut f)?;
                let dtype = match dt { 0 => "BF16", 3 => "F32", _ => "?" };
                let shape_str = format!("{:?}", shape);
                ("TRAW", dtype.to_string(), shape_str, None)
            }
            m if m == &CHUNK_TBLK => {
                let dim_count = read_u32(&mut f)? as usize;
                let mut shape: Vec<u32> = Vec::with_capacity(dim_count);
                for _ in 0..dim_count {
                    shape.push(read_u32(&mut f)?);
                }
                let dt = read_u32(&mut f)?;
                let dtype = match dt { 12 => "Q4_K", 14 => "Q6_K", 8 => "Q8_0", _ => "?" };
                let shape_str = format!("{:?}", shape);
                ("TBLK", dtype.to_string(), shape_str, None)
            }
            other => {
                panic!("unknown chunk: {:?}", other);
            }
        };

        *counts.entry(format!("{}/{}", kind, dtype_str)).or_insert(0) += 1;

        if cli.filter.is_empty() || name.contains(&cli.filter) {
            let rank_s = rank.map(|r| r.to_string()).unwrap_or_else(|| "-".to_string());
            println!("{:<3} {:<7} {:<6} {:<60} {:<25} {:<4}",
                i, kind, dtype_str, name, shape, rank_s);
            shown += 1;
        }

        // Skip to next tensor
        f.seek(SeekFrom::Start(payload_start + payload_len))?;
    }
    println!("\nShown {} tensors", shown);
    println!("\nCounts by kind/dtype:");
    for (k, v) in counts.iter() {
        println!("  {} = {}", k, v);
    }
    Ok(())
}
