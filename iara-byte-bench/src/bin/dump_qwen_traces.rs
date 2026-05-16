//! # dump_qwen_traces
//!
//! Coleta `(prompt, response)` do Qwen 2.5 1.5B via iara-engine HTTP /api/generate
//! e salva num formato binário simples consumido por `byte-language-lab/exp16`.
//!
//! ## Formato do arquivo (little-endian)
//!
//! ```text
//! magic        : [u8; 4] = b"QTRC"
//! version      : u32     = 1
//! n_records    : u32
//! repeat n_records:
//!   prompt_len : u32
//!   prompt     : [u8; prompt_len]    (UTF-8)
//!   response_len : u32
//!   response   : [u8; response_len]  (UTF-8, output greedy do Qwen)
//! ```
//!
//! ## Uso
//!
//! 1. Em outro terminal: `./scripts/run_iara.sh release`
//! 2. Carregar modelo: `curl -X POST http://localhost:3000/api/load-model -d '{"name":"qwen2.5-1.5b"}'`
//! 3. Rodar: `cargo run --release -p iara-byte-bench --bin dump_qwen_traces -- \
//!      --prompts ../byte-language-lab/data/exp16_prompts.txt \
//!      --out ../byte-language-lab/data/qwen_traces.bin`

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(version, about = "Dump Qwen 1.5B (prompt, response) traces", long_about = None)]
struct Args {
    /// URL base do iara-engine
    #[arg(long, default_value = "http://localhost:3000")]
    server: String,

    /// Nome do modelo (deve estar carregado no engine)
    #[arg(long, default_value = "qwen2.5-1.5b")]
    model: String,

    /// Arquivo com 1 prompt por linha
    #[arg(long)]
    prompts: PathBuf,

    /// Arquivo binário de saída
    #[arg(long)]
    out: PathBuf,

    /// Max tokens por resposta
    #[arg(long, default_value = "32")]
    max_tokens: usize,

    /// Greedy (T=0). Top_k=1 garante determinismo.
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Timeout por request (segundos)
    #[arg(long, default_value = "60")]
    timeout_secs: u64,
}

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct GenerateResponse {
    response: String,
    #[serde(default)]
    tokens_per_second: f64,
}

async fn ask(
    client: &reqwest::Client,
    server: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> anyhow::Result<String> {
    let req = GenerateRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        max_tokens,
        temperature,
        top_k: 1,
        top_p: 1.0,
        stream: false,
    };

    let resp = client
        .post(format!("{server}/api/generate"))
        .json(&req)
        .send()
        .await?
        .error_for_status()?;

    let body: GenerateResponse = resp.json().await?;
    Ok(body.response)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let raw = fs::read_to_string(&args.prompts)
        .map_err(|e| anyhow::anyhow!("read prompts {}: {e}", args.prompts.display()))?;
    let prompts: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();

    println!("[dump] {} prompts carregados de {}", prompts.len(), args.prompts.display());
    println!("[dump] server={} model={}", args.server, args.model);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(args.timeout_secs))
        .build()?;

    let mut records: Vec<(String, String)> = Vec::with_capacity(prompts.len());
    let mut errors = 0usize;
    let total = prompts.len();
    let t0 = std::time::Instant::now();

    for (i, p) in prompts.iter().enumerate() {
        match ask(&client, &args.server, &args.model, p, args.max_tokens, args.temperature).await {
            Ok(resp) => {
                let preview: String = resp.chars().take(50).collect();
                println!("[{:>3}/{}] {:.40} → {:?}", i + 1, total, p, preview);
                records.push((p.clone(), resp));
            }
            Err(e) => {
                eprintln!("[{:>3}/{}] ERR {p:?}: {e}", i + 1, total);
                errors += 1;
            }
        }
    }

    println!(
        "[dump] coletados {}/{} ({} erros) em {:.1}s",
        records.len(),
        total,
        errors,
        t0.elapsed().as_secs_f64()
    );

    // Serializa formato binário
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(&args.out)?;
    f.write_all(b"QTRC")?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(records.len() as u32).to_le_bytes())?;
    for (p, r) in &records {
        let pb = p.as_bytes();
        let rb = r.as_bytes();
        f.write_all(&(pb.len() as u32).to_le_bytes())?;
        f.write_all(pb)?;
        f.write_all(&(rb.len() as u32).to_le_bytes())?;
        f.write_all(rb)?;
    }
    f.flush()?;
    println!("[dump] gravado: {}", args.out.display());
    Ok(())
}
