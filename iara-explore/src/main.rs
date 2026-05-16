//! tateador — explora os pesos e vocabulário do Qwen
//! Mede, calcula, sente o que tem dentro dos arquivos.

use std::{collections::HashMap, fs, io::{Read, Seek, SeekFrom}};

const MODEL: &str = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b";

fn main() {
    let vocab = load_vocab();
    parte1_byte_graph(&vocab);
    parte2_safetensors();
    parte3_embeddings(&vocab);
}

// ═══════════════════════════════════════════════════════════════
// PARTE 1 — Byte Graph: o que o vocabulário sabe sobre bytes
// ═══════════════════════════════════════════════════════════════

fn parte1_byte_graph(vocab: &HashMap<String, u32>) {
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║   PARTE 1 — BYTE GRAPH DO VOCABULÁRIO QWEN      ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    // Para cada byte: quais tokens começam com ele
    let mut byte_tokens: Vec<u32> = vec![0; 256]; // contagem
    // Bigrama: bigram[a][b] = quantos tokens contêm a sequência byte_a→byte_b
    let mut bigram = vec![[0u32; 256]; 256];

    for token_str in vocab.keys() {
        let bytes = token_str.as_bytes();
        if bytes.is_empty() { continue; }
        byte_tokens[bytes[0] as usize] += 1;
        for w in bytes.windows(2) {
            bigram[w[0] as usize][w[1] as usize] += 1;
        }
    }

    // Top bytes por riqueza semântica (mais tokens começam com eles)
    println!("▸ Bytes mais ricos (mais tokens começam com eles):");
    let mut counts: Vec<(u8, u32)> = (0u8..=255).map(|b| (b, byte_tokens[b as usize])).collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (byte, n) in counts.iter().take(12).filter(|(_, n)| *n > 0) {
        let label = byte_label(*byte);
        println!("  {label:>8}  →  {n:4} tokens");
    }

    // Sucessores mais prováveis para bytes chave do PT-BR
    println!("\n▸ Top sucessores (bigrama no vocabulário):");
    for &b in &[b'o', b'O', b'n', b'a', b'q', 0xC3u8] {
        let row = &bigram[b as usize];
        let total: u32 = row.iter().sum();
        if total == 0 { continue; }
        let mut succs: Vec<(u8, f32)> = (0u8..=255)
            .map(|b2| (b2, row[b2 as usize] as f32 / total as f32))
            .filter(|(_, p)| *p > 0.005)
            .collect();
        succs.sort_by(|a, c| c.1.partial_cmp(&a.1).unwrap());
        let top: Vec<String> = succs.iter().take(6)
            .map(|(b2, p)| format!("{}({:.0}%)", byte_label(*b2), p * 100.0))
            .collect();
        println!("  {:>8}  →  {}", byte_label(b), top.join("  "));
    }

    // Entropia de cada byte (H baixa = muito previsível = contexto forte)
    println!("\n▸ Entropia por byte — H baixa = próximo byte é previsível:");
    println!("  (bytes com contexto semântico forte no vocabulário Qwen)");
    let mut entropies: Vec<(u8, f32)> = (0u8..=255).filter_map(|b| {
        let row = &bigram[b as usize];
        let total: u32 = row.iter().sum();
        if total < 20 { return None; }
        let h: f32 = row.iter()
            .filter(|&&c| c > 0)
            .map(|&c| { let p = c as f32 / total as f32; -p * p.log2() })
            .sum();
        Some((b, h))
    }).collect();
    entropies.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (byte, h) in entropies.iter().take(10) {
        println!("  {:>8}  H = {:.3} bits  (de max {:.1})", byte_label(*byte), h, 8.0_f32);
    }

    // Tokens PT-BR que existem como unidade ou são divididos pelo BPE
    println!("\n▸ Tokens PT-BR — unidade semântica ou sub-tokens?");
    let check = [
        "oi", "OI", "Oi", "olá", "Olá", "ola", "Ola",
        "não", "Não", "sim", "Sim", "tudo", "bem",
        "você", "obrigado", "tchau", "o", "i", "a", "e",
    ];
    for t in &check {
        match vocab.get(*t) {
            Some(id) => println!("  {t:>12}  →  token {id:6}  ✓ unidade"),
            None     => println!("  {t:>12}  →  (BPE divide em sub-tokens)"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PARTE 2 — Safetensors: anatomia dos pesos
// ═══════════════════════════════════════════════════════════════

fn parte2_safetensors() {
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║   PARTE 2 — ANATOMIA DOS PESOS (SAFETENSORS)    ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    let path = format!("{MODEL}/model.safetensors");
    let mut f = fs::File::open(&path).expect("model.safetensors não encontrado");

    let hlen = read_hlen(&mut f);
    let mut hbuf = vec![0u8; hlen];
    f.read_exact(&mut hbuf).unwrap();
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&hbuf).unwrap();

    // Coleta todos os tensores com tamanho e shape
    let mut tensors: Vec<(String, String, Vec<u64>, u64)> = header.iter()
        .filter(|(k, _)| *k != "__metadata__")
        .filter_map(|(name, val)| {
            let dtype = val["dtype"].as_str()?.to_string();
            let shape: Vec<u64> = val["shape"].as_array()?
                .iter().filter_map(|v| v.as_u64()).collect();
            let offs = val["data_offsets"].as_array()?;
            let size = offs[1].as_u64()? - offs[0].as_u64()?;
            Some((name.clone(), dtype, shape, size))
        }).collect();

    let total: u64 = tensors.iter().map(|t| t.3).sum();
    tensors.sort_by(|a, b| b.3.cmp(&a.3));

    println!("▸ Maiores tensores (top 20):");
    println!("  {:<45}  {:<6}  {:<22}  {:>8}", "Nome", "Dtype", "Shape", "Tamanho");
    println!("  {}", "─".repeat(90));
    for (name, dtype, shape, size) in tensors.iter().take(20) {
        let shape_s = shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("×");
        println!("  {name:<45}  {dtype:<6}  [{shape_s:<20}]  {:>6.1} MB",
            *size as f64 / 1e6);
    }
    println!("\n  Total: {:.1} MB em {} tensores", total as f64 / 1e6, tensors.len());

    // Camadas únicas (extrai prefixo layers.N)
    let mut layer_nums: std::collections::BTreeSet<u32> = Default::default();
    for (name, ..) in &tensors {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(n) = rest.split('.').next().and_then(|s| s.parse().ok()) {
                layer_nums.insert(n);
            }
        }
    }
    println!("\n▸ Camadas transformer: {} (layers 0..{})",
        layer_nums.len(),
        layer_nums.iter().last().copied().unwrap_or(0));
}

// ═══════════════════════════════════════════════════════════════
// PARTE 3 — Embeddings: medindo significado
// ═══════════════════════════════════════════════════════════════

fn parte3_embeddings(vocab: &HashMap<String, u32>) {
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║   PARTE 3 — GEOMETRIA DOS EMBEDDINGS            ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    let path = format!("{MODEL}/model.safetensors");
    let mut f = fs::File::open(&path).unwrap();

    let hlen = read_hlen(&mut f);
    let mut hbuf = vec![0u8; hlen];
    f.read_exact(&mut hbuf).unwrap();
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&hbuf).unwrap();

    let data_start = 8 + hlen;
    let embed_info  = &header["model.embed_tokens.weight"];
    let embed_off   = embed_info["data_offsets"][0].as_u64().unwrap() as usize;
    let hidden: usize = embed_info["shape"][1].as_u64().unwrap() as usize;

    // Lê embedding de um token específico
    let read_embed = |f: &mut fs::File, id: u32| -> Vec<f32> {
        let off = data_start + embed_off + id as usize * hidden * 2;
        f.seek(SeekFrom::Start(off as u64)).unwrap();
        let mut buf = vec![0u8; hidden * 2];
        f.read_exact(&mut buf).unwrap();
        buf.chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };

    // Métricas
    let cosine = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32  = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32  = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb + 1e-8)
    };
    let norm = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
    let add  = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x,y)| x+y).collect() };
    let sub  = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x,y)| x-y).collect() };

    println!("▸ Questão central: token composto ≠ soma das partes?");
    println!("  Se cosine(E[AB], E[A]+E[B]) < 1.0 → 'AB' carrega");
    println!("  significado EXTRA além de A e B separados.\n");

    // Pares para analisar
    let pares: &[(&str, &str, &str)] = &[
        ("oi",  "o",  "i"),
        ("ola", "o",  "a"),
        ("ou",  "o",  "u"),
        ("os",  "o",  "s"),
        ("na",  "n",  "a"),
        ("no",  "n",  "o"),
        ("de",  "d",  "e"),
        ("se",  "s",  "e"),
    ];

    for (ab, ta, tb) in pares {
        let Some(&id_ab) = vocab.get(*ab) else {
            println!("  {ab:>6}: não é token único");
            continue;
        };
        let (Some(&id_a), Some(&id_b)) = (vocab.get(*ta), vocab.get(*tb)) else { continue; };

        let e_ab  = read_embed(&mut f, id_ab);
        let e_a   = read_embed(&mut f, id_a);
        let e_b   = read_embed(&mut f, id_b);
        let e_sum = add(&e_a, &e_b);
        let residual = sub(&e_ab, &e_sum);

        let sim = cosine(&e_ab, &e_sum);
        let sem_extra = norm(&residual);

        // Barra visual de significado extra
        let bar_len = ((1.0 - sim.clamp(0.0, 1.0)) * 40.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(40 - bar_len);

        println!("  \"{ab}\" (token {id_ab:6})");
        println!("    cosine(E[{ab}], E[{ta}]+E[{tb}]) = {sim:+.4}");
        println!("    significado extra   = {sem_extra:7.2}  |{bar}|");
        println!();
    }

    // Cluster: tokens semânticamente próximos de "oi"
    if let Some(&id_oi) = vocab.get("oi") {
        println!("▸ Tokens mais próximos de \"oi\" no espaço de embedding:");
        println!("  (amostragem de 5000 tokens por velocidade)\n");

        let e_oi = read_embed(&mut f, id_oi);
        let mut sims: Vec<(f32, String, u32)> = Vec::new();

        // Amostra tokens — pega um a cada ~30 do vocab total
        let step = vocab.len() / 5000 + 1;
        let mut sample: Vec<(&String, u32)> = vocab.iter()
            .map(|(k, &v)| (k, v))
            .enumerate()
            .filter(|(i, _)| i % step == 0)
            .map(|(_, x)| x)
            .collect();
        sample.sort_by_key(|(_, id)| *id);

        for (tok_str, tok_id) in &sample {
            if *tok_id == id_oi { continue; }
            let e = read_embed(&mut f, *tok_id);
            let s = cosine(&e_oi, &e);
            if s > 0.4 {
                sims.push((s, tok_str.to_string(), *tok_id));
            }
        }
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (sim, tok, id) in sims.iter().take(15) {
            println!("    sim={sim:.4}  id={id:6}  {:?}", tok);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

fn load_vocab() -> HashMap<String, u32> {
    let raw = fs::read_to_string(format!("{MODEL}/vocab.json"))
        .expect("vocab.json não encontrado");
    serde_json::from_str(&raw).expect("vocab.json inválido")
}

fn read_hlen(f: &mut fs::File) -> usize {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).unwrap();
    u64::from_le_bytes(buf) as usize
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn byte_label(b: u8) -> String {
    if b.is_ascii_graphic() {
        format!("'{}'", b as char)
    } else if b == b' ' {
        "' '".to_string()
    } else {
        format!("0x{b:02X}")
    }
}
