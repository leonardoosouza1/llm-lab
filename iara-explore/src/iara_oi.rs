//! iara_oi.rs — Responde "oi" usando os pesos do Qwen em Rust puro
//!
//! Estratégia: 0-layer embedding inference
//!   1. E_oi = embedding do token "oi" (6728)
//!   2. E_ola = embedding do token "ola" (7924) — alvo da resposta
//!   3. R = direção de resposta = E_ola
//!   4. Byte graph dos bigramas do vocab
//!   5. byte_avg_embed[b] = média dos embeddings de tokens que começam com b
//!   6. Geração byte a byte: score = graph_prob × cosine(R, byte_avg_embed[b])
//!   7. Resultado em bytes + UTF-8

use std::{collections::HashMap, fs, io::{BufRead, BufReader, Read, Seek, SeekFrom}, time::Instant};

const MODEL:   &str  = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b";
const HIDDEN:  usize = 1536;
const VOCAB_N: usize = 151936;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA — Respondendo 'oi' com pesos Qwen em Rust puro       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();

    // ── Carrega vocab ──────────────────────────────────────────────
    print!("▸ Carregando vocab.json... ");
    let vocab: HashMap<String, u32> = {
        let raw = fs::read_to_string(format!("{MODEL}/vocab.json")).unwrap();
        serde_json::from_str(&raw).unwrap()
    };
    println!("{} tokens  ({:.1}s)", vocab.len(), t0.elapsed().as_secs_f32());

    // Mapa inverso id → string
    let id_to_str: HashMap<u32, String> = vocab.iter().map(|(k,&v)| (v, k.clone())).collect();

    // ── Abre safetensors ───────────────────────────────────────────
    let st = load_meta();

    // ── Embeddings dos tokens âncora ──────────────────────────────
    print!("▸ Lendo embeddings âncora... ");
    let mut f = fs::File::open(format!("{MODEL}/model.safetensors")).unwrap();

    let id_oi  = *vocab.get("oi").expect("token 'oi' não encontrado");
    let id_ola = *vocab.get("ola").expect("token 'ola' não encontrado");
    let id_Oi  = vocab.get("Oi").copied().unwrap_or(id_oi);
    let id_sim = vocab.get("sim").copied().unwrap_or(0);

    let e_oi  = read_embed(&mut f, &st, id_oi);
    let e_ola = read_embed(&mut f, &st, id_ola);
    let e_Oi  = read_embed(&mut f, &st, id_Oi);
    let e_sim = read_embed(&mut f, &st, id_sim);

    // Direção de resposta = média de embeddings de respostas conhecidas
    let resp_target: Vec<f32> = (0..HIDDEN)
        .map(|i| (e_ola[i] + e_Oi[i] + e_sim[i]) / 3.0)
        .collect();

    println!("ok  ({:.1}s)", t0.elapsed().as_secs_f32());

    // ── Top-N tokens por dot product com resp_target ───────────────
    println!("▸ Procurando melhores tokens de resposta (0-layer inference)...");
    let t1 = Instant::now();

    // Lê a tabela inteira de embeddings de uma vez (466MB BF16)
    let embed_offset = st.embed_off;
    let embed_bytes  = VOCAB_N * HIDDEN * 2;
    f.seek(SeekFrom::Start((st.data_start + embed_offset) as u64)).unwrap();
    let mut raw = vec![0u8; embed_bytes];
    f.read_exact(&mut raw).unwrap();
    println!("  tabela lida: {:.0}MB em {:.1}s", embed_bytes as f32 / 1e6, t1.elapsed().as_secs_f32());

    // Calcula dot product de cada token com resp_target
    let t2 = Instant::now();
    let mut scores: Vec<(f32, u32)> = Vec::with_capacity(VOCAB_N);
    // byte_sum e byte_count para byte_avg_embed
    let mut byte_sum:   Vec<Vec<f32>> = vec![vec![0.0f32; HIDDEN]; 256];
    let mut byte_count: Vec<u32>      = vec![0u32; 256];

    for token_id in 0..VOCAB_N as u32 {
        let base = token_id as usize * HIDDEN * 2;
        // Decodifica BF16 → f32 inline
        let emb: Vec<f32> = (0..HIDDEN).map(|i| {
            let bits = u16::from_le_bytes([raw[base + i*2], raw[base + i*2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        }).collect();

        // Dot product com resp_target (cosine aproximado sem normalizar — mais rápido)
        let dot: f32 = emb.iter().zip(&resp_target).map(|(a,b)| a*b).sum();
        scores.push((dot, token_id));

        // Acumula para byte_avg_embed
        if let Some(s) = id_to_str.get(&token_id) {
            let bytes = s.as_bytes();
            if !bytes.is_empty() {
                let b = bytes[0] as usize;
                for i in 0..HIDDEN { byte_sum[b][i] += emb[i]; }
                byte_count[b] += 1;
            }
        }
    }

    scores.sort_unstable_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
    println!("  scores calculados em {:.1}s", t2.elapsed().as_secs_f32());

    // ── Exibe top tokens ─────────────────────────────────────────
    println!("\n▸ Top tokens por similaridade com alvo de resposta:\n");
    println!("  {:>6}  {:>7}  {:>20}  bytes", "score", "id", "token");
    println!("  {}", "─".repeat(56));
    let mut mostrados = 0;
    for (score, tid) in scores.iter().take(2000) {
        let s = match id_to_str.get(tid) { Some(s) => s, None => continue };
        // Filtra: só tokens que parecem PT-BR ou resposta plausível
        let bytes = s.as_bytes();
        if bytes.is_empty() { continue; }
        let first = bytes[0];
        // Aceita: letra maiúscula, letra minúscula comum, pontuação básica
        if !first.is_ascii_alphabetic() && first != b'!' && first != b'O' { continue; }
        // Evita tokens muito longos (não são palavras simples)
        if s.len() > 12 { continue; }
        // Evita tokens de código (_, camelCase, etc)
        if s.contains('_') || s.contains('{') { continue; }

        let hex: String = bytes.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ");
        println!("  {:>6.1}  {:>7}  {:>20}  [{hex}]", score, tid, format!("{s:?}"));
        mostrados += 1;
        if mostrados >= 20 { break; }
    }

    // ── byte_avg_embed normalizados ────────────────────────────────
    let byte_avg: Vec<Vec<f32>> = (0..256).map(|b| {
        let n = byte_count[b].max(1) as f32;
        byte_sum[b].iter().map(|&v| v / n).collect()
    }).collect();

    // ── Byte graph do vocab ────────────────────────────────────────
    println!("\n▸ Construindo byte graph...");
    let mut bigram = vec![[0u32; 256]; 256];
    for (token_str, _) in &vocab {
        let bytes = token_str.as_bytes();
        for w in bytes.windows(2) {
            bigram[w[0] as usize][w[1] as usize] += 1;
        }
    }
    // Normaliza para probabilidades
    let byte_graph: Vec<Vec<f32>> = bigram.iter().map(|row| {
        let total = row.iter().sum::<u32>().max(1) as f32;
        row.iter().map(|&c| c as f32 / total).collect()
    }).collect();

    // ── Geração byte a byte ────────────────────────────────────────
    println!("\n▸ Gerando resposta byte a byte (graph × embedding)...\n");

    // Normaliza resp_target
    let r_norm: f32 = resp_target.iter().map(|x| x*x).sum::<f32>().sqrt();
    let r_unit: Vec<f32> = resp_target.iter().map(|x| x / r_norm).collect();

    // Similaridade semântica de cada byte com o alvo de resposta
    let byte_sem: Vec<f32> = (0u16..256).map(|b| {
        let avg = &byte_avg[b as usize];
        let n: f32 = avg.iter().map(|x| x*x).sum::<f32>().sqrt();
        if n < 1e-6 { return 0.0; }
        let dot: f32 = avg.iter().zip(&r_unit).map(|(a,b)| a*b).sum::<f32>();
        (dot / n).max(0.0)
    }).collect();

    // Mostra mapa de afinidade semântica por byte
    println!("  Afinidade semântica bytes → resposta (top 15):");
    let mut sem_ord: Vec<(f32, u8)> = (0u8..=255).map(|b| (byte_sem[b as usize], b)).collect();
    sem_ord.sort_by(|a,c| c.0.partial_cmp(&a.0).unwrap());
    for (sem, b) in sem_ord.iter().take(15) {
        let label = if b.is_ascii_graphic() { format!("'{}'", *b as char) } else { format!("0x{b:02X}") };
        println!("    {label:>6}  afinidade={sem:.4}  {}", bar(*sem * 3.0));
    }

    // Geração greedy com temperatura
    println!("\n  Gerando resposta a 'oi':\n");
    let temp = 0.7f32;

    for run in 0..5 {
        let seed = (run * 7919 + 31337) as u64;
        let response = gera_resposta(&byte_graph, &byte_sem, temp, seed, 40);
        let decoded  = String::from_utf8_lossy(&response);
        let hex_str: String = response.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ");
        println!("  Run {}: {:?}", run+1, decoded.trim());
        println!("         bytes: [{hex_str}]\n");
    }

    // ── Estratégia 2: Token-seeded (top tokens semânticos → bytes) ───
    println!("\n▸ Estratégia 2 — Token-seeded (top tokens semânticos → bytes):\n");

    // Coleta top tokens de resposta: só ASCII alfabético, até 10 chars
    let top_tok_list: Vec<(f32, Vec<u8>)> = {
        let mut v = Vec::new();
        for (score, tid) in scores.iter().take(10000) {
            if v.len() >= 300 { break; }
            let s = match id_to_str.get(tid) { Some(s) => s, None => continue };
            let bytes = s.as_bytes().to_vec();
            if bytes.is_empty() || bytes.len() > 10 { continue; }
            if !bytes[0].is_ascii_alphabetic() { continue; }
            if bytes.iter().any(|&b| !b.is_ascii_alphabetic()) { continue; }
            v.push((*score, bytes));
        }
        v
    };

    println!("  {} tokens coletados. Top 10:", top_tok_list.len());
    for (sc, bytes) in top_tok_list.iter().take(10) {
        let s = String::from_utf8_lossy(bytes);
        println!("    score={:.3}  {:?}", sc, s.as_ref());
    }

    // Distribuição de bytes por posição (score²-weighted)
    let maxpos = 8usize;
    let mut pos_dist: Vec<[f32; 256]> = vec![[0.0f32; 256]; maxpos];
    for (score, bytes) in &top_tok_list {
        let w = score * score;
        for (pos, &b) in bytes.iter().enumerate() {
            if pos < maxpos { pos_dist[pos][b as usize] += w; }
        }
    }
    for row in pos_dist.iter_mut() {
        let total: f32 = row.iter().sum();
        if total > 1e-9 { row.iter_mut().for_each(|v| *v /= total); }
    }

    println!("\n  Top bytes por posição:");
    for pos in 0..4 {
        let mut top: Vec<(f32, u8)> = (b'A'..=b'z')
            .map(|b| (pos_dist[pos][b as usize], b))
            .filter(|(p, _)| *p > 0.005)
            .collect();
        top.sort_by(|a, b_| b_.0.partial_cmp(&a.0).unwrap());
        let s: String = top.iter().take(6)
            .map(|(p, b)| format!("'{}' {:.0}%", *b as char, p * 100.0))
            .collect::<Vec<_>>().join("  ");
        println!("    pos[{}]: {}", pos, s);
    }

    // Geração condicional (trie-style): filtra tokens pelo prefixo gerado
    println!("\n  Gerando respostas a 'oi' (condicional / trie-style):\n");
    let mut rng_ts = 42u64;
    for run in 0..8 {
        let mut word: Vec<u8> = Vec::new();
        // Começa com todos os tokens candidatos
        let mut remaining: Vec<(f32, &[u8])> = top_tok_list.iter()
            .map(|(sc, b)| (*sc, b.as_slice()))
            .collect();

        for pos in 0..maxpos {
            // Distribuição condicional: só tokens que ainda sobrevivem
            let mut cdist = [0.0f32; 256];
            let mut total_w = 0.0f32;
            for (sc, bytes) in &remaining {
                if bytes.len() > pos {
                    let b = bytes[pos] as usize;
                    cdist[b] += sc * sc;
                    total_w += sc * sc;
                }
            }
            if total_w < 1e-9 { break; }
            for v in cdist.iter_mut() { *v /= total_w; }

            // Amostra
            rng_ts = rng_ts.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (rng_ts >> 33) as f32 / (u32::MAX as f32);
            let mut cum = 0.0f32;
            let mut chosen = 0u8;
            for b in b'A'..=b'z' {
                cum += cdist[b as usize];
                if r < cum { chosen = b; break; }
            }
            if chosen == 0 || !chosen.is_ascii_alphabetic() { break; }
            word.push(chosen);

            // Filtra remaining para tokens que têm esse byte nessa posição
            remaining.retain(|(_, bytes)| bytes.len() > pos && bytes[pos] == chosen);
            if remaining.is_empty() { break; }
        }
        let s = String::from_utf8_lossy(&word);
        let hex: String = word.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ");
        println!("  Run {}: {:?}  [{}]", run + 1, s.as_ref(), hex);
    }

    // Resposta greedy condicional
    let mut greedy: Vec<u8> = Vec::new();
    let mut rem: Vec<(f32, &[u8])> = top_tok_list.iter()
        .map(|(sc, b)| (*sc, b.as_slice()))
        .collect();
    for pos in 0..maxpos {
        let mut cdist = [0.0f32; 256];
        let mut tw = 0.0f32;
        for (sc, bytes) in &rem {
            if bytes.len() > pos {
                cdist[bytes[pos] as usize] += sc * sc;
                tw += sc * sc;
            }
        }
        if tw < 1e-9 { break; }
        let best = (b'A'..=b'z')
            .max_by(|&a, &b_| cdist[a as usize].partial_cmp(&cdist[b_ as usize]).unwrap())
            .unwrap();
        if cdist[best as usize] / tw < 0.05 || !best.is_ascii_alphabetic() { break; }
        greedy.push(best);
        rem.retain(|(_, bytes)| bytes.len() > pos && bytes[pos] == best);
        if rem.is_empty() { break; }
    }
    let greedy_str = String::from_utf8_lossy(&greedy);
    let hex: String = greedy.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ");
    println!("\n  ★ GREEDY condicional: {:?}  [{}]", greedy_str.as_ref(), hex);

    // ── Análise final: input vs output em bytes ────────────────────
    println!("\n▸ Análise entrada/saída em bytes:\n");
    let entrada = b"oi";
    let saida = b"Ol\xC3\xA1!";  // "Olá!" em UTF-8
    println!("  ENTRADA 'oi':  {:?}", entrada.iter().map(|b| format!("{b:08b}")).collect::<Vec<_>>().join(" "));
    println!("  SAÍDA   'Olá!': {:?}", saida.iter().map(|b| format!("{b:08b}")).collect::<Vec<_>>().join(" "));
    println!();
    println!("  Byte 'o' = 0x{:02X}: b7={} b6={} case={} grp={} nl={} (vogal)",
        b'o', 0, 1, 1, 0, b'o' & 0x0F);
    println!("  Byte 'O' = 0x{:02X}: b7={} b6={} case={} grp={} nl={} (vogal, upper)",
        b'O', 0, 1, 0, 0, b'O' & 0x0F);
    println!("  Byte 'l' = 0x{:02X}: b7={} b6={} case={} grp={} nl={} (cons_com)",
        b'l', 0, 1, 1, 0, b'l' & 0x0F);
    println!("  Byte 0xC3: prefixo UTF-8 de á,ã,â,é,ê,í,ó,ô,ú,ç → ESTRUTURAL");
    println!("  Byte 0xA1: sufixo UTF-8 de 'á'");

    println!("\n  Total: {:.1}s", t0.elapsed().as_secs_f32());
}

// ── Geração greedy+temperatura ──────────────────────────────────────
fn gera_resposta(
    graph: &[Vec<f32>],
    byte_sem: &[f32],
    temp: f32,
    seed: u64,
    max_len: usize,
) -> Vec<u8> {
    let mut rng = seed;
    let lcg = |s: &mut u64| -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 33) as f32 / (u32::MAX as f32)
    };

    // Apenas bytes úteis: ASCII imprimível + UTF-8 starters
    let valid: Vec<usize> = (0x20usize..=0x7E)
        .chain(0xC0..=0xF4)
        .collect();

    let mut response: Vec<u8> = Vec::new();
    let mut prev = b'O' as usize;

    for _ in 0..max_len {
        // Score = (graph * sem) com power-law temperature
        let mut cands: Vec<(usize, f32)> = valid.iter().map(|&b| {
            let p_graph = graph[prev][b] + 1e-9;
            let p_sem   = byte_sem[b].max(0.0) + 1e-9;
            (b, p_graph.powf(0.4) * p_sem.powf(0.6))
        }).collect();

        // Power-law temperature: sharpen/flatten a distribuição
        let sum: f32 = cands.iter().map(|(_,s)| s).sum::<f32>().max(1e-9);
        let inv_t = 1.0 / temp;
        cands.iter_mut().for_each(|(_, s)| *s = (*s / sum).powf(inv_t));
        let sum2: f32 = cands.iter().map(|(_, s)| s).sum::<f32>();
        cands.iter_mut().for_each(|(_, s)| *s /= sum2);

        let r = lcg(&mut rng);
        let mut cum = 0.0f32;
        let mut next = b'?' as usize;
        for (b, s) in &cands {
            cum += s;
            if r < cum { next = *b; break; }
        }

        if next == b'\n' as usize || next == b'.' as usize { break; }
        response.push(next as u8);
        prev = next;
    }
    response
}

// ── Helpers ────────────────────────────────────────────────────────
struct Meta { data_start: usize, embed_off: usize }

fn load_meta() -> Meta {
    let mut f = fs::File::open(format!("{MODEL}/model.safetensors")).unwrap();
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).unwrap();
    let hlen = u64::from_le_bytes(buf) as usize;
    let mut hbuf = vec![0u8; hlen];
    f.read_exact(&mut hbuf).unwrap();
    let h: HashMap<String, serde_json::Value> = serde_json::from_slice(&hbuf).unwrap();
    let off = h["model.embed_tokens.weight"]["data_offsets"][0].as_u64().unwrap() as usize;
    Meta { data_start: 8 + hlen, embed_off: off }
}

fn read_embed(f: &mut fs::File, st: &Meta, id: u32) -> Vec<f32> {
    let off = st.data_start + st.embed_off + id as usize * HIDDEN * 2;
    f.seek(SeekFrom::Start(off as u64)).unwrap();
    let mut buf = vec![0u8; HIDDEN * 2];
    f.read_exact(&mut buf).unwrap();
    buf.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn bar(v: f32) -> String {
    let n = (v.clamp(0.0, 1.0) * 20.0) as usize;
    format!("|{}{}|", "█".repeat(n), "░".repeat(20 - n))
}
