//! medir.rs — Valida a tese dos nibbles e dos bits "livres" do ASCII
//!
//! Mede 4 coisas:
//!   1. Pares case (A/a): confirmação que bit5 é trivial
//!   2. Nibble_l vs similaridade de embedding
//!   3. Os 2 bits "livres" do ASCII — quanto valem?
//!   4. Proposta de re-encoding semântico dos 4 bits altos

use std::{collections::HashMap, fs, io::{Read, Seek, SeekFrom}};

const MODEL: &str = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b";
const HIDDEN: usize = 1536;

fn main() {
    let vocab = load_vocab();
    let safetensors = load_safetensors_meta();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   MEDIÇÃO 1 — Pares Maiúsculo/Minúsculo                 ║");
    println!("║   Tese: bit5 é trivial, upper≈lower em embedding        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    medir_case_pairs(&vocab, &safetensors);

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║   MEDIÇÃO 2 — Nibble_l: identidade ou ruído?            ║");
    println!("║   Tese: mesmo nibble_l = mesma letra, diferente grupo   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    medir_nibble_l(&vocab, &safetensors);

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║   MEDIÇÃO 3 — Os 2 bits livres do ASCII                 ║");
    println!("║   bit7=0, bit6=1 sempre para letras → podemos roubar   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    medir_bits_livres();

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║   MEDIÇÃO 4 — Re-encoding: 4 bits semânticos            ║");
    println!("║   Substituir nibble_h por vogal/consoante/freq/posição  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    medir_reencoding(&vocab, &safetensors);
}

// ═══════════════════════════════════════════════════
// MEDIÇÃO 1: pares upper/lower
// ═══════════════════════════════════════════════════

fn medir_case_pairs(vocab: &HashMap<String, u32>, st: &SafetensorsMeta) {
    let mut path = fs::File::open(format!("{MODEL}/model.safetensors")).unwrap();
    let mut resultados: Vec<(char, f32, f32)> = Vec::new(); // letra, cosine, bit_dist

    for c in b'a'..=b'z' {
        let lower = (c as char).to_string();
        let upper = (c as char).to_ascii_uppercase().to_string();

        let (Some(&id_l), Some(&id_u)) = (vocab.get(&lower), vocab.get(&upper)) else { continue };

        let e_l = read_embed(&mut path, st, id_l);
        let e_u = read_embed(&mut path, st, id_u);
        let sim = cosine(&e_l, &e_u);

        // distância de bits: só o bit5 muda (valor 32)
        // lower = 0110xxxx, upper = 0100xxxx — diferem em bit5
        let bit_dist = 1u32; // sempre 1 bit de diferença para case

        resultados.push((c as char, sim, bit_dist as f32));
    }

    // Ordena por similaridade
    resultados.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Letra   cosine(E[upper], E[lower])   nibble_l");
    println!("  {}", "─".repeat(48));
    for (letra, sim, _) in &resultados {
        let nibble_l = *letra as u8 & 0x0F;
        let barra = sim_bar(*sim);
        println!("  '{letra}'/{c}      {sim:+.4}   {barra}   nibble_l={nibble_l}",
            c = letra.to_ascii_uppercase());
    }

    let media: f32 = resultados.iter().map(|(_, s, _)| s).sum::<f32>() / resultados.len() as f32;
    let min = resultados.iter().map(|(_, s, _)| *s).fold(f32::INFINITY, f32::min);
    let max = resultados.iter().map(|(_, s, _)| *s).fold(f32::NEG_INFINITY, f32::max);

    println!("\n  ▸ Média:  {media:.4}");
    println!("  ▸ Mínimo: {min:.4}");
    println!("  ▸ Máximo: {max:.4}");
    println!("\n  → Bit5 (case) muda o embedding em média {:.1}%", (1.0 - media) * 100.0);
}

// ═══════════════════════════════════════════════════
// MEDIÇÃO 2: nibble_l — identidade ou coincidência?
// ═══════════════════════════════════════════════════

fn medir_nibble_l(vocab: &HashMap<String, u32>, st: &SafetensorsMeta) {
    let mut path = fs::File::open(format!("{MODEL}/model.safetensors")).unwrap();

    println!("  Grupos de letras com mesmo nibble_l:\n");
    println!("  nibble_l  letras (grupo0/grupo1)  cosine intra-grupo");
    println!("  {}", "─".repeat(60));

    for nl in 1u8..=15 {
        // Grupo 0 (A-O): nibble_h=6 (lower), 4 (upper) — letras 1-15
        // Grupo 1 (P-Z): nibble_h=7 (lower), 5 (upper) — posições 0-10
        let byte_g0_low = 0x60u8 | nl; // ex: nl=1 → 0x61 = 'a'
        let byte_g1_low = 0x70u8 | nl; // ex: nl=1 → 0x71 = 'q'

        if !byte_g0_low.is_ascii_alphabetic() || !byte_g1_low.is_ascii_alphabetic() {
            continue;
        }

        let c0 = byte_g0_low as char; // ex: 'a'
        let c1 = byte_g1_low as char; // ex: 'q'

        let (Some(&id0), Some(&id1)) = (
            vocab.get(&c0.to_string()),
            vocab.get(&c1.to_string()),
        ) else { continue };

        let e0 = read_embed(&mut path, st, id0);
        let e1 = read_embed(&mut path, st, id1);
        let sim = cosine(&e0, &e1);

        let barra = sim_bar(sim);
        println!("  nl={nl:2}  '{c0}' vs '{c1}'  (mesmo nibble_l, grupos diferentes)  {sim:+.4}  {barra}");
    }

    println!("\n  → Se cosine baixo: nibble_l SOZINHO não basta — precisa do bit4 (grupo)");
    println!("  → Se cosine alto: a posição mod 16 captura algo real\n");

    // Compara: par case ('a'/'A') vs par nibble ('a'/'q')
    println!("  Comparação direta:\n");
    for (label, ta, tb) in [
        ("case: 'a'/'A' (só bit5 diferente)", "a", "A"),
        ("nibble: 'a'/'q' (mesmo nl=1, grupos dif)", "a", "q"),
        ("nibble: 'e'/'u' (mesmo nl=5, grupos dif)", "e", "u"),
        ("vizinhos: 'a'/'b' (nl=1 vs nl=2)", "a", "b"),
        ("distantes: 'a'/'z' (nl=1 vs nl=10)", "a", "z"),
    ] {
        let (Some(&id_a), Some(&id_b)) = (vocab.get(ta), vocab.get(tb)) else { continue };
        let ea = read_embed(&mut path, st, id_a);
        let eb = read_embed(&mut path, st, id_b);
        let sim = cosine(&ea, &eb);
        println!("  {label:48}  cosine = {sim:+.4}  {}", sim_bar(sim));
    }
}

// ═══════════════════════════════════════════════════
// MEDIÇÃO 3: os 2 bits livres
// ═══════════════════════════════════════════════════

fn medir_bits_livres() {
    println!("  Análise bit a bit das letras ASCII:\n");
    println!("  {:>4}  {:>10}  b7 b6 | b5 b4 | b3 b2 b1 b0  nibble_h  nibble_l  pos_alfa",
        "byte", "char");
    println!("  {}", "─".repeat(72));

    for &c in b"aeiouAEIOUqzQZ pN".iter() {
        let label = format!("{:?}", c as char);
        let b7 = (c >> 7) & 1;
        let b6 = (c >> 6) & 1;
        let b5 = (c >> 5) & 1; // case
        let b4 = (c >> 4) & 1; // grupo
        let b3 = (c >> 3) & 1;
        let b2 = (c >> 2) & 1;
        let b1 = (c >> 1) & 1;
        let b0 = c & 1;
        let nh = c >> 4;
        let nl = c & 0x0F;
        let pos = if c.is_ascii_alphabetic() {
            let base = if c.is_ascii_uppercase() { b'A' } else { b'a' };
            format!("{}", c - base + 1)
        } else { "—".to_string() };

        // Marca bits "livres" (b7=0, b6=1 para letras)
        let b7_mark = if c.is_ascii_alphabetic() && b7 == 0 { "0✓" } else { &format!("{b7} ") };
        let b6_mark = if c.is_ascii_alphabetic() && b6 == 1 { "1✓" } else { &format!("{b6} ") };

        println!("  {c:>4}  {label:>10}  {b7_mark} {b6_mark} | {b5}  {b4} | {b3}  {b2}  {b1}  {b0}  \
                  nh={nh:2}  nl={nl:2}  pos={pos}");
    }

    println!("\n  ✓ = bits SEMPRE fixos para letras ASCII → 0 entropia → LIVRES para roubar\n");
    println!("  Proposta de re-encoding (mantém 8 bits, muda o significado):");
    println!();
    println!("  ORIGINAL:   [ b7=0 | b6=1 | case | grp | nl3 | nl2 | nl1 | nl0 ]");
    println!("                 fixo   fixo   info   info  ←——— identidade ———→");
    println!();
    println!("  NOVO:       [ sem1 | sem0 | case | grp | nl3 | nl2 | nl1 | nl0 ]");
    println!("               ←semântica→   info   info  ←——— identidade ———→");
    println!();
    println!("  sem1,sem0 pode codificar:");
    println!("    00 = vogal              (a, e, i, o, u)");
    println!("    01 = consoante comum    (n, r, s, t, l, ...)");
    println!("    10 = consoante rara     (q, x, z, w, k)");
    println!("    11 = acento/especial    (ã, ç, á, é, UTF-8 seq)");
}

// ═══════════════════════════════════════════════════
// MEDIÇÃO 4: re-encoding semântico
// ═══════════════════════════════════════════════════

fn medir_reencoding(vocab: &HashMap<String, u32>, st: &SafetensorsMeta) {
    let mut path = fs::File::open(format!("{MODEL}/model.safetensors")).unwrap();

    // Categorias semânticas PT-BR
    let vogais:      &[u8] = b"aeiouAEIOU";
    let cons_comuns: &[u8] = b"nrstlmcpdbNRSTLMCPDB";
    let cons_raras:  &[u8] = b"qxzwkfvjQXZWKFVJ";

    let sem_class = |b: u8| -> (&str, u8) {
        if vogais.contains(&b)      { ("vogal    ", 0b00) }
        else if cons_comuns.contains(&b) { ("cons_com ", 0b01) }
        else if cons_raras.contains(&b)  { ("cons_rara", 0b10) }
        else                             { ("especial ", 0b11) }
    };

    println!("  Re-encoding proposto para letras básicas:\n");
    println!("  {:>4}  {:>6}  classe      sem  case  grp  nl  → novo_byte",
        "byte", "char");
    println!("  {}", "─".repeat(60));

    for c in b'a'..=b'z' {
        let (class_name, sem_bits) = sem_class(c);
        let case_bit  = 1u8; // minúsculo
        let group_bit = if c > b'o' { 1u8 } else { 0u8 };
        let nl        = c & 0x0F;

        // Novo byte: [sem1|sem0|case|grp|nl3|nl2|nl1|nl0]
        let novo = (sem_bits << 6) | (case_bit << 5) | (group_bit << 4) | nl;

        println!("  {:>4}  {:>6}  {class_name}  {:02b}   {:1}     {:1}    {:2}  → 0x{novo:02X} = {:08b}",
            c, c as char, sem_bits, case_bit, group_bit, nl, novo);
    }

    // Valida: vogais têm embeddings similares entre si?
    println!("\n  ▸ Validação: vogais PT-BR formam cluster no espaço Qwen?\n");
    let vogais_chars = ['a', 'e', 'i', 'o', 'u'];
    let mut vogal_embeds: Vec<(char, Vec<f32>)> = Vec::new();
    for &v in &vogais_chars {
        if let Some(&id) = vocab.get(&v.to_string()) {
            vogal_embeds.push((v, read_embed(&mut path, st, id)));
        }
    }

    println!("  Similaridade entre vogais:");
    for i in 0..vogal_embeds.len() {
        for j in (i+1)..vogal_embeds.len() {
            let sim = cosine(&vogal_embeds[i].1, &vogal_embeds[j].1);
            let barra = sim_bar(sim);
            println!("    '{}' ↔ '{}'  {sim:+.4}  {barra}",
                vogal_embeds[i].0, vogal_embeds[j].0);
        }
    }

    // Vogal vs consoante comum
    println!("\n  Vogal vs consoante (esperado: menos similar):");
    let pares_cross = [('a','n'), ('e','s'), ('i','t'), ('o','r')];
    for (v, c) in pares_cross {
        let (Some(&iv), Some(&ic)) = (vocab.get(&v.to_string()), vocab.get(&c.to_string())) else { continue };
        let ev = read_embed(&mut path, st, iv);
        let ec = read_embed(&mut path, st, ic);
        let sim = cosine(&ev, &ec);
        println!("    '{v}' ↔ '{c}'  {sim:+.4}  {}", sim_bar(sim));
    }

    println!("\n  → Se vogais são mais similares entre si que vogal/consoante:");
    println!("    sem_bits=00 (vogal) captura algo REAL do espaço semântico Qwen");
    println!("    = os 2 bits livres podem carregar informação semântica genuína");
}

// ═══════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════

struct SafetensorsMeta {
    data_start: usize,
    embed_offset: usize,
}

fn load_safetensors_meta() -> SafetensorsMeta {
    let mut f = fs::File::open(format!("{MODEL}/model.safetensors")).unwrap();
    let mut hlen_buf = [0u8; 8];
    f.read_exact(&mut hlen_buf).unwrap();
    let hlen = u64::from_le_bytes(hlen_buf) as usize;
    let mut hbuf = vec![0u8; hlen];
    f.read_exact(&mut hbuf).unwrap();
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&hbuf).unwrap();
    let embed_offset = header["model.embed_tokens.weight"]["data_offsets"][0]
        .as_u64().unwrap() as usize;
    SafetensorsMeta { data_start: 8 + hlen, embed_offset }
}

fn load_vocab() -> HashMap<String, u32> {
    let raw = fs::read_to_string(format!("{MODEL}/vocab.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn read_embed(f: &mut fs::File, st: &SafetensorsMeta, id: u32) -> Vec<f32> {
    let off = st.data_start + st.embed_offset + id as usize * HIDDEN * 2;
    f.seek(SeekFrom::Start(off as u64)).unwrap();
    let mut buf = vec![0u8; HIDDEN * 2];
    f.read_exact(&mut buf).unwrap();
    buf.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32  = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32  = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-8)
}

fn sim_bar(sim: f32) -> String {
    let clamped = sim.clamp(-1.0, 1.0);
    let filled = ((clamped + 1.0) / 2.0 * 20.0) as usize;
    format!("|{}{}|", "█".repeat(filled), "░".repeat(20 - filled))
}
