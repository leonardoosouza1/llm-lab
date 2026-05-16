//! validar.rs — Duas validações:
//! 1. Distância alfabética → distância no embedding (Qwen encoda ordem?)
//! 2. Novo 8-bit semântico → os grupos se separam no espaço Qwen?

use std::{collections::HashMap, fs, io::{Read, Seek, SeekFrom}};

const MODEL: &str  = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b";
const HIDDEN: usize = 1536;

fn main() {
    let vocab = load_vocab();
    let st    = load_meta();

    validar_ordem_alfabetica(&vocab, &st);
    validar_novo_8bit(&vocab, &st);
}

// ═══════════════════════════════════════════════════════════════
// VALIDAÇÃO 1 — Ordem alfabética como dimensão geométrica
// ═══════════════════════════════════════════════════════════════

fn validar_ordem_alfabetica(vocab: &HashMap<String, u32>, st: &Meta) {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  VALIDAÇÃO 1 — Qwen encoda posição no alfabeto?         ║");
    println!("║  Se sim: cosine(E[a], E[x]) cai conforme x se afasta   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let mut f = fs::File::open(format!("{MODEL}/model.safetensors")).unwrap();

    // Carrega embeddings de todas as letras minúsculas que são tokens únicos
    let mut embeds: Vec<(char, usize, Vec<f32>)> = Vec::new(); // (letra, pos_alfa, embed)
    for (i, c) in (b'a'..=b'z').enumerate() {
        let s = (c as char).to_string();
        if let Some(&id) = vocab.get(&s) {
            embeds.push((c as char, i + 1, read_embed(&mut f, st, id)));
        }
    }

    // Para cada âncora, mede cosine com todas as outras e agrupa por distância
    let ancoras = ['a', 'e', 'm', 'z'];
    for ancora in ancoras {
        let Some((_, pos_a, e_a)) = embeds.iter().find(|(c,_,_)| *c == ancora) else { continue };
        println!("  Âncora: '{ancora}' (posição {pos_a} no alfabeto)\n");
        println!("  dist_alfa  letra  cosine  barra");
        println!("  {}", "─".repeat(52));

        let mut pares: Vec<(i32, char, f32)> = embeds.iter()
            .filter(|(c,_,_)| *c != ancora)
            .map(|(c, pos_b, e_b)| {
                let dist = (*pos_b as i32 - *pos_a as i32).abs();
                let sim  = cosine(e_a, e_b);
                (dist, *c, sim)
            })
            .collect();

        pares.sort_by_key(|(d,_,_)| *d);

        for (dist, letra, sim) in &pares {
            println!("  dist={dist:2}  '{letra}'  {sim:+.4}  {}", bar(*sim));
        }

        // Correlação de Pearson: dist vs (1-cosine)
        let n = pares.len() as f32;
        let xs: Vec<f32> = pares.iter().map(|(d,_,_)| *d as f32).collect();
        let ys: Vec<f32> = pares.iter().map(|(_,_,s)| 1.0 - s).collect(); // distância embedding
        let pearson = pearson_r(&xs, &ys);
        println!("\n  ▸ Pearson r(dist_alfa, dist_embed) = {pearson:+.4}");
        println!("    {} = Qwen {} encoda ordem alfabética geometricamente\n",
            if pearson > 0.5 { "r > 0.5" } else { "r ≤ 0.5" },
            if pearson > 0.5 { "SIM" } else { "NÃO MUITO" });
    }

    // Resumo por faixas de distância
    println!("  Resumo: cosine médio por faixa de distância alfabética\n");
    println!("  faixa    letras incluídas       cosine_médio  barra");
    println!("  {}", "─".repeat(56));

    let e_a = &embeds.iter().find(|(c,_,_)| *c == 'a').unwrap().2;
    let mut por_faixa: Vec<(String, Vec<f32>)> = vec![
        ("dist 1-3  ".to_string(),  Vec::new()),
        ("dist 4-8  ".to_string(),  Vec::new()),
        ("dist 9-16 ".to_string(), Vec::new()),
        ("dist 17-25".to_string(), Vec::new()),
    ];

    for (c, pos, e) in &embeds {
        if *c == 'a' { continue; }
        let dist = (*pos as i32 - 1).abs();
        let sim  = cosine(e_a, e);
        let bucket = match dist {
            1..=3   => 0,
            4..=8   => 1,
            9..=16  => 2,
            _       => 3,
        };
        por_faixa[bucket].1.push(sim);
    }

    for (label, sims) in &por_faixa {
        if sims.is_empty() { continue; }
        let media = sims.iter().sum::<f32>() / sims.len() as f32;
        println!("  {label}   {media:+.4}  {}", bar(media));
    }
}

// ═══════════════════════════════════════════════════════════════
// VALIDAÇÃO 2 — Novo 8-bit semântico
// ═══════════════════════════════════════════════════════════════

fn validar_novo_8bit(vocab: &HashMap<String, u32>, st: &Meta) {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  VALIDAÇÃO 2 — Novo 8-bit semântico                     ║");
    println!("║  Os 2 bits livres com classe semântica separam grupos?  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let mut f = fs::File::open(format!("{MODEL}/model.safetensors")).unwrap();

    // Re-encoding: byte → (sem_class: u8, label: &str)
    // bit7,bit6 originais (fixos em 01 para letras) → substituídos por sem1,sem0
    let sem_class = |b: u8| -> (u8, &'static str) {
        match b {
            b'a'|b'e'|b'i'|b'o'|b'u' => (0b00, "vogal    "),
            b'A'|b'E'|b'I'|b'O'|b'U' => (0b00, "vogal    "),
            b'n'|b'r'|b's'|b't'|b'l'|b'm'|b'c'|b'p'|b'd'|b'b'
            |b'N'|b'R'|b'S'|b'T'|b'L'|b'M'|b'C'|b'P'|b'D'|b'B' => (0b01, "cons_com "),
            b'q'|b'x'|b'z'|b'w'|b'k'|b'f'|b'v'|b'j'|b'h'|b'g'|b'y'
            |b'Q'|b'X'|b'Z'|b'W'|b'K'|b'F'|b'V'|b'J'|b'H'|b'G'|b'Y' => (0b10, "cons_rara"),
            _ => (0b11, "especial "),
        }
    };

    // Novo byte: [sem1|sem0|case|grp|nl3|nl2|nl1|nl0]
    let novo_byte = |b: u8| -> u8 {
        let (sem, _) = sem_class(b);
        let case_bit = if b.is_ascii_uppercase() { 0u8 } else { 1u8 };
        let grp_bit  = if b & 0x0F == 0 && b.is_ascii_alphabetic() { 1u8 }
                       else if (b & 0xF0) >= 0x70 && b.is_ascii_alphabetic() { 1u8 }
                       else { 0u8 };
        let nl = b & 0x0F;
        (sem << 6) | (case_bit << 5) | (grp_bit << 4) | nl
    };

    println!("  Tabela de re-encoding (letras minúsculas a-z):\n");
    println!("  orig   char  sem_class  orig_byte    novo_byte   mudou?");
    println!("  {}", "─".repeat(60));
    for c in b'a'..=b'z' {
        let (_, label) = sem_class(c);
        let nb = novo_byte(c);
        let mudou = if nb == c { "  —" } else { "  ✎" };
        println!("  0x{c:02X}  '{}'  {label}  {:08b}  {:08b}{mudou}",
            c as char, c, nb);
    }

    // Carrega embeddings por classe semântica
    let classes = ["vogal    ", "cons_com ", "cons_rara", "especial "];
    let mut por_classe: Vec<Vec<(u8, Vec<f32>)>> = vec![vec![]; 4];

    let letras_test: &[u8] = b"aeiounsrtlmcpdbqxzwkfvjhgy";
    for &c in letras_test {
        let s = (c as char).to_string();
        if let Some(&id) = vocab.get(&s) {
            let (sem, _) = sem_class(c);
            let e = read_embed(&mut f, st, id);
            por_classe[sem as usize].push((c, e));
        }
    }

    // Cosine médio intra-classe vs inter-classe
    println!("\n  Similaridade intra-classe (mesmo sem_bits):");
    println!("  {}", "─".repeat(56));

    let mut intra_medias: Vec<f32> = Vec::new();
    for (cls_idx, membros) in por_classe.iter().enumerate() {
        if membros.len() < 2 { continue; }
        let mut sims: Vec<f32> = Vec::new();
        for i in 0..membros.len() {
            for j in (i+1)..membros.len() {
                sims.push(cosine(&membros[i].1, &membros[j].1));
            }
        }
        let media = sims.iter().sum::<f32>() / sims.len() as f32;
        let letras: String = membros.iter().map(|(c,_)| *c as char).collect::<Vec<_>>()
            .iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
        println!("  {} {}  média={media:+.4}  {}  [{}]",
            classes[cls_idx], bar(media), format!("n={}", membros.len()), letras);
        intra_medias.push(media);
    }

    // Inter-classe: pega dois grupos e mede entre eles
    println!("\n  Similaridade inter-classe (sem_bits diferentes):");
    println!("  {}", "─".repeat(56));

    let pares_inter = [(0usize, 1usize), (0, 2), (1, 2)];
    let mut inter_medias: Vec<f32> = Vec::new();
    for (ci, cj) in pares_inter {
        if por_classe[ci].is_empty() || por_classe[cj].is_empty() { continue; }
        let mut sims: Vec<f32> = Vec::new();
        for (_, ei) in &por_classe[ci] {
            for (_, ej) in &por_classe[cj] {
                sims.push(cosine(ei, ej));
            }
        }
        let media = sims.iter().sum::<f32>() / sims.len() as f32;
        println!("  {} vs {}  média={media:+.4}  {}",
            classes[ci], classes[cj], bar(media));
        inter_medias.push(media);
    }

    let media_intra = intra_medias.iter().sum::<f32>() / intra_medias.len() as f32;
    let media_inter = inter_medias.iter().sum::<f32>() / inter_medias.len() as f32;
    let separacao   = media_intra - media_inter;

    println!("\n  ▸ Média intra-classe:  {media_intra:+.4}");
    println!("  ▸ Média inter-classe:  {media_inter:+.4}");
    println!("  ▸ Separação:          {separacao:+.4}  (positivo = grupos reais)");
    println!();

    if separacao > 0.05 {
        println!("  ✓ VALIDADO: os 2 bits semânticos capturam estrutura REAL");
        println!("    Grupos (vogal/cons_com/cons_rara) existem no espaço Qwen");
        println!("    → Vale a pena substituir os bits fixos por esses 2 bits");
    } else {
        println!("  ✗ Separação fraca: os grupos não se destacam o suficiente");
        println!("    → Refinar a classificação semântica");
    }

    // Demonstração visual do novo espaço
    println!("\n  Novo byte de 'a' (vogal, min, grp0, pos1):");
    let nb_a = novo_byte(b'a');
    println!("  original: 0x61 = 01100001  [0|1|1|0|0|0|0|1]  (bit6=fixo,bit7=fixo)");
    println!("  novo:     0x{nb_a:02X} = {nb_a:08b}  [0|0|1|0|0|0|0|1]  (bit6=vogal0,bit7=vogal1)");
    println!();
    println!("  Novo byte de 's' (cons_com, min, grp1, pos3):");
    let nb_s = novo_byte(b's');
    println!("  original: 0x73 = {:08b}", b's');
    println!("  novo:     0x{nb_s:02X} = {nb_s:08b}");
}

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

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

fn load_vocab() -> HashMap<String, u32> {
    let raw = fs::read_to_string(format!("{MODEL}/vocab.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x,y)| x*y).sum();
    let na: f32  = a.iter().map(|x| x*x).sum::<f32>().sqrt();
    let nb: f32  = b.iter().map(|x| x*x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-8)
}

fn bar(sim: f32) -> String {
    let filled = ((sim.clamp(-0.2, 0.8) + 0.2) / 1.0 * 24.0) as usize;
    format!("|{}{}|", "█".repeat(filled.min(24)), "░".repeat(24usize.saturating_sub(filled)))
}

fn pearson_r(xs: &[f32], ys: &[f32]) -> f32 {
    let n  = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let num: f32 = xs.iter().zip(ys).map(|(x,y)| (x-mx)*(y-my)).sum();
    let dx: f32  = xs.iter().map(|x| (x-mx).powi(2)).sum::<f32>().sqrt();
    let dy: f32  = ys.iter().map(|y| (y-my).powi(2)).sum::<f32>().sqrt();
    num / (dx * dy + 1e-8)
}
