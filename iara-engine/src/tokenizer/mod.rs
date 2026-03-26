// ============================================================================
// tokenizer.rs — Tokenizador BPE real via HuggingFace tokenizers
// ============================================================================
// Carrega o tokenizer.json do modelo e faz:
//   - Encoding: texto → token IDs reais do modelo
//   - Decoding: token ID → texto legível
// Suporta recarga ao trocar de modelo (usa RwLock em vez de OnceLock).
// ============================================================================

use std::path::Path;
use std::sync::{RwLock, OnceLock};
use tokenizers::Tokenizer;

/// Tokenizer global recarregável (None quando nenhum modelo está carregado)
static TOKENIZER: OnceLock<RwLock<Option<Tokenizer>>> = OnceLock::new();

fn get_lock() -> &'static RwLock<Option<Tokenizer>> {
    TOKENIZER.get_or_init(|| RwLock::new(None))
}

/// Token: ID numérico + representação textual
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u32,
    pub text: String,
}

/// Carrega (ou recarrega) o tokenizer a partir do tokenizer.json do modelo.
/// Pode ser chamado múltiplas vezes para trocar de modelo.
pub fn load_tokenizer(model_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !tokenizer_path.exists() {
        return Err(format!(
            "tokenizer.json não encontrado em {}",
            model_dir.display()
        )
        .into());
    }

    let tok = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Erro ao carregar tokenizer: {}", e))?;

    let vocab_size = tok.get_vocab_size(true);
    log::info!("🔤 Tokenizer carregado: {} tokens no vocabulário", vocab_size);

    *get_lock().write().unwrap() = Some(tok);

    Ok(())
}

/// Verifica se o tokenizer está carregado
pub fn is_loaded() -> bool {
    get_lock().read().unwrap().is_some()
}

/// Tokeniza texto usando BPE real do modelo
pub fn tokenize(text: &str) -> Vec<Token> {
    let guard = get_lock().read().unwrap();
    let tok = match guard.as_ref() {
        Some(t) => t,
        None => {
            log::warn!("⚠️ Tokenizer não carregado, usando fallback byte-level");
            drop(guard);
            return tokenize_fallback(text);
        }
    };

    let encoding = match tok.encode(text, false) {
        Ok(e) => e,
        Err(err) => {
            log::error!("❌ Erro ao tokenizar '{}': {}", text, err);
            drop(guard);
            return tokenize_fallback(text);
        }
    };

    let ids = encoding.get_ids();
    let token_strs = encoding.get_tokens();

    let tokens: Vec<Token> = ids
        .iter()
        .zip(token_strs.iter())
        .map(|(&id, text)| Token { id, text: text.clone() })
        .collect();

    log::debug!(
        "🔤 Tokenizado: '{}' → {} tokens",
        &text[..text.len().min(40)],
        tokens.len(),
    );

    tokens
}

/// Converte um token ID de volta para texto usando o vocabulário real
pub fn token_id_to_text(id: u32) -> String {
    let guard = get_lock().read().unwrap();
    if let Some(tok) = guard.as_ref() {
        match tok.decode(&[id], false) {
            Ok(text) if !text.is_empty() => return text,
            _ => {}
        }
    }
    // Fallback
    if id < 128 {
        let b = id as u8;
        if b.is_ascii_graphic() || b == b' ' {
            return String::from(b as char);
        }
    }
    format!("[{}]", id)
}

/// Decodifica múltiplos token IDs em texto
pub fn decode_ids(ids: &[u32]) -> String {
    let guard = get_lock().read().unwrap();
    if let Some(tok) = guard.as_ref() {
        tok.decode(ids, true).unwrap_or_else(|_| {
            ids.iter().map(|&id| format!("[{}]", id)).collect::<Vec<_>>().join("")
        })
    } else {
        ids.iter().map(|&id| token_id_to_text(id)).collect::<Vec<_>>().join("")
    }
}

/// Fallback: tokenize byte-level (usado quando tokenizer.json não está disponível)
fn tokenize_fallback(text: &str) -> Vec<Token> {
    text.bytes()
        .map(|b| Token {
            id: b as u32,
            text: if b.is_ascii_graphic() || b == b' ' {
                String::from(b as char)
            } else {
                format!("<{:02X}>", b)
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_fallback() {
        let tokens = tokenize_fallback("Hi");
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_token_id_to_text_no_tokenizer() {
        let text = token_id_to_text(65);
        assert!(!text.is_empty());
    }
}
