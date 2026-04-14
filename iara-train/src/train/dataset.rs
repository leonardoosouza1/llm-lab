//! Dataset and data pipeline for causal language model training.
//!
//! Supports two input formats:
//! - Pre-tokenized binary files: flat sequence of little-endian u32 token IDs
//! - Plain text files: one document per line, converted to byte indices
//!
//! Each dataset item is a sliding window of `seq_len + 1` tokens, yielding
//! `input_ids = tokens[..seq_len]` and `target_ids = tokens[1..seq_len+1]`.

use std::fs;
use std::io::Read;
use std::path::Path;

// ---------------------------------------------------------------------------
// Dataset item: a single training window
// ---------------------------------------------------------------------------

/// A single training sample -- a window of token IDs.
#[derive(Clone, Debug)]
pub struct TokenWindow {
    /// Input token IDs: `tokens[i .. i + seq_len]`
    pub input_ids: Vec<u32>,
    /// Target token IDs: `tokens[i+1 .. i + seq_len + 1]`
    pub target_ids: Vec<u32>,
}

// ---------------------------------------------------------------------------
// TextDataset
// ---------------------------------------------------------------------------

/// Simple dataset that slices a flat token stream into overlapping windows
/// for next-token prediction training.
pub struct TextDataset {
    /// Flat token stream loaded from file.
    tokens: Vec<u32>,
    /// Context window length (each item has seq_len input tokens).
    seq_len: usize,
}

impl TextDataset {
    /// Load pre-tokenized binary data.
    ///
    /// The file must be a flat sequence of **little-endian u32** token IDs.
    /// This is the recommended format for large datasets: tokenize offline with
    /// llama.cpp or HuggingFace tokenizers, then write the IDs to a `.bin` file.
    pub fn from_token_file(path: &Path, seq_len: usize) -> Self {
        let mut file = fs::File::open(path)
            .unwrap_or_else(|e| panic!("cannot open token file {}: {e}", path.display()));
        let meta = file
            .metadata()
            .unwrap_or_else(|e| panic!("cannot stat {}: {e}", path.display()));
        let n_bytes = meta.len() as usize;
        assert!(n_bytes % 4 == 0, "token file size must be a multiple of 4 bytes");

        let n_tokens = n_bytes / 4;
        let mut buf = vec![0u8; n_bytes];
        file.read_exact(&mut buf)
            .unwrap_or_else(|e| panic!("read error on {}: {e}", path.display()));

        let tokens: Vec<u32> = buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert!(
            n_tokens > seq_len,
            "token file has {n_tokens} tokens but seq_len is {seq_len} -- need at least seq_len+1"
        );

        log::info!(
            "TextDataset: loaded {} tokens from {} ({} windows of seq_len={})",
            tokens.len(),
            path.display(),
            tokens.len().saturating_sub(seq_len),
            seq_len
        );

        Self { tokens, seq_len }
    }

    /// Load a plain-text file, treating each byte as a token ID.
    ///
    /// This is a **debug / smoke-test** path: it maps raw UTF-8 bytes to
    /// token IDs (0..255). For real training, pre-tokenize with a proper BPE
    /// tokenizer and use [`from_token_file`](Self::from_token_file).
    pub fn from_text_file(path: &Path, seq_len: usize) -> Self {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read text file {}: {e}", path.display()));

        let tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();

        assert!(
            tokens.len() > seq_len,
            "text file has {} bytes but seq_len is {seq_len}",
            tokens.len()
        );

        log::info!(
            "TextDataset: loaded {} byte-tokens from {} ({} windows)",
            tokens.len(),
            path.display(),
            tokens.len().saturating_sub(seq_len),
        );

        Self { tokens, seq_len }
    }

    /// Build from a pre-existing token vector (useful for tests / programmatic use).
    pub fn from_tokens(tokens: Vec<u32>, seq_len: usize) -> Self {
        assert!(
            tokens.len() > seq_len,
            "need at least seq_len+1 tokens, got {}",
            tokens.len()
        );
        Self { tokens, seq_len }
    }

    /// Total number of sliding-window samples.
    pub fn num_samples(&self) -> usize {
        self.tokens.len().saturating_sub(self.seq_len)
    }

    /// Get a single training window by index.
    pub fn get(&self, index: usize) -> Option<TokenWindow> {
        if index >= self.num_samples() {
            return None;
        }

        let start = index;
        let end = start + self.seq_len + 1; // +1 because target is shifted by 1

        let window = &self.tokens[start..end];
        let input_ids: Vec<u32> = window[..self.seq_len].to_vec();
        let target_ids: Vec<u32> = window[1..].to_vec();

        Some(TokenWindow {
            input_ids,
            target_ids,
        })
    }

    /// Return the sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Return reference to the raw token stream.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_from_tokens() {
        let tokens: Vec<u32> = (0..20).collect();
        let ds = TextDataset::from_tokens(tokens, 4);

        // 20 tokens, seq_len=4 => 20-4 = 16 windows
        assert_eq!(ds.num_samples(), 16);

        let item = ds.get(0).unwrap();
        assert_eq!(item.input_ids, vec![0, 1, 2, 3]);
        assert_eq!(item.target_ids, vec![1, 2, 3, 4]);

        let item = ds.get(15).unwrap();
        assert_eq!(item.input_ids, vec![15, 16, 17, 18]);
        assert_eq!(item.target_ids, vec![16, 17, 18, 19]);

        assert!(ds.get(16).is_none());
    }
}
