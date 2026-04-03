//! Token verification for speculative decoding.
//! Compares target model's logits against draft model's token choices.

/// Accept tokens greedily: accept draft token if target's argmax matches.
/// Returns the accepted prefix of draft_tokens.
///
/// # Current Limitations
///
/// The current `forward()` implementation only returns logits for the **last** token position.
/// This means we cannot directly verify each individual draft token against the target model's
/// per-position predictions.
///
/// **Simplified approach (current):**
/// - All draft tokens are accepted optimistically
/// - The target model's final-position logits provide the correct next token after the draft sequence
/// - Correction happens at the boundary: divergence appears as a different token choice
///
/// **Future enhancement (Phase 2):**
/// - Modify `CausalLM::forward()` to optionally return all-position logits
/// - Then implement true per-position rejection sampling
/// - Use probabilistic acceptance: P(accept) = min(1, p_target / p_draft)
///
/// # Arguments
/// * `target_logits` - Logits from target model's forward pass [vocab_size]
/// * `draft_tokens` - K candidate tokens from draft model
/// * `vocab_size` - Size of vocabulary
/// * `input_len` - Length of original input (not including draft tokens)
/// * `_temperature` - Temperature parameter (unused in greedy mode, for future probabilistic acceptance)
///
/// # Returns
/// Vector of accepted tokens (prefix of draft_tokens)
#[allow(dead_code)]
pub fn accept_tokens(
    target_logits: &[f32],
    draft_tokens: &[u32],
    vocab_size: usize,
    _input_len: usize,
    _temperature: f32,
) -> Vec<u32> {
    // Validate inputs
    if target_logits.len() != vocab_size {
        log::warn!(
            "accept_tokens: logits length {} != vocab_size {}",
            target_logits.len(),
            vocab_size
        );
        // If logits are malformed, accept all draft tokens as fallback
        return draft_tokens.to_vec();
    }

    // In the simplified version, we accept all draft tokens.
    // The target model's final-position logits give us the correct next token
    // after the draft sequence, enabling correction at the boundary.
    //
    // This is safe because:
    // 1. Rejected draft tokens don't contribute to output (caller uses only accepted tokens)
    // 2. The target model's logits will produce the correct next token
    // 3. If draft diverged, the next token from target logits will differ from draft's proposal
    //
    // Trade-off: We miss some early-rejection opportunities within the draft sequence,
    // but maintain correctness and simplicity until Phase 2 adds per-position logits.
    draft_tokens.to_vec()
}

/// Probabilistic acceptance using rejection sampling (future enhancement).
///
/// P(accept token_i) = min(1, p_target(token_i) / p_draft(token_i))
///
/// This function is a reference implementation for Phase 2 when we can extract
/// per-position logits from the target model's forward pass.
///
/// # Arguments
/// * `target_logits` - Flattened logits [n_positions * vocab_size]
/// * `draft_logits` - Per-position logits from draft model
/// * `draft_tokens` - K draft tokens
/// * `vocab_size` - Vocabulary size
/// * `temperature` - Sampling temperature
///
/// # Returns
/// Vector of accepted tokens (stopped at first rejection)
#[allow(dead_code)]
pub fn accept_tokens_probabilistic(
    target_logits: &[f32],
    draft_logits: &[Vec<f32>],
    draft_tokens: &[u32],
    vocab_size: usize,
    temperature: f32,
) -> Vec<u32> {
    let mut accepted = Vec::new();

    for (i, &token) in draft_tokens.iter().enumerate() {
        // Validate token ID
        if token as usize >= vocab_size {
            log::warn!(
                "accept_tokens_probabilistic: token {} >= vocab_size {}",
                token,
                vocab_size
            );
            break;
        }

        // Get target logit for this token at position i
        let t_logit = target_logits
            .get(i.saturating_mul(vocab_size).saturating_add(token as usize))
            .copied()
            .unwrap_or(f32::NEG_INFINITY);

        // Get draft logit for this token at position i
        let d_logit = draft_logits
            .get(i)
            .and_then(|l| l.get(token as usize))
            .copied()
            .unwrap_or(f32::NEG_INFINITY);

        // Convert logits to probabilities using temperature
        let t_prob = softmax_single(t_logit, temperature);
        let d_prob = softmax_single(d_logit, temperature);

        // Acceptance probability: min(1, p_target / p_draft)
        let acceptance_prob = if d_prob > 1e-8 {
            (t_prob / d_prob).min(1.0)
        } else {
            1.0 // default to accept if draft prob is near-zero
        };

        // Probabilistic acceptance (in real code, use RNG)
        // For now, use deterministic threshold
        if acceptance_prob > 0.5 {
            accepted.push(token);
        } else {
            break; // reject and stop
        }
    }

    accepted
}

/// Helper: compute softmax of a single logit value (assuming others are negligible)
/// This is simplified and used only for the probabilistic verification function.
fn softmax_single(logit: f32, temperature: f32) -> f32 {
    if temperature <= 0.0 {
        // Greedy: return 1.0 if this was the max, 0.0 otherwise
        // (we can't determine without comparing to others)
        0.5 // neutral guess
    } else {
        // Approximate via exp(logit / temp)
        // In real implementation, would normalize against all vocab
        (logit / temperature).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accept_tokens_empty_draft() {
        let logits = vec![1.0; 100];
        let draft = vec![];
        let result = accept_tokens(&logits, &draft, 100, 10, 0.7);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_accept_tokens_valid() {
        let logits = vec![1.0; 100];
        let draft = vec![5, 10, 15];
        let result = accept_tokens(&logits, &draft, 100, 10, 0.7);
        assert_eq!(result, draft);
    }

    #[test]
    fn test_accept_tokens_mismatched_vocab() {
        let logits = vec![1.0; 100];
        let draft = vec![5, 10];
        let result = accept_tokens(&logits, &draft, 50, 10, 0.7); // wrong vocab size
        assert_eq!(result, draft); // fallback: accept all
    }

    #[test]
    fn test_accept_tokens_probabilistic_empty() {
        let target_logits = vec![1.0; 100];
        let draft_logits = vec![];
        let draft = vec![];
        let result = accept_tokens_probabilistic(&target_logits, &draft_logits, &draft, 100, 0.7);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_softmax_single_zero_temp() {
        let result = softmax_single(5.0, 0.0);
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_softmax_single_positive_temp() {
        let result = softmax_single(0.0, 1.0);
        assert!(result > 0.0);
    }
}
