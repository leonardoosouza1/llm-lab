//! Continuous batching server for llama.cpp
//!
//! TRUE interleaved batching: multiple sequences share a single llama_decode call.
//! Each request gets its own sequence ID (0..MAX_SEQ-1) and KV cache slot.
//! The decode loop builds a merged batch with one token per active sequence,
//! processes them all at once, then samples independently for each.
//!
//! This reuses weight reads across sequences — the GPU reads model weights ONCE
//! but produces tokens for ALL active sequences. Aggregate throughput scales
//! nearly linearly with batch size for memory-bound workloads.

use std::ffi::CString;
use std::time::Instant;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};

use llama_sys::*;
use super::engine::GenerateParams;
use super::GenerateResult;

/// Maximum concurrent sequences
const MAX_SEQ: usize = 4;

struct PendingRequest {
    prompt: String,
    params: GenerateParams,
    response_tx: oneshot::Sender<Result<GenerateResult, String>>,
}

/// Handle for submitting requests to the batch server
#[derive(Clone)]
pub struct BatchHandle {
    tx: mpsc::Sender<PendingRequest>,
}

impl BatchHandle {
    pub async fn submit(
        &self,
        prompt: String,
        params: GenerateParams,
    ) -> Result<GenerateResult, String> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(PendingRequest { prompt, params, response_tx: resp_tx })
            .await
            .map_err(|_| "batch server shut down".to_string())?;
        resp_rx.await.map_err(|_| "batch server dropped response".to_string())?
    }
}

/// Active sequence state during interleaved decode
struct ActiveSeq {
    seq_id: i32,
    next_token: llama_token,
    n_past: i32,
    n_decoded: usize,
    max_tokens: usize,
    text: String,
    done: bool,
    t0: Instant,
    n_prompt: usize,
    prefill_ms: u128,
    response_tx: Option<oneshot::Sender<Result<GenerateResult, String>>>,
    sampler: *mut llama_sampler,
}

pub fn start_batch_server(
    engine: Arc<Mutex<Option<super::LlamaEngine>>>,
) -> BatchHandle {
    let (tx, mut rx) = mpsc::channel::<PendingRequest>(32);

    tokio::spawn(async move {
        loop {
            // Wait for at least one request
            let first = match rx.recv().await {
                Some(r) => r,
                None => break,
            };

            let mut incoming = vec![first];

            // Drain more (non-blocking), up to MAX_SEQ
            while incoming.len() < MAX_SEQ {
                match rx.try_recv() {
                    Ok(r) => incoming.push(r),
                    Err(_) => break,
                }
            }

            let n_requests = incoming.len();

            // Lock engine for the entire batch processing
            let mut guard = engine.lock().await;
            let engine = match guard.as_mut() {
                Some(e) => e,
                None => {
                    for req in incoming {
                        let _ = req.response_tx.send(Err("model not loaded".into()));
                    }
                    continue;
                }
            };

            if n_requests == 1 {
                // Single request — use standard path (no interleave overhead)
                let req = incoming.pop().unwrap();
                let result = engine.generate(&req.prompt, &req.params);
                let _ = req.response_tx.send(result);
                continue;
            }

            // ── Interleaved batch processing ────────────────────────────
            log::info!("batch_server: interleaved decode for {} sequences", n_requests);

            // Access raw llama context
            let ctx = engine.target_ctx();
            let vocab = engine.target_vocab();

            // Clear ALL KV cache
            let mem = unsafe { llama_get_memory(ctx) };
            unsafe { llama_memory_clear(mem, false) };

            // Prefill each sequence separately (different prompt lengths)
            let mut active: Vec<ActiveSeq> = Vec::with_capacity(n_requests);

            for (i, req) in incoming.into_iter().enumerate() {
                let seq_id = i as i32;
                let t0 = Instant::now();

                // Tokenize
                let prompt_tokens = match tokenize(vocab, &req.prompt) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = req.response_tx.send(Err(e));
                        continue;
                    }
                };

                let n_prompt = prompt_tokens.len();
                if n_prompt == 0 {
                    let _ = req.response_tx.send(Err("empty prompt".into()));
                    continue;
                }

                // Prefill this sequence
                let batch = unsafe { llama_batch_init(n_prompt as i32, 0, 1) };
                unsafe {
                    for j in 0..n_prompt {
                        let b = &batch as *const llama_batch as *mut llama_batch;
                        *(*b).token.add(j) = prompt_tokens[j];
                        *(*b).pos.add(j) = j as i32;
                        *(*b).n_seq_id.add(j) = 1;
                        *(*(*b).seq_id.add(j)).add(0) = seq_id;
                        *(*b).logits.add(j) = if j == n_prompt - 1 { 1 } else { 0 };
                    }
                    let b = &batch as *const llama_batch as *mut llama_batch;
                    (*b).n_tokens = n_prompt as i32;
                }

                let ret = unsafe { llama_decode(ctx, batch) };
                unsafe { llama_batch_free(batch) };

                if ret != 0 {
                    log::warn!("prefill failed for seq {}", seq_id);
                    let _ = req.response_tx.send(Err(format!("prefill failed: {ret}")));
                    continue;
                }

                let prefill_ms = t0.elapsed().as_millis();

                // Sample first token
                let sampler = build_sampler(&req.params);
                let first_tok = unsafe { llama_sampler_sample(sampler, ctx, -1) };

                let is_eog = unsafe { llama_vocab_is_eog(vocab, first_tok) };

                active.push(ActiveSeq {
                    seq_id,
                    next_token: first_tok,
                    n_past: n_prompt as i32,
                    n_decoded: 0,
                    max_tokens: req.params.max_tokens,
                    text: String::new(),
                    done: is_eog,
                    t0,
                    n_prompt,
                    prefill_ms,
                    response_tx: Some(req.response_tx),
                    sampler,
                });
            }

            // ── Interleaved decode loop ─────────────────────────────────
            let decode_t0 = Instant::now();

            loop {
                // Collect tokens from all active (not done) sequences
                let active_count = active.iter().filter(|s| !s.done).count();
                if active_count == 0 { break; }

                // Build merged batch: one token per active sequence
                let batch = unsafe { llama_batch_init(active_count as i32, 0, 1) };
                let mut batch_idx = 0;

                for seq in active.iter_mut() {
                    if seq.done { continue; }

                    // Append token text
                    let piece = token_to_str(vocab, seq.next_token);
                    seq.text.push_str(&piece);
                    seq.n_decoded += 1;

                    // Add to batch
                    unsafe {
                        let b = &batch as *const llama_batch as *mut llama_batch;
                        *(*b).token.add(batch_idx) = seq.next_token;
                        *(*b).pos.add(batch_idx) = seq.n_past;
                        *(*b).n_seq_id.add(batch_idx) = 1;
                        *(*(*b).seq_id.add(batch_idx)).add(0) = seq.seq_id;
                        *(*b).logits.add(batch_idx) = 1;
                    }

                    seq.n_past += 1;
                    batch_idx += 1;
                }

                unsafe {
                    let b = &batch as *const llama_batch as *mut llama_batch;
                    (*b).n_tokens = batch_idx as i32;
                }

                // Single decode call for ALL sequences
                let ret = unsafe { llama_decode(ctx, batch) };
                unsafe { llama_batch_free(batch) };

                if ret != 0 {
                    log::warn!("interleaved decode failed: {ret}");
                    break;
                }

                // Sample next token for each active sequence
                let mut sample_idx = 0;
                for seq in active.iter_mut() {
                    if seq.done { continue; }

                    let tok = unsafe { llama_sampler_sample(seq.sampler, ctx, sample_idx) };
                    sample_idx += 1;

                    if unsafe { llama_vocab_is_eog(vocab, tok) } || seq.n_decoded >= seq.max_tokens {
                        seq.done = true;
                    } else {
                        seq.next_token = tok;
                    }
                }
            }

            let total_decode_ms = decode_t0.elapsed().as_millis();
            let total_tokens: usize = active.iter().map(|s| s.n_decoded).sum();

            log::info!("batch_server: {} sequences, {} total tokens in {}ms ({:.1} tok/s aggregate)",
                n_requests, total_tokens, total_decode_ms,
                total_tokens as f64 / (total_decode_ms.max(1) as f64 / 1000.0));

            // Send results and cleanup
            for mut seq in active {
                unsafe { llama_sampler_free(seq.sampler) };

                let eval_ms = seq.t0.elapsed().as_millis() - seq.prefill_ms;
                let result = Ok(GenerateResult {
                    text: seq.text,
                    prompt_tokens: seq.n_prompt,
                    generated_tokens: seq.n_decoded,
                    prompt_eval_ms: seq.prefill_ms,
                    eval_ms,
                });

                if let Some(tx) = seq.response_tx.take() {
                    let _ = tx.send(result);
                }
            }

            // Clear KV cache for next batch
            let mem = unsafe { llama_get_memory(ctx) };
            unsafe { llama_memory_clear(mem, false) };
        }

        log::info!("batch_server: shut down");
    });

    BatchHandle { tx }
}

// ── Helpers (standalone, don't need &self) ───────────────────────────────

fn tokenize(vocab: *const llama_vocab, text: &str) -> Result<Vec<llama_token>, String> {
    let c_text = CString::new(text).map_err(|e| format!("invalid text: {e}"))?;
    let n = unsafe {
        llama_tokenize(vocab, c_text.as_ptr(), text.len() as i32,
            std::ptr::null_mut(), 0, true, true)
    };
    let n = (-n) as usize;
    let mut tokens = vec![0 as llama_token; n];
    let actual = unsafe {
        llama_tokenize(vocab, c_text.as_ptr(), text.len() as i32,
            tokens.as_mut_ptr(), tokens.len() as i32, true, true)
    };
    if actual < 0 { return Err(format!("tokenize failed: {actual}")); }
    tokens.truncate(actual as usize);
    Ok(tokens)
}

fn token_to_str(vocab: *const llama_vocab, token: llama_token) -> String {
    let mut buf = [0i8; 128];
    let n = unsafe {
        llama_token_to_piece(vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
    };
    if n > 0 {
        unsafe {
            String::from_utf8_lossy(std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize))
                .into_owned()
        }
    } else {
        String::new()
    }
}

fn build_sampler(params: &GenerateParams) -> *mut llama_sampler {
    unsafe {
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if params.temperature < 0.001 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(0));
        }
        chain
    }
}
