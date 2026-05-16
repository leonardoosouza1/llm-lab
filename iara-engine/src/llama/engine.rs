use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use llama_sys::*;

// ── Initialization guard ─────────────────────────────────────────────────────

static BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

fn ensure_backend_init() {
    if !BACKEND_INITIALIZED.swap(true, Ordering::SeqCst) {
        unsafe { llama_backend_init() };
        log::info!("llama.cpp backend initialized");
    }
}

// ── Types ────────────────────────────────────────────────────────────────────

pub struct GenerateParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 40, top_p: 0.9, max_tokens: 100 }
    }
}

pub struct GenerateResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_eval_ms: u128,
    pub eval_ms: u128,
}

pub struct ModelInfo {
    pub n_params: u64,
    pub n_layers: i32,
    pub n_embd: i32,
    pub vocab_size: i32,
    pub model_size: u64,
}

// ── Internal context wrapper ─────────────────────────────────────────────────

struct LlamaCtx {
    model: *mut llama_model,
    ctx: *mut llama_context,
    vocab: *const llama_vocab,
}

impl Drop for LlamaCtx {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() { llama_free(self.ctx); }
            if !self.model.is_null() { llama_model_free(self.model); }
        }
    }
}

impl LlamaCtx {
    fn load(path: &str, n_ctx: u32) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|e| format!("invalid path: {e}"))?;
        let mut model_params = unsafe { llama_model_default_params() };
        model_params.n_gpu_layers = 99;

        let model = unsafe { llama_model_load_from_file(c_path.as_ptr(), model_params) };
        if model.is_null() {
            return Err(format!("failed to load: {path}"));
        }

        let mut ctx_params = unsafe { llama_context_default_params() };
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = 512;
        ctx_params.n_ubatch = 512;
        ctx_params.n_seq_max = 4; // Support up to 4 concurrent sequences for interleaved batching
        ctx_params.flash_attn_type = llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED;

        let ctx = unsafe { llama_init_from_model(model, ctx_params) };
        if ctx.is_null() {
            unsafe { llama_model_free(model) };
            return Err("failed to create context".into());
        }

        let vocab = unsafe { llama_model_get_vocab(model) };
        Ok(Self { model, ctx, vocab })
    }

    fn clear_kv(&self) {
        let mem = unsafe { llama_get_memory(self.ctx) };
        unsafe { llama_memory_clear(mem, false) };
    }
}

// ── LlamaEngine ──────────────────────────────────────────────────────────────

pub struct LlamaEngine {
    target: LlamaCtx,
    draft: Option<LlamaCtx>,
    n_ctx: u32,
    draft_k: usize,
    cfc_draft: super::cfc_draft::CfcDraft,
}

unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    /// Load target model.
    pub fn load(gguf_path: &str, n_ctx: u32) -> Result<Self, String> {
        ensure_backend_init();

        let t0 = Instant::now();
        log::info!("LlamaEngine: loading target from {gguf_path}");
        let target = LlamaCtx::load(gguf_path, n_ctx)?;
        log::info!("LlamaEngine: target loaded in {}ms ({}M params, {} layers)",
            t0.elapsed().as_millis(),
            unsafe { llama_model_n_params(target.model) } / 1_000_000,
            unsafe { llama_model_n_layer(target.model) });

        let vocab_size = unsafe { llama_vocab_n_tokens(target.vocab) } as usize;
        let cfc_draft = super::cfc_draft::CfcDraft::new(super::cfc_draft::CfcDraftConfig {
            vocab_size,
            ..Default::default()
        });

        Ok(Self { target, draft: None, n_ctx, draft_k: 2, cfc_draft })
    }

    /// Load draft model for speculative decoding.
    pub fn load_draft(&mut self, gguf_path: &str) -> Result<(), String> {
        let t0 = Instant::now();
        log::info!("LlamaEngine: loading draft from {gguf_path}");
        let draft = LlamaCtx::load(gguf_path, self.n_ctx)?;
        log::info!("LlamaEngine: draft loaded in {}ms ({}M params)",
            t0.elapsed().as_millis(),
            unsafe { llama_model_n_params(draft.model) } / 1_000_000);
        self.draft = Some(draft);
        Ok(())
    }

    pub fn has_draft(&self) -> bool { self.draft.is_some() }

    /// Generate text — dispatches to speculative or standard path.
    ///
    /// Speculative decoding only helps when draft/target speed ratio > 3x.
    /// For small models (1.5B target + 0.5B draft = 1.6x ratio), standard is faster.
    /// Speculative becomes valuable for 7B+ targets with 0.5B-1.5B drafts.
    pub fn generate(&mut self, prompt: &str, params: &GenerateParams) -> Result<GenerateResult, String> {
        // Speculative if draft + max_tokens > 4. Gate by env IARA_SPEC_FORCE=1 to
        // bypass the 3B-params heuristic (lets us benchmark small-model spec too).
        let force = std::env::var("IARA_SPEC_FORCE").map(|v| v == "1" || v == "true").unwrap_or(false);
        let use_spec = self.draft.is_some()
            && params.max_tokens > 4
            && (force || unsafe { llama_model_n_params(self.target.model) } > 3_000_000_000);

        if use_spec {
            self.generate_speculative(prompt, params)
        } else {
            self.generate_standard(prompt, params)
        }
    }

    /// Standard autoregressive generation (no speculation).
    fn generate_standard(&mut self, prompt: &str, params: &GenerateParams) -> Result<GenerateResult, String> {
        let prompt_tokens = self.tokenize(prompt, true)?;
        let n_prompt = prompt_tokens.len();
        if n_prompt == 0 { return Err("empty prompt".into()); }

        self.target.clear_kv();
        self.cfc_draft.reset_state();
        let sampler = self.build_sampler(params);

        // Feed prompt tokens to CfC for context
        for &tok in &prompt_tokens {
            self.cfc_draft.observe(tok as u32);
        }

        // Prefill
        let prefill_t0 = Instant::now();
        let batch = unsafe { llama_batch_get_one(prompt_tokens.as_ptr() as *mut _, n_prompt as i32) };
        if unsafe { llama_decode(self.target.ctx, batch) } != 0 {
            unsafe { llama_sampler_free(sampler) };
            return Err("prefill failed".into());
        }
        let prefill_ms = prefill_t0.elapsed().as_millis();

        // Decode loop
        let decode_t0 = Instant::now();
        let mut text = String::new();
        let mut n_decoded = 0;

        for _ in 0..params.max_tokens {
            let tok = unsafe { llama_sampler_sample(sampler, self.target.ctx, -1) };
            if unsafe { llama_vocab_is_eog(self.target.vocab, tok) } { break; }

            // CfC online learning: observe the token and train
            self.cfc_draft.observe(tok as u32);
            self.cfc_draft.train_step(tok as u32);

            text.push_str(&self.token_to_str(tok));
            n_decoded += 1;

            let mut t = tok;
            let batch = unsafe { llama_batch_get_one(&mut t, 1) };
            if unsafe { llama_decode(self.target.ctx, batch) } != 0 { break; }
        }

        unsafe { llama_sampler_free(sampler) };

        // Log CfC state
        let (_, total, _) = self.cfc_draft.accuracy();
        if total > 10 {
            log::info!("CfC: {} samples trained, h_norm={:.2}",
                total, self.cfc_draft.h_norm());
        }

        Ok(GenerateResult {
            text, prompt_tokens: n_prompt, generated_tokens: n_decoded,
            prompt_eval_ms: prefill_ms, eval_ms: decode_t0.elapsed().as_millis(),
        })
    }

    /// Speculative decoding with BATCH verification.
    ///
    /// Following the llama.cpp speculative-simple pattern:
    /// 1. Keep id_last separate (last token from previous round)
    /// 2. Draft generates k tokens autoregressively
    /// 3. Build target batch: [id_last, draft0, ..., draftN-1] with logits=true for ALL
    /// 4. Decode batch ONCE on target
    /// 5. Sample from each position to verify drafts
    /// 6. On rejection: trim KV cache after rejection point
    fn generate_speculative(&mut self, prompt: &str, params: &GenerateParams) -> Result<GenerateResult, String> {
        let draft = self.draft.as_ref().ok_or("no draft model")?;
        let prompt_tokens = self.tokenize(prompt, true)?;
        let n_prompt = prompt_tokens.len();
        if n_prompt < 2 { return Err("prompt too short".into()); }

        self.target.clear_kv();
        draft.clear_kv();

        let target_sampler = self.build_sampler(params);
        let draft_sampler = self.build_greedy_sampler();

        // ── Prefill: decode all tokens EXCEPT the last one ──────────────
        // Following llama.cpp pattern: keep id_last separate
        let prefill_t0 = Instant::now();
        let n_prefill = n_prompt - 1;

        // Target prefill (all but last token)
        let batch = unsafe { llama_batch_get_one(prompt_tokens.as_ptr() as *mut _, n_prefill as i32) };
        if unsafe { llama_decode(self.target.ctx, batch) } != 0 {
            unsafe { llama_sampler_free(target_sampler); llama_sampler_free(draft_sampler) };
            return Err("target prefill failed".into());
        }

        // Draft prefill (all but last token)
        let batch_d = unsafe { llama_batch_get_one(prompt_tokens.as_ptr() as *mut _, n_prefill as i32) };
        if unsafe { llama_decode(draft.ctx, batch_d) } != 0 {
            unsafe { llama_sampler_free(target_sampler); llama_sampler_free(draft_sampler) };
            return Err("draft prefill failed".into());
        }

        // Also feed last prompt token to draft so it can start drafting
        let mut last_tok = prompt_tokens[n_prompt - 1];
        let b_draft_last = unsafe { llama_batch_get_one(&mut last_tok, 1) };
        if unsafe { llama_decode(draft.ctx, b_draft_last) } != 0 {
            unsafe { llama_sampler_free(target_sampler); llama_sampler_free(draft_sampler) };
            return Err("draft last-token prefill failed".into());
        }

        let prefill_ms = prefill_t0.elapsed().as_millis();

        let decode_t0 = Instant::now();
        let mut text = String::new();
        let mut n_decoded = 0usize;
        let mut n_accepted = 0usize;
        let mut n_drafted = 0usize;
        let k = self.draft_k;
        let mut has_eos = false;

        // id_last: the last token not yet evaluated by target (starts as last prompt token)
        let mut id_last = prompt_tokens[n_prompt - 1];
        // n_past tracks how many positions are in the target KV cache
        let mut n_past = n_prefill as i32; // target has evaluated 0..n_prefill-1

        // Allocate reusable batch for target verification
        let mut batch_tgt = unsafe { llama_batch_init((k + 1) as i32, 0, 1) };

        while n_decoded < params.max_tokens && !has_eos {
            // ── Step 1: Draft generates k tokens greedily ───────────────
            let mut draft_tokens: Vec<llama_token> = Vec::with_capacity(k);
            for _ in 0..k {
                let tok = unsafe { llama_sampler_sample(draft_sampler, draft.ctx, -1) };
                if unsafe { llama_vocab_is_eog(draft.vocab, tok) } { break; }
                draft_tokens.push(tok);

                let mut t = tok;
                let b = unsafe { llama_batch_get_one(&mut t, 1) };
                if unsafe { llama_decode(draft.ctx, b) } != 0 { break; }
            }

            n_drafted += draft_tokens.len();

            // ── Step 2: Build target batch [id_last, draft0, ..., draftN-1] ─
            // All positions get logits=true so we can sample from each
            unsafe {
                batch_tgt.n_tokens = 0;
            }

            // Helper: add a token to the batch
            let batch_size = 1 + draft_tokens.len();
            unsafe {
                // Position 0: id_last
                let idx = 0;
                *batch_tgt.token.add(idx) = id_last;
                *batch_tgt.pos.add(idx) = n_past;
                *batch_tgt.n_seq_id.add(idx) = 1;
                *(*batch_tgt.seq_id.add(idx)) = 0;
                *batch_tgt.logits.add(idx) = 1; // true

                // Positions 1..k: draft tokens
                for (i, &draft_tok) in draft_tokens.iter().enumerate() {
                    let idx = i + 1;
                    *batch_tgt.token.add(idx) = draft_tok;
                    *batch_tgt.pos.add(idx) = n_past + 1 + i as i32;
                    *batch_tgt.n_seq_id.add(idx) = 1;
                    *(*batch_tgt.seq_id.add(idx)) = 0;
                    *batch_tgt.logits.add(idx) = 1; // true
                }

                batch_tgt.n_tokens = batch_size as i32;
            }

            // ── Step 3: Decode target batch in ONE call ─────────────────
            if unsafe { llama_decode(self.target.ctx, batch_tgt) } != 0 {
                log::error!("target batch decode failed");
                break;
            }

            // ── Step 4: Sample from each position and verify ────────────
            // Position 0 in batch = logits after seeing id_last
            //   -> sample here tells us what target wants NEXT (should match draft[0])
            // Position i in batch = logits after seeing draft[i-1]
            //   -> sample here should match draft[i]

            let mut n_accept_this_round = 0usize;

            for i in 0..draft_tokens.len() {
                let target_tok = unsafe {
                    llama_sampler_sample(target_sampler, self.target.ctx, i as i32)
                };

                if target_tok != draft_tokens[i] || unsafe { llama_vocab_is_eog(self.target.vocab, target_tok) } {
                    // Rejection at position i
                    // Use target's token instead (if not EOS)
                    if !unsafe { llama_vocab_is_eog(self.target.vocab, target_tok) } {
                        text.push_str(&self.token_to_str(target_tok));
                        n_decoded += 1;
                        id_last = target_tok;
                    } else {
                        has_eos = true;
                    }
                    break;
                }

                // Draft token accepted
                text.push_str(&self.token_to_str(draft_tokens[i]));
                n_decoded += 1;
                n_accepted += 1;
                n_accept_this_round += 1;
                id_last = draft_tokens[i];
            }

            if !has_eos && n_accept_this_round == draft_tokens.len() {
                // All drafts accepted — get bonus token from last batch position
                let bonus_idx = draft_tokens.len() as i32; // last position in batch
                let bonus = unsafe {
                    llama_sampler_sample(target_sampler, self.target.ctx, bonus_idx)
                };
                if !unsafe { llama_vocab_is_eog(self.target.vocab, bonus) } {
                    text.push_str(&self.token_to_str(bonus));
                    n_decoded += 1;
                    id_last = bonus;
                } else {
                    has_eos = true;
                }
            }

            // If no drafts were generated and no accept happened, do standard sample
            if draft_tokens.is_empty() && !has_eos {
                let tok = unsafe {
                    llama_sampler_sample(target_sampler, self.target.ctx, 0)
                };
                if unsafe { llama_vocab_is_eog(self.target.vocab, tok) } {
                    has_eos = true;
                } else {
                    text.push_str(&self.token_to_str(tok));
                    n_decoded += 1;
                    id_last = tok;
                }
            }

            // ── Step 5: Update n_past and trim KV caches ────────────────
            // n_past advances by: 1 (for id_last we fed) + n_accept_this_round
            // (or +1 for bonus if all accepted)
            let tokens_accepted = if !has_eos && n_accept_this_round == draft_tokens.len() && !draft_tokens.is_empty() {
                n_accept_this_round + 1 // accepted drafts + bonus
            } else if has_eos && n_accept_this_round == draft_tokens.len() && !draft_tokens.is_empty() {
                n_accept_this_round // all accepted but EOS bonus, no extra position needed
            } else if draft_tokens.is_empty() {
                if has_eos { 0 } else { 1 } // just the token from standard sample
            } else {
                n_accept_this_round + 1 // accepted drafts + the replacement token
            };

            n_past += tokens_accepted as i32;

            // Trim target KV cache: remove everything from n_past onwards
            let tgt_mem = unsafe { llama_get_memory(self.target.ctx) };
            unsafe { llama_memory_seq_rm(tgt_mem, 0, n_past, -1) };

            // ── Step 6: Sync draft model's KV cache ─────────────────────
            // Draft may have extra tokens in its KV from the drafting phase.
            // Trim to match target's n_past, then feed id_last so draft is
            // ready to generate next round's drafts.
            if !has_eos {
                let draft_mem = unsafe { llama_get_memory(draft.ctx) };
                // Draft KV has: n_prefill positions from prefill + 1 (last prompt tok) + k draft tokens
                // We need it to have exactly n_past positions
                unsafe { llama_memory_seq_rm(draft_mem, 0, n_past, -1) };

                // Feed id_last to draft so it's ready to generate next drafts
                let mut t = id_last;
                let b = unsafe { llama_batch_get_one(&mut t, 1) };
                if unsafe { llama_decode(draft.ctx, b) } != 0 {
                    log::error!("draft sync decode failed");
                    break;
                }
            }
        }

        unsafe { llama_batch_free(batch_tgt) };

        let eval_ms = decode_t0.elapsed().as_millis();
        let accept_rate = if n_drafted > 0 { n_accepted as f32 / n_drafted as f32 } else { 0.0 };
        log::info!("speculative: {} tokens, {} drafted, {} accepted ({:.0}%), k={}",
            n_decoded, n_drafted, n_accepted, accept_rate * 100.0, k);

        unsafe { llama_sampler_free(target_sampler); llama_sampler_free(draft_sampler) };

        Ok(GenerateResult {
            text, prompt_tokens: n_prompt, generated_tokens: n_decoded,
            prompt_eval_ms: prefill_ms, eval_ms,
        })
    }

    pub fn model_info(&self) -> ModelInfo {
        unsafe {
            ModelInfo {
                n_params: llama_model_n_params(self.target.model),
                n_layers: llama_model_n_layer(self.target.model),
                n_embd: llama_model_n_embd(self.target.model),
                vocab_size: llama_vocab_n_tokens(self.target.vocab),
                model_size: llama_model_size(self.target.model),
            }
        }
    }

    pub fn print_perf(&self) {
        unsafe { llama_perf_context_print(self.target.ctx) };
    }

    /// Raw context pointer for interleaved batching (batch_server).
    pub fn target_ctx(&self) -> *mut llama_context { self.target.ctx }

    /// Raw vocab pointer for interleaved batching.
    pub fn target_vocab(&self) -> *const llama_vocab { self.target.vocab }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<llama_token>, String> {
        let c_text = CString::new(text).map_err(|e| format!("invalid text: {e}"))?;
        let n = unsafe {
            llama_tokenize(self.target.vocab, c_text.as_ptr(), text.len() as i32,
                std::ptr::null_mut(), 0, add_special, true)
        };
        let n = (-n) as usize;
        let mut tokens = vec![0 as llama_token; n];
        let actual = unsafe {
            llama_tokenize(self.target.vocab, c_text.as_ptr(), text.len() as i32,
                tokens.as_mut_ptr(), tokens.len() as i32, add_special, true)
        };
        if actual < 0 { return Err(format!("tokenize failed: {actual}")); }
        tokens.truncate(actual as usize);
        Ok(tokens)
    }

    fn token_to_str(&self, token: llama_token) -> String {
        let mut buf = [0i8; 128];
        let n = unsafe {
            llama_token_to_piece(self.target.vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
        };
        if n > 0 {
            unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize)).into_owned() }
        } else {
            String::new()
        }
    }

    fn build_sampler(&self, params: &GenerateParams) -> *mut llama_sampler {
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

    fn build_greedy_sampler(&self) -> *mut llama_sampler {
        unsafe {
            let chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            chain
        }
    }
}
