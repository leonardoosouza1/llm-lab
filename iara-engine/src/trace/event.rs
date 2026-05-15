//! Trace event types — the structured record emitted at each forward stage.

use std::time::{SystemTime, UNIX_EPOCH};

/// Which side of the IARA stack produced the event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// byte-language-lab — byte-level HDC brain.
    Brain,
    /// llm-lab — token-level transformer engine.
    Engine,
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Source::Brain => "brain",
            Source::Engine => "engine",
        }
    }
}

/// A pipeline stage. `Layer` carries the layer index + sub-stage name.
#[derive(Debug, Clone)]
pub enum Stage {
    /// text → bytes → BPE merges → token ids
    Tokenize,
    /// embedding lookup: token id → hidden vector
    Embed,
    /// inside a transformer layer; `sub` is e.g. "input_norm", "q", "attn_out", "mlp"
    Layer { idx: u32, sub: String },
    /// final norm + lm_head → logit vector
    Logits,
    /// logits → sampled token
    Sample,
}

impl Stage {
    pub fn label(&self) -> String {
        match self {
            Stage::Tokenize => "tokenize".to_string(),
            Stage::Embed => "embed".to_string(),
            Stage::Layer { idx, sub } => format!("layer.{}.{}", idx, sub),
            Stage::Logits => "logits".to_string(),
            Stage::Sample => "sample".to_string(),
        }
    }

    pub fn layer_idx(&self) -> Option<u32> {
        match self {
            Stage::Layer { idx, .. } => Some(*idx),
            _ => None,
        }
    }
}

/// Compact per-stage statistics — the default detail level.
#[derive(Debug, Clone, Default)]
pub struct StageStats {
    /// L2 norm of the (first row of the) tensor.
    pub l2: f32,
    /// First-N values, for eyeballing structure.
    pub head: Vec<f32>,
    /// Top-k (index, value) by magnitude — reconstructs "which neurons fired".
    pub top_k: Vec<(u32, f32)>,
    /// Wall time spent in this stage, microseconds.
    pub timing_us: u64,
    /// True if any non-finite value was seen — catches kernel corruption.
    pub nan: bool,
}

/// One structured trace record. Serialized as a single JSONL line.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Microseconds since UNIX epoch.
    pub ts_us: u64,
    pub source: Source,
    /// Shared across brain + engine for a single user query.
    pub trace_id: u64,
    /// Token index within the generation (0-based).
    pub seq: u32,
    pub stage: Stage,
    pub stats: StageStats,
    /// Full activation dump — only populated with `--trace-full`.
    pub full: Option<Vec<f32>>,
}

impl TraceEvent {
    pub fn new(source: Source, trace_id: u64, seq: u32, stage: Stage, stats: StageStats) -> Self {
        let ts_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        Self { ts_us, source, trace_id, seq, stage, stats, full: None }
    }

    /// Serialize to a single JSONL line (no trailing newline).
    pub fn to_jsonl(&self) -> String {
        let head: Vec<String> = self.stats.head.iter().map(|v| fmt_f32(*v)).collect();
        let topk: Vec<String> = self.stats.top_k.iter()
            .map(|(i, v)| format!("[{},{}]", i, fmt_f32(*v)))
            .collect();
        let full_part = match &self.full {
            Some(vals) => {
                let s: Vec<String> = vals.iter().map(|v| fmt_f32(*v)).collect();
                format!(",\"full\":[{}]", s.join(","))
            }
            None => String::new(),
        };
        format!(
            "{{\"ts\":{},\"src\":\"{}\",\"tid\":{},\"seq\":{},\"stage\":\"{}\",\
             \"l2\":{},\"timing_us\":{},\"nan\":{},\"head\":[{}],\"top_k\":[{}]{}}}",
            self.ts_us,
            self.source.as_str(),
            self.trace_id,
            self.seq,
            self.stage.label(),
            fmt_f32(self.stats.l2),
            self.stats.timing_us,
            self.stats.nan,
            head.join(","),
            topk.join(","),
            full_part,
        )
    }
}

/// Format an f32 compactly, finite-safe (JSON has no NaN/Inf literal).
fn fmt_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{:.6}", v)
    } else if v.is_nan() {
        "\"NaN\"".to_string()
    } else if v > 0.0 {
        "\"Inf\"".to_string()
    } else {
        "\"-Inf\"".to_string()
    }
}

/// Compute compact stats from a tensor slice.
pub fn compute_stats(data: &[f32], head_n: usize, top_k: usize, timing_us: u64) -> StageStats {
    let mut nan = false;
    let mut sumsq = 0.0f64;
    for &v in data {
        if !v.is_finite() {
            nan = true;
        } else {
            sumsq += (v as f64) * (v as f64);
        }
    }
    let l2 = (sumsq.sqrt()) as f32;
    let head: Vec<f32> = data.iter().take(head_n).copied().collect();

    // top-k by magnitude
    let mut idx: Vec<(u32, f32)> = data.iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    idx.sort_by(|a, b| {
        b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(top_k);

    StageStats { l2, head, top_k: idx, timing_us, nan }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_basic() {
        let s = compute_stats(&[3.0, 4.0, 0.0], 2, 2, 100);
        assert!((s.l2 - 5.0).abs() < 1e-4);
        assert_eq!(s.head, vec![3.0, 4.0]);
        assert_eq!(s.top_k[0].0, 1); // 4.0 has largest magnitude
        assert!(!s.nan);
    }

    #[test]
    fn stats_detects_nan() {
        let s = compute_stats(&[1.0, f32::NAN, 2.0], 3, 1, 0);
        assert!(s.nan);
    }

    #[test]
    fn jsonl_roundtrip_shape() {
        let stats = compute_stats(&[1.0, 2.0], 2, 2, 50);
        let ev = TraceEvent::new(Source::Engine, 42, 7,
            Stage::Layer { idx: 3, sub: "mlp".into() }, stats);
        let line = ev.to_jsonl();
        assert!(line.starts_with('{') && line.ends_with('}'));
        assert!(line.contains("\"stage\":\"layer.3.mlp\""));
        assert!(line.contains("\"tid\":42"));
        assert!(line.contains("\"src\":\"engine\""));
    }

    #[test]
    fn jsonl_nan_is_quoted() {
        let stats = compute_stats(&[f32::NAN], 1, 1, 0);
        let ev = TraceEvent::new(Source::Engine, 1, 0, Stage::Embed, stats);
        let line = ev.to_jsonl();
        // NaN must be quoted — bare NaN is invalid JSON.
        assert!(line.contains("\"NaN\""));
    }
}
