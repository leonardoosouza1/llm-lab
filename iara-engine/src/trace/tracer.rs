//! Tracer — holds the active level + sinks; the forward path drives it.

use std::sync::Arc;

use super::event::{Source, Stage, StageStats, TraceEvent};
use super::sink::TraceSink;

/// Trace verbosity. Levels are ordered: each includes the cheaper ones.
///
/// `Off` is the default and is zero-cost — `enabled()` short-circuits before
/// any stat computation or device download.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TraceLevel {
    /// No tracing.
    Off,
    /// Timing per kernel / section only.
    Perf,
    /// Per-layer stats: norm energy, attention, MLP activation, hot neurons.
    Layer,
    /// Byte/token level: tokenization, embed, logits → sampled token.
    Byte,
    /// Everything.
    Full,
}

impl TraceLevel {
    /// Parse from env-var string (`IARA_TRACE`): off|perf|layer|byte|full.
    pub fn from_env() -> Self {
        match std::env::var("IARA_TRACE").as_deref() {
            Ok("perf") => TraceLevel::Perf,
            Ok("layer") => TraceLevel::Layer,
            Ok("byte") => TraceLevel::Byte,
            Ok("full") => TraceLevel::Full,
            _ => TraceLevel::Off,
        }
    }

    /// Does this level capture per-layer activation stats?
    pub fn wants_layer(&self) -> bool {
        *self >= TraceLevel::Layer
    }
    /// Does this level capture byte/token-level events?
    pub fn wants_byte(&self) -> bool {
        *self >= TraceLevel::Byte
    }
    /// Does this level dump full activation tensors?
    pub fn wants_full(&self) -> bool {
        *self == TraceLevel::Full
    }
}

/// The trace driver. Cheap to clone (Arc-shared sinks).
#[derive(Clone)]
pub struct Tracer {
    level: TraceLevel,
    source: Source,
    trace_id: u64,
    sinks: Vec<Arc<dyn TraceSink>>,
}

impl Tracer {
    /// A disabled tracer — `enabled()` is always false, all emits no-op.
    pub fn off() -> Self {
        Self { level: TraceLevel::Off, source: Source::Engine, trace_id: 0, sinks: Vec::new() }
    }

    pub fn new(level: TraceLevel, source: Source, trace_id: u64) -> Self {
        Self { level, source, trace_id, sinks: Vec::new() }
    }

    pub fn with_sink(mut self, sink: Arc<dyn TraceSink>) -> Self {
        self.sinks.push(sink);
        self
    }

    pub fn level(&self) -> TraceLevel {
        self.level
    }

    /// True if any tracing is active. Guard expensive stat work behind this.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.level != TraceLevel::Off && !self.sinks.is_empty()
    }

    /// True if the given stage should be traced at the current level.
    #[inline]
    pub fn wants(&self, stage: &Stage) -> bool {
        if !self.enabled() {
            return false;
        }
        match stage {
            Stage::Layer { .. } => self.level.wants_layer(),
            Stage::Tokenize | Stage::Embed | Stage::Logits | Stage::Sample => {
                self.level.wants_byte() || matches!(self.level, TraceLevel::Perf)
            }
        }
    }

    /// Emit an event to all sinks. Caller should gate on `wants()` first.
    pub fn emit(&self, seq: u32, stage: Stage, stats: StageStats, full: Option<Vec<f32>>) {
        let mut ev = TraceEvent::new(self.source, self.trace_id, seq, stage, stats);
        ev.full = if self.level.wants_full() { full } else { None };
        for s in &self.sinks {
            s.emit(&ev);
        }
    }

    pub fn flush(&self) {
        for s in &self.sinks {
            s.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_disabled() {
        let t = Tracer::off();
        assert!(!t.enabled());
        assert!(!t.wants(&Stage::Embed));
        assert!(!t.wants(&Stage::Layer { idx: 0, sub: "q".into() }));
    }

    #[test]
    fn level_ordering() {
        assert!(TraceLevel::Layer.wants_layer());
        assert!(!TraceLevel::Perf.wants_layer());
        assert!(TraceLevel::Byte.wants_byte());
        assert!(!TraceLevel::Layer.wants_byte());
        assert!(TraceLevel::Full.wants_full());
        assert!(!TraceLevel::Byte.wants_full());
    }

    #[test]
    fn layer_level_traces_layers_not_bytes() {
        // A tracer with a dummy sink so enabled() is true.
        use crate::trace::sink::RingSink;
        let t = Tracer::new(TraceLevel::Layer, Source::Engine, 1)
            .with_sink(Arc::new(RingSink::new(4)));
        assert!(t.wants(&Stage::Layer { idx: 0, sub: "mlp".into() }));
        assert!(!t.wants(&Stage::Embed)); // byte-level, needs Byte+
    }
}
