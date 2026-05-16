//! trace — first-class introspection subsystem.
//!
//! Replaces the ad-hoc `IARA_DBG_HIDDEN` / `IARA_SECTION_PROF` env-var hacks
//! with a structured, leveled, sink-routed trace pipeline.
//!
//! Design: docs/architecture/DNA-V6-TRACING.md
//!
//! - `TraceLevel`  — Off | Perf | Layer | Byte | Full. Off is zero-cost.
//! - `TraceEvent`  — one structured record per forward stage.
//! - `TraceSink`   — where events go: FileSink (.trace JSONL) or RingSink (live).
//! - `Tracer`      — holds the active level + sinks; the forward path calls it.
//!
//! Cross-project: `byte-language-lab` (brain, byte-level) and this engine share
//! the JSONL schema and a `trace_id`, so a replay tool stitches both into one
//! timeline (byte ⇄ token transition).

mod event;
mod sink;
mod tracer;

pub use event::{compute_stats, Source, Stage, StageStats, TraceEvent};
pub use sink::{FileSink, RingSink, TraceSink};
pub use tracer::{TraceLevel, Tracer};
