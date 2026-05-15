//! Trace sinks — where `TraceEvent`s go.
//!
//! - `FileSink`: appends JSONL to a `.trace` file (offline replay).
//! - `RingSink`: fixed-size ring buffer + optional live broadcast (UI).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use super::event::TraceEvent;

/// A destination for trace events.
pub trait TraceSink: Send + Sync {
    fn emit(&self, ev: &TraceEvent);
    /// Flush any buffered data (called at end of generation).
    fn flush(&self) {}
}

/// Appends each event as one JSONL line to a `.trace` file.
pub struct FileSink {
    writer: Mutex<BufWriter<File>>,
}

impl FileSink {
    /// Open (truncating) a `.trace` file for writing.
    pub fn create(path: &str) -> std::io::Result<Self> {
        let f = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;
        Ok(Self { writer: Mutex::new(BufWriter::new(f)) })
    }
}

impl TraceSink for FileSink {
    fn emit(&self, ev: &TraceEvent) {
        if let Ok(mut w) = self.writer.lock() {
            let _ = writeln!(w, "{}", ev.to_jsonl());
        }
    }
    fn flush(&self) {
        if let Ok(mut w) = self.writer.lock() {
            let _ = w.flush();
        }
    }
}

/// Fixed-capacity ring buffer of recent events — for the live UI to poll/drain.
pub struct RingSink {
    buf: Mutex<Vec<TraceEvent>>,
    cap: usize,
}

impl RingSink {
    pub fn new(cap: usize) -> Self {
        Self { buf: Mutex::new(Vec::with_capacity(cap)), cap }
    }

    /// Snapshot current events (most recent last).
    pub fn snapshot(&self) -> Vec<TraceEvent> {
        self.buf.lock().map(|b| b.clone()).unwrap_or_default()
    }

    /// Drain all buffered events.
    pub fn drain(&self) -> Vec<TraceEvent> {
        self.buf.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default()
    }
}

impl TraceSink for RingSink {
    fn emit(&self, ev: &TraceEvent) {
        if let Ok(mut b) = self.buf.lock() {
            if b.len() >= self.cap {
                b.remove(0); // drop oldest
            }
            b.push(ev.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::event::{compute_stats, Source, Stage, TraceEvent};

    fn ev(seq: u32) -> TraceEvent {
        TraceEvent::new(Source::Engine, 1, seq, Stage::Embed,
            compute_stats(&[1.0, 2.0], 2, 2, 0))
    }

    #[test]
    fn ring_caps_and_evicts() {
        let r = RingSink::new(3);
        for i in 0..5 { r.emit(&ev(i)); }
        let snap = r.snapshot();
        assert_eq!(snap.len(), 3);
        // oldest (seq 0,1) evicted; keeps 2,3,4
        assert_eq!(snap[0].seq, 2);
        assert_eq!(snap[2].seq, 4);
    }

    #[test]
    fn ring_drain_empties() {
        let r = RingSink::new(8);
        r.emit(&ev(0));
        r.emit(&ev(1));
        assert_eq!(r.drain().len(), 2);
        assert_eq!(r.drain().len(), 0);
    }

    #[test]
    fn file_sink_writes_jsonl() {
        let tmp = std::env::temp_dir().join("iara_trace_test.trace");
        let path = tmp.to_str().unwrap();
        {
            let s = FileSink::create(path).unwrap();
            s.emit(&ev(0));
            s.emit(&ev(1));
            s.flush();
        }
        let content = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with('{'));
        let _ = std::fs::remove_file(path);
    }
}
