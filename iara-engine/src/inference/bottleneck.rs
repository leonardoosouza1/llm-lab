// ============================================================================
// inference/bottleneck.rs — Bottleneck Detection for IARA Inference Pipeline
// ============================================================================
//
// Lightweight profiling system that identifies GPU stalls, CPU fallbacks, and
// performance issues during inference. Designed for production use:
//
//   - Zero-cost when disabled: #[inline(always)] + early atomic check
//   - Lock-free counters via AtomicU64 (Relaxed ordering — no barrier needed)
//   - Fixed-size ring buffer for events (no allocation in hot path)
//   - RAII ProfileSection guard for automatic timing
//   - profile_section! macro for ergonomic instrumentation
//
// Usage:
//   1. Enable profiling:  BOTTLENECK_PROFILER.enable()
//   2. Instrument code:   profile_section!(GpuSync, "file:line", "description")
//   3. Get report:        BOTTLENECK_PROFILER.report()
//   4. Disable:           BOTTLENECK_PROFILER.disable()
//
// Future: expose via /api/debug/bottlenecks endpoints once integrated into
// AppState (enable, disable, reset, report as JSON).
//
// ============================================================================

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

// ── Ring buffer capacity ─────────────────────────────────────────────────

/// Maximum events stored in the ring buffer. Old events are overwritten.
const RING_BUFFER_CAPACITY: usize = 10_000;

// ── Bottleneck categories ────────────────────────────────────────────────

/// Categories of bottlenecks detected during inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BottleneckKind {
    /// GPU-to-CPU synchronization (download_f32, to_host, etc.)
    GpuSync,
    /// Too many small kernel launches (dispatch overhead dominates)
    KernelLaunch,
    /// Operation running on CPU instead of GPU (fallback path)
    CpuFallback,
    /// Large data transfer between host and device memory
    MemoryTransfer,
    /// Mutex/lock wait time in hot path
    LockContention,
    /// Memory allocation in hot path (Vec growth, new tensors)
    Allocation,
}

impl BottleneckKind {
    /// Human-readable label for report columns.
    pub fn label(&self) -> &'static str {
        match self {
            BottleneckKind::GpuSync => "GPU Sync",
            BottleneckKind::KernelLaunch => "Kernel Launch",
            BottleneckKind::CpuFallback => "CPU Fallback",
            BottleneckKind::MemoryTransfer => "Mem Transfer",
            BottleneckKind::LockContention => "Lock Contention",
            BottleneckKind::Allocation => "Allocation",
        }
    }

    /// All variants in display order.
    pub fn all() -> &'static [BottleneckKind] {
        &[
            BottleneckKind::GpuSync,
            BottleneckKind::KernelLaunch,
            BottleneckKind::CpuFallback,
            BottleneckKind::MemoryTransfer,
            BottleneckKind::LockContention,
            BottleneckKind::Allocation,
        ]
    }
}

impl fmt::Display for BottleneckKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ── Event and stat structs ───────────────────────────────────────────────

/// A single bottleneck event recorded during inference.
#[derive(Debug, Clone)]
pub struct BottleneckEvent {
    pub kind: BottleneckKind,
    /// Source location, e.g. "model/mod.rs:2133"
    pub location: &'static str,
    /// What was happening, e.g. "phase_coherence download_f32"
    pub description: &'static str,
    /// Duration of the bottleneck in microseconds
    pub duration_us: u64,
    /// Layer index if within a transformer block
    pub layer_idx: Option<usize>,
    /// Which token step this occurred on
    pub token_step: u64,
}

/// Aggregated statistics for a specific bottleneck location.
#[derive(Debug, Clone)]
pub struct BottleneckStat {
    pub kind: BottleneckKind,
    pub location: &'static str,
    pub description: &'static str,
    pub count: u64,
    pub total_us: u64,
    pub max_us: u64,
    pub avg_us: f64,
    /// Percentage of total forward time consumed by this bottleneck
    pub pct_of_total: f64,
}

// ── Per-kind atomic counters (lock-free) ─────────────────────────────────

/// Lock-free counters for a single BottleneckKind.
struct KindCounters {
    count: AtomicU64,
    total_us: AtomicU64,
    max_us: AtomicU64,
}

impl KindCounters {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_us: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    fn record(&self, duration_us: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(duration_us, Ordering::Relaxed);
        // Update max via CAS loop (lock-free)
        let mut current = self.max_us.load(Ordering::Relaxed);
        while duration_us > current {
            match self.max_us.compare_exchange_weak(
                current,
                duration_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_us.store(0, Ordering::Relaxed);
        self.max_us.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.count.load(Ordering::Relaxed),
            self.total_us.load(Ordering::Relaxed),
            self.max_us.load(Ordering::Relaxed),
        )
    }
}

// ── Ring buffer ──────────────────────────────────────────────────────────

/// Fixed-size ring buffer that overwrites oldest events. No allocation after
/// initial construction (Vec is pre-allocated to capacity).
struct RingBuffer {
    events: Vec<BottleneckEvent>,
    write_pos: usize,
    len: usize,
}

impl RingBuffer {
    fn new() -> Self {
        Self {
            events: Vec::with_capacity(RING_BUFFER_CAPACITY),
            write_pos: 0,
            len: 0,
        }
    }

    fn push(&mut self, event: BottleneckEvent) {
        if self.events.len() < RING_BUFFER_CAPACITY {
            // Still filling initial capacity — push without overwriting
            self.events.push(event);
            self.len = self.events.len();
            self.write_pos = self.len % RING_BUFFER_CAPACITY;
        } else {
            // Ring buffer full — overwrite oldest
            self.events[self.write_pos] = event;
            self.write_pos = (self.write_pos + 1) % RING_BUFFER_CAPACITY;
            self.len = RING_BUFFER_CAPACITY;
        }
    }

    /// Returns events in chronological order (oldest first).
    fn snapshot(&self) -> Vec<BottleneckEvent> {
        if self.events.len() < RING_BUFFER_CAPACITY {
            return self.events.clone();
        }
        // Ring buffer wrapped: read_pos..end then 0..read_pos
        let mut out = Vec::with_capacity(RING_BUFFER_CAPACITY);
        out.extend_from_slice(&self.events[self.write_pos..]);
        out.extend_from_slice(&self.events[..self.write_pos]);
        out
    }

    fn clear(&mut self) {
        self.events.clear();
        self.write_pos = 0;
        self.len = 0;
    }
}

// ── BottleneckProfiler ───────────────────────────────────────────────────

/// Thread-safe bottleneck profiler for the IARA inference pipeline.
///
/// All public methods are safe to call from any thread. The profiler uses
/// lock-free atomics for counters and a Mutex-protected ring buffer for
/// detailed event storage. When disabled, all recording methods short-circuit
/// on a Relaxed atomic load (effectively zero-cost).
pub struct BottleneckProfiler {
    enabled: AtomicBool,

    // Ring buffer of recent events (Mutex only taken when profiling is enabled)
    events: Mutex<RingBuffer>,

    // Per-token timing
    token_start_us: AtomicU64,
    total_forward_us: AtomicU64,
    token_count: AtomicU64,

    // Per-kind lock-free counters (indexed by BottleneckKind discriminant)
    gpu_sync: KindCounters,
    kernel_launch: KindCounters,
    cpu_fallback: KindCounters,
    memory_transfer: KindCounters,
    lock_contention: KindCounters,
    allocation: KindCounters,
}

// Safety: all fields are either Atomic or Mutex-protected.
unsafe impl Sync for BottleneckProfiler {}

impl BottleneckProfiler {
    /// Create a new profiler (disabled by default).
    pub const fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            events: Mutex::new(RingBuffer {
                events: Vec::new(),
                write_pos: 0,
                len: 0,
            }),
            token_start_us: AtomicU64::new(0),
            total_forward_us: AtomicU64::new(0),
            token_count: AtomicU64::new(0),
            gpu_sync: KindCounters::new(),
            kernel_launch: KindCounters::new(),
            cpu_fallback: KindCounters::new(),
            memory_transfer: KindCounters::new(),
            lock_contention: KindCounters::new(),
            allocation: KindCounters::new(),
        }
    }

    /// Enable profiling. Subsequent calls to record() will capture events.
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable profiling. All record() calls become no-ops.
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Check if profiling is currently enabled.
    #[inline(always)]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Mark the start of a token's forward pass.
    /// Call this before processing each token.
    #[inline(always)]
    pub fn token_start(&self) {
        if !self.is_enabled() {
            return;
        }
        let now = timestamp_us();
        self.token_start_us.store(now, Ordering::Relaxed);
    }

    /// Mark the end of a token's forward pass.
    /// Call this after processing each token.
    #[inline(always)]
    pub fn token_end(&self) {
        if !self.is_enabled() {
            return;
        }
        let start = self.token_start_us.load(Ordering::Relaxed);
        if start == 0 {
            return;
        }
        let now = timestamp_us();
        let elapsed = now.saturating_sub(start);
        self.total_forward_us.fetch_add(elapsed, Ordering::Relaxed);
        self.token_count.fetch_add(1, Ordering::Relaxed);
        self.token_start_us.store(0, Ordering::Relaxed);
    }

    /// Record a bottleneck event. This is a no-op when profiling is disabled.
    ///
    /// The #[inline(always)] ensures the disabled check is inlined at the call
    /// site, making the disabled path a single atomic load + branch.
    #[inline(always)]
    pub fn record(
        &self,
        kind: BottleneckKind,
        location: &'static str,
        description: &'static str,
        duration_us: u64,
        layer_idx: Option<usize>,
    ) {
        if !self.is_enabled() {
            return;
        }
        self.record_inner(kind, location, description, duration_us, layer_idx);
    }

    /// Inner recording path — not inlined to keep the hot path small.
    #[cold]
    fn record_inner(
        &self,
        kind: BottleneckKind,
        location: &'static str,
        description: &'static str,
        duration_us: u64,
        layer_idx: Option<usize>,
    ) {
        // Update lock-free per-kind counters
        self.counters_for(kind).record(duration_us);

        // Store event in ring buffer
        let token_step = self.token_count.load(Ordering::Relaxed);
        let event = BottleneckEvent {
            kind,
            location,
            description,
            duration_us,
            layer_idx,
            token_step,
        };

        if let Ok(mut ring) = self.events.lock() {
            ring.push(event);
        }
        // If lock is poisoned, silently drop the event (profiling should
        // never crash the inference pipeline).
    }

    /// Get the counters for a given BottleneckKind.
    fn counters_for(&self, kind: BottleneckKind) -> &KindCounters {
        match kind {
            BottleneckKind::GpuSync => &self.gpu_sync,
            BottleneckKind::KernelLaunch => &self.kernel_launch,
            BottleneckKind::CpuFallback => &self.cpu_fallback,
            BottleneckKind::MemoryTransfer => &self.memory_transfer,
            BottleneckKind::LockContention => &self.lock_contention,
            BottleneckKind::Allocation => &self.allocation,
        }
    }

    /// Get aggregated statistics per bottleneck kind.
    pub fn kind_stats(&self) -> Vec<BottleneckStat> {
        let total_fwd = self.total_forward_us.load(Ordering::Relaxed) as f64;

        BottleneckKind::all()
            .iter()
            .map(|&kind| {
                let (count, total_us, max_us) = self.counters_for(kind).snapshot();
                let avg_us = if count > 0 {
                    total_us as f64 / count as f64
                } else {
                    0.0
                };
                let pct = if total_fwd > 0.0 {
                    (total_us as f64 / total_fwd) * 100.0
                } else {
                    0.0
                };

                BottleneckStat {
                    kind,
                    location: "",
                    description: "",
                    count,
                    total_us,
                    max_us,
                    avg_us,
                    pct_of_total: pct,
                }
            })
            .collect()
    }

    /// Get aggregated statistics per (location, description) pair, sorted by
    /// total time descending. Useful for identifying the hottest call sites.
    pub fn location_stats(&self) -> Vec<BottleneckStat> {
        let events = match self.events.lock() {
            Ok(ring) => ring.snapshot(),
            Err(_) => return Vec::new(),
        };

        let total_fwd = self.total_forward_us.load(Ordering::Relaxed) as f64;

        // Aggregate by (location, description)
        let mut map: std::collections::HashMap<
            (&'static str, &'static str),
            (BottleneckKind, u64, u64, u64),
        > = std::collections::HashMap::new();

        for ev in &events {
            let entry = map
                .entry((ev.location, ev.description))
                .or_insert((ev.kind, 0, 0, 0));
            entry.1 += 1; // count
            entry.2 += ev.duration_us; // total
            if ev.duration_us > entry.3 {
                entry.3 = ev.duration_us; // max
            }
        }

        let mut stats: Vec<BottleneckStat> = map
            .into_iter()
            .map(|((loc, desc), (kind, count, total_us, max_us))| {
                let avg_us = if count > 0 {
                    total_us as f64 / count as f64
                } else {
                    0.0
                };
                let pct = if total_fwd > 0.0 {
                    (total_us as f64 / total_fwd) * 100.0
                } else {
                    0.0
                };
                BottleneckStat {
                    kind,
                    location: loc,
                    description: desc,
                    count,
                    total_us,
                    max_us,
                    avg_us,
                    pct_of_total: pct,
                }
            })
            .collect();

        stats.sort_by(|a, b| b.total_us.cmp(&a.total_us));
        stats
    }

    /// Generate a formatted report string.
    ///
    /// Produces a human-readable table showing per-kind breakdown and top
    /// bottleneck locations, suitable for logging or debug endpoints.
    pub fn report(&self) -> String {
        let token_count = self.token_count.load(Ordering::Relaxed);
        let total_fwd_us = self.total_forward_us.load(Ordering::Relaxed);
        let total_fwd_ms = total_fwd_us as f64 / 1000.0;
        let ms_per_token = if token_count > 0 {
            total_fwd_ms / token_count as f64
        } else {
            0.0
        };

        let kind_stats = self.kind_stats();
        let location_stats = self.location_stats();

        let mut out = String::with_capacity(2048);

        // Header
        let bar = "\u{2550}".repeat(62);
        out.push_str(&format!("\n  {bar}\n"));
        out.push_str(&format!(
            "    IARA Bottleneck Report \u{2014} {} tokens analyzed\n",
            token_count
        ));
        out.push_str(&format!("  {bar}\n"));
        out.push_str(&format!(
            "    Total forward time:  {:.1}ms ({:.1}ms/token)\n",
            total_fwd_ms, ms_per_token
        ));
        out.push('\n');

        // Table header
        out.push_str("    \u{250c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{252c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{252c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{252c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{252c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}\n");
        out.push_str("    \u{2502} Bottleneck       \u{2502} Count  \u{2502} Total    \u{2502} Avg    \u{2502} % Time  \u{2502}\n");
        out.push_str("    \u{251c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}\n");

        // Per-kind rows
        let mut accounted_us: u64 = 0;
        for stat in &kind_stats {
            accounted_us += stat.total_us;
            let total_ms = stat.total_us as f64 / 1000.0;
            let avg_ms = stat.avg_us / 1000.0;
            out.push_str(&format!(
                "    \u{2502} {:<16} \u{2502} {:>6} \u{2502} {:>7.0}ms \u{2502} {:>5.1}ms \u{2502} {:>6.1}%  \u{2502}\n",
                stat.kind.label(),
                format_count(stat.count),
                total_ms,
                avg_ms,
                stat.pct_of_total,
            ));
        }

        // Pure compute row (time not attributed to any bottleneck)
        let pure_us = total_fwd_us.saturating_sub(accounted_us);
        let pure_ms = pure_us as f64 / 1000.0;
        let pure_pct = if total_fwd_us > 0 {
            (pure_us as f64 / total_fwd_us as f64) * 100.0
        } else {
            0.0
        };
        out.push_str(&format!(
            "    \u{2502} {:<16} \u{2502} {:>6} \u{2502} {:>7.0}ms \u{2502} {:>6} \u{2502} {:>6.1}%  \u{2502}\n",
            "Pure Compute", "\u{2014}", pure_ms, "\u{2014}", pure_pct,
        ));

        // Table footer
        out.push_str("    \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2534}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2534}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2534}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2534}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}\n");

        // Top bottleneck locations
        if !location_stats.is_empty() {
            out.push('\n');
            out.push_str("    Top bottleneck locations:\n");
            let top_n = location_stats.len().min(10);
            let tok = if token_count > 0 { token_count } else { 1 };
            for (i, stat) in location_stats[..top_n].iter().enumerate() {
                let per_tok = stat.count as f64 / tok as f64;
                let avg_ms = stat.avg_us / 1000.0;
                out.push_str(&format!(
                    "    {:>2}. {:<22} {:<20} {:>4.0}\u{00d7}/tok  {:.1}ms avg  [{}]\n",
                    i + 1,
                    stat.location,
                    stat.description,
                    per_tok,
                    avg_ms,
                    stat.kind.label(),
                ));
            }
        }

        out.push_str(&format!("  {bar}\n"));
        out
    }

    /// Reset all counters and clear the event buffer.
    pub fn reset(&self) {
        self.token_start_us.store(0, Ordering::Relaxed);
        self.total_forward_us.store(0, Ordering::Relaxed);
        self.token_count.store(0, Ordering::Relaxed);

        self.gpu_sync.reset();
        self.kernel_launch.reset();
        self.cpu_fallback.reset();
        self.memory_transfer.reset();
        self.lock_contention.reset();
        self.allocation.reset();

        if let Ok(mut ring) = self.events.lock() {
            ring.clear();
        }
    }
}

// ── Global profiler instance ─────────────────────────────────────────────

/// Global bottleneck profiler. Use this directly or via the profile_section! macro.
///
/// Disabled by default. Call `BOTTLENECK_PROFILER.enable()` to start recording.
pub static BOTTLENECK_PROFILER: BottleneckProfiler = BottleneckProfiler::new();

// ── RAII profile guard ───────────────────────────────────────────────────

/// RAII guard that times a code section and records it as a bottleneck event
/// when dropped. Created by the `profile_section!` macro.
///
/// If profiling is disabled at construction time, `start` is set to None and
/// the Drop impl is effectively a no-op (no timing overhead).
pub struct ProfileSection {
    start: Option<Instant>,
    kind: BottleneckKind,
    location: &'static str,
    description: &'static str,
    layer_idx: Option<usize>,
    profiler: &'static BottleneckProfiler,
}

impl ProfileSection {
    /// Create a new profile section. Returns immediately if profiling is disabled.
    #[inline(always)]
    pub fn new(
        profiler: &'static BottleneckProfiler,
        kind: BottleneckKind,
        location: &'static str,
        description: &'static str,
        layer_idx: Option<usize>,
    ) -> Self {
        let start = if profiler.is_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        Self {
            start,
            kind,
            location,
            description,
            layer_idx,
            profiler,
        }
    }
}

impl Drop for ProfileSection {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed_us = start.elapsed().as_micros() as u64;
            self.profiler
                .record(self.kind, self.location, self.description, elapsed_us, self.layer_idx);
        }
    }
}

// ── Convenience macros ───────────────────────────────────────────────────

/// Profile a code section using RAII. The section is timed from macro invocation
/// until the returned guard is dropped (end of scope).
///
/// # Examples
///
/// ```ignore
/// // Basic usage (uses global BOTTLENECK_PROFILER):
/// let _guard = profile_section!(GpuSync, "forward.rs:123", "attention download");
///
/// // With layer index:
/// let _guard = profile_section!(GpuSync, "forward.rs:123", "qkv download", layer = 5);
///
/// // With explicit profiler:
/// let _guard = profile_section!(my_profiler, GpuSync, "forward.rs:123", "qkv download");
/// ```
#[macro_export]
macro_rules! profile_section {
    // Global profiler, no layer
    ($kind:ident, $loc:literal, $desc:literal) => {
        $crate::inference::bottleneck::ProfileSection::new(
            &$crate::inference::bottleneck::BOTTLENECK_PROFILER,
            $crate::inference::bottleneck::BottleneckKind::$kind,
            $loc,
            $desc,
            None,
        )
    };
    // Global profiler, with layer
    ($kind:ident, $loc:literal, $desc:literal, layer = $layer:expr) => {
        $crate::inference::bottleneck::ProfileSection::new(
            &$crate::inference::bottleneck::BOTTLENECK_PROFILER,
            $crate::inference::bottleneck::BottleneckKind::$kind,
            $loc,
            $desc,
            Some($layer),
        )
    };
    // Explicit profiler, no layer
    ($profiler:expr, $kind:ident, $loc:literal, $desc:literal) => {
        $crate::inference::bottleneck::ProfileSection::new(
            $profiler,
            $crate::inference::bottleneck::BottleneckKind::$kind,
            $loc,
            $desc,
            None,
        )
    };
    // Explicit profiler, with layer
    ($profiler:expr, $kind:ident, $loc:literal, $desc:literal, layer = $layer:expr) => {
        $crate::inference::bottleneck::ProfileSection::new(
            $profiler,
            $crate::inference::bottleneck::BottleneckKind::$kind,
            $loc,
            $desc,
            Some($layer),
        )
    };
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Get current timestamp in microseconds (monotonic, relative to process start).
fn timestamp_us() -> u64 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_micros() as u64
}

/// Format a count with thousands separators for display.
fn format_count(n: u64) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_by_default() {
        let p = BottleneckProfiler::new();
        assert!(!p.is_enabled());
    }

    #[test]
    fn test_enable_disable() {
        let p = BottleneckProfiler::new();
        p.enable();
        assert!(p.is_enabled());
        p.disable();
        assert!(!p.is_enabled());
    }

    #[test]
    fn test_record_when_disabled_is_noop() {
        let p = BottleneckProfiler::new();
        p.record(BottleneckKind::GpuSync, "test:1", "noop", 100, None);
        let stats = p.kind_stats();
        let gpu = stats.iter().find(|s| s.kind == BottleneckKind::GpuSync).unwrap();
        assert_eq!(gpu.count, 0);
    }

    #[test]
    fn test_record_when_enabled() {
        let p = BottleneckProfiler::new();
        p.enable();
        p.record(BottleneckKind::GpuSync, "test:1", "download", 500, None);
        p.record(BottleneckKind::GpuSync, "test:1", "download", 300, None);
        p.record(BottleneckKind::KernelLaunch, "test:2", "dispatch", 50, Some(0));

        let stats = p.kind_stats();
        let gpu = stats.iter().find(|s| s.kind == BottleneckKind::GpuSync).unwrap();
        assert_eq!(gpu.count, 2);
        assert_eq!(gpu.total_us, 800);
        assert_eq!(gpu.max_us, 500);

        let kern = stats.iter().find(|s| s.kind == BottleneckKind::KernelLaunch).unwrap();
        assert_eq!(kern.count, 1);
        assert_eq!(kern.total_us, 50);
    }

    #[test]
    fn test_token_timing() {
        let p = BottleneckProfiler::new();
        p.enable();
        p.token_start();
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(5));
        p.token_end();

        let count = p.token_count.load(Ordering::Relaxed);
        let total = p.total_forward_us.load(Ordering::Relaxed);
        assert_eq!(count, 1);
        assert!(total >= 4000, "expected >= 4ms, got {}us", total);
    }

    #[test]
    fn test_reset() {
        let p = BottleneckProfiler::new();
        p.enable();
        p.record(BottleneckKind::GpuSync, "test:1", "dl", 100, None);
        p.token_start();
        p.token_end();
        p.reset();

        let stats = p.kind_stats();
        let gpu = stats.iter().find(|s| s.kind == BottleneckKind::GpuSync).unwrap();
        assert_eq!(gpu.count, 0);
        assert_eq!(p.token_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut ring = RingBuffer::new();
        // Fill beyond capacity
        for i in 0..(RING_BUFFER_CAPACITY + 500) {
            ring.push(BottleneckEvent {
                kind: BottleneckKind::GpuSync,
                location: "test",
                description: "overflow",
                duration_us: i as u64,
                layer_idx: None,
                token_step: 0,
            });
        }
        assert_eq!(ring.len, RING_BUFFER_CAPACITY);
        let snap = ring.snapshot();
        assert_eq!(snap.len(), RING_BUFFER_CAPACITY);
        // Oldest event should be i=500 (first 500 overwritten)
        assert_eq!(snap[0].duration_us, 500);
        // Newest should be capacity + 499
        assert_eq!(
            snap[RING_BUFFER_CAPACITY - 1].duration_us,
            (RING_BUFFER_CAPACITY + 499) as u64
        );
    }

    #[test]
    fn test_report_not_empty() {
        let p = BottleneckProfiler::new();
        p.enable();
        p.token_start();
        p.record(BottleneckKind::GpuSync, "model/mod.rs:2133", "phase_coherence", 3000, Some(5));
        p.token_end();

        let report = p.report();
        assert!(report.contains("IARA Bottleneck Report"));
        assert!(report.contains("GPU Sync"));
        assert!(report.contains("phase_coherence"));
    }

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(42), "42");
        assert_eq!(format_count(1400), "1,400");
        assert_eq!(format_count(50400), "50,400");
        assert_eq!(format_count(1000000), "1,000,000");
    }

    #[test]
    fn test_location_stats_aggregation() {
        let p = BottleneckProfiler::new();
        p.enable();
        // Same location, different durations
        p.record(BottleneckKind::GpuSync, "a.rs:1", "dl", 100, None);
        p.record(BottleneckKind::GpuSync, "a.rs:1", "dl", 200, None);
        p.record(BottleneckKind::GpuSync, "b.rs:2", "ul", 50, None);

        let stats = p.location_stats();
        assert_eq!(stats.len(), 2);
        // First should be a.rs:1 (higher total)
        assert_eq!(stats[0].location, "a.rs:1");
        assert_eq!(stats[0].count, 2);
        assert_eq!(stats[0].total_us, 300);
        assert_eq!(stats[0].max_us, 200);
    }
}
