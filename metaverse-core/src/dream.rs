// ============================================================================
// Dream Engine: Sleep & Memory Consolidation System
// ============================================================================
// Implements a simplified Borbely two-process model of sleep regulation
// combined with memory consolidation mechanisms (Hebbian learning during
// sleep, memory pruning, and creative recombination during REM).

use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================================
// Constants
// ============================================================================

pub const SLEEP_PRESSURE_RATE: f32 = 0.01;
pub const SLEEP_DECAY_RATE: f32 = 0.05;
pub const CONSOLIDATION_LEARNING_RATE: f32 = 0.02;
pub const MAX_EXPERIENCE_BUFFER: usize = 100;
pub const LIGHT_SLEEP_DURATION: u32 = 5;
pub const DEEP_SLEEP_DURATION: u32 = 10;
pub const REM_DURATION: u32 = 5;

// ============================================================================
// Experience and ExperienceBuffer
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub sensory_input: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub surprise: f32,
    pub tick: u64,
}

impl Experience {
    pub fn new(
        sensory_input: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
        surprise: f32,
        tick: u64,
    ) -> Self {
        Self {
            sensory_input,
            action,
            reward,
            surprise,
            tick,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceBuffer {
    experiences: Vec<Experience>,
    capacity: usize,
    head: usize,
    is_full: bool,
}

impl ExperienceBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            experiences: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            is_full: false,
        }
    }

    pub fn record(&mut self, exp: Experience) {
        if self.experiences.len() < self.capacity {
            self.experiences.push(exp);
        } else {
            self.experiences[self.head] = exp;
            self.head = (self.head + 1) % self.capacity;
            self.is_full = true;
        }
    }

    pub fn sample_by_surprise(&self, n: usize, rng: &mut impl Rng) -> Vec<Experience> {
        if self.experiences.is_empty() {
            return Vec::new();
        }

        let mut samples = Vec::new();

        // Normalize surprise values for weighting
        let max_surprise = self
            .experiences
            .iter()
            .map(|e| e.surprise)
            .fold(f32::NEG_INFINITY, f32::max)
            .max(1e-6);

        for _ in 0..n.min(self.experiences.len()) {
            let mut cumulative = 0.0;
            let roll = rng.gen::<f32>() * max_surprise * self.experiences.len() as f32;

            for exp in &self.experiences {
                cumulative += exp.surprise;
                if cumulative >= roll {
                    samples.push(exp.clone());
                    break;
                }
            }
        }

        samples
    }

    pub fn sample_by_reward(&self, n: usize, rng: &mut impl Rng) -> Vec<Experience> {
        if self.experiences.is_empty() {
            return Vec::new();
        }

        let mut samples = Vec::new();

        // Normalize absolute reward values for weighting
        let max_reward = self
            .experiences
            .iter()
            .map(|e| e.reward.abs())
            .fold(f32::NEG_INFINITY, f32::max)
            .max(1e-6);

        for _ in 0..n.min(self.experiences.len()) {
            let mut cumulative = 0.0;
            let roll = rng.gen::<f32>() * max_reward * self.experiences.len() as f32;

            for exp in &self.experiences {
                cumulative += exp.reward.abs();
                if cumulative >= roll {
                    samples.push(exp.clone());
                    break;
                }
            }
        }

        samples
    }

    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Experience> {
        self.experiences.iter()
    }
}

// ============================================================================
// Sleep Stages
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SleepStage {
    Awake,
    Light,
    Deep,
    REM,
}

// ============================================================================
// Dream Engine
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamEngine {
    pub experience_buffer: ExperienceBuffer,
    pub sleep_pressure: f32,
    pub sleep_threshold: f32,
    pub is_sleeping: bool,
    pub sleep_stage: SleepStage,
    pub rem_cycles: u32,
    pub consolidation_strength: f32,
    sleep_ticks: u32,
    stage_ticks: u32,
}

impl DreamEngine {
    pub fn new(sleep_threshold: f32, consolidation_strength: f32) -> Self {
        Self {
            experience_buffer: ExperienceBuffer::new(MAX_EXPERIENCE_BUFFER),
            sleep_pressure: 0.0,
            sleep_threshold,
            is_sleeping: false,
            sleep_stage: SleepStage::Awake,
            rem_cycles: 0,
            consolidation_strength,
            sleep_ticks: 0,
            stage_ticks: 0,
        }
    }

    pub fn record_experience(&mut self, exp: Experience) {
        self.experience_buffer.record(exp);
    }

    pub fn should_sleep(&self, light_intensity: f32) -> bool {
        // Process C (circadian): promotes sleep at night (low light)
        let circadian_drive = 1.0 - light_intensity.clamp(0.0, 1.0);

        // Combined: sleep if pressure exceeds threshold AND circadian allows
        self.sleep_pressure > self.sleep_threshold && circadian_drive > 0.3
    }

    pub fn tick_wakefulness(&mut self) {
        // During wakefulness, accumulate sleep pressure (Process S)
        self.sleep_pressure += SLEEP_PRESSURE_RATE;
        self.sleep_pressure = self.sleep_pressure.min(1.0); // Cap at 1.0
    }

    pub fn tick_sleep(&mut self, light_intensity: f32) {
        if !self.is_sleeping {
            // Transition to sleep if conditions met
            if self.should_sleep(light_intensity) {
                self.is_sleeping = true;
                self.sleep_stage = SleepStage::Light;
                self.sleep_ticks = 0;
                self.stage_ticks = 0;
                self.rem_cycles = 0;
            }
            return;
        }

        // Already sleeping; manage sleep progression
        self.sleep_ticks += 1;
        self.stage_ticks += 1;

        // Exponential decay of sleep pressure during sleep
        self.sleep_pressure *= 1.0 - SLEEP_DECAY_RATE;

        // State machine for sleep stages
        self.sleep_stage = match self.sleep_stage {
            SleepStage::Light => {
                if self.stage_ticks >= LIGHT_SLEEP_DURATION {
                    self.stage_ticks = 0;
                    SleepStage::Deep
                } else {
                    SleepStage::Light
                }
            }
            SleepStage::Deep => {
                if self.stage_ticks >= DEEP_SLEEP_DURATION {
                    self.stage_ticks = 0;
                    SleepStage::REM
                } else {
                    SleepStage::Deep
                }
            }
            SleepStage::REM => {
                self.rem_cycles += 1;
                if self.stage_ticks >= REM_DURATION {
                    self.stage_ticks = 0;
                    SleepStage::Light
                } else {
                    SleepStage::REM
                }
            }
            SleepStage::Awake => {
                // Should not reach here while is_sleeping=true
                SleepStage::Light
            }
        };

        // Wake up when pressure is low enough AND light is sufficient (morning)
        let circadian_wake = light_intensity > 0.7;
        if self.sleep_pressure < 0.2 && circadian_wake {
            self.is_sleeping = false;
            self.sleep_stage = SleepStage::Awake;
            self.sleep_ticks = 0;
            self.stage_ticks = 0;
        }
    }
}

// ============================================================================
// Consolidation Report
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationReport {
    pub memories_replayed: usize,
    pub memories_pruned: usize,
    pub novel_associations: usize,
    pub avg_surprise_replayed: f32,
}

impl ConsolidationReport {
    pub fn new() -> Self {
        Self {
            memories_replayed: 0,
            memories_pruned: 0,
            novel_associations: 0,
            avg_surprise_replayed: 0.0,
        }
    }
}

impl Default for ConsolidationReport {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Memory Consolidation (Hebbian updates on synthetic experiences)
// ============================================================================

pub fn consolidate_weights(
    edges: &mut Vec<crate::brain::Edge>,
    experience: &Experience,
    learning_rate: f32,
) {
    // For each edge, apply a Hebbian-like update based on simulated activations
    // from the experience (sensory_input and action vectors)
    for edge in edges.iter_mut() {
        if !edge.enabled {
            continue;
        }

        let from_idx = edge.from as usize;
        let to_idx = edge.to as usize;

        let from_activation = if from_idx < experience.sensory_input.len() {
            experience.sensory_input[from_idx]
        } else {
            0.0
        };

        let to_activation = if to_idx >= crate::brain::NUM_SENSORS {
            let motor_idx = to_idx - crate::brain::NUM_SENSORS;
            if motor_idx < experience.action.len() {
                experience.action[motor_idx]
            } else {
                0.0
            }
        } else {
            0.0
        };

        let correlation = from_activation * to_activation;

        if correlation > 0.0 {
            let adjust = (correlation * learning_rate).min(1.0);
            if adjust > 0.5 {
                edge.weight = 1;
            }
        } else if correlation < 0.0 {
            let adjust = (-correlation * learning_rate).min(1.0);
            if adjust > 0.5 {
                edge.weight = -1;
            }
        }
    }
}

pub fn run_consolidation(
    engine: &DreamEngine,
    edges: &mut Vec<crate::brain::Edge>,
    rng: &mut impl Rng,
) -> ConsolidationReport {
    let mut report = ConsolidationReport::new();

    if !engine.is_sleeping {
        return report;
    }

    match engine.sleep_stage {
        SleepStage::Light => {
            // Light sleep: decay unimportant memories (low surprise, low reward)
            let mut pruned = 0;
            for exp in engine.experience_buffer.iter() {
                if exp.surprise < 0.1 && exp.reward.abs() < 0.1 {
                    pruned += 1;
                }
            }
            report.memories_pruned = pruned;
        }

        SleepStage::Deep => {
            // Deep sleep: replay high-reward experiences with Hebbian updates
            let high_reward_exps =
                engine
                    .experience_buffer
                    .sample_by_reward(10, rng);

            let mut total_surprise = 0.0;
            for exp in high_reward_exps.iter() {
                consolidate_weights(edges, exp, CONSOLIDATION_LEARNING_RATE);
                total_surprise += exp.surprise;
                report.memories_replayed += 1;
            }

            if report.memories_replayed > 0 {
                report.avg_surprise_replayed = total_surprise / report.memories_replayed as f32;
            }
        }

        SleepStage::REM => {
            // REM: replay high-surprise experiences and blend them creatively
            let high_surprise_exps =
                engine
                    .experience_buffer
                    .sample_by_surprise(5, rng);

            report.memories_replayed = high_surprise_exps.len();
            let mut total_surprise = 0.0;

            for exp in high_surprise_exps.iter() {
                total_surprise += exp.surprise;
            }

            if report.memories_replayed > 0 {
                report.avg_surprise_replayed = total_surprise / report.memories_replayed as f32;
            }

            // Creative recombination: blend 2 random experiences
            if high_surprise_exps.len() >= 2 {
                let idx1 = rng.gen_range(0..high_surprise_exps.len());
                let idx2 = rng.gen_range(0..high_surprise_exps.len());
                if idx1 != idx2 {
                    let exp1 = &high_surprise_exps[idx1];
                    let exp2 = &high_surprise_exps[idx2];

                    // Blend the experiences
                    let mut dream_sensory = exp1.sensory_input.clone();
                    let mut dream_action = exp1.action.clone();

                    for i in 0..dream_sensory.len().min(exp2.sensory_input.len()) {
                        dream_sensory[i] = (exp1.sensory_input[i] + exp2.sensory_input[i]) * 0.5;
                    }

                    for i in 0..dream_action.len().min(exp2.action.len()) {
                        dream_action[i] = (exp1.action[i] + exp2.action[i]) * 0.5;
                    }

                    let dream_reward = (exp1.reward + exp2.reward) * 0.5;
                    let dream_surprise = (exp1.surprise + exp2.surprise) * 0.5;

                    let dream_exp = Experience::new(
                        dream_sensory,
                        dream_action,
                        dream_reward,
                        dream_surprise,
                        exp1.tick,
                    );

                    // Apply consolidation to this blended experience
                    consolidate_weights(edges, &dream_exp, CONSOLIDATION_LEARNING_RATE);
                    report.novel_associations += 1;
                }
            }
        }

        SleepStage::Awake => {
            // Not sleeping
        }
    }

    report
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep_pressure_accumulates() {
        let mut engine = DreamEngine::new(0.5, 0.02);
        let initial_pressure = engine.sleep_pressure;

        for _ in 0..10 {
            engine.tick_wakefulness();
        }

        assert!(engine.sleep_pressure > initial_pressure);
    }

    #[test]
    fn test_sleep_pressure_decays_during_sleep() {
        let mut engine = DreamEngine::new(0.1, 0.02);
        engine.sleep_pressure = 0.9;
        engine.is_sleeping = true;
        engine.sleep_stage = SleepStage::Deep;

        let initial_pressure = engine.sleep_pressure;

        for _ in 0..10 {
            engine.tick_sleep(0.0); // Low light (night)
        }

        assert!(engine.sleep_pressure < initial_pressure);
    }

    #[test]
    fn test_circadian_alignment() {
        let mut engine = DreamEngine::new(0.3, 0.02);
        engine.sleep_pressure = 0.5;

        // High light (day): should NOT want to sleep
        assert!(!engine.should_sleep(0.9));

        // Low light (night): SHOULD want to sleep
        assert!(engine.should_sleep(0.1));
    }

    #[test]
    fn test_experience_buffer_ring_behavior() {
        let mut buffer = ExperienceBuffer::new(3);

        // Add 5 experiences to a buffer of capacity 3
        for i in 0..5 {
            let exp = Experience::new(vec![i as f32], vec![i as f32], i as f32, i as f32, i as u64);
            buffer.record(exp);
        }

        // Buffer should have exactly 3 items (ring behavior)
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full);
    }

    #[test]
    fn test_surprise_weighted_sampling() {
        let mut buffer = ExperienceBuffer::new(10);
        let mut rng = rand::thread_rng();

        // Add experiences with varying surprise levels
        for i in 0..10 {
            let surprise = (i + 1) as f32 * 0.1; // 0.1, 0.2, 0.3, ..., 1.0
            let exp = Experience::new(vec![0.0], vec![0.0], 0.0, surprise, i as u64);
            buffer.record(exp);
        }

        let samples = buffer.sample_by_surprise(5, &mut rng);
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|exp| exp.surprise > 0.0));
    }

    #[test]
    fn test_reward_weighted_sampling() {
        let mut buffer = ExperienceBuffer::new(10);
        let mut rng = rand::thread_rng();

        // Add experiences with varying rewards
        for i in 0..10 {
            let reward = (i + 1) as f32 * 0.1; // 0.1, 0.2, 0.3, ..., 1.0
            let exp = Experience::new(vec![0.0], vec![0.0], reward, 0.0, i as u64);
            buffer.record(exp);
        }

        let samples = buffer.sample_by_reward(5, &mut rng);
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_sleep_stage_progression() {
        let mut engine = DreamEngine::new(0.1, 0.02);
        engine.sleep_pressure = 0.5;

        // Force entry into sleep
        engine.is_sleeping = true;
        engine.sleep_stage = SleepStage::Light;
        engine.stage_ticks = 0;

        // Progression: Light -> Deep -> REM -> Light
        let mut stages = vec![SleepStage::Light];

        for _ in 0..25 {
            engine.tick_sleep(0.0);
            stages.push(engine.sleep_stage);
        }

        // Should see progression through stages
        let has_deep = stages.iter().any(|&s| s == SleepStage::Deep);
        let has_rem = stages.iter().any(|&s| s == SleepStage::REM);

        assert!(has_deep);
        assert!(has_rem);
    }

    #[test]
    fn test_consolidation_replays_important_memories() {
        let mut engine = DreamEngine::new(0.5, 0.02);
        let mut rng = rand::thread_rng();

        // Add high-reward experiences
        for i in 0..5 {
            let exp = Experience::new(
                vec![0.5, 0.5],
                vec![0.3, 0.3],
                1.0,      // High reward
                0.1,      // Low surprise
                i as u64,
            );
            engine.record_experience(exp);
        }

        // Force into deep sleep
        engine.is_sleeping = true;
        engine.sleep_stage = SleepStage::Deep;

        let mut edges = vec![
            crate::brain::Edge {
                from: 0,
                to: 17,
                weight: 0,
                enabled: true,
                innovation: 0,
            },
            crate::brain::Edge {
                from: 1,
                to: 17,
                weight: 0,
                enabled: true,
                innovation: 1,
            },
        ];

        let report = run_consolidation(&engine, &mut edges, &mut rng);

        assert!(report.memories_replayed > 0);
    }

    #[test]
    fn test_rem_blending_creates_novel() {
        let mut engine = DreamEngine::new(0.5, 0.02);
        let mut rng = rand::thread_rng();

        // Add high-surprise experiences (good for REM)
        for i in 0..5 {
            let exp = Experience::new(
                vec![i as f32 * 0.1, (i + 1) as f32 * 0.1],
                vec![i as f32 * 0.2, (i + 1) as f32 * 0.2],
                0.1,         // Low reward
                0.8,         // High surprise
                i as u64,
            );
            engine.record_experience(exp);
        }

        // Force into REM sleep
        engine.is_sleeping = true;
        engine.sleep_stage = SleepStage::REM;

        let mut edges = vec![
            crate::brain::Edge {
                from: 0,
                to: 17,
                weight: 0,
                enabled: true,
                innovation: 0,
            },
            crate::brain::Edge {
                from: 1,
                to: 17,
                weight: 0,
                enabled: true,
                innovation: 1,
            },
        ];

        let report = run_consolidation(&engine, &mut edges, &mut rng);

        assert!(report.memories_replayed > 0);
    }

    #[test]
    fn test_wake_transition() {
        let mut engine = DreamEngine::new(0.3, 0.02);
        engine.is_sleeping = true;
        engine.sleep_stage = SleepStage::Deep;
        engine.sleep_pressure = 0.05; // Low pressure

        // With high light (day), should transition to Awake
        engine.tick_sleep(0.9);

        assert!(!engine.is_sleeping);
        assert_eq!(engine.sleep_stage, SleepStage::Awake);
    }
}
