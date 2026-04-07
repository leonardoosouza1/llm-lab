// ============================================================================
// Physics Module — MetaVerse Core
// ============================================================================
// Implements core physics systems: diffusion, reaction-diffusion, thermodynamics,
// criticality, synchronization, and harmonic oscillations.
// ============================================================================

use std::f32::consts::PI;
use serde::{Serialize, Deserialize};

const BOLTZMANN: f32 = 1.380649e-23; // J/K, normalized to ~1e-4 in our scale
const GAS_CONSTANT: f32 = 8.314; // J/(mol*K)

/// ============================================================================
/// 1. DIFFUSION (Fick's Law)
/// ============================================================================

/// Applies discrete Fick diffusion on a 2D scalar field.
/// C(x,t+1) = C(x,t) + D * [C(x+1,t) - 2*C(x,t) + C(x-1,t)]
/// Grid wraps toroidally.
///
/// # Arguments
/// * `field` - Mutable flattened 2D grid (row-major: field[y * width + x])
/// * `width` - Grid width
/// * `height` - Grid height
/// * `diffusion_coeff` - Diffusion coefficient D (typically 0.1-0.5)
pub fn diffuse_scalar_field(field: &mut Vec<f32>, width: usize, height: usize, diffusion_coeff: f32) {
    if field.is_empty() || width == 0 || height == 0 {
        return;
    }

    let mut new_field = field.clone();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;

            // Toroidal wrapping for neighbors
            let left = (if x == 0 { width - 1 } else { x - 1 }) + y * width;
            let right = ((x + 1) % width) + y * width;
            let up = x + (if y == 0 { height - 1 } else { y - 1 }) * width;
            let down = x + ((y + 1) % height) * width;

            // Laplacian: ∇²C = C(left) + C(right) + C(up) + C(down) - 4*C(center)
            let laplacian = field[left] + field[right] + field[up] + field[down] - 4.0 * field[idx];

            // Update: C_new = C + D * ∇²C
            new_field[idx] = field[idx] + diffusion_coeff * laplacian;

            // Clamp to [0, 1]
            new_field[idx] = new_field[idx].max(0.0).min(1.0);
        }
    }

    *field = new_field;
}

/// ============================================================================
/// 2. GRAY-SCOTT REACTION-DIFFUSION (Turing Patterns)
/// ============================================================================

/// Parameters for Gray-Scott reaction-diffusion system
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GrayScottParams {
    /// Feed rate F: 0.01-0.07 (controls nutrient supply)
    pub feed_rate: f32,
    /// Kill rate k: 0.03-0.07 (removal rate of inhibitor)
    pub kill_rate: f32,
    /// Diffusion coefficient of activator u: ~0.16
    pub du: f32,
    /// Diffusion coefficient of inhibitor v: ~0.08
    pub dv: f32,
}

impl Default for GrayScottParams {
    fn default() -> Self {
        Self {
            feed_rate: 0.055,
            kill_rate: 0.062,
            du: 0.16,
            dv: 0.08,
        }
    }
}

impl GrayScottParams {
    pub fn turbulent() -> Self {
        Self {
            feed_rate: 0.034,
            kill_rate: 0.063,
            du: 0.16,
            dv: 0.08,
        }
    }

    pub fn stripes() -> Self {
        Self {
            feed_rate: 0.037,
            kill_rate: 0.060,
            du: 0.16,
            dv: 0.08,
        }
    }
}

/// Gray-Scott reaction-diffusion field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrayScottField {
    /// Activator concentration (typically U)
    pub u: Vec<f32>,
    /// Inhibitor concentration (typically V)
    pub v: Vec<f32>,
    /// Grid width
    pub width: usize,
    /// Grid height
    pub height: usize,
}

impl GrayScottField {
    /// Create a new Gray-Scott field with u=1.0 everywhere, v=0.0 with seed perturbations
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        let mut u = vec![1.0; size];
        let mut v = vec![0.0; size];

        // Add small perturbations at center to seed pattern formation
        if width > 10 && height > 10 {
            let cx = width / 2;
            let cy = height / 2;
            let radius = 5;

            for dy in 0..radius {
                for dx in 0..radius {
                    if let Some(x) = cx.checked_sub(radius / 2).and_then(|x| x.checked_add(dx)) {
                        if let Some(y) = cy.checked_sub(radius / 2).and_then(|y| y.checked_add(dy)) {
                            if x < width && y < height {
                                let idx = y * width + x;
                                u[idx] = 0.5;
                                v[idx] = 0.25;
                            }
                        }
                    }
                }
            }
        }

        Self { u, v, width, height }
    }

    /// Apply one timestep of Gray-Scott reaction-diffusion
    /// du/dt = Du * ∇²u - u*v² + F*(1-u)
    /// dv/dt = Dv * ∇²v + u*v² - (F+k)*v
    pub fn step(&mut self, params: &GrayScottParams) {
        let size = self.u.len();
        if size == 0 || self.width == 0 || self.height == 0 {
            return;
        }

        let mut du = vec![0.0; size];
        let mut dv = vec![0.0; size];

        // Compute Laplacians
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;

                let left = (if x == 0 { self.width - 1 } else { x - 1 }) + y * self.width;
                let right = ((x + 1) % self.width) + y * self.width;
                let up = x + (if y == 0 { self.height - 1 } else { y - 1 }) * self.width;
                let down = x + ((y + 1) % self.height) * self.width;

                let laplacian_u =
                    self.u[left] + self.u[right] + self.u[up] + self.u[down] - 4.0 * self.u[idx];
                let laplacian_v =
                    self.v[left] + self.v[right] + self.v[up] + self.v[down] - 4.0 * self.v[idx];

                // Reaction terms
                let reaction = self.u[idx] * self.v[idx] * self.v[idx];

                // du/dt = Du * ∇²u - u*v² + F*(1-u)
                du[idx] = params.du * laplacian_u - reaction + params.feed_rate * (1.0 - self.u[idx]);

                // dv/dt = Dv * ∇²v + u*v² - (F+k)*v
                dv[idx] = params.dv * laplacian_v + reaction - (params.feed_rate + params.kill_rate) * self.v[idx];
            }
        }

        // Update concentrations (dt = 1.0)
        for i in 0..size {
            self.u[i] = (self.u[i] + du[i]).max(0.0).min(1.0);
            self.v[i] = (self.v[i] + dv[i]).max(0.0).min(1.0);
        }
    }

    /// Convert u field to a topographic height map [0..1]
    pub fn generate_topography(&self) -> Vec<f32> {
        self.u.iter().map(|&val| val.max(0.0).min(1.0)).collect()
    }

    /// Get topography by taking U channel (alias)
    pub fn topography(&self) -> Vec<f32> {
        self.u.clone()
    }
}

/// ============================================================================
/// 3. THERMODYNAMICS
/// ============================================================================

/// Landauer limit: minimum energy cost per irreversible bit operation
/// E_min = k_B * T * ln(2) ≈ 0.693 * k_B * T
///
/// # Arguments
/// * `n_bits` - Number of bits being irreversibly erased
/// * `temperature` - Temperature in Kelvin
///
/// # Returns
/// Minimum energy required in Joules (normalized scale)
pub fn landauer_cost(n_bits: usize, temperature: f32) -> f32 {
    let n = n_bits as f32;
    let ln2 = 0.693147;
    n * ln2 * BOLTZMANN * temperature.max(1.0)
}

/// Arrhenius equation: reaction rate as function of temperature
/// k = A * exp(-Ea / (R * T))
///
/// # Arguments
/// * `base_rate` - Pre-exponential factor A
/// * `activation_energy` - Activation energy in J/mol
/// * `temperature` - Temperature in Kelvin
///
/// # Returns
/// Reaction rate k
pub fn arrhenius_rate(base_rate: f32, activation_energy: f32, temperature: f32) -> f32 {
    let t = temperature.max(1.0);
    let exponent = -activation_energy / (GAS_CONSTANT * t);
    base_rate * exponent.exp()
}

/// Gibbs free energy: spontaneity of reaction
/// ΔG = ΔH - T * ΔS
/// Reaction is spontaneous if ΔG < 0
///
/// # Arguments
/// * `enthalpy` - Enthalpy change ΔH in J/mol
/// * `temperature` - Temperature in Kelvin
/// * `entropy` - Entropy change ΔS in J/(mol*K)
///
/// # Returns
/// Gibbs free energy ΔG in J/mol
pub fn gibbs_free_energy(enthalpy: f32, temperature: f32, entropy: f32) -> f32 {
    let t = temperature.max(1.0);
    enthalpy - t * entropy
}

/// ============================================================================
/// 4. CRITICALITY (Self-Organized Criticality)
/// ============================================================================

/// Monitor and adjust system for self-organized criticality
/// Tracks branching ratio σ (target = 1.0)
#[derive(Debug, Clone)]
pub struct CriticalityMonitor {
    /// Branching ratio σ: average number of events triggered per event
    pub branching_ratio: f32,
    /// Recorded avalanche sizes
    avalanche_sizes: Vec<usize>,
    /// Window size for moving average
    window_size: usize,
}

impl CriticalityMonitor {
    /// Create a new criticality monitor
    pub fn new() -> Self {
        Self {
            branching_ratio: 0.5,
            avalanche_sizes: Vec::new(),
            window_size: 100,
        }
    }

    /// Record an avalanche event
    pub fn record_avalanche(&mut self, size: usize) {
        self.avalanche_sizes.push(size);
        if self.avalanche_sizes.len() > self.window_size * 10 {
            self.avalanche_sizes.drain(0..self.window_size);
        }
    }

    /// Compute branching ratio from recent avalanches
    /// σ = (mean avalanche size at time t+1) / (mean avalanche size at time t)
    pub fn compute_branching_ratio(&self) -> f32 {
        if self.avalanche_sizes.len() < 2 {
            return 0.5;
        }

        let n = self.avalanche_sizes.len();
        let window = self.window_size.min(n / 2).max(2);

        let mean_before = self.avalanche_sizes[n - 2 * window..n - window]
            .iter()
            .map(|&s| s as f32)
            .sum::<f32>()
            / window as f32;

        let mean_after = self.avalanche_sizes[n - window..n]
            .iter()
            .map(|&s| s as f32)
            .sum::<f32>()
            / window as f32;

        if mean_before > 1e-6 {
            (mean_after / mean_before).max(0.01).min(2.0)
        } else {
            1.0
        }
    }

    /// Check if system is in critical state (σ ≈ 1.0 ± 0.1)
    pub fn is_critical(&self) -> bool {
        let sigma = self.compute_branching_ratio();
        (sigma - 1.0).abs() < 0.1
    }

    /// Return adjustment factor to push toward criticality
    /// Returns value in [0.5, 1.5] to adjust system parameters
    pub fn criticality_adjustment(&self) -> f32 {
        let sigma = self.compute_branching_ratio();
        // If σ > 1, system is supercritical, reduce feedback (multiply by <1)
        // If σ < 1, system is subcritical, increase feedback (multiply by >1)
        let target = 1.0;
        let error = sigma - target;
        let pid_p = 0.2; // Proportional gain
        (1.0 - pid_p * error).max(0.5).min(1.5)
    }
}

impl Default for CriticalityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// ============================================================================
/// 5. KURAMOTO SYNCHRONIZATION
/// ============================================================================

/// Kuramoto order parameter for a set of oscillators
/// r * exp(i*ψ) = (1/N) * Σ exp(i*θ_j)
/// Returns (r, ψ) where r is the magnitude [0,1] and ψ is the phase
///
/// # Arguments
/// * `phases` - Array of oscillator phases in radians
///
/// # Returns
/// Tuple (r, psi) where r is order parameter magnitude, psi is mean phase
pub fn kuramoto_order_parameter(phases: &[f32]) -> (f32, f32) {
    if phases.is_empty() {
        return (0.0, 0.0);
    }

    let n = phases.len() as f32;
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;

    for &theta in phases {
        sin_sum += theta.sin();
        cos_sum += theta.cos();
    }

    let r = ((sin_sum * sin_sum + cos_sum * cos_sum) / (n * n)).sqrt();
    let psi = sin_sum.atan2(cos_sum);

    (r, psi)
}

/// Update oscillator phases with Kuramoto coupling
/// dθ_i/dt = ω_i + (K/N) * Σ sin(θ_j - θ_i)
///
/// # Arguments
/// * `phases` - Mutable array of oscillator phases
/// * `frequencies` - Natural frequencies of oscillators
/// * `coupling` - Coupling strength K
/// * `dt` - Time step
pub fn kuramoto_step(phases: &mut [f32], frequencies: &[f32], coupling: f32, dt: f32) {
    if phases.is_empty() || phases.len() != frequencies.len() {
        return;
    }

    let n = phases.len() as f32;
    let mut deltas = vec![0.0; phases.len()];

    for i in 0..phases.len() {
        let mut coupling_sum = 0.0;

        for j in 0..phases.len() {
            if i != j {
                let phase_diff = phases[j] - phases[i];
                coupling_sum += phase_diff.sin();
            }
        }

        // dθ_i/dt = ω_i + (K/N) * Σ sin(θ_j - θ_i)
        deltas[i] = frequencies[i] + (coupling / n) * coupling_sum;
    }

    // Update phases
    for i in 0..phases.len() {
        phases[i] += deltas[i] * dt;

        // Normalize to [-π, π]
        while phases[i] > PI {
            phases[i] -= 2.0 * PI;
        }
        while phases[i] < -PI {
            phases[i] += 2.0 * PI;
        }
    }
}

/// ============================================================================
/// 6. HARMONIC OSCILLATOR
/// ============================================================================

/// A damped harmonic oscillator with external forcing
/// Equation: x'' + γ*x' + ω²*x = F
#[derive(Debug, Clone, Copy)]
pub struct HarmonicOscillator {
    /// Current position
    pub position: f32,
    /// Current velocity
    pub velocity: f32,
    /// Natural frequency ω in rad/s
    pub natural_freq: f32,
    /// Damping coefficient γ (energy dissipation rate)
    pub damping: f32,
}

impl HarmonicOscillator {
    /// Create a new harmonic oscillator
    /// # Arguments
    /// * `freq` - Natural frequency ω in rad/s
    /// * `damping` - Damping coefficient γ (0=undamped, 1=critically damped)
    pub fn new(freq: f32, damping: f32) -> Self {
        Self {
            position: 0.0,
            velocity: 0.0,
            natural_freq: freq.max(0.01),
            damping: damping.max(0.0).min(2.0),
        }
    }

    /// Advance oscillator by one timestep with external forcing
    /// x'' + γ*x' + ω²*x = F
    /// Using Verlet integration for stability
    ///
    /// # Arguments
    /// * `forcing` - External force F
    /// * `dt` - Time step
    pub fn step(&mut self, forcing: f32, dt: f32) {
        let omega_sq = self.natural_freq * self.natural_freq;

        // Acceleration: x'' = F - γ*x' - ω²*x
        let acceleration = forcing - self.damping * self.velocity - omega_sq * self.position;

        // Update velocity and position (Verlet)
        self.velocity += acceleration * dt;
        self.position += self.velocity * dt;

        // Damp velocity (energy dissipation)
        self.velocity *= (1.0 - self.damping * dt).max(0.0);
    }

    /// Calculate total energy (kinetic + potential)
    /// E = (1/2)*m*v² + (1/2)*k*x² where k = m*ω²
    /// Assuming unit mass: E = (1/2)*v² + (1/2)*ω²*x²
    pub fn energy(&self) -> f32 {
        let kinetic = 0.5 * self.velocity * self.velocity;
        let potential = 0.5 * self.natural_freq * self.natural_freq * self.position * self.position;
        kinetic + potential
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Diffusion Tests
    // ========================================================================

    #[test]
    fn test_diffuse_scalar_field_mass_conservation() {
        // Test that total mass is approximately conserved during diffusion
        let width = 32;
        let height = 32;
        let mut field = vec![0.5; width * height];

        let initial_sum: f32 = field.iter().sum();

        diffuse_scalar_field(&mut field, width, height, 0.1);

        let final_sum: f32 = field.iter().sum();

        // Mass should be approximately conserved (within clamping tolerance)
        assert!((initial_sum - final_sum).abs() < initial_sum * 0.05);
    }

    #[test]
    fn test_diffuse_scalar_field_smoothing() {
        // Test that diffusion smooths out sharp gradients
        let width = 16;
        let height = 16;
        let mut field = vec![0.0; width * height];

        // Create a spike
        field[width / 2 + (height / 2) * width] = 1.0;
        field[width / 2 + 1 + (height / 2) * width] = 0.0;

        let spike_before = field[width / 2 + (height / 2) * width];

        diffuse_scalar_field(&mut field, width, height, 0.2);

        let spike_after = field[width / 2 + (height / 2) * width];

        // Spike should decrease due to diffusion
        assert!(spike_after < spike_before);
        assert!(spike_after > 0.0); // But not disappear
    }

    // ========================================================================
    // Gray-Scott Tests
    // ========================================================================

    #[test]
    fn test_gray_scott_field_initialization() {
        let gs = GrayScottField::new(64, 64);

        assert_eq!(gs.u.len(), 64 * 64);
        assert_eq!(gs.v.len(), 64 * 64);

        // u should be mostly 1.0
        let u_mean: f32 = gs.u.iter().sum::<f32>() / gs.u.len() as f32;
        assert!(u_mean > 0.95);

        // v should be mostly 0.0 with some perturbations
        let v_max = gs.v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(v_max > 0.0);
    }

    #[test]
    fn test_gray_scott_pattern_formation() {
        // Test that Gray-Scott produces non-trivial patterns after 1000 steps
        let mut gs = GrayScottField::new(128, 128);
        let params = GrayScottParams::default();

        for _ in 0..1000 {
            gs.step(&params);
        }

        // After 1000 steps, u field should have significant variance
        let u_mean: f32 = gs.u.iter().sum::<f32>() / gs.u.len() as f32;
        let variance = gs.u.iter().map(|&v| (v - u_mean).powi(2)).sum::<f32>() / gs.u.len() as f32;

        assert!(variance > 0.01, "Gray-Scott pattern variance too low: {}", variance);
    }

    #[test]
    fn test_gray_scott_topography_generation() {
        let gs = GrayScottField::new(32, 32);
        let topo = gs.generate_topography();

        assert_eq!(topo.len(), 32 * 32);
        assert!(topo.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    // ========================================================================
    // Thermodynamics Tests
    // ========================================================================

    #[test]
    fn test_landauer_cost_increases_with_bits() {
        let temp = 310.0;
        let cost_1 = landauer_cost(1, temp);
        let cost_10 = landauer_cost(10, temp);
        let cost_100 = landauer_cost(100, temp);

        assert!(cost_1 < cost_10);
        assert!(cost_10 < cost_100);
    }

    #[test]
    fn test_landauer_cost_increases_with_temperature() {
        let temp_low = 100.0;
        let temp_high = 400.0;

        let cost_low = landauer_cost(10, temp_low);
        let cost_high = landauer_cost(10, temp_high);

        assert!(cost_low < cost_high);
    }

    #[test]
    fn test_arrhenius_rate_increases_with_temperature() {
        let base_rate = 1.0;
        let activation_energy = 50000.0; // J/mol

        let rate_300 = arrhenius_rate(base_rate, activation_energy, 300.0);
        let rate_400 = arrhenius_rate(base_rate, activation_energy, 400.0);

        assert!(rate_300 > 0.0);
        assert!(rate_400 > rate_300);
    }

    #[test]
    fn test_arrhenius_rate_decreases_with_activation_energy() {
        let base_rate = 1.0;
        let temperature = 350.0;

        let rate_low_ea = arrhenius_rate(base_rate, 20000.0, temperature);
        let rate_high_ea = arrhenius_rate(base_rate, 100000.0, temperature);

        assert!(rate_low_ea > rate_high_ea);
    }

    #[test]
    fn test_gibbs_free_energy_spontaneity() {
        // Exothermic + entropy increase (always spontaneous)
        let dg_spontaneous = gibbs_free_energy(-100.0, 300.0, 0.5);
        assert!(dg_spontaneous < 0.0);

        // Endothermic + low entropy increase (non-spontaneous at low T)
        let dg_non_spontaneous = gibbs_free_energy(100.0, 300.0, -0.2);
        assert!(dg_non_spontaneous > 0.0);
    }

    #[test]
    fn test_gibbs_free_energy_temperature_dependence() {
        // For endothermic but entropy-favorable reaction
        let dg_low_temp = gibbs_free_energy(50.0, 100.0, 0.5);
        let dg_high_temp = gibbs_free_energy(50.0, 400.0, 0.5);

        // Higher temperature favors entropy, so dG becomes more negative
        assert!(dg_high_temp < dg_low_temp);
    }

    // ========================================================================
    // Criticality Tests
    // ========================================================================

    #[test]
    fn test_criticality_monitor_branching_ratio() {
        let mut monitor = CriticalityMonitor::new();

        // Record some avalanches
        for i in 0..50 {
            monitor.record_avalanche(10 + i);
        }
        for i in 0..50 {
            monitor.record_avalanche(10 + i);
        }

        let sigma = monitor.compute_branching_ratio();
        assert!(sigma > 0.0);
        assert!(sigma < 3.0);
    }

    #[test]
    fn test_criticality_monitor_critical_detection() {
        let mut monitor = CriticalityMonitor::new();

        // Record consistent avalanches (near critical)
        for _ in 0..200 {
            monitor.record_avalanche(50);
        }

        assert!(monitor.is_critical());
    }

    #[test]
    fn test_criticality_adjustment_factor() {
        let mut monitor = CriticalityMonitor::new();

        // Subcritical: record two epochs of avalanches where second is smaller than first
        // First epoch: larger avalanches (100 events)
        for _ in 0..100 {
            monitor.record_avalanche(50);
        }
        // Second epoch: smaller avalanches (100 events) - means branching ratio σ < 1.0
        for _ in 0..100 {
            monitor.record_avalanche(10);
        }
        let subcritical_adj = monitor.criticality_adjustment();
        assert!(subcritical_adj > 1.0, "Subcritical adjustment should be > 1.0, got {}", subcritical_adj);

        // Supercritical: first epoch small, second epoch large (σ > 1.0)
        let mut monitor_super = CriticalityMonitor::new();
        for _ in 0..100 {
            monitor_super.record_avalanche(10);
        }
        for _ in 0..100 {
            monitor_super.record_avalanche(100);
        }
        let supercritical_adj = monitor_super.criticality_adjustment();
        assert!(supercritical_adj < 1.0, "Supercritical adjustment should be < 1.0, got {}", supercritical_adj);
    }

    // ========================================================================
    // Kuramoto Tests
    // ========================================================================

    #[test]
    fn test_kuramoto_order_parameter_synchronized() {
        // All oscillators at same phase
        let phases = vec![0.0; 50];
        let (r, _psi) = kuramoto_order_parameter(&phases);

        assert!(r > 0.99); // Should be nearly 1.0
    }

    #[test]
    fn test_kuramoto_order_parameter_random() {
        // Uniformly distributed phases
        let phases = vec![0.0, PI / 2.0, PI, -PI / 2.0];
        let (r, _psi) = kuramoto_order_parameter(&phases);

        assert!(r < 0.2); // Should be close to 0
    }

    #[test]
    fn test_kuramoto_synchronization() {
        // Identical oscillators with strong coupling should synchronize
        let mut phases = vec![0.0, 0.5, -0.3, 0.1, -0.8, 0.0, 0.5, -0.3, 0.1, -0.8];
        let frequencies = vec![1.0; 10]; // Identical frequencies (must match phases length)
        let coupling = 5.0; // Strong coupling

        let (r_before, _) = kuramoto_order_parameter(&phases);

        // Run 200 steps to allow synchronization
        for _ in 0..200 {
            kuramoto_step(&mut phases, &frequencies, coupling, 0.01);
        }

        let (r_after, _) = kuramoto_order_parameter(&phases);

        // Should improve synchronization
        assert!(r_after > r_before, "Synchronization did not improve: r_before = {}, r_after = {}", r_before, r_after);
        assert!(r_after > 0.9, "Failed to synchronize: r = {}", r_after);
    }

    #[test]
    fn test_kuramoto_frequency_mismatch() {
        // Oscillators with different frequencies should not fully sync
        // This tests the case where intrinsic frequency mismatch prevents synchronization
        let mut phases = vec![0.0; 10];
        let mut frequencies = vec![1.0; 10];
        // Half at different frequency - larger mismatch effect
        frequencies[0] = 2.0;
        frequencies[1] = 2.0;
        frequencies[2] = 2.0;
        frequencies[3] = 2.0;
        frequencies[4] = 2.0;

        let coupling = 0.5; // Very weak coupling - insufficient to overcome frequency mismatch

        for _ in 0..500 {
            kuramoto_step(&mut phases, &frequencies, coupling, 0.01);
        }

        let (r, _) = kuramoto_order_parameter(&phases);

        // Should not achieve full synchronization due to large frequency mismatch
        assert!(r < 0.95, "Expected incomplete synchronization due to frequency mismatch, got r = {}", r);
    }

    // ========================================================================
    // Harmonic Oscillator Tests
    // ========================================================================

    #[test]
    fn test_harmonic_oscillator_creation() {
        let osc = HarmonicOscillator::new(1.0, 0.1);

        assert_eq!(osc.position, 0.0);
        assert_eq!(osc.velocity, 0.0);
        assert!(osc.natural_freq > 0.0);
        assert!(osc.damping >= 0.0);
    }

    #[test]
    fn test_harmonic_oscillator_free_oscillation() {
        let mut osc = HarmonicOscillator::new(2.0, 0.05);
        osc.position = 1.0;

        let initial_energy = osc.energy();

        // Run for 10 steps with no forcing
        for _ in 0..10 {
            osc.step(0.0, 0.01);
        }

        let final_energy = osc.energy();

        // Energy should decrease due to damping
        assert!(final_energy < initial_energy);
        assert!(final_energy > 0.0); // But not go to zero immediately
    }

    #[test]
    fn test_harmonic_oscillator_energy_dissipation() {
        // Undamped should maintain energy better than damped
        let mut undamped = HarmonicOscillator::new(2.0, 0.001);
        undamped.position = 1.0;

        let mut damped = HarmonicOscillator::new(2.0, 0.5);
        damped.position = 1.0;

        let undamped_initial = undamped.energy();
        let damped_initial = damped.energy();

        for _ in 0..50 {
            undamped.step(0.0, 0.01);
            damped.step(0.0, 0.01);
        }

        let undamped_final = undamped.energy();
        let damped_final = damped.energy();

        // Damped should lose more energy
        let undamped_loss = undamped_initial - undamped_final;
        let damped_loss = damped_initial - damped_final;

        assert!(damped_loss > undamped_loss);
    }

    #[test]
    fn test_harmonic_oscillator_forced_response() {
        let mut osc = HarmonicOscillator::new(1.0, 0.2);

        // Apply continuous forcing
        let forcing = 0.5;
        for _ in 0..100 {
            osc.step(forcing, 0.01);
        }

        // Should reach steady state with non-zero position
        assert!(osc.position.abs() > 0.1);
    }

    #[test]
    fn test_harmonic_oscillator_resonance() {
        // Apply forcing at natural frequency
        let freq = 2.0;
        let mut osc = HarmonicOscillator::new(freq, 0.05); // Lower damping for better resonance

        // Forcing at resonance frequency
        let forcing_freq = freq;
        let mut phase: f32 = 0.0;
        let dt = 0.01;

        let mut max_amplitude: f32 = 0.0;

        // Run more steps to allow resonance buildup and observe steady-state
        for _ in 0..1000 {
            let forcing = 0.5 * (phase).sin(); // Stronger forcing amplitude
            osc.step(forcing, dt);
            phase += forcing_freq * dt;
            max_amplitude = max_amplitude.max(osc.position.abs());
        }

        // Should build up significant amplitude at resonance
        assert!(max_amplitude > 0.5, "Max amplitude at resonance: {}", max_amplitude);
    }
}
