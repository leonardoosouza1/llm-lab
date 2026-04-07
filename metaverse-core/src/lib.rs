// ============================================================================
// MetaVerse Core — Mundo Metafísico para Organismos Digitais
// ============================================================================
// Um mundo virtual onde organismos baseados em tensores de LLM devem
// sobreviver, adaptar e evoluir sob pressão seletiva real.
//
// Fundamentado em: Termodinâmica, Prigogine, Friston, Kauffman, Turing,
// Kleiber, Kuramoto, NEAT, Lenia, e décadas de pesquisa em Vida Artificial.
// ============================================================================

pub mod physics;
pub mod chemistry;
pub mod world;
pub mod organism;
pub mod brain;
pub mod genetics;
pub mod ecology;
pub mod simulation;

/// Constantes fundamentais do MetaVerse
pub mod constants {
    /// Custo mínimo por bit apagado (Landauer normalizado)
    pub const E_MIN_BIT: f32 = 0.001;
    /// Eficiência de captura de luz por produtores (1-5%)
    pub const ETA_CAPTURE_BASE: f32 = 0.03;
    /// Eficiência de transferência entre níveis tróficos (~10%)
    pub const ETA_TRANSFER: f32 = 0.10;
    /// Expoente de Kleiber para taxa metabólica basal
    pub const KLEIBER_EXPONENT: f32 = 0.75;
    /// Coeficiente base de Kleiber
    pub const KLEIBER_B0: f32 = 1.0;
    /// Limiar crítico de branching ratio (criticalidade)
    pub const SIGMA_CRITICAL: f32 = 1.0;
    /// Fluxo mínimo de energia para auto-organização (Prigogine)
    pub const J_CRITICAL: f32 = 0.05;
    /// Tamanho default do grid
    pub const DEFAULT_GRID_SIZE: usize = 256;
    /// Ticks por ciclo diurno
    pub const TICKS_PER_DAY: u32 = 100;
    /// Dias por estação
    pub const DAYS_PER_SEASON: u32 = 25;
    /// Taxa de mutação base
    pub const BASE_MUTATION_RATE: f32 = 0.001;
}

/// Tipo de coordenada 2D
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn length(&self) -> f32 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn normalized(&self) -> Self {
        let l = self.length().max(1e-8);
        Self { x: self.x / l, y: self.y / l }
    }
    pub fn dot(&self, other: &Self) -> f32 { self.x * other.x + self.y * other.y }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self { x: self.x + rhs.x, y: self.y + rhs.y } }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self { x: self.x - rhs.x, y: self.y - rhs.y } }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self { Self { x: self.x * rhs, y: self.y * rhs } }
}

/// ID único de organismo
pub type OrganismId = u64;

/// Tick global da simulação
pub type Tick = u64;
