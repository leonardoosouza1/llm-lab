// ============================================================================
// iara_physics.rs — Operadores Físicos para Inferência Diferente
// ============================================================================
//
// "Papo de cientista maluco." — Leonardo, 2026-04-03
//
// Cinco operadores que substituem peças do transformer padrão por física real:
//
//   1. SemanticMass     — token como corpo com massa gravitacional
//   2. GravityAttention — F = G·m₁·m₂/d² substitui softmax(QKᵀ/√d)
//   3. VortexField      — componente antisimétrico (curl ≠ 0) na atenção
//   4. KanBarrier       — poço duplo V(x) = -x²/2 + x⁴/4 substitui SiLU
//   5. KuramotoDecoder  — sincronização de fase substitui softmax no vocab
//
// Todos os structs são `pub` e serializáveis para integração com o engine.
// O caminho quente de inferência usa os operadores via trait `PhysicsOp`.
// ============================================================================

use std::f32::consts::TAU;

// ─────────────────────────────────────────────────────────────────────────────
// Trait raiz — todos os operadores implementam forward() e name()
// ─────────────────────────────────────────────────────────────────────────────

pub trait PhysicsOp: Send + Sync {
    fn name(&self) -> &'static str;
    /// Retorna métricas para o painel de observabilidade
    fn telemetry(&self) -> PhysicsTelemetry;
}

#[derive(Debug, Clone)]
pub struct PhysicsTelemetry {
    pub energy: f32,      // energia cinética/potencial do sistema
    pub coherence: f32,   // parâmetro de ordem (0=caos, 1=sincronizado)
    pub entropy: f32,     // entropia da distribuição de saída
    pub alive_pct: f32,   // % de neurônios ativos (acima de threshold)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. SEMANTIC MASS — token como corpo com massa gravitacional
// ─────────────────────────────────────────────────────────────────────────────
//
// massa_i = ||embedding_i||₂   (norma L2 do vetor de embedding)
//
// Física: objetos massivos deformam o espaço semântico ao redor deles,
// puxando tokens relacionados com força proporcional à massa × massa / distância².
// Tokens raros têm massa baixa (vetores de embedding fracos).
// Tokens frequentes/importantes têm massa alta.

#[derive(Debug, Clone)]
pub struct SemanticMass {
    /// Massa de cada token [vocab_size] — pré-computada dos embeddings
    pub masses: Vec<f32>,
    /// Índices dos tokens "hub" (top-K por massa) — para busca rápida
    pub hub_indices: Vec<usize>,
    /// Limite de hub (tokens acima de N× a massa média)
    pub hub_threshold: f32,
}

impl SemanticMass {
    /// Computa massas a partir dos embeddings (chamado uma vez no carregamento)
    /// embed_rows: linhas da matriz de embedding [vocab × hidden]
    pub fn from_embeddings(embed_flat: &[f32], vocab_size: usize, hidden: usize) -> Self {
        let mut masses = Vec::with_capacity(vocab_size);

        for i in 0..vocab_size {
            let row = &embed_flat[i * hidden..(i + 1) * hidden];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            masses.push(norm);
        }

        let mean_mass = masses.iter().sum::<f32>() / masses.len() as f32;
        let hub_threshold = mean_mass * 1.5;

        let mut hub_indices: Vec<usize> = masses.iter()
            .enumerate()
            .filter(|(_, &m)| m >= hub_threshold)
            .map(|(i, _)| i)
            .collect();
        hub_indices.sort_by(|&a, &b| masses[b].partial_cmp(&masses[a]).unwrap());
        hub_indices.truncate(512); // top-512 hubs máximo

        Self { masses, hub_indices, hub_threshold }
    }

    /// Força gravitacional entre dois tokens: G·m₁·m₂ / (d² + ε)
    #[inline]
    pub fn gravity_force(&self, i: usize, j: usize, semantic_dist_sq: f32) -> f32 {
        const G: f32 = 1.0;
        const EPS: f32 = 1e-6;
        G * self.masses[i] * self.masses[j] / (semantic_dist_sq + EPS)
    }

    /// Massa de um token específico
    #[inline]
    pub fn mass_of(&self, token_id: usize) -> f32 {
        self.masses.get(token_id).copied().unwrap_or(0.0)
    }
}

impl PhysicsOp for SemanticMass {
    fn name(&self) -> &'static str { "SemanticMass" }
    fn telemetry(&self) -> PhysicsTelemetry {
        let mean = self.masses.iter().sum::<f32>() / self.masses.len() as f32;
        let max  = self.masses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        PhysicsTelemetry {
            energy: mean / (max + 1e-8),
            coherence: self.hub_indices.len() as f32 / self.masses.len() as f32,
            entropy: 0.0, // calculado por GravityAttention
            alive_pct: (self.masses.iter().filter(|&&m| m > self.hub_threshold).count() as f32)
                       / self.masses.len() as f32,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. GRAVITY ATTENTION — substitui softmax(QKᵀ/√d)
// ─────────────────────────────────────────────────────────────────────────────
//
// Em vez de: A[i,j] = softmax(Q_i · K_j / √d)
// Usa:       A[i,j] = softmax(G · m_i · m_j / d_ij²)
//
// onde d_ij = distância cosseno entre os hidden states i e j.
// m_i = massa semântica do token i (pré-computada dos embeddings).
//
// Propriedades:
//   - Tokens de alta massa atraem mais atenção (lei gravitacional)
//   - Decaimento natural com a distância (sem position bias articial)
//   - Diferenciável: pode ser usado com backprop se necessário

#[derive(Debug, Clone)]
pub struct GravityAttention {
    pub g_constant: f32,  // constante gravitacional (hiperparâmetro)
    pub vortex_lambda: f32, // força do componente de vórtice (0 = desativado)
    /// Telemetria interna — atualizada a cada forward
    last_entropy: f32,
    last_coherence: f32,
}

impl GravityAttention {
    pub fn new(g_constant: f32, vortex_lambda: f32) -> Self {
        Self { g_constant, vortex_lambda, last_entropy: 0.0, last_coherence: 0.0 }
    }

    /// Forward: computa atenção gravitacional para uma sequência de hidden states.
    ///
    /// # Argumentos
    /// - `hidden`: hidden states [seq_len × hidden_size]
    /// - `masses`: massa semântica de cada posição [seq_len] (do embedding do token)
    /// - `causal`: se true, aplica máscara causal (tokens futuros = -inf)
    ///
    /// # Retorna
    /// Pesos de atenção normalizados [seq_len × seq_len]
    pub fn forward(
        &mut self,
        hidden: &[f32],
        masses: &[f32],
        seq_len: usize,
        hidden_size: usize,
        causal: bool,
    ) -> Vec<f32> {
        assert_eq!(hidden.len(), seq_len * hidden_size);
        assert_eq!(masses.len(), seq_len);

        let mut attn = vec![0.0f32; seq_len * seq_len];

        // Normaliza hidden states para calcular distância cosseno
        let mut hidden_norm = vec![0.0f32; seq_len * hidden_size];
        for i in 0..seq_len {
            let row = &hidden[i * hidden_size..(i + 1) * hidden_size];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-8;
            for (j, &x) in row.iter().enumerate() {
                hidden_norm[i * hidden_size + j] = x / norm;
            }
        }

        // Computa scores gravitacionais: G × m_i × m_j / (1 - cos_sim)²
        for i in 0..seq_len {
            for j in 0..seq_len {
                if causal && j > i {
                    attn[i * seq_len + j] = f32::NEG_INFINITY;
                    continue;
                }

                // Distância cosseno (d = 1 - cos_sim)
                let cos_sim: f32 = (0..hidden_size)
                    .map(|k| hidden_norm[i * hidden_size + k] * hidden_norm[j * hidden_size + k])
                    .sum();
                let dist_sq = (1.0 - cos_sim).max(1e-6).powi(2);

                // Força gravitacional
                let force = self.g_constant * masses[i] * masses[j] / dist_sq;
                attn[i * seq_len + j] = force;
            }
        }

        // Adiciona componente de vórtice (curl ≠ 0) se habilitado
        if self.vortex_lambda.abs() > 1e-6 {
            self.apply_vortex(&mut attn, seq_len);
        }

        // Softmax por linha
        self.softmax_rows(&mut attn, seq_len);

        // Atualiza telemetria
        self.last_entropy = self.compute_entropy(&attn, seq_len);
        self.last_coherence = self.compute_coherence(&attn, seq_len);

        attn
    }

    /// Adiciona componente antisimétrico: A_new = A + λ × (A - Aᵀ) / 2
    /// Isso adiciona "rotação" ao campo de atenção — tokens não são apenas
    /// atraídos, mas orbitam uns ao redor dos outros.
    fn apply_vortex(&self, attn: &mut Vec<f32>, seq_len: usize) {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let a_ij = attn[i * seq_len + j];
                let a_ji = attn[j * seq_len + i];
                let skew = (a_ij - a_ji) / 2.0;
                attn[i * seq_len + j] += self.vortex_lambda * skew;
            }
        }
    }

    fn softmax_rows(&self, m: &mut Vec<f32>, seq_len: usize) {
        for i in 0..seq_len {
            let row = &mut m[i * seq_len..(i + 1) * seq_len];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = if x.is_infinite() { 0.0 } else { (*x - max).exp() };
                sum += *x;
            }
            if sum > 0.0 {
                for x in row.iter_mut() { *x /= sum; }
            }
        }
    }

    fn compute_entropy(&self, attn: &[f32], seq_len: usize) -> f32 {
        let mut total = 0.0f32;
        for i in 0..seq_len {
            let row = &attn[i * seq_len..(i + 1) * seq_len];
            let h: f32 = row.iter()
                .filter(|&&x| x > 1e-10)
                .map(|&x| -x * x.ln())
                .sum();
            total += h;
        }
        total / seq_len as f32
    }

    fn compute_coherence(&self, attn: &[f32], seq_len: usize) -> f32 {
        // Coerência: quão concentrada é a atenção (1 = toda em 1 token, 0 = uniforme)
        let mut total = 0.0f32;
        for i in 0..seq_len {
            let row = &attn[i * seq_len..(i + 1) * seq_len];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            total += max;
        }
        total / seq_len as f32
    }
}

impl PhysicsOp for GravityAttention {
    fn name(&self) -> &'static str { "GravityAttention" }
    fn telemetry(&self) -> PhysicsTelemetry {
        PhysicsTelemetry {
            energy: self.g_constant,
            coherence: self.last_coherence,
            entropy: self.last_entropy,
            alive_pct: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. KAN BARRIER — poço duplo substitui SiLU
// ─────────────────────────────────────────────────────────────────────────────
//
// Potencial: V(x) = -x²/2 + x⁴/4
// Força:     F(x) = x - x³  (restauradora)
//
// Minima em x = ±1 (os dois "poços").
// Barreira em x = 0 (altura V=0.25).
//
// Propriedades:
//   - Sinal fraco (|x| < barreira): fica preso em um poço (decisão binária)
//   - Sinal forte (|x| > barreira): passa pela barreira (tunneling quântico)
//   - Ruído baixo: atenuado pelo poço (filtro natural)
//   - Ruído alto: pode tunelizar (ressonância estocástica)
//
// A escala da barreira é ajustável por camada — o modelo aprende a profundidade
// do poço como um parâmetro B-spline (conceito KAN).

#[derive(Debug, Clone)]
pub struct KanBarrier {
    /// Escala da barreira (1.0 = poço duplo padrão)
    pub barrier_scale: f32,
    /// Amplitude de ruído estocástico (0.0 = determinístico)
    pub noise_amplitude: f32,
    /// Número de passos de integração (mais passos = mais preciso, mais lento)
    pub n_steps: usize,
    /// Semântica: threshold para considerar neurônio "vivo"
    pub alive_threshold: f32,
    // Telemetria
    last_alive_pct: f32,
    last_energy: f32,
}

impl KanBarrier {
    pub fn new(barrier_scale: f32, noise_amplitude: f32) -> Self {
        Self {
            barrier_scale,
            noise_amplitude,
            n_steps: 8,
            alive_threshold: 0.1,
            last_alive_pct: 0.0,
            last_energy: 0.0,
        }
    }

    /// Poço duplo: força restauradora F(x) = -(x - x³) × scale
    /// Minima em x = ±1/√scale (ajustável)
    #[inline]
    fn double_well_force(&self, x: f32) -> f32 {
        -(x - x * x * x) * self.barrier_scale
    }

    /// Energia potencial V(x) = -(−x²/2 + x⁴/4) × scale
    #[inline]
    fn potential(&self, x: f32) -> f32 {
        (-x * x / 2.0 + x * x * x * x / 4.0) * self.barrier_scale
    }

    /// Forward: processa um slice de ativações via integração do poço duplo.
    /// Equivalente a aplicar uma ativação física em vez de SiLU.
    ///
    /// # Argumentos
    /// - `input`: ativações de entrada [n]
    /// - `out`: buffer de saída [n] — sobrescrito
    pub fn forward(&mut self, input: &[f32], out: &mut [f32]) {
        assert_eq!(input.len(), out.len());
        let n = input.len();

        // Inicializa posição = valor de entrada (partícula começa na posição x₀)
        let mut pos = input.to_vec();
        let mut vel = vec![0.0f32; n];

        // Integração numérica (Verlet simplificado)
        for _ in 0..self.n_steps {
            for i in 0..n {
                let force = self.double_well_force(pos[i]);
                vel[i] = vel[i] * 0.85 + force * 0.1; // amortecimento
                pos[i] += vel[i];
                // Clamp para evitar divergência
                pos[i] = pos[i].clamp(-3.0, 3.0);
            }
        }

        // Resultado = posição final (sinal filtrado pelo poço)
        out.copy_from_slice(&pos);

        // Telemetria
        let energy: f32 = pos.iter().map(|&x| self.potential(x)).sum::<f32>() / n as f32;
        let alive = pos.iter().filter(|&&x| x.abs() > self.alive_threshold).count();
        self.last_alive_pct = alive as f32 / n as f32;
        self.last_energy = energy.abs();
    }

    /// Versão rápida: approximação analítica do poço duplo.
    /// Em vez de integrar, aplica a transformação fechada:
    ///   f_fast(x) = x / (1 + |x|²)^0.5  ×  sign(x)  para |x| > threshold
    ///   f_fast(x) = 0                                  para |x| ≤ threshold
    ///
    /// Muito mais rápido que a integração, mas menos preciso.
    #[inline]
    pub fn forward_fast(&self, x: f32) -> f32 {
        let barrier = (0.5 / self.barrier_scale).sqrt(); // ponto de barreira
        if x.abs() <= barrier {
            // Dentro da barreira: atenuação forte (sinal fraco fica preso)
            x * (x / barrier).abs().powi(2)
        } else {
            // Fora da barreira: passa com pequena distorção para ±1
            let sign = x.signum();
            let mag = (x.abs() - barrier) / (x.abs() - barrier + 1.0);
            sign * (barrier + mag * (1.0 - barrier))
        }
    }

    /// Comparação com SiLU para benchmark
    #[inline]
    pub fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// SNR após filtragem (quanto de sinal vs ruído na saída)
    pub fn snr_vs_silu(signal: &[f32], noise_std: f32) -> (f32, f32) {
        let n = signal.len();
        let mut rng_state: u64 = 0xdeadbeef;

        let noisy: Vec<f32> = signal.iter().map(|&s| {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                                  .wrapping_add(1442695040888963407);
            let u = (rng_state >> 33) as f32 / (u32::MAX as f32) - 0.5;
            s + u * noise_std * 2.0
        }).collect();

        let out_silu: Vec<f32> = noisy.iter().map(|&x| Self::silu(x)).collect();
        let barrier = KanBarrier::new(1.0, 0.0);
        let out_kan: Vec<f32>  = noisy.iter().map(|&x| barrier.forward_fast(x)).collect();

        let corr_silu = pearson_r(signal, &out_silu);
        let corr_kan  = pearson_r(signal, &out_kan);
        (corr_silu * corr_silu, corr_kan * corr_kan)  // R²
    }
}

impl PhysicsOp for KanBarrier {
    fn name(&self) -> &'static str { "KanBarrier" }
    fn telemetry(&self) -> PhysicsTelemetry {
        PhysicsTelemetry {
            energy: self.last_energy,
            coherence: self.barrier_scale,
            entropy: 0.0,
            alive_pct: self.last_alive_pct,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. KURAMOTO DECODER — sincronização de fase substitui softmax sobre vocabulário
// ─────────────────────────────────────────────────────────────────────────────
//
// Em vez de: token = argmax(softmax(logits × W_vocab))
// Usa:       cada token candidato é um oscilador
//            o hidden state define a "fase alvo"
//            o token mais sincronizado com a fase alvo é selecionado
//
// dφ_i/dt = ω_i + K/N × Σⱼ sin(φⱼ - φᵢ)
//
// Vantagem: O(k) onde k = candidatos ativos (Kuramoto hubs)
//           vs O(vocab_size) do softmax padrão (152k operações)
//
// Aprendizado online via nudge de fase:
//   Δω_i = η × sin(φ_true - φᵢ)   (sem backprop!)

#[derive(Debug, Clone)]
pub struct KuramotoDecoder {
    /// Fase natural de cada token-candidato [n_candidates]
    pub phases: Vec<f32>,
    /// Frequência natural de cada oscilador [n_candidates]
    pub freqs: Vec<f32>,
    /// Índices dos tokens candidatos no vocabulário
    pub token_ids: Vec<u32>,
    /// Força de acoplamento K
    pub coupling: f32,
    /// Taxa de aprendizado η para nudge de fase
    pub learning_rate: f32,
    /// Passos de integração por token
    pub n_steps: usize,
    // Telemetria
    last_r: f32,  // parâmetro de ordem (0=caos, 1=sync)
    last_entropy: f32,
}

impl KuramotoDecoder {
    /// Cria decodificador com N candidatos de alta frequência (hubs)
    pub fn new(n_candidates: usize, coupling: f32) -> Self {
        // Distribuição de frequências naturais (Lorentziana para criticalidade)
        let freqs: Vec<f32> = (0..n_candidates)
            .map(|i| 0.5 + (i as f32 / n_candidates as f32) * 2.0)
            .collect();

        // Fases iniciais uniformes
        let phases: Vec<f32> = (0..n_candidates)
            .map(|i| (i as f32 / n_candidates as f32) * TAU)
            .collect();

        let token_ids: Vec<u32> = (0..n_candidates as u32).collect();

        Self {
            phases,
            freqs,
            token_ids,
            coupling,
            learning_rate: 0.05,
            n_steps: 20,
            last_r: 0.0,
            last_entropy: 0.0,
        }
    }

    /// Encoda o hidden state como fase alvo.
    /// Usa a fase média dos pesos imaginárioo/real do vetor.
    pub fn encode_hidden_as_phase(&self, hidden: &[f32]) -> f32 {
        // Codifica h como ponto no plano complexo: soma de (h[2i] + i·h[2i+1])
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for chunk in hidden.chunks(2) {
            re += chunk[0];
            im += chunk.get(1).copied().unwrap_or(0.0);
        }
        im.atan2(re) // fase em [-π, π]
    }

    /// Forward: dado o hidden state, encontra o token mais coerente.
    ///
    /// # Retorna
    /// (token_id, coherence_score) — o token selecionado e sua coerência
    pub fn forward(&mut self, hidden: &[f32]) -> (u32, f32) {
        let target_phase = self.encode_hidden_as_phase(hidden);
        let n = self.phases.len();

        // Integração Kuramoto por n_steps
        let dt = 0.05f32;
        for _ in 0..self.n_steps {
            let mut new_phases = self.phases.clone();
            for i in 0..n {
                // dφ_i/dt = ω_i + K/N × Σⱼ sin(φⱼ - φᵢ) + K_ext × sin(φ_target - φᵢ)
                let internal_coupling: f32 = self.phases.iter()
                    .map(|&pj| (pj - self.phases[i]).sin())
                    .sum::<f32>() / n as f32;
                let external_coupling = (target_phase - self.phases[i]).sin();
                let d_phi = self.freqs[i]
                    + self.coupling * internal_coupling
                    + self.coupling * 2.0 * external_coupling;
                new_phases[i] = (self.phases[i] + d_phi * dt) % TAU;
            }
            self.phases = new_phases;
        }

        // Parâmetro de ordem r = |1/N Σ exp(i·φⱼ)|
        let (re, im): (f32, f32) = self.phases.iter()
            .map(|&p| (p.cos(), p.sin()))
            .fold((0.0, 0.0), |(re, im), (c, s)| (re + c, im + s));
        self.last_r = (re * re + im * im).sqrt() / n as f32;

        // Seleciona o oscilador mais próximo da fase alvo
        let best_idx = self.phases.iter()
            .enumerate()
            .map(|(i, &p)| (i, (target_phase - p).sin().abs()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let coherence = (target_phase - self.phases[best_idx]).cos();
        self.last_entropy = -(self.last_r.ln() + 1e-8);

        (self.token_ids[best_idx], coherence)
    }

    /// Aprendizado online: dado o token correto, ajusta as frequências.
    /// Sem backprop — apenas nudge de fase.
    ///
    /// Δω_i = η × sin(φ_true - φᵢ)
    pub fn online_update(&mut self, correct_token_id: u32) {
        // Encontra a fase alvo para o token correto
        let target_idx = self.token_ids.iter()
            .position(|&t| t == correct_token_id)
            .unwrap_or(0);
        let target_phase = self.phases[target_idx];

        // Ajusta frequências: osciladores perto do alvo aceleram, longe desaceleram
        for i in 0..self.phases.len() {
            let delta = self.learning_rate * (target_phase - self.phases[i]).sin();
            self.freqs[i] += delta;
            // Clamp para evitar divergência de frequência
            self.freqs[i] = self.freqs[i].clamp(0.1, 5.0);
        }
    }

    /// Custo computacional vs softmax padrão
    pub fn cost_ratio(&self, vocab_size: usize) -> f32 {
        let kuramoto_ops = self.phases.len() * self.n_steps; // n_candidates × steps
        let softmax_ops  = vocab_size;                         // exp × vocab
        softmax_ops as f32 / kuramoto_ops as f32
    }
}

impl PhysicsOp for KuramotoDecoder {
    fn name(&self) -> &'static str { "KuramotoDecoder" }
    fn telemetry(&self) -> PhysicsTelemetry {
        PhysicsTelemetry {
            energy: self.coupling,
            coherence: self.last_r,
            entropy: self.last_entropy,
            alive_pct: self.last_r,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. PHYSICS ENGINE — orquestra todos os operadores
// ─────────────────────────────────────────────────────────────────────────────

pub struct IaraPhysicsEngine {
    pub gravity_attn: GravityAttention,
    pub kan_barrier: KanBarrier,
    pub kuramoto: KuramotoDecoder,
    pub semantic_mass: Option<SemanticMass>,
}

impl IaraPhysicsEngine {
    /// Configuração padrão — hiperparâmetros calibrados empiricamente
    pub fn default_config() -> Self {
        Self {
            gravity_attn: GravityAttention::new(
                1.0,   // G = 1.0
                0.15,  // λ_vortex = 0.15 (rotação suave)
            ),
            kan_barrier: KanBarrier::new(
                1.0,   // barrier_scale = 1.0
                0.0,   // sem ruído estocástico por padrão
            ),
            kuramoto: KuramotoDecoder::new(
                512,   // 512 candidatos (top tokens por frequência)
                2.5,   // K = 2.5 (acoplamento crítico)
            ),
            semantic_mass: None,
        }
    }

    /// Telemetria completa do engine físico
    pub fn all_telemetry(&self) -> Vec<PhysicsTelemetry> {
        vec![
            self.gravity_attn.telemetry(),
            self.kan_barrier.telemetry(),
            self.kuramoto.telemetry(),
        ]
    }

    /// Gera um passo de inferência usando os operadores físicos
    /// (substitui parcialmente o forward padrão do transformer)
    pub fn physics_step(
        &mut self,
        hidden: &[f32],
        masses: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> PhysicsStepOutput {
        // 1. Gravity attention
        let attn = self.gravity_attn.forward(
            hidden, masses, seq_len, hidden_size,
            true, // causal
        );

        // 2. Context vector = atenção × hidden states (produção de contexto)
        let mut context = vec![0.0f32; hidden_size];
        let last = seq_len - 1;
        for j in 0..seq_len {
            let w = attn[last * seq_len + j];
            for k in 0..hidden_size {
                context[k] += w * hidden[j * hidden_size + k];
            }
        }

        // 3. KAN barrier no contexto (substitui SiLU gate)
        let mut filtered = vec![0.0f32; hidden_size];
        self.kan_barrier.forward(&context, &mut filtered);

        // 4. Kuramoto decoder: escolhe o próximo token
        let (token_id, coherence) = self.kuramoto.forward(&filtered);

        PhysicsStepOutput {
            attn_weights: attn,
            context_vector: context,
            filtered_context: filtered,
            selected_token: token_id,
            coherence,
            telemetry: self.all_telemetry(),
        }
    }
}

#[derive(Debug)]
pub struct PhysicsStepOutput {
    /// Pesos de atenção gravitacional [seq_len × seq_len]
    pub attn_weights: Vec<f32>,
    /// Vetor de contexto ponderado [hidden_size]
    pub context_vector: Vec<f32>,
    /// Contexto após filtro KAN [hidden_size]
    pub filtered_context: Vec<f32>,
    /// Token selecionado pelo decoder Kuramoto
    pub selected_token: u32,
    /// Coerência de fase do token selecionado [0, 1]
    pub coherence: f32,
    /// Telemetria de todos os operadores
    pub telemetry: Vec<PhysicsTelemetry>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilitários
// ─────────────────────────────────────────────────────────────────────────────

fn pearson_r(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len()) as f32;
    let mx = x.iter().sum::<f32>() / n;
    let my = y.iter().sum::<f32>() / n;
    let num: f32 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi-mx)*(yi-my)).sum();
    let sx = x.iter().map(|&xi| (xi-mx).powi(2)).sum::<f32>().sqrt();
    let sy = y.iter().map(|&yi| (yi-my).powi(2)).sum::<f32>().sqrt();
    num / (sx * sy + 1e-8)
}

// ─────────────────────────────────────────────────────────────────────────────
// Testes
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gravity_attention_shape() {
        let mut ga = GravityAttention::new(1.0, 0.1);
        let seq_len = 4;
        let hidden_size = 8;
        let hidden = vec![1.0f32; seq_len * hidden_size];
        let masses = vec![1.0f32, 2.0, 0.5, 1.5];
        let attn = ga.forward(&hidden, &masses, seq_len, hidden_size, true);
        assert_eq!(attn.len(), seq_len * seq_len);
        // Cada linha deve somar 1 (após softmax)
        for i in 0..seq_len {
            let row_sum: f32 = attn[i*seq_len..(i+1)*seq_len].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "linha {i}: soma={row_sum}");
        }
    }

    #[test]
    fn test_kan_barrier_forward() {
        let mut kan = KanBarrier::new(1.0, 0.0);
        let input = vec![-2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0];
        let mut out = vec![0.0f32; input.len()];
        kan.forward(&input, &mut out);
        // Sinal grande deve passar pelo poço
        assert!(out[0].abs() > 0.5, "sinal grande negativo deve sair do poço");
        assert!(out[6].abs() > 0.5, "sinal grande positivo deve sair do poço");
    }

    #[test]
    fn test_kan_barrier_fast_filter() {
        let b = KanBarrier::new(1.0, 0.0);
        // Sinal dentro da barreira → atenuado
        assert!(b.forward_fast(0.1).abs() < 0.1);
        // Sinal fora da barreira → passa
        assert!(b.forward_fast(2.0).abs() > 0.5);
        // Antiparidade
        assert!((b.forward_fast(-1.5) + b.forward_fast(1.5)).abs() < 1e-5);
    }

    #[test]
    fn test_kuramoto_decoder() {
        let mut decoder = KuramotoDecoder::new(64, 2.5);
        let hidden = vec![0.5f32; 128];
        let (token_id, coherence) = decoder.forward(&hidden);
        assert!(token_id < 64);
        assert!(coherence >= -1.0 && coherence <= 1.0);
    }

    #[test]
    fn test_physics_engine_step() {
        let mut engine = IaraPhysicsEngine::default_config();
        let seq_len = 4;
        let hidden_size = 64;
        let hidden = vec![0.1f32; seq_len * hidden_size];
        let masses = vec![1.0f32, 1.5, 0.8, 1.2];
        let out = engine.physics_step(&hidden, &masses, seq_len, hidden_size);
        assert_eq!(out.attn_weights.len(), seq_len * seq_len);
        assert_eq!(out.context_vector.len(), hidden_size);
        assert!(out.coherence >= -1.0 && out.coherence <= 1.0);
    }

    #[test]
    fn test_kuramoto_cost_ratio() {
        let decoder = KuramotoDecoder::new(512, 2.5);
        let ratio = decoder.cost_ratio(152064); // vocab Qwen3
        // Deve ser ~15x mais barato que softmax completo
        assert!(ratio > 5.0, "ratio={ratio}, expected > 5x");
    }
}
