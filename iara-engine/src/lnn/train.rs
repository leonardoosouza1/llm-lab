// ============================================================================
// lnn/train.rs — Treinamento Online do Sistema Líquido (ESN + CfC)
// ============================================================================
//
// Fase 3: Treinamento online por reinforcement signal.
//
// Arquitetura de treinamento:
//   1. Qwen gera N tokens → coleta telemetria (entropy, gravity, CfC state)
//   2. Computa reward signal da qualidade da geração
//   3. Treina ESN readout (W_out) via ridge regression (CPU, ~256 pesos)
//   4. Treina CfC weights via backprop manual + SGD (CPU, ~6M pesos)
//   5. Upload pesos atualizados para GPU
//   6. Próxima geração usa pesos melhorados
//
// Tudo em CPU — baixo custo, não bloqueia inferência GPU.
// ============================================================================

use std::sync::Arc;
use compute_core::ComputeBackend;
use serde::Serialize;

use super::esn::EsnObserver;
use super::cfc::{CfcLayer, CfcConfig};

// ── Constantes de treinamento ──────────────────────────────────────────────

/// Regularização L2 para ridge regression do ESN
const ESN_RIDGE_LAMBDA: f32 = 1e-4;

/// Learning rate base para CfC SGD
const CFC_LR_BASE: f32 = 1e-4;

/// Momentum para CfC SGD
const CFC_MOMENTUM: f32 = 0.9;

/// Clip de gradientes (max norm)
const CFC_GRAD_CLIP: f32 = 1.0;

/// Mínimo de steps para treinar (precisa de dados suficientes)
const MIN_STEPS_FOR_TRAINING: usize = 8;

// ── Reward Signal ──────────────────────────────────────────────────────────

/// Sinal de recompensa computado a partir da telemetria de geração.
///
/// Reward alto = geração de qualidade (confiante, fluente, sem repetição).
/// Reward baixo = geração ruim (entropia alta, repetitiva, incoerente).
#[derive(Debug, Clone, Serialize)]
pub struct RewardSignal {
    /// Reward por token [N_steps] — usado como target para ESN
    pub per_token: Vec<f32>,
    /// Reward médio da geração inteira
    pub mean_reward: f32,
    /// Entropia média (quanto menor, mais confiante)
    pub mean_entropy: f32,
    /// Diversidade léxica (tokens únicos / total)
    pub lexical_diversity: f32,
    /// Pontuação de repetição (0 = sem repetição, 1 = tudo repetido)
    pub repetition_score: f32,
}

impl RewardSignal {
    /// Computa reward signal a partir do histórico de observações do ESN.
    ///
    /// Recompensa:
    ///   - Baixa entropia → modelo confiante → reward alto
    ///   - Entropia decrescente → modelo está convergindo → reward extra
    ///   - Repetição → penalidade
    ///   - Diversidade léxica → bônus
    pub fn compute(
        obs_history: &[[f32; 3]],
        generated_tokens: &[u32],
    ) -> Self {
        let n = obs_history.len();
        if n == 0 {
            return Self {
                per_token: vec![],
                mean_reward: 0.0,
                mean_entropy: 0.0,
                lexical_diversity: 0.0,
                repetition_score: 0.0,
            };
        }

        // Entropia média
        let mean_entropy = obs_history.iter().map(|o| o[0]).sum::<f32>() / n as f32;

        // Reward por token: baseado na confiança (top1_prob) e redução de entropia
        let mut per_token = Vec::with_capacity(n);
        for i in 0..n {
            let confidence = obs_history[i][1]; // top1_prob ∈ [0,1]
            let entropy = obs_history[i][0];

            // Componente 1: confiança do modelo (reward proporcional)
            let conf_reward = confidence;

            // Componente 2: entropia baixa é bom
            // Normaliza: entropy típica de LLM ∈ [0, 12], mapeia para [1, 0]
            let ent_reward = (1.0 - entropy / 12.0).clamp(0.0, 1.0);

            // Componente 3: entropia decrescente (convergência)
            let trend_reward = if i > 0 {
                let delta = obs_history[i - 1][0] - entropy; // positivo = entropia caiu
                (delta * 0.5).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            per_token.push(conf_reward * 0.4 + ent_reward * 0.4 + trend_reward * 0.2);
        }

        // Diversidade léxica
        let unique_tokens: std::collections::HashSet<u32> =
            generated_tokens.iter().copied().collect();
        let lexical_diversity = if generated_tokens.is_empty() {
            0.0
        } else {
            unique_tokens.len() as f32 / generated_tokens.len() as f32
        };

        // Repetição: conta n-gramas repetidos (bigrams)
        let mut bigram_count = std::collections::HashMap::new();
        for pair in generated_tokens.windows(2) {
            *bigram_count.entry((pair[0], pair[1])).or_insert(0u32) += 1;
        }
        let total_bigrams = generated_tokens.len().saturating_sub(1).max(1) as f32;
        let repeated = bigram_count.values().filter(|&&c| c > 1).count() as f32;
        let repetition_score = (repeated / total_bigrams).clamp(0.0, 1.0);

        // Ajusta reward com diversidade e penalidade de repetição
        let diversity_bonus = lexical_diversity * 0.2;
        let repetition_penalty = repetition_score * 0.3;
        for r in per_token.iter_mut() {
            *r = (*r + diversity_bonus - repetition_penalty).clamp(-1.0, 1.0);
        }

        let mean_reward = per_token.iter().sum::<f32>() / n as f32;

        Self {
            per_token,
            mean_reward,
            mean_entropy,
            lexical_diversity,
            repetition_score,
        }
    }
}

// ── ESN Readout Training (Ridge Regression) ────────────────────────────────

/// Treina o readout W_out do ESN via ridge regression.
///
/// Minimiza ||H · w - y||² + λ||w||²
/// Solução: w = (H^T H + λI)^{-1} H^T y
///
/// H: [N × R] — histórico de estados do reservatório
/// y: [N] — reward signal por token
/// w: [R] — pesos do readout (saída)
///
/// Retorna os novos pesos W_out ou None se dados insuficientes.
pub fn train_esn_readout(esn: &EsnObserver, reward: &RewardSignal) -> Option<Vec<f32>> {
    let n = esn.state_history.len();
    let targets = &reward.per_token;

    if n < MIN_STEPS_FOR_TRAINING || targets.len() < n {
        log::debug!("ESN training: dados insuficientes ({} steps, min {})", n, MIN_STEPS_FOR_TRAINING);
        return None;
    }

    let r = esn.state[0..].len(); // R = 256

    // H^T H: [R × R]
    let mut hth = vec![0.0f32; r * r];
    for t in 0..n {
        let h = &esn.state_history[t];
        for i in 0..r {
            for j in 0..r {
                hth[i * r + j] += h[i] * h[j];
            }
        }
    }

    // Regularização: H^T H + λI
    for i in 0..r {
        hth[i * r + i] += ESN_RIDGE_LAMBDA;
    }

    // H^T y: [R]
    let mut hty = vec![0.0f32; r];
    for t in 0..n {
        let h = &esn.state_history[t];
        let y = targets[t];
        for i in 0..r {
            hty[i] += h[i] * y;
        }
    }

    // Resolve (H^T H + λI) w = H^T y via Cholesky
    // Tenta com regularização crescente se a matriz for singular
    let lambdas = [ESN_RIDGE_LAMBDA, 1e-3, 1e-2, 1e-1];
    let mut result: Option<Vec<f32>> = None;

    for (attempt, &lambda) in lambdas.iter().enumerate() {
        // Na primeira tentativa já temos λ aplicado; nas demais, incrementamos a diagonal
        if attempt > 0 {
            let extra = lambda - lambdas[attempt - 1];
            for i in 0..r {
                hth[i * r + i] += extra;
            }
            log::debug!("ESN Cholesky retry #{} com λ={:.0e}", attempt, lambda);
        }

        if let Some(w) = solve_positive_definite(&hth, &hty, r) {
            log::info!(
                "ESN readout treinado: {} steps, mean_reward={:.4}, ||w||={:.4}, λ={:.0e}",
                n,
                reward.mean_reward,
                w.iter().map(|x| x * x).sum::<f32>().sqrt(),
                lambda,
            );
            result = Some(w);
            break;
        }
    }

    if result.is_none() {
        log::warn!("ESN ridge regression falhou após {} tentativas (reservatório degenerado)", lambdas.len());
    }
    result
}

/// Resolve Ax = b via Cholesky (A simétrica positiva-definida).
/// A: [n × n] row-major, b: [n], retorna x: [n].
fn solve_positive_definite(a: &[f32], b: &[f32], n: usize) -> Option<Vec<f32>> {
    // Cholesky: A = L L^T
    let mut l = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    return None; // Não é positiva definida
                }
                l[i * n + j] = diag.sqrt();
            } else {
                let denom = l[j * n + j];
                if denom.abs() < 1e-12 {
                    return None;
                }
                l[i * n + j] = (a[i * n + j] - sum) / denom;
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..i {
            sum += l[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }

    // Backward substitution: L^T x = y
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = 0.0f32;
        for j in (i + 1)..n {
            sum += l[j * n + i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }

    Some(x)
}

// ── CfC Backprop (CPU-side gradient descent) ───────────────────────────────

/// Estado de treinamento do CfC — mantém momentum e estatísticas.
pub struct CfcTrainer {
    /// Learning rate atual (pode ser ajustado por schedule)
    pub lr: f32,
    /// Momentum buffers para cada grupo de pesos
    pub momentum: CfcGradients,
    /// Histórico de loss para monitoramento
    pub loss_history: Vec<f32>,
    /// Epoch counter
    pub epoch: usize,
    /// Acumulador de ativações para backward pass
    activations: Vec<CfcActivationSnapshot>,
}

/// Snapshot das ativações de um step CfC (para backprop).
#[derive(Clone)]
struct CfcActivationSnapshot {
    /// Input x [hidden_dim]
    x: Vec<f32>,
    /// Estado anterior h_prev [cfc_dim]
    h_prev: Vec<f32>,
    /// Gate f = sigmoid(f_proj) [cfc_dim]
    f_gate: Vec<f32>,
    /// Candidate g = tanh(g_proj) [cfc_dim]
    g_candidate: Vec<f32>,
    /// Estado resultante h_new [cfc_dim]
    h_new: Vec<f32>,
    /// Correction output [hidden_dim]
    correction: Vec<f32>,
    /// Target: reward signal para este step
    reward: f32,
}

/// Gradientes para todos os pesos do CfC (mesma estrutura que CfcWeights, mas CPU).
pub struct CfcGradients {
    pub wf: Vec<f32>,   // [cd × hd]
    pub wf2: Vec<f32>,  // [cd × cd]
    pub bf: Vec<f32>,   // [cd]
    pub wg: Vec<f32>,   // [cd × hd]
    pub wg2: Vec<f32>,  // [cd × cd]
    pub bg: Vec<f32>,   // [cd]
    pub wout: Vec<f32>, // [hd × cd]
    pub bout: Vec<f32>, // [hd]
}

impl CfcGradients {
    fn zeros(cfc_dim: usize, hidden_dim: usize) -> Self {
        let cd = cfc_dim;
        let hd = hidden_dim;
        Self {
            wf: vec![0.0; cd * hd],
            wf2: vec![0.0; cd * cd],
            bf: vec![0.0; cd],
            wg: vec![0.0; cd * hd],
            wg2: vec![0.0; cd * cd],
            bg: vec![0.0; cd],
            wout: vec![0.0; hd * cd],
            bout: vec![0.0; hd],
        }
    }

    fn clip_norm(&mut self, max_norm: f32) {
        let all_grads: Vec<&mut Vec<f32>> = vec![
            &mut self.wf, &mut self.wf2, &mut self.bf,
            &mut self.wg, &mut self.wg2, &mut self.bg,
            &mut self.wout, &mut self.bout,
        ];

        // Compute total gradient norm
        let total_sq: f32 = all_grads.iter()
            .map(|g| g.iter().map(|x| x * x).sum::<f32>())
            .sum();
        let total_norm = total_sq.sqrt();

        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for g in vec![
                &mut self.wf, &mut self.wf2, &mut self.bf,
                &mut self.wg, &mut self.wg2, &mut self.bg,
                &mut self.wout, &mut self.bout,
            ] {
                for v in g.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }

    fn grad_norm(&self) -> f32 {
        let total: f32 = [
            &self.wf, &self.wf2, &self.bf,
            &self.wg, &self.wg2, &self.bg,
            &self.wout, &self.bout,
        ].iter().map(|g| g.iter().map(|x| x * x).sum::<f32>()).sum();
        total.sqrt()
    }
}

impl CfcTrainer {
    /// Cria um novo trainer para CfC.
    pub fn new(config: &CfcConfig) -> Self {
        Self {
            lr: config.lr.max(CFC_LR_BASE),
            momentum: CfcGradients::zeros(config.cfc_dim, config.hidden_dim),
            loss_history: Vec::new(),
            epoch: 0,
            activations: Vec::new(),
        }
    }

    /// Registra ativações de um forward pass para uso no backward.
    ///
    /// Chamado durante a geração, APÓS cfc.forward().
    /// Requer download dos buffers GPU → CPU.
    pub fn record_activation(
        &mut self,
        x: Vec<f32>,
        h_prev: Vec<f32>,
        h_new: Vec<f32>,
        correction: Vec<f32>,
        f_gate: Vec<f32>,
        g_candidate: Vec<f32>,
        reward: f32,
    ) {
        self.activations.push(CfcActivationSnapshot {
            x,
            h_prev,
            f_gate,
            g_candidate,
            h_new,
            correction,
            reward,
        });
    }

    /// Executa backward pass e atualiza pesos do CfC.
    ///
    /// Fluxo:
    ///   1. Download pesos atuais da GPU → CPU
    ///   2. Compute gradients via backprop manual
    ///   3. Aplica SGD + momentum
    ///   4. Upload pesos atualizados para GPU
    ///
    /// Loss: REINFORCE-style — maximize reward × log_likelihood
    /// Na prática: gradient da correction ponderado pelo reward signal.
    ///
    /// Retorna (loss, grad_norm) ou None se dados insuficientes.
    pub fn train_step(
        &mut self,
        cfc: &mut CfcLayer,
        backend: &Arc<dyn ComputeBackend>,
    ) -> Option<CfcTrainResult> {
        if self.activations.len() < MIN_STEPS_FOR_TRAINING {
            log::debug!(
                "CfC training: dados insuficientes ({} activations, min {})",
                self.activations.len(), MIN_STEPS_FOR_TRAINING
            );
            return None;
        }

        let cd = cfc.config.cfc_dim;
        let hd = cfc.config.hidden_dim;
        let n = self.activations.len();

        // Download pesos atuais da GPU
        let wf_cpu = backend.download_f32(&cfc.weights.wf);     // [cd, hd]
        let wf2_cpu = backend.download_f32(&cfc.weights.wf2);   // [cd, cd]
        let wg_cpu = backend.download_f32(&cfc.weights.wg);     // [cd, hd]
        let wg2_cpu = backend.download_f32(&cfc.weights.wg2);   // [cd, cd]
        let wout_cpu = backend.download_f32(&cfc.weights.wout); // [hd, cd]

        // Acumula gradientes sobre todos os steps
        let mut grads = CfcGradients::zeros(cd, hd);
        let mut total_loss = 0.0f32;

        for snap in &self.activations {
            // ── Forward (verificação, usando valores salvos) ──
            // f = sigmoid(x @ Wf^T + h_prev @ Wf2^T + bf)
            // g = tanh(x @ Wg^T + h_prev @ Wg2^T + bg)
            // h_new = f * h_prev + (1-f) * g
            // correction = h_new @ Wout^T + bout

            // Loss: -reward * ||correction||² (queremos maximizar reward,
            // quanto mais reward, mais queremos que a correction seja grande/útil)
            // Se reward é positivo: loss é negativo → gradientes amplificam correction
            // Se reward é negativo: loss é positivo → gradientes reduzem correction
            let corr_sq: f32 = snap.correction.iter().map(|c| c * c).sum();
            let loss = -snap.reward * corr_sq / hd as f32;
            total_loss += loss;

            // ── Backward through correction = h_new @ Wout^T + bout ──
            // dL/d(correction) = -2 * reward * correction / hd
            let d_correction: Vec<f32> = snap.correction.iter()
                .map(|c| -2.0 * snap.reward * c / hd as f32)
                .collect();

            // dL/d(Wout) = d_correction^T @ h_new
            // Wout: [hd, cd], d_correction: [hd], h_new: [cd]
            for i in 0..hd {
                for j in 0..cd {
                    grads.wout[i * cd + j] += d_correction[i] * snap.h_new[j];
                }
            }
            // dL/d(bout) = d_correction
            for i in 0..hd {
                grads.bout[i] += d_correction[i];
            }

            // dL/d(h_new) = d_correction @ Wout
            // [cd] = [hd] @ [hd, cd]
            let mut d_h_new = vec![0.0f32; cd];
            for j in 0..cd {
                let mut acc = 0.0f32;
                for i in 0..hd {
                    acc += d_correction[i] * wout_cpu[i * cd + j];
                }
                d_h_new[j] = acc;
            }

            // ── Backward through h_new = f * h_prev + (1-f) * g ──
            // dL/df = d_h_new * (h_prev - g)
            // dL/dg = d_h_new * (1-f)
            let d_f: Vec<f32> = (0..cd).map(|k| {
                d_h_new[k] * (snap.h_prev[k] - snap.g_candidate[k])
            }).collect();
            let d_g: Vec<f32> = (0..cd).map(|k| {
                d_h_new[k] * (1.0 - snap.f_gate[k])
            }).collect();

            // ── Backward through f = sigmoid(f_proj) ──
            // df/d(f_proj) = f * (1 - f)
            let d_f_proj: Vec<f32> = (0..cd).map(|k| {
                d_f[k] * snap.f_gate[k] * (1.0 - snap.f_gate[k])
            }).collect();

            // ── Backward through g = tanh(g_proj) ──
            // dg/d(g_proj) = 1 - g²
            let d_g_proj: Vec<f32> = (0..cd).map(|k| {
                d_g[k] * (1.0 - snap.g_candidate[k] * snap.g_candidate[k])
            }).collect();

            // ── Backward through f_proj = x @ Wf^T + h_prev @ Wf2^T + bf ──
            // dL/d(Wf) = d_f_proj^T @ x  → [cd, hd]
            for i in 0..cd {
                for j in 0..hd {
                    grads.wf[i * hd + j] += d_f_proj[i] * snap.x[j];
                }
            }
            // dL/d(Wf2) = d_f_proj^T @ h_prev → [cd, cd]
            for i in 0..cd {
                for j in 0..cd {
                    grads.wf2[i * cd + j] += d_f_proj[i] * snap.h_prev[j];
                }
            }
            // dL/d(bf) = d_f_proj
            for i in 0..cd {
                grads.bf[i] += d_f_proj[i];
            }

            // ── Backward through g_proj = x @ Wg^T + h_prev @ Wg2^T + bg ──
            for i in 0..cd {
                for j in 0..hd {
                    grads.wg[i * hd + j] += d_g_proj[i] * snap.x[j];
                }
            }
            for i in 0..cd {
                for j in 0..cd {
                    grads.wg2[i * cd + j] += d_g_proj[i] * snap.h_prev[j];
                }
            }
            for i in 0..cd {
                grads.bg[i] += d_g_proj[i];
            }
        }

        // Normaliza gradientes pelo número de steps
        let inv_n = 1.0 / n as f32;
        for g in [
            &mut grads.wf, &mut grads.wf2, &mut grads.bf,
            &mut grads.wg, &mut grads.wg2, &mut grads.bg,
            &mut grads.wout, &mut grads.bout,
        ] {
            for v in g.iter_mut() {
                *v *= inv_n;
            }
        }

        // Clip gradientes
        let pre_clip_norm = grads.grad_norm();
        grads.clip_norm(CFC_GRAD_CLIP);
        let post_clip_norm = grads.grad_norm();

        // SGD + Momentum: v = μ·v + grad; w = w - lr·v
        let lr = self.lr;
        let mu = CFC_MOMENTUM;

        fn apply_sgd(weight_cpu: &mut Vec<f32>, grad: &[f32], momentum: &mut Vec<f32>, lr: f32, mu: f32) {
            for i in 0..weight_cpu.len() {
                momentum[i] = mu * momentum[i] + grad[i];
                weight_cpu[i] -= lr * momentum[i];
            }
        }

        let mut wf_new = wf_cpu;
        let mut wf2_new = wf2_cpu;
        let mut bf_new = backend.download_f32(&cfc.weights.bf);
        let mut wg_new = wg_cpu;
        let mut wg2_new = wg2_cpu;
        let mut bg_new = backend.download_f32(&cfc.weights.bg);
        let mut wout_new = wout_cpu;
        let mut bout_new = backend.download_f32(&cfc.weights.bout);

        apply_sgd(&mut wf_new, &grads.wf, &mut self.momentum.wf, lr, mu);
        apply_sgd(&mut wf2_new, &grads.wf2, &mut self.momentum.wf2, lr, mu);
        apply_sgd(&mut bf_new, &grads.bf, &mut self.momentum.bf, lr, mu);
        apply_sgd(&mut wg_new, &grads.wg, &mut self.momentum.wg, lr, mu);
        apply_sgd(&mut wg2_new, &grads.wg2, &mut self.momentum.wg2, lr, mu);
        apply_sgd(&mut bg_new, &grads.bg, &mut self.momentum.bg, lr, mu);
        apply_sgd(&mut wout_new, &grads.wout, &mut self.momentum.wout, lr, mu);
        apply_sgd(&mut bout_new, &grads.bout, &mut self.momentum.bout, lr, mu);

        // Upload pesos atualizados para GPU
        cfc.weights.wf = backend.upload_f32(&wf_new, &[cd, hd]);
        cfc.weights.wf2 = backend.upload_f32(&wf2_new, &[cd, cd]);
        cfc.weights.bf = backend.upload_f32(&bf_new, &[cd]);
        cfc.weights.wg = backend.upload_f32(&wg_new, &[cd, hd]);
        cfc.weights.wg2 = backend.upload_f32(&wg2_new, &[cd, cd]);
        cfc.weights.bg = backend.upload_f32(&bg_new, &[cd]);
        cfc.weights.wout = backend.upload_f32(&wout_new, &[hd, cd]);
        cfc.weights.bout = backend.upload_f32(&bout_new, &[hd]);

        let avg_loss = total_loss / n as f32;
        self.loss_history.push(avg_loss);
        self.epoch += 1;

        let result = CfcTrainResult {
            epoch: self.epoch,
            loss: avg_loss,
            grad_norm_pre_clip: pre_clip_norm,
            grad_norm_post_clip: post_clip_norm,
            lr: self.lr,
            n_steps: n,
            param_count: cfc.param_count(),
        };

        log::info!(
            "CfC training epoch {}: loss={:.6}, grad_norm={:.4} → {:.4}, lr={:.6}, steps={}",
            result.epoch, result.loss, pre_clip_norm, post_clip_norm, lr, n
        );

        self.activations.clear();
        Some(result)
    }

    /// Retorna telemetria de treinamento.
    pub fn telemetry(&self) -> CfcTrainTelemetry {
        let recent_loss = self.loss_history.last().copied().unwrap_or(0.0);
        let loss_trend = if self.loss_history.len() >= 2 {
            let last = self.loss_history[self.loss_history.len() - 1];
            let prev = self.loss_history[self.loss_history.len() - 2];
            last - prev // negativo = melhorando
        } else {
            0.0
        };
        CfcTrainTelemetry {
            epoch: self.epoch,
            recent_loss,
            loss_trend,
            lr: self.lr,
            n_activations: self.activations.len(),
            total_epochs: self.loss_history.len(),
        }
    }
}

/// Resultado de um step de treinamento CfC.
#[derive(Debug, Clone, Serialize)]
pub struct CfcTrainResult {
    pub epoch: usize,
    pub loss: f32,
    pub grad_norm_pre_clip: f32,
    pub grad_norm_post_clip: f32,
    pub lr: f32,
    pub n_steps: usize,
    pub param_count: usize,
}

/// Telemetria de treinamento CfC para broadcast.
#[derive(Debug, Clone, Serialize)]
pub struct CfcTrainTelemetry {
    pub epoch: usize,
    pub recent_loss: f32,
    pub loss_trend: f32,
    pub lr: f32,
    pub n_activations: usize,
    pub total_epochs: usize,
}

// ── Training Orchestrator ──────────────────────────────────────────────────

/// Orquestrador de treinamento do sistema LNN completo.
///
/// Coordena:
///   - ESN readout training (ridge regression, cada geração)
///   - CfC weight training (backprop + SGD, cada geração)
///   - Reward signal computation
///   - Training schedule & telemetry
pub struct LnnTrainer {
    /// Trainer do CfC (mantém momentum e loss history)
    pub cfc_trainer: Option<CfcTrainer>,
    /// Histórico de rewards para tracking de progresso
    pub reward_history: Vec<f32>,
    /// Se o treinamento está habilitado
    pub enabled: bool,
    /// Treinar ESN automaticamente após cada geração
    pub auto_train_esn: bool,
    /// Treinar CfC automaticamente após cada geração
    pub auto_train_cfc: bool,
    /// Contador de gerações desde o início do treinamento
    pub generation_count: usize,
}

impl LnnTrainer {
    pub fn new(cfc_config: Option<&CfcConfig>) -> Self {
        Self {
            cfc_trainer: cfc_config.map(|c| CfcTrainer::new(c)),
            reward_history: Vec::new(),
            enabled: true,
            auto_train_esn: true,
            auto_train_cfc: true,
            generation_count: 0,
        }
    }

    /// Executa treinamento completo após uma geração.
    ///
    /// Chamado no final do decode loop, com todos os dados acumulados.
    ///
    /// Retorna resultado do treinamento para telemetria.
    pub fn train_after_generation(
        &mut self,
        esn: &mut EsnObserver,
        cfc: Option<&mut CfcLayer>,
        backend: &Arc<dyn ComputeBackend>,
        generated_tokens: &[u32],
    ) -> LnnTrainResult {
        if !self.enabled {
            return LnnTrainResult::default();
        }

        self.generation_count += 1;

        // 1. Compute reward signal
        let reward = RewardSignal::compute(&esn.obs_history, generated_tokens);
        self.reward_history.push(reward.mean_reward);

        log::info!(
            "LNN Training gen #{}: reward={:.4}, entropy={:.2}, diversity={:.2}, repetition={:.2}",
            self.generation_count, reward.mean_reward, reward.mean_entropy,
            reward.lexical_diversity, reward.repetition_score
        );

        let mut result = LnnTrainResult {
            generation: self.generation_count,
            reward: reward.clone(),
            esn_trained: false,
            cfc_trained: None,
        };

        // 2. Train ESN readout
        if self.auto_train_esn {
            if let Some(new_weights) = train_esn_readout(esn, &reward) {
                esn.set_readout(new_weights);
                result.esn_trained = true;
                log::info!("ESN readout atualizado (gen #{})", self.generation_count);
            }
        }

        // 3. Train CfC
        if self.auto_train_cfc {
            if let (Some(trainer), Some(cfc_layer)) = (&mut self.cfc_trainer, cfc) {
                result.cfc_trained = trainer.train_step(cfc_layer, backend);
            }
        }

        result
    }

    /// Registra ativações do CfC para treinamento posterior.
    pub fn record_cfc_activation(
        &mut self,
        x: Vec<f32>,
        h_prev: Vec<f32>,
        h_new: Vec<f32>,
        correction: Vec<f32>,
        f_gate: Vec<f32>,
        g_candidate: Vec<f32>,
        reward: f32,
    ) {
        if let Some(ref mut trainer) = self.cfc_trainer {
            trainer.record_activation(x, h_prev, h_new, correction, f_gate, g_candidate, reward);
        }
    }

    /// Telemetria geral do treinamento.
    pub fn telemetry(&self) -> LnnTrainTelemetry {
        LnnTrainTelemetry {
            enabled: self.enabled,
            generation_count: self.generation_count,
            mean_reward: if self.reward_history.is_empty() {
                0.0
            } else {
                self.reward_history.iter().sum::<f32>() / self.reward_history.len() as f32
            },
            recent_reward: self.reward_history.last().copied().unwrap_or(0.0),
            reward_trend: if self.reward_history.len() >= 2 {
                let n = self.reward_history.len();
                self.reward_history[n - 1] - self.reward_history[n - 2]
            } else {
                0.0
            },
            esn_auto: self.auto_train_esn,
            cfc_auto: self.auto_train_cfc,
            cfc_telemetry: self.cfc_trainer.as_ref().map(|t| t.telemetry()),
        }
    }
}

/// Resultado completo de um ciclo de treinamento.
#[derive(Debug, Clone, Serialize)]
pub struct LnnTrainResult {
    pub generation: usize,
    pub reward: RewardSignal,
    pub esn_trained: bool,
    pub cfc_trained: Option<CfcTrainResult>,
}

impl Default for LnnTrainResult {
    fn default() -> Self {
        Self {
            generation: 0,
            reward: RewardSignal {
                per_token: vec![],
                mean_reward: 0.0,
                mean_entropy: 0.0,
                lexical_diversity: 0.0,
                repetition_score: 0.0,
            },
            esn_trained: false,
            cfc_trained: None,
        }
    }
}

/// Telemetria geral do sistema de treinamento LNN.
#[derive(Debug, Clone, Serialize)]
pub struct LnnTrainTelemetry {
    pub enabled: bool,
    pub generation_count: usize,
    pub mean_reward: f32,
    pub recent_reward: f32,
    pub reward_trend: f32,
    pub esn_auto: bool,
    pub cfc_auto: bool,
    pub cfc_telemetry: Option<CfcTrainTelemetry>,
}

// ── Utilities ──────────────────────────────────────────────────────────────

/// Salva pesos do LNN treinado em disco (safetensors-compatible JSON).
pub fn save_lnn_weights(
    esn: &EsnObserver,
    cfc: Option<&CfcLayer>,
    backend: &Arc<dyn ComputeBackend>,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut data = serde_json::Map::new();

    // ESN readout
    data.insert("esn_w_out".into(), serde_json::json!(esn.w_out));

    // CfC weights (download from GPU)
    if let Some(cfc) = cfc {
        data.insert("cfc_wf".into(), serde_json::json!(backend.download_f32(&cfc.weights.wf)));
        data.insert("cfc_wf2".into(), serde_json::json!(backend.download_f32(&cfc.weights.wf2)));
        data.insert("cfc_bf".into(), serde_json::json!(backend.download_f32(&cfc.weights.bf)));
        data.insert("cfc_wg".into(), serde_json::json!(backend.download_f32(&cfc.weights.wg)));
        data.insert("cfc_wg2".into(), serde_json::json!(backend.download_f32(&cfc.weights.wg2)));
        data.insert("cfc_bg".into(), serde_json::json!(backend.download_f32(&cfc.weights.bg)));
        data.insert("cfc_wout".into(), serde_json::json!(backend.download_f32(&cfc.weights.wout)));
        data.insert("cfc_bout".into(), serde_json::json!(backend.download_f32(&cfc.weights.bout)));
        data.insert("cfc_dim".into(), serde_json::json!(cfc.config.cfc_dim));
        data.insert("hidden_dim".into(), serde_json::json!(cfc.config.hidden_dim));
    }

    let json = serde_json::Value::Object(data);
    let mut file = std::fs::File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json)?.as_bytes())?;

    log::info!("LNN weights salvos em {:?}", path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_signal_basic() {
        let obs = vec![
            [2.0, 0.6, 0.1],
            [1.8, 0.7, 0.2],
            [1.5, 0.8, 0.3],
            [1.2, 0.85, 0.4],
            [1.0, 0.9, 0.5],
            [0.8, 0.92, 0.6],
            [0.6, 0.95, 0.7],
            [0.5, 0.97, 0.8],
        ];
        let tokens = vec![10, 20, 30, 40, 50, 60, 70, 80];

        let reward = RewardSignal::compute(&obs, &tokens);

        assert!(reward.mean_reward > 0.0, "Geração convergente deve ter reward positivo");
        assert!(reward.lexical_diversity > 0.9, "Tokens únicos → alta diversidade");
        assert!(reward.repetition_score < 0.1, "Sem repetição");
        assert_eq!(reward.per_token.len(), 8);
    }

    #[test]
    fn test_reward_repetitive() {
        let obs = vec![
            [3.0, 0.3, 0.1],
            [3.1, 0.3, 0.2],
            [3.0, 0.3, 0.3],
            [3.1, 0.3, 0.4],
            [3.0, 0.3, 0.5],
            [3.1, 0.3, 0.6],
            [3.0, 0.3, 0.7],
            [3.1, 0.3, 0.8],
        ];
        // Tokens repetitivos
        let tokens = vec![10, 20, 10, 20, 10, 20, 10, 20];

        let reward = RewardSignal::compute(&obs, &tokens);

        assert!(reward.repetition_score > 0.2, "Bigrams repetidos devem ser detectados: got {}", reward.repetition_score);
        assert!(reward.mean_reward < 0.6, "Geração repetitiva deve ter reward baixo: got {}", reward.mean_reward);
    }

    #[test]
    fn test_cholesky_solve() {
        // A = [[4, 2], [2, 3]], b = [1, 2]
        // Solução: x = [-0.125, 0.75]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let x = solve_positive_definite(&a, &b, 2).unwrap();
        assert!((x[0] - (-0.125)).abs() < 1e-4);
        assert!((x[1] - 0.75).abs() < 1e-4);
    }

    #[test]
    fn test_esn_ridge_regression() {
        // Simula ESN com estado artificial
        let mut esn = EsnObserver::new(42);
        esn.reset();

        // Cria histórico artificial (10 steps)
        for i in 0..10 {
            let mut state = vec![0.0f32; 256];
            state[0] = i as f32 * 0.1;
            state[1] = (i as f32 * 0.2).sin();
            esn.state_history.push(state);
            esn.obs_history.push([1.0 - i as f32 * 0.05, 0.5 + i as f32 * 0.03, i as f32 / 10.0]);
        }

        let reward = RewardSignal {
            per_token: (0..10).map(|i| 0.5 + i as f32 * 0.05).collect(),
            mean_reward: 0.725,
            mean_entropy: 1.5,
            lexical_diversity: 0.9,
            repetition_score: 0.1,
        };

        let weights = train_esn_readout(&esn, &reward);
        assert!(weights.is_some(), "Ridge regression deve convergir com dados suficientes");

        let w = weights.unwrap();
        assert_eq!(w.len(), 256);
        assert!(w.iter().any(|&v| v.abs() > 1e-6), "Pesos não devem ser todos zero");
    }
}
