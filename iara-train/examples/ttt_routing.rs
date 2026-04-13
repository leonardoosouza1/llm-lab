//! TTT Routing — adaptação online dos pesos da heurística A*.
//!
//! Simula um loop de inferência onde:
//!   1. A* roteia com pesos atuais
//!   2. A execução da rota gera feedback de qualidade real
//!   3. TTT atualiza os pesos [w_sparsity, w_confidence]
//!   4. Próxima rota usa pesos melhorados
//!
//! Objetivo: pesos convergem para minimizar erro entre
//! qualidade prevista e qualidade real observada.

use iara_train::navigator::{
    router::{route_astar, RouterConfig, N_SUPER_LAYERS, SL_NAMES},
    ttt::{TttOptimizer, TttConfig, RouteObservation},
};

/// Simula "qualidade real" de uma rota executada.
/// Em produção: mede cosine_sim entre output completo vs output roteado.
fn simulate_quality_loss(path: &[usize], sparsity: &[f32; 4], rng: &mut u64) -> f32 {
    // Perda base: número de layers puladas × custo por layer
    let n_skipped = path.windows(2)
        .map(|w| w[1].saturating_sub(w[0]).saturating_sub(1))
        .sum::<usize>();

    // Adiciona ruído realístico (medição de qualidade tem variância)
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let noise = ((*rng >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.01;

    // Esparsidade reduz perda (layers esparsas contribuem menos ao output)
    let sparsity_factor = sparsity.iter().sum::<f32>() / 4.0;
    let base_loss = n_skipped as f32 * 0.025 * (1.0 - sparsity_factor * 0.5);

    (base_loss + noise).max(0.0)
}

fn main() {
    println!("=== IARA TTT Routing — Adaptação Online de Heurísticas ===\n");

    let n_steps    = 200;
    let mut rng    = 0xFEED_BEEF_u64;

    // Pesos iniciais (igual ao default do RouterConfig)
    let init_weights = [0.6f32, 0.4];
    let ttt_cfg  = TttConfig { lr: 0.02, window: 32, ..Default::default() };
    let mut ttt  = TttOptimizer::new(init_weights, ttt_cfg);

    // Histórico para relatório
    let mut weight_history = Vec::<[f32; 2]>::new();
    let mut loss_history   = Vec::<f32>::new();
    let mut flops_saved_history = Vec::<f32>::new();

    println!("{:<6} {:<10} {:<10} {:<8} {:<8} {:<10}",
        "Step", "w_spars", "w_conf", "loss", "flops%", "route");
    println!("{}", "-".repeat(65));

    for step in 0..n_steps {
        // Gerar features de ambiente variáveis
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let base = (rng >> 33) as f32 / u32::MAX as f32;

        let sparsity: [f32; 4] = [
            (base * 0.6).min(0.8),
            (base * 0.5 + 0.1).min(0.8),
            (base * 0.7).min(0.8),
            (base * 0.4 + 0.05).min(0.8),
        ];
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let logit_entropy = (rng >> 33) as f32 / u32::MAX as f32 * 0.8;

        // A* com pesos TTT atuais
        let config = RouterConfig {
            quality_threshold:  0.95,
            heuristic_weights:  ttt.weights,
        };
        let route = route_astar(&sparsity, logit_entropy, &config);

        // Simular qualidade real (feedback)
        let obs_ql  = simulate_quality_loss(&route.path, &sparsity, &mut rng);
        let pred_ql = route.quality_loss;

        // TTT update
        let obs = RouteObservation {
            sparsity_per_sl:        sparsity,
            logit_entropy,
            observed_quality_loss:  obs_ql,
            predicted_quality_loss: pred_ql,
        };
        let loss = ttt.update(&obs);

        // Log a cada 20 steps
        if step % 20 == 0 || step == n_steps - 1 {
            let route_names: Vec<_> = route.path.iter()
                .map(|&i| SL_NAMES[i.min(N_SUPER_LAYERS - 1)])
                .collect();
            println!("{:<6} {:<10.4} {:<10.4} {:<8.5} {:<8.1} {:?}",
                step, ttt.weights[0], ttt.weights[1], loss,
                route.flops_saved_pct, route_names);
        }

        weight_history.push(ttt.weights);
        loss_history.push(loss);
        flops_saved_history.push(route.flops_saved_pct);
    }

    // ── Análise de convergência ───────────────────────────────────────────────
    println!("\n{}", "─".repeat(65));
    let stats = ttt.stats();

    let avg_flops_early: f32 = flops_saved_history[..20].iter().sum::<f32>() / 20.0;
    let avg_flops_late:  f32 = flops_saved_history[180..].iter().sum::<f32>() / 20.0;
    let avg_loss_early:  f32 = loss_history[..20].iter().sum::<f32>() / 20.0;
    let avg_loss_late:   f32 = loss_history[180..].iter().sum::<f32>() / 20.0;

    let w0_drift = (weight_history.last().unwrap()[0] - weight_history[0][0]).abs();
    let w1_drift = (weight_history.last().unwrap()[1] - weight_history[0][1]).abs();

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║           TTT Routing — Convergência                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Steps:           {:>6}                               ║", n_steps);
    println!("║  Updates:         {:>6}                               ║", stats.n_updates);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Pesos iniciais:  [{:.3}, {:.3}]                    ║", init_weights[0], init_weights[1]);
    println!("║  Pesos finais:    [{:.3}, {:.3}]                    ║", stats.weights[0], stats.weights[1]);
    println!("║  Drift w_spars:   {:.4}                             ║", w0_drift);
    println!("║  Drift w_conf:    {:.4}                             ║", w1_drift);
    println!("║  Convergido:      {}                               ║", if stats.converged { "SIM ✓" } else { "NÃO" });
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Loss early:      {:.6}                          ║", avg_loss_early);
    println!("║  Loss late:       {:.6}                          ║", avg_loss_late);
    println!("║  FLOPs% early:    {:.1}%                            ║", avg_flops_early);
    println!("║  FLOPs% late:     {:.1}%                            ║", avg_flops_late);
    println!("╚══════════════════════════════════════════════════════════╝");

    if !stats.converged {
        println!("\nNota: aumentar n_steps ou reduzir lr para mais convergência.");
    }
}
