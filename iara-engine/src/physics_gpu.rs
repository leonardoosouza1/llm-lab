// ============================================================================
// physics_gpu.rs — Kernels Nativos para GPU AMD (Navi 22)
// ============================================================================
// Implementa a Malha KAN e a Pressão de Pauli usando rocm-rs.
// ============================================================================

use rocm_kernel_macros::{amdgpu_kernel_attr, amdgpu_kernel_init};
use std::path::PathBuf;

// Inicializa a infraestrutura de kernel
amdgpu_kernel_init!();

/// Kernel de Pressão de Degenerescência de Pauli
/// Calcula a força repulsiva entre neurônios em paralelo bruto.
#[amdgpu_kernel_attr]
fn pauli_repulsion_kernel(
    states: *const f32,
    repulsion_out: *mut f32,
    n: usize,
    k_pauli: f32
) {
    // Pegamos o ID da thread (cada thread processa um neurônio)
    let i = read_by_workitem_id_x(states); // Simplificação da macro
    
    if i < n {
        let mut force = 0.0f32;
        let xi = unsafe { *states.add(i) };
        
        for j in 0..n {
            if i != j {
                let xj = unsafe { *states.add(j) };
                let dist = (xi - xj).abs();
                if dist > 0.0 && dist < 0.1 {
                    let direction = if xi > xj { 1.0 } else { -1.0 };
                    force += direction * (k_pauli / (dist * dist + 1e-4));
                }
            }
        }
        unsafe { *repulsion_out.add(i) = force; }
    }
}

/// Kernel de Malha KAN (B-Splines)
/// Processa a interpolação elástica das sinapses.
#[amdgpu_kernel_attr]
fn kan_spline_kernel(
    states: *const f32,
    weights_dna: *const f32,
    weights_plastic: *const f32,
    output: *mut f32,
    n: usize
) {
    let i = read_by_workitem_id_x(states);
    if i < n {
        let mut coupling = 0.0f32;
        let xi = unsafe { *states.add(i) };
        
        for j in 0..n {
            let xj = unsafe { *states.add(j) };
            let w_dna = unsafe { *weights_dna.add(i * n + j) };
            let w_p = unsafe { *weights_plastic.add(i * n + j) };
            
            // Função Spline Axiomática: DNA (Sino) + Plástico (Cúbico)
            coupling += w_dna * xj * (-xj.abs()).exp() + w_p * xj * xj * xj;
        }
        unsafe { *output.add(i) = coupling; }
    }
}
