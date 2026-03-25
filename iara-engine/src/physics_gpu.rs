// ============================================================================
// physics_gpu.rs — Kernels Nativos para GPU AMD (Navi 22)
// ============================================================================

use rocm_kernel_macros::{amdgpu_kernel_attr, amdgpu_kernel_init, amdgpu_kernel_finalize};

// Inicializa a infraestrutura de kernel
amdgpu_kernel_init!();

/// Binário compilado do kernel (HSaco) — Usado pelo GpuOrchestrator
pub const AMDGPU_KERNEL_BINARY_PATH: &str = amdgpu_kernel_finalize!();

/// Kernel 1: Mapeamento de Grade Espacial (Spatial Hashing)
/// Atribui cada neurônio a uma célula 3D baseada em sua posição (x, y, z).
#[amdgpu_kernel_attr]
fn spatial_hash_kernel(
    positions: *const f32, // [n * 3] (x, y, z)
    grid_cells: *mut i32,  // [n] IDs das células
    grid_res: i32,         // Resolução da grade (ex: 128)
    n: usize
) {
    let i = read_by_workitem_id_x(positions);
    if i < n {
        let px = unsafe { *positions.add(i * 3) };
        let py = unsafe { *positions.add(i * 3 + 1) };
        let pz = unsafe { *positions.add(i * 3 + 2) };

        // Normalização e Hash (Mapeia 0.0-1.0 para 0-grid_res)
        let gx = (px * grid_res as f32) as i32;
        let gy = (py * grid_res as f32) as i32;
        let gz = (pz * grid_res as f32) as i32;

        let cell_id = gx + gy * grid_res + gz * grid_res * grid_res;
        unsafe { *grid_cells.add(i) = cell_id; }
    }
}

/// Kernel 2: Pressão de Pauli com Otimização Espacial (O(N) Local)
#[amdgpu_kernel_attr]
fn pauli_repulsion_spatial_kernel(
    states: *const f32,
    grid_cells: *const i32,
    repulsion_out: *mut f32,
    n: usize,
    k_pauli: f32
) {
    let i = read_by_workitem_id_x(states);
    if i < n {
        let mut force = 0.0f32;
        let xi = unsafe { *states.add(i) };
        let my_cell = unsafe { *grid_cells.add(i) };
        
        // Em um sistema real, aqui iteraríamos apenas nas células vizinhas.
        // Para a V1, filtramos por cell_id para reduzir o loop global.
        for j in 0..n {
            if i != j {
                let neighbor_cell = unsafe { *grid_cells.add(j) };
                if (my_cell - neighbor_cell).abs() <= 1 {
                    let xj = unsafe { *states.add(j) };
                    let dist = (xi - xj).abs();
                    if dist > 0.0 && dist < 0.05 {
                        force += (k_pauli / (dist * dist + 1e-4));
                    }
                }
            }
        }
        unsafe { *repulsion_out.add(i) = force; }
    }
}

/// Kernel 3: Malha KAN Elástica
#[amdgpu_kernel_attr]
fn kan_spline_kernel(
    states: *const f32,
    weights_dna: *const f32,
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
            // Função Spline na GPU: DNA * Sino(x)
            coupling += w_dna * xj * (-xj.abs()).exp();
        }
        unsafe { *output.add(i) = coupling; }
    }
}
