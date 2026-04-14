//! IARA WGSL Ternary Matmul GPU Kernel — Production Module
//!
//! Optimized WGSL compute shader for ternary matrix-vector multiplication.
//! Uses column-major packed layout, shared memory tiling, LUT decode, and
//! fused batch encoding for minimal dispatch overhead.
//!
//! Architecture (validated on AMD RX 6750 XT, RADV NAVI22):
//!   - WG=128 threads, Tile_K=256 (2 loads/thread)
//!   - No reduction tree (1 thread = 1 output element)
//!   - Column-major packed layout for coalesced reads
//!   - LUT decode: const array replaces arithmetic tsign
//!   - vec4 dot products for hardware FMA
//!
//! Performance:
//!   Single dispatch: ~125-178 µs for [8960 × 1536]
//!   Fused 14-layer batch: 751 µs (53 µs/layer)
//!   Bandwidth: 19-27 GB/s (7-10% of 288 GB/s peak)

/// The production WGSL shader source code.
/// WG=128, Tile_K=256, column-major packed, LUT decode, vec4 dot.
pub const TERNARY_MATMUL_SHADER: &str = r#"
struct Uniforms { out_dim: u32, in_dim: u32, scale: f32, packed_cols: u32, };

@group(0) @binding(0) var<storage, read>       packed : array<u32>;
@group(0) @binding(1) var<storage, read>       x_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out  : array<f32>;
@group(0) @binding(3) var<uniform>             u      : Uniforms;

var<workgroup> x_tile: array<f32, 256>;

const LUT = array<f32, 4>(0.0, 1.0, -1.0, 0.0);

fn dot16_lut(w: u32, off: u32) -> f32 {
    let a0 = vec4<f32>(LUT[(w >>  0u) & 3u], LUT[(w >>  2u) & 3u],
                       LUT[(w >>  4u) & 3u], LUT[(w >>  6u) & 3u]);
    let a1 = vec4<f32>(LUT[(w >>  8u) & 3u], LUT[(w >> 10u) & 3u],
                       LUT[(w >> 12u) & 3u], LUT[(w >> 14u) & 3u]);
    let a2 = vec4<f32>(LUT[(w >> 16u) & 3u], LUT[(w >> 18u) & 3u],
                       LUT[(w >> 20u) & 3u], LUT[(w >> 22u) & 3u]);
    let a3 = vec4<f32>(LUT[(w >> 24u) & 3u], LUT[(w >> 26u) & 3u],
                       LUT[(w >> 28u) & 3u], LUT[(w >> 30u) & 3u]);

    let x0 = vec4<f32>(x_tile[off +  0u], x_tile[off +  1u], x_tile[off +  2u], x_tile[off +  3u]);
    let x1 = vec4<f32>(x_tile[off +  4u], x_tile[off +  5u], x_tile[off +  6u], x_tile[off +  7u]);
    let x2 = vec4<f32>(x_tile[off +  8u], x_tile[off +  9u], x_tile[off + 10u], x_tile[off + 11u]);
    let x3 = vec4<f32>(x_tile[off + 12u], x_tile[off + 13u], x_tile[off + 14u], x_tile[off + 15u]);

    return dot(a0, x0) + dot(a1, x1) + dot(a2, x2) + dot(a3, x3);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(global_invocation_id) gid : vec3<u32>,
) {
    let out_idx = gid.x;
    let tid = lid.x;
    var acc: f32 = 0.0;

    for (var tile_start: u32 = 0u; tile_start < u.in_dim; tile_start += 256u) {
        // Cooperative load: 128 threads × 2 loads = 256 elements
        let k0 = tile_start + tid;
        let k1 = tile_start + tid + 128u;
        if k0 < u.in_dim { x_tile[tid] = x_in[k0]; } else { x_tile[tid] = 0.0; }
        if k1 < u.in_dim { x_tile[tid + 128u] = x_in[k1]; } else { x_tile[tid + 128u] = 0.0; }
        workgroupBarrier();

        if out_idx < u.out_dim {
            let tile_packed_count = min(256u, u.in_dim - tile_start) / 16u;
            for (var p: u32 = 0u; p < tile_packed_count; p++) {
                let k_block = (tile_start / 16u) + p;
                acc += dot16_lut(packed[k_block * u.out_dim + out_idx], p * 16u);
            }
        }
        workgroupBarrier();
    }

    if out_idx < u.out_dim {
        y_out[out_idx] = acc * u.scale;
    }
}
"#;

/// Workgroup size for the primary shader (WG=128, good for out_dim >= 2048).
pub const WORKGROUP_SIZE: u32 = 128;

/// Small-output shader variant: WG=32 for matrices with small out_dim.
/// Dispatches 4× more workgroups → better CU occupancy for down_proj [1536 × 8960].
pub const TERNARY_MATMUL_SHADER_SMALL: &str = r#"
struct Uniforms { out_dim: u32, in_dim: u32, scale: f32, packed_cols: u32, };

@group(0) @binding(0) var<storage, read>       packed : array<u32>;
@group(0) @binding(1) var<storage, read>       x_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out  : array<f32>;
@group(0) @binding(3) var<uniform>             u      : Uniforms;

var<workgroup> x_tile: array<f32, 256>;

const LUT = array<f32, 4>(0.0, 1.0, -1.0, 0.0);

fn dot16_lut(w: u32, off: u32) -> f32 {
    let a0 = vec4<f32>(LUT[(w >>  0u) & 3u], LUT[(w >>  2u) & 3u],
                       LUT[(w >>  4u) & 3u], LUT[(w >>  6u) & 3u]);
    let a1 = vec4<f32>(LUT[(w >>  8u) & 3u], LUT[(w >> 10u) & 3u],
                       LUT[(w >> 12u) & 3u], LUT[(w >> 14u) & 3u]);
    let a2 = vec4<f32>(LUT[(w >> 16u) & 3u], LUT[(w >> 18u) & 3u],
                       LUT[(w >> 20u) & 3u], LUT[(w >> 22u) & 3u]);
    let a3 = vec4<f32>(LUT[(w >> 24u) & 3u], LUT[(w >> 26u) & 3u],
                       LUT[(w >> 28u) & 3u], LUT[(w >> 30u) & 3u]);
    let x0 = vec4<f32>(x_tile[off +  0u], x_tile[off +  1u], x_tile[off +  2u], x_tile[off +  3u]);
    let x1 = vec4<f32>(x_tile[off +  4u], x_tile[off +  5u], x_tile[off +  6u], x_tile[off +  7u]);
    let x2 = vec4<f32>(x_tile[off +  8u], x_tile[off +  9u], x_tile[off + 10u], x_tile[off + 11u]);
    let x3 = vec4<f32>(x_tile[off + 12u], x_tile[off + 13u], x_tile[off + 14u], x_tile[off + 15u]);
    return dot(a0, x0) + dot(a1, x1) + dot(a2, x2) + dot(a3, x3);
}

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(global_invocation_id) gid : vec3<u32>,
) {
    let out_idx = gid.x;
    let tid = lid.x;
    var acc: f32 = 0.0;

    for (var tile_start: u32 = 0u; tile_start < u.in_dim; tile_start += 256u) {
        // Cooperative load: 32 threads × 8 loads = 256 elements
        for (var ld: u32 = 0u; ld < 8u; ld++) {
            let local_idx = ld * 32u + tid;
            let k_global = tile_start + local_idx;
            if k_global < u.in_dim { x_tile[local_idx] = x_in[k_global]; }
            else { x_tile[local_idx] = 0.0; }
        }
        workgroupBarrier();

        if out_idx < u.out_dim {
            let tile_packed_count = min(256u, u.in_dim - tile_start) / 16u;
            for (var p: u32 = 0u; p < tile_packed_count; p++) {
                let k_block = (tile_start / 16u) + p;
                acc += dot16_lut(packed[k_block * u.out_dim + out_idx], p * 16u);
            }
        }
        workgroupBarrier();
    }

    if out_idx < u.out_dim {
        y_out[out_idx] = acc * u.scale;
    }
}
"#;

/// Workgroup size for the small-output shader.
pub const WORKGROUP_SIZE_SMALL: u32 = 32;

/// Threshold: use SMALL shader when out_dim < this value.
pub const SMALL_THRESHOLD: usize = 2048;

/// Select the appropriate shader and workgroup config for a given output dimension.
/// Returns (shader_source, workgroup_size, n_workgroups).
pub fn select_kernel(out_dim: usize) -> (&'static str, u32, u32) {
    if out_dim < SMALL_THRESHOLD {
        let n_wg = ((out_dim + WORKGROUP_SIZE_SMALL as usize - 1) / WORKGROUP_SIZE_SMALL as usize) as u32;
        (TERNARY_MATMUL_SHADER_SMALL, WORKGROUP_SIZE_SMALL, n_wg)
    } else {
        let n_wg = ((out_dim + WORKGROUP_SIZE as usize - 1) / WORKGROUP_SIZE as usize) as u32;
        (TERNARY_MATMUL_SHADER, WORKGROUP_SIZE, n_wg)
    }
}

/// Compute the number of workgroups needed for the primary (WG=128) shader.
pub fn compute_n_workgroups(out_dim: usize) -> u32 {
    ((out_dim + WORKGROUP_SIZE as usize - 1) / WORKGROUP_SIZE as usize) as u32
}

/// Build the uniform buffer contents for the ternary matmul kernel.
/// Returns [out_dim, in_dim, scale_bits, packed_cols].
pub fn build_uniforms(out_dim: usize, in_dim: usize, scale: f32) -> [u32; 4] {
    let packed_cols = in_dim / 16;
    [out_dim as u32, in_dim as u32, scale.to_bits(), packed_cols as u32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroups() {
        assert_eq!(compute_n_workgroups(128), 1);
        assert_eq!(compute_n_workgroups(129), 2);
        assert_eq!(compute_n_workgroups(8960), 70);
    }

    #[test]
    fn test_uniforms() {
        let u = build_uniforms(8960, 1536, 0.211);
        assert_eq!(u[0], 8960);
        assert_eq!(u[1], 1536);
        assert_eq!(u[3], 96); // 1536/16
    }

    #[test]
    fn test_select_kernel() {
        // Large output → primary shader
        let (shader, wg, n) = select_kernel(8960);
        assert!(std::ptr::eq(shader, TERNARY_MATMUL_SHADER));
        assert_eq!(wg, 128);
        assert_eq!(n, 70);

        // Small output → small shader
        let (shader, wg, n) = select_kernel(1536);
        assert!(std::ptr::eq(shader, TERNARY_MATMUL_SHADER_SMALL));
        assert_eq!(wg, 32);
        assert_eq!(n, 48);
    }
}
