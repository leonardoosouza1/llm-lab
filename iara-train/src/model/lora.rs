//! LoRA (Low-Rank Adaptation) adapter for burn Linear layers.
//!
//! Implements the technique from "LoRA: Low-Rank Adaptation of Large Language Models"
//! (Hu et al., 2021). A frozen base linear layer is augmented with two small trainable
//! matrices A and B such that:
//!
//!   y = base(x) + (scaling) * x @ A^T @ B^T
//!
//! where A is [rank, in_features], B is [out_features, rank], and scaling = alpha / rank.
//! Only A and B receive gradients; the base weights are frozen.

use burn::module::{Module, Param, ParamId};
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

/// LoRA adapter wrapping a frozen burn `Linear` layer.
///
/// The base linear layer's parameters are frozen (no gradients). Two low-rank
/// matrices `lora_a` (rank x in_features) and `lora_b` (out_features x rank)
/// are the only trainable parameters. The LoRA contribution is scaled by
/// `alpha / rank`.
#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    /// Frozen base linear layer. We call `.no_grad()` at construction so its
    /// weight (and bias, if present) will not receive gradients.
    base: Linear<B>,

    /// Low-rank matrix A of shape [rank, in_features].
    /// Initialized with Kaiming uniform so the initial LoRA output has variance.
    lora_a: Param<Tensor<B, 2>>,

    /// Low-rank matrix B of shape [out_features, rank].
    /// Initialized with zeros so LoRA starts as identity (delta = 0).
    lora_b: Param<Tensor<B, 2>>,

    /// Scaling factor = alpha / rank, stored as a plain f64 (not a parameter).
    scaling: f64,
}

impl<B: Backend> LoraLinear<B> {
    /// Create a new LoRA adapter around an existing `Linear` layer.
    ///
    /// # Arguments
    ///
    /// * `base`   - The pre-trained linear layer to wrap. Its weights are frozen.
    /// * `rank`   - The rank of the low-rank decomposition.
    /// * `alpha`  - Scaling numerator; effective scaling is `alpha / rank`.
    /// * `device` - The device on which to allocate LoRA parameters.
    pub fn new(base: Linear<B>, rank: usize, alpha: f64, device: &B::Device) -> Self {
        let [d_input, d_output] = base.weight.shape().dims();

        // Freeze the base layer: all its Param tensors lose require_grad.
        let base = base.no_grad();

        // A: [rank, d_input] -- Kaiming uniform init
        // The bound is sqrt(1 / d_input) which matches the default Kaiming uniform
        // with gain = 1/sqrt(3) and fan_in = d_input. We use the simpler uniform
        // formulation: U(-1/sqrt(d_input), 1/sqrt(d_input)).
        let bound = 1.0 / (d_input as f64).sqrt();
        let lora_a = Tensor::<B, 2>::random(
            [rank, d_input],
            Distribution::Uniform(-bound, bound),
            device,
        );
        let lora_a = Param::from_tensor(lora_a);

        // B: [d_output, rank] -- zero init so LoRA starts as identity
        let lora_b = Tensor::<B, 2>::zeros([d_output, rank], device);
        let lora_b = Param::from_tensor(lora_b);

        let scaling = alpha / rank as f64;

        Self {
            base,
            lora_a,
            lora_b,
            scaling,
        }
    }

    /// Forward pass: base(x) + scaling * x @ A^T @ B^T
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch, seq_len, in_features].
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch, seq_len, out_features].
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Base linear: uses weight [d_input, d_output], computes x @ W (+ bias)
        let base_out = self.base.forward(x.clone());

        // LoRA delta: x @ A^T @ B^T
        // x:    [batch, seq, d_input]
        // A^T:  [d_input, rank]       (A is [rank, d_input])
        // B^T:  [rank, d_output]      (B is [d_output, rank])
        //
        // Step 1: x @ A^T -> [batch, seq, rank]
        let a_t = self.lora_a.val().transpose();
        let h = x.matmul(a_t.unsqueeze());

        // Step 2: h @ B^T -> [batch, seq, d_output]
        let b_t = self.lora_b.val().transpose();
        let delta = h.matmul(b_t.unsqueeze());

        // Combine: base + scaling * delta
        base_out + delta.mul_scalar(self.scaling)
    }

    /// Merge LoRA weights back into the base linear layer.
    ///
    /// Produces a single `Linear` with weight W_new = W_old + scaling * (B @ A).
    /// After merging there is no LoRA overhead at inference time.
    ///
    /// Note: the burn Linear weight is stored as [d_input, d_output] (row-major),
    /// so we need (B @ A)^T = A^T @ B^T which has shape [d_input, d_output].
    pub fn merge(self) -> Linear<B> {
        // W_old: [d_input, d_output]
        let w_old = self.base.weight.val();

        // B @ A: [d_output, rank] @ [rank, d_input] = [d_output, d_input]
        // We need [d_input, d_output], so transpose: (B @ A)^T = A^T @ B^T
        let a = self.lora_a.val(); // [rank, d_input]
        let b = self.lora_b.val(); // [d_output, rank]
        let delta = a.transpose().matmul(b.transpose()); // [d_input, d_output]

        let w_new = w_old + delta.mul_scalar(self.scaling);

        // Rebuild the Linear with merged weight
        Linear {
            weight: Param::initialized(ParamId::new(), w_new),
            bias: self.base.bias,
        }
    }

    /// Return the LoRA rank.
    pub fn rank(&self) -> usize {
        self.lora_a.shape().dims::<2>()[0]
    }

    /// Return the scaling factor (alpha / rank).
    pub fn scaling(&self) -> f64 {
        self.scaling
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::LinearConfig;
    use burn::tensor::Shape;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn lora_identity_at_init() {
        // Because B is initialized to zeros, LoRA should initially be identity:
        // forward(x) == base.forward(x)
        let device = Default::default();
        let base = LinearConfig::new(8, 4).with_bias(false).init(&device);

        let x = Tensor::<TestBackend, 3>::random(
            [2, 3, 8],
            Distribution::Default,
            &device,
        );
        let expected = base.forward(x.clone());

        let lora = LoraLinear::new(base, 2, 1.0, &device);
        let actual = lora.forward(x);

        // They should be exactly equal since B=0 means delta=0
        assert_eq!(actual.to_data(), expected.to_data());
    }

    #[test]
    fn lora_merge_matches_forward() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        let base = LinearConfig::new(8, 4).with_bias(true).init(&device);

        // Create LoRA and manually set A and B to non-zero values
        let mut lora = LoraLinear::new(base, 2, 2.0, &device);

        // Override A and B with known values for deterministic test
        let a = Tensor::<TestBackend, 2>::random(
            [2, 8],
            Distribution::Uniform(-0.5, 0.5),
            &device,
        );
        let b = Tensor::<TestBackend, 2>::random(
            [4, 2],
            Distribution::Uniform(-0.5, 0.5),
            &device,
        );
        lora.lora_a = Param::from_tensor(a);
        lora.lora_b = Param::from_tensor(b);

        let x = Tensor::<TestBackend, 3>::random(
            [1, 5, 8],
            Distribution::Default,
            &device,
        );

        // Forward through LoRA
        let lora_out = lora.forward(x.clone());

        // Merge and forward through the merged linear
        let merged = lora.merge();
        let merged_out = merged.forward(x);

        // They should be approximately equal (floating point)
        let diff = (lora_out - merged_out).abs().max();
        let max_diff: f32 = diff.into_scalar();
        assert!(
            max_diff < 1e-5,
            "Merge output differs from LoRA forward by {max_diff}"
        );
    }

    #[test]
    fn lora_shapes() {
        let device = Default::default();
        let base = LinearConfig::new(16, 8).with_bias(false).init(&device);
        let lora = LoraLinear::new(base, 4, 1.0, &device);

        assert_eq!(lora.lora_a.shape(), Shape::from([4, 16]));
        assert_eq!(lora.lora_b.shape(), Shape::from([8, 4]));
        assert_eq!(lora.rank(), 4);
        assert!((lora.scaling() - 0.25).abs() < 1e-10);

        // Forward should produce correct output shape
        let x = Tensor::<TestBackend, 3>::zeros([2, 5, 16], &device);
        let y = lora.forward(x);
        assert_eq!(y.shape(), Shape::from([2, 5, 8]));
    }
}
