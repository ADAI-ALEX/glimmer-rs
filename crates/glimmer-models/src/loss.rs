//! Language-modelling losses.

use burn::tensor::{Int, Tensor};

/// Numerically stable log-softmax over the last dimension of a `[N, V]` tensor
/// (subtracts the row maximum before exponentiating).
pub fn log_softmax_stable(x: Tensor<2>) -> Tensor<2> {
    let max = x.clone().max_dim(1).detach();
    let shifted = x - max;
    let lse = shifted.clone().exp().sum_dim(1).log();
    shifted - lse
}

/// Weighted negative log-likelihood **sum**: `Σ wᵢ · −log p(targetᵢ)`.
///
/// Returning a sum (not a mean) lets the trainer normalise by the total number of supervised
/// tokens across all gradient-accumulation micro-batches, so every supervised token carries
/// equal weight even when micro-batches contain different numbers of targets (as packed SFT
/// batches do).
pub fn weighted_nll_sum(logits: Tensor<3>, targets: Tensor<2, Int>, weights: Tensor<2>) -> Tensor<1> {
    let [b, t, v] = logits.dims();
    let logp = log_softmax_stable(logits.reshape([b * t, v]));
    let picked = logp.gather(1, targets.reshape([b * t, 1])).reshape([b * t]);
    (picked * weights.reshape([b * t])).sum().neg()
}
