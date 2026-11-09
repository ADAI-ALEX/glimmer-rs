//! Samplers: random search and the Tree-structured Parzen Estimator (TPE).
//!
//! TPE (Bergstra et al., 2011) models `p(x | good)` and `p(x | bad)` instead of `p(y | x)`.
//! Completed trials are split at the γ-quantile of the objective; each group gets a Parzen
//! (kernel density) estimator per parameter, and the next point is the candidate drawn from the
//! "good" density that maximises `l(x) / g(x)`. Parameters are modelled independently, as in
//! Hyperopt's default TPE. Until `startup` trials have completed, it falls back to random search.

use crate::space::ParamSpec;
use glimmer_core::Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SamplerKind {
    Random,
    Tpe,
}

/// A completed observation: parameter values and the objective (lower is better).
#[derive(Debug, Clone)]
pub struct Observation {
    pub params: BTreeMap<String, f64>,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct TpeConfig {
    pub gamma: f64,
    pub candidates: usize,
    pub startup: usize,
}

impl Default for TpeConfig {
    fn default() -> Self {
        Self { gamma: 0.25, candidates: 32, startup: 5 }
    }
}

fn gaussian_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
}

/// Parzen estimator in the parameter's internal space, with a uniform prior component.
struct Parzen {
    points: Vec<f64>,
    sigmas: Vec<f64>,
    low: f64,
    high: f64,
}

impl Parzen {
    fn new(mut points: Vec<f64>, low: f64, high: f64) -> Self {
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let range = (high - low).max(1e-12);
        // Bandwidth per point: distance to the farther neighbour (Hyperopt heuristic), clamped.
        let sigmas = (0..points.len())
            .map(|i| {
                let left = if i > 0 { points[i] - points[i - 1] } else { points[i] - low };
                let right = if i + 1 < points.len() { points[i + 1] - points[i] } else { high - points[i] };
                left.max(right).clamp(range / (1.0 + points.len() as f64).powi(2).max(1.0) * 0.5, range)
            })
            .collect();
        Self { points, sigmas, low, high }
    }

    fn pdf(&self, x: f64) -> f64 {
        let prior = 1.0 / (self.high - self.low).max(1e-12);
        let n = self.points.len() as f64 + 1.0;
        let kde: f64 = self.points.iter().zip(&self.sigmas).map(|(&m, &s)| gaussian_pdf(x, m, s)).sum();
        (kde + prior) / n
    }

    fn sample(&self, rng: &mut Rng) -> f64 {
        // Mixture: pick the prior with weight 1/(n+1), otherwise one kernel.
        let k = rng.below(self.points.len() + 1);
        if k == self.points.len() {
            return rng.uniform(self.low, self.high);
        }
        (self.points[k] + rng.normal() * self.sigmas[k]).clamp(self.low, self.high)
    }
}

fn categorical_weights(values: &[f64], obs: &[f64]) -> Vec<f64> {
    // Laplace-smoothed frequencies.
    values.iter().map(|v| 1.0 + obs.iter().filter(|o| (*o - v).abs() < 1e-12).count() as f64).collect()
}

/// Propose the next parameter set.
pub fn propose(
    kind: SamplerKind,
    space: &[ParamSpec],
    history: &[Observation],
    cfg: &TpeConfig,
    rng: &mut Rng,
) -> BTreeMap<String, f64> {
    let random = |rng: &mut Rng| space.iter().map(|p| (p.name().to_string(), p.sample(rng))).collect();
    if kind == SamplerKind::Random || history.len() < cfg.startup {
        return random(rng);
    }
    let mut sorted: Vec<&Observation> = history.iter().filter(|o| o.value.is_finite()).collect();
    if sorted.len() < 2 {
        return random(rng);
    }
    sorted.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
    let n_good = ((cfg.gamma * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len() - 1);
    let (good, bad) = sorted.split_at(n_good);
    let mut out = BTreeMap::new();
    for p in space {
        let name = p.name();
        let g: Vec<f64> = good.iter().filter_map(|o| o.params.get(name)).map(|&v| p.to_internal(v)).collect();
        let b: Vec<f64> = bad.iter().filter_map(|o| o.params.get(name)).map(|&v| p.to_internal(v)).collect();
        let value = match p {
            ParamSpec::Choice { values, .. } => {
                let lw = categorical_weights(values, &g);
                let gw = categorical_weights(values, &b);
                let lt: f64 = lw.iter().sum();
                let gt: f64 = gw.iter().sum();
                // Sample candidates from l, keep the best ratio.
                let mut best = (f64::NEG_INFINITY, values[0]);
                for _ in 0..cfg.candidates {
                    let mut r = rng.next_f64() * lt;
                    let mut idx = 0;
                    for (i, w) in lw.iter().enumerate() {
                        if r < *w {
                            idx = i;
                            break;
                        }
                        r -= w;
                    }
                    let ratio = (lw[idx] / lt) / (gw[idx] / gt);
                    if ratio > best.0 {
                        best = (ratio, values[idx]);
                    }
                }
                best.1
            }
            _ => {
                let (lo, hi) = p.bounds_internal();
                let l = Parzen::new(g, lo, hi);
                let gd = Parzen::new(b, lo, hi);
                let mut best = (f64::NEG_INFINITY, lo);
                for _ in 0..cfg.candidates {
                    let x = l.sample(rng);
                    let score = l.pdf(x).ln() - gd.pdf(x).ln();
                    if score > best.0 {
                        best = (score, x);
                    }
                }
                p.from_internal(best.1)
            }
        };
        out.insert(name.to_string(), value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TPE should concentrate samples near the optimum of a simple 1-D objective far better
    /// than random search.
    #[test]
    fn tpe_beats_random_on_quadratic() {
        let space = vec![ParamSpec::LogUniform { name: "lr".into(), low: 1e-5, high: 1e-1 }];
        let objective = |p: &BTreeMap<String, f64>| {
            let x = p["lr"].log10();
            (x + 3.0).powi(2) // optimum at lr = 1e-3
        };
        let run = |kind: SamplerKind, seed: u64| {
            let mut rng = Rng::new(seed);
            let mut hist: Vec<Observation> = Vec::new();
            for _ in 0..30 {
                let p = propose(kind, &space, &hist, &TpeConfig::default(), &mut rng);
                let v = objective(&p);
                hist.push(Observation { params: p, value: v });
            }
            // Mean objective of the last 10 proposals.
            hist[20..].iter().map(|o| o.value).sum::<f64>() / 10.0
        };
        let (mut tpe, mut rnd) = (0.0, 0.0);
        for seed in 0..5 {
            tpe += run(SamplerKind::Tpe, seed);
            rnd += run(SamplerKind::Random, seed);
        }
        assert!(tpe < rnd * 0.5, "TPE {tpe} should beat random {rnd}");
    }

    #[test]
    fn categorical_tpe_prefers_good_choice() {
        let space = vec![ParamSpec::Choice { name: "batch_size".into(), values: vec![4.0, 8.0, 16.0, 32.0] }];
        let mut rng = Rng::new(3);
        let mut hist = Vec::new();
        for _ in 0..40 {
            let p = propose(SamplerKind::Tpe, &space, &hist, &TpeConfig::default(), &mut rng);
            let v = if p["batch_size"] == 16.0 { 1.0 } else { 2.0 + rng.next_f64() };
            hist.push(Observation { params: p, value: v });
        }
        let late16 = hist[20..].iter().filter(|o| o.params["batch_size"] == 16.0).count();
        assert!(late16 >= 10, "TPE chose 16 only {late16}/20 times");
    }
}
