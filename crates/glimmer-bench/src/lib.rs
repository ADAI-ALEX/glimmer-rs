//! # glimmer-bench
//!
//! Benchmark methodology:
//!
//! * **Warm-up** iterations run first (kernel compilation, autotuning, allocator growth) and are
//!   excluded from timing.
//! * Every timed iteration ends with a **device synchronisation** (reading a scalar back), so
//!   asynchronous GPU queues cannot make work look free.
//! * Each benchmark is **repeated**; we report mean, standard deviation and minimum.
//! * Results are recorded with the full configuration and hardware description.

pub mod suite;

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub name: String,
    pub backend: String,
    pub config: String,
    pub warmup: usize,
    pub iterations: usize,
    pub mean_ms: f64,
    pub std_ms: f64,
    pub min_ms: f64,
    /// Work per iteration (e.g. tokens or FLOPs) and its unit, for throughput reporting.
    pub work: Option<(f64, String)>,
}

impl Measurement {
    pub fn throughput(&self) -> Option<(f64, String)> {
        self.work.as_ref().map(|(w, unit)| (w / (self.mean_ms / 1e3), format!("{unit}/s")))
    }
}

/// Time `f` with warm-up and repetitions. `f` must synchronise the device before returning.
pub fn measure(warmup: usize, iterations: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len().max(1) as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    (mean, var.sqrt(), min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measure_reports_sensible_statistics() {
        let mut n = 0;
        let (mean, std, min) = measure(2, 5, || {
            n += 1;
            std::thread::sleep(std::time::Duration::from_millis(2));
        });
        assert_eq!(n, 7);
        assert!(mean >= 2.0 && min >= 2.0 && std >= 0.0);
    }
}
