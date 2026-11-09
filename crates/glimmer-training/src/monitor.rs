//! Training-health monitor.
//!
//! Watches the loss, gradient norms and validation curve and produces *advice*. It never changes
//! the configuration: every recommendation is logged as an `advice` event for the user (or the
//! hyperparameter search) to act on, which keeps runs reproducible. The only automatic actions
//! the trainer takes — skipping non-finite steps and early stopping — are configured up front
//! and logged as `auto_action` events.

use glimmer_core::metrics::{Ema, EvalMetrics, StepMetrics};
use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq)]
pub struct Advice {
    pub code: &'static str,
    pub message: String,
}

pub struct Monitor {
    fast: Ema,
    slow: Ema,
    clipped: VecDeque<bool>,
    vals: Vec<(usize, f64, f64)>,
    warmup: usize,
    last_spike: Option<usize>,
    issued: std::collections::HashMap<&'static str, usize>,
}

const CLIP_WINDOW: usize = 100;

impl Monitor {
    pub fn new(warmup_steps: usize) -> Self {
        Self {
            fast: Ema::new(0.3),
            slow: Ema::new(0.02),
            clipped: VecDeque::with_capacity(CLIP_WINDOW),
            vals: Vec::new(),
            warmup: warmup_steps,
            last_spike: None,
            issued: Default::default(),
        }
    }

    /// Rate-limit advice: each code at most once per `every` steps.
    fn once(&mut self, code: &'static str, step: usize, every: usize, message: String) -> Option<Advice> {
        match self.issued.get(code) {
            Some(&last) if step < last + every => None,
            _ => {
                self.issued.insert(code, step);
                Some(Advice { code, message })
            }
        }
    }

    pub fn observe_step(&mut self, m: &StepMetrics) -> Vec<Advice> {
        let mut out = Vec::new();
        let slow_before = self.slow.value;
        self.fast.update(m.loss);
        self.slow.update(m.loss);
        if self.clipped.len() == CLIP_WINDOW {
            self.clipped.pop_front();
        }
        self.clipped.push_back(m.clipped);
        let past_warmup = m.step > self.warmup + 20;

        if let Some(slow) = slow_before {
            if past_warmup && m.loss > slow * 1.5 && m.loss - slow > 0.5 {
                self.last_spike = Some(m.step);
                if let Some(a) = self.once(
                    "loss_spike",
                    m.step,
                    200,
                    format!(
                        "Loss spike at step {}: {:.3} vs running average {:.3}. Isolated spikes are \
                         normal; if they recur, lower the peak learning rate (now {:.2e}) or lengthen warmup.",
                        m.step, m.loss, slow, m.lr
                    ),
                ) {
                    out.push(a);
                }
            }
        }
        if self.clipped.len() == CLIP_WINDOW && past_warmup {
            let frac = self.clipped.iter().filter(|&&c| c).count() as f64 / CLIP_WINDOW as f64;
            if frac > 0.6 {
                if let Some(a) = self.once(
                    "frequent_clipping",
                    m.step,
                    1000,
                    format!(
                        "Gradients were clipped on {:.0}% of the last {CLIP_WINDOW} steps (norm {:.2}). \
                         The learning rate may be too high for this model and batch size.",
                        frac * 100.0,
                        m.grad_norm
                    ),
                ) {
                    out.push(a);
                }
            }
        }
        if m.grad_norm.is_finite() && m.grad_norm > 50.0 && past_warmup {
            if let Some(a) = self.once(
                "large_gradients",
                m.step,
                500,
                format!("Very large gradient norm ({:.1}) at step {}. Check for instability.", m.grad_norm, m.step),
            ) {
                out.push(a);
            }
        }
        out
    }

    pub fn observe_eval(&mut self, e: &EvalMetrics) -> Vec<Advice> {
        let mut out = Vec::new();
        self.vals.push((e.step, e.val_loss, e.train_loss));
        let n = self.vals.len();
        if n >= 3 {
            let (s0, v0, t0) = self.vals[n - 3];
            let (_, v1, _) = self.vals[n - 2];
            let (s2, v2, t2) = self.vals[n - 1];
            if v2 > v1 && v1 > v0 && t2 < t0 {
                if let Some(a) = self.once(
                    "overfitting",
                    e.step,
                    1,
                    format!(
                        "Validation loss rose for two evaluations ({v0:.3} → {v2:.3}, steps {s0}–{s2}) while \
                         training loss fell ({t0:.3} → {t2:.3}): the model is overfitting. The best checkpoint \
                         is kept; consider early stopping, more data, dropout or weight decay."
                    ),
                ) {
                    out.push(a);
                }
            }
            let gap = v2 - t2;
            if gap > 0.5 && t2 < t0 {
                if let Some(a) = self.once(
                    "generalisation_gap",
                    e.step,
                    2000,
                    format!("Validation loss exceeds training loss by {gap:.2}; generalisation is degrading."),
                ) {
                    out.push(a);
                }
            }
        }
        if n >= 4 {
            let recent: Vec<f64> = self.vals[n - 4..].iter().map(|v| v.1).collect();
            let improvement = recent[0] - recent[3];
            if improvement.abs() < 0.005 && e.step > self.warmup {
                if let Some(a) = self.once(
                    "plateau",
                    e.step,
                    3000,
                    format!(
                        "Validation loss has plateaued around {:.3} over the last four evaluations. \
                         Training longer at this learning rate is unlikely to help much.",
                        recent[3]
                    ),
                ) {
                    out.push(a);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(step: usize, loss: f64, clipped: bool) -> StepMetrics {
        StepMetrics {
            step,
            tokens_seen: 0,
            loss,
            lr: 1e-3,
            grad_norm: 1.0,
            clipped,
            tokens_per_sec: 0.0,
            step_time_ms: 0.0,
            elapsed_secs: 0.0,
        }
    }

    #[test]
    fn detects_spike_once() {
        let mut m = Monitor::new(10);
        for s in 0..100 {
            assert!(m.observe_step(&step(s, 3.0, false)).is_empty());
        }
        let a = m.observe_step(&step(100, 6.0, false));
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].code, "loss_spike");
        assert!(m.observe_step(&step(101, 6.0, false)).is_empty(), "rate limited");
    }

    #[test]
    fn detects_frequent_clipping() {
        let mut m = Monitor::new(0);
        let mut found = false;
        for s in 0..300 {
            found |= m.observe_step(&step(s, 3.0, true)).iter().any(|a| a.code == "frequent_clipping");
        }
        assert!(found);
    }

    #[test]
    fn detects_overfitting() {
        let mut m = Monitor::new(0);
        let ev = |step, val, train| EvalMetrics {
            step,
            val_loss: val,
            val_ppl: 0.0,
            train_loss: train,
            is_best: false,
            elapsed_secs: 0.0,
        };
        assert!(m.observe_eval(&ev(100, 2.0, 2.1)).is_empty());
        assert!(m.observe_eval(&ev(200, 2.1, 1.8)).is_empty());
        let a = m.observe_eval(&ev(300, 2.3, 1.5));
        assert!(a.iter().any(|a| a.code == "overfitting"), "{a:?}");
    }
}
