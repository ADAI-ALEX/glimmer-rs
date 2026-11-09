//! Metric records written by the trainer and consumed by the CLI, the TUI and the evaluator.
//!
//! Records are stored one JSON object per line in `runs/<id>/metrics.jsonl`, so a run can be
//! inspected (or plotted) while it is still training and after the process exits.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// One optimiser step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepMetrics {
    pub step: usize,
    pub tokens_seen: u64,
    /// Mean training loss over the supervised tokens of this step (all micro-batches).
    pub loss: f64,
    pub lr: f64,
    /// Global L2 norm of the gradient before clipping.
    pub grad_norm: f64,
    /// Whether clipping scaled this step's gradient.
    pub clipped: bool,
    pub tokens_per_sec: f64,
    pub step_time_ms: f64,
    pub elapsed_secs: f64,
}

/// One validation pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalMetrics {
    pub step: usize,
    pub val_loss: f64,
    pub val_ppl: f64,
    /// Smoothed training loss at the time of evaluation (for overfitting detection).
    pub train_loss: f64,
    pub is_best: bool,
    pub elapsed_secs: f64,
}

/// A generated sample captured during training (genuine model output).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SampleRecord {
    pub step: usize,
    pub prompt: String,
    pub completion: String,
}

/// Something noteworthy that is not a number: checkpoints, advice, automatic actions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventRecord {
    pub step: usize,
    pub kind: EventKind,
    pub message: String,
    pub unix: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Info,
    Checkpoint,
    /// An automatic control acted (e.g. early stopping). Always logged for reproducibility.
    AutoAction,
    /// A recommendation from the training monitor; never applied silently.
    Advice,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MetricRecord {
    Step(StepMetrics),
    Eval(EvalMetrics),
    Sample(SampleRecord),
    Event(EventRecord),
}

/// Append-only JSONL writer.
pub struct MetricsWriter {
    file: std::io::BufWriter<std::fs::File>,
}

impl MetricsWriter {
    pub fn append(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file: std::io::BufWriter::new(file),
        })
    }

    pub fn write(&mut self, record: &MetricRecord) -> Result<()> {
        serde_json::to_writer(&mut self.file, record)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        Ok(())
    }
}

/// Read every record of a metrics file, skipping malformed lines (e.g. a partially written last
/// line while training is still running).
pub fn read_metrics(path: impl AsRef<Path>) -> Result<Vec<MetricRecord>> {
    let file = std::fs::File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if let Ok(rec) = serde_json::from_str::<MetricRecord>(&line) {
            out.push(rec);
        }
    }
    Ok(out)
}

/// Exponential moving average helper used for smoothed loss displays.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Ema {
    pub value: Option<f64>,
    pub alpha: f64,
}

impl Ema {
    pub fn new(alpha: f64) -> Self {
        Self { value: None, alpha }
    }
    pub fn update(&mut self, x: f64) -> f64 {
        let v = match self.value {
            None => x,
            Some(prev) => self.alpha * x + (1.0 - self.alpha) * prev,
        };
        self.value = Some(v);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_roundtrip() {
        let dir = std::env::temp_dir().join(format!("glimmer-metrics-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.jsonl");
        let _ = std::fs::remove_file(&path);
        let mut w = MetricsWriter::append(&path).unwrap();
        let rec = MetricRecord::Step(StepMetrics {
            step: 1,
            tokens_seen: 100,
            loss: 2.5,
            lr: 1e-3,
            grad_norm: 0.7,
            clipped: false,
            tokens_per_sec: 1000.0,
            step_time_ms: 10.0,
            elapsed_secs: 1.0,
        });
        w.write(&rec).unwrap();
        w.write(&MetricRecord::Event(EventRecord {
            step: 1,
            kind: EventKind::Checkpoint,
            message: "saved".into(),
            unix: 0,
        }))
        .unwrap();
        drop(w);
        // Simulate a torn final line.
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"type\":\"step\",\"st")
            .unwrap();
        let recs = read_metrics(&path).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0], rec);
    }
}
