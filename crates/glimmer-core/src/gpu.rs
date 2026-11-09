//! Live GPU counters.
//!
//! On macOS the Apple GPU driver publishes `PerformanceStatistics` in the IORegistry, readable
//! without elevated privileges. These are the same counters Activity Monitor uses. On any other
//! platform (or if the read fails) the sampler returns `None` and the UI shows "not measurable".

use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct GpuSample {
    /// Overall GPU busy percentage reported by the driver.
    pub device_utilization: Option<f64>,
    pub renderer_utilization: Option<f64>,
    pub tiler_utilization: Option<f64>,
    /// System memory currently in use by GPU allocations (bytes).
    pub in_use_memory: Option<u64>,
    /// System memory allocated by the GPU driver (bytes).
    pub allocated_memory: Option<u64>,
}

impl GpuSample {
    pub fn is_empty(&self) -> bool {
        self.device_utilization.is_none() && self.in_use_memory.is_none()
    }
}

fn stat(block: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"=");
    let start = block.find(&needle)? + needle.len();
    let rest = &block[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Parse the `PerformanceStatistics` dictionary out of `ioreg` output.
pub fn parse_performance_statistics(text: &str) -> Option<GpuSample> {
    let start = text.find("\"PerformanceStatistics\" = {")?;
    let block = &text[start..];
    let end = block.find('}')?;
    let block = &block[..=end];
    let sample = GpuSample {
        device_utilization: stat(block, "Device Utilization %"),
        renderer_utilization: stat(block, "Renderer Utilization %"),
        tiler_utilization: stat(block, "Tiler Utilization %"),
        in_use_memory: stat(block, "In use system memory").map(|v| v as u64),
        allocated_memory: stat(block, "Alloc system memory").map(|v| v as u64),
    };
    (!sample.is_empty()).then_some(sample)
}

/// Read one sample of GPU counters. Takes ~10–30 ms (spawns `ioreg`); call at most ~1 Hz.
pub fn sample() -> Option<GpuSample> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = Command::new("ioreg").args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]).output().ok()?;
    parse_performance_statistics(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_ioreg_output() {
        let text = r#"    | "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=940359680,"Tiler Utilization %"=68,"recoveryCount"=0,"Renderer Utilization %"=68,"Device Utilization %"=76,"In use system memory"=205832192}"#;
        let s = parse_performance_statistics(text).unwrap();
        assert_eq!(s.device_utilization, Some(76.0));
        assert_eq!(s.renderer_utilization, Some(68.0));
        assert_eq!(s.in_use_memory, Some(205_832_192));
        assert_eq!(s.allocated_memory, Some(940_359_680));
    }

    #[test]
    fn missing_block_is_none() {
        assert!(parse_performance_statistics("nothing here").is_none());
    }
}
