//! Hardware inspection.
//!
//! Everything reported here is measured or queried from the operating system at runtime.
//! Fields that cannot be determined are `None` and are displayed as unavailable — never guessed.

use serde::{Deserialize, Serialize};
use std::process::Command;
use sysinfo::{MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub cores: Option<u32>,
    /// Metal feature set reported by the OS (e.g. "Metal 4"), when available.
    pub metal_support: Option<String>,
    /// Apple Silicon GPUs share system memory with the CPU.
    pub unified_memory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareReport {
    pub os_name: String,
    pub os_version: Option<String>,
    pub kernel_version: Option<String>,
    pub arch: String,
    pub cpu_brand: Option<String>,
    pub logical_cores: usize,
    pub performance_cores: Option<usize>,
    pub efficiency_cores: Option<usize>,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub gpu: Option<GpuInfo>,
    pub rustc_version: String,
    pub build_profile: String,
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn sysctl(key: &str) -> Option<String> {
    run("sysctl", &["-n", key])
}

/// Extract `"key" = value` from `ioreg` text output.
fn ioreg_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\" = ");
    let start = text.find(&needle)? + needle.len();
    let rest = &text[start..];
    let end = rest.find(['\n', ',', '}']).unwrap_or(rest.len());
    Some(rest[..end].trim().trim_matches('"'))
}

fn detect_apple_gpu(with_metal_query: bool) -> Option<GpuInfo> {
    let text = run("ioreg", &["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])?;
    let name = ioreg_value(&text, "model")?.to_string();
    let cores = ioreg_value(&text, "gpu-core-count").and_then(|v| v.parse().ok());
    let metal_support = if with_metal_query {
        run("system_profiler", &["SPDisplaysDataType"]).and_then(|t| {
            t.lines().find_map(|l| l.trim().strip_prefix("Metal Support:").map(|v| v.trim().to_string())).or_else(
                || t.lines().find_map(|l| l.trim().strip_prefix("Metal Family:").map(|v| v.trim().to_string())),
            )
        })
    } else {
        None
    };
    Some(GpuInfo { unified_memory: name.starts_with("Apple"), name, cores, metal_support })
}

impl HardwareReport {
    /// Inspect the machine. `thorough` additionally queries `system_profiler` for the Metal
    /// feature set, which takes around a second on macOS.
    pub fn detect(thorough: bool) -> Self {
        let mem = System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::everything())).ok();
        let (total, available) = mem.as_ref().map(|s| (s.total_memory(), s.available_memory())).unwrap_or((0, 0));

        let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let macos = cfg!(target_os = "macos");

        let (os_version, cpu_brand, perf, eff, gpu) = if macos {
            (
                run("sw_vers", &["-productVersion"]).map(|v| format!("macOS {v}")),
                sysctl("machdep.cpu.brand_string"),
                sysctl("hw.perflevel0.physicalcpu").and_then(|v| v.parse().ok()),
                sysctl("hw.perflevel1.physicalcpu").and_then(|v| v.parse().ok()),
                detect_apple_gpu(thorough),
            )
        } else {
            (
                System::long_os_version().ok(),
                std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|t| {
                    t.lines()
                        .find(|l| l.starts_with("model name"))
                        .and_then(|l| l.split(':').nth(1))
                        .map(|s| s.trim().to_string())
                }),
                None,
                None,
                None,
            )
        };

        HardwareReport {
            os_name: System::name().unwrap_or_else(|_| std::env::consts::OS.to_string()),
            os_version,
            kernel_version: System::kernel_version().ok(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_brand,
            logical_cores: logical,
            performance_cores: perf,
            efficiency_cores: eff,
            total_memory_bytes: total,
            available_memory_bytes: available,
            gpu,
            rustc_version: crate::RUSTC_VERSION.to_string(),
            build_profile: crate::BUILD_PROFILE.to_string(),
        }
    }

    pub fn total_memory_gib(&self) -> f64 {
        self.total_memory_bytes as f64 / (1u64 << 30) as f64
    }

    pub fn has_apple_gpu(&self) -> bool {
        self.gpu.as_ref().is_some_and(|g| g.unified_memory)
    }

    /// Recommend a model preset for local training, with a human-readable rationale.
    ///
    /// The goal is meaningful progress in a few hours, not the largest model that fits: on a
    /// laptop GPU, a model that sees more tokens usually beats a bigger under-trained one.
    pub fn recommend_preset(&self) -> (&'static str, String) {
        let gib = self.total_memory_gib();
        match (self.has_apple_gpu(), gib) {
            (true, g) if g >= 31.0 => (
                "small",
                format!(
                    "{g:.0} GiB unified memory and an Apple GPU: `small` (~30M params) fits comfortably; \
                     use `tiny` if you want results within a couple of hours."
                ),
            ),
            (true, g) if g >= 12.0 => (
                "tiny",
                format!(
                    "{g:.0} GiB unified memory and an Apple GPU: `tiny` (~14M params) trains on tens of \
                     millions of tokens in a few hours, leaving memory for the OS and the TUI."
                ),
            ),
            (true, g) => ("micro", format!("Only {g:.0} GiB memory: `micro` keeps training within memory limits.")),
            (false, g) => (
                "micro",
                format!(
                    "No supported GPU detected ({g:.0} GiB RAM): CPU training is roughly an order of \
                     magnitude slower, so start with `micro` and scale up deliberately."
                ),
            ),
        }
    }
}

/// Resident memory of the current process in bytes (measured via the OS).
pub struct ProcessMemory {
    sys: Option<System>,
    pid: Option<Pid>,
}

impl ProcessMemory {
    pub fn new() -> Self {
        let sys = System::new().ok();
        let pid = sysinfo::get_current_pid().ok();
        Self { sys, pid }
    }

    pub fn resident_bytes(&mut self) -> Option<u64> {
        let pid = self.pid?;
        let sys = self.sys.as_mut()?;
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            false,
            ProcessRefreshKind::nothing().with_memory(),
        );
        sys.process(pid).map(|p| p.memory())
    }
}

impl Default for ProcessMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// System memory snapshot: (used, total) bytes.
pub fn system_memory() -> Option<(u64, u64)> {
    let s = System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::everything())).ok()?;
    Some((s.used_memory(), s.total_memory()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ioreg_values() {
        let text = "  \"model\" = \"Apple M1 Pro\"\n  \"gpu-core-count\" = 16\n";
        assert_eq!(ioreg_value(text, "model"), Some("Apple M1 Pro"));
        assert_eq!(ioreg_value(text, "gpu-core-count"), Some("16"));
        assert_eq!(ioreg_value(text, "missing"), None);
    }

    #[test]
    fn detect_reports_real_memory() {
        let r = HardwareReport::detect(false);
        assert!(r.total_memory_bytes > 0);
        assert!(r.logical_cores >= 1);
        let (preset, why) = r.recommend_preset();
        assert!(crate::config::PRESETS.contains(&preset));
        assert!(!why.is_empty());
    }
}
