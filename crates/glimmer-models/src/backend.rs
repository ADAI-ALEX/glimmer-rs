//! Compute backend selection and verification.
//!
//! GLIMMER never claims acceleration it cannot demonstrate. [`select`] opens the requested
//! backend and runs a numerical self-test: a matrix multiplication is executed on the device and
//! compared with a reference computed in plain Rust on the host. The returned [`ComputeDevice`]
//! records the adapter identity reported by the driver, the self-test error and timing. If Metal
//! is requested (or `auto`) but fails to initialise or verify, the reason is recorded and the
//! CPU backend is used instead — the UI shows the fallback explicitly.

use anyhow::{Result, anyhow, bail};
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use glimmer_core::BackendKind;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Verification {
    /// Max absolute error of a 128×96·96×80 matmul against the host reference.
    pub max_abs_error: f32,
    pub passed: bool,
    /// Wall time of the first (cold) device round-trip, including kernel compilation.
    pub cold_ms: f64,
    /// Wall time of a warm round-trip.
    pub warm_ms: f64,
}

#[derive(Debug, Clone)]
pub struct ComputeDevice {
    pub device: Device,
    /// Resolved backend (never `Auto`).
    pub kind: BackendKind,
    /// Adapter name as reported by the driver (e.g. "Apple M1 Pro"), or a CPU description.
    pub adapter: String,
    /// How kernels reach the hardware (e.g. "CubeCL → MSL → Metal (wgpu)").
    pub target: Option<String>,
    pub verification: Verification,
    /// Why a fallback happened, if it did.
    pub fallback_reason: Option<String>,
}

impl ComputeDevice {
    pub fn label(&self) -> String {
        match self.kind {
            BackendKind::Metal => format!("Metal · {}", self.adapter),
            _ => format!("CPU · {}", self.adapter),
        }
    }
}

fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    c
}

/// Run the numerical self-test on a device.
pub fn verify(device: &Device) -> Result<Verification> {
    let (m, k, n) = (128, 96, 80);
    let mut rng = glimmer_core::Rng::new(0xC0FFEE);
    let a: Vec<f32> = (0..m * k).map(|_| rng.uniform(-1.0, 1.0) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.uniform(-1.0, 1.0) as f32).collect();
    let expected = reference_matmul(&a, &b, m, k, n);
    let run = || -> Result<Vec<f32>> {
        let ta = Tensor::<2>::from_data(TensorData::new(a.clone(), [m, k]), device);
        let tb = Tensor::<2>::from_data(TensorData::new(b.clone(), [k, n]), device);
        let c = ta.matmul(tb).into_data();
        c.try_to_vec::<f32>().map_err(|e| anyhow!("reading result: {e:?}"))
    };
    let t0 = Instant::now();
    let got = run()?;
    let cold_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let got2 = run()?;
    let warm_ms = t1.elapsed().as_secs_f64() * 1e3;
    let err = got
        .iter()
        .chain(got2.iter())
        .zip(expected.iter().chain(expected.iter()))
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    Ok(Verification { max_abs_error: err, passed: err < 1e-3 && got.len() == expected.len(), cold_ms, warm_ms })
}

fn cpu_device() -> Result<ComputeDevice> {
    let device = Device::flex();
    let verification = verify(&device)?;
    if !verification.passed {
        bail!("CPU backend failed its numerical self-test (max error {})", verification.max_abs_error);
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let amx = if cfg!(target_os = "macos") { ", Apple AMX" } else { "" };
    Ok(ComputeDevice {
        device,
        kind: BackendKind::Cpu,
        adapter: format!("Burn Flex ({threads} threads, SIMD{amx})"),
        target: None,
        verification,
        fallback_reason: None,
    })
}

fn metal_device() -> Result<ComputeDevice> {
    if !cfg!(target_os = "macos") {
        bail!("Metal is only available on macOS");
    }
    // Device initialisation panics inside wgpu if no adapter exists; contain it.
    let probe = std::panic::catch_unwind(|| {
        let device = Device::metal(DeviceKind::DefaultDevice);
        let identity = device.identity();
        let verification = verify(&device);
        (device, identity, verification)
    });
    let (device, identity, verification) =
        probe.map_err(|_| anyhow!("Metal device initialisation panicked (no compatible GPU?)"))?;
    let verification = verification?;
    if !verification.passed {
        bail!("Metal backend failed its numerical self-test (max error {})", verification.max_abs_error);
    }
    // `Device::metal` pins wgpu's Metal backend with CubeCL's native MSL compiler; device
    // initialisation fails (rather than silently falling back to WGSL) if MSL is unavailable.
    // The runtime's identity "fingerprint" is only a cache namespace, so it is not reported.
    let adapter = identity.map(|id| id.name).unwrap_or_else(|| "unknown Metal adapter".into());
    let target = Some("CubeCL kernels → MSL → Metal (wgpu), with kernel fusion + autotune".to_string());
    Ok(ComputeDevice { device, kind: BackendKind::Metal, adapter, target, verification, fallback_reason: None })
}

/// Open and verify a backend. `Auto` prefers Metal and falls back to the CPU with a recorded
/// reason; an explicit `Metal` request fails loudly instead of silently using the CPU.
pub fn select(kind: BackendKind) -> Result<ComputeDevice> {
    match kind {
        BackendKind::Cpu => cpu_device(),
        BackendKind::Metal => metal_device(),
        BackendKind::Auto => match metal_device() {
            Ok(d) => Ok(d),
            Err(e) => {
                let mut cpu = cpu_device()?;
                cpu.fallback_reason = Some(e.to_string());
                Ok(cpu)
            }
        },
    }
}

/// Open a backend for **single-sequence inference** (chat, generation, evaluation sampling).
///
/// For `Auto`, this prefers the CPU backend: autoregressive decoding of a small model is a long
/// chain of tiny, latency-bound kernels whose shapes change every token (the KV cache grows), so
/// the GPU pays kernel-launch and JIT-compilation costs that dominate its arithmetic advantage.
/// On an M1 Pro, Flex with Apple AMX decodes STEVIE far faster than Metal (see
/// `docs/performance.md`). Training keeps using Metal. An explicit backend is always honoured.
pub fn select_for_inference(kind: BackendKind) -> Result<ComputeDevice> {
    match kind {
        BackendKind::Auto => {
            let mut d = cpu_device()?;
            d.fallback_reason = None;
            Ok(d)
        }
        other => select(other),
    }
}

/// Seed the device RNG (used by dropout). Weight initialisation uses GLIMMER's own host RNG.
pub fn seed(device: &Device, seed: u64) {
    device.seed(seed);
}
