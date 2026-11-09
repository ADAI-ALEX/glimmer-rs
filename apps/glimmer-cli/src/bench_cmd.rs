//! `glimmer bench` and `glimmer doctor`.

use crate::style::{self, bold, dim, heading, kv};
use anyhow::Result;
use clap::Args;
use glimmer_bench::suite::{self, Ctx};
use glimmer_bench::Measurement;
use glimmer_core::config::ModelConfig;
use glimmer_core::hardware::HardwareReport;
use glimmer_core::time::format_bytes;
use glimmer_core::{BackendKind, Workspace};
use glimmer_models::backend;
use serde::Serialize;

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Backends to benchmark: auto, metal, cpu, or all.
    #[arg(long, default_value = "all")]
    pub backend: String,
    /// Model preset for model-level benchmarks.
    #[arg(long, default_value = "tiny")]
    pub preset: String,
    /// Fewer iterations (for a quick check).
    #[arg(long)]
    pub quick: bool,
    /// Only run a training-throughput sweep over these batch sizes (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub sweep_batch: Vec<usize>,
    /// Context lengths for the sweep (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub sweep_context: Vec<usize>,
    /// Write results as JSON to this path (default: runs/bench/<stamp>.json).
    #[arg(long)]
    pub out: Option<String>,
}

#[derive(Serialize)]
struct BenchReport {
    hardware: HardwareReport,
    preset: ModelConfig,
    measurements: Vec<Measurement>,
    unix: u64,
}

fn print(m: &Measurement) {
    let tp = m
        .throughput()
        .map(|(v, u)| format!("{:>10.1} {u}", v))
        .unwrap_or_default();
    println!(
        "  {:<28} {:<34} {:>9.2} ms ± {:>6.2}  (min {:>8.2})  {}",
        m.name,
        dim(&m.config),
        m.mean_ms,
        m.std_ms,
        m.min_ms,
        style::accent(&tp)
    );
}

pub fn run(ws: &Workspace, args: BenchArgs) -> Result<()> {
    let hw = HardwareReport::detect(false);
    let kinds: Vec<BackendKind> = match args.backend.as_str() {
        "all" => vec![BackendKind::Metal, BackendKind::Cpu],
        other => vec![other.parse()?],
    };
    let mut preset = ModelConfig::preset(&args.preset).ok_or_else(|| anyhow::anyhow!("unknown preset"))?;
    let (warmup, iters) = if args.quick { (2, 5) } else { (5, 20) };
    let mut all = Vec::new();
    for kind in kinds {
        let dev = match backend::select(kind) {
            Ok(d) => d,
            Err(e) => {
                println!("  {} {kind}: {e}", style::warn("skipped"));
                continue;
            }
        };
        heading(&format!("benchmarks on {}", dev.label()));
        let c = Ctx { device: &dev, warmup, iterations: iters };
        if !args.sweep_batch.is_empty() {
            let contexts = if args.sweep_context.is_empty() { vec![preset.context_length] } else { args.sweep_context.clone() };
            for &ctx in &contexts {
                for &b in &args.sweep_batch {
                    let cfg = ModelConfig { context_length: ctx, ..preset.clone() };
                    let m = suite::train_step(&Ctx { device: &dev, warmup: 3, iterations: if args.quick { 5 } else { 10 } }, &cfg, b);
                    print(&m);
                    all.push(m);
                }
            }
            continue;
        }
        let is_cpu = kind == BackendKind::Cpu;
        for n in [512, 1024, 2048] {
            let m = suite::matmul(&c, n);
            print(&m);
            all.push(m);
        }
        let h = preset.n_heads;
        let dh = preset.head_dim();
        for t in [256, 512] {
            let m = suite::attention_fwd(&c, 8, h, t, dh);
            print(&m);
            all.push(m);
        }
        if is_cpu {
            // Keep CPU runs short: the CPU backend is an order of magnitude slower.
            preset.context_length = preset.context_length.min(256);
        }
        let batch = if is_cpu { 4 } else { 16 };
        let m = suite::forward(&c, &preset, batch);
        print(&m);
        all.push(m);
        let tc = Ctx { device: &dev, warmup: 3, iterations: if is_cpu { 3 } else { iters.min(10) } };
        let m = suite::train_step(&tc, &preset, batch);
        print(&m);
        all.push(m);
        let (a, b) = suite::decode(&c, &preset, 64)?;
        print(&a);
        print(&b);
        all.push(a);
        all.push(b);
        let (s, l) = suite::checkpoint_io(&c, &preset, &ws.runs_dir())?;
        print(&s);
        print(&l);
        all.push(s);
        all.push(l);
        preset = ModelConfig::preset(&args.preset).unwrap();
    }
    // Host-side benchmarks (backend-independent).
    let train_bin = ws.datasets_dir().join("pretrain-mix").join("train.bin");
    let tok_path = ws.tokenizers_dir().join("glimmer-8k.json");
    if train_bin.is_file() || tok_path.is_file() {
        heading("host-side benchmarks");
        let cpu = backend::select(BackendKind::Cpu)?;
        let c = Ctx { device: &cpu, warmup, iterations: iters };
        if train_bin.is_file() {
            let m = suite::data_loading(&c, &train_bin, 16, preset.context_length)?;
            print(&m);
            all.push(m);
        }
        if tok_path.is_file() {
            let tok = glimmer_tokenizer::Tokenizer::load(&tok_path)?;
            let text = sample_text(ws).unwrap_or_else(|| "G'day mate, how ya going? ".repeat(4000));
            let m = suite::tokenizer(&c, &tok, &text);
            print(&m);
            all.push(m);
        }
    }
    let out = args.out.map(|o| ws.resolve(o)).unwrap_or_else(|| {
        ws.runs_dir().join("bench").join(format!("{}.json", glimmer_core::time::compact_stamp(glimmer_core::time::now_unix())))
    });
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p)?;
    }
    let report = BenchReport { hardware: hw, preset: ModelConfig::preset(&args.preset).unwrap(), measurements: all, unix: glimmer_core::time::now_unix() };
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("\n  {} {}", dim("results →"), ws.relative(&out));
    Ok(())
}

fn sample_text(ws: &Workspace) -> Option<String> {
    let docs = glimmer_data::corpus::read_documents(ws, "gutenberg-au").ok()?;
    let mut s = String::new();
    for d in docs.iter().take(60) {
        s.push_str(&d.text);
        s.push('\n');
    }
    Some(s)
}

#[derive(Args, Debug)]
pub struct DoctorArgs {
    /// Print the report as JSON.
    #[arg(long)]
    pub json: bool,
}

pub fn doctor(ws: &Workspace, args: DoctorArgs) -> Result<()> {
    let hw = HardwareReport::detect(true);
    let metal = backend::select(BackendKind::Metal);
    let cpu = backend::select(BackendKind::Cpu);
    if args.json {
        let v = serde_json::json!({
            "hardware": hw,
            "metal": metal.as_ref().map(|d| serde_json::json!({"adapter": d.adapter, "target": d.target, "verification": d.verification})).map_err(|e| e.to_string()),
            "cpu": cpu.as_ref().map(|d| serde_json::json!({"adapter": d.adapter, "verification": d.verification})).map_err(|e| e.to_string()),
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    heading("system");
    kv("os", format!("{} {}", hw.os_version.clone().unwrap_or(hw.os_name.clone()), dim(&hw.kernel_version.clone().unwrap_or_default())));
    kv("architecture", &hw.arch);
    kv("cpu", hw.cpu_brand.clone().unwrap_or_else(|| "unknown".into()));
    let cores = match (hw.performance_cores, hw.efficiency_cores) {
        (Some(p), Some(e)) => format!("{} ({p} performance + {e} efficiency)", hw.logical_cores),
        _ => hw.logical_cores.to_string(),
    };
    kv("cores", cores);
    kv("memory", format!("{} total · {} available", format_bytes(hw.total_memory_bytes), format_bytes(hw.available_memory_bytes)));
    match &hw.gpu {
        Some(g) => {
            kv("gpu", format!("{}{}", g.name, g.cores.map(|c| format!(" · {c} cores")).unwrap_or_default()));
            kv("metal support", g.metal_support.clone().unwrap_or_else(|| "unknown".into()));
            kv("unified memory", if g.unified_memory { "yes" } else { "no" });
        }
        None => kv("gpu", dim("none detected")),
    }
    kv("rust toolchain", &hw.rustc_version);
    kv("build profile", &hw.build_profile);
    kv("burn", "0.22.1 (Metal via wgpu/MSL · Flex CPU)");

    heading("compute backends (numerical self-test vs host reference)");
    match &metal {
        Ok(d) => {
            kv("metal", format!("{} {}", style::ok("✓ verified"), d.adapter));
            kv("  kernel pipeline", d.target.clone().unwrap_or_default());
            kv("  max abs error", format!("{:.2e}", d.verification.max_abs_error));
            kv("  first / warm run", format!("{:.0} ms / {:.2} ms", d.verification.cold_ms, d.verification.warm_ms));
        }
        Err(e) => kv("metal", format!("{} {e}", style::err("✗ unavailable"))),
    }
    match &cpu {
        Ok(d) => {
            kv("cpu", format!("{} {}", style::ok("✓ verified"), d.adapter));
            kv("  max abs error", format!("{:.2e}", d.verification.max_abs_error));
        }
        Err(e) => kv("cpu", format!("{} {e}", style::err("✗"))),
    }
    let selected = backend::select(BackendKind::Auto)?;
    kv("auto selects", bold(&selected.label()));
    if let Some(r) = &selected.fallback_reason {
        kv("fallback reason", style::warn(r));
    }

    heading("recommendation");
    let (preset, why) = hw.recommend_preset();
    let cfg = ModelConfig::preset(preset).unwrap();
    kv("model preset", format!("{} ({:.1}M parameters at vocab 8192)", bold(preset), ModelConfig { vocab_size: 8192, ..cfg }.param_count() as f64 / 1e6));
    println!("  {}", dim(&why));

    heading("workspace");
    kv("root", ws.root.display());
    let datasets = glimmer_data::DatasetManifest::list(ws.datasets_dir());
    kv("datasets", if datasets.is_empty() { dim("none — run `glimmer data pipeline`") } else { datasets.iter().map(|d| d.0.clone()).collect::<Vec<_>>().join(", ") });
    let models = glimmer_core::registry::list_models(&ws.models_dir());
    kv("models", if models.is_empty() { dim("none trained yet") } else { models.iter().map(|m| m.name.clone()).collect::<Vec<_>>().join(", ") });
    if let Some(s) = glimmer_core::gpu::sample() {
        kv("gpu utilisation now", format!("{:.0}%", s.device_utilization.unwrap_or(0.0)));
    }
    Ok(())
}
