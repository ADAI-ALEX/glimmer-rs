//! `glimmer models`, `glimmer chat`, `glimmer generate`, `glimmer evaluate`, `glimmer tune`.

use crate::style::{self, bold, dim, heading, kv};
use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use glimmer_core::config::ModelConfig;
use glimmer_core::registry::{list_models, resolve_model};
use glimmer_core::time::{format_count, format_duration, iso8601};
use glimmer_core::{BackendKind, Workspace};
use glimmer_evaluation::{EvalOptions, EvalReport, ModelSource, comparison_markdown, evaluate_model};
use glimmer_inference::{ChatSession, Engine, SamplingConfig};
use glimmer_optimisation::{StudyConfig, StudyEvent, run_study};
use glimmer_tokenizer::chat::{Message, render_prompt};
use glimmer_training::{TrainControl, TrainEvent};
use std::io::{BufRead, Write};
use std::sync::Arc;

#[derive(Subcommand, Debug)]
pub enum ModelsCmd {
    /// List trained models in models/.
    List,
    /// Show a model's architecture, parameters and training lineage.
    Inspect { name: String },
}

pub fn run_models(ws: &Workspace, cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::List => {
            heading("models");
            let models = list_models(&ws.models_dir());
            if models.is_empty() {
                println!("  {}", dim("no trained models yet — `glimmer train --model stevie`"));
            }
            for m in models {
                let desc = m.card.as_ref().map(|c| c.description.clone()).unwrap_or_default();
                println!(
                    "  {:<12} {:>7} params  ctx {:<5} {}",
                    bold(&m.name),
                    format_count(m.config.param_count() as u64),
                    m.config.context_length,
                    dim(&desc)
                );
            }
            println!("\n  {}", dim("presets for new models: micro, tiny, small (see configs/training/)"));
            for p in glimmer_core::config::PRESETS {
                let c = ModelConfig { vocab_size: 8192, ..ModelConfig::preset(p).unwrap() };
                println!(
                    "    {:<6} {:>6} params · {} layers · d_model {} · {} heads · ctx {}",
                    p,
                    format_count(c.param_count() as u64),
                    c.n_layers,
                    c.d_model,
                    c.n_heads,
                    c.context_length
                );
            }
        }
        ModelsCmd::Inspect { name } => {
            let m = resolve_model(&ws.models_dir(), &ws.root, &name)?;
            let c = &m.config;
            heading(&format!("model {}", m.name));
            if let Some(card) = &m.card {
                println!("  {}\n", card.description);
            }
            kv("parameters", format!("{} ({})", format_count(c.param_count() as u64), c.param_count()));
            kv(
                "architecture",
                format!(
                    "decoder-only transformer · {} layers · d_model {} · {} heads (dim {}) · d_ff {}",
                    c.n_layers,
                    c.d_model,
                    c.n_heads,
                    c.head_dim(),
                    c.d_ff
                ),
            );
            kv(
                "components",
                format!("{:?} · {:?} · RoPE θ={} · tied embeddings: {}", c.norm, c.ffn, c.rope_theta, c.tie_embeddings),
            );
            kv("context", format!("{} tokens", c.context_length));
            kv("vocabulary", format!("{} (byte-level BPE)", c.vocab_size));
            kv("files", ws.relative(&m.dir));
            if let Some(card) = &m.card {
                kv("hardware", card.hardware.clone().unwrap_or_default());
                kv("created", iso8601(card.created_unix));
                if let Some(s) = &card.system_prompt {
                    kv("system prompt", s);
                }
                println!("\n  {}", bold("training lineage"));
                for (i, s) in card.stages.iter().enumerate() {
                    println!(
                        "    {}. {:<9} {} steps · {} tokens · best val {} · {} · {}",
                        i + 1,
                        s.stage,
                        s.steps,
                        format_count(s.tokens),
                        s.best_val_loss.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into()),
                        format_duration(s.wall_secs),
                        s.backend
                    );
                    println!(
                        "       {}",
                        dim(&format!(
                            "dataset {} · run {} · from {}",
                            s.dataset,
                            s.run_id,
                            s.initialised_from.clone().unwrap_or_else(|| "random initialisation".into())
                        ))
                    );
                }
                kv("total tokens", format_count(card.total_tokens()));
            }
        }
    }
    Ok(())
}

#[derive(Args, Debug)]
pub struct ChatArgs {
    /// Model name in models/ or a model/checkpoint directory.
    #[arg(long, short, default_value = "stevie")]
    pub model: String,
    /// Plain line-based chat instead of the full-screen interface.
    #[arg(long)]
    pub plain: bool,
    #[arg(long, default_value = "auto")]
    pub backend: BackendKind,
    #[arg(long)]
    pub temperature: Option<f32>,
    #[arg(long)]
    pub seed: Option<u64>,
}

pub fn chat(ws: &Workspace, args: ChatArgs) -> Result<()> {
    if !args.plain {
        return glimmer_terminal::run(ws.clone(), glimmer_terminal::Screen::Chat, Some(args.model), args.backend);
    }
    let mut engine = Engine::load(ws, &args.model, args.backend)?;
    let mut sampling = SamplingConfig::default();
    if let Some(t) = args.temperature {
        sampling.temperature = t;
    }
    if let Some(s) = args.seed {
        sampling.seed = s;
    }
    let system = engine.card.as_ref().and_then(|c| c.system_prompt.clone());
    let mut session = ChatSession::new(&engine.name, system, sampling);
    println!(
        "{} {} {}",
        style::gum("❦"),
        bold(&engine.name),
        dim(&format!("· {} · type /quit to exit", engine.device.label()))
    );
    println!("{} {}\n", style::gum("stevie ›"), glimmer_terminal::app::WELCOME);
    let stdin = std::io::stdin();
    loop {
        print!("{} ", style::info("you ›"));
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/quit" | "/exit" => break,
            "/clear" => {
                session.clear();
                engine.reset_cache();
                println!("{}", dim("(cleared)"));
                continue;
            }
            _ => {}
        }
        print!("{} ", style::gum("stevie ›"));
        std::io::stdout().flush()?;
        let (g, _) = session.respond(&mut engine, line, |piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
            true
        })?;
        println!(
            "\n{}\n",
            dim(&format!(
                "{} tokens · {:.1} tok/s · {:.2}s",
                g.stats.generated_tokens,
                g.stats.decode_tokens_per_sec,
                g.stats.total_ms / 1e3
            ))
        );
    }
    Ok(())
}

#[derive(Args, Debug)]
pub struct GenerateArgs {
    pub prompt: String,
    #[arg(long, short, default_value = "stevie")]
    pub model: String,
    /// Treat the prompt as a chat message (default) or raw text continuation.
    #[arg(long)]
    pub raw: bool,
    #[arg(long, default_value = "auto")]
    pub backend: BackendKind,
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    #[arg(long, default_value_t = 120)]
    pub max_tokens: usize,
}

pub fn generate(ws: &Workspace, a: GenerateArgs) -> Result<()> {
    let mut engine = Engine::load(ws, &a.model, a.backend)?;
    let ids = if a.raw {
        engine.tokenizer.encode(&a.prompt)
    } else {
        let mut msgs = Vec::new();
        if let Some(s) = engine.card.as_ref().and_then(|c| c.system_prompt.clone()) {
            msgs.push(Message::system(s));
        }
        msgs.push(Message::user(a.prompt.clone()));
        render_prompt(&engine.tokenizer, &msgs)
    };
    let cfg =
        SamplingConfig { temperature: a.temperature, seed: a.seed, max_new_tokens: a.max_tokens, ..Default::default() };
    let g = engine.generate(&ids, &cfg, |p| {
        print!("{p}");
        let _ = std::io::stdout().flush();
        true
    })?;
    println!(
        "\n{}",
        dim(&format!(
            "[{:?} · {} tokens · {:.1} tok/s]",
            g.stop, g.stats.generated_tokens, g.stats.decode_tokens_per_sec
        ))
    );
    Ok(())
}

#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Models or checkpoint directories to evaluate (default: stevie).
    #[arg(long = "model", short)]
    pub models: Vec<String>,
    /// Also evaluate an untrained model with the same architecture (the "before" baseline).
    #[arg(long)]
    pub baseline: bool,
    /// Labels for the models, in order (default: their names).
    #[arg(long = "label")]
    pub labels: Vec<String>,
    #[arg(long, default_value = "datasets/stevie/eval_prompts.toml")]
    pub prompts: String,
    #[arg(long, default_value_t = 1)]
    pub samples: usize,
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,
    #[arg(long, default_value = "auto")]
    pub backend: BackendKind,
    /// Output name under runs/eval/.
    #[arg(long, default_value = "stevie")]
    pub name: String,
}

pub fn evaluate(ws: &Workspace, a: EvalArgs) -> Result<()> {
    let device = glimmer_models::backend::select(a.backend)?;
    let prompts = glimmer_data::stevie::load_eval_prompts(&ws.resolve(&a.prompts))?;
    let models = if a.models.is_empty() { vec!["stevie".to_string()] } else { a.models.clone() };
    let mut sources: Vec<(String, ModelSource)> = Vec::new();
    if a.baseline {
        let first = resolve_model(&ws.models_dir(), &ws.root, &models[0])?;
        sources.push((
            "random init".into(),
            ModelSource::RandomInit { config: first.config.clone(), tokenizer: first.tokenizer(), seed: 1337 },
        ));
    }
    for (i, m) in models.iter().enumerate() {
        let reg = resolve_model(&ws.models_dir(), &ws.root, m)?;
        let label = a.labels.get(i).cloned().unwrap_or_else(|| reg.name.clone());
        sources.push((label, ModelSource::Dir(reg.dir.clone())));
    }
    let sampling = SamplingConfig { temperature: a.temperature, seed: 42, max_new_tokens: 96, ..Default::default() };
    let sft = glimmer_data::build::dataset_dir(ws, "stevie-sft");
    let pre = glimmer_data::build::dataset_dir(ws, "pretrain-mix");
    let mut reports: Vec<EvalReport> = Vec::new();
    for (label, src) in &sources {
        heading(&format!("evaluating {label}"));
        let opts = EvalOptions {
            label: label.clone(),
            prompts: prompts.clone(),
            samples_per_prompt: a.samples,
            sampling: sampling.clone(),
            system_prompt: Some(glimmer_data::stevie::SYSTEM_PROMPT.to_string()),
            sft_dataset: sft.join("test.sft").is_file().then(|| sft.clone()),
            pretrain_dataset: pre.join("test.bin").is_file().then(|| pre.clone()),
            pretrain_batches: 24,
        };
        let r = evaluate_model(src, &opts, &device, &mut |p| {
            print!("\r  {} {:<40}", dim("generating"), p);
            let _ = std::io::stdout().flush();
        })?;
        println!();
        kv(
            "conversation test",
            r.sft_test_loss.map(|v| format!("loss {v:.4} · ppl {:.2}", v.exp())).unwrap_or_else(|| "—".into()),
        );
        kv(
            "general text test",
            r.pretrain_test_loss.map(|v| format!("loss {v:.4} · ppl {:.2}", v.exp())).unwrap_or_else(|| "—".into()),
        );
        kv("ends turn", format!("{:.0}%", r.aggregate.end_of_turn_rate * 100.0));
        kv("aussie markers", format!("{:.0}% of replies", r.aggregate.australian_marker_rate * 100.0));
        kv(
            "continuity",
            r.aggregate.continuity_rate.map(|v| format!("{:.0}%", v * 100.0)).unwrap_or_else(|| "—".into()),
        );
        reports.push(r);
    }
    let out = ws.runs_dir().join("eval");
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join(format!("{}.json", a.name)), serde_json::to_string_pretty(&reports)?)?;
    std::fs::write(out.join(format!("{}.md", a.name)), comparison_markdown(&reports, ws))?;
    println!("\n  {} {}", dim("report →"), ws.relative(out.join(format!("{}.md", a.name))));
    Ok(())
}

#[derive(Args, Debug)]
pub struct TuneArgs {
    /// Study config (configs/tuning/<name>.toml).
    pub config: String,
}

pub fn tune(ws: &Workspace, a: TuneArgs) -> Result<()> {
    let p = {
        let p = ws.resolve(&a.config);
        if p.is_file() { p } else { ws.configs_dir().join("tuning").join(format!("{}.toml", a.config)) }
    };
    let cfg = StudyConfig::load(&p).with_context(|| format!("loading {}", p.display()))?;
    if cfg.space.is_empty() {
        bail!("study has an empty search space");
    }
    heading(&format!("study {} — {:?} search, {} trials, rungs {:?}", cfg.name, cfg.sampler, cfg.trials, cfg.rungs()));
    let control = Arc::new(TrainControl::new());
    let c2 = control.clone();
    let _ = ctrlc::set_handler(move || c2.stop());
    let summary = run_study(ws, &cfg, &control, &mut |e| match e {
        StudyEvent::TrialStarted { trial, rung, budget, params } => {
            let ps: Vec<String> = params
                .iter()
                .map(|(k, v)| format!("{k}={}", glimmer_optimisation::study::format_param(k, *v)))
                .collect();
            println!("\n  {} trial {trial} rung {rung} → {budget} steps  {}", style::accent("▸"), dim(&ps.join(" ")));
        }
        StudyEvent::Train { event: TrainEvent::Step(s), .. } => {
            print!("\r    step {:>5}  loss {:.4}  {:.0} tok/s   ", s.step, s.loss, s.tokens_per_sec);
            let _ = std::io::stdout().flush();
        }
        StudyEvent::TrialFinished(r) => {
            println!(
                "\n    {} val loss {}",
                style::info("◆"),
                bold(&r.val_loss.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into()))
            );
        }
        StudyEvent::Promoted { rung, trials } => {
            println!("\n  {} promoted to rung {rung}: {:?}", style::ok("↑"), trials)
        }
        _ => {}
    })?;
    heading("study complete");
    kv("best trial", summary.best_trial.map(|t| t.to_string()).unwrap_or_else(|| "—".into()));
    kv("best val loss", summary.best_val_loss.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into()));
    for (k, v) in &summary.best_params {
        kv(&format!("  {k}"), glimmer_optimisation::study::format_param(k, *v));
    }
    kv("wall time", format_duration(summary.wall_secs));
    kv("report", ws.relative(summary.dir.join("report.md")));
    Ok(())
}
