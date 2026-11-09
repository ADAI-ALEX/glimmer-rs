//! `glimmer` — command-line entry point for the GLIMMER laboratory.

mod bench_cmd;
mod data_cmd;
mod eval_cmd;
mod model_cmd;
mod style;
mod train_cmd;

use anyhow::Result;
use clap::{Parser, Subcommand};
use glimmer_core::Workspace;

#[derive(Parser, Debug)]
#[command(
    name = "glimmer",
    version,
    about = "GLIMMER — a Rust-native LLM training and experimentation laboratory",
    long_about = "Build, train, evaluate and chat with small language models — trained from scratch, \
                  entirely in Rust.\n\nRun `glimmer` with no arguments to open the interactive laboratory."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Acquire and prepare training data.
    #[command(subcommand)]
    Data(data_cmd::DataCmd),
    /// Train or inspect tokenizers.
    #[command(subcommand)]
    Tokenizer(data_cmd::TokenizerCmd),
    /// List and inspect prepared datasets.
    #[command(subcommand)]
    Datasets(data_cmd::DatasetsCmd),
    /// List dataset sources with licences and status.
    Sources,
    /// Train a model headlessly (pretraining or fine-tuning).
    Train(train_cmd::TrainArgs),
    /// List training runs.
    Runs,
    /// Benchmark kernels, training and inference on the available backends.
    Bench(bench_cmd::BenchArgs),
    /// Inspect hardware and verify compute backends.
    Doctor(bench_cmd::DoctorArgs),
    /// Open the interactive laboratory (same as running `glimmer` with no arguments).
    Tui {
        #[arg(long, default_value = "auto")]
        backend: glimmer_core::BackendKind,
        /// Screen to open: home, train, chat, models, datasets, experiments, evaluation, hardware, settings.
        #[arg(long, default_value = "home")]
        screen: String,
        /// Model to load for chat.
        #[arg(long)]
        model: Option<String>,
    },
    /// Chat with a trained model (full-screen, or --plain for a simple REPL).
    Chat(model_cmd::ChatArgs),
    /// Generate a single reply or continuation from the command line.
    Generate(model_cmd::GenerateArgs),
    /// List and inspect trained models.
    #[command(subcommand)]
    Models(model_cmd::ModelsCmd),
    /// Evaluate models on held-out prompts and test splits (before/after comparisons).
    Evaluate(model_cmd::EvalArgs),
    /// Run a hyperparameter search study (random / TPE, optional successive halving).
    Tune(model_cmd::TuneArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let ws = Workspace::discover()?;
    match cli.command {
        Some(Command::Data(c)) => data_cmd::run_data(&ws, c),
        Some(Command::Tokenizer(c)) => data_cmd::run_tokenizer(&ws, c),
        Some(Command::Datasets(c)) => data_cmd::run_datasets(&ws, c),
        Some(Command::Sources) => data_cmd::run_sources(&ws),
        Some(Command::Train(a)) => train_cmd::run(&ws, a),
        Some(Command::Runs) => train_cmd::list_runs(&ws),
        Some(Command::Bench(a)) => bench_cmd::run(&ws, a),
        Some(Command::Doctor(a)) => bench_cmd::doctor(&ws, a),
        Some(Command::Tui { backend, screen, model }) => {
            let s = glimmer_terminal::Screen::ALL
                .into_iter()
                .find(|s| s.title().starts_with(screen.as_str()))
                .ok_or_else(|| anyhow::anyhow!("unknown screen '{screen}'"))?;
            glimmer_terminal::run(ws, s, model, backend)
        }
        Some(Command::Chat(a)) => model_cmd::chat(&ws, a),
        Some(Command::Generate(a)) => model_cmd::generate(&ws, a),
        Some(Command::Models(c)) => model_cmd::run_models(&ws, c),
        Some(Command::Evaluate(a)) => model_cmd::evaluate(&ws, a),
        Some(Command::Tune(a)) => model_cmd::tune(&ws, a),
        None => {
            use std::io::IsTerminal;
            if std::io::stdout().is_terminal() {
                glimmer_terminal::run(ws, glimmer_terminal::Screen::Home, None, glimmer_core::BackendKind::Auto)
            } else {
                use clap::CommandFactory;
                Cli::command().print_help()?;
                Ok(())
            }
        }
    }
}
