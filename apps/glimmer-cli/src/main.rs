//! `glimmer` — command-line entry point for the GLIMMER laboratory.

mod bench_cmd;
mod data_cmd;
mod eval_cmd;
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
        None => {
            use clap::CommandFactory;
            Cli::command().print_help()?;
            Ok(())
        }
    }
}
