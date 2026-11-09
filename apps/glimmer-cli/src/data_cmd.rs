//! `glimmer data …`, `glimmer tokenizer …`, `glimmer datasets …`, `glimmer sources …`

use crate::style::{self, bold, dim, heading, kv};
use anyhow::{Result, bail};
use clap::Subcommand;
use glimmer_core::Workspace;
use glimmer_core::time::{format_bytes, format_count, iso8601};
use glimmer_data::build::{self, DatasetSpec, TokenizerSpec};
use glimmer_data::corpus;
use glimmer_data::source::{SourceKind, SourceSpec};
use glimmer_data::split::SplitSpec;
use glimmer_data::{DatasetKind, DatasetManifest};
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
pub enum DataCmd {
    /// Download approved sources into data/raw/ (licences in datasets/sources/).
    Fetch {
        /// Source ids (default: all remote sources).
        #[arg(long = "source", short)]
        sources: Vec<String>,
        /// Re-download even if files exist.
        #[arg(long)]
        force: bool,
    },
    /// Clean, filter and deduplicate fetched sources into data/corpora/.
    Prepare {
        #[arg(long = "source", short)]
        sources: Vec<String>,
    },
    /// Tokenise prepared corpora into a training dataset (configs/datasets/<name>.toml).
    Build {
        /// Dataset config path or name.
        config: String,
    },
    /// Run the complete pipeline: fetch → prepare → tokenizer → build every dataset config.
    Pipeline {
        #[arg(long, default_value = "configs/tokenizers/glimmer-8k.toml")]
        tokenizer: String,
        /// Skip the download step (use already-fetched files).
        #[arg(long)]
        offline: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum TokenizerCmd {
    /// Train a byte-level BPE tokenizer from prepared corpora.
    Train {
        #[arg(default_value = "configs/tokenizers/glimmer-8k.toml")]
        config: String,
    },
    /// Show how a tokenizer splits some text.
    Encode {
        text: String,
        #[arg(long, default_value = "data/tokenizers/glimmer-8k.json")]
        tokenizer: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum DatasetsCmd {
    /// List prepared datasets.
    List,
    /// Show a dataset's manifest: sources, licences, splits and token counts.
    Inspect { name: String },
}

fn log_line(s: &str) {
    println!("{s}");
}

fn selected_sources(ws: &Workspace, ids: &[String], remote_only: bool) -> Result<Vec<SourceSpec>> {
    let all = SourceSpec::load_dir(ws.dataset_sources_dir().join("sources"))?;
    if ids.is_empty() {
        return Ok(all.into_iter().filter(|s| !remote_only || s.is_remote()).collect());
    }
    ids.iter()
        .map(|id| {
            all.iter()
                .find(|s| &s.id == id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("unknown source '{id}' (see `glimmer sources`)"))
        })
        .collect()
}

fn resolve_config(ws: &Workspace, value: &str, dir: &str) -> PathBuf {
    let p = ws.resolve(value);
    if p.is_file() { p } else { ws.configs_dir().join(dir).join(format!("{value}.toml")) }
}

pub fn run_data(ws: &Workspace, cmd: DataCmd) -> Result<()> {
    match cmd {
        DataCmd::Fetch { sources, force } => {
            for s in selected_sources(ws, &sources, true)? {
                heading(&format!("fetch {} — {}", s.id, s.name));
                kv("licence", &s.license);
                corpus::fetch_source(ws, &s, force, &mut |l| log_line(l))?;
            }
        }
        DataCmd::Prepare { sources } => {
            for s in selected_sources(ws, &sources, false)? {
                heading(&format!("prepare {}", s.id));
                corpus::prepare_source(ws, &s, &mut |l| log_line(l))?;
            }
        }
        DataCmd::Build { config } => {
            let path = resolve_config(ws, &config, "datasets");
            let spec = DatasetSpec::load(&path)?;
            heading(&format!("build dataset {}", spec.name));
            build::build_dataset(ws, &spec, &mut |l| log_line(l))?;
        }
        DataCmd::Pipeline { tokenizer, offline } => {
            let sources = selected_sources(ws, &[], false)?;
            if !offline {
                for s in sources.iter().filter(|s| s.is_remote()) {
                    heading(&format!("fetch {}", s.id));
                    corpus::fetch_source(ws, s, false, &mut |l| log_line(l))?;
                }
            }
            for s in &sources {
                heading(&format!("prepare {}", s.id));
                corpus::prepare_source(ws, s, &mut |l| log_line(l))?;
            }
            run_tokenizer(ws, TokenizerCmd::Train { config: tokenizer })?;
            let mut configs: Vec<PathBuf> = std::fs::read_dir(ws.configs_dir().join("datasets"))?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "toml"))
                .collect();
            configs.sort();
            for c in configs {
                let spec = DatasetSpec::load(&c)?;
                heading(&format!("build dataset {}", spec.name));
                build::build_dataset(ws, &spec, &mut |l| log_line(l))?;
            }
        }
    }
    Ok(())
}

pub fn run_tokenizer(ws: &Workspace, cmd: TokenizerCmd) -> Result<()> {
    match cmd {
        TokenizerCmd::Train { config } => {
            let path = resolve_config(ws, &config, "tokenizers");
            let spec = TokenizerSpec::load(&path)?;
            heading(&format!("train tokenizer {} (vocab {})", spec.name, spec.vocab_size));
            // Use the same split assignment as the datasets so held-out text never shapes the vocabulary.
            let split = first_split_spec(ws).unwrap_or_default();
            build::train_tokenizer(ws, &spec, &split, &mut |l| log_line(l))?;
        }
        TokenizerCmd::Encode { text, tokenizer } => {
            let tok = glimmer_tokenizer::Tokenizer::load(ws.resolve(&tokenizer))?;
            let ids = tok.encode(&text);
            println!(
                "{} tokens ({:.2} bytes/token)",
                bold(&ids.len().to_string()),
                text.len() as f64 / ids.len().max(1) as f64
            );
            let pieces: Vec<String> = ids
                .iter()
                .map(|&i| format!("{}{}", style::accent(&format!("{:?}", tok.token_str(i))), dim(&format!("·{i}"))))
                .collect();
            println!("{}", pieces.join(" "));
        }
    }
    Ok(())
}

fn first_split_spec(ws: &Workspace) -> Option<SplitSpec> {
    let dir = ws.configs_dir().join("datasets");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).collect();
    paths.sort();
    paths.iter().filter_map(|p| DatasetSpec::load(p).ok()).find(|s| s.kind == DatasetKind::Pretrain).map(|s| s.split)
}

pub fn run_datasets(ws: &Workspace, cmd: DatasetsCmd) -> Result<()> {
    match cmd {
        DatasetsCmd::List => {
            heading("datasets");
            let list = DatasetManifest::list(ws.datasets_dir());
            if list.is_empty() {
                println!("  {}", dim("none prepared yet — run `glimmer data pipeline`"));
            }
            for (name, m) in list {
                match m {
                    Ok(m) => println!(
                        "  {:<20} {:<9} {:>9} tokens  {}",
                        bold(&name),
                        format!("{:?}", m.kind).to_lowercase(),
                        format_count(m.total_tokens()),
                        dim(&m.description)
                    ),
                    Err(e) => println!("  {:<20} {}", name, style::err(&e.to_string())),
                }
            }
        }
        DatasetsCmd::Inspect { name } => {
            let dir = build::dataset_dir(ws, &name);
            if !dir.join("manifest.json").is_file() {
                bail!("dataset '{name}' not found in {}", ws.relative(ws.datasets_dir()));
            }
            let m = DatasetManifest::load(&dir)?;
            heading(&format!("dataset {}", m.name));
            kv("kind", format!("{:?}", m.kind).to_lowercase());
            kv("description", &m.description);
            kv("tokenizer", format!("{} (vocab {})", m.tokenizer, m.vocab_size));
            kv("tokenizer sha256", &m.tokenizer_sha256[..16]);
            kv("created", iso8601(m.created_unix));
            kv("split seed", m.split.seed);
            println!("\n  {}", bold("splits"));
            for (split, s) in &m.splits {
                let extra = if m.kind == DatasetKind::Sft {
                    format!("  {} messages, {} supervised tokens", s.messages, format_count(s.supervised_tokens))
                } else {
                    String::new()
                };
                println!("    {:<6} {:>8} docs  {:>9} tokens{extra}", split, s.documents, format_count(s.tokens));
            }
            println!("\n  {}", bold("sources"));
            for s in &m.sources {
                println!("    {} {}", style::accent("•"), bold(&s.name));
                println!("      {:<12} {}", dim("licence"), s.license);
                println!("      {:<12} {}", dim("weight"), s.weight);
                println!("      {:<12} {}", dim("provenance"), s.provenance);
                println!("      {:<12} {}", dim("homepage"), s.homepage);
            }
            println!("\n  {}", bold("preprocessing"));
            for (i, p) in m.preprocessing.iter().enumerate() {
                println!("    {}. {p}", i + 1);
            }
        }
    }
    Ok(())
}

pub fn run_sources(ws: &Workspace) -> Result<()> {
    heading("dataset sources");
    for s in SourceSpec::load_dir(ws.dataset_sources_dir().join("sources"))? {
        let fetched = corpus::raw_dir(ws, &s.id).join("provenance.json").is_file() || !s.is_remote();
        let stats = corpus::read_stats(ws, &s.id);
        let status = match (&stats, fetched) {
            (Some(st), _) => style::ok(&format!("prepared · {} docs", st.documents)),
            (None, true) => style::warn("fetched"),
            (None, false) => dim("not fetched"),
        };
        println!("  {} {:<22} {}", style::accent("•"), bold(&s.id), status);
        println!("      {:<12} {}", dim("name"), s.name);
        println!("      {:<12} {} ({})", dim("licence"), s.license, s.kind_label());
        if let SourceKind::Gutenberg { books } = &s.kind {
            println!("      {:<12} {} books", dim("books"), books.len());
        }
        if let Some(st) = stats {
            println!(
                "      {:<12} {} chars, {} exact + {} near duplicates removed",
                dim("corpus"),
                format_bytes(st.characters),
                st.exact_duplicates,
                st.near_duplicates
            );
        }
    }
    Ok(())
}
