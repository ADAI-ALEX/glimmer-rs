//! Building tokenised training datasets and training tokenizers from prepared corpora.

use crate::corpus;
use crate::download::sha256_file;
use crate::fnv1a;
use crate::manifest::{DatasetKind, DatasetManifest, SourceEntry, SplitStats};
use crate::sft::{SftExample, write_examples};
use crate::source::SourceSpec;
use crate::split::{Split, SplitSpec};
use crate::stevie;
use crate::tokens::write_tokens;
use anyhow::{Context, Result, bail};
use glimmer_core::{Rng, Workspace};
use glimmer_tokenizer::chat::{Message, render_with_mask};
use glimmer_tokenizer::{ENDOFTEXT, Tokenizer, TrainerConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceWeight {
    pub id: String,
    /// Upsampling factor: 2.0 includes every document twice, 0.5 includes about half.
    #[serde(default = "one")]
    pub weight: f64,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SftOptions {
    /// Probability that an example is prefixed with the STEVIE system prompt. Training with and
    /// without it lets the model work either way.
    pub system_prompt_probability: f64,
    /// Conversations longer than this many tokens are dropped (they would not fit the context).
    pub max_tokens: usize,
    /// Held-out prompts file used to check for train/eval leakage.
    pub eval_prompts: Option<String>,
    pub leakage_threshold: f64,
}

impl Default for SftOptions {
    fn default() -> Self {
        Self {
            system_prompt_probability: 0.5,
            max_tokens: 512,
            eval_prompts: None,
            leakage_threshold: 0.7,
        }
    }
}

/// `configs/datasets/<name>.toml`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetSpec {
    pub name: String,
    pub kind: DatasetKind,
    pub description: String,
    pub tokenizer: String,
    #[serde(default)]
    pub split: SplitSpec,
    pub sources: Vec<SourceWeight>,
    #[serde(default)]
    pub sft: SftOptions,
}

impl DatasetSpec {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let t = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        toml::from_str(&t).with_context(|| format!("parsing {}", p.display()))
    }
}

/// `configs/tokenizers/<name>.toml`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerSpec {
    pub name: String,
    pub vocab_size: usize,
    #[serde(default = "two")]
    pub min_frequency: u64,
    pub sources: Vec<TokenizerSource>,
}

fn two() -> u64 {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerSource {
    pub id: String,
    /// Use at most this many bytes of the source's training documents.
    pub max_bytes: Option<u64>,
}

impl TokenizerSpec {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let t = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        toml::from_str(&t).with_context(|| format!("parsing {}", p.display()))
    }
}

pub fn source_spec(ws: &Workspace, id: &str) -> Result<SourceSpec> {
    SourceSpec::load(ws.dataset_sources_dir().join("sources").join(format!("{id}.toml")))
}

/// Train a tokenizer on the *training split* of the listed corpora (validation and test text
/// never influences the vocabulary).
pub fn train_tokenizer(
    ws: &Workspace,
    spec: &TokenizerSpec,
    split: &SplitSpec,
    log: &mut dyn FnMut(&str),
) -> Result<PathBuf> {
    let mut texts: Vec<String> = Vec::new();
    let mut total = 0u64;
    let mut names = Vec::new();
    for src in &spec.sources {
        let source = source_spec(ws, &src.id)?;
        let mut used = 0u64;
        let docs: Vec<String> = if matches!(source.kind, crate::source::SourceKind::Conversations { .. }) {
            corpus::read_conversations(ws, &src.id)?
                .into_iter()
                .filter(|c| split.assign(&c.id) == Split::Train)
                .map(|c| stevie::conversation_text(&c))
                .collect()
        } else {
            corpus::read_documents(ws, &src.id)?
                .into_iter()
                .filter(|d| split.assign(&d.id) == Split::Train)
                .map(|d| d.text)
                .collect()
        };
        for d in docs {
            if src.max_bytes.is_some_and(|m| used >= m) {
                break;
            }
            used += d.len() as u64;
            texts.push(d);
        }
        log(&format!("  {}: {:.1} MB", src.id, used as f64 / 1e6));
        total += used;
        names.push(format!("{} ({} bytes)", src.id, used));
    }
    log(&format!("  counting pre-tokens over {:.1} MB", total as f64 / 1e6));
    let counts = glimmer_tokenizer::count_pretokens(&texts, 64);
    drop(texts);
    log(&format!("  {} unique pre-tokens; learning merges", counts.len()));
    let cfg = TrainerConfig {
        vocab_size: spec.vocab_size,
        min_frequency: spec.min_frequency,
        max_pretoken_bytes: 64,
        trained_on: names,
    };
    let started = std::time::Instant::now();
    let tok = glimmer_tokenizer::train_from_counts(counts, total, &cfg, |done, all| {
        if done % 1024 == 0 || done == all {
            eprint!("\r    merges {done}/{all}");
        }
    })?;
    eprintln!();
    let path = ws.tokenizers_dir().join(format!("{}.json", spec.name));
    tok.save(&path)?;
    log(&format!(
        "  ✓ vocab {} in {:.1}s → {}",
        tok.vocab_size(),
        started.elapsed().as_secs_f64(),
        ws.relative(&path)
    ));
    Ok(path)
}

fn repetitions(weight: f64, id: &str, seed: u64) -> usize {
    let base = weight.floor() as usize;
    let frac = weight - weight.floor();
    let mut rng = Rng::new(crate::mix64(fnv1a(id.as_bytes()) ^ seed));
    base + usize::from(frac > 0.0 && rng.next_f64() < frac)
}

fn entry(spec: &SourceSpec, weight: f64) -> SourceEntry {
    SourceEntry {
        id: spec.id.clone(),
        name: spec.name.clone(),
        license: spec.license.clone(),
        license_url: spec.license_url.clone(),
        provenance: spec.provenance.clone(),
        homepage: spec.homepage.clone(),
        weight,
        documents: BTreeMap::new(),
        tokens: BTreeMap::new(),
    }
}

pub fn dataset_dir(ws: &Workspace, name: &str) -> PathBuf {
    ws.datasets_dir().join(name)
}

/// Build a dataset described by a [`DatasetSpec`].
pub fn build_dataset(ws: &Workspace, spec: &DatasetSpec, log: &mut dyn FnMut(&str)) -> Result<DatasetManifest> {
    let tok_path = ws.resolve(&spec.tokenizer);
    let tok = Tokenizer::load(&tok_path)?;
    if tok.vocab_size() > u16::MAX as usize {
        bail!("vocabulary too large for u16 token storage");
    }
    let out = dataset_dir(ws, &spec.name);
    std::fs::create_dir_all(&out)?;
    let mut manifest = DatasetManifest {
        name: spec.name.clone(),
        kind: spec.kind,
        description: spec.description.clone(),
        tokenizer: ws.relative(&tok_path),
        tokenizer_sha256: sha256_file(&tok_path)?,
        vocab_size: tok.vocab_size(),
        split: spec.split,
        splits: BTreeMap::new(),
        sources: Vec::new(),
        preprocessing: vec![
            "Unicode NFC normalisation; quotes, spaces and newlines unified; control and zero-width characters removed".into(),
            "Quality filters: length bounds and minimum alphabetic ratio".into(),
            "Paragraphs containing blocklisted slurs dropped (datasets/filters/blocklist.rot13)".into(),
            "Exact deduplication (FNV-1a of case/whitespace-normalised text)".into(),
            "Deterministic hash-based train/val/test assignment by document id".into(),
            format!("Byte-level BPE tokenisation with {}", ws.relative(&tok_path)),
        ],
        created_unix: glimmer_core::time::now_unix(),
        glimmer_version: glimmer_core::VERSION.to_string(),
    };
    match spec.kind {
        DatasetKind::Pretrain => build_pretrain(ws, spec, &tok, &out, &mut manifest, log)?,
        DatasetKind::Sft => build_sft(ws, spec, &tok, &out, &mut manifest, log)?,
    }
    manifest.save(&out)?;
    log(&format!("  ✓ manifest → {}", ws.relative(out.join("manifest.json"))));
    Ok(manifest)
}

fn build_pretrain(
    ws: &Workspace,
    spec: &DatasetSpec,
    tok: &Tokenizer,
    out: &Path,
    manifest: &mut DatasetManifest,
    log: &mut dyn FnMut(&str),
) -> Result<()> {
    use rayon::prelude::*;
    manifest
        .preprocessing
        .push("Documents separated by <|endoftext|>; training documents shuffled; stored as u16".into());
    let mut per_split: BTreeMap<Split, Vec<Vec<u16>>> = BTreeMap::new();
    for sw in &spec.sources {
        let src = source_spec(ws, &sw.id)?;
        let docs = corpus::read_documents(ws, &sw.id)?;
        log(&format!("  tokenising {} ({} documents, weight {})", sw.id, docs.len(), sw.weight));
        let encoded: Vec<(Split, usize, Vec<u16>)> = docs
            .par_iter()
            .map(|d| {
                let split = spec.split.assign(&d.id);
                let reps = repetitions(sw.weight, &d.id, spec.split.seed);
                let ids: Vec<u16> = tok.encode(&d.text).into_iter().map(|t| t as u16).collect();
                (split, reps, ids)
            })
            .collect();
        let mut e = entry(&src, sw.weight);
        for (split, reps, ids) in encoded {
            if reps == 0 {
                continue;
            }
            *e.documents.entry(split.as_str().into()).or_insert(0) += reps;
            *e.tokens.entry(split.as_str().into()).or_insert(0) += (reps * (ids.len() + 1)) as u64;
            let bucket = per_split.entry(split).or_default();
            for _ in 1..reps {
                bucket.push(ids.clone());
            }
            bucket.push(ids);
        }
        manifest.sources.push(e);
    }
    let mut rng = Rng::new(spec.split.seed ^ 0x5EED);
    for split in Split::ALL {
        let mut docs = per_split.remove(&split).unwrap_or_default();
        if split == Split::Train {
            rng.shuffle(&mut docs);
        }
        let mut stream: Vec<u16> = Vec::with_capacity(docs.iter().map(|d| d.len() + 1).sum());
        for d in &docs {
            stream.extend_from_slice(d);
            stream.push(ENDOFTEXT as u16);
        }
        write_tokens(&out.join(format!("{}.bin", split.as_str())), &stream)?;
        log(&format!(
            "  {}: {} documents, {:.2}M tokens",
            split.as_str(),
            docs.len(),
            stream.len() as f64 / 1e6
        ));
        manifest.splits.insert(
            split.as_str().into(),
            SplitStats {
                documents: docs.len(),
                tokens: stream.len() as u64,
                ..Default::default()
            },
        );
    }
    Ok(())
}

fn build_sft(
    ws: &Workspace,
    spec: &DatasetSpec,
    tok: &Tokenizer,
    out: &Path,
    manifest: &mut DatasetManifest,
    log: &mut dyn FnMut(&str),
) -> Result<()> {
    let opts = &spec.sft;
    let prompts = match &opts.eval_prompts {
        Some(p) => stevie::load_eval_prompts(&ws.resolve(p))?,
        None => Vec::new(),
    };
    manifest.preprocessing.push(format!(
        "Chat template <|system|>/<|user|>/<|assistant|>/<|end|>; loss on assistant tokens and their <|end|> only; system prompt on {:.0}% of examples",
        opts.system_prompt_probability * 100.0
    ));
    manifest
        .preprocessing
        .push("Near-duplicate conversations removed with MinHash (word 3-gram shingles)".into());
    let mut per_split: BTreeMap<Split, (Vec<SftExample>, SplitStats)> = BTreeMap::new();
    let mut dropped_long = 0usize;
    let mut leaked = 0usize;
    for sw in &spec.sources {
        let src = source_spec(ws, &sw.id)?;
        let convs = corpus::read_conversations(ws, &sw.id)?;
        let leaks = stevie::leakage(&convs, &prompts, opts.leakage_threshold);
        let leaked_ids: std::collections::HashSet<String> = leaks.iter().map(|l| l.0.clone()).collect();
        for (c, p, s) in &leaks {
            log(&format!("  ! conversation {c} overlaps eval prompt {p} ({s:.2}) — excluded"));
        }
        leaked += leaked_ids.len();
        let mut e = entry(&src, sw.weight);
        for c in convs.iter().filter(|c| !leaked_ids.contains(&c.id)) {
            let split = spec.split.assign(&c.id);
            let mut rng = Rng::new(fnv1a(c.id.as_bytes()) ^ spec.split.seed);
            let mut messages = Vec::with_capacity(c.messages.len() + 1);
            if rng.next_f64() < opts.system_prompt_probability {
                messages.push(Message::system(stevie::SYSTEM_PROMPT));
            }
            messages.extend(c.messages.iter().cloned());
            let (tokens, mask) = render_with_mask(tok, &messages);
            if tokens.len() > opts.max_tokens {
                dropped_long += 1;
                continue;
            }
            let reps = repetitions(sw.weight, &c.id, spec.split.seed);
            let ex = SftExample { tokens, mask };
            let (list, stats) = per_split.entry(split).or_default();
            for _ in 0..reps {
                stats.documents += 1;
                stats.messages += c.messages.len();
                stats.tokens += ex.tokens.len() as u64;
                stats.supervised_tokens += ex.supervised() as u64;
                list.push(ex.clone());
            }
            *e.documents.entry(split.as_str().into()).or_insert(0) += reps;
            *e.tokens.entry(split.as_str().into()).or_insert(0) += (reps * ex.tokens.len()) as u64;
        }
        manifest.sources.push(e);
    }
    if dropped_long > 0 {
        log(&format!("  ! {dropped_long} conversations exceeded {} tokens and were dropped", opts.max_tokens));
    }
    if leaked > 0 {
        manifest
            .preprocessing
            .push(format!("{leaked} conversations excluded for overlapping held-out eval prompts"));
    }
    for split in Split::ALL {
        let (examples, stats) = per_split.remove(&split).unwrap_or_default();
        write_examples(&out.join(format!("{}.sft", split.as_str())), &examples)?;
        log(&format!(
            "  {}: {} conversations, {} messages, {} tokens ({} supervised)",
            split.as_str(),
            stats.documents,
            stats.messages,
            stats.tokens,
            stats.supervised_tokens
        ));
        manifest.splits.insert(split.as_str().into(), stats);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_weights() {
        assert_eq!(repetitions(2.0, "a", 1), 2);
        assert_eq!(repetitions(1.0, "a", 1), 1);
        let n: usize = (0..10_000).map(|i| repetitions(0.3, &i.to_string(), 1)).sum();
        assert!((2_600..3_400).contains(&n), "{n}");
        // Deterministic per id.
        assert_eq!(repetitions(0.5, "x", 9), repetitions(0.5, "x", 9));
    }
}
