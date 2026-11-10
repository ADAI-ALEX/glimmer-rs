//! # glimmer-evaluation
//!
//! Measures more than whether the loss went down:
//!
//! * **Loss / perplexity** on held-out test splits (conversation and general text).
//! * **Generation quality proxies** on a fixed set of held-out prompts: length, end-of-turn
//!   termination, repetition (distinct-n, repeated trigrams), topical overlap, multi-turn
//!   continuity probes and generation speed.
//! * **Australian English style**: rate of responses containing Australian vocabulary or
//!   spelling, and American spellings.
//!
//! Every generated response is stored verbatim in the report, so results can be inspected
//! rather than taken on trust. Reports for several checkpoints (random initialisation,
//! after pretraining, after fine-tuning) can be compared side by side.

pub mod metrics;

use anyhow::{Context, Result};
use glimmer_core::{ModelConfig, Workspace};
use glimmer_data::batch::{PretrainBatcher, SftBatcher};
use glimmer_data::sft::read_examples;
use glimmer_data::stevie::EvalPrompt;
use glimmer_inference::engine::generate_once;
use glimmer_inference::{SamplingConfig, StopReason};
use glimmer_models::{ComputeDevice, Transformer, store};
use glimmer_tokenizer::Tokenizer;
use glimmer_tokenizer::chat::{Message, Role, render_prompt};
use metrics::ResponseMetrics;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What to evaluate.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// A registered model or checkpoint directory (`model.bpk`, `model.json`, `tokenizer.json`).
    Dir(PathBuf),
    /// An untrained model with the given architecture — the "before training" baseline.
    RandomInit { config: ModelConfig, tokenizer: PathBuf, seed: u64 },
}

#[derive(Debug, Clone)]
pub struct EvalOptions {
    pub label: String,
    pub prompts: Vec<EvalPrompt>,
    pub samples_per_prompt: usize,
    pub sampling: SamplingConfig,
    pub system_prompt: Option<String>,
    /// SFT dataset directory whose `test.sft` is scored.
    pub sft_dataset: Option<PathBuf>,
    /// Pretraining dataset directory whose `test.bin` is scored.
    pub pretrain_dataset: Option<PathBuf>,
    pub pretrain_batches: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sample {
    pub seed: u64,
    pub text: String,
    pub stop: StopReason,
    pub metrics: ResponseMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptResult {
    pub id: String,
    pub category: String,
    pub conversation: Vec<Message>,
    pub samples: Vec<Sample>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Aggregate {
    pub responses: usize,
    pub mean_words: f64,
    pub end_of_turn_rate: f64,
    pub mean_distinct_2: f64,
    pub mean_repeated_trigrams: f64,
    pub australian_marker_rate: f64,
    pub mean_australian_markers: f64,
    pub us_spelling_rate: f64,
    pub topical_rate: f64,
    pub continuity_rate: Option<f64>,
    pub mean_tokens_per_sec: f64,
}

impl Aggregate {
    fn from(samples: &[&Sample]) -> Self {
        let n = samples.len().max(1) as f64;
        let mean = |f: &dyn Fn(&Sample) -> f64| samples.iter().map(|s| f(s)).sum::<f64>() / n;
        let cont: Vec<bool> = samples.iter().filter_map(|s| s.metrics.continuity).collect();
        Self {
            responses: samples.len(),
            mean_words: mean(&|s| s.metrics.words as f64),
            end_of_turn_rate: mean(&|s| s.metrics.ended_turn as u8 as f64),
            mean_distinct_2: mean(&|s| s.metrics.distinct_2),
            mean_repeated_trigrams: mean(&|s| s.metrics.repeated_trigrams),
            australian_marker_rate: mean(&|s| s.metrics.has_australian_marker as u8 as f64),
            mean_australian_markers: mean(&|s| s.metrics.australian_markers as f64),
            us_spelling_rate: mean(&|s| (s.metrics.us_spellings > 0) as u8 as f64),
            topical_rate: mean(&|s| s.metrics.topical as u8 as f64),
            continuity_rate: (!cont.is_empty()).then(|| cont.iter().filter(|&&c| c).count() as f64 / cont.len() as f64),
            mean_tokens_per_sec: mean(&|s| s.metrics.tokens_per_sec),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalReport {
    pub label: String,
    pub source: String,
    pub parameters: usize,
    pub backend: String,
    pub sft_test_loss: Option<f64>,
    pub pretrain_test_loss: Option<f64>,
    pub aggregate: Aggregate,
    pub per_category: BTreeMap<String, Aggregate>,
    pub prompts: Vec<PromptResult>,
    pub created_unix: u64,
}

impl EvalReport {
    pub fn sft_test_ppl(&self) -> Option<f64> {
        self.sft_test_loss.map(f64::exp)
    }
    pub fn pretrain_test_ppl(&self) -> Option<f64> {
        self.pretrain_test_loss.map(f64::exp)
    }
}

fn load(source: &ModelSource, device: &ComputeDevice) -> Result<(Transformer, Tokenizer, String)> {
    match source {
        ModelSource::Dir(dir) => {
            let cfg: ModelConfig = serde_json::from_str(
                &std::fs::read_to_string(dir.join("model.json"))
                    .with_context(|| format!("{} has no model.json", dir.display()))?,
            )?;
            let tok = Tokenizer::load(dir.join("tokenizer.json"))?;
            let model = store::load_weights(&cfg, &device.device, &dir.join("model.bpk"))?;
            Ok((model, tok, dir.display().to_string()))
        }
        ModelSource::RandomInit { config, tokenizer, seed } => {
            let tok = Tokenizer::load(tokenizer)?;
            let cfg = ModelConfig { vocab_size: tok.vocab_size(), ..config.clone() };
            Ok((Transformer::init(&cfg, &device.device, *seed), tok, format!("random init (seed {seed})")))
        }
    }
}

/// Score `test.sft` of an SFT dataset (mean loss per supervised token).
pub fn sft_test_loss(model: &Transformer, dataset: &Path, batch: usize, device: &ComputeDevice) -> Result<f64> {
    let examples = read_examples(&dataset.join("test.sft"))?;
    let batcher = SftBatcher::new(examples, model.context_length(), &mut glimmer_core::Rng::new(0))?;
    Ok(glimmer_training::evaluate(model, &batcher.eval_batches(batch), &device.device))
}

/// The lexical style metrics measured on the corpus's own reference replies (STEVIE's turns in the
/// held-out test split): the yardstick for what fine-tuning aims at. The corpus uses Australian
/// vocabulary sparingly by design, so a model's marker rate should be read against this.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferenceStyle {
    pub replies: usize,
    pub australian_marker_rate: f64,
    pub mean_australian_markers: f64,
    pub us_spelling_rate: f64,
    pub mean_words: f64,
}

/// What the fine-tuning corpus looks like: reference style from the test split, and the word
/// n-grams of every training reply, to detect responses copied from training data.
#[derive(Debug, Clone)]
pub struct CorpusReference {
    pub style: ReferenceStyle,
    pub train_replies: usize,
    train_ngrams: std::collections::HashSet<String>,
}

/// Length of the word span that counts as copied from the training replies.
pub const COPY_SPAN_WORDS: usize = 8;

fn ngrams(text: &str, n: usize) -> Vec<String> {
    let w = metrics::words(text);
    if w.len() < n { Vec::new() } else { w.windows(n).map(|g| g.join(" ")).collect() }
}

/// Decode every supervised reply (assistant turn, without its `<|end|>`) in `<split>.sft`.
fn corpus_replies(ws: &Workspace, dataset: &Path, split: &str) -> Result<Vec<String>> {
    let manifest = glimmer_data::manifest::DatasetManifest::load(dataset)?;
    let tok = Tokenizer::load(ws.root.join(&manifest.tokenizer))?;
    let mut replies = Vec::new();
    for ex in read_examples(&dataset.join(format!("{split}.sft")))? {
        let mut cur: Vec<u32> = Vec::new();
        for (&t, &m) in ex.tokens.iter().zip(&ex.mask) {
            if m == 1 && t != glimmer_tokenizer::END {
                cur.push(t);
            } else if !cur.is_empty() {
                replies.push(tok.decode(&cur));
                cur.clear();
            }
        }
        if !cur.is_empty() {
            replies.push(tok.decode(&cur));
        }
    }
    Ok(replies)
}

/// Build the corpus reference for an SFT dataset directory.
pub fn corpus_reference(ws: &Workspace, dataset: &Path) -> Result<CorpusReference> {
    let test = corpus_replies(ws, dataset, "test")?;
    anyhow::ensure!(!test.is_empty(), "no reference replies in {}", dataset.display());
    let n = test.len() as f64;
    let styles: Vec<_> = test.iter().map(|r| glimmer_data::lexicon::analyse(r)).collect();
    let style = ReferenceStyle {
        replies: test.len(),
        australian_marker_rate: styles.iter().filter(|s| s.has_australian_marker()).count() as f64 / n,
        mean_australian_markers: styles.iter().map(|s| (s.slang_hits + s.au_spellings) as f64).sum::<f64>() / n,
        us_spelling_rate: styles.iter().filter(|s| s.us_spellings > 0).count() as f64 / n,
        mean_words: styles.iter().map(|s| s.words as f64).sum::<f64>() / n,
    };
    let train = corpus_replies(ws, dataset, "train")?;
    let train_ngrams = train.iter().flat_map(|r| ngrams(r, COPY_SPAN_WORDS)).collect();
    Ok(CorpusReference { style, train_replies: train.len(), train_ngrams })
}

impl CorpusReference {
    /// Whether `text` contains a span of `COPY_SPAN_WORDS` words that appears in a training reply.
    pub fn copies_training(&self, text: &str) -> bool {
        ngrams(text, COPY_SPAN_WORDS).iter().any(|g| self.train_ngrams.contains(g))
    }

    /// Fraction of a report's responses that copy a training span.
    pub fn copied_rate(&self, report: &EvalReport) -> f64 {
        let texts: Vec<&str> = report.prompts.iter().flat_map(|p| p.samples.iter().map(|s| s.text.as_str())).collect();
        if texts.is_empty() {
            return 0.0;
        }
        texts.iter().filter(|t| self.copies_training(t)).count() as f64 / texts.len() as f64
    }
}

/// Score `test.bin` of a pretraining dataset over fixed windows.
pub fn pretrain_test_loss(model: &Transformer, dataset: &Path, batches: usize, device: &ComputeDevice) -> Result<f64> {
    let b = PretrainBatcher::open(&dataset.join("test.bin"), model.context_length())?;
    Ok(glimmer_training::evaluate(model, &b.eval_batches(8, batches), &device.device))
}

/// Evaluate one model.
pub fn evaluate_model(
    source: &ModelSource,
    opts: &EvalOptions,
    device: &ComputeDevice,
    progress: &mut dyn FnMut(&str),
) -> Result<EvalReport> {
    let (model, tok, source_label) = load(source, device)?;
    use glimmer_models::burn::module::Module;
    let parameters = model.num_params();
    let sft_test_loss = match &opts.sft_dataset {
        Some(d) => Some(sft_test_loss(&model, d, 8, device)?),
        None => None,
    };
    let pretrain_test_loss = match &opts.pretrain_dataset {
        Some(d) => Some(pretrain_test_loss(&model, d, opts.pretrain_batches, device)?),
        None => None,
    };
    let mut prompts = Vec::new();
    for (i, p) in opts.prompts.iter().enumerate() {
        progress(&format!("[{}/{}] {}", i + 1, opts.prompts.len(), p.id));
        let mut msgs = Vec::new();
        if let Some(s) = &opts.system_prompt {
            msgs.push(Message::system(s.clone()));
        }
        msgs.extend(p.messages.iter().cloned());
        let ids = render_prompt(&tok, &msgs);
        let last_user =
            p.messages.iter().rev().find(|m| m.role == Role::User).map(|m| m.content.clone()).unwrap_or_default();
        let mut samples = Vec::new();
        for k in 0..opts.samples_per_prompt {
            let seed = opts.sampling.seed.wrapping_add(1000 * k as u64 + i as u64 + 1);
            let cfg = SamplingConfig { seed, ..opts.sampling.clone() };
            let t0 = std::time::Instant::now();
            let (tokens, text, stop) = generate_once(&model, &tok, &device.device, &ids, &cfg)?;
            let secs = t0.elapsed().as_secs_f64();
            let text = text.trim().to_string();
            let m = metrics::measure(
                &last_user,
                &text,
                tokens.len(),
                stop == StopReason::EndOfTurn,
                &p.expect_any,
                tokens.len() as f64 / secs.max(1e-9),
            );
            samples.push(Sample { seed, text, stop, metrics: m });
        }
        prompts.push(PromptResult {
            id: p.id.clone(),
            category: p.category.clone(),
            conversation: p.messages.clone(),
            samples,
        });
    }
    let all: Vec<&Sample> = prompts.iter().flat_map(|p| p.samples.iter()).collect();
    let mut per_category = BTreeMap::new();
    let mut cats: Vec<&str> = prompts.iter().map(|p| p.category.as_str()).collect();
    cats.sort();
    cats.dedup();
    for c in cats {
        let s: Vec<&Sample> = prompts.iter().filter(|p| p.category == c).flat_map(|p| p.samples.iter()).collect();
        per_category.insert(c.to_string(), Aggregate::from(&s));
    }
    Ok(EvalReport {
        label: opts.label.clone(),
        source: source_label,
        parameters,
        backend: device.label(),
        sft_test_loss,
        pretrain_test_loss,
        aggregate: Aggregate::from(&all),
        per_category,
        prompts,
        created_unix: glimmer_core::time::now_unix(),
    })
}

fn pct(v: f64) -> String {
    format!("{:.0}%", v * 100.0)
}

fn opt(v: Option<f64>, digits: usize) -> String {
    v.map(|x| format!("{x:.digits$}")).unwrap_or_else(|| "—".into())
}

/// Side-by-side Markdown comparison of several reports (e.g. before and after training).
pub fn comparison_markdown(reports: &[EvalReport], reference: Option<&CorpusReference>, ws: &Workspace) -> String {
    let mut out = String::from("# STEVIE evaluation\n\n");
    out.push_str(&format!(
        "Generated by `glimmer evaluate` on {}. Every response below is genuine model output (sampled with fixed seeds); nothing is edited or cherry-picked.\n\n",
        glimmer_core::time::iso8601(glimmer_core::time::now_unix())
    ));
    out.push_str("## Summary\n\n| metric | ");
    out.push_str(&reports.iter().map(|r| r.label.clone()).collect::<Vec<_>>().join(" | "));
    out.push_str(" |\n|---|");
    out.push_str(&"---|".repeat(reports.len()));
    out.push('\n');
    let row = |name: &str, f: &dyn Fn(&EvalReport) -> String| {
        format!("| {name} | {} |\n", reports.iter().map(f).collect::<Vec<_>>().join(" | "))
    };
    out.push_str(&row("conversation test loss (ppl)", &|r| {
        format!("{} ({})", opt(r.sft_test_loss, 3), opt(r.sft_test_ppl(), 1))
    }));
    out.push_str(&row("general-text test loss (ppl)", &|r| {
        format!("{} ({})", opt(r.pretrain_test_loss, 3), opt(r.pretrain_test_ppl(), 1))
    }));
    out.push_str(&row("ends its turn (`<|end|>`)", &|r| pct(r.aggregate.end_of_turn_rate)));
    out.push_str(&row("responses with Australian markers", &|r| pct(r.aggregate.australian_marker_rate)));
    out.push_str(&row("Australian markers per response", &|r| format!("{:.2}", r.aggregate.mean_australian_markers)));
    out.push_str(&row("responses with US spellings", &|r| pct(r.aggregate.us_spelling_rate)));
    out.push_str(&row("topical overlap with prompt", &|r| pct(r.aggregate.topical_rate)));
    out.push_str(&row("multi-turn continuity probe", &|r| {
        r.aggregate.continuity_rate.map(pct).unwrap_or_else(|| "—".into())
    }));
    out.push_str(&row("distinct bigrams (higher = less repetitive)", &|r| {
        format!("{:.2}", r.aggregate.mean_distinct_2)
    }));
    out.push_str(&row("repeated trigram rate", &|r| format!("{:.2}", r.aggregate.mean_repeated_trigrams)));
    if let Some(c) = reference {
        out.push_str(&row(
            &format!("responses repeating {COPY_SPAN_WORDS}+ consecutive words of a training reply"),
            &|r| pct(c.copied_rate(r)),
        ));
    }
    out.push_str(&row("mean response length (words)", &|r| format!("{:.1}", r.aggregate.mean_words)));
    out.push_str(&row("generation speed (tokens/s)", &|r| format!("{:.0}", r.aggregate.mean_tokens_per_sec)));
    if let Some(c) = reference {
        let r = &c.style;
        out.push_str(&format!(
            "\n**Reference.** The same lexical measures on the corpus's own {} held-out reference replies (STEVIE's turns in the test split, never trained on): Australian markers in {} of replies ({:.2} per reply), US spellings in {}, {:.1} words on average. The corpus uses Australian vocabulary sparingly by design, so this is the target, not 100%. Copying is checked against all {} training replies.\n",
            r.replies,
            pct(r.australian_marker_rate),
            r.mean_australian_markers,
            pct(r.us_spelling_rate),
            r.mean_words,
            c.train_replies
        ));
    }
    out.push_str("\n**How to read this.** Losses are mean negative log-likelihood per supervised token on held-out splits. Style, topicality and continuity are lexical proxies (see `crates/glimmer-evaluation/src/metrics.rs`): they show whether STEVIE *sounds* Australian, stays near the topic and reuses context words — not whether it is factually right. Small models make confident mistakes.\n\n");
    out.push_str("## Responses to held-out prompts\n\n");
    let n = reports.iter().map(|r| r.prompts.len()).min().unwrap_or(0);
    for i in 0..n {
        let p = &reports[0].prompts[i];
        out.push_str(&format!("### `{}` · {}\n\n", p.id, p.category));
        for m in &p.conversation {
            let who = match m.role {
                Role::User => "**User**",
                Role::Assistant => "**STEVIE** *(given context)*",
                Role::System => "**System**",
            };
            out.push_str(&format!("> {who}: {}\n>\n", m.content));
        }
        out.push('\n');
        for r in reports {
            if let Some(s) = r.prompts[i].samples.first() {
                let text = if s.text.is_empty() { "*(empty)*".to_string() } else { s.text.replace('\n', " ") };
                out.push_str(&format!(
                    "* **{}**: {}{}\n",
                    r.label,
                    text,
                    match s.stop {
                        StopReason::EndOfTurn => "",
                        StopReason::SpecialToken => " *[stopped on a special token, not `<|end|>`]*",
                        _ => " *[no end-of-turn]*",
                    }
                ));
            }
        }
        out.push('\n');
    }
    let _ = ws;
    out
}

/// Generates genuine samples from fixed prompts at each evaluation during training, so progress
/// is visible as real model output in the CLI and the terminal UI.
pub fn training_sampler(stage: glimmer_core::Stage) -> glimmer_training::Sampler {
    let cfg = SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.15,
        max_new_tokens: 48,
        seed: 7,
        repetition_window: 64,
    };
    Box::new(move |model, tok, device| {
        let prompts: Vec<(String, Vec<u32>)> = match stage {
            glimmer_core::Stage::Pretrain => ["Once upon a time", "The kangaroo is", "In Australia,"]
                .iter()
                .map(|p| (p.to_string(), tok.encode(p)))
                .collect(),
            glimmer_core::Stage::Sft => ["G'day, who are you?", "Tell me about the bush."]
                .iter()
                .map(|p| (p.to_string(), render_prompt(tok, &[Message::user(*p)])))
                .collect(),
        };
        prompts
            .into_iter()
            .filter_map(|(label, ids)| {
                generate_once(model, tok, device, &ids, &cfg).ok().map(|(_, text, _)| (label, text))
            })
            .collect()
    })
}
