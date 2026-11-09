//! Dataset manifests: everything needed to reproduce and audit a prepared dataset.

use crate::split::SplitSpec;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DatasetKind {
    Pretrain,
    Sft,
}

/// Licence and provenance of one contributing source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceEntry {
    pub id: String,
    pub name: String,
    pub license: String,
    pub license_url: String,
    pub provenance: String,
    pub homepage: String,
    pub weight: f64,
    /// Documents (or conversations) contributed per split, after weighting.
    pub documents: BTreeMap<String, usize>,
    /// Tokens contributed per split, after weighting.
    pub tokens: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SplitStats {
    pub documents: usize,
    pub tokens: u64,
    /// SFT only: number of messages and supervised (assistant) tokens.
    #[serde(default)]
    pub messages: usize,
    #[serde(default)]
    pub supervised_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetManifest {
    pub name: String,
    pub kind: DatasetKind,
    pub description: String,
    /// Tokenizer path relative to the workspace root, and its SHA-256.
    pub tokenizer: String,
    pub tokenizer_sha256: String,
    pub vocab_size: usize,
    pub split: SplitSpec,
    pub splits: BTreeMap<String, SplitStats>,
    pub sources: Vec<SourceEntry>,
    /// Ordered list of processing steps applied (human-readable).
    pub preprocessing: Vec<String>,
    pub created_unix: u64,
    pub glimmer_version: String,
}

impl DatasetManifest {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let p = dir.as_ref().join("manifest.json");
        let text = std::fs::read_to_string(&p).with_context(|| format!("reading dataset manifest {}", p.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        std::fs::create_dir_all(dir.as_ref())?;
        std::fs::write(dir.as_ref().join("manifest.json"), serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn total_tokens(&self) -> u64 {
        self.splits.values().map(|s| s.tokens).sum()
    }

    /// List the prepared datasets under `data/datasets/`.
    pub fn list(datasets_dir: impl AsRef<Path>) -> Vec<(String, Result<DatasetManifest>)> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(datasets_dir.as_ref()) {
            for e in rd.flatten() {
                if e.path().join("manifest.json").is_file() {
                    let name = e.file_name().to_string_lossy().into_owned();
                    out.push((name, DatasetManifest::load(e.path())));
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}
