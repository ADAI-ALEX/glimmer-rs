//! Declarative dataset sources.
//!
//! Each source lives in `datasets/sources/<id>.toml` and records *where* data comes from, *under
//! what licence*, and *how* it is cleaned. Nothing is downloaded that is not described here.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceKind {
    /// A plain-text file over HTTP(S), optionally truncated to `max_bytes` with a range request.
    HttpText {
        url: String,
        #[serde(default)]
        max_bytes: Option<u64>,
        /// Literal line separating documents (e.g. `<|endoftext|>`). Paragraph-split if absent.
        #[serde(default)]
        document_separator: Option<String>,
    },
    /// A Parquet file over HTTP(S) with one document per row.
    HttpParquet {
        url: String,
        text_column: String,
        #[serde(default)]
        title_column: Option<String>,
        #[serde(default)]
        max_rows: Option<usize>,
    },
    /// Project Gutenberg plain-text e-books by id.
    Gutenberg { books: Vec<GutenbergBook> },
    /// Local conversation files in the STEVIE text format.
    Conversations { path: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GutenbergBook {
    pub id: u32,
    pub title: String,
    pub author: String,
    /// Author's year of death — recorded to document public-domain status in Australia
    /// (authors who died before 1955) as well as the United States.
    pub died: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CleaningSpec {
    pub min_chars: usize,
    pub max_chars: usize,
    /// Minimum fraction of alphabetic characters among non-whitespace characters.
    pub min_alpha_ratio: f64,
    /// Remove near-duplicates with MinHash (Jaccard ≥ `near_dedup_threshold`).
    pub near_dedup: bool,
    pub near_dedup_threshold: f64,
    /// Drop paragraphs containing terms from the offensive-language blocklist.
    pub blocklist: bool,
    /// Split long documents into chunks of roughly this many characters (0 = never).
    pub chunk_chars: usize,
}

impl Default for CleaningSpec {
    fn default() -> Self {
        Self {
            min_chars: 64,
            max_chars: 200_000,
            min_alpha_ratio: 0.6,
            near_dedup: false,
            near_dedup_threshold: 0.8,
            blocklist: true,
            chunk_chars: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceSpec {
    pub id: String,
    pub name: String,
    #[serde(flatten)]
    pub kind: SourceKind,
    /// SPDX identifier or licence name.
    pub license: String,
    pub license_url: String,
    pub homepage: String,
    /// Who created the data and how.
    pub provenance: String,
    /// Whether and how the data may be redistributed, and what GLIMMER does.
    pub redistribution: String,
    #[serde(default)]
    pub citation: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub cleaning: CleaningSpec,
}

impl SourceSpec {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("reading source {}", path.display()))?;
        let spec: Self = toml::from_str(&text).with_context(|| format!("parsing source {}", path.display()))?;
        if spec.id.is_empty() || spec.license.is_empty() {
            bail!("source {} must declare an id and a licence", path.display());
        }
        Ok(spec)
    }

    /// Load every `*.toml` in a directory, sorted by id.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Vec<Self>> {
        let dir = dir.as_ref();
        let mut out = Vec::new();
        if !dir.is_dir() {
            return Ok(out);
        }
        for entry in std::fs::read_dir(dir)? {
            let p = entry?.path();
            if p.extension().is_some_and(|e| e == "toml") {
                out.push(Self::load(&p)?);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    pub fn kind_label(&self) -> &'static str {
        match self.kind {
            SourceKind::HttpText { .. } => "http-text",
            SourceKind::HttpParquet { .. } => "http-parquet",
            SourceKind::Gutenberg { .. } => "gutenberg",
            SourceKind::Conversations { .. } => "conversations",
        }
    }

    pub fn is_remote(&self) -> bool {
        !matches!(self.kind, SourceKind::Conversations { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_http_text_source() {
        let s: SourceSpec = toml::from_str(
            r#"
            id = "x"
            name = "X"
            kind = "http_text"
            url = "https://example.com/a.txt"
            max_bytes = 1000
            document_separator = "<|endoftext|>"
            license = "CC0-1.0"
            license_url = "u"
            homepage = "h"
            provenance = "p"
            redistribution = "r"
            [cleaning]
            min_chars = 10
            "#,
        )
        .unwrap();
        assert_eq!(s.cleaning.min_chars, 10);
        assert!(s.cleaning.blocklist);
        assert!(matches!(s.kind, SourceKind::HttpText { max_bytes: Some(1000), .. }));
    }
}
