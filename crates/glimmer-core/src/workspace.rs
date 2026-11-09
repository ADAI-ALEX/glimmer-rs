//! Project workspace discovery.
//!
//! A GLIMMER workspace is a directory containing a `glimmer.toml` marker. All relative paths in
//! configuration files (datasets, tokenizers, checkpoints) resolve against the workspace root,
//! so commands behave the same from any subdirectory.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MARKER: &str = "glimmer.toml";

/// Optional settings stored in `glimmer.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct WorkspaceSettings {
    /// Default model to open in chat.
    pub default_model: Option<String>,
    /// Override the data directory (downloads, tokenized shards). Relative to the root.
    pub data_dir: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Workspace {
    pub root: PathBuf,
    pub settings: WorkspaceSettings,
}

impl Workspace {
    /// Find the workspace by walking up from the current directory. Honours `GLIMMER_HOME`.
    /// Falls back to the current directory when no marker is found.
    pub fn discover() -> Result<Self> {
        if let Ok(home) = std::env::var("GLIMMER_HOME") {
            return Self::open(PathBuf::from(home));
        }
        let cwd = std::env::current_dir()?;
        let mut dir: Option<&Path> = Some(&cwd);
        while let Some(d) = dir {
            if d.join(MARKER).is_file() {
                return Self::open(d.to_path_buf());
            }
            dir = d.parent();
        }
        Self::open(cwd)
    }

    pub fn open(root: PathBuf) -> Result<Self> {
        let marker = root.join(MARKER);
        let settings = if marker.is_file() {
            toml::from_str(&std::fs::read_to_string(&marker)?)?
        } else {
            WorkspaceSettings::default()
        };
        Ok(Self { root, settings })
    }

    /// Resolve a possibly-relative path against the workspace root.
    pub fn resolve(&self, p: impl AsRef<Path>) -> PathBuf {
        let p = p.as_ref();
        if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) }
    }

    /// Express a path relative to the root when possible (for display and portable manifests).
    pub fn relative(&self, p: impl AsRef<Path>) -> String {
        let p = p.as_ref();
        p.strip_prefix(&self.root).unwrap_or(p).to_string_lossy().into_owned()
    }

    pub fn data_dir(&self) -> PathBuf {
        self.resolve(self.settings.data_dir.as_deref().unwrap_or("data"))
    }
    pub fn raw_dir(&self) -> PathBuf {
        self.data_dir().join("raw")
    }
    pub fn corpora_dir(&self) -> PathBuf {
        self.data_dir().join("corpora")
    }
    pub fn datasets_dir(&self) -> PathBuf {
        self.data_dir().join("datasets")
    }
    pub fn tokenizers_dir(&self) -> PathBuf {
        self.data_dir().join("tokenizers")
    }
    pub fn runs_dir(&self) -> PathBuf {
        self.root.join("runs")
    }
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }
    pub fn experiments_dir(&self) -> PathBuf {
        self.root.join("experiments")
    }
    pub fn configs_dir(&self) -> PathBuf {
        self.root.join("configs")
    }
    /// Version-controlled dataset sources (manifests, the STEVIE corpus, eval prompts).
    pub fn dataset_sources_dir(&self) -> PathBuf {
        self.root.join("datasets")
    }
    pub fn conversations_dir(&self) -> PathBuf {
        self.root.join("conversations")
    }
}
