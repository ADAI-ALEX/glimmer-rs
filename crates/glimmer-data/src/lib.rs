//! # glimmer-data
//!
//! Reproducible dataset pipeline:
//!
//! ```text
//! datasets/sources/*.toml ──fetch──▶ data/raw/<source>/          (original bytes + provenance)
//!                         ──prepare─▶ data/corpora/<source>.jsonl (cleaned, filtered, deduplicated)
//! configs/datasets/*.toml ──build───▶ data/datasets/<name>/       (tokenised splits + manifest)
//! ```
//!
//! * [`source`] — declarative source definitions with licence and provenance metadata.
//! * [`download`] — HTTP fetching with byte limits, checksums and provenance records.
//! * [`clean`] — Unicode normalisation, Gutenberg boilerplate removal, quality filters.
//! * [`dedup`] — exact (FNV-1a) and near-duplicate (MinHash + LSH) removal.
//! * [`corpus`] — turning raw downloads into cleaned document collections.
//! * [`build`] — tokenising, splitting and writing training datasets.
//! * [`batch`] — random-window pretraining batches and packed SFT batches.
//! * [`stevie`] — the STEVIE conversation corpus format and quality checks.
//! * [`lexicon`] — Australian English vocabulary and spelling lists.

pub mod batch;
pub mod build;
pub mod clean;
pub mod corpus;
pub mod dedup;
pub mod download;
pub mod lexicon;
pub mod manifest;
pub mod sft;
pub mod source;
pub mod split;
pub mod stevie;
pub mod tokens;

pub use batch::{Batch, PretrainBatcher, SftBatcher};
pub use manifest::{DatasetKind, DatasetManifest};
pub use source::SourceSpec;

/// SplitMix64 finaliser: turns a weakly mixed hash into well-distributed bits. FNV-1a alone has
/// poorly mixed high bits for short, similar inputs (e.g. `doc-1`, `doc-2`), which biases any
/// hash-to-uniform mapping.
pub fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Uniform `[0, 1)` value derived deterministically from bytes.
pub fn hash_unit(bytes: &[u8]) -> f64 {
    (mix64(fnv1a(bytes)) >> 11) as f64 / (1u64 << 53) as f64
}

/// Stable 64-bit FNV-1a hash (identical across platforms and Rust versions).
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
