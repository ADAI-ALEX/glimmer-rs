//! Deduplication.
//!
//! * **Exact**: FNV-1a of a canonical form (lower-cased, whitespace-collapsed).
//! * **Near-duplicate**: MinHash signatures over word shingles with locality-sensitive hashing
//!   (banding) to find candidate pairs, verified by estimated Jaccard similarity. This catches
//!   templated variations that exact hashing misses.

use crate::fnv1a;
use std::collections::{HashMap, HashSet};

/// Canonical form used for exact duplicate detection.
pub fn canonical(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            space = true;
        } else {
            if space && !out.is_empty() {
                out.push(' ');
            }
            space = false;
            for l in c.to_lowercase() {
                out.push(l);
            }
        }
    }
    out
}

pub fn exact_hash(text: &str) -> u64 {
    fnv1a(canonical(text).as_bytes())
}

/// Indices of documents to keep after exact deduplication (first occurrence wins).
pub fn exact_dedup<S: AsRef<str>>(docs: &[S]) -> Vec<usize> {
    let mut seen = HashSet::with_capacity(docs.len());
    (0..docs.len())
        .filter(|&i| seen.insert(exact_hash(docs[i].as_ref())))
        .collect()
}

/// MinHash parameters.
#[derive(Debug, Clone, Copy)]
pub struct MinHashConfig {
    pub num_hashes: usize,
    pub bands: usize,
    pub shingle: usize,
}

impl Default for MinHashConfig {
    fn default() -> Self {
        // 32 bands × 4 rows: candidate probability ≈ 1-(1-J^4)^32, ~0.96 at J=0.6.
        Self {
            num_hashes: 128,
            bands: 32,
            shingle: 3,
        }
    }
}

fn mix(mut x: u64) -> u64 {
    // SplitMix64 finaliser as a fast 64-bit permutation.
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// MinHash signature of a text's word shingles.
pub fn signature(text: &str, cfg: &MinHashConfig) -> Vec<u64> {
    let canon = canonical(text);
    let words: Vec<&str> = canon
        .split(' ')
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();
    let mut shingles: Vec<u64> = Vec::new();
    if words.len() < cfg.shingle {
        shingles.push(fnv1a(words.join(" ").as_bytes()));
    } else {
        for w in words.windows(cfg.shingle) {
            shingles.push(fnv1a(w.join(" ").as_bytes()));
        }
    }
    let mut sig = vec![u64::MAX; cfg.num_hashes];
    for s in shingles {
        for (i, slot) in sig.iter_mut().enumerate() {
            let h = mix(s ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            if h < *slot {
                *slot = h;
            }
        }
    }
    sig
}

pub fn estimated_jaccard(a: &[u64], b: &[u64]) -> f64 {
    let same = a.iter().zip(b).filter(|(x, y)| x == y).count();
    same as f64 / a.len().max(1) as f64
}

/// Near-duplicate clustering result.
#[derive(Debug, Clone)]
pub struct NearDedup {
    /// Indices to keep (first member of every cluster).
    pub keep: Vec<usize>,
    /// (dropped index, kept index it duplicates, similarity).
    pub dropped: Vec<(usize, usize, f64)>,
}

/// Remove near-duplicates with Jaccard ≥ `threshold`. Documents earlier in the slice win.
pub fn near_dedup<S: AsRef<str> + Sync>(docs: &[S], threshold: f64, cfg: &MinHashConfig) -> NearDedup {
    use rayon::prelude::*;
    let sigs: Vec<Vec<u64>> = docs.par_iter().map(|d| signature(d.as_ref(), cfg)).collect();
    let rows = cfg.num_hashes / cfg.bands;
    let mut buckets: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    let mut keep = Vec::new();
    let mut dropped = Vec::new();
    for (i, sig) in sigs.iter().enumerate() {
        let band_keys: Vec<(usize, u64)> = (0..cfg.bands)
            .map(|b| {
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                for v in &sig[b * rows..(b + 1) * rows] {
                    h = mix(h ^ v);
                }
                (b, h)
            })
            .collect();
        let mut best: Option<(usize, f64)> = None;
        let mut checked = HashSet::new();
        for key in &band_keys {
            if let Some(cands) = buckets.get(key) {
                for &j in cands {
                    if checked.insert(j) {
                        let sim = estimated_jaccard(sig, &sigs[j]);
                        if sim >= threshold && best.is_none_or(|(_, s)| sim > s) {
                            best = Some((j, sim));
                        }
                    }
                }
            }
        }
        match best {
            Some((j, sim)) => dropped.push((i, j, sim)),
            None => {
                keep.push(i);
                for key in band_keys {
                    buckets.entry(key).or_default().push(i);
                }
            }
        }
    }
    NearDedup { keep, dropped }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ignores_case_and_whitespace() {
        let docs = ["Hello  World", "hello world", "Hello there"];
        assert_eq!(exact_dedup(&docs), vec![0, 2]);
    }

    #[test]
    fn minhash_similarity_tracks_overlap() {
        let cfg = MinHashConfig::default();
        let a = signature("the quick brown fox jumps over the lazy dog near the river bank today", &cfg);
        let b = signature("the quick brown fox jumps over the lazy dog near the river bank tonight", &cfg);
        let c = signature("completely different sentence about koalas eating eucalyptus leaves slowly", &cfg);
        assert!(estimated_jaccard(&a, &b) > 0.6);
        assert!(estimated_jaccard(&a, &c) < 0.1);
    }

    #[test]
    fn near_dedup_drops_variants_keeps_distinct() {
        let docs = vec![
            "G'day mate, how's your arvo going? Hope the footy was good on the weekend.",
            "G'day mate, how's your arvo going? Hope the footy was great on the weekend.",
            "Koalas sleep up to twenty hours a day because eucalyptus is low in energy.",
        ];
        let r = near_dedup(&docs, 0.5, &MinHashConfig::default());
        assert_eq!(r.keep, vec![0, 2]);
        assert_eq!(r.dropped.len(), 1);
        assert_eq!(r.dropped[0].1, 0);
    }
}
