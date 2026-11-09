//! BPE training with incremental pair-count updates.
//!
//! 1. Count pre-token frequencies over the corpus (parallel).
//! 2. Represent each unique pre-token as a sequence of byte ids weighted by its frequency.
//! 3. Repeatedly merge the most frequent adjacent pair. Only words containing the merged pair
//!    are revisited; pair counts are updated by diffing each affected word before and after the
//!    merge, and a lazily-invalidated max-heap yields the next best pair.
//!
//! Ties are broken by the smallest `(left, right)` id pair, making training deterministic.

use crate::{BYTE_OFFSET, FIRST_MERGE, Tokenizer, TokenizerMetadata, pretokenize::pretokenize};
use anyhow::{Result, bail};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Target vocabulary size including the 8 special tokens and 256 bytes.
    pub vocab_size: usize,
    /// Stop early when the best pair occurs fewer times than this.
    pub min_frequency: u64,
    /// Pre-tokens longer than this many bytes are ignored during training (they are still
    /// encodable; this only keeps pathological runs from dominating memory).
    pub max_pretoken_bytes: usize,
    pub trained_on: Vec<String>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            min_frequency: 2,
            max_pretoken_bytes: 64,
            trained_on: Vec::new(),
        }
    }
}

/// Count pre-tokens across documents in parallel.
pub fn count_pretokens<S: AsRef<str> + Sync>(docs: &[S], max_bytes: usize) -> HashMap<String, u64> {
    docs.par_iter()
        .fold(HashMap::new, |mut acc: HashMap<String, u64>, doc| {
            for chunk in pretokenize(doc.as_ref()) {
                if chunk.len() <= max_bytes {
                    *acc.entry(chunk.to_string()).or_insert(0) += 1;
                }
            }
            acc
        })
        .reduce(HashMap::new, |mut a, b| {
            if a.len() < b.len() {
                return merge_counts(b, a);
            }
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        })
}

fn merge_counts(mut a: HashMap<String, u64>, b: HashMap<String, u64>) -> HashMap<String, u64> {
    for (k, v) in b {
        *a.entry(k).or_insert(0) += v;
    }
    a
}

/// Train on an iterator of texts (sequential counting; convenient for small inputs and tests).
pub fn train<'a>(texts: impl Iterator<Item = &'a str>, cfg: &TrainerConfig) -> Result<Tokenizer> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    let mut total_bytes = 0u64;
    for t in texts {
        total_bytes += t.len() as u64;
        for chunk in pretokenize(t) {
            if chunk.len() <= cfg.max_pretoken_bytes {
                *counts.entry(chunk.to_string()).or_insert(0) += 1;
            }
        }
    }
    train_from_counts(counts, total_bytes, cfg, |_, _| {})
}

fn pair_counts_of(word: &[u32]) -> impl Iterator<Item = (u32, u32)> + '_ {
    word.windows(2).map(|w| (w[0], w[1]))
}

/// Train from pre-computed pre-token counts. `progress(done, total)` is called periodically.
pub fn train_from_counts(
    counts: HashMap<String, u64>,
    total_bytes: u64,
    cfg: &TrainerConfig,
    mut progress: impl FnMut(usize, usize),
) -> Result<Tokenizer> {
    let base = FIRST_MERGE as usize;
    if cfg.vocab_size < base {
        bail!("vocab_size must be at least {base} (8 special tokens + 256 bytes)");
    }
    let target_merges = cfg.vocab_size - base;
    let unique = counts.len() as u64;

    // Deterministic word order (HashMap iteration order is randomised).
    let mut entries: Vec<(String, u64)> = counts.into_iter().collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut words: Vec<Vec<u32>> = Vec::with_capacity(entries.len());
    let mut freqs: Vec<i64> = Vec::with_capacity(entries.len());
    for (w, c) in entries {
        words.push(w.bytes().map(|b| BYTE_OFFSET + b as u32).collect());
        freqs.push(c as i64);
    }

    let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::new();
    let mut where_: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (idx, w) in words.iter().enumerate() {
        for p in pair_counts_of(w) {
            *pair_counts.entry(p).or_insert(0) += freqs[idx];
            where_.entry(p).or_default().push(idx as u32);
        }
    }
    let mut heap: BinaryHeap<(i64, Reverse<(u32, u32)>)> = pair_counts
        .iter()
        .map(|(&p, &c)| (c, Reverse(p)))
        .collect();

    let mut merges: Vec<(u32, u32)> = Vec::with_capacity(target_merges);
    let mut local_before: HashMap<(u32, u32), i64> = HashMap::new();
    let mut local_after: HashMap<(u32, u32), i64> = HashMap::new();

    while merges.len() < target_merges {
        let Some((count, Reverse(pair))) = heap.pop() else { break };
        let current = pair_counts.get(&pair).copied().unwrap_or(0);
        if current != count {
            // Stale heap entry: re-queue with the true count if still alive.
            if current > 0 {
                heap.push((current, Reverse(pair)));
            }
            continue;
        }
        if (count as u64) < cfg.min_frequency {
            break;
        }
        let new_id = FIRST_MERGE + merges.len() as u32;
        merges.push(pair);

        let mut idxs = where_.remove(&pair).unwrap_or_default();
        idxs.sort_unstable();
        idxs.dedup();
        for idx in idxs {
            let idx_u = idx as usize;
            let w = &words[idx_u];
            if !w.windows(2).any(|x| x[0] == pair.0 && x[1] == pair.1) {
                continue; // stale index entry
            }
            let f = freqs[idx_u];
            local_before.clear();
            for p in pair_counts_of(w) {
                *local_before.entry(p).or_insert(0) += 1;
            }
            let mut merged = Vec::with_capacity(w.len());
            let mut i = 0;
            while i < w.len() {
                if i + 1 < w.len() && w[i] == pair.0 && w[i + 1] == pair.1 {
                    merged.push(new_id);
                    i += 2;
                } else {
                    merged.push(w[i]);
                    i += 1;
                }
            }
            local_after.clear();
            for p in pair_counts_of(&merged) {
                *local_after.entry(p).or_insert(0) += 1;
            }
            for (p, n) in local_before.iter() {
                let after = local_after.get(p).copied().unwrap_or(0);
                if after != *n {
                    let e = pair_counts.entry(*p).or_insert(0);
                    *e += (after - n) * f;
                    if *e > 0 {
                        heap.push((*e, Reverse(*p)));
                    }
                }
            }
            for (p, n) in local_after.iter() {
                if !local_before.contains_key(p) {
                    let e = pair_counts.entry(*p).or_insert(0);
                    *e += n * f;
                    heap.push((*e, Reverse(*p)));
                    where_.entry(*p).or_default().push(idx);
                }
            }
            words[idx_u] = merged;
        }
        pair_counts.remove(&pair);
        if merges.len() % 256 == 0 {
            progress(merges.len(), target_merges);
        }
    }
    progress(merges.len(), target_merges);

    let metadata = TokenizerMetadata {
        trained_on: cfg.trained_on.clone(),
        training_bytes: total_bytes,
        unique_pretokens: unique,
        created_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    Tokenizer::from_merges(merges, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_merge_is_most_frequent_pair() {
        // "ab" occurs far more than anything else.
        let text = "ab ab ab ab ab ab cd";
        let t = train(
            [text].into_iter(),
            &TrainerConfig {
                vocab_size: FIRST_MERGE as usize + 1,
                min_frequency: 1,
                ..Default::default()
            },
        )
        .unwrap();
        // " a"+"b" (with leading space) or "a"+"b": both chunks "ab" and " ab" contain (a, b).
        let a = BYTE_OFFSET + b'a' as u32;
        let b = BYTE_OFFSET + b'b' as u32;
        assert_eq!(t.merges[0], (a, b));
    }

    #[test]
    fn deterministic_training() {
        let text = "the cat sat on the mat with the hat ".repeat(50);
        let cfg = TrainerConfig {
            vocab_size: FIRST_MERGE as usize + 40,
            ..Default::default()
        };
        let a = train([text.as_str()].into_iter(), &cfg).unwrap();
        let b = train([text.as_str()].into_iter(), &cfg).unwrap();
        assert_eq!(a.merges, b.merges);
    }

    #[test]
    fn incremental_counts_match_bruteforce() {
        // Compare the incremental trainer with a naive recount-every-step implementation.
        let text = "low lower lowest newer wider new news ".repeat(7);
        let n = 25;
        let cfg = TrainerConfig {
            vocab_size: FIRST_MERGE as usize + n,
            min_frequency: 1,
            ..Default::default()
        };
        let fast = train([text.as_str()].into_iter(), &cfg).unwrap();

        let mut counts: HashMap<String, i64> = HashMap::new();
        for c in pretokenize(&text) {
            *counts.entry(c.to_string()).or_insert(0) += 1;
        }
        let mut words: Vec<(Vec<u32>, i64)> = counts
            .into_iter()
            .map(|(w, c)| (w.bytes().map(|b| BYTE_OFFSET + b as u32).collect(), c))
            .collect();
        let mut slow = Vec::new();
        for k in 0..n {
            let mut pc: HashMap<(u32, u32), i64> = HashMap::new();
            for (w, c) in &words {
                for p in pair_counts_of(w) {
                    *pc.entry(p).or_insert(0) += c;
                }
            }
            let Some((&best, _)) = pc
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            else {
                break;
            };
            let id = FIRST_MERGE + k as u32;
            slow.push(best);
            for (w, _) in words.iter_mut() {
                let mut m = Vec::new();
                let mut i = 0;
                while i < w.len() {
                    if i + 1 < w.len() && (w[i], w[i + 1]) == best {
                        m.push(id);
                        i += 2;
                    } else {
                        m.push(w[i]);
                        i += 1;
                    }
                }
                *w = m;
            }
        }
        assert_eq!(fast.merges, slow);
    }

    #[test]
    fn parallel_counts_match_sequential() {
        let docs: Vec<String> = (0..50).map(|i| format!("doc {i} has words and more words")).collect();
        let par = count_pretokens(&docs, 64);
        let mut seq: HashMap<String, u64> = HashMap::new();
        for d in &docs {
            for c in pretokenize(d) {
                *seq.entry(c.to_string()).or_insert(0) += 1;
            }
        }
        assert_eq!(par, seq);
    }
}
