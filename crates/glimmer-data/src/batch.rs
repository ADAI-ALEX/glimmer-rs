//! Training batches.
//!
//! Batches are plain host-side vectors; the trainer moves them onto the compute device. Keeping
//! batching independent of the tensor library makes it easy to test exhaustively.

use crate::sft::SftExample;
use crate::tokens::TokenFile;
use anyhow::{Result, bail};
use glimmer_core::Rng;
use glimmer_tokenizer::PAD;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub batch_size: usize,
    pub seq_len: usize,
    /// `[batch_size * seq_len]` input ids.
    pub inputs: Vec<u32>,
    /// `[batch_size * seq_len]` next-token targets.
    pub targets: Vec<u32>,
    /// `[batch_size * seq_len]` loss weights (1 = supervised, 0 = ignored).
    pub weights: Vec<f32>,
    /// Packed batches: segment id per token (0 = padding). Tokens attend only within a segment.
    pub segments: Option<Vec<u32>>,
    /// Packed batches: position of each token within its segment (restarts at 0).
    pub positions: Option<Vec<u32>>,
}

impl Batch {
    pub fn supervised_tokens(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0.0).count()
    }
    pub fn tokens(&self) -> usize {
        self.batch_size * self.seq_len
    }
}

/// Random contiguous windows from a token stream (standard causal-LM pretraining).
pub struct PretrainBatcher {
    file: TokenFile,
    pub seq_len: usize,
}

impl PretrainBatcher {
    pub fn open(path: &Path, seq_len: usize) -> Result<Self> {
        let file = TokenFile::open(path)?;
        if file.len() < seq_len + 2 {
            bail!("{} has {} tokens, fewer than one window of {}", path.display(), file.len(), seq_len + 1);
        }
        Ok(Self { file, seq_len })
    }

    pub fn num_tokens(&self) -> usize {
        self.file.len()
    }

    fn window(&self, start: usize, b: &mut Batch) {
        let mut buf = Vec::with_capacity(self.seq_len + 1);
        self.file.read_into(start, self.seq_len + 1, &mut buf);
        b.inputs.extend_from_slice(&buf[..self.seq_len]);
        b.targets.extend_from_slice(&buf[1..]);
    }

    fn empty(&self, batch_size: usize) -> Batch {
        Batch {
            batch_size,
            seq_len: self.seq_len,
            inputs: Vec::with_capacity(batch_size * self.seq_len),
            targets: Vec::with_capacity(batch_size * self.seq_len),
            weights: vec![1.0; batch_size * self.seq_len],
            segments: None,
            positions: None,
        }
    }

    /// Uniformly random windows.
    pub fn sample(&self, batch_size: usize, rng: &mut Rng) -> Batch {
        let mut b = self.empty(batch_size);
        let max_start = self.file.len() - self.seq_len - 1;
        for _ in 0..batch_size {
            self.window(rng.below(max_start + 1), &mut b);
        }
        b
    }

    /// Deterministic, evenly spaced windows for validation — identical across runs so validation
    /// losses are comparable between experiments.
    pub fn eval_batches(&self, batch_size: usize, num_batches: usize) -> Vec<Batch> {
        let windows = batch_size * num_batches;
        let max_start = self.file.len() - self.seq_len - 1;
        let stride = (max_start / windows.max(1)).max(1);
        (0..num_batches)
            .map(|bi| {
                let mut b = self.empty(batch_size);
                for k in 0..batch_size {
                    let start = ((bi * batch_size + k) * stride).min(max_start);
                    self.window(start, &mut b);
                }
                b
            })
            .collect()
    }
}

/// Serializable position of an [`SftBatcher`] for exact resumption.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SftCursor {
    pub order: Vec<u32>,
    pub pos: usize,
    pub epoch: usize,
}

/// Packs whole conversations into fixed-length rows.
pub struct SftBatcher {
    examples: Vec<SftExample>,
    pub seq_len: usize,
    cursor: SftCursor,
}

const LOOKAHEAD: usize = 16;

impl SftBatcher {
    pub fn new(examples: Vec<SftExample>, seq_len: usize, rng: &mut Rng) -> Result<Self> {
        if examples.is_empty() {
            bail!("SFT dataset has no examples");
        }
        let mut order: Vec<u32> = (0..examples.len() as u32).collect();
        rng.shuffle(&mut order);
        Ok(Self { examples, seq_len, cursor: SftCursor { order, pos: 0, epoch: 0 } })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    pub fn epoch(&self) -> usize {
        self.cursor.epoch
    }

    /// Fractional epochs completed.
    pub fn progress(&self) -> f64 {
        self.cursor.epoch as f64 + self.cursor.pos as f64 / self.examples.len() as f64
    }

    pub fn cursor(&self) -> &SftCursor {
        &self.cursor
    }

    pub fn restore(&mut self, cursor: SftCursor) -> Result<()> {
        if cursor.order.len() != self.examples.len() {
            bail!("SFT cursor does not match the dataset size");
        }
        self.cursor = cursor;
        Ok(())
    }

    /// Supervised (input, target, weight) triples of one example, truncated to `seq_len`.
    fn pairs(&self, idx: usize) -> (&[u32], &[u32], &[u8]) {
        let ex = &self.examples[idx];
        let n = (ex.tokens.len().saturating_sub(1)).min(self.seq_len);
        (&ex.tokens[..n], &ex.tokens[1..=n], &ex.mask[1..=n])
    }

    fn pack_row(b: &mut Batch, rows: &[(&[u32], &[u32], &[u8])], seq_len: usize) {
        let segs = b.segments.as_mut().unwrap();
        let pos = b.positions.as_mut().unwrap();
        let mut used = 0;
        for (seg, (inp, tgt, msk)) in rows.iter().enumerate() {
            b.inputs.extend_from_slice(inp);
            b.targets.extend_from_slice(tgt);
            b.weights.extend(msk.iter().map(|&m| m as f32));
            segs.extend(std::iter::repeat_n(seg as u32 + 1, inp.len()));
            pos.extend(0..inp.len() as u32);
            used += inp.len();
        }
        let pad = seq_len - used;
        b.inputs.extend(std::iter::repeat_n(PAD, pad));
        b.targets.extend(std::iter::repeat_n(PAD, pad));
        b.weights.extend(std::iter::repeat_n(0.0, pad));
        segs.extend(std::iter::repeat_n(0, pad));
        pos.extend(0..pad as u32);
    }

    fn empty(&self, batch_size: usize) -> Batch {
        let n = batch_size * self.seq_len;
        Batch {
            batch_size,
            seq_len: self.seq_len,
            inputs: Vec::with_capacity(n),
            targets: Vec::with_capacity(n),
            weights: Vec::with_capacity(n),
            segments: Some(Vec::with_capacity(n)),
            positions: Some(Vec::with_capacity(n)),
        }
    }

    /// Next training batch. Each row is filled greedily with whole conversations; a short
    /// look-ahead finds conversations that fit the remaining space. Epoch boundaries reshuffle.
    pub fn next_batch(&mut self, batch_size: usize, rng: &mut Rng) -> Batch {
        let mut b = self.empty(batch_size);
        for _ in 0..batch_size {
            let mut row: Vec<usize> = Vec::new();
            let mut used = 0;
            loop {
                if self.cursor.pos >= self.cursor.order.len() {
                    self.cursor.epoch += 1;
                    self.cursor.pos = 0;
                    rng.shuffle(&mut self.cursor.order);
                }
                let remaining = self.seq_len - used;
                let end = (self.cursor.pos + LOOKAHEAD).min(self.cursor.order.len());
                let found = (self.cursor.pos..end).find(|&k| {
                    let idx = self.cursor.order[k] as usize;
                    self.pairs(idx).0.len() <= remaining
                });
                match found {
                    Some(k) => {
                        self.cursor.order.swap(self.cursor.pos, k);
                        let idx = self.cursor.order[self.cursor.pos] as usize;
                        used += self.pairs(idx).0.len();
                        row.push(idx);
                        self.cursor.pos += 1;
                        if used == self.seq_len {
                            break;
                        }
                    }
                    None if row.is_empty() => {
                        // Nothing fits an empty row only if seq_len is tiny; take it truncated.
                        let idx = self.cursor.order[self.cursor.pos] as usize;
                        row.push(idx);
                        self.cursor.pos += 1;
                        break;
                    }
                    None => break,
                }
            }
            let parts: Vec<_> = row.iter().map(|&i| self.pairs(i)).collect();
            Self::pack_row(&mut b, &parts, self.seq_len);
        }
        b
    }

    /// Deterministic packed batches covering every example exactly once (in dataset order).
    pub fn eval_batches(&self, batch_size: usize) -> Vec<Batch> {
        let mut batches = Vec::new();
        let mut b = self.empty(batch_size);
        let mut rows_in_batch = 0;
        let mut row: Vec<usize> = Vec::new();
        let mut used = 0;
        let flush_row = |b: &mut Batch, row: &mut Vec<usize>, this: &Self| {
            let parts: Vec<_> = row.iter().map(|&i| this.pairs(i)).collect();
            Self::pack_row(b, &parts, this.seq_len);
            row.clear();
        };
        for idx in 0..self.examples.len() {
            let len = self.pairs(idx).0.len();
            if used + len > self.seq_len && !row.is_empty() {
                flush_row(&mut b, &mut row, self);
                used = 0;
                rows_in_batch += 1;
                if rows_in_batch == batch_size {
                    batches.push(std::mem::replace(&mut b, self.empty(batch_size)));
                    rows_in_batch = 0;
                }
            }
            row.push(idx);
            used += len;
        }
        if !row.is_empty() {
            flush_row(&mut b, &mut row, self);
            rows_in_batch += 1;
        }
        if rows_in_batch > 0 {
            // Fill the final batch with empty (fully padded) rows.
            while rows_in_batch < batch_size {
                Self::pack_row(&mut b, &[], self.seq_len);
                rows_in_batch += 1;
            }
            b.batch_size = batch_size;
            batches.push(b);
        }
        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::write_tokens;

    #[test]
    fn pretrain_windows_are_shifted_by_one() {
        let dir = std::env::temp_dir().join(format!("glimmer-batch-{}", std::process::id()));
        let p = dir.join("t.bin");
        let toks: Vec<u16> = (0..500).collect();
        write_tokens(&p, &toks).unwrap();
        let b = PretrainBatcher::open(&p, 16).unwrap();
        let mut rng = Rng::new(1);
        let batch = b.sample(4, &mut rng);
        assert_eq!(batch.inputs.len(), 64);
        for i in 0..64 {
            assert_eq!(batch.targets[i], batch.inputs[i] + 1);
        }
        let evals = b.eval_batches(2, 3);
        assert_eq!(evals.len(), 3);
        assert_eq!(evals, b.eval_batches(2, 3), "eval batches must be deterministic");
        std::fs::remove_dir_all(dir).ok();
    }

    fn ex(len: usize, base: u32) -> SftExample {
        SftExample {
            tokens: (0..len as u32).map(|i| base + i).collect(),
            mask: (0..len).map(|i| u8::from(i % 2 == 1)).collect(),
        }
    }

    #[test]
    fn sft_packing_preserves_alignment_and_segments() {
        let examples = vec![ex(5, 100), ex(7, 200), ex(4, 300), ex(9, 400)];
        let mut rng = Rng::new(3);
        let mut sb = SftBatcher::new(examples.clone(), 12, &mut rng).unwrap();
        let b = sb.next_batch(3, &mut rng);
        let segs = b.segments.as_ref().unwrap();
        let pos = b.positions.as_ref().unwrap();
        assert_eq!(b.inputs.len(), 36);
        for i in 0..36 {
            if segs[i] == 0 {
                assert_eq!(b.weights[i], 0.0, "padding must not be supervised");
                assert_eq!(b.inputs[i], PAD);
            } else {
                // Within a segment, the target is the next token of the same example.
                assert_eq!(b.targets[i], b.inputs[i] + 1);
                // Positions restart at the start of each segment.
                if i % 12 == 0 || segs[i - 1] != segs[i] {
                    assert_eq!(pos[i], 0);
                } else {
                    assert_eq!(pos[i], pos[i - 1] + 1);
                }
            }
        }
    }

    #[test]
    fn sft_epochs_cover_every_example() {
        let examples: Vec<SftExample> = (0..10).map(|i| ex(3 + i % 4, 1000 * (i as u32 + 1))).collect();
        let mut rng = Rng::new(5);
        let mut sb = SftBatcher::new(examples, 8, &mut rng).unwrap();
        let mut seen = std::collections::HashSet::new();
        while sb.epoch() == 0 {
            let b = sb.next_batch(1, &mut rng);
            let segs = b.segments.unwrap();
            for (i, s) in segs.iter().enumerate() {
                if *s > 0 && b.positions.as_ref().unwrap()[i] == 0 {
                    seen.insert(b.inputs[i] / 1000);
                }
            }
        }
        assert_eq!(seen.len(), 10);
    }

    #[test]
    fn sft_cursor_restores_exactly() {
        let examples: Vec<SftExample> = (0..20).map(|i| ex(4, 10 * i)).collect();
        let mut rng = Rng::new(9);
        let mut a = SftBatcher::new(examples.clone(), 10, &mut rng).unwrap();
        a.next_batch(2, &mut rng);
        let saved_cursor = a.cursor().clone();
        let saved_rng = rng.clone();
        let next_a = a.next_batch(2, &mut rng);
        let mut rng2 = saved_rng;
        let mut b = SftBatcher::new(examples, 10, &mut Rng::new(0)).unwrap();
        b.restore(saved_cursor).unwrap();
        assert_eq!(b.next_batch(2, &mut rng2), next_a);
    }

    #[test]
    fn sft_eval_covers_all_examples_once() {
        let examples: Vec<SftExample> = (0..13).map(|i| ex(3 + i % 5, 100 * (i as u32 + 1))).collect();
        let sb = SftBatcher::new(examples, 10, &mut Rng::new(1)).unwrap();
        let batches = sb.eval_batches(2);
        let starts: usize = batches
            .iter()
            .map(|b| {
                let segs = b.segments.as_ref().unwrap();
                let pos = b.positions.as_ref().unwrap();
                (0..segs.len()).filter(|&i| segs[i] > 0 && pos[i] == 0).count()
            })
            .sum();
        assert_eq!(starts, 13);
    }
}
