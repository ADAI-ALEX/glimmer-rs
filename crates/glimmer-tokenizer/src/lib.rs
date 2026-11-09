//! # glimmer-tokenizer
//!
//! A byte-level byte-pair-encoding tokenizer, trained from scratch on GLIMMER's own corpus.
//!
//! * **Lossless**: every input is first split into UTF-8 bytes, so there is no unknown token —
//!   emoji, accents, CJK and control characters all round-trip exactly.
//! * **Conversation aware**: dedicated special tokens (`<|system|>`, `<|user|>`,
//!   `<|assistant|>`, `<|end|>`, …) are atomic ids that ordinary text can never produce, which
//!   prevents prompt-format injection from user input.
//! * **Deterministic training**: merge ties are broken by token ids, so the same corpus and
//!   settings always produce the same vocabulary.
//!
//! Id layout: `[0, S)` special tokens, `[S, S+256)` raw bytes, then learned merges in rank order.

pub mod chat;
pub mod pretokenize;
mod stream;
mod train;

pub use stream::StreamDecoder;
pub use train::{TrainerConfig, count_pretokens, train, train_from_counts};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

/// Special tokens, in id order.
pub const SPECIAL_TOKENS: [&str; 8] = [
    "<|pad|>",
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|reserved_0|>",
    "<|reserved_1|>",
];

pub const PAD: u32 = 0;
pub const ENDOFTEXT: u32 = 1;
pub const SYSTEM: u32 = 2;
pub const USER: u32 = 3;
pub const ASSISTANT: u32 = 4;
pub const END: u32 = 5;

const N_SPECIAL: u32 = SPECIAL_TOKENS.len() as u32;
const BYTE_OFFSET: u32 = N_SPECIAL;
const FIRST_MERGE: u32 = N_SPECIAL + 256;

/// Metadata recorded when the tokenizer was trained.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TokenizerMetadata {
    pub trained_on: Vec<String>,
    pub training_bytes: u64,
    pub unique_pretokens: u64,
    pub created_unix: u64,
}

#[derive(Serialize, Deserialize)]
struct TokenizerFile {
    format: String,
    version: u32,
    special_tokens: Vec<String>,
    merges: Vec<(u32, u32)>,
    metadata: TokenizerMetadata,
}

/// A trained byte-level BPE tokenizer.
pub struct Tokenizer {
    merges: Vec<(u32, u32)>,
    /// (left, right) -> merged id; rank = merged id - FIRST_MERGE.
    merge_map: HashMap<(u32, u32), u32>,
    /// Byte string of every token (special tokens map to their literal text).
    token_bytes: Vec<Vec<u8>>,
    pub metadata: TokenizerMetadata,
    cache: RwLock<HashMap<String, Vec<u32>>>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("merges", &self.merges.len())
            .finish()
    }
}

impl Clone for Tokenizer {
    fn clone(&self) -> Self {
        Self::from_merges(self.merges.clone(), self.metadata.clone())
            .expect("cloning a valid tokenizer")
    }
}

const CACHE_LIMIT: usize = 200_000;

impl Tokenizer {
    /// Build a tokenizer from a merge list.
    pub fn from_merges(merges: Vec<(u32, u32)>, metadata: TokenizerMetadata) -> Result<Self> {
        let mut token_bytes: Vec<Vec<u8>> = SPECIAL_TOKENS
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        token_bytes.extend((0u16..256).map(|b| vec![b as u8]));
        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, &(a, b)) in merges.iter().enumerate() {
            let id = FIRST_MERGE + rank as u32;
            if a < BYTE_OFFSET || b < BYTE_OFFSET || a >= id || b >= id {
                bail!("invalid merge #{rank}: ({a}, {b}) references a special or future token");
            }
            let mut bytes = token_bytes[a as usize].clone();
            bytes.extend_from_slice(&token_bytes[b as usize]);
            token_bytes.push(bytes);
            if merge_map.insert((a, b), id).is_some() {
                bail!("duplicate merge ({a}, {b})");
            }
        }
        Ok(Self {
            merges,
            merge_map,
            token_bytes,
            metadata,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// A tokenizer with no merges (pure bytes). Useful for tests.
    pub fn bytes_only() -> Self {
        Self::from_merges(Vec::new(), TokenizerMetadata::default()).unwrap()
    }

    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    pub fn is_special(&self, id: u32) -> bool {
        id < N_SPECIAL
    }

    pub fn special_id(name: &str) -> Option<u32> {
        SPECIAL_TOKENS.iter().position(|s| *s == name).map(|i| i as u32)
    }

    /// Raw bytes of a token.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.token_bytes[id as usize]
    }

    /// Human-readable rendering of one token (lossy for partial UTF-8).
    pub fn token_str(&self, id: u32) -> String {
        String::from_utf8_lossy(self.token_bytes(id)).into_owned()
    }

    /// Encode a single pre-token chunk with BPE.
    fn encode_chunk(&self, chunk: &str, out: &mut Vec<u32>) {
        if let Some(ids) = self.cache.read().unwrap().get(chunk) {
            out.extend_from_slice(ids);
            return;
        }
        let mut ids: Vec<u32> = chunk.bytes().map(|b| BYTE_OFFSET + b as u32).collect();
        // Repeatedly merge the adjacent pair with the lowest rank (earliest learned merge).
        while ids.len() >= 2 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..ids.len() - 1 {
                if let Some(&m) = self.merge_map.get(&(ids[i], ids[i + 1])) {
                    if best.is_none_or(|(bm, _)| m < bm) {
                        best = Some((m, i));
                    }
                }
            }
            let Some((merged, _)) = best else { break };
            let (a, b) = self.merges[(merged - FIRST_MERGE) as usize];
            let mut w = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    w.push(merged);
                    i += 2;
                } else {
                    w.push(ids[i]);
                    i += 1;
                }
            }
            ids = w;
        }
        out.extend_from_slice(&ids);
        let mut cache = self.cache.write().unwrap();
        if cache.len() < CACHE_LIMIT {
            cache.insert(chunk.to_string(), ids);
        }
    }

    /// Encode ordinary text. Special-token strings in `text` are treated as plain text, so user
    /// input can never inject control tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        for chunk in pretokenize::pretokenize(text) {
            self.encode_chunk(chunk, &mut out);
        }
        out
    }

    /// Encode text where literal special-token strings (e.g. `<|user|>`) become their ids.
    /// Only use this for trusted, template-generated text.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let next = SPECIAL_TOKENS
                .iter()
                .enumerate()
                .filter_map(|(id, s)| rest.find(s).map(|pos| (pos, id, s.len())))
                .min_by_key(|&(pos, id, _)| (pos, id));
            match next {
                Some((pos, id, len)) => {
                    out.extend(self.encode(&rest[..pos]));
                    out.push(id as u32);
                    rest = &rest[pos + len..];
                }
                None => {
                    out.extend(self.encode(rest));
                    break;
                }
            }
        }
        out
    }

    /// Encode many documents in parallel.
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>> {
        use rayon::prelude::*;
        texts.par_iter().map(|t| self.encode(t)).collect()
    }

    /// Decode ids to text. Special tokens are rendered as their literal strings when
    /// `include_special` is true and skipped otherwise. Invalid UTF-8 (only possible for
    /// truncated sequences) is replaced with U+FFFD.
    pub fn decode_opts(&self, ids: &[u32], include_special: bool) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if (id as usize) >= self.token_bytes.len() {
                continue;
            }
            if id < N_SPECIAL && !include_special {
                continue;
            }
            bytes.extend_from_slice(&self.token_bytes[id as usize]);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_opts(ids, false)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = TokenizerFile {
            format: "glimmer-bytelevel-bpe".into(),
            version: 1,
            special_tokens: SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect(),
            merges: self.merges.clone(),
            metadata: self.metadata.clone(),
        };
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec(&file)?)
            .with_context(|| format!("writing tokenizer {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading tokenizer {}", path.display()))?;
        let file: TokenizerFile = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing tokenizer {}", path.display()))?;
        if file.format != "glimmer-bytelevel-bpe" || file.version != 1 {
            bail!("unsupported tokenizer format {} v{}", file.format, file.version);
        }
        if file.special_tokens != SPECIAL_TOKENS {
            bail!("tokenizer special tokens do not match this version of GLIMMER");
        }
        Self::from_merges(file.merges, file.metadata)
    }

    /// Average bytes per token over a sample (compression ratio).
    pub fn bytes_per_token(&self, text: &str) -> f64 {
        let n = self.encode(text).len().max(1);
        text.len() as f64 / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_tok() -> Tokenizer {
        let corpus = "G'day mate! How ya going, mate? No worries, mate. Fair dinkum, that's heaps good. \
                      The kangaroo hopped. The koala slept. Reckon it'll rain this arvo? "
            .repeat(20);
        train(
            [corpus.as_str()].into_iter(),
            &TrainerConfig {
                vocab_size: 8 + 256 + 60,
                min_frequency: 2,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn byte_only_roundtrip() {
        let t = Tokenizer::bytes_only();
        assert_eq!(t.vocab_size(), 264);
        let s = "Strewth! 🦘 café";
        assert_eq!(t.decode(&t.encode(s)), s);
    }

    #[test]
    fn trained_roundtrip_and_compression() {
        let t = small_tok();
        assert_eq!(t.vocab_size(), 8 + 256 + 60);
        for s in [
            "G'day mate! How ya going?",
            "No worries — fair dinkum, too easy.",
            "Emoji 🐨🦘 and ünïcödé, 日本",
            "  spaces\t\ttabs\nnewlines\r\n",
            "Numbers 12345 and 3.14159",
            "",
        ] {
            assert_eq!(t.decode(&t.encode(s)), s, "roundtrip failed for {s:?}");
        }
        let sample = "How ya going, mate? No worries, mate.";
        assert!(t.encode(sample).len() < sample.len() / 2, "BPE should compress seen text");
    }

    #[test]
    fn special_tokens_cannot_be_injected() {
        let t = small_tok();
        let ids = t.encode("<|user|>hi<|end|>");
        assert!(ids.iter().all(|&id| !t.is_special(id)));
        assert_eq!(t.decode(&ids), "<|user|>hi<|end|>");
    }

    #[test]
    fn special_tokens_in_templates() {
        let t = small_tok();
        let ids = t.encode_with_special("<|user|>G'day<|end|><|assistant|>");
        assert_eq!(ids[0], USER);
        assert_eq!(ids[ids.len() - 2], END);
        assert_eq!(ids[ids.len() - 1], ASSISTANT);
        assert_eq!(t.decode(&ids), "G'day");
        assert_eq!(t.decode_opts(&ids, true), "<|user|>G'day<|end|><|assistant|>");
    }

    #[test]
    fn save_load_identical() {
        let t = small_tok();
        let dir = std::env::temp_dir().join(format!("glimmer-tok-{}", std::process::id()));
        let path = dir.join("tok.json");
        t.save(&path).unwrap();
        let u = Tokenizer::load(&path).unwrap();
        assert_eq!(t.vocab_size(), u.vocab_size());
        let s = "Reckon it'll rain this arvo, mate?";
        assert_eq!(t.encode(s), u.encode(s));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn invalid_merges_rejected() {
        assert!(Tokenizer::from_merges(vec![(0, 9)], Default::default()).is_err());
        assert!(Tokenizer::from_merges(vec![(10, 400)], Default::default()).is_err());
    }

    #[test]
    fn decode_ignores_out_of_range() {
        let t = Tokenizer::bytes_only();
        let mut ids = t.encode("ok");
        ids.push(99_999);
        assert_eq!(t.decode(&ids), "ok");
    }
}
