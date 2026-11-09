//! Deterministic train/validation/test assignment.
//!
//! A document's split is a pure function of its id and the split seed, so adding or removing
//! other documents never moves an existing document between splits, and upsampled copies of a
//! document always land in the same split (no train/validation leakage).

use crate::hash_unit;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
    Test,
}

impl Split {
    pub const ALL: [Split; 3] = [Split::Train, Split::Val, Split::Test];

    pub fn as_str(&self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SplitSpec {
    pub val_fraction: f64,
    pub test_fraction: f64,
    pub seed: u64,
}

impl Default for SplitSpec {
    fn default() -> Self {
        Self {
            val_fraction: 0.01,
            test_fraction: 0.01,
            seed: 7,
        }
    }
}

impl SplitSpec {
    pub fn assign(&self, doc_id: &str) -> Split {
        let mut key = self.seed.to_le_bytes().to_vec();
        key.extend_from_slice(doc_id.as_bytes());
        let u = hash_unit(&key);
        if u < self.test_fraction {
            Split::Test
        } else if u < self.test_fraction + self.val_fraction {
            Split::Val
        } else {
            Split::Train
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_proportional() {
        let spec = SplitSpec {
            val_fraction: 0.1,
            test_fraction: 0.05,
            seed: 1,
        };
        let mut counts = std::collections::HashMap::new();
        for i in 0..20_000 {
            let id = format!("doc-{i}");
            let s = spec.assign(&id);
            assert_eq!(s, spec.assign(&id));
            *counts.entry(s).or_insert(0usize) += 1;
        }
        let val = counts[&Split::Val] as f64 / 20_000.0;
        let test = counts[&Split::Test] as f64 / 20_000.0;
        assert!((val - 0.1).abs() < 0.015, "{val}");
        assert!((test - 0.05).abs() < 0.01, "{test}");
    }

    #[test]
    fn seed_changes_assignment() {
        let a = SplitSpec { seed: 1, ..Default::default() };
        let b = SplitSpec { seed: 2, ..Default::default() };
        let differ = (0..2000)
            .filter(|i| a.assign(&i.to_string()) != b.assign(&i.to_string()))
            .count();
        assert!(differ > 0);
    }
}
