//! Text-level metrics for generated responses.
//!
//! These are deliberately simple and transparent. They measure observable properties of the
//! output — length, repetition, whether the turn ended properly, dialect markers, overlap with
//! the prompt — not "intelligence". Each is documented so results can be interpreted honestly.

use glimmer_data::lexicon;
use std::collections::HashSet;

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "with",
    "at", "by", "it", "its", "this", "that", "i", "you", "me", "my", "your", "we", "do", "does", "did", "what", "how",
    "why", "who", "when", "where", "can", "could", "should", "would", "will", "have", "has", "had", "so", "if", "not",
    "no", "yes", "about", "any", "some", "just", "like", "get", "got", "im", "i'm", "it's", "there", "they", "them",
    "he", "she", "his", "her", "our", "us", "up", "out", "as", "from", "than", "then", "too", "very", "really", "much",
];

pub fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .map(|w| w.trim_matches('\'').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

fn content_words(text: &str) -> HashSet<String> {
    words(text).into_iter().filter(|w| w.len() > 2 && !STOPWORDS.contains(&w.as_str())).collect()
}

/// Fraction of distinct n-grams among all n-grams (1.0 = no repetition).
pub fn distinct_n(text: &str, n: usize) -> f64 {
    let w = words(text);
    if w.len() < n {
        return 1.0;
    }
    let grams: Vec<&[String]> = w.windows(n).collect();
    let unique: HashSet<&[String]> = grams.iter().copied().collect();
    unique.len() as f64 / grams.len() as f64
}

/// Fraction of word trigrams that occur more than once (degenerate looping indicator).
pub fn repeated_trigram_rate(text: &str) -> f64 {
    let w = words(text);
    if w.len() < 3 {
        return 0.0;
    }
    let mut seen = HashSet::new();
    let mut repeats = 0;
    let total = w.len() - 2;
    for g in w.windows(3) {
        if !seen.insert(g.to_vec()) {
            repeats += 1;
        }
    }
    repeats as f64 / total as f64
}

/// Whether the response shares at least one content word with the prompt (a crude topicality
/// proxy — it cannot judge correctness or coherence).
pub fn topical_overlap(prompt: &str, response: &str) -> bool {
    let p = content_words(prompt);
    let r = content_words(response);
    !p.is_disjoint(&r)
}

/// Per-response measurements.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ResponseMetrics {
    pub words: usize,
    pub tokens: usize,
    pub ended_turn: bool,
    pub distinct_1: f64,
    pub distinct_2: f64,
    pub repeated_trigrams: f64,
    pub australian_markers: usize,
    pub has_australian_marker: bool,
    pub us_spellings: usize,
    pub topical: bool,
    /// For multi-turn probes: whether an expected context word appears.
    pub continuity: Option<bool>,
    pub tokens_per_sec: f64,
}

pub fn measure(
    prompt: &str,
    response: &str,
    tokens: usize,
    ended: bool,
    expect_any: &[String],
    tps: f64,
) -> ResponseMetrics {
    let style = lexicon::analyse(response);
    let lower = response.to_lowercase();
    ResponseMetrics {
        words: words(response).len(),
        tokens,
        ended_turn: ended,
        distinct_1: distinct_n(response, 1),
        distinct_2: distinct_n(response, 2),
        repeated_trigrams: repeated_trigram_rate(response),
        australian_markers: style.slang_hits + style.au_spellings,
        has_australian_marker: style.has_australian_marker(),
        us_spellings: style.us_spellings,
        topical: topical_overlap(prompt, response),
        continuity: (!expect_any.is_empty()).then(|| expect_any.iter().any(|w| lower.contains(&w.to_lowercase()))),
        tokens_per_sec: tps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_metrics() {
        assert_eq!(distinct_n("a b c d", 1), 1.0);
        assert!(distinct_n("the the the the", 1) < 0.3);
        assert!(repeated_trigram_rate("go go go go go go") > 0.5);
        assert_eq!(repeated_trigram_rate("one two three four five"), 0.0);
    }

    #[test]
    fn topicality_and_continuity() {
        assert!(topical_overlap("Tell me about wombats", "Wombats dig burrows."));
        assert!(!topical_overlap("Tell me about wombats", "The sky is blue."));
        let m = measure("What's my name?", "It's Tom, mate!", 5, true, &["Tom".into()], 10.0);
        assert_eq!(m.continuity, Some(true));
        assert!(m.has_australian_marker);
    }
}
