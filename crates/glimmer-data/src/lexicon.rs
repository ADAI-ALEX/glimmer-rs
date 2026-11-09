//! Australian English lexicon used to measure dialect style in data and model outputs.
//!
//! The style metric is deliberately simple and transparent: it counts recognisable Australian
//! vocabulary and British/Australian spellings, and penalises American spellings. It measures
//! whether STEVIE *sounds* Australian, not whether it is correct or helpful.

use std::collections::HashSet;
use std::sync::OnceLock;

/// Australian colloquialisms and vocabulary (lower-case; multi-word phrases allowed).
pub const SLANG: &[&str] = &[
    "mate",
    "reckon",
    "heaps",
    "no worries",
    "fair dinkum",
    "dinkum",
    "arvo",
    "brekkie",
    "good on ya",
    "good on you",
    "too easy",
    "strewth",
    "bloody",
    "g'day",
    "ripper",
    "bonza",
    "stoked",
    "she'll be right",
    "how ya going",
    "how are you going",
    "keen",
    "chuck",
    "servo",
    "bottle-o",
    "maccas",
    "barbie",
    "snag",
    "snags",
    "thongs",
    "esky",
    "ute",
    "footy",
    "cuppa",
    "tucker",
    "bush",
    "outback",
    "sheila",
    "bloke",
    "fella",
    "cobber",
    "crook",
    "dunny",
    "stubby",
    "tinnie",
    "sunnies",
    "brekky",
    "avo",
    "smoko",
    "suss",
    "sussed",
    "spewin",
    "spewing",
    "wombat",
    "kangaroo",
    "roo",
    "koala",
    "galah",
    "drongo",
    "larrikin",
    "true blue",
    "rellies",
    "rello",
    "chockers",
    "chockablock",
    "flat out",
    "hard yakka",
    "yakka",
    "knackered",
    "stuffed",
    "rapt",
    "sweet as",
    "yeah nah",
    "nah yeah",
    "deadset",
    "dead set",
    "ta",
    "cheers",
    "legend",
    "you beauty",
    "beauty",
    "cooked",
    "that's cooked",
    "devo",
    "devastated",
    "brolly",
    "doona",
    "lollies",
    "mozzie",
    "mozzies",
    "bogan",
    "bludger",
    "ripsnorter",
    "billabong",
    "billy",
    "swag",
    "jumbuck",
    "bushwalk",
    "togs",
    "cossie",
    "cozzie",
    "bathers",
    "capsicum",
    "rockmelon",
    "petrol station",
    "uni",
    "trackies",
    "op shop",
    "chook",
    "chooks",
    "lamington",
    "pavlova",
    "vegemite",
    "tim tam",
    "sanger",
    "sanga",
    "pash",
    "dag",
    "daggy",
    "grouse",
    "aussie",
    "straya",
    "hooroo",
    "catch ya",
    "see ya later",
    "onya",
    "fair go",
    "good one",
];

/// Spellings preferred in Australian English.
pub const AU_SPELLINGS: &[&str] = &[
    "colour",
    "colours",
    "favourite",
    "favourites",
    "realise",
    "realised",
    "organise",
    "organised",
    "organisation",
    "behaviour",
    "neighbour",
    "neighbours",
    "honour",
    "humour",
    "centre",
    "metre",
    "metres",
    "litre",
    "litres",
    "theatre",
    "travelling",
    "travelled",
    "analyse",
    "recognise",
    "recognised",
    "apologise",
    "programme",
    "defence",
    "licence",
    "mum",
    "mums",
    "maths",
    "aluminium",
    "jewellery",
    "grey",
    "tyre",
    "tyres",
    "cheque",
    "learnt",
    "practise",
    "optimise",
    "optimisation",
    "flavour",
    "labour",
    "harbour",
];

/// American spellings that an Australian writer would not use.
pub const US_SPELLINGS: &[&str] = &[
    "color",
    "colors",
    "favorite",
    "favorites",
    "realize",
    "realized",
    "organize",
    "organized",
    "organization",
    "behavior",
    "neighbor",
    "neighbors",
    "honor",
    "humor",
    "center",
    "meter",
    "meters",
    "liter",
    "liters",
    "theater",
    "traveling",
    "traveled",
    "analyze",
    "recognize",
    "recognized",
    "apologize",
    "defense",
    "mom",
    "moms",
    "math",
    "aluminum",
    "jewelry",
    "gray",
    "tire",
    "tires",
    "optimize",
    "optimization",
    "flavor",
    "labor",
    "harbor",
];

fn set(words: &'static [&'static str]) -> HashSet<&'static str> {
    words.iter().copied().filter(|w| !w.contains(' ')).collect()
}

fn slang_words() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| set(SLANG))
}
fn au_spell() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| set(AU_SPELLINGS))
}
fn us_spell() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| set(US_SPELLINGS))
}

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '-'))
        .map(|w| w.trim_matches('\'').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Dialect features found in a text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StyleCounts {
    pub words: usize,
    pub slang_hits: usize,
    pub au_spellings: usize,
    pub us_spellings: usize,
    pub matched: Vec<String>,
}

impl StyleCounts {
    pub fn has_australian_marker(&self) -> bool {
        self.slang_hits + self.au_spellings > 0
    }
}

pub fn analyse(text: &str) -> StyleCounts {
    let ws = words(text);
    let lower = format!(" {} ", ws.join(" "));
    let mut c = StyleCounts { words: ws.len(), ..Default::default() };
    for w in &ws {
        if slang_words().contains(w.as_str()) {
            c.slang_hits += 1;
            c.matched.push(w.clone());
        }
        if au_spell().contains(w.as_str()) {
            c.au_spellings += 1;
            c.matched.push(w.clone());
        }
        if us_spell().contains(w.as_str()) {
            c.us_spellings += 1;
        }
    }
    for phrase in SLANG.iter().filter(|p| p.contains(' ')) {
        let needle = format!(" {phrase} ");
        let n = lower.matches(&needle).count();
        if n > 0 {
            c.slang_hits += n;
            c.matched.push(phrase.to_string());
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_slang_and_spelling() {
        let c = analyse("G'day mate! No worries, my favourite colour is blue. Reckon it'll rain this arvo.");
        assert!(c.slang_hits >= 4, "{c:?}");
        assert_eq!(c.au_spellings, 2);
        assert_eq!(c.us_spellings, 0);
        assert!(c.has_australian_marker());
    }

    #[test]
    fn detects_american_spelling() {
        let c = analyse("My favorite color is gray.");
        assert_eq!(c.us_spellings, 3);
        assert!(!c.has_australian_marker());
    }
}
