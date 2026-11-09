//! Text cleaning and quality filtering.

use crate::source::CleaningSpec;
use std::collections::HashSet;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

/// Normalise text: NFC, unified newlines and quotes, no control or zero-width characters,
/// trailing whitespace trimmed, at most one blank line in a row.
pub fn normalize(text: &str) -> String {
    let mut s = String::with_capacity(text.len());
    for c in text.nfc() {
        let mapped = match c {
            '\r' => None,
            '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' => Some('\''),
            '\u{201C}' | '\u{201D}' | '\u{201F}' | '\u{2033}' => Some('"'),
            '\u{00A0}' | '\u{2007}' | '\u{202F}' => Some(' '),
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' | '\u{00AD}' => None,
            '\t' => Some(' '),
            '\n' => Some('\n'),
            c if c.is_control() => None,
            c => Some(c),
        };
        if let Some(c) = mapped {
            s.push(c);
        }
    }
    let s = s.replace('\u{2026}', "...");
    let mut out = String::with_capacity(s.len());
    let mut blank_run = 0;
    for line in s.split('\n') {
        let line = line.trim_end();
        if line.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_string()
}

/// Fraction of alphabetic characters among non-whitespace characters.
pub fn alpha_ratio(text: &str) -> f64 {
    let mut alpha = 0usize;
    let mut total = 0usize;
    for c in text.chars() {
        if !c.is_whitespace() {
            total += 1;
            if c.is_alphabetic() {
                alpha += 1;
            }
        }
    }
    if total == 0 { 0.0 } else { alpha as f64 / total as f64 }
}

/// The blocklist is stored ROT13-encoded in `datasets/filters/blocklist.rot13` so that slurs are
/// not displayed when browsing the repository. It targets racial slurs that appear in some
/// 19th-century texts; paragraphs containing them are dropped rather than edited.
const BLOCKLIST_ROT13: &str = include_str!("../../../datasets/filters/blocklist.rot13");

fn rot13(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='m' | 'A'..='M' => (c as u8 + 13) as char,
            'n'..='z' | 'N'..='Z' => (c as u8 - 13) as char,
            c => c,
        })
        .collect()
}

fn blocklist() -> &'static HashSet<String> {
    static SET: OnceLock<HashSet<String>> = OnceLock::new();
    SET.get_or_init(|| {
        BLOCKLIST_ROT13
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| rot13(l).to_lowercase())
            .collect()
    })
}

/// True if `text` contains a blocklisted term as a whole word (case-insensitive).
pub fn contains_blocked(text: &str) -> bool {
    let set = blocklist();
    text.split(|c: char| !(c.is_alphanumeric() || c == '-'))
        .filter(|w| !w.is_empty())
        .any(|w| set.contains(&w.to_lowercase()))
}

/// Remove Project Gutenberg boilerplate and transcription notes.
pub fn strip_gutenberg(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let is_start = |l: &str| {
        let u = l.to_uppercase();
        u.contains("*** START OF") || u.contains("***START OF") || u.contains("*END*THE SMALL PRINT")
    };
    let is_end = |l: &str| {
        let u = l.to_uppercase();
        u.contains("*** END OF") || u.contains("***END OF") || u.contains("END OF THE PROJECT GUTENBERG")
            || u.contains("END OF PROJECT GUTENBERG")
    };
    let start = lines.iter().position(|l| is_start(l)).map(|i| i + 1).unwrap_or(0);
    let end = lines[start..]
        .iter()
        .position(|l| is_end(l))
        .map(|i| start + i)
        .unwrap_or(lines.len());
    lines[start..end].join("\n")
}

fn is_boilerplate_paragraph(p: &str) -> bool {
    let l = p.to_lowercase();
    const MARKERS: [&str; 10] = [
        "project gutenberg",
        "produced by",
        "distributed proofread",
        "transcriber's note",
        "transcribers note",
        "e-text",
        "etext",
        "ebook",
        "this file was",
        "www.",
    ];
    MARKERS.iter().any(|m| l.contains(m))
}

/// Unwrap hard-wrapped paragraphs (Gutenberg wraps at ~70 columns). Verse — paragraphs whose
/// lines are mostly short — keeps its line breaks.
pub fn unwrap_paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split("\n\n") {
        let lines: Vec<&str> = para
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if lines.is_empty() {
            continue;
        }
        let short = lines.iter().filter(|l| l.chars().count() < 50).count();
        let verse = lines.len() >= 2 && short * 10 >= lines.len() * 7;
        let joined = if verse {
            lines.join("\n")
        } else {
            lines.join(" ")
        };
        // Strip italics markers and illustration tags.
        let cleaned = joined.replace('_', "");
        if cleaned.starts_with("[Illustration") || cleaned.starts_with("[Footnote") {
            continue;
        }
        out.push(cleaned);
    }
    out
}

/// Split a sequence of paragraphs into documents of roughly `target` characters, breaking only
/// on paragraph boundaries.
pub fn chunk_paragraphs(paragraphs: &[String], target: usize) -> Vec<String> {
    let mut docs = Vec::new();
    let mut cur = String::new();
    for p in paragraphs {
        if !cur.is_empty() && cur.len() + p.len() > target {
            docs.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(p);
    }
    if !cur.is_empty() {
        docs.push(cur);
    }
    docs
}

/// Why a document was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rejection {
    TooShort,
    TooLong,
    LowAlpha,
}

/// Apply paragraph-level blocklist filtering and document-level quality filters.
pub fn filter_document(text: &str, spec: &CleaningSpec) -> Result<String, Rejection> {
    let text = if spec.blocklist && contains_blocked(text) {
        text.split("\n\n")
            .filter(|p| !contains_blocked(p))
            .collect::<Vec<_>>()
            .join("\n\n")
    } else {
        text.to_string()
    };
    let n = text.chars().count();
    if n < spec.min_chars {
        return Err(Rejection::TooShort);
    }
    if n > spec.max_chars {
        return Err(Rejection::TooLong);
    }
    if alpha_ratio(&text) < spec.min_alpha_ratio {
        return Err(Rejection::LowAlpha);
    }
    Ok(text)
}

/// Returns true if a paragraph is Gutenberg boilerplate (exposed for the corpus builder).
pub fn is_boilerplate(p: &str) -> bool {
    is_boilerplate_paragraph(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_quotes_whitespace_and_controls() {
        let s = "\u{FEFF}It\u{2019}s \u{201C}grand\u{201D}\r\n\r\n\r\n\r\nNext\u{0007} line \t\n";
        assert_eq!(normalize(s), "It's \"grand\"\n\nNext line");
    }

    #[test]
    fn nfc_composes_accents() {
        assert_eq!(normalize("e\u{0301}"), "\u{00e9}");
    }

    #[test]
    fn rot13_is_involution() {
        assert_eq!(rot13(&rot13("Hello, World")), "Hello, World");
        assert_eq!(rot13("abc"), "nop");
    }

    #[test]
    fn blocklist_loaded_and_whole_word() {
        assert!(!blocklist().is_empty());
        let word = blocklist().iter().next().unwrap().clone();
        assert!(contains_blocked(&format!("some {word} text")));
        assert!(!contains_blocked("perfectly ordinary text about kangaroos"));
    }

    #[test]
    fn strips_gutenberg_header_footer() {
        let t = "Header junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nBody text\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicence";
        assert_eq!(strip_gutenberg(t), "Body text");
    }

    #[test]
    fn unwraps_prose_keeps_verse() {
        let t = "This is a long line of prose that was hard-wrapped by the transcriber\nat roughly seventy columns, as Project Gutenberg texts usually are, and\nit ends here.\n\nThe man from Snowy River\nwas a hard and wiry man\nand rode a pony too";
        let p = unwrap_paragraphs(t);
        assert_eq!(p.len(), 2);
        assert!(!p[0].contains('\n'));
        assert_eq!(p[1].lines().count(), 3);
    }

    #[test]
    fn chunks_on_paragraph_boundaries() {
        let paras: Vec<String> = (0..10).map(|i| format!("para {i} {}", "x".repeat(40))).collect();
        let docs = chunk_paragraphs(&paras, 120);
        assert!(docs.len() > 1);
        assert_eq!(docs.join("\n\n"), paras.join("\n\n"));
    }

    #[test]
    fn quality_filters() {
        let spec = CleaningSpec {
            min_chars: 10,
            ..Default::default()
        };
        assert_eq!(filter_document("short", &spec), Err(Rejection::TooShort));
        assert_eq!(
            filter_document("1234 5678 9012 3456 7890 !!!!", &spec),
            Err(Rejection::LowAlpha)
        );
        assert!(filter_document("A perfectly fine sentence about wombats.", &spec).is_ok());
    }
}
