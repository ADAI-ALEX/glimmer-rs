//! Pre-tokenisation: split text into chunks that BPE merges never cross.
//!
//! The rules follow the spirit of GPT-2's regex
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//! implemented as a hand-written scanner (Rust's `regex` crate has no look-ahead):
//!
//! * English contractions are split off (`don't` → `don` `'t`).
//! * A single leading space attaches to the following word, digit or punctuation run.
//! * Digits are split individually, which helps small models with arithmetic.
//! * Runs of whitespace are kept together, except that the final space before a word is
//!   donated to that word.
//!
//! Every byte of the input belongs to exactly one chunk, so concatenating chunks reproduces the
//! input exactly — the property that makes byte-level BPE lossless.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Letter,
    Digit,
    Space,
    Other,
}

fn class(c: char) -> Class {
    if c.is_alphabetic() {
        Class::Letter
    } else if c.is_numeric() {
        Class::Digit
    } else if c.is_whitespace() {
        Class::Space
    } else {
        Class::Other
    }
}

const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

/// Length in bytes of a contraction starting at `s`, if any (ASCII case-insensitive).
fn contraction_len(s: &str) -> Option<usize> {
    for c in CONTRACTIONS {
        if s.len() >= c.len() && s.as_bytes()[..c.len()].eq_ignore_ascii_case(c.as_bytes()) {
            // Only treat as a contraction when not followed by another letter run that would make
            // it part of a longer word (e.g. "'sup" stays punctuation + word).
            let rest = &s[c.len()..];
            let next_is_letter = rest.chars().next().is_some_and(|ch| ch.is_alphabetic());
            if !next_is_letter {
                return Some(c.len());
            }
        }
    }
    None
}

/// Split `text` into pre-tokens (borrowed slices).
pub fn pretokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::with_capacity(text.len() / 4 + 1);
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| if i < n { chars[i].0 } else { text.len() };
    let mut i = 0;
    while i < n {
        let (start, c) = chars[i];
        // Contractions.
        if c == '\'' {
            if let Some(len) = contraction_len(&text[start..]) {
                out.push(&text[start..start + len]);
                // Advance by however many chars the contraction spans (all ASCII).
                i += len;
                continue;
            }
        }
        let cls = class(c);
        match cls {
            Class::Space => {
                // Whitespace run [i, j).
                let mut j = i;
                while j < n && class(chars[j].1) == Class::Space {
                    j += 1;
                }
                // If followed by a non-space and the run ends with a plain space, donate it.
                let donate = j < n && chars[j - 1].1 == ' ';
                let end = if donate { j - 1 } else { j };
                if end > i {
                    out.push(&text[start..byte_at(end)]);
                }
                i = end;
                if donate {
                    // The space starts the next chunk.
                    let next_cls = class(chars[j].1);
                    let mut k = j;
                    match next_cls {
                        Class::Letter => {
                            while k < n && class(chars[k].1) == Class::Letter {
                                k += 1;
                            }
                        }
                        Class::Digit => k += 1,
                        Class::Other => {
                            if chars[k].1 == '\'' && contraction_len(&text[chars[k].0..]).is_some()
                            {
                                // " 's" — keep the space alone; the contraction is handled next.
                                out.push(&text[byte_at(i)..byte_at(j)]);
                                i = j;
                                continue;
                            }
                            while k < n && class(chars[k].1) == Class::Other {
                                k += 1;
                            }
                        }
                        Class::Space => unreachable!(),
                    }
                    out.push(&text[byte_at(i)..byte_at(k)]);
                    i = k;
                }
            }
            Class::Letter => {
                let mut j = i;
                while j < n && class(chars[j].1) == Class::Letter {
                    j += 1;
                }
                out.push(&text[start..byte_at(j)]);
                i = j;
            }
            Class::Digit => {
                out.push(&text[start..byte_at(i + 1)]);
                i += 1;
            }
            Class::Other => {
                let mut j = i;
                while j < n && class(chars[j].1) == Class::Other {
                    if j > i && chars[j].1 == '\'' && contraction_len(&text[chars[j].0..]).is_some()
                    {
                        break;
                    }
                    j += 1;
                }
                out.push(&text[start..byte_at(j)]);
                i = j;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_lossless(s: &str) {
        let parts = pretokenize(s);
        assert_eq!(parts.concat(), s, "parts: {parts:?}");
        assert!(parts.iter().all(|p| !p.is_empty()), "empty chunk in {parts:?}");
    }

    #[test]
    fn basic_words_and_spaces() {
        assert_eq!(
            pretokenize("How ya going, mate?"),
            vec!["How", " ya", " going", ",", " mate", "?"]
        );
    }

    #[test]
    fn contractions_split() {
        assert_eq!(pretokenize("I'm sure it's"), vec!["I", "'m", " sure", " it", "'s"]);
        assert_eq!(pretokenize("don't"), vec!["don", "'t"]);
    }

    #[test]
    fn australian_slang() {
        let s = "G'day! Fair dinkum, she'll be right. Arvo brekkie?";
        check_lossless(s);
        let parts = pretokenize(s);
        assert!(parts.contains(&" dinkum"));
        assert!(parts.contains(&"'ll"));
    }

    #[test]
    fn digits_individual() {
        assert_eq!(pretokenize("in 2026"), vec!["in", " 2", "0", "2", "6"]);
    }

    #[test]
    fn whitespace_runs() {
        assert_eq!(pretokenize("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pretokenize("a\n\nb"), vec!["a", "\n\n", "b"]);
        assert_eq!(pretokenize("end   "), vec!["end", "   "]);
    }

    #[test]
    fn unicode_and_emoji_lossless() {
        for s in [
            "Café naïve résumé",
            "🦘 kangaroos! 🐨🐨",
            "日本語のテキスト",
            "tab\tsep\r\nwindows",
            "  leading",
            "'quoted'",
            "it's  's",
            "x '",
            "",
            "\u{00a0}nbsp",
            "e\u{0301}",
        ] {
            check_lossless(s);
        }
    }

    #[test]
    fn random_strings_lossless() {
        let alphabet: Vec<char> = "ab Z9'’.,!? \n\t🦘é-_".chars().collect();
        let mut state = 12345u64;
        for _ in 0..2000 {
            let len = (state % 24) as usize;
            let mut s = String::new();
            for _ in 0..len {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s.push(alphabet[(state >> 33) as usize % alphabet.len()]);
            }
            check_lossless(&s);
        }
    }
}
