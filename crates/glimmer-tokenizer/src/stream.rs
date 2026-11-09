//! Incremental decoding for streaming generation.
//!
//! A single token may end in the middle of a multi-byte UTF-8 character (e.g. half of an emoji).
//! [`StreamDecoder`] buffers incomplete trailing bytes and only emits complete characters, so the
//! chat UI never displays replacement glyphs for text that is still being generated.

use crate::Tokenizer;

#[derive(Debug, Default, Clone)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one token; returns any newly completed text. Special tokens produce no text.
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        if tok.is_special(id) || (id as usize) >= tok.vocab_size() {
            return String::new();
        }
        self.pending.extend_from_slice(tok.token_bytes(id));
        self.drain_valid()
    }

    fn drain_valid(&mut self) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        // Incomplete sequence at the end: wait for more bytes.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        // Genuinely invalid bytes: replace and continue.
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + n);
                        }
                    }
                }
            }
        }
    }

    /// Flush whatever remains (lossy) at the end of generation.
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emoji_split_across_tokens() {
        let tok = Tokenizer::bytes_only();
        let text = "hi 🦘!";
        let ids = tok.encode(text);
        let mut d = StreamDecoder::new();
        let mut out = String::new();
        for id in ids {
            let piece = d.push(&tok, id);
            assert!(!piece.contains('\u{FFFD}'));
            out.push_str(&piece);
        }
        out.push_str(&d.finish());
        assert_eq!(out, text);
    }
}
