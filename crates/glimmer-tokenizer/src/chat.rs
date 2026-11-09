//! Conversation format shared by data preparation, training, inference and evaluation.
//!
//! ```text
//! <|system|>You are Stevie…<|end|><|user|>G'day!<|end|><|assistant|>How ya going, mate?<|end|>
//! ```
//!
//! For supervised fine-tuning, [`render_with_mask`] marks which tokens are training targets:
//! only assistant message content and the `<|end|>` that closes each assistant turn. The role
//! tags, system prompt and user turns are context, never targets — the model learns to answer
//! users, not to imitate them, and learns *when to stop* from the supervised `<|end|>`.

use crate::{ASSISTANT, END, SYSTEM, Tokenizer, USER};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn token(self) -> u32 {
        match self {
            Role::System => SYSTEM,
            Role::User => USER,
            Role::Assistant => ASSISTANT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn user(s: impl Into<String>) -> Self {
        Self { role: Role::User, content: s.into() }
    }
    pub fn assistant(s: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: s.into() }
    }
    pub fn system(s: impl Into<String>) -> Self {
        Self { role: Role::System, content: s.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    #[serde(default)]
    pub topic: String,
    pub messages: Vec<Message>,
}

/// Render a conversation to token ids and a per-token target mask (1 = supervised).
pub fn render_with_mask(tok: &Tokenizer, messages: &[Message]) -> (Vec<u32>, Vec<u8>) {
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for m in messages {
        ids.push(m.role.token());
        mask.push(0);
        let content = tok.encode(&m.content);
        let target = u8::from(m.role == Role::Assistant);
        mask.extend(std::iter::repeat_n(target, content.len()));
        ids.extend(content);
        ids.push(END);
        mask.push(target);
    }
    (ids, mask)
}

/// Render a prompt for generation: the conversation so far followed by `<|assistant|>`.
pub fn render_prompt(tok: &Tokenizer, messages: &[Message]) -> Vec<u32> {
    let (mut ids, _) = render_with_mask(tok, messages);
    ids.push(ASSISTANT);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_covers_assistant_content_and_end_only() {
        let tok = Tokenizer::bytes_only();
        let msgs = vec![Message::system("S"), Message::user("hi"), Message::assistant("yo")];
        let (ids, mask) = render_with_mask(&tok, &msgs);
        // <sys> S <end> <user> h i <end> <asst> y o <end>
        assert_eq!(ids.len(), 11);
        assert_eq!(mask, vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]);
        assert_eq!(ids[0], SYSTEM);
        assert_eq!(ids[3], USER);
        assert_eq!(ids[7], ASSISTANT);
        assert_eq!(*ids.last().unwrap(), END);
        assert_eq!(tok.decode(&ids), "Shiyo");
    }

    #[test]
    fn multi_turn_masks_every_assistant_turn() {
        let tok = Tokenizer::bytes_only();
        let msgs = vec![Message::user("a"), Message::assistant("b"), Message::user("c"), Message::assistant("d")];
        let (ids, mask) = render_with_mask(&tok, &msgs);
        let supervised: Vec<u32> = ids.iter().zip(&mask).filter(|(_, m)| **m == 1).map(|(i, _)| *i).collect();
        assert_eq!(tok.decode(&supervised), "bd");
        assert_eq!(supervised.iter().filter(|&&i| i == END).count(), 2);
    }

    #[test]
    fn prompt_ends_with_assistant_tag() {
        let tok = Tokenizer::bytes_only();
        let p = render_prompt(&tok, &[Message::user("G'day")]);
        assert_eq!(*p.last().unwrap(), ASSISTANT);
        assert_eq!(p[0], USER);
    }

    #[test]
    fn json_roundtrip() {
        let c = Conversation {
            id: "x".into(),
            topic: "t".into(),
            messages: vec![Message::user("hi"), Message::assistant("g'day")],
        };
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.contains("\"role\":\"user\""));
        assert_eq!(serde_json::from_str::<Conversation>(&s).unwrap(), c);
    }
}
