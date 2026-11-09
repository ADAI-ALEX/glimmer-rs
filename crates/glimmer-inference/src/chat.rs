//! Chat sessions: conversation history, context-window management and persistence.

use crate::engine::{Engine, Generation};
use crate::sampling::SamplingConfig;
use anyhow::{Context, Result, bail};
use glimmer_tokenizer::Tokenizer;
use glimmer_tokenizer::chat::{Message, Role, render_prompt};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub sampling: SamplingConfig,
    pub created_unix: u64,
}

/// A prompt assembled to fit the context window.
#[derive(Debug, Clone)]
pub struct FittedPrompt {
    pub tokens: Vec<u32>,
    /// Number of oldest messages left out to make room.
    pub dropped_messages: usize,
}

impl ChatSession {
    pub fn new(model: &str, system_prompt: Option<String>, sampling: SamplingConfig) -> Self {
        let now = glimmer_core::time::now_unix();
        Self {
            id: format!("chat-{}", glimmer_core::time::compact_stamp(now)),
            model: model.to_string(),
            system_prompt,
            messages: Vec::new(),
            sampling,
            created_unix: now,
        }
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Build the prompt for the next assistant turn. The system prompt is always kept; the
    /// oldest user/assistant exchanges are dropped until the prompt plus the generation budget
    /// fits within `context_length`.
    pub fn fit_prompt(&self, tok: &Tokenizer, context_length: usize, reserve: usize) -> Result<FittedPrompt> {
        if self.messages.last().map(|m| m.role) != Some(Role::User) {
            bail!("the conversation must end with a user message");
        }
        let budget = context_length.saturating_sub(reserve).max(16);
        let mut start = 0;
        loop {
            let mut msgs = Vec::with_capacity(self.messages.len() + 1);
            if let Some(s) = &self.system_prompt {
                msgs.push(Message::system(s.clone()));
            }
            msgs.extend(self.messages[start..].iter().cloned());
            let tokens = render_prompt(tok, &msgs);
            if tokens.len() <= budget || start + 1 >= self.messages.len() {
                if tokens.len() >= context_length {
                    bail!("the latest message alone ({} tokens) exceeds the context window", tokens.len());
                }
                return Ok(FittedPrompt {
                    tokens,
                    dropped_messages: start,
                });
            }
            // Drop the oldest exchange (user + assistant) to keep turns aligned.
            start += 2.min(self.messages.len() - 1 - start);
        }
    }

    /// Add a user message, generate STEVIE's reply with streaming, and record it.
    pub fn respond(
        &mut self,
        engine: &mut Engine,
        user: &str,
        on_text: impl FnMut(&str) -> bool,
    ) -> Result<(Generation, usize)> {
        self.messages.push(Message::user(user.trim()));
        let reserve = self.sampling.max_new_tokens.min(engine.context_length() / 2);
        let fitted = self.fit_prompt(&engine.tokenizer, engine.context_length(), reserve)?;
        let generation = engine.generate(&fitted.tokens, &self.sampling, on_text)?;
        self.messages.push(Message::assistant(generation.text.trim()));
        Ok((generation, fitted.dropped_messages))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_recent_turns_within_budget() {
        let tok = Tokenizer::bytes_only();
        let mut s = ChatSession::new("m", Some("SYS".into()), SamplingConfig::default());
        for i in 0..10 {
            s.messages.push(Message::user(format!("question number {i} padded out")));
            s.messages.push(Message::assistant(format!("answer number {i} padded out too")));
        }
        s.messages.push(Message::user("latest?"));
        let fitted = s.fit_prompt(&tok, 200, 40).unwrap();
        assert!(fitted.tokens.len() <= 160);
        assert!(fitted.dropped_messages > 0 && fitted.dropped_messages % 2 == 0);
        let text = tok.decode(&fitted.tokens);
        assert!(text.starts_with("SYS"), "system prompt must be kept");
        assert!(text.ends_with("latest?"), "latest message must be kept");
        assert!(text.contains("answer number 9"), "most recent exchange kept");
        assert!(!text.contains("question number 0 "), "oldest exchange dropped");
    }

    #[test]
    fn requires_user_turn_last() {
        let tok = Tokenizer::bytes_only();
        let mut s = ChatSession::new("m", None, SamplingConfig::default());
        s.messages.push(Message::user("hi"));
        s.messages.push(Message::assistant("g'day"));
        assert!(s.fit_prompt(&tok, 100, 10).is_err());
    }

    #[test]
    fn save_load_roundtrip() {
        let mut s = ChatSession::new("stevie", None, SamplingConfig::default());
        s.messages.push(Message::user("G'day"));
        s.messages.push(Message::assistant("How ya going?"));
        let p = std::env::temp_dir().join(format!("glimmer-chat-{}.json", std::process::id()));
        s.save(&p).unwrap();
        let t = ChatSession::load(&p).unwrap();
        assert_eq!(t.messages, s.messages);
        assert_eq!(t.id, s.id);
        std::fs::remove_file(p).ok();
    }
}
