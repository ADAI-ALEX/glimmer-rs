//! The STEVIE conversation corpus format, validation and quality control.
//!
//! Conversations are authored as plain text for easy review and diffing:
//!
//! ```text
//! # Comments start with '#'.
//! ### greet-001 | greetings
//! U: How ya going, mate?
//! S: Not too bad at all! Having a ripper of a day. How's yours?
//! ```
//!
//! * `### <id> | <topic>` starts a conversation; ids are unique across all files.
//! * `U:` lines are the user, `S:` lines are STEVIE. Turns must alternate, start with the user
//!   and end with STEVIE. One message per line.

use crate::dedup::{self, MinHashConfig};
use crate::source::SourceSpec;
use anyhow::{Context, Result};
use glimmer_tokenizer::chat::{Conversation, Message, Role};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

/// Default system prompt used for STEVIE (kept short: every token costs context).
pub const SYSTEM_PROMPT: &str = "You are Stevie, a friendly Australian AI mate.";

#[derive(Debug, Default)]
pub struct LoadReport {
    pub conversations: Vec<Conversation>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub files: usize,
}

/// Phrases STEVIE must never use: it is an original character, not an impersonation.
const FORBIDDEN: [&str; 4] = ["steve irwin", "crikey", "i'm steve", "i am steve"];

pub fn parse(text: &str, file: &str, report: &mut LoadReport) {
    let mut current: Option<(Conversation, usize)> = None;
    let finish = |conv: Option<(Conversation, usize)>, report: &mut LoadReport| {
        if let Some((c, line)) = conv {
            if c.messages.is_empty() {
                report.errors.push(format!("{file}:{line}: conversation {} has no messages", c.id));
            } else if c.messages.last().unwrap().role != Role::Assistant {
                report.errors.push(format!("{file}:{line}: conversation {} must end with S:", c.id));
            } else {
                report.conversations.push(c);
            }
        }
    };
    for (n, raw) in text.lines().enumerate() {
        let lineno = n + 1;
        let line = raw.trim_end();
        if line.trim().is_empty() || line.starts_with('#') && !line.starts_with("###") {
            continue;
        }
        if let Some(head) = line.strip_prefix("###") {
            finish(current.take(), report);
            let mut parts = head.splitn(2, '|');
            let id = parts.next().unwrap_or("").trim().to_string();
            let topic = parts.next().unwrap_or("").trim().to_string();
            if id.is_empty() {
                report.errors.push(format!("{file}:{lineno}: missing conversation id"));
            }
            current = Some((
                Conversation {
                    id,
                    topic,
                    messages: Vec::new(),
                },
                lineno,
            ));
            continue;
        }
        let (role, content) = if let Some(c) = line.strip_prefix("U:") {
            (Role::User, c.trim())
        } else if let Some(c) = line.strip_prefix("S:") {
            (Role::Assistant, c.trim())
        } else {
            report
                .errors
                .push(format!("{file}:{lineno}: expected 'U:', 'S:' or '###', got: {line}"));
            continue;
        };
        let Some((conv, _)) = current.as_mut() else {
            report.errors.push(format!("{file}:{lineno}: message outside a conversation"));
            continue;
        };
        if content.is_empty() {
            report.errors.push(format!("{file}:{lineno}: empty message"));
            continue;
        }
        let expected = if conv.messages.len() % 2 == 0 {
            Role::User
        } else {
            Role::Assistant
        };
        if role != expected {
            report.errors.push(format!(
                "{file}:{lineno}: turns must alternate U/S starting with U (conversation {})",
                conv.id
            ));
            continue;
        }
        if role == Role::Assistant {
            let lower = content.to_lowercase();
            for f in FORBIDDEN {
                if lower.contains(f) {
                    report.warnings.push(format!(
                        "{file}:{lineno}: STEVIE uses '{f}' — avoid impersonating a real person"
                    ));
                }
            }
        }
        conv.messages.push(Message {
            role,
            content: content.to_string(),
        });
    }
    finish(current.take(), report);
}

/// Load and validate every `*.txt` file in a directory.
pub fn load_dir(dir: &Path) -> Result<LoadReport> {
    let mut report = LoadReport::default();
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading conversation directory {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    files.sort();
    for f in &files {
        let text = std::fs::read_to_string(f)?;
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        parse(&text, &name, &mut report);
    }
    report.files = files.len();
    let mut seen = HashSet::new();
    for c in &report.conversations {
        if !seen.insert(c.id.clone()) {
            report.errors.push(format!("duplicate conversation id {}", c.id));
        }
    }
    Ok(report)
}

/// Text used for duplicate detection: all message contents.
pub fn conversation_text(c: &Conversation) -> String {
    c.messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join(" \n ")
}

pub struct FilterResult {
    pub kept: Vec<Conversation>,
    pub rejected: BTreeMap<String, usize>,
    pub exact_duplicates: usize,
    pub near_duplicates: usize,
}

/// Length limits, exact and near-duplicate removal.
pub fn quality_filter(convs: Vec<Conversation>, spec: &SourceSpec, log: &mut dyn FnMut(&str)) -> FilterResult {
    let mut rejected = BTreeMap::new();
    let max_chars = spec.cleaning.max_chars;
    let convs: Vec<Conversation> = convs
        .into_iter()
        .filter(|c| {
            let ok = c.messages.iter().all(|m| m.content.chars().count() <= max_chars);
            if !ok {
                *rejected.entry("MessageTooLong".to_string()).or_insert(0) += 1;
            }
            ok
        })
        .collect();
    let texts: Vec<String> = convs.iter().map(conversation_text).collect();
    let keep = dedup::exact_dedup(&texts);
    let exact = convs.len() - keep.len();
    let convs: Vec<Conversation> = keep.into_iter().map(|i| convs[i].clone()).collect();
    let mut near = 0;
    let convs = if spec.cleaning.near_dedup {
        let texts: Vec<String> = convs.iter().map(conversation_text).collect();
        let r = dedup::near_dedup(&texts, spec.cleaning.near_dedup_threshold, &MinHashConfig::default());
        for (d, k, s) in &r.dropped {
            log(&format!(
                "  near-duplicate ({s:.2}): {} ≈ {} — dropped {}",
                convs[*d].id, convs[*k].id, convs[*d].id
            ));
        }
        near = r.dropped.len();
        r.keep.into_iter().map(|i| convs[i].clone()).collect()
    } else {
        convs
    };
    FilterResult {
        kept: convs,
        rejected,
        exact_duplicates: exact,
        near_duplicates: near,
    }
}

/// A fixed held-out evaluation prompt. Never used for training.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalPrompt {
    pub id: String,
    pub category: String,
    /// Conversation so far; must end with a user message.
    pub messages: Vec<Message>,
    /// Optional words a contextually-aware reply would plausibly contain (continuity probe).
    #[serde(default)]
    pub expect_any: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvalFile {
    prompt: Vec<EvalPrompt>,
}

pub fn load_eval_prompts(path: &Path) -> Result<Vec<EvalPrompt>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading eval prompts {}", path.display()))?;
    let f: EvalFile = toml::from_str(&text)?;
    for p in &f.prompt {
        anyhow::ensure!(
            p.messages.last().is_some_and(|m| m.role == Role::User),
            "eval prompt {} must end with a user message",
            p.id
        );
    }
    Ok(f.prompt)
}

/// Training conversations whose user turns closely match a held-out eval prompt.
pub fn leakage(train: &[Conversation], prompts: &[EvalPrompt], threshold: f64) -> Vec<(String, String, f64)> {
    let cfg = MinHashConfig::default();
    let prompt_sigs: Vec<(String, Vec<u64>, String)> = prompts
        .iter()
        .flat_map(|p| {
            p.messages
                .iter()
                .filter(|m| m.role == Role::User)
                .map(|m| (p.id.clone(), dedup::signature(&m.content, &cfg), dedup::canonical(&m.content)))
                .collect::<Vec<_>>()
        })
        .collect();
    let mut hits = Vec::new();
    for c in train {
        for m in c.messages.iter().filter(|m| m.role == Role::User) {
            let canon = dedup::canonical(&m.content);
            let sig = dedup::signature(&m.content, &cfg);
            for (pid, psig, pcanon) in &prompt_sigs {
                let sim = if &canon == pcanon {
                    1.0
                } else {
                    dedup::estimated_jaccard(&sig, psig)
                };
                if sim >= threshold {
                    hits.push((c.id.clone(), pid.clone(), sim));
                }
            }
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_file() {
        let text = "# comment\n### a-1 | greetings\nU: G'day\nS: How ya going?\nU: Good\nS: Ripper.\n\n### a-2 | food\nU: Brekkie?\nS: Vegemite toast, mate.\n";
        let mut r = LoadReport::default();
        parse(text, "t.txt", &mut r);
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert_eq!(r.conversations.len(), 2);
        assert_eq!(r.conversations[0].topic, "greetings");
        assert_eq!(r.conversations[0].messages.len(), 4);
        assert_eq!(r.conversations[1].messages[1].role, Role::Assistant);
    }

    #[test]
    fn reports_structural_errors() {
        let text = "### x | t\nS: starts with stevie\n### y | t\nU: no reply\n### z | t\nU: a\nU: b\nS: c\nfree text\n";
        let mut r = LoadReport::default();
        parse(text, "t.txt", &mut r);
        assert!(r.errors.len() >= 3, "{:?}", r.errors);
    }

    #[test]
    fn warns_on_impersonation() {
        let mut r = LoadReport::default();
        parse("### x | t\nU: who are you\nS: Crikey, I'm Steve Irwin!\n", "t.txt", &mut r);
        assert!(!r.warnings.is_empty());
    }

    #[test]
    fn detects_leakage() {
        let train = vec![Conversation {
            id: "c1".into(),
            topic: String::new(),
            messages: vec![Message::user("How are you going, mate?"), Message::assistant("Good!")],
        }];
        let prompts = vec![EvalPrompt {
            id: "p1".into(),
            category: "greeting".into(),
            messages: vec![Message::user("how are you going, mate?")],
            expect_any: vec![],
        }];
        assert_eq!(leakage(&train, &prompts, 0.8).len(), 1);
    }
}
