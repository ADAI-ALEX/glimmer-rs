//! Minimal ANSI styling for headless CLI output. Honours `NO_COLOR` and non-terminal stdout.

use std::io::IsTerminal;
use std::sync::OnceLock;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal())
}

fn paint(code: &str, s: &str) -> String {
    if enabled() { format!("\x1b[{code}m{s}\x1b[0m") } else { s.to_string() }
}

pub fn bold(s: &str) -> String {
    paint("1", s)
}
pub fn dim(s: &str) -> String {
    paint("2", s)
}
/// GLIMMER accent (soft gold).
pub fn accent(s: &str) -> String {
    paint("38;5;221", s)
}
/// STEVIE accent (eucalyptus green).
pub fn gum(s: &str) -> String {
    paint("38;5;114", s)
}
pub fn ok(s: &str) -> String {
    paint("38;5;114", s)
}
pub fn warn(s: &str) -> String {
    paint("38;5;214", s)
}
pub fn err(s: &str) -> String {
    paint("38;5;203", s)
}
pub fn info(s: &str) -> String {
    paint("38;5;111", s)
}

pub fn heading(s: &str) {
    println!("\n{} {}", accent("◆"), bold(s));
}

pub fn kv(key: &str, value: impl std::fmt::Display) {
    println!("  {:<22} {}", dim(key), value);
}
