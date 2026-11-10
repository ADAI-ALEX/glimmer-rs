//! # glimmer-terminal
//!
//! The interactive GLIMMER laboratory, built with Ratatui. The render loop only draws state and
//! drains channels; training, inference, hardware probing and system monitoring each run on their
//! own threads (see [`workers`]), so the interface stays responsive while the GPU is busy.

pub mod app;
pub mod theme;
pub mod ui;
pub mod workers;

pub use app::Screen;

use anyhow::Result;
use glimmer_core::{BackendKind, Workspace};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Scripted input for demos and screen recordings (`GLIMMER_TUI_SCRIPT=<file>`). Each line is one
/// step, fed through exactly the same key handling as a person typing:
///
/// ```text
/// wait 1500          # milliseconds
/// screen 3           # press a screen number
/// type How ya going? # type text (into the chat box when it has focus)
/// key enter          # enter, esc, tab, up, down, pgup, pgdn, or a single character
/// idle               # wait until STEVIE finishes replying
/// train-idle         # wait until the current training run finishes
/// quit
/// ```
enum Step {
    Wait(u64),
    Keys(Vec<KeyEvent>),
    Idle,
    TrainIdle,
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn parse_script(text: &str) -> VecDeque<Step> {
    let mut steps = VecDeque::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let (cmd, arg) = line.split_once(' ').unwrap_or((line, ""));
        match cmd {
            "wait" => steps.push_back(Step::Wait(arg.trim().parse().unwrap_or(500))),
            "idle" => steps.push_back(Step::Idle),
            "train-idle" => steps.push_back(Step::TrainIdle),
            "type" => steps.push_back(Step::Keys(arg.chars().map(|c| key(KeyCode::Char(c))).collect())),
            "screen" => steps.push_back(Step::Keys(arg.trim().chars().map(|c| key(KeyCode::Char(c))).collect())),
            "quit" => steps.push_back(Step::Keys(vec![KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)])),
            "key" => {
                let k = match arg.trim() {
                    "enter" => KeyCode::Enter,
                    "esc" => KeyCode::Esc,
                    "tab" => KeyCode::Tab,
                    "up" => KeyCode::Up,
                    "down" => KeyCode::Down,
                    "pgup" => KeyCode::PageUp,
                    "pgdn" => KeyCode::PageDown,
                    other => KeyCode::Char(other.chars().next().unwrap_or(' ')),
                };
                steps.push_back(Step::Keys(vec![key(k)]));
            }
            _ => {}
        }
    }
    steps
}

/// Run the terminal UI until the user quits.
pub fn run(ws: Workspace, initial: Screen, model: Option<String>, backend: BackendKind) -> Result<()> {
    let mut script = std::env::var("GLIMMER_TUI_SCRIPT")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| parse_script(&t))
        .unwrap_or_default();
    let mut next_step = Instant::now();
    let mut terminal = ratatui::init();
    let mut app = app::App::new(ws, initial, model, backend);
    let result = (|| -> Result<()> {
        while app.running {
            app.update();
            if Instant::now() >= next_step {
                match script.pop_front() {
                    Some(Step::Wait(ms)) => next_step = Instant::now() + Duration::from_millis(ms),
                    Some(Step::Keys(keys)) => {
                        for k in keys {
                            app.on_key(k);
                        }
                        next_step = Instant::now() + Duration::from_millis(30);
                    }
                    Some(Step::Idle) => {
                        if app.chat.streaming || app.chat.loading.is_some() {
                            script.push_front(Step::Idle);
                        }
                        next_step = Instant::now() + Duration::from_millis(100);
                    }
                    Some(Step::TrainIdle) => {
                        if app.training_active() {
                            script.push_front(Step::TrainIdle);
                        }
                        next_step = Instant::now() + Duration::from_millis(250);
                    }
                    None => {}
                }
            }
            terminal.draw(|f| ui::draw(f, &app))?;
            if event::poll(Duration::from_millis(40))? {
                match event::read()? {
                    Event::Key(k) if k.kind == KeyEventKind::Press => app.on_key(k),
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            }
        }
        Ok(())
    })();
    ratatui::restore();
    // Give a running trainer a moment to save its checkpoint after a graceful stop.
    if let Some(job) = &app.train.job
        && app.training_active()
    {
        eprintln!("waiting for training to save a checkpoint…");
        let start = std::time::Instant::now();
        while start.elapsed() < Duration::from_secs(120) {
            match job.rx.recv_timeout(Duration::from_millis(500)) {
                Ok(workers::TrainMsg::Event(glimmer_training::TrainEvent::Finished(s))) => {
                    eprintln!("saved: {}", s.last_checkpoint.map(|p| p.display().to_string()).unwrap_or_default());
                    break;
                }
                Ok(workers::TrainMsg::Error(_)) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                _ => {}
            }
        }
    }
    result
}
