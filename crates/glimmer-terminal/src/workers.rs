//! Background workers. Expensive work never runs on the render thread:
//!
//! * **training** — the trainer runs on its own thread and streams events over a channel;
//!   pause/stop/checkpoint go through lock-free atomics.
//! * **chat** — an inference thread owns the model and streams tokens back as they decode.
//! * **monitor** — samples GPU counters and memory once a second.
//! * **probe** — inspects hardware and verifies backends once at start-up.

use glimmer_core::config::TrainingConfig;
use glimmer_core::hardware::{HardwareReport, ProcessMemory, system_memory};
use glimmer_core::registry::ModelCard;
use glimmer_core::{BackendKind, Workspace, gpu};
use glimmer_inference::{ChatSession, Engine, GenerationStats, SamplingConfig, StopReason};
use glimmer_models::backend::{self, Verification};
use glimmer_tokenizer::chat::Message;
use glimmer_training::{TrainControl, TrainEvent, TrainOptions, train};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::Duration;

// ── Training ────────────────────────────────────────────────────────────────────────────────

pub enum TrainMsg {
    Event(TrainEvent),
    Error(String),
}

pub struct TrainJob {
    pub control: Arc<TrainControl>,
    pub rx: Receiver<TrainMsg>,
}

pub fn spawn_training(ws: Workspace, config: TrainingConfig, resume: Option<PathBuf>) -> TrainJob {
    let control = Arc::new(TrainControl::new());
    let (tx, rx) = channel();
    let ctl = control.clone();
    thread::Builder::new()
        .name("glimmer-train".into())
        .spawn(move || {
            let mut opts = TrainOptions::new(config.clone());
            opts.resume = resume;
            if config.eval.sample_prompts {
                opts.sampler = Some(glimmer_evaluation::training_sampler(config.run.stage));
            }
            let etx = tx.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                train(&ws, opts, &ctl, &mut |e| {
                    let _ = etx.send(TrainMsg::Event(e));
                })
            }));
            match result {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    let _ = tx.send(TrainMsg::Error(format!("{e:#}")));
                }
                Err(_) => {
                    let _ = tx.send(TrainMsg::Error("training thread panicked".into()));
                }
            }
        })
        .expect("spawning training thread");
    TrainJob { control, rx }
}

// ── Chat ────────────────────────────────────────────────────────────────────────────────────

pub enum ChatCmd {
    Load { reference: String, backend: BackendKind },
    Send(String),
    Clear,
    Sampling(SamplingConfig),
    System(Option<String>),
    Save(PathBuf),
    Restore(PathBuf),
}

pub struct LoadedModel {
    pub name: String,
    pub backend: String,
    pub parameters: usize,
    pub context: usize,
    pub card: Option<ModelCard>,
}

pub enum ChatEvt {
    Loading(String),
    Loaded(Box<LoadedModel>),
    Token(String),
    Done { stats: GenerationStats, stop: StopReason, dropped: usize },
    Info(String),
    Error(String),
    Restored(Vec<Message>),
}

pub struct ChatWorker {
    pub tx: Sender<ChatCmd>,
    pub rx: Receiver<ChatEvt>,
    pub cancel: Arc<AtomicBool>,
}

pub fn spawn_chat(ws: Workspace) -> ChatWorker {
    let (ctx, crx) = channel::<ChatCmd>();
    let (etx, erx) = channel::<ChatEvt>();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel2 = cancel.clone();
    thread::Builder::new()
        .name("glimmer-chat".into())
        .spawn(move || {
            let mut engine: Option<Engine> = None;
            let mut session = ChatSession::new("stevie", None, SamplingConfig::default());
            while let Ok(cmd) = crx.recv() {
                match cmd {
                    ChatCmd::Load { reference, backend } => {
                        let _ = etx.send(ChatEvt::Loading(reference.clone()));
                        match Engine::load(&ws, &reference, backend) {
                            Ok(e) => {
                                use glimmer_models::burn::module::Module;
                                let sys = e.card.as_ref().and_then(|c| c.system_prompt.clone());
                                session.system_prompt = sys;
                                session.model = e.name.clone();
                                let _ = etx.send(ChatEvt::Loaded(Box::new(LoadedModel {
                                    name: e.name.clone(),
                                    backend: e.device.label(),
                                    parameters: e.model.num_params(),
                                    context: e.context_length(),
                                    card: e.card.clone(),
                                })));
                                engine = Some(e);
                            }
                            Err(err) => {
                                let _ = etx.send(ChatEvt::Error(format!("could not load '{reference}': {err:#}")));
                            }
                        }
                    }
                    ChatCmd::Send(text) => {
                        let Some(e) = engine.as_mut() else {
                            let _ = etx.send(ChatEvt::Error(
                                "no model loaded — train STEVIE or pick a model with /model".into(),
                            ));
                            continue;
                        };
                        cancel2.store(false, Ordering::SeqCst);
                        let tx2 = etx.clone();
                        let c = cancel2.clone();
                        match session.respond(e, &text, |piece| {
                            let _ = tx2.send(ChatEvt::Token(piece.to_string()));
                            !c.load(Ordering::SeqCst)
                        }) {
                            Ok((g, dropped)) => {
                                let _ = etx.send(ChatEvt::Done { stats: g.stats, stop: g.stop, dropped });
                            }
                            Err(err) => {
                                session.messages.pop();
                                let _ = etx.send(ChatEvt::Error(format!("{err:#}")));
                            }
                        }
                    }
                    ChatCmd::Clear => {
                        session.clear();
                        if let Some(e) = engine.as_mut() {
                            e.reset_cache();
                        }
                    }
                    ChatCmd::Sampling(s) => session.sampling = s,
                    ChatCmd::System(s) => session.system_prompt = s,
                    ChatCmd::Save(p) => match session.save(&p) {
                        Ok(()) => {
                            let _ = etx.send(ChatEvt::Info(format!("saved conversation to {}", p.display())));
                        }
                        Err(err) => {
                            let _ = etx.send(ChatEvt::Error(format!("{err:#}")));
                        }
                    },
                    ChatCmd::Restore(p) => match ChatSession::load(&p) {
                        Ok(s) => {
                            let msgs = s.messages.clone();
                            session.messages = s.messages;
                            if let Some(e) = engine.as_mut() {
                                e.reset_cache();
                            }
                            let _ = etx.send(ChatEvt::Restored(msgs));
                        }
                        Err(err) => {
                            let _ = etx.send(ChatEvt::Error(format!("{err:#}")));
                        }
                    },
                }
            }
        })
        .expect("spawning chat thread");
    ChatWorker { tx: ctx, rx: erx, cancel }
}

// ── System monitor ──────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct SystemSample {
    pub gpu: Option<gpu::GpuSample>,
    pub process_rss: Option<u64>,
    pub system_used: Option<u64>,
    pub system_total: Option<u64>,
}

pub fn spawn_monitor() -> Receiver<SystemSample> {
    let (tx, rx) = channel();
    thread::Builder::new()
        .name("glimmer-monitor".into())
        .spawn(move || {
            let mut pm = ProcessMemory::new();
            loop {
                let (used, total) = system_memory().map(|(u, t)| (Some(u), Some(t))).unwrap_or((None, None));
                let s = SystemSample {
                    gpu: gpu::sample(),
                    process_rss: pm.resident_bytes(),
                    system_used: used,
                    system_total: total,
                };
                if tx.send(s).is_err() {
                    break;
                }
                thread::sleep(Duration::from_secs(1));
            }
        })
        .expect("spawning monitor thread");
    rx
}

// ── Start-up probe ──────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub kind: BackendKind,
    pub result: Result<(String, Option<String>, Verification), String>,
}

pub struct Probe {
    pub hardware: HardwareReport,
    pub backends: Vec<BackendStatus>,
}

pub fn spawn_probe() -> Receiver<Probe> {
    let (tx, rx) = channel();
    thread::Builder::new()
        .name("glimmer-probe".into())
        .spawn(move || {
            let hardware = HardwareReport::detect(true);
            let backends = [BackendKind::Metal, BackendKind::Cpu]
                .into_iter()
                .map(|kind| BackendStatus {
                    kind,
                    result: backend::select(kind)
                        .map(|d| (d.adapter, d.target, d.verification))
                        .map_err(|e| e.to_string()),
                })
                .collect();
            let _ = tx.send(Probe { hardware, backends });
        })
        .expect("spawning probe thread");
    rx
}
