//! Application state and input handling.

use crate::workers::{self, ChatCmd, ChatEvt, ChatWorker, LoadedModel, Probe, SystemSample, TrainJob, TrainMsg};
use glimmer_core::config::TrainingConfig;
use glimmer_core::metrics::{EvalMetrics, EventKind, MetricRecord, SampleRecord, StepMetrics, read_metrics};
use glimmer_core::registry::{RegisteredModel, list_models};
use glimmer_core::{BackendKind, Workspace};
use glimmer_data::DatasetManifest;
use glimmer_evaluation::EvalReport;
use glimmer_inference::{GenerationStats, SamplingConfig, StopReason};
use glimmer_tokenizer::chat::Role;
use glimmer_training::{RunInfo, RunSummary, TrainEvent};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Home,
    Train,
    Chat,
    Models,
    Datasets,
    Experiments,
    Evaluation,
    Hardware,
    Settings,
}

impl Screen {
    pub const ALL: [Screen; 9] = [
        Screen::Home,
        Screen::Train,
        Screen::Chat,
        Screen::Models,
        Screen::Datasets,
        Screen::Experiments,
        Screen::Evaluation,
        Screen::Hardware,
        Screen::Settings,
    ];
    pub fn title(self) -> &'static str {
        match self {
            Screen::Home => "home",
            Screen::Train => "training lab",
            Screen::Chat => "chat",
            Screen::Models => "models",
            Screen::Datasets => "datasets",
            Screen::Experiments => "experiments",
            Screen::Evaluation => "evaluation",
            Screen::Hardware => "hardware",
            Screen::Settings => "settings",
        }
    }
    pub fn index(self) -> usize {
        Screen::ALL.iter().position(|s| *s == self).unwrap()
    }
}

pub const WELCOME: &str = "How ya going, mate? I'm Stevie. What's on your mind?";

#[derive(Debug, Clone)]
pub struct ChatLine {
    pub role: Option<Role>,
    pub text: String,
    pub meta: Option<String>,
}

pub struct ChatView {
    pub worker: ChatWorker,
    pub lines: Vec<ChatLine>,
    pub input: String,
    pub focus_input: bool,
    pub streaming: bool,
    pub model: Option<LoadedModel>,
    pub loading: Option<String>,
    pub last: Option<GenerationStats>,
    pub scroll: u16,
    pub sampling: SamplingConfig,
    pub requested_model: Option<String>,
}

#[derive(Default)]
pub struct TrainView {
    pub configs: Vec<(PathBuf, Result<TrainingConfig, String>)>,
    pub selected: usize,
    pub job: Option<TrainJob>,
    pub info: Option<RunInfo>,
    pub steps: Vec<StepMetrics>,
    pub evals: Vec<EvalMetrics>,
    pub log: VecDeque<(EventKind, String)>,
    pub samples: Vec<SampleRecord>,
    pub paused: bool,
    pub summary: Option<RunSummary>,
    pub error: Option<String>,
    pub show_config: bool,
    /// Following a run started by another process (read-only).
    pub watch: Option<Watch>,
}

/// Incrementally tails `metrics.jsonl` of a run in another process.
pub struct Watch {
    pub dir: PathBuf,
    offset: u64,
    last_poll: Instant,
}

pub struct RunEntry {
    pub name: String,
    pub dir: PathBuf,
}

pub struct RunCurve {
    pub name: String,
    pub train: Vec<(f64, f64)>,
    pub val: Vec<(f64, f64)>,
    pub lr: Vec<(f64, f64)>,
    pub last: Option<StepMetrics>,
    pub best_val: Option<f64>,
    pub events: Vec<String>,
}

pub struct App {
    pub ws: Workspace,
    pub screen: Screen,
    pub running: bool,
    pub tick: u64,
    pub train: TrainView,
    pub chat: ChatView,
    probe_rx: Receiver<Probe>,
    pub probe: Option<Probe>,
    mon_rx: Receiver<SystemSample>,
    pub sys: SystemSample,
    pub gpu_history: VecDeque<u64>,
    /// GPU-resident memory history in MiB.
    pub gpu_mem_history: VecDeque<u64>,
    pub models: Vec<RegisteredModel>,
    pub model_sel: usize,
    pub datasets: Vec<(String, anyhow::Result<DatasetManifest>)>,
    pub dataset_sel: usize,
    pub runs: Vec<RunEntry>,
    pub run_sel: usize,
    pub curve: Option<RunCurve>,
    pub studies: Vec<(String, String)>,
    pub reports: Vec<(String, EvalReport)>,
    pub report_sel: usize,
    pub report_scroll: u16,
    pub settings_sel: usize,
    pub backend: BackendKind,
    pub toast: Option<(String, Instant)>,
    pub confirm_quit: bool,
    pub home_sel: usize,
}

pub const HOME_ACTIONS: [(&str, &str, Screen); 5] = [
    ("Chat with STEVIE", "talk to the fine-tuned model", Screen::Chat),
    ("Open the training lab", "pick a config and train from scratch", Screen::Train),
    ("Browse experiments", "loss curves of every run and search", Screen::Experiments),
    ("Inspect evaluation", "before/after comparisons with real outputs", Screen::Evaluation),
    ("Check hardware", "verified backends and live GPU counters", Screen::Hardware),
];

impl App {
    pub fn new(ws: Workspace, initial: Screen, model: Option<String>, backend: BackendKind) -> Self {
        let chat_worker = workers::spawn_chat(ws.clone());
        let mut app = Self {
            screen: initial,
            running: true,
            tick: 0,
            train: TrainView::default(),
            chat: ChatView {
                worker: chat_worker,
                lines: vec![ChatLine {
                    role: Some(Role::Assistant),
                    text: WELCOME.into(),
                    meta: Some("scripted greeting".into()),
                }],
                input: String::new(),
                focus_input: true,
                streaming: false,
                model: None,
                loading: None,
                last: None,
                scroll: 0,
                sampling: SamplingConfig::default(),
                requested_model: None,
            },
            probe_rx: workers::spawn_probe(),
            probe: None,
            mon_rx: workers::spawn_monitor(),
            sys: SystemSample::default(),
            gpu_history: VecDeque::with_capacity(120),
            gpu_mem_history: VecDeque::with_capacity(120),
            models: Vec::new(),
            model_sel: 0,
            datasets: Vec::new(),
            dataset_sel: 0,
            runs: Vec::new(),
            run_sel: 0,
            curve: None,
            studies: Vec::new(),
            reports: Vec::new(),
            report_sel: 0,
            report_scroll: 0,
            settings_sel: 0,
            backend,
            toast: None,
            confirm_quit: false,
            home_sel: 0,
            ws,
        };
        app.refresh();
        if app.screen == Screen::Train {
            app.watch_active();
        }
        let target = model
            .or_else(|| app.ws.settings.default_model.clone())
            .filter(|m| app.models.iter().any(|r| &r.name == m))
            .or_else(|| app.models.first().map(|m| m.name.clone()));
        if let Some(m) = target {
            app.load_model(&m);
        }
        app
    }

    pub fn toast(&mut self, s: impl Into<String>) {
        self.toast = Some((s.into(), Instant::now()));
    }

    /// Re-scan configs, models, datasets, runs, studies and evaluation reports.
    pub fn refresh(&mut self) {
        let ws = &self.ws;
        let mut configs: Vec<(PathBuf, Result<TrainingConfig, String>)> =
            std::fs::read_dir(ws.configs_dir().join("training"))
                .map(|rd| {
                    rd.flatten()
                        .map(|e| e.path())
                        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
                        .map(|p| {
                            let c = TrainingConfig::load(&p).map_err(|e| format!("{e:#}"));
                            (p, c)
                        })
                        .collect()
                })
                .unwrap_or_default();
        configs.sort_by(|a, b| a.0.cmp(&b.0));
        self.train.configs = configs;
        self.models = list_models(&ws.models_dir());
        self.datasets = DatasetManifest::list(ws.datasets_dir());
        let mut runs: Vec<RunEntry> = std::fs::read_dir(ws.runs_dir())
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.join("metrics.jsonl").is_file())
                    .map(|p| RunEntry { name: p.file_name().unwrap().to_string_lossy().into_owned(), dir: p })
                    .collect()
            })
            .unwrap_or_default();
        runs.sort_by(|a, b| b.name.cmp(&a.name));
        self.runs = runs;
        self.studies = std::fs::read_dir(ws.experiments_dir())
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| {
                        let report = std::fs::read_to_string(e.path().join("report.md")).ok()?;
                        Some((e.file_name().to_string_lossy().into_owned(), report))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut reports: Vec<(String, EvalReport)> = std::fs::read_dir(ws.runs_dir().join("eval"))
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                    .filter_map(|e| {
                        let r: Vec<EvalReport> = serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok()?;
                        let stem = e.path().file_stem()?.to_string_lossy().into_owned();
                        Some(r.into_iter().map(move |rep| (stem.clone(), rep)))
                    })
                    .flatten()
                    .collect()
            })
            .unwrap_or_default();
        reports.sort_by(|a, b| a.0.cmp(&b.0));
        self.reports = reports;
        self.load_curve();
    }

    pub fn load_curve(&mut self) {
        let Some(run) = self.runs.get(self.run_sel) else {
            self.curve = None;
            return;
        };
        let recs = read_metrics(run.dir.join("metrics.jsonl")).unwrap_or_default();
        let mut c = RunCurve {
            name: run.name.clone(),
            train: vec![],
            val: vec![],
            lr: vec![],
            last: None,
            best_val: None,
            events: vec![],
        };
        for r in recs {
            match r {
                MetricRecord::Step(s) => {
                    c.train.push((s.step as f64, s.loss));
                    c.lr.push((s.step as f64, s.lr));
                    c.last = Some(s);
                }
                MetricRecord::Eval(e) => {
                    c.val.push((e.step as f64, e.val_loss));
                    c.best_val = Some(c.best_val.map_or(e.val_loss, |b: f64| b.min(e.val_loss)));
                }
                MetricRecord::Event(e) if matches!(e.kind, EventKind::Advice | EventKind::AutoAction) => {
                    c.events.push(format!("step {}: {}", e.step, e.message));
                }
                _ => {}
            }
        }
        self.curve = Some(c);
    }

    pub fn load_model(&mut self, name: &str) {
        self.chat.requested_model = Some(name.to_string());
        let _ = self.chat.worker.tx.send(ChatCmd::Load { reference: name.to_string(), backend: self.backend });
    }

    pub fn training_active(&self) -> bool {
        self.train.job.is_some() && self.train.summary.is_none() && self.train.error.is_none()
    }

    /// Drain worker channels. Called every frame.
    pub fn update(&mut self) {
        self.tick += 1;
        if let Ok(p) = self.probe_rx.try_recv() {
            self.probe = Some(p);
        }
        while let Ok(s) = self.mon_rx.try_recv() {
            if let Some(u) = s.gpu.and_then(|g| g.device_utilization) {
                if self.gpu_history.len() == 120 {
                    self.gpu_history.pop_front();
                }
                self.gpu_history.push_back(u as u64);
            }
            if let Some(m) = s.gpu.and_then(|g| g.in_use_memory) {
                if self.gpu_mem_history.len() == 120 {
                    self.gpu_mem_history.pop_front();
                }
                self.gpu_mem_history.push_back(m >> 20);
            }
            self.sys = s;
        }
        // Training events.
        let mut finished = false;
        if let Some(job) = &self.train.job {
            let mut n = 0;
            while let Ok(msg) = job.rx.try_recv() {
                n += 1;
                match msg {
                    TrainMsg::Event(e) => match e {
                        TrainEvent::Started(info) => {
                            self.train.info = Some(*info);
                            self.train.steps.clear();
                            self.train.evals.clear();
                            self.train.samples.clear();
                        }
                        TrainEvent::Step(s) => self.train.steps.push(s),
                        TrainEvent::Eval(e) => self.train.evals.push(e),
                        TrainEvent::Sample(s) => {
                            self.train.samples.push(s);
                            if self.train.samples.len() > 12 {
                                self.train.samples.remove(0);
                            }
                        }
                        TrainEvent::Event(ev) => {
                            self.train.log.push_back((ev.kind, format!("step {} · {}", ev.step, ev.message)));
                            if self.train.log.len() > 200 {
                                self.train.log.pop_front();
                            }
                        }
                        TrainEvent::Paused(p) => self.train.paused = p,
                        TrainEvent::Finished(s) => {
                            self.train.summary = Some(*s);
                            finished = true;
                        }
                    },
                    TrainMsg::Error(e) => {
                        self.train.error = Some(e);
                        finished = true;
                    }
                }
                if n > 2000 {
                    break;
                }
            }
        }
        if finished {
            self.toast("training finished");
            self.refresh();
        }
        self.poll_watch();
        // Chat events.
        while let Ok(evt) = self.chat.worker.rx.try_recv() {
            match evt {
                ChatEvt::Loading(m) => self.chat.loading = Some(m),
                ChatEvt::Loaded(m) => {
                    self.chat.loading = None;
                    self.chat.lines.push(ChatLine {
                        role: None,
                        text: format!(
                            "loaded {} · {} params · {}",
                            m.name,
                            glimmer_core::time::format_count(m.parameters as u64),
                            m.backend
                        ),
                        meta: None,
                    });
                    self.chat.model = Some(*m);
                }
                ChatEvt::Token(t) => {
                    if let Some(last) = self.chat.lines.last_mut() {
                        last.text.push_str(&t);
                    }
                    self.chat.scroll = 0;
                }
                ChatEvt::Done { stats, stop, dropped } => {
                    self.chat.streaming = false;
                    if let Some(last) = self.chat.lines.last_mut() {
                        last.text = last.text.trim().to_string();
                        let mut meta = format!(
                            "{} tok · {:.0} tok/s · {:.1}s",
                            stats.generated_tokens,
                            stats.decode_tokens_per_sec,
                            stats.total_ms / 1e3
                        );
                        if stop == StopReason::MaxTokens {
                            meta.push_str(" · hit max tokens");
                        }
                        if stop == StopReason::Cancelled {
                            meta.push_str(" · stopped");
                        }
                        if dropped > 0 {
                            meta.push_str(&format!(" · {dropped} old messages trimmed"));
                        }
                        last.meta = Some(meta);
                        if last.text.is_empty() {
                            last.text = "…".into();
                        }
                    }
                    self.chat.last = Some(stats);
                }
                ChatEvt::Info(s) => self.chat.lines.push(ChatLine { role: None, text: s, meta: None }),
                ChatEvt::Error(e) => {
                    self.chat.loading = None;
                    if self.chat.streaming {
                        self.chat.streaming = false;
                        if self.chat.lines.last().is_some_and(|l| l.role == Some(Role::Assistant) && l.text.is_empty())
                        {
                            self.chat.lines.pop();
                        }
                    }
                    self.chat.lines.push(ChatLine { role: None, text: format!("⚠ {e}"), meta: None });
                }
                ChatEvt::Restored(msgs) => {
                    self.chat.lines.retain(|l| l.role.is_none());
                    for m in msgs {
                        self.chat.lines.push(ChatLine { role: Some(m.role), text: m.content, meta: None });
                    }
                }
            }
        }
        if self.toast.as_ref().is_some_and(|(_, t)| t.elapsed().as_secs() > 4) {
            self.toast = None;
        }
    }

    /// Follow the most recent active run started elsewhere (e.g. `glimmer train` in a shell).
    pub fn watch_active(&mut self) -> bool {
        if self.train.job.is_some() {
            return false;
        }
        let Some(dir) = active_run(&self.runs) else { return false };
        let Some(info) = reconstruct_info(&dir) else { return false };
        self.train = TrainView {
            configs: std::mem::take(&mut self.train.configs),
            selected: self.train.selected,
            ..Default::default()
        };
        self.train.log.push_back((
            EventKind::Info,
            format!("watching {} (started in another process; controls disabled)", info.run_id),
        ));
        self.train.info = Some(info);
        self.train.watch =
            Some(Watch { dir, offset: 0, last_poll: Instant::now() - std::time::Duration::from_secs(5) });
        self.poll_watch();
        true
    }

    fn poll_watch(&mut self) {
        use std::io::{Read, Seek, SeekFrom};
        let Some(w) = self.train.watch.as_mut() else { return };
        if w.last_poll.elapsed().as_millis() < 500 {
            return;
        }
        w.last_poll = Instant::now();
        let path = w.dir.join("metrics.jsonl");
        let Ok(mut f) = std::fs::File::open(&path) else { return };
        if f.seek(SeekFrom::Start(w.offset)).is_err() {
            return;
        }
        let mut buf = String::new();
        if f.read_to_string(&mut buf).is_err() {
            return;
        }
        // Only consume complete lines; a partially written last line is read next time.
        let complete = buf.rfind('\n').map(|i| i + 1).unwrap_or(0);
        w.offset += complete as u64;
        let finished = w.dir.join("summary.json").exists();
        for line in buf[..complete].lines() {
            match serde_json::from_str::<MetricRecord>(line) {
                Ok(MetricRecord::Step(s)) => self.train.steps.push(s),
                Ok(MetricRecord::Eval(e)) => self.train.evals.push(e),
                Ok(MetricRecord::Sample(s)) => {
                    self.train.samples.push(s);
                    if self.train.samples.len() > 12 {
                        self.train.samples.remove(0);
                    }
                }
                Ok(MetricRecord::Event(e)) => {
                    self.train.log.push_back((e.kind, format!("step {} · {}", e.step, e.message)));
                    if self.train.log.len() > 200 {
                        self.train.log.pop_front();
                    }
                }
                Err(_) => {}
            }
        }
        if finished && self.train.summary.is_none() {
            let dir = self.train.watch.as_ref().map(|w| w.dir.clone()).unwrap();
            self.train.summary =
                std::fs::read_to_string(dir.join("summary.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
        }
    }

    fn start_training(&mut self, resume: bool) {
        if self.training_active() {
            self.toast("a training run is already in progress");
            return;
        }
        let Some((path, cfg)) = self.train.configs.get(self.train.selected) else { return };
        let path = path.clone();
        let cfg = match cfg {
            Ok(c) => c.clone(),
            Err(e) => {
                self.toast(format!("invalid config: {e}"));
                return;
            }
        };
        let resume_dir = if resume {
            let prefix = format!("{}-", cfg.run.name);
            let mut runs: Vec<PathBuf> =
                self.runs.iter().filter(|r| r.name.starts_with(&prefix)).map(|r| r.dir.clone()).collect();
            runs.sort();
            match runs.last().and_then(|r| glimmer_training::checkpoint::latest(r)) {
                Some(c) => Some(c),
                None => {
                    self.toast("no checkpoint to resume for this config");
                    return;
                }
            }
        } else {
            None
        };
        self.train = TrainView {
            configs: std::mem::take(&mut self.train.configs),
            selected: self.train.selected,
            ..Default::default()
        };
        self.train.log.push_back((EventKind::Info, format!("starting {}", self.ws.relative(&path))));
        self.train.job = Some(workers::spawn_training(self.ws.clone(), cfg, resume_dir));
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && key.code == KeyCode::Char('c') {
            self.request_quit();
            return;
        }
        if key.code == KeyCode::Tab {
            self.screen = Screen::ALL[(self.screen.index() + 1) % Screen::ALL.len()];
            return;
        }
        if key.code == KeyCode::BackTab {
            self.screen = Screen::ALL[(self.screen.index() + Screen::ALL.len() - 1) % Screen::ALL.len()];
            return;
        }
        let typing = self.screen == Screen::Chat && self.chat.focus_input;
        if !typing && let KeyCode::Char(c) = key.code {
            if let Some(d) = c.to_digit(10)
                && (1..=9).contains(&d)
            {
                self.screen = Screen::ALL[d as usize - 1];
                if self.screen == Screen::Chat {
                    self.chat.focus_input = true;
                }
                return;
            }
            if c == 'q' {
                self.request_quit();
                return;
            }
            if c == 'R' {
                self.refresh();
                self.toast("refreshed");
                return;
            }
        }
        if key.code != KeyCode::Char('q') {
            self.confirm_quit = false;
        }
        match self.screen {
            Screen::Home => self.key_home(key),
            Screen::Train => self.key_train(key),
            Screen::Chat => self.key_chat(key),
            Screen::Models => self.key_models(key),
            Screen::Datasets => list_nav(key, &mut self.dataset_sel, self.datasets.len()),
            Screen::Experiments => {
                let before = self.run_sel;
                list_nav(key, &mut self.run_sel, self.runs.len());
                if before != self.run_sel {
                    self.load_curve();
                }
            }
            Screen::Evaluation => self.key_eval(key),
            Screen::Hardware => {}
            Screen::Settings => self.key_settings(key),
        }
    }

    fn request_quit(&mut self) {
        if self.training_active() && !self.confirm_quit {
            self.confirm_quit = true;
            self.toast("training is running — press q again to stop it gracefully and quit");
            return;
        }
        if let Some(job) = &self.train.job {
            job.control.stop();
        }
        self.chat.worker.cancel.store(true, Ordering::SeqCst);
        self.running = false;
    }

    fn key_home(&mut self, key: KeyEvent) {
        list_nav(key, &mut self.home_sel, HOME_ACTIONS.len());
        if key.code == KeyCode::Enter {
            self.screen = HOME_ACTIONS[self.home_sel].2;
        }
    }

    fn key_train(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Down | KeyCode::Char('k') | KeyCode::Char('j') if !self.training_active() => {
                list_nav(key, &mut self.train.selected, self.train.configs.len())
            }
            KeyCode::Enter => self.start_training(false),
            KeyCode::Char('r') => self.start_training(true),
            KeyCode::Char('p') | KeyCode::Char(' ') => {
                if let Some(job) = &self.train.job {
                    let paused = job.control.toggle_pause();
                    self.toast(if paused { "pausing after this step…" } else { "resuming" });
                }
            }
            KeyCode::Char('s') => {
                if let Some(job) = &self.train.job {
                    job.control.request_checkpoint();
                    self.toast("checkpoint requested");
                }
            }
            KeyCode::Char('x') => {
                if let Some(job) = &self.train.job {
                    job.control.stop();
                    self.toast("stopping gracefully — a checkpoint will be saved");
                }
            }
            KeyCode::Char('c') => self.train.show_config = !self.train.show_config,
            KeyCode::Char('w') if !self.watch_active() => {
                self.toast("no active run to watch (runs/ with metrics written in the last 2 minutes)");
            }
            _ => {}
        }
    }

    fn key_models(&mut self, key: KeyEvent) {
        list_nav(key, &mut self.model_sel, self.models.len());
        if key.code == KeyCode::Enter
            && let Some(m) = self.models.get(self.model_sel)
        {
            let name = m.name.clone();
            self.load_model(&name);
            self.screen = Screen::Chat;
            self.chat.focus_input = true;
        }
    }

    fn key_eval(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::PageDown => self.report_scroll = self.report_scroll.saturating_add(10),
            KeyCode::PageUp => self.report_scroll = self.report_scroll.saturating_sub(10),
            _ => {
                let before = self.report_sel;
                list_nav(key, &mut self.report_sel, self.reports.len());
                if before != self.report_sel {
                    self.report_scroll = 0;
                }
            }
        }
    }

    pub const SETTINGS: [&'static str; 6] =
        ["temperature", "top-k", "top-p", "repetition penalty", "max new tokens", "backend"];

    fn key_settings(&mut self, key: KeyEvent) {
        list_nav(key, &mut self.settings_sel, Self::SETTINGS.len());
        let delta = match key.code {
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('+') => 1.0,
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('-') => -1.0,
            _ => return,
        };
        let s = &mut self.chat.sampling;
        match self.settings_sel {
            0 => s.temperature = (s.temperature + 0.05 * delta as f32).clamp(0.0, 2.0),
            1 => s.top_k = (s.top_k as i64 + 5 * delta as i64).clamp(0, 500) as usize,
            2 => s.top_p = (s.top_p + 0.02 * delta as f32).clamp(0.05, 1.0),
            3 => s.repetition_penalty = (s.repetition_penalty + 0.05 * delta as f32).clamp(1.0, 2.0),
            4 => s.max_new_tokens = (s.max_new_tokens as i64 + 16 * delta as i64).clamp(16, 400) as usize,
            5 => {
                self.backend = match (self.backend, delta > 0.0) {
                    (BackendKind::Auto, true) => BackendKind::Metal,
                    (BackendKind::Metal, true) => BackendKind::Cpu,
                    (BackendKind::Cpu, true) => BackendKind::Auto,
                    (BackendKind::Auto, false) => BackendKind::Cpu,
                    (BackendKind::Cpu, false) => BackendKind::Metal,
                    (BackendKind::Metal, false) => BackendKind::Auto,
                };
                if let Some(m) = self.chat.model.as_ref().map(|m| m.name.clone()) {
                    self.load_model(&m);
                }
                return;
            }
            _ => {}
        }
        let _ = self.chat.worker.tx.send(ChatCmd::Sampling(self.chat.sampling.clone()));
    }

    fn key_chat(&mut self, key: KeyEvent) {
        if !self.chat.focus_input {
            match key.code {
                KeyCode::Enter | KeyCode::Char('i') => self.chat.focus_input = true,
                KeyCode::Up | KeyCode::Char('k') => self.chat.scroll = self.chat.scroll.saturating_add(1),
                KeyCode::Down | KeyCode::Char('j') => self.chat.scroll = self.chat.scroll.saturating_sub(1),
                KeyCode::PageUp => self.chat.scroll = self.chat.scroll.saturating_add(10),
                KeyCode::PageDown => self.chat.scroll = self.chat.scroll.saturating_sub(10),
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Esc => {
                if self.chat.streaming {
                    self.chat.worker.cancel.store(true, Ordering::SeqCst);
                } else {
                    self.chat.focus_input = false;
                }
            }
            KeyCode::PageUp => self.chat.scroll = self.chat.scroll.saturating_add(10),
            KeyCode::PageDown => self.chat.scroll = self.chat.scroll.saturating_sub(10),
            KeyCode::Backspace => {
                self.chat.input.pop();
            }
            KeyCode::Enter => self.submit_chat(),
            KeyCode::Char(c) => self.chat.input.push(c),
            _ => {}
        }
    }

    fn system(&mut self, s: impl Into<String>) {
        self.chat.lines.push(ChatLine { role: None, text: s.into(), meta: None });
    }

    fn submit_chat(&mut self) {
        let text = self.chat.input.trim().to_string();
        if text.is_empty() || self.chat.streaming {
            return;
        }
        self.chat.input.clear();
        self.chat.scroll = 0;
        if let Some(cmd) = text.strip_prefix('/') {
            self.command(cmd);
            return;
        }
        if self.chat.model.is_none() {
            self.system("no model loaded yet — train STEVIE first, or pick one with /model <name>");
            return;
        }
        self.chat.lines.push(ChatLine { role: Some(Role::User), text: text.clone(), meta: None });
        self.chat.lines.push(ChatLine { role: Some(Role::Assistant), text: String::new(), meta: None });
        self.chat.streaming = true;
        let _ = self.chat.worker.tx.send(ChatCmd::Send(text));
    }

    fn command(&mut self, cmd: &str) {
        let mut parts = cmd.splitn(2, ' ');
        let name = parts.next().unwrap_or("");
        let arg = parts.next().map(str::trim).filter(|s| !s.is_empty());
        let conv_dir = self.ws.conversations_dir();
        match name {
            "help" => self.system(
                "commands: /clear · /save [name] · /load [name] · /model [name] · /stats · /temperature <t> · /topk <k> · /topp <p> · /system [text|off] · /quit — Esc to stop a reply or leave the input, PgUp/PgDn to scroll",
            ),
            "clear" => {
                let _ = self.chat.worker.tx.send(ChatCmd::Clear);
                self.chat.lines = vec![ChatLine { role: Some(Role::Assistant), text: WELCOME.into(), meta: Some("scripted greeting".into()) }];
            }
            "save" => {
                let n = arg.map(str::to_string).unwrap_or_else(|| format!("chat-{}", glimmer_core::time::compact_stamp(glimmer_core::time::now_unix())));
                let _ = self.chat.worker.tx.send(ChatCmd::Save(conv_dir.join(format!("{n}.json"))));
            }
            "load" => match arg {
                Some(n) => {
                    let _ = self.chat.worker.tx.send(ChatCmd::Restore(conv_dir.join(format!("{}.json", n.trim_end_matches(".json")))));
                }
                None => {
                    let saved: Vec<String> = std::fs::read_dir(&conv_dir)
                        .map(|rd| rd.flatten().filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned())).collect())
                        .unwrap_or_default();
                    self.system(if saved.is_empty() { "no saved conversations".to_string() } else { format!("saved: {}", saved.join(", ")) });
                }
            },
            "model" => match arg {
                Some(m) => self.load_model(m),
                None => {
                    let names: Vec<String> = self.models.iter().map(|m| m.name.clone()).collect();
                    let current = self.chat.model.as_ref().map(|m| m.name.clone()).unwrap_or_else(|| "none".into());
                    self.system(format!("current: {current} · available: {}", if names.is_empty() { "none".into() } else { names.join(", ") }));
                }
            },
            "stats" => {
                let s = match &self.chat.last {
                    Some(g) => format!(
                        "last reply: {} tokens in {:.2}s ({:.1} tok/s decode), prompt {} tokens ({} reused from cache), prefill {:.0} ms, context {}/{}",
                        g.generated_tokens, g.total_ms / 1e3, g.decode_tokens_per_sec, g.prompt_tokens, g.reused_tokens, g.prefill_ms, g.context_used, g.context_length
                    ),
                    None => "no replies yet".into(),
                };
                self.system(s);
            }
            "temperature" | "temp" => self.set_sampling(arg, |s, v| s.temperature = v.clamp(0.0, 2.0) as f32),
            "topk" => self.set_sampling(arg, |s, v| s.top_k = v.max(0.0) as usize),
            "topp" => self.set_sampling(arg, |s, v| s.top_p = v.clamp(0.05, 1.0) as f32),
            "system" => {
                let sys = match arg {
                    Some("off") => None,
                    Some(t) => Some(t.to_string()),
                    None => Some(glimmer_data::stevie::SYSTEM_PROMPT.to_string()),
                };
                self.system(format!("system prompt: {}", sys.clone().unwrap_or_else(|| "off".into())));
                let _ = self.chat.worker.tx.send(ChatCmd::System(sys));
            }
            "quit" | "exit" => self.request_quit(),
            other => self.system(format!("unknown command /{other} — try /help")),
        }
    }

    fn set_sampling(&mut self, arg: Option<&str>, f: impl Fn(&mut SamplingConfig, f64)) {
        match arg.and_then(|a| a.parse::<f64>().ok()) {
            Some(v) => {
                f(&mut self.chat.sampling, v);
                let s = self.chat.sampling.clone();
                self.system(format!(
                    "sampling: temperature {:.2} · top-k {} · top-p {:.2}",
                    s.temperature, s.top_k, s.top_p
                ));
                let _ = self.chat.worker.tx.send(ChatCmd::Sampling(s));
            }
            None => self.system("expected a number"),
        }
    }
}

/// A run is active if its metrics were written in the last two minutes and it has no summary.
pub fn active_run(runs: &[RunEntry]) -> Option<PathBuf> {
    runs.iter()
        .filter(|r| !r.dir.join("summary.json").exists())
        .filter_map(|r| {
            let m = std::fs::metadata(r.dir.join("metrics.jsonl")).ok()?.modified().ok()?;
            (m.elapsed().ok()?.as_secs() < 120).then(|| (m, r.dir.clone()))
        })
        .max_by_key(|(m, _)| *m)
        .map(|(_, d)| d)
}

fn reconstruct_info(dir: &std::path::Path) -> Option<RunInfo> {
    if let Some(i) = std::fs::read_to_string(dir.join("run_info.json")).ok().and_then(|t| serde_json::from_str(&t).ok())
    {
        return Some(i);
    }
    // Runs started before run_info.json existed: rebuild from the saved configs.
    let config = TrainingConfig::load(dir.join("training.toml")).ok()?;
    let model: glimmer_core::ModelConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join("model.json")).ok()?).ok()?;
    let first_info = read_metrics(dir.join("metrics.jsonl")).ok()?.into_iter().find_map(|r| match r {
        MetricRecord::Event(e) if e.kind == EventKind::Info => Some(e.message),
        _ => None,
    });
    let backend_label = first_info
        .as_deref()
        .map(|m| m.split(" · ").skip(1).take(2).collect::<Vec<_>>().join(" · "))
        .unwrap_or_else(|| "unknown".into());
    let backend = if backend_label.starts_with("Metal") { BackendKind::Metal } else { BackendKind::Cpu };
    Some(RunInfo {
        run_id: dir.file_name()?.to_string_lossy().into_owned(),
        run_dir: dir.to_path_buf(),
        stage: config.run.stage,
        parameters: model.param_count(),
        backend,
        backend_label,
        fallback_reason: None,
        dataset: config.data.dataset.clone(),
        train_tokens: 0,
        start_step: 0,
        max_steps: config.schedule.max_steps,
        tokens_per_step: config.tokens_per_step(model.context_length),
        resumed_from: None,
        initialised_from: config.init.from_checkpoint.clone(),
        model,
        config,
    })
}

pub fn list_nav(key: KeyEvent, sel: &mut usize, len: usize) {
    if len == 0 {
        return;
    }
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => *sel = (*sel + len - 1) % len,
        KeyCode::Down | KeyCode::Char('j') => *sel = (*sel + 1) % len,
        KeyCode::Home => *sel = 0,
        KeyCode::End => *sel = len - 1,
        _ => {}
    }
}
