//! Rendering. Every screen is a pure function of [`App`] state.

use crate::app::{App, HOME_ACTIONS, Screen};
use crate::theme::{self, *};
use glimmer_core::metrics::EventKind;
use glimmer_core::time::{format_bytes, format_count, format_duration};
use glimmer_tokenizer::chat::Role;
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Chart, Clear, Dataset, Gauge, GraphType, List, ListItem, ListState, Padding,
    Paragraph, Row, Sparkline, Table, Wrap,
};
use unicode_width::UnicodeWidthStr;

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    f.render_widget(Block::new().style(theme::base()), area);
    let [header, body, status] =
        Layout::vertical([Constraint::Length(2), Constraint::Min(5), Constraint::Length(1)]).areas(area);
    draw_header(f, header, app);
    match app.screen {
        Screen::Home => home(f, body, app),
        Screen::Train => train(f, body, app),
        Screen::Chat => chat(f, body, app),
        Screen::Models => models(f, body, app),
        Screen::Datasets => datasets(f, body, app),
        Screen::Experiments => experiments(f, body, app),
        Screen::Evaluation => evaluation(f, body, app),
        Screen::Hardware => hardware(f, body, app),
        Screen::Settings => settings(f, body, app),
    }
    draw_status(f, status, app);
}

fn panel<'a>(title: &'a str, focused: bool) -> Block<'a> {
    Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border(focused))
        .title(Line::from(vec![
            Span::raw(" "),
            Span::styled(title, if focused { theme::title() } else { theme::muted() }),
            Span::raw(" "),
        ]))
        .padding(Padding::horizontal(1))
        .style(Style::new().bg(BG))
}

fn kv<'a>(k: &'a str, v: impl Into<String>) -> Line<'a> {
    Line::from(vec![Span::styled(format!("{k:<15}"), theme::muted()), Span::styled(v.into(), Style::new().fg(TEXT))])
}

fn kv_styled<'a>(k: &'a str, v: impl Into<String>, s: Style) -> Line<'a> {
    Line::from(vec![Span::styled(format!("{k:<15}"), theme::muted()), Span::styled(v.into(), s)])
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let mut spans =
        vec![Span::styled(" ✦ ", accent()), Span::styled("GLIMMER", theme::title()), Span::styled("   ", faint())];
    for (i, s) in Screen::ALL.iter().enumerate() {
        let label = format!(" {} {} ", i + 1, s.title());
        if *s == app.screen {
            spans.push(Span::styled(label, theme::selected()));
        } else {
            spans.push(Span::styled(format!(" {}", i + 1), faint()));
            spans.push(Span::styled(format!(" {} ", s.title()), theme::muted()));
        }
    }
    let right = backend_badge(app);
    let [l, r] = Layout::horizontal([Constraint::Min(10), Constraint::Length(right.width() as u16 + 1)])
        .areas(Rect { height: 1, ..area });
    f.render_widget(Paragraph::new(Line::from(spans)), l);
    f.render_widget(Paragraph::new(right).alignment(Alignment::Right), r);
    let rule = "─".repeat(area.width as usize);
    f.render_widget(Paragraph::new(Span::styled(rule, faint())), Rect { y: area.y + 1, height: 1, ..area });
}

fn backend_badge(app: &App) -> Line<'static> {
    match &app.probe {
        None => Line::from(Span::styled("probing hardware… ", faint())),
        Some(p) => {
            let metal = p.backends.iter().find(|b| b.kind == glimmer_core::BackendKind::Metal);
            match metal.map(|m| &m.result) {
                Some(Ok((adapter, _, _))) => Line::from(vec![
                    Span::styled("● ", Style::new().fg(GUM)),
                    Span::styled(format!("Metal · {adapter} "), theme::muted()),
                ]),
                _ => Line::from(vec![
                    Span::styled("● ", Style::new().fg(OCHRE)),
                    Span::styled("CPU only ", theme::muted()),
                ]),
            }
        }
    }
}

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let hints = match app.screen {
        Screen::Home => "↑↓ select · enter open · tab/1-9 switch · q quit",
        Screen::Train if app.train.watch.is_some() => {
            "watching a run from another process · enter starts a new run · c config"
        }
        Screen::Train => {
            "↑↓ config · enter start · r resume · w watch active run · p pause · s checkpoint · x stop · c config"
        }
        Screen::Chat if app.chat.focus_input => "enter send · /help commands · esc stop/leave input · pgup/pgdn scroll",
        Screen::Chat => "enter/i type · ↑↓ scroll · tab/1-9 switch · q quit",
        Screen::Models => "↑↓ select · enter chat with model · R refresh",
        Screen::Experiments => "↑↓ select run · R refresh",
        Screen::Evaluation => "↑↓ report · pgup/pgdn scroll",
        Screen::Settings => "↑↓ select · ←→ adjust",
        _ => "tab/1-9 switch · R refresh · q quit",
    };
    let mut right: Vec<Span> = Vec::new();
    if let (Some(info), Some(s)) = (&app.train.info, app.train.steps.last())
        && app.training_active()
    {
        right.push(Span::styled(
            format!("▸ {} {}/{} loss {:.3}  ", info.stage.as_str(), s.step, info.max_steps, s.loss),
            accent(),
        ));
    }
    if let Some(u) = app.sys.gpu.and_then(|g| g.device_utilization) {
        right.push(Span::styled(format!("GPU {u:.0}%  "), theme::muted()));
    }
    if let (Some(used), Some(total)) = (app.sys.system_used, app.sys.system_total) {
        right.push(Span::styled(format!("mem {}/{}  ", format_bytes(used), format_bytes(total)), theme::muted()));
    }
    let left = match &app.toast {
        Some((t, _)) => Span::styled(format!(" {t}"), Style::new().fg(GOLD)),
        None => Span::styled(format!(" {hints}"), faint()),
    };
    let rw: u16 = right.iter().map(|s| s.width() as u16).sum();
    let [l, r] = Layout::horizontal([Constraint::Min(10), Constraint::Length(rw)]).areas(area);
    f.render_widget(Paragraph::new(Line::from(left)).style(Style::new().bg(SURFACE)), l);
    f.render_widget(Paragraph::new(Line::from(right)).style(Style::new().bg(SURFACE)), r);
}

// ── Home ────────────────────────────────────────────────────────────────────────────────────

fn home(f: &mut Frame, area: Rect, app: &App) {
    let [top, bottom] = Layout::vertical([Constraint::Length(10), Constraint::Min(6)]).areas(area);
    let mut logo: Vec<Line> = vec![Line::raw("")];
    for (i, row) in LOGO.iter().enumerate() {
        logo.push(Line::from(Span::styled(*row, Style::new().fg(logo_color(i)))).alignment(Alignment::Center));
    }
    logo.push(Line::raw(""));
    logo.push(
        Line::from(vec![Span::styled(
            "a Rust-native laboratory for training small language models from scratch",
            theme::muted(),
        )])
        .alignment(Alignment::Center),
    );
    f.render_widget(Paragraph::new(logo), top);

    let [left, right] = Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(bottom);
    let items: Vec<ListItem> = HOME_ACTIONS
        .iter()
        .map(|(t, d, _)| {
            ListItem::new(Line::from(vec![Span::styled(format!("{t:<26}"), bold()), Span::styled(*d, theme::muted())]))
        })
        .collect();
    let mut st = ListState::default().with_selected(Some(app.home_sel));
    f.render_stateful_widget(
        List::new(items)
            .block(panel("start here", true))
            .highlight_style(Style::new().bg(RAISED).fg(GOLD))
            .highlight_symbol("▸ "),
        left,
        &mut st,
    );

    let mut lines = vec![];
    let stevie = app.models.iter().find(|m| m.name == "stevie");
    match stevie.and_then(|m| m.card.as_ref()) {
        Some(card) => {
            lines.push(Line::from(vec![
                Span::styled("❦ STEVIE ", theme::stevie().add_modifier(Modifier::BOLD)),
                Span::styled("a little Aussie AI mate", theme::muted()),
            ]));
            lines.push(kv("parameters", format_count(card.parameters as u64)));
            lines.push(kv("trained on", format!("{} tokens", format_count(card.total_tokens()))));
            lines.push(kv("steps", format_count(card.total_steps() as u64)));
            lines.push(kv("stages", card.stages.iter().map(|s| s.stage.clone()).collect::<Vec<_>>().join(" → ")));
        }
        None => {
            lines.push(Line::from(Span::styled("❦ STEVIE has not been trained yet", theme::stevie())));
            lines.push(Line::from(Span::styled(
                "open the training lab and run stevie-pretrain, then stevie-sft",
                theme::muted(),
            )));
        }
    }
    lines.push(Line::raw(""));
    lines.push(kv("models", app.models.len().to_string()));
    lines.push(kv("datasets", app.datasets.len().to_string()));
    lines.push(kv("training runs", app.runs.len().to_string()));
    lines.push(kv("studies", app.studies.len().to_string()));
    f.render_widget(Paragraph::new(lines).block(panel("laboratory", false)), right);
}

// ── Charts ──────────────────────────────────────────────────────────────────────────────────

fn downsample(points: &[(f64, f64)], max: usize) -> Vec<(f64, f64)> {
    if points.len() <= max {
        return points.to_vec();
    }
    let bucket = points.len().div_ceil(max);
    points.chunks(bucket).map(|c| (c[c.len() - 1].0, c.iter().map(|p| p.1).sum::<f64>() / c.len() as f64)).collect()
}

fn ema(points: &[(f64, f64)], alpha: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(points.len());
    let mut v: Option<f64> = None;
    for &(x, y) in points {
        let n = v.map_or(y, |p| alpha * y + (1.0 - alpha) * p);
        v = Some(n);
        out.push((x, n));
    }
    out
}

struct Series<'a> {
    name: &'a str,
    color: ratatui::style::Color,
    points: Vec<(f64, f64)>,
    scatter: bool,
}

fn chart(f: &mut Frame, area: Rect, title: &str, series: Vec<Series>, fmt: fn(f64) -> String, log_hint: bool) {
    let all: Vec<&(f64, f64)> = series.iter().flat_map(|s| s.points.iter()).filter(|p| p.1.is_finite()).collect();
    let block = panel(title, false);
    if all.len() < 2 {
        f.render_widget(Paragraph::new(Span::styled("waiting for data…", faint())).block(block), area);
        return;
    }
    let (xmin, xmax) = all.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let (mut ymin, mut ymax) = all.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.1), a.1.max(p.1)));
    if (ymax - ymin).abs() < 1e-12 {
        ymax += 1.0;
        ymin -= 1.0;
    }
    let pad = (ymax - ymin) * 0.06;
    let (ymin, ymax) = (if log_hint { ymin.max(0.0) } else { ymin - pad }, ymax + pad);
    let datasets: Vec<Dataset> = series
        .iter()
        .map(|s| {
            Dataset::default()
                .name(s.name.to_string())
                .marker(if s.scatter { Marker::Dot } else { Marker::Braille })
                .graph_type(if s.scatter { GraphType::Scatter } else { GraphType::Line })
                .style(Style::new().fg(s.color))
                .data(&s.points)
        })
        .collect();
    let xl = vec![Span::styled(format!("{xmin:.0}"), faint()), Span::styled(format!("{xmax:.0}"), faint())];
    let yl = vec![
        Span::styled(fmt(ymin), faint()),
        Span::styled(fmt((ymin + ymax) / 2.0), faint()),
        Span::styled(fmt(ymax), faint()),
    ];
    let c = Chart::new(datasets)
        .style(Style::new().bg(BG).fg(MUTED))
        .block(block)
        .x_axis(Axis::default().bounds([xmin, xmax.max(xmin + 1.0)]).labels(xl).style(faint()))
        .y_axis(Axis::default().bounds([ymin, ymax]).labels(yl).style(faint()))
        .legend_position(Some(ratatui::widgets::LegendPosition::TopRight))
        .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)));
    f.render_widget(c, area);
}

fn f3(v: f64) -> String {
    format!("{v:.2}")
}
fn sci(v: f64) -> String {
    format!("{v:.1e}")
}
fn kilo(v: f64) -> String {
    format!("{:.1}k", v / 1e3)
}

// ── Training lab ────────────────────────────────────────────────────────────────────────────

fn train(f: &mut Frame, area: Rect, app: &App) {
    let t = &app.train;
    let [left, right] = Layout::horizontal([Constraint::Length(46), Constraint::Min(30)]).areas(area);
    let [configs, stats] =
        Layout::vertical([Constraint::Length((t.configs.len() as u16 + 2).clamp(4, 10)), Constraint::Min(8)])
            .areas(left);

    let items: Vec<ListItem> = t
        .configs
        .iter()
        .map(|(p, c)| {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            let detail = match c {
                Ok(c) => format!("{} · {}", c.run.stage.as_str(), format_count(c.schedule.max_steps as u64)),
                Err(_) => "invalid".into(),
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!("{name:<22}"), bold()),
                Span::styled(detail, theme::muted()),
            ]))
        })
        .collect();
    let mut st = ListState::default().with_selected(Some(t.selected));
    f.render_stateful_widget(
        List::new(items)
            .block(panel("training configs", !app.training_active()))
            .highlight_style(Style::new().bg(RAISED).fg(GOLD))
            .highlight_symbol("▸ "),
        configs,
        &mut st,
    );

    // Stats panel.
    let mut lines: Vec<Line> = Vec::new();
    match &t.info {
        None => {
            if t.show_config {
                if let Some((_, Ok(c))) = t.configs.get(t.selected) {
                    for l in c.to_toml().unwrap_or_default().lines() {
                        lines.push(Line::from(Span::styled(l.to_string(), theme::muted())));
                    }
                }
            } else if let Some(e) = &t.error {
                lines.push(Line::from(Span::styled(format!("⚠ {e}"), Style::new().fg(RED))));
            } else {
                lines.push(Line::from(Span::styled("No run in progress.", theme::muted())));
                lines.push(Line::raw(""));
                lines.push(Line::from(Span::styled("Press enter to start the selected config from", faint())));
                lines.push(Line::from(Span::styled("random initialisation (or its init checkpoint),", faint())));
                lines.push(Line::from(Span::styled("r to resume its latest checkpoint, c to view it.", faint())));
                lines.push(Line::from(Span::styled("w watches a run started from another shell.", faint())));
            }
        }
        Some(info) => {
            let last = t.steps.last();
            let val = t.evals.last();
            let status = if t.summary.is_some() {
                ("finished", GUM)
            } else if t.watch.is_some() {
                ("training · watching", GOLD)
            } else if t.error.is_some() {
                ("failed", RED)
            } else if t.paused {
                ("paused", OCHRE)
            } else {
                ("training", GOLD)
            };
            lines.push(kv_styled("status", status.0, Style::new().fg(status.1).add_modifier(Modifier::BOLD)));
            lines.push(kv("run", info.run_id.clone()));
            lines.push(kv("stage", info.stage.as_str()));
            lines.push(kv("model", format!("{} · {}", info.model.name, format_count(info.parameters as u64))));
            lines.push(kv("dataset", info.dataset.clone()));
            lines.push(kv_styled(
                "backend",
                info.backend_label.clone(),
                Style::new().fg(if info.backend == glimmer_core::BackendKind::Metal { GUM } else { OCHRE }),
            ));
            lines.push(Line::raw(""));
            if let Some(s) = last {
                let remaining = info.max_steps.saturating_sub(s.step) as f64;
                let avg_ms =
                    t.steps.iter().rev().take(50).map(|s| s.step_time_ms).sum::<f64>() / t.steps.len().min(50) as f64;
                lines.push(kv("step", format!("{} / {}", s.step, info.max_steps)));
                lines.push(kv_styled("train loss", format!("{:.4}", s.loss), accent()));
                lines.push(kv_styled(
                    "val loss",
                    val.map(|v| format!("{:.4}", v.val_loss)).unwrap_or_else(|| "—".into()),
                    Style::new().fg(BLUE),
                ));
                lines.push(kv("perplexity", val.map(|v| format!("{:.2}", v.val_ppl)).unwrap_or_else(|| "—".into())));
                lines.push(kv_styled("learning rate", format!("{:.2e}", s.lr), Style::new().fg(VIOLET)));
                lines.push(kv("grad norm", format!("{:.3}{}", s.grad_norm, if s.clipped { " (clipped)" } else { "" })));
                lines.push(kv_styled("tokens/sec", format!("{:.0}", s.tokens_per_sec), Style::new().fg(GUM)));
                lines.push(kv("steps/sec", format!("{:.2}", 1000.0 / s.step_time_ms.max(1e-6))));
                lines.push(kv("tokens seen", format_count(s.tokens_seen)));
                lines.push(kv("elapsed", format_duration(s.elapsed_secs)));
                lines.push(kv(
                    "eta",
                    if t.summary.is_some() { "—".into() } else { format_duration(remaining * avg_ms / 1e3) },
                ));
            } else {
                lines.push(Line::from(Span::styled("compiling kernels and warming up…", faint())));
            }
            if t.watch.is_none() {
                if let Some(rss) = app.sys.process_rss {
                    lines.push(kv("process memory", format_bytes(rss)));
                }
            }
            if let Some(g) = app.sys.gpu {
                if let Some(u) = g.device_utilization {
                    lines.push(kv("gpu busy", format!("{u:.0}%")));
                }
                if let Some(m) = g.in_use_memory {
                    lines.push(kv("gpu memory", format_bytes(m)));
                }
            }
            if let Some(s) = &t.summary {
                lines.push(Line::raw(""));
                lines.push(kv_styled("result", format!("{:?}", s.reason), Style::new().fg(GUM)));
                if let Some(b) = s.best_val_loss {
                    lines.push(kv("best val", format!("{b:.4} @ step {}", s.best_step.unwrap_or(0))));
                }
                if let Some(m) = &s.exported_model {
                    lines.push(kv_styled("exported", app.ws.relative(m), Style::new().fg(GUM)));
                }
            }
        }
    }
    let [st_area, gauge_area] = Layout::vertical([Constraint::Min(4), Constraint::Length(3)]).areas(stats);
    f.render_widget(Paragraph::new(lines).block(panel("run", app.training_active())), st_area);
    let ratio = match (&t.info, t.steps.last()) {
        (Some(i), Some(s)) => (s.step as f64 / i.max_steps.max(1) as f64).clamp(0.0, 1.0),
        _ => 0.0,
    };
    f.render_widget(
        Gauge::default()
            .block(panel("progress", false))
            .gauge_style(Style::new().fg(GOLD).bg(RAISED))
            .ratio(ratio)
            .label(Span::styled(format!("{:.1}%", ratio * 100.0), bold())),
        gauge_area,
    );

    // Right: charts + log/samples.
    let [charts, bottom] = Layout::vertical([Constraint::Percentage(68), Constraint::Percentage(32)]).areas(right);
    let [row1, row2] = Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(charts);
    let [lr_area, tp_area] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(row2);
    let raw: Vec<(f64, f64)> = t.steps.iter().map(|s| (s.step as f64, s.loss)).collect();
    let val: Vec<(f64, f64)> = t.evals.iter().map(|e| (e.step as f64, e.val_loss)).collect();
    chart(
        f,
        row1,
        "loss",
        vec![
            Series { name: "train", color: GOLD_DEEP, points: downsample(&raw, 600), scatter: false },
            Series { name: "train (ema)", color: GOLD, points: downsample(&ema(&raw, 0.05), 600), scatter: false },
            Series { name: "validation", color: BLUE, points: val, scatter: false },
        ],
        f3,
        true,
    );
    let lr: Vec<(f64, f64)> = t.steps.iter().map(|s| (s.step as f64, s.lr)).collect();
    chart(
        f,
        lr_area,
        "learning rate",
        vec![Series { name: "lr", color: VIOLET, points: downsample(&lr, 400), scatter: false }],
        sci,
        true,
    );
    let tp: Vec<(f64, f64)> = t.steps.iter().skip(1).map(|s| (s.step as f64, s.tokens_per_sec)).collect();
    chart(
        f,
        tp_area,
        "throughput",
        vec![Series { name: "tokens/s", color: GUM, points: downsample(&ema(&tp, 0.1), 400), scatter: false }],
        kilo,
        true,
    );

    let [log_area, sample_area] =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(bottom);
    let h = log_area.height.saturating_sub(2) as usize;
    let max_w = log_area.width.saturating_sub(10) as usize;
    let log_lines: Vec<Line> = t
        .log
        .iter()
        .rev()
        .take(h)
        .rev()
        .map(|(k, m)| {
            let (tag, c) = match k {
                EventKind::Checkpoint => ("ckpt ", MUTED),
                EventKind::Advice => ("advice", OCHRE),
                EventKind::AutoAction => ("auto ", BLUE),
                EventKind::Warning => ("warn ", OCHRE),
                EventKind::Error => ("error", RED),
                EventKind::Info => ("info ", FAINT),
            };
            let msg: String = m.replace(" → checkpoints/best", "").chars().take(max_w).collect();
            Line::from(vec![Span::styled(format!("{tag} "), Style::new().fg(c)), Span::styled(msg, theme::muted())])
        })
        .collect();
    // One line per event, newest at the bottom, so the latest events are always visible.
    f.render_widget(Paragraph::new(log_lines).block(panel("events", false)), log_area);
    let mut sl: Vec<Line> = Vec::new();
    for s in t.samples.iter().rev().take(3) {
        sl.push(Line::from(vec![
            Span::styled(format!("step {} ", s.step), faint()),
            Span::styled(s.prompt.clone(), bold()),
        ]));
        sl.push(Line::from(Span::styled(s.completion.replace('\n', " ⏎ "), theme::stevie())));
    }
    if sl.is_empty() {
        sl.push(Line::from(Span::styled("samples appear at each evaluation", faint())));
    }
    f.render_widget(Paragraph::new(sl).wrap(Wrap { trim: true }).block(panel("live samples", false)), sample_area);
}

// ── Chat ────────────────────────────────────────────────────────────────────────────────────

/// Greedy word wrap that preserves explicit newlines.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out = Vec::new();
    for para in text.split('\n') {
        let mut line = String::new();
        for word in para.split(' ') {
            let w = word.width();
            if line.is_empty() {
                line.push_str(word);
            } else if line.width() + 1 + w <= width {
                line.push(' ');
                line.push_str(word);
            } else {
                out.push(std::mem::take(&mut line));
                line.push_str(word);
            }
            while line.width() > width {
                let cut: String = line.chars().take(width).collect();
                let rest: String = line.chars().skip(width).collect();
                out.push(cut);
                line = rest;
            }
        }
        out.push(line);
    }
    out
}

fn chat(f: &mut Frame, area: Rect, app: &App) {
    let c = &app.chat;
    let [main, side] = Layout::horizontal([Constraint::Min(40), Constraint::Length(36)]).areas(area);
    let [msgs_area, input_area] = Layout::vertical([Constraint::Min(5), Constraint::Length(3)]).areas(main);

    let title = match &c.model {
        Some(m) if m.name == "stevie" => "❦ STEVIE · a little Aussie AI mate".to_string(),
        Some(m) => format!("{} · chat", m.name),
        None => "chat".into(),
    };
    let block = panel("", !c.focus_input).title(Line::from(vec![
        Span::raw(" "),
        Span::styled(title, theme::stevie().add_modifier(Modifier::BOLD)),
        Span::raw(" "),
    ]));
    let inner = block.inner(msgs_area);
    f.render_widget(block, msgs_area);
    let width = inner.width.saturating_sub(4) as usize;
    let mut lines: Vec<Line> = Vec::new();
    for (i, l) in c.lines.iter().enumerate() {
        match l.role {
            Some(Role::User) => {
                lines.push(Line::from(Span::styled("you", Style::new().fg(BLUE).add_modifier(Modifier::BOLD))));
                for w in wrap(&l.text, width) {
                    lines.push(Line::from(vec![
                        Span::styled("│ ", Style::new().fg(BLUE)),
                        Span::styled(w, Style::new().fg(TEXT)),
                    ]));
                }
            }
            Some(_) => {
                let streaming = c.streaming && i == c.lines.len() - 1;
                let mut head = vec![Span::styled("stevie", theme::stevie().add_modifier(Modifier::BOLD))];
                if let Some(m) = &l.meta {
                    head.push(Span::styled(format!("  {m}"), faint()));
                }
                lines.push(Line::from(head));
                let mut body = l.text.clone();
                if streaming {
                    body.push(if (app.tick / 4).is_multiple_of(2) { '▍' } else { ' ' });
                }
                if body.is_empty() && streaming {
                    body = "thinking…".into();
                }
                for w in wrap(&body, width) {
                    lines.push(Line::from(vec![
                        Span::styled("│ ", Style::new().fg(GUM)),
                        Span::styled(w, Style::new().fg(TEXT)),
                    ]));
                }
            }
            None => {
                for w in wrap(&l.text, width) {
                    lines.push(Line::from(Span::styled(format!("  {w}"), faint())));
                }
            }
        }
        lines.push(Line::raw(""));
    }
    if let Some(m) = &c.loading {
        lines.push(Line::from(Span::styled(format!("  loading {m}…"), faint())));
    }
    let visible = inner.height as usize;
    let total = lines.len();
    let max_scroll = total.saturating_sub(visible);
    let scroll = max_scroll.saturating_sub((c.scroll as usize).min(max_scroll));
    f.render_widget(Paragraph::new(lines).scroll((scroll as u16, 0)), inner);

    let prompt = if c.streaming { "  stevie is typing… (esc to stop)".to_string() } else { c.input.clone() };
    let cursor = if c.focus_input && !c.streaming && (app.tick / 5).is_multiple_of(2) { "▏" } else { " " };
    let input = Paragraph::new(Line::from(vec![
        Span::styled("› ", accent()),
        Span::styled(prompt, if c.streaming { faint() } else { Style::new().fg(TEXT) }),
        Span::styled(cursor, accent()),
    ]))
    .block(panel("message", c.focus_input));
    f.render_widget(input, input_area);

    // Sidebar.
    let mut sl: Vec<Line> = Vec::new();
    match &c.model {
        Some(m) => {
            sl.push(kv_styled("model", m.name.clone(), theme::stevie()));
            sl.push(kv("parameters", format_count(m.parameters as u64)));
            sl.push(kv("context", format!("{} tokens", m.context)));
            sl.push(kv("backend", m.backend.clone()));
            if let Some(card) = &m.card {
                sl.push(kv("trained on", format!("{} tok", format_count(card.total_tokens()))));
                sl.push(kv("lineage", card.stages.iter().map(|s| s.stage.clone()).collect::<Vec<_>>().join("→")));
            }
        }
        None => sl.push(Line::from(Span::styled("no model loaded", faint()))),
    }
    sl.push(Line::raw(""));
    sl.push(Line::from(Span::styled("last reply", bold())));
    match &c.last {
        Some(g) => {
            sl.push(kv_styled("speed", format!("{:.1} tok/s", g.decode_tokens_per_sec), Style::new().fg(GUM)));
            sl.push(kv("duration", format!("{:.2} s", g.total_ms / 1e3)));
            sl.push(kv("tokens", g.generated_tokens.to_string()));
            sl.push(kv("prefill", format!("{:.0} ms", g.prefill_ms)));
            sl.push(kv("cache reuse", format!("{} tok", g.reused_tokens)));
            sl.push(kv("context used", format!("{}/{}", g.context_used, g.context_length)));
        }
        None => sl.push(Line::from(Span::styled("—", faint()))),
    }
    sl.push(Line::raw(""));
    sl.push(Line::from(Span::styled("sampling", bold())));
    let s = &c.sampling;
    sl.push(kv("temperature", format!("{:.2}", s.temperature)));
    sl.push(kv("top-k / top-p", format!("{} / {:.2}", s.top_k, s.top_p)));
    sl.push(kv("repetition", format!("{:.2}", s.repetition_penalty)));
    sl.push(kv("max tokens", s.max_new_tokens.to_string()));
    let [info_area, ctx_area] = Layout::vertical([Constraint::Min(6), Constraint::Length(3)]).areas(side);
    f.render_widget(Paragraph::new(sl).block(panel("session", false)), info_area);
    let ratio = c.last.as_ref().map(|g| g.context_used as f64 / g.context_length.max(1) as f64).unwrap_or(0.0);
    f.render_widget(
        Gauge::default()
            .block(panel("context window", false))
            .gauge_style(Style::new().fg(GUM).bg(RAISED))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(Span::styled(format!("{:.0}%", ratio * 100.0), bold())),
        ctx_area,
    );
}

// ── Models ──────────────────────────────────────────────────────────────────────────────────

fn models(f: &mut Frame, area: Rect, app: &App) {
    let [left, right] = Layout::horizontal([Constraint::Length(34), Constraint::Min(30)]).areas(area);
    let items: Vec<ListItem> = app
        .models
        .iter()
        .map(|m| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:<16}", m.name), bold()),
                Span::styled(format_count(m.config.param_count() as u64), theme::muted()),
            ]))
        })
        .collect();
    let mut st = ListState::default().with_selected(Some(app.model_sel));
    let list = if items.is_empty() {
        List::new(vec![ListItem::new(Span::styled("no models yet", faint()))])
    } else {
        List::new(items)
    };
    f.render_stateful_widget(
        list.block(panel("models", true)).highlight_style(Style::new().bg(RAISED).fg(GOLD)).highlight_symbol("▸ "),
        left,
        &mut st,
    );
    let mut lines = Vec::new();
    if let Some(m) = app.models.get(app.model_sel) {
        let c = &m.config;
        lines.push(Line::from(Span::styled(m.name.clone(), theme::title())));
        if let Some(card) = &m.card {
            lines.push(Line::from(Span::styled(card.description.clone(), theme::muted())));
        }
        lines.push(Line::raw(""));
        lines.push(kv("parameters", format!("{} ({})", format_count(c.param_count() as u64), c.param_count())));
        lines.push(kv(
            "architecture",
            format!("{} layers · d_model {} · {} heads · d_ff {}", c.n_layers, c.d_model, c.n_heads, c.d_ff),
        ));
        lines.push(kv(
            "components",
            format!("{:?} · {:?} · RoPE θ={} · tied={}", c.norm, c.ffn, c.rope_theta, c.tie_embeddings),
        ));
        lines.push(kv("context", format!("{} tokens", c.context_length)));
        lines.push(kv("vocabulary", format!("{} tokens (byte-level BPE)", c.vocab_size)));
        lines.push(kv("location", app.ws.relative(&m.dir)));
        if let Some(card) = &m.card {
            lines.push(kv("hardware", card.hardware.clone().unwrap_or_default()));
            lines.push(Line::raw(""));
            lines.push(Line::from(Span::styled("training lineage", bold())));
            for (i, s) in card.stages.iter().enumerate() {
                let from = s.initialised_from.clone().unwrap_or_else(|| "random initialisation".into());
                lines.push(Line::from(vec![
                    Span::styled(format!(" {}. ", i + 1), accent()),
                    Span::styled(format!("{:<9}", s.stage), bold()),
                    Span::styled(
                        format!(
                            "{} steps · {} tokens · best val {} · {} · {}",
                            s.steps,
                            format_count(s.tokens),
                            s.best_val_loss.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into()),
                            format_duration(s.wall_secs),
                            s.backend
                        ),
                        theme::muted(),
                    ),
                ]));
                lines.push(Line::from(Span::styled(format!("      dataset {} · from {}", s.dataset, from), faint())));
            }
        }
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled("press enter to chat with this model", faint())));
    }
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).block(panel("details", false)), right);
}

// ── Datasets ────────────────────────────────────────────────────────────────────────────────

fn datasets(f: &mut Frame, area: Rect, app: &App) {
    let [left, right] = Layout::horizontal([Constraint::Length(34), Constraint::Min(30)]).areas(area);
    let items: Vec<ListItem> = app
        .datasets
        .iter()
        .map(|(n, m)| {
            let detail = m
                .as_ref()
                .map(|m| format!("{:?} · {}", m.kind, format_count(m.total_tokens())).to_lowercase())
                .unwrap_or_else(|_| "error".into());
            ListItem::new(Line::from(vec![
                Span::styled(format!("{n:<16}"), bold()),
                Span::styled(detail, theme::muted()),
            ]))
        })
        .collect();
    let mut st = ListState::default().with_selected(Some(app.dataset_sel));
    f.render_stateful_widget(
        List::new(items)
            .block(panel("datasets", true))
            .highlight_style(Style::new().bg(RAISED).fg(GOLD))
            .highlight_symbol("▸ "),
        left,
        &mut st,
    );
    let mut lines = Vec::new();
    if let Some((_, Ok(m))) = app.datasets.get(app.dataset_sel) {
        lines.push(Line::from(Span::styled(m.name.clone(), theme::title())));
        lines.push(Line::from(Span::styled(m.description.clone(), theme::muted())));
        lines.push(Line::raw(""));
        lines.push(kv("tokenizer", format!("{} (vocab {})", m.tokenizer, m.vocab_size)));
        lines.push(kv("split seed", m.split.seed.to_string()));
        let rows: Vec<Row> = m
            .splits
            .iter()
            .map(|(k, s)| {
                Row::new(vec![
                    k.clone(),
                    s.documents.to_string(),
                    format_count(s.tokens),
                    if s.supervised_tokens > 0 { format_count(s.supervised_tokens) } else { "—".into() },
                ])
            })
            .collect();
        let [info, table, src] =
            Layout::vertical([Constraint::Length(6), Constraint::Length(rows.len() as u16 + 3), Constraint::Min(4)])
                .areas(right);
        f.render_widget(Paragraph::new(lines).block(panel("dataset", false)), info);
        f.render_widget(
            Table::new(
                rows,
                [Constraint::Length(8), Constraint::Length(12), Constraint::Length(12), Constraint::Length(14)],
            )
            .header(Row::new(vec!["split", "documents", "tokens", "supervised"]).style(theme::muted()))
            .block(panel("splits", false)),
            table,
        );
        let mut sl = Vec::new();
        for s in &m.sources {
            sl.push(Line::from(vec![Span::styled("• ", accent()), Span::styled(s.name.clone(), bold())]));
            sl.push(kv("  licence", s.license.clone()));
            sl.push(kv("  weight", format!("{}", s.weight)));
            sl.push(Line::from(Span::styled(format!("  {}", s.provenance), faint())));
            sl.push(Line::raw(""));
        }
        f.render_widget(Paragraph::new(sl).wrap(Wrap { trim: true }).block(panel("sources & licences", false)), src);
    } else {
        f.render_widget(
            Paragraph::new(Span::styled("run `glimmer data pipeline` to prepare datasets", faint()))
                .block(panel("dataset", false)),
            right,
        );
    }
}

// ── Experiments ─────────────────────────────────────────────────────────────────────────────

fn experiments(f: &mut Frame, area: Rect, app: &App) {
    let [left, right] = Layout::horizontal([Constraint::Length(44), Constraint::Min(30)]).areas(area);
    let [runs_area, studies_area] =
        Layout::vertical([Constraint::Percentage(65), Constraint::Percentage(35)]).areas(left);
    let items: Vec<ListItem> =
        app.runs.iter().map(|r| ListItem::new(Span::styled(r.name.clone(), Style::new().fg(TEXT)))).collect();
    let mut st = ListState::default().with_selected(Some(app.run_sel));
    f.render_stateful_widget(
        List::new(items)
            .block(panel("training runs", true))
            .highlight_style(Style::new().bg(RAISED).fg(GOLD))
            .highlight_symbol("▸ "),
        runs_area,
        &mut st,
    );
    let sl: Vec<Line> = if app.studies.is_empty() {
        vec![Line::from(Span::styled("no hyperparameter studies yet — `glimmer tune`", faint()))]
    } else {
        app.studies
            .iter()
            .map(|(n, r)| {
                let best = r.lines().find(|l| l.starts_with("**Best:**")).unwrap_or("").replace("**Best:**", "best:");
                Line::from(vec![Span::styled(format!("{n} "), bold()), Span::styled(best, theme::muted())])
            })
            .collect()
    };
    f.render_widget(
        Paragraph::new(sl).wrap(Wrap { trim: true }).block(panel("hyperparameter studies", false)),
        studies_area,
    );

    let [chart_area, info_area] =
        Layout::vertical([Constraint::Percentage(70), Constraint::Percentage(30)]).areas(right);
    match &app.curve {
        Some(c) => {
            chart(
                f,
                chart_area,
                &c.name,
                vec![
                    Series {
                        name: "train (ema)",
                        color: GOLD,
                        points: downsample(&ema(&c.train, 0.05), 600),
                        scatter: false,
                    },
                    Series { name: "validation", color: BLUE, points: c.val.clone(), scatter: false },
                ],
                f3,
                true,
            );
            let mut lines = Vec::new();
            if let Some(s) = &c.last {
                lines.push(kv(
                    "last step",
                    format!(
                        "{} · loss {:.4} · {:.0} tok/s · {}",
                        s.step,
                        s.loss,
                        s.tokens_per_sec,
                        format_duration(s.elapsed_secs)
                    ),
                ));
            }
            lines.push(kv(
                "best val",
                c.best_val.map(|v| format!("{v:.4} (ppl {:.2})", v.exp())).unwrap_or_else(|| "—".into()),
            ));
            for e in c.events.iter().rev().take(4) {
                lines.push(Line::from(Span::styled(e.clone(), Style::new().fg(OCHRE))));
            }
            f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }).block(panel("summary", false)), info_area);
        }
        None => f.render_widget(Paragraph::new(Span::styled("no runs yet", faint())).block(panel("run", false)), right),
    }
}

// ── Evaluation ──────────────────────────────────────────────────────────────────────────────

fn evaluation(f: &mut Frame, area: Rect, app: &App) {
    let [left, right] = Layout::horizontal([Constraint::Length(34), Constraint::Min(30)]).areas(area);
    let items: Vec<ListItem> = app
        .reports
        .iter()
        .map(|(file, r)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:<18}", r.label), bold()),
                Span::styled(file.clone(), faint()),
            ]))
        })
        .collect();
    let mut st = ListState::default().with_selected(Some(app.report_sel));
    f.render_stateful_widget(
        List::new(items)
            .block(panel("reports", true))
            .highlight_style(Style::new().bg(RAISED).fg(GOLD))
            .highlight_symbol("▸ "),
        left,
        &mut st,
    );
    let Some((_, r)) = app.reports.get(app.report_sel) else {
        f.render_widget(
            Paragraph::new(Span::styled("no evaluation reports yet — `glimmer evaluate`", faint()))
                .block(panel("report", false)),
            right,
        );
        return;
    };
    let a = &r.aggregate;
    let pct = |v: f64| format!("{:.0}%", v * 100.0);
    let mut lines = vec![
        Line::from(Span::styled(r.label.clone(), theme::title())),
        kv("source", r.source.clone()),
        kv(
            "conversation",
            r.sft_test_loss.map(|v| format!("test loss {v:.3} · ppl {:.1}", v.exp())).unwrap_or_else(|| "—".into()),
        ),
        kv(
            "general text",
            r.pretrain_test_loss
                .map(|v| format!("test loss {v:.3} · ppl {:.1}", v.exp()))
                .unwrap_or_else(|| "—".into()),
        ),
        kv("ends turn", pct(a.end_of_turn_rate)),
        kv(
            "aussie markers",
            format!("{} of replies · {:.2} per reply", pct(a.australian_marker_rate), a.mean_australian_markers),
        ),
        kv("topical", pct(a.topical_rate)),
        kv("continuity", a.continuity_rate.map(pct).unwrap_or_else(|| "—".into())),
        kv("distinct-2", format!("{:.2}", a.mean_distinct_2)),
        kv("speed", format!("{:.0} tok/s", a.mean_tokens_per_sec)),
        Line::raw(""),
    ];
    for p in &r.prompts {
        let user =
            p.conversation.iter().rev().find(|m| m.role == Role::User).map(|m| m.content.clone()).unwrap_or_default();
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", p.category), faint()),
            Span::styled(user, Style::new().fg(BLUE)),
        ]));
        if let Some(s) = p.samples.first() {
            lines.push(Line::from(Span::styled(
                format!("  {}", if s.text.is_empty() { "(empty)" } else { &s.text }),
                theme::stevie(),
            )));
        }
    }
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).scroll((app.report_scroll, 0)).block(panel("report", false)),
        right,
    );
}

// ── Hardware ────────────────────────────────────────────────────────────────────────────────

fn hardware(f: &mut Frame, area: Rect, app: &App) {
    let [left, right] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(area);
    let mut lines = Vec::new();
    match &app.probe {
        None => lines.push(Line::from(Span::styled("inspecting hardware and verifying backends…", faint()))),
        Some(p) => {
            let h = &p.hardware;
            lines.push(kv("os", h.os_version.clone().unwrap_or_else(|| h.os_name.clone())));
            lines.push(kv("architecture", h.arch.clone()));
            lines.push(kv("cpu", h.cpu_brand.clone().unwrap_or_else(|| "unknown".into())));
            lines.push(kv(
                "cores",
                match (h.performance_cores, h.efficiency_cores) {
                    (Some(p), Some(e)) => format!("{} ({p}P + {e}E)", h.logical_cores),
                    _ => h.logical_cores.to_string(),
                },
            ));
            lines.push(kv("memory", format_bytes(h.total_memory_bytes)));
            match &h.gpu {
                Some(g) => {
                    lines.push(kv(
                        "gpu",
                        format!("{}{}", g.name, g.cores.map(|c| format!(" · {c} cores")).unwrap_or_default()),
                    ));
                    lines.push(kv("metal", g.metal_support.clone().unwrap_or_else(|| "unknown".into())));
                    lines.push(kv("unified memory", if g.unified_memory { "yes" } else { "no" }));
                }
                None => lines.push(kv("gpu", "none detected")),
            }
            lines.push(kv("rust", h.rustc_version.clone()));
            lines.push(Line::raw(""));
            lines.push(Line::from(Span::styled("backend verification (matmul vs host reference)", bold())));
            for b in &p.backends {
                match &b.result {
                    Ok((adapter, target, v)) => {
                        lines.push(kv_styled(b.kind.as_str(), format!("✓ {adapter}"), Style::new().fg(GUM)));
                        lines.push(Line::from(Span::styled(
                            format!(
                                "               max error {:.1e} · cold {:.0} ms · warm {:.2} ms",
                                v.max_abs_error, v.cold_ms, v.warm_ms
                            ),
                            faint(),
                        )));
                        if let Some(t) = target {
                            lines.push(Line::from(Span::styled(format!("               {t}"), faint())));
                        }
                    }
                    Err(e) => lines.push(kv_styled(b.kind.as_str(), format!("✗ {e}"), Style::new().fg(RED))),
                }
            }
            let (preset, why) = h.recommend_preset();
            lines.push(Line::raw(""));
            lines.push(kv_styled("recommended", preset, accent()));
            lines.push(Line::from(Span::styled(why, faint())));
        }
    }
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }).block(panel("hardware", true)), left);

    let [spark, live] = Layout::vertical([Constraint::Length(8), Constraint::Min(4)]).areas(right);
    let data: Vec<u64> = app.gpu_history.iter().copied().collect();
    if data.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("GPU utilisation is not measurable on this system", faint()))
                .block(panel("gpu utilisation", false)),
            spark,
        );
    } else {
        f.render_widget(
            Sparkline::default()
                .block(panel("gpu utilisation (IORegistry, 1 Hz)", false))
                .data(&data)
                .max(100)
                .style(Style::new().fg(GUM)),
            spark,
        );
    }
    let mut l = Vec::new();
    if let Some(g) = app.sys.gpu {
        l.push(kv("device busy", g.device_utilization.map(|v| format!("{v:.0}%")).unwrap_or_else(|| "—".into())));
        l.push(kv("renderer", g.renderer_utilization.map(|v| format!("{v:.0}%")).unwrap_or_else(|| "—".into())));
        l.push(kv("gpu memory", g.in_use_memory.map(format_bytes).unwrap_or_else(|| "—".into())));
        l.push(kv("gpu allocated", g.allocated_memory.map(format_bytes).unwrap_or_else(|| "—".into())));
    } else {
        l.push(Line::from(Span::styled("GPU counters unavailable", faint())));
    }
    l.push(kv("process memory", app.sys.process_rss.map(format_bytes).unwrap_or_else(|| "—".into())));
    if let (Some(u), Some(t)) = (app.sys.system_used, app.sys.system_total) {
        l.push(kv("system memory", format!("{} / {}", format_bytes(u), format_bytes(t))));
    }
    l.push(Line::raw(""));
    l.push(Line::from(Span::styled("All values are read from the OS; nothing is estimated.", faint())));
    f.render_widget(Paragraph::new(l).block(panel("live", false)), live);
}

// ── Settings ────────────────────────────────────────────────────────────────────────────────

fn settings(f: &mut Frame, area: Rect, app: &App) {
    let s = &app.chat.sampling;
    let values = [
        format!("{:.2}", s.temperature),
        s.top_k.to_string(),
        format!("{:.2}", s.top_p),
        format!("{:.2}", s.repetition_penalty),
        s.max_new_tokens.to_string(),
        app.backend.as_str().to_string(),
    ];
    let help = [
        "randomness of sampling (0 = always pick the most likely token)",
        "sample only among the k most likely tokens (0 = off)",
        "nucleus sampling: smallest set of tokens covering this probability",
        "discourages repeating recent tokens (1.0 = off)",
        "upper bound on the length of a reply",
        "compute backend for chat (reloads the model)",
    ];
    let items: Vec<ListItem> = App::SETTINGS
        .iter()
        .zip(values.iter())
        .zip(help.iter())
        .map(|((k, v), h)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{k:<22}"), bold()),
                Span::styled(format!("◂ {v:^8} ▸   "), accent()),
                Span::styled(*h, faint()),
            ]))
        })
        .collect();
    let mut st = ListState::default().with_selected(Some(app.settings_sel));
    let [list, about] = Layout::vertical([Constraint::Length(10), Constraint::Min(4)]).areas(area);
    f.render_stateful_widget(
        List::new(items)
            .block(panel("chat sampling", true))
            .highlight_style(Style::new().bg(RAISED))
            .highlight_symbol("▸ "),
        list,
        &mut st,
    );
    let lines = vec![
        kv("workspace", app.ws.root.display().to_string()),
        kv("glimmer", glimmer_core::VERSION),
        kv("build", format!("{} · {}", glimmer_core::BUILD_PROFILE, glimmer_core::RUSTC_VERSION)),
    ];
    f.render_widget(Paragraph::new(lines).block(panel("about", false)), about);
    let _ = Clear;
}
