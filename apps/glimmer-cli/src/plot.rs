//! `glimmer runs plot`: render a run's loss curves to a standalone SVG.

use anyhow::{Result, bail};
use glimmer_core::metrics::{MetricRecord, read_metrics};
use std::fmt::Write;
use std::path::Path;

fn ema(points: &[(f64, f64)], alpha: f64) -> Vec<(f64, f64)> {
    let mut v: Option<f64> = None;
    points
        .iter()
        .map(|&(x, y)| {
            let n = v.map_or(y, |p| alpha * y + (1.0 - alpha) * p);
            v = Some(n);
            (x, n)
        })
        .collect()
}

fn nice_ticks(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let span = (hi - lo).max(1e-9);
    let raw = span / n as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * mag).find(|s| span / s <= n as f64).unwrap_or(mag * 10.0);
    let mut t = (lo / step).ceil() * step;
    let mut out = Vec::new();
    while t <= hi + 1e-9 {
        out.push(t);
        t += step;
    }
    out
}

/// Render training (EMA) and validation loss of a run as an SVG.
pub fn plot_run(run_dir: &Path, title: &str, out: &Path) -> Result<()> {
    let recs = read_metrics(run_dir.join("metrics.jsonl"))?;
    let mut train = Vec::new();
    let mut val = Vec::new();
    for r in recs {
        match r {
            MetricRecord::Step(s) if s.loss.is_finite() => train.push((s.step as f64, s.loss)),
            MetricRecord::Eval(e) if e.val_loss.is_finite() => val.push((e.step as f64, e.val_loss)),
            _ => {}
        }
    }
    if train.len() < 2 {
        bail!("{} has fewer than two steps", run_dir.display());
    }
    let smooth = ema(&train, if train.len() > 2000 { 0.02 } else { 0.1 });
    let stride = (smooth.len() / 800).max(1);
    let smooth: Vec<(f64, f64)> = smooth.iter().step_by(stride).copied().collect();
    let (w, h) = (900.0, 420.0);
    let (ml, mr, mt, mb) = (64.0, 24.0, 52.0, 48.0);
    let xmax = train.last().unwrap().0;
    let xmin = 0.0;
    // Skip the first few steps when scaling the y axis so early transients don't flatten the curve.
    let skip = (smooth.len() / 50).max(1);
    let ys: Vec<f64> = smooth.iter().skip(skip).map(|p| p.1).chain(val.iter().map(|p| p.1)).collect();
    let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let ymax = smooth.first().map(|p| p.1).unwrap_or(0.0).max(ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (ymin, ymax) = (ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.03);
    let sx = |x: f64| ml + (x - xmin) / (xmax - xmin).max(1.0) * (w - ml - mr);
    let sy = |y: f64| mt + (1.0 - (y.clamp(ymin, ymax) - ymin) / (ymax - ymin).max(1e-9)) * (h - mt - mb);
    let mut s = String::new();
    writeln!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">"##
    )?;
    writeln!(s, r##"<rect width="{w}" height="{h}" rx="14" fill="#0D0E12"/>"##)?;
    let font = "font-family=\"'SF Mono','JetBrains Mono',Menlo,Consolas,monospace\"";
    writeln!(s, r##"<text x="{ml}" y="30" {font} font-size="15" font-weight="700" fill="#F2C46E">{title}</text>"##)?;
    for t in nice_ticks(ymin, ymax, 5) {
        let y = sy(t);
        writeln!(s, r##"<line x1="{ml}" x2="{}" y1="{y:.1}" y2="{y:.1}" stroke="#1F222B"/>"##, w - mr)?;
        writeln!(
            s,
            r##"<text x="{}" y="{:.1}" {font} font-size="11" fill="#6E7486" text-anchor="end">{t:.2}</text>"##,
            ml - 8.0,
            y + 4.0
        )?;
    }
    for t in nice_ticks(xmin, xmax, 6) {
        let x = sx(t);
        writeln!(s, r##"<line x1="{x:.1}" x2="{x:.1}" y1="{mt}" y2="{}" stroke="#1F222B"/>"##, h - mb)?;
        let label = if t >= 1000.0 { format!("{}k", t / 1000.0) } else { format!("{t}") };
        writeln!(
            s,
            r##"<text x="{x:.1}" y="{}" {font} font-size="11" fill="#6E7486" text-anchor="middle">{label}</text>"##,
            h - mb + 18.0
        )?;
    }
    writeln!(
        s,
        r##"<text x="{}" y="{}" {font} font-size="11" fill="#6E7486" text-anchor="middle">optimiser step</text>"##,
        (ml + w - mr) / 2.0,
        h - 10.0
    )?;
    let path = |pts: &[(f64, f64)]| {
        pts.iter().map(|&(x, y)| format!("{:.1},{:.1}", sx(x), sy(y))).collect::<Vec<_>>().join(" ")
    };
    writeln!(s, r##"<polyline points="{}" fill="none" stroke="#F2C46E" stroke-width="2"/>"##, path(&smooth))?;
    if !val.is_empty() {
        writeln!(
            s,
            r##"<polyline points="{}" fill="none" stroke="#7DA6F7" stroke-width="2" stroke-dasharray="1 0"/>"##,
            path(&val)
        )?;
        for &(x, y) in &val {
            writeln!(s, r##"<circle cx="{:.1}" cy="{:.1}" r="2.6" fill="#7DA6F7"/>"##, sx(x), sy(y))?;
        }
    }
    let lx = w - mr - 210.0;
    writeln!(s, r##"<rect x="{lx}" y="14" width="200" height="26" rx="6" fill="#151820" stroke="#2E323E"/>"##)?;
    writeln!(
        s,
        r##"<line x1="{}" x2="{}" y1="27" y2="27" stroke="#F2C46E" stroke-width="2"/><text x="{}" y="31" {font} font-size="11" fill="#C8CBD4">train (EMA)</text>"##,
        lx + 10.0,
        lx + 28.0,
        lx + 34.0
    )?;
    writeln!(
        s,
        r##"<line x1="{}" x2="{}" y1="27" y2="27" stroke="#7DA6F7" stroke-width="2"/><text x="{}" y="31" {font} font-size="11" fill="#C8CBD4">validation</text>"##,
        lx + 112.0,
        lx + 130.0,
        lx + 136.0
    )?;
    writeln!(s, "</svg>")?;
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(out, s)?;
    Ok(())
}
