//! GLIMMER visual identity for the terminal.
//!
//! A restrained dark palette: near-black surfaces, cool greys for structure, and a single warm
//! accent (GLIMMER gold). STEVIE has its own accent — eucalyptus green with an ochre highlight —
//! so the model's voice is visually distinct from the framework around it.

use ratatui::style::{Color, Modifier, Style};

pub const BG: Color = Color::Rgb(13, 14, 18);
pub const SURFACE: Color = Color::Rgb(20, 22, 28);
pub const RAISED: Color = Color::Rgb(28, 31, 39);
pub const BORDER: Color = Color::Rgb(46, 50, 62);
pub const BORDER_FOCUS: Color = Color::Rgb(116, 104, 74);
pub const TEXT: Color = Color::Rgb(222, 224, 230);
pub const MUTED: Color = Color::Rgb(132, 137, 152);
pub const FAINT: Color = Color::Rgb(80, 85, 100);

/// GLIMMER accent.
pub const GOLD: Color = Color::Rgb(242, 196, 110);
pub const GOLD_DEEP: Color = Color::Rgb(222, 160, 82);
/// Validation / information.
pub const BLUE: Color = Color::Rgb(125, 166, 247);
/// Learning rate.
pub const VIOLET: Color = Color::Rgb(184, 152, 245);
/// STEVIE: eucalyptus green and ochre.
pub const GUM: Color = Color::Rgb(128, 198, 154);
pub const OCHRE: Color = Color::Rgb(214, 134, 78);
pub const RED: Color = Color::Rgb(238, 112, 120);

pub fn base() -> Style {
    Style::new().fg(TEXT).bg(BG)
}
pub fn muted() -> Style {
    Style::new().fg(MUTED)
}
pub fn faint() -> Style {
    Style::new().fg(FAINT)
}
pub fn accent() -> Style {
    Style::new().fg(GOLD)
}
pub fn bold() -> Style {
    Style::new().fg(TEXT).add_modifier(Modifier::BOLD)
}
pub fn title() -> Style {
    Style::new().fg(GOLD).add_modifier(Modifier::BOLD)
}
pub fn stevie() -> Style {
    Style::new().fg(GUM)
}
pub fn selected() -> Style {
    Style::new().fg(BG).bg(GOLD).add_modifier(Modifier::BOLD)
}
pub fn border(focused: bool) -> Style {
    Style::new().fg(if focused { BORDER_FOCUS } else { BORDER })
}

/// The GLIMMER wordmark, drawn with a warm vertical gradient.
pub const LOGO: [&str; 5] = [
    " ██████  ██      ██ ███    ███ ███    ███ ███████ ██████ ",
    "██       ██      ██ ████  ████ ████  ████ ██      ██   ██",
    "██   ███ ██      ██ ██ ████ ██ ██ ████ ██ █████   ██████ ",
    "██    ██ ██      ██ ██  ██  ██ ██  ██  ██ ██      ██   ██",
    " ██████  ███████ ██ ██      ██ ██      ██ ███████ ██   ██",
];

pub fn logo_color(row: usize) -> Color {
    // Interpolate gold → deep amber down the wordmark.
    let t = row as f32 / 4.0;
    let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t) as u8;
    Color::Rgb(lerp(250, 214), lerp(214, 140), lerp(140, 70))
}
