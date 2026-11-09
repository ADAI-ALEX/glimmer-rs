//! Every screen must render without panicking at small and large terminal sizes.

use glimmer_core::{BackendKind, Workspace};
use glimmer_terminal::app::App;
use glimmer_terminal::{Screen, ui};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

#[test]
fn all_screens_render_at_various_sizes() {
    let root = std::env::temp_dir().join(format!("glimmer-tui-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("glimmer.toml"), "").unwrap();
    let ws = Workspace::open(root.clone()).unwrap();
    let mut app = App::new(ws, Screen::Home, None, BackendKind::Cpu);
    for (w, h) in [(80, 24), (120, 36), (200, 60)] {
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        for screen in Screen::ALL {
            app.screen = screen;
            app.update();
            term.draw(|f| ui::draw(f, &app)).unwrap_or_else(|e| panic!("{screen:?} at {w}x{h}: {e}"));
            let buf = term.backend().buffer().clone();
            let text: String = buf.content().iter().map(|c| c.symbol()).collect();
            assert!(text.contains("GLIMMER"), "header missing on {screen:?}");
        }
    }
    std::fs::remove_dir_all(root).ok();
}
