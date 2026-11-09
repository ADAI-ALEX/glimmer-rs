//! Embeds the compiler version so `glimmer doctor` can report the toolchain that built the binary,
//! even on machines without Rust installed.
use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=GLIMMER_RUSTC_VERSION={version}");
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".into());
    println!("cargo:rustc-env=GLIMMER_BUILD_PROFILE={profile}");
    println!("cargo:rerun-if-changed=build.rs");
}
