//! End-to-end test of the `glimmer` binary in a throwaway workspace (CPU backend):
//! prepare a corpus → train a tokenizer → build an SFT dataset → train and export a model →
//! list it → generate → evaluate.

use std::path::{Path, PathBuf};
use std::process::Command;

fn glimmer(ws: &Path, args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_glimmer"))
        .args(args)
        .env("GLIMMER_HOME", ws)
        .env("NO_COLOR", "1")
        .output()
        .expect("running glimmer");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "glimmer {args:?} failed:\n{text}");
    text
}

fn write(path: PathBuf, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

#[test]
fn full_pipeline_through_the_cli() {
    let ws = std::env::temp_dir().join(format!("glimmer-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    write(ws.join("glimmer.toml"), "");

    let mut convs = String::new();
    let pairs = [
        ("How ya going?", "Not bad at all, mate. How about you?"),
        ("What's a wombat?", "A stocky burrowing marsupial with a backwards pouch."),
        ("Tell me a joke.", "What do you call a lazy kangaroo? A pouch potato."),
        ("What's brekkie?", "Brekkie is breakfast, mate."),
        ("Any tips for sleep?", "Keep a regular bedtime and put the phone away."),
        ("What's an esky?", "A cooler for keeping drinks cold at the beach."),
    ];
    for i in 0..30 {
        let (u, s) = pairs[i % pairs.len()];
        convs.push_str(&format!("### c-{i} | test\nU: {u} ({i})\nS: {s} Number {i}.\n\n"));
    }
    write(ws.join("datasets/demo/conversations/a.txt"), &convs);
    write(
        ws.join("datasets/demo/eval.toml"),
        "[[prompt]]\nid = \"p1\"\ncategory = \"greeting\"\nmessages = [{ role = \"user\", content = \"G'day there\" }]\n",
    );
    write(
        ws.join("datasets/sources/convs.toml"),
        r#"id = "convs"
name = "Test conversations"
kind = "conversations"
path = "datasets/demo/conversations"
license = "Apache-2.0"
license_url = "u"
homepage = "h"
provenance = "test"
redistribution = "test"
[cleaning]
min_chars = 1
max_chars = 1000
min_alpha_ratio = 0.0
near_dedup = false
blocklist = false
"#,
    );
    write(
        ws.join("configs/tokenizers/t.toml"),
        "name = \"t\"\nvocab_size = 320\nmin_frequency = 2\n[[sources]]\nid = \"convs\"\n",
    );
    write(
        ws.join("configs/datasets/chat.toml"),
        r#"name = "chat"
kind = "sft"
description = "test chat"
tokenizer = "data/tokenizers/t.json"
[split]
val_fraction = 0.15
test_fraction = 0.15
seed = 3
[[sources]]
id = "convs"
[sft]
eval_prompts = "datasets/demo/eval.toml"
"#,
    );
    write(
        ws.join("configs/training/sft.toml"),
        r#"[run]
name = "e2e"
stage = "sft"
backend = "cpu"
export_as = "demo"
[model]
preset = "micro"
context_length = 64
[data]
dataset = "chat"
[optimizer]
lr = 3e-3
min_lr = 3e-4
[schedule]
warmup_steps = 5
max_steps = 40
[batch]
batch_size = 4
[eval]
every = 20
batches = 2
[checkpoint]
every = 0
"#,
    );

    let out = glimmer(&ws, &["data", "prepare", "--source", "convs"]);
    assert!(out.contains("30 conversations"), "{out}");
    let out = glimmer(&ws, &["tokenizer", "train", "t"]);
    assert!(out.contains("vocab 320"), "{out}");
    glimmer(&ws, &["data", "build", "chat"]);
    let out = glimmer(&ws, &["datasets", "inspect", "chat"]);
    assert!(out.contains("Apache-2.0") && out.contains("train"), "{out}");

    let out = glimmer(&ws, &["train", "--config", "configs/training/sft.toml"]);
    assert!(out.contains("finished") && out.contains("Completed"), "{out}");
    assert!(ws.join("models/demo/model.bpk").is_file(), "model was not exported");
    let card = std::fs::read_to_string(ws.join("models/demo/card.json")).unwrap();
    assert!(card.contains("\"stage\": \"sft\""), "{card}");

    let out = glimmer(&ws, &["models", "list"]);
    assert!(out.contains("demo"), "{out}");
    let out = glimmer(&ws, &["models", "inspect", "demo"]);
    assert!(out.contains("training lineage"), "{out}");
    let out = glimmer(&ws, &["generate", "--model", "demo", "--backend", "cpu", "--max-tokens", "12", "How ya going?"]);
    assert!(out.contains("tok/s"), "{out}");
    glimmer(
        &ws,
        &["evaluate", "--model", "demo", "--prompts", "datasets/demo/eval.toml", "--backend", "cpu", "--name", "e2e"],
    );
    assert!(ws.join("runs/eval/e2e.md").is_file());
    let out = glimmer(&ws, &["runs"]);
    assert!(out.contains("e2e-"), "{out}");
    std::fs::remove_dir_all(&ws).ok();
}

#[test]
fn help_and_doctor_json() {
    let ws = std::env::temp_dir().join(format!("glimmer-help-{}", std::process::id()));
    write(ws.join("glimmer.toml"), "");
    let out = glimmer(&ws, &["--help"]);
    for cmd in ["train", "chat", "evaluate", "tune", "bench", "doctor", "data", "models"] {
        assert!(out.contains(cmd), "help is missing `{cmd}`");
    }
    if std::env::var_os("GLIMMER_SKIP_METAL_TESTS").is_some() {
        std::fs::remove_dir_all(&ws).ok();
        return; // doctor probes the GPU; hosted CI runners have no reliable one
    }
    let out = glimmer(&ws, &["doctor", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("doctor --json prints JSON");
    assert!(v["hardware"]["total_memory_bytes"].as_u64().unwrap() > 0);
    std::fs::remove_dir_all(&ws).ok();
}
