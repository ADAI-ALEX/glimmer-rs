<p align="center"><img src="assets/branding/banner.svg" alt="GLIMMER" width="100%"></p>

# GLIMMER

A Rust-native laboratory for building, training, evaluating and chatting with small language
models — trained from random initialisation, entirely in Rust (Burn on Apple Metal, with a
pure-Rust CPU fallback), driven from an interactive terminal application.

Its first demonstration model is **STEVIE**, a little Aussie AI mate, trained from scratch.

> 🚧 The demonstration run is in progress; full documentation, results, benchmarks and screenshots
> will be published here shortly. See [`docs/`](docs) for the architecture and methodology.

```bash
cargo build --release
./target/release/glimmer doctor          # inspect hardware, verify backends
./target/release/glimmer data pipeline   # fetch → clean → tokenizer → datasets
./target/release/glimmer train --model stevie
./target/release/glimmer                 # open the laboratory
```

Licensed under Apache-2.0.
