# Contributing to noddle

Contributions are welcome. This document covers how to get set up, the project conventions, and where the most useful work is.

---

## Getting started

### Prerequisites

- Rust (edition 2024, rustc 1.85+) — install via [rustup](https://rustup.rs/)
- `protoc` (Protocol Buffers compiler)

```sh
# Debian/Ubuntu
sudo apt install protobuf-compiler

# macOS
brew install protobuf
```

### Build and test

```sh
cargo build
cargo test
```

---

## Project conventions

### Code style

- Run `cargo fmt` before committing
- Run `cargo clippy` and address warnings before opening a PR
- All public items should have doc comments (`///`)
- Internal logic that isn't self-evident should have inline comments

### Commits

Keep commits focused. One logical change per commit. Commit messages should say *why*, not just *what* — the diff already shows what changed.

### Pull requests

- Open a PR for any non-trivial change
- Reference the relevant GitHub issue if one exists
- PRs should pass `cargo test` and `cargo clippy` cleanly

---

## Architecture overview

The codebase is a Cargo workspace. Each crate has a single responsibility:

| Crate | Responsibility |
|---|---|
| `noddle-proto` | Wire types generated from `proto/noddle.proto`. The source of truth for every message in the system. |
| `noddle-core` | The `InferenceBackend` trait and all implementations. Tensor handling. Model manifest loading. |
| `noddle-registry` | The `Registry` struct: node capability storage, LWW merge, diff/sync, JSON persistence, gossip loop. |
| `noddle-router` | Fan-out job dispatch, node selection (ping + load + dedup), cancel propagation. |
| `noddle-node` | The daemon binary. Wires all crates together, runs the gRPC server, manages node lifecycle. |
| `noddle-cli` | The `noddle` CLI binary. Thin client — connects to local daemon, submits prompts, streams output. |

The dependency flow is strictly one-way: `proto → core/registry → router → node`. The CLI only depends on `proto`.

### Model agnosticism

Adding a new model never requires code changes:
1. Add a JSON manifest file (see README for the schema)
2. Download the weight file
3. The node picks it up on restart

Adding a new inference backend (e.g. a new runtime beyond llama.cpp):
1. Implement `InferenceBackend` in `noddle-core/src/backend.rs`
2. Register it — any model manifest with the matching `weight_format` will route through it

### Wire protocol

All node-to-node and CLI-to-node communication is defined in `proto/noddle.proto`. If you need to add a new message field or RPC, start there. The generated Rust code in `noddle-proto` is derived automatically at build time — never edit it by hand.

---

## Where to contribute

Good starting points:

- **`LlamaCppBackend`** — implement the real llama.cpp inference backend in `noddle-core`. The `StubBackend` is the reference for what the trait expects.
- **`TransformersBackend`** — HuggingFace Transformers backend via a Python subprocess or PyO3
- **Node daemon** — `noddle-node/src/main.rs` is currently a stub. The gRPC server, TLS setup, config loading, and role assessment all need implementing.
- **CLI** — `noddle-cli/src/main.rs` is a stub. Prompt input, Unix socket connection, streaming output renderer.
- **Tests** — unit tests for registry merge logic, tensor validation, and manifest loading are the highest priority

If you're unsure where to start, open an issue describing what you'd like to work on.
