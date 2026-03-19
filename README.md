# noddle

A distributed AI inference network — a compute cooperative where users run nodes that collectively perform AI inference. No single company controls it. The people running the infrastructure are the users.

Read [EXPLANATION.md](./EXPLANATION.md) for the full motivation and architecture design.

---

## How it works

Inference is split across the network by transformer model layers. Node A runs layers 1–10 and passes its tensor output to the next node, which runs layers 11–20, and so on. Multiple redundant parallel chains race through the network simultaneously — the first complete response wins.

Contributing a node earns you access to the tool. Your machine participates in the network, you get free access to the AI coding assistant.

---

## Project structure

```
noddle/
  proto/                    # Protobuf definitions — source of truth for all wire types
  crates/
    noddle-proto/           # Generated Rust types from proto/ (tonic + prost)
    noddle-core/            # Inference engine abstraction, tensor handling, model manifests
    noddle-registry/        # Registry data structure, LWW merge, gossip sync
    noddle-router/          # Fan-out job routing, node selection, cancel propagation
    noddle-node/            # Node daemon binary
    noddle-cli/             # CLI binary (`noddle` command)
  ui/                       # Tauri + Svelte desktop UI (coming soon)
  bootstrap-registry.json   # Seed list of known nodes shipped with the client
  EXPLANATION.md            # Architecture design notes
```

---

## Prerequisites

- [Rust](https://rustup.rs/) (edition 2024 — rustc 1.85+)
- [protoc](https://grpc.io/docs/protoc-installation/) (Protocol Buffers compiler)

```sh
# Debian/Ubuntu
sudo apt install protobuf-compiler

# macOS
brew install protobuf
```

---

## Running locally

### Build everything

```sh
cargo build
```

### Run the node daemon

```sh
cargo run --bin noddle-node
```

The node will:
1. Load the bootstrap registry from `bootstrap-registry.json`
2. Assess your hardware and assign itself a role (WORKER or ADMIN)
3. Start the gRPC server (default: `0.0.0.0:7900`)
4. Begin gossiping with peers every 60 seconds

### Run the CLI

```sh
cargo run --bin noddle -- "your prompt here"
```

The CLI connects to the local node daemon via Unix socket and streams the response back.

### Run the tests

```sh
cargo test
```

---

## Adding a new model

No code changes required. Create a manifest file in the manifests directory:

```json
{
  "model_id": "org/model-name",
  "model_version": "1.0.0",
  "total_layers": 32,
  "weight_format": "gguf",
  "tokenizer": "llama3",
  "min_vram_mb": 5000,
  "tensor_mb_per_layer_per_512_tokens": 4.0,
  "description": "Human-readable description shown in the UI"
}
```

The node will pick it up on next start. Supported `weight_format` values: `gguf`, `safetensors`. Supported `tokenizer` values: `llama3`, `mistral`, `gemma`, `phi3`, `bpe_json`.

---

## Adding a new inference backend

Implement the `InferenceBackend` trait in `crates/noddle-core/src/backend.rs`:

```rust
pub trait InferenceBackend: Send + Sync {
    fn load_model(&mut self, manifest: &ModelManifest, weight_path: &Path) -> Result<()>;
    fn run_layers(&self, layer_range: Range<u32>, input_tensor: &Tensor, tokenized_prompt: &[u32]) -> Result<Tensor>;
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    // ... see backend.rs for the full trait
}
```

Register your implementation and it will work with any model whose manifest specifies the matching `weight_format`.

---

## Configuration

The node reads `~/.config/noddle/config.toml` on startup (created with defaults if absent):

```toml
[node]
weights_dir = "~/.local/share/noddle/weights"
listen_addr = "0.0.0.0:7900"
role = "auto"          # auto | worker | admin

[privacy]
advertise_gpu_specs = false
advertise_bandwidth = false

[routing]
fan_out_width = 3
min_success_count = 2
ping_timeout_ms = 100
```

---

## License

[GNU Affero General Public License v3.0](./LICENSE) — you can use, modify, and distribute this software freely, but any derivative work must also be open source under the same license.
