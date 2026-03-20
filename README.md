# noddle
### Distributed AI inference: a compute collective

A distributed AI inference network where users run nodes that collectively perform AI inference. No single company controls it. The people running the infrastructure are the users.

Read [ARCHITECTURE.md](./ARCHITECTURE.md) for a detailed breakdown of how the system works.

---

## How it works

Inference is split across the network by transformer model layers. Node A runs layers 0–9 and passes its hidden state tensor to the next node, which runs layers 10–18, and so on until a node produces the final token. Multiple redundant parallel chains race through the network simultaneously — the first complete response wins, and a cancellation broadcast kills the rest.

If no network peers are available, the node runs all layers locally as a fallback — single-node still works.

Contributing a node earns you access to the tool. Your machine participates in the network, you get free access to the AI assistant.

---

## Architecture

The core primitive is layer-range execution: instead of one machine running an entire model, noddle assigns each node a contiguous range of transformer layers. The hidden state tensor travels hop by hop across the network.

For fault tolerance, the router fans out to multiple independent chains in parallel. The first chain to return a complete response wins; the others are cancelled. See [ARCHITECTURE.md](./ARCHITECTURE.md) for ASCII diagrams of the full data flow.

---

## Project structure

```
noddle/
  proto/                       # Protobuf definitions — source of truth for all wire types
  crates/
    noddle-proto/              # Generated Rust types from proto/ (tonic + prost)
    noddle-core/               # InferenceAdapter trait, tensor wire format, model manifests
    noddle-registry/           # Registry data structure, LWW merge, gossip sync
    noddle-router/             # Fan-out job routing, node selection, cancel propagation
    noddle-node/               # Node daemon binary
    noddle-adapter-candle/     # Real inference via candle-core: GGUF loading, transformer forward pass
    noddle-cli/                # CLI binary (`noddle` command)
  manifests/                   # Bundled model manifests (JSON)
  bootstrap-registry.json      # Seed list of known nodes shipped with the client
  ARCHITECTURE.md              # Detailed architecture reference
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

Supported platforms: Linux, macOS. Windows support is not yet implemented (Unix socket IPC is platform-specific).

---

## Getting started

### 1. Build

```sh
cargo build
```

### 2. Download a model

```sh
cargo run --bin noddle -- pull meta-llama/Llama-3.2-3B-Instruct
```

Downloads the GGUF weights (~2GB) to `~/.local/share/noddle/weights/` with a progress bar. To see all available models:

```sh
cargo run --bin noddle -- models
```

### 3. Start the node daemon

```sh
cargo run --bin noddle-node
```

The node will:
1. Load the bootstrap registry from `bootstrap-registry.json`
2. Detect VRAM (NVIDIA via `nvidia-smi`, AMD via sysfs) and assign itself a role: WORKER or ADMIN
3. Scan `~/.local/share/noddle/weights/` for GGUF files matching known manifests and load them
4. Start the gRPC server (default: `0.0.0.0:7900`) with self-signed TLS
5. Listen on a Unix socket for local CLI connections (`$XDG_RUNTIME_DIR/noddle.sock` on Linux, `/tmp/noddle.sock` on macOS)
6. Begin gossiping with peers every 60 seconds

### 4. Send a prompt

Single-shot:

```sh
cargo run --bin noddle -- "your prompt here"
```

Interactive session (type prompts, Ctrl+D or `exit` to quit):

```sh
cargo run --bin noddle
```

Options:

```sh
noddle [OPTIONS] [PROMPT]

Options:
  -m, --model <MODEL_ID>      Model to use (default: meta-llama/Llama-3.2-3B-Instruct)
  -s, --session <SESSION_ID>  Resume a named session (conversation history is maintained)
```

### 5. Run the tests

```sh
cargo test
```

---

## GPU acceleration

For NVIDIA GPUs, build `noddle-adapter-candle` with the `cuda` feature:

```sh
cargo build -p noddle-node -p noddle-cli --features noddle-adapter-candle/cuda
```

For Apple Silicon (Metal):

```sh
cargo build -p noddle-node -p noddle-cli --features noddle-adapter-candle/metal
```

AMD GPUs are not currently accelerated (candle does not support ROCm). AMD nodes participate as ADMIN/routing nodes and fall back to CPU for inference.

---

## Configuration

The node reads `~/.config/noddle/config.toml` on startup (created with defaults if absent). **The file is watched at runtime — changes take effect without a restart.**

```toml
[node]
listen_addr = "0.0.0.0:7900"
role = "auto"          # auto | worker | admin
active = true          # set to false to pause participation without stopping the process

[privacy]
advertise_gpu_specs = false
advertise_bandwidth = false

[routing]
fan_out_width = 3
min_success_count = 2
ping_timeout_ms = 100
```

---

## Adding a new model

No code changes required. Create a manifest file under `manifests/`:

```json
{
  "model_id": "org/model-name",
  "model_version": "1.0.0",
  "total_layers": 32,
  "weight_format": "gguf",
  "gguf_url": "https://huggingface.co/org/model-name/resolve/main/model.Q4_K_M.gguf",
  "tokenizer": "llama3",
  "min_vram_mb": 5000,
  "tensor_mb_per_layer_per_512_tokens": 4.0,
  "description": "Human-readable description shown in the UI"
}
```

The node picks it up on next start. `noddle pull <model_id>` will download the weights automatically.

Supported `weight_format`: `gguf`, `safetensors`. Supported `tokenizer`: `llama3`, `mistral`, `gemma`, `phi3`, `bpe_json`.

---

## Adding a new inference backend

Implement the `InferenceAdapter` trait in `crates/noddle-core/src/adapter.rs`:

```rust
pub trait InferenceAdapter: Send + Sync {
    fn load_model(&mut self, manifest: &ModelManifest, weight_path: &Path) -> Result<()>;
    fn run_layers(&self, layer_range: Range<u32>, input_tensor: &Tensor, tokenized_prompt: &[u32]) -> Result<Tensor>;
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    fn eos_token_id(&self) -> Option<u32> { None }
    fn apply_chat_template(&self, user_prompt: &str) -> String { user_prompt.to_string() }
    // ... see adapter.rs for the full trait
}
```

The existing implementations are `CandleAdapter` (real inference via candle-core) and `StubAdapter` (testing). A new backend needs to implement `run_layers` — the ability to execute an arbitrary range of transformer layers and return the output tensor. That is the primitive that makes distributed inference possible.

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for setup instructions, code conventions, and where the most useful work is right now.

---

## License

[GNU Affero General Public License v3.0](./LICENSE) — you can use, modify, and distribute this software freely, but any derivative work must also be open source under the same license.
