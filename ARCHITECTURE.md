# noddle Architecture

**DRAFT — this document reflects the current state of the implementation and ongoing design decisions. Details will change.**

---

## Table of contents

1. [Overview](#overview)
2. [Distributed inference model](#distributed-inference-model)
3. [Fan-out redundancy](#fan-out-redundancy)
4. [Node roles](#node-roles)
5. [Registry and gossip](#registry-and-gossip)
6. [Routing](#routing)
7. [Inference engine](#inference-engine)
8. [Security](#security)
9. [Local IPC and admin API](#local-ipc-and-admin-api)
10. [Configuration and live reload](#configuration-and-live-reload)
11. [Node startup sequence](#node-startup-sequence)
12. [Single-node fallback](#single-node-fallback)
13. [Known limitations](#known-limitations)
14. [Planned work](#planned-work)

---

## Overview

noddle is a distributed AI inference network. Rather than a single machine running an entire language model, multiple volunteer nodes each run a contiguous range of transformer layers. The hidden state tensor passes hop by hop through the chain until the final node produces an output token.

```
  User
   |
   v
+-------+         Unix socket (local IPC)
|  CLI  | <-------------------------------------> [ noddle.sock ]
+-------+                                               |
                                                        v
                                               +----------------+
                                               |  Local node    |
                                               |  (noddle-node) |
                                               +----------------+
                                                   |        |
                                    gRPC/TLS        |        |  gRPC/TLS
                                                   v        v
                                          +----------+  +----------+
                                          |  Node B  |  |  Node B' |   <- parallel chains
                                          | layer    |  | layer    |
                                          | 0..9     |  | 0..9     |
                                          +----------+  +----------+
                                               |              |
                                               v              v
                                          +----------+  +----------+
                                          |  Node C  |  |  Node C' |
                                          | layer    |  | layer    |
                                          | 10..18   |  | 10..18   |
                                          +----------+  +----------+
                                               |              |
                                               v              v
                                          +----------+  +----------+
                                          |  Node D  |  |  Node D' |
                                          | layer    |  | layer    |   first complete
                                          | 19..28   |  | 19..28   |   response wins
                                          +----------+  +----------+
                                               |
                                               v
                                          [ token ] --> streamed back to CLI
```

---

## Distributed inference model

A transformer model is a sequence of identical-structure layers (attention + feed-forward). Each layer takes a hidden state tensor as input and produces a hidden state tensor as output. This means layers are composable and the split point between nodes is well-defined.

**Layer assignment**

The router assigns each node in a chain a `LayerRange { start: u32, end: u32 }`. Node A runs layers `[start, end)`, produces a tensor, and forwards it to the next node in the chain. The last node in the chain runs the LM head to produce logits and performs argmax to select the next token.

**Token generation loop**

For a single token generation step:

```
Prompt tokens
     |
     v
[ Node 0: embedding + layers 0..9 ]
     |
     | hidden state tensor (serialized as bytes on the wire)
     v
[ Node 1: layers 10..18 ]
     |
     | hidden state tensor
     v
[ Node 2: layers 19..28 + LM head ]
     |
     v
  next token ID
     |
     v
[ detokenize + append to response ]
     |
     v
  repeat until EOS or MAX_NEW_TOKENS
```

**Wire format**

Tensors are serialized to raw bytes for transmission in `JobMessage` proto fields. The `noddle-core` crate owns the tensor wire format. The receiving node deserializes and continues the forward pass from the correct layer.

**Current model**

The implemented model is Llama 3.2 3B Instruct, loaded in Q4_K_M GGUF quantization (~2 GB weights). The `CandleAdapter` in `crates/noddle-adapter-candle/` implements the full forward pass using `candle-core`.

Key implementation notes:
- Tokenizer is extracted from GGUF metadata at load time — no separate tokenizer download needed
- Architecture metadata uses the GGUF prefix `llama.` (e.g. `llama.embedding_length`), not the generic `llm.` prefix
- Llama 3.2 uses weight tying: `token_embd.weight` doubles as the LM head; there is no separate `output.weight`
- RoPE (Rotary Position Embeddings) cos/sin tables are expanded to the full `head_dim`
- The Llama 3 Instruct chat template is applied automatically before tokenization
- Decoding is greedy (argmax); no sampling or temperature is implemented yet

---

## Fan-out redundancy

To tolerate node failures and slow nodes, the router fans out to multiple independent chains simultaneously. Each chain is a complete path from input to output token.

```
                         [ Router ]
                        /     |     \
                       /      |      \
                      v       v       v
                  Chain 1  Chain 2  Chain 3     <- fan_out_width = 3
                    |         |        |
                    |         |        |
                  Done!     ...      ...        <- Chain 1 wins
                    |
                    v
              [ Response ]
                    |
              [ Cancel broadcast ]
             /       |        \
            v         v        v
       Chain 2    Chain 2    Chain 3            <- killed
```

Configuration:
- `fan_out_width`: how many parallel chains to launch (default: 3)
- `min_success_count`: minimum chains that must complete before the result is accepted (default: 2; currently the first complete chain wins and the rest are cancelled)

The cancellation signal propagates hop by hop through each chain using the `CancelJob` RPC defined in `proto/noddle.proto`.

**Node selection**

For each chain, the router selects nodes from the registry using a combination of:
- Model availability: node must advertise that it has the model weights loaded
- Load: prefer nodes with lower current job count
- Deduplication: avoid reusing the same node within a single chain

---

## Node roles

Every node self-assigns a role at startup based on hardware detection.

| Role   | Requirements           | Responsibilities                              |
|--------|------------------------|-----------------------------------------------|
| WORKER | Sufficient VRAM/GPU    | Runs inference layers, participates in chains |
| ADMIN  | Any hardware           | Routes jobs, maintains registry, no inference |

**VRAM detection**
- NVIDIA: `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
- AMD: reads from `/sys/class/drm/card*/device/mem_info_vram_total`
- If neither is available, the node falls back to ADMIN role

**AMD GPUs**

AMD nodes are assigned ADMIN role regardless of VRAM. `candle-core` does not support ROCm, so there is no GPU-accelerated inference path for AMD hardware. Assigning them WORKER role and falling back to CPU inference is not useful in practice (~5 min/token without KV cache). AMD nodes participate as routing and registry nodes only. WORKER role will be enabled for AMD once ROCm support lands in candle.

**Role override**

The config file `role` field can force a node to `worker` or `admin` regardless of hardware.

---

## Registry and gossip

Every node maintains a full copy of the network registry. The registry is a map of node IDs to `NodeCapability` records (address, role, models, load, sequence number).

**NodeCapability fields** (from `proto/noddle.proto`)
- `node_id`: stable identifier (derived from TLS cert public key)
- `listen_addr`: gRPC address
- `role`: WORKER or ADMIN
- `available_models`: list of model IDs with loaded weights
- `current_load`: active job count
- `sequence_number`: monotonically increasing, used for LWW merge

**LWW merge**

When a node receives a registry update with a higher sequence number for a given node ID, it replaces the local record. Lower or equal sequence numbers are discarded. This is a simple last-write-wins strategy with no conflict resolution beyond sequence ordering.

**Gossip**

```
  Node A                      Node B
    |                            |
    |--- RegistrySync(diff) ---->|
    |                            |  merge, update local registry
    |<-- RegistrySync(diff) -----|
    |                            |
  (periodic, ~60s)
```

Registry diffs are also piggybacked on job messages to reduce gossip overhead.

**Bootstrap**

On first start, a node loads `bootstrap-registry.json` (bundled with the binary) to get an initial set of known peers. It then gossips with those peers to discover the rest of the network.

---

## Routing

The router lives in `crates/noddle-router/`. Its responsibilities:

1. Accept an incoming `PromptRequest` from the local node
2. Query the registry for available WORKER nodes that have the requested model
3. Partition the model's total layer count across selected nodes to build chain(s)
4. Fan out `fan_out_width` parallel chains
5. Collect the first successful response
6. Broadcast cancellation to remaining chains
7. Return the response token stream to the local node

**Layer partitioning**

Given N nodes and L total layers, each node is assigned approximately `L / N` layers. The last node in the chain takes any remainder. Layer boundaries are chosen to minimize tensor transfer size (all boundaries are equivalent for uniform-width transformers).

**Retry / fallback**

If the registry has no available peers (empty or all nodes unavailable), the router falls back to local execution (see [Single-node fallback](#single-node-fallback)).

---

## Inference engine

The inference abstraction lives in `crates/noddle-core/src/adapter.rs`.

```rust
pub trait InferenceAdapter: Send + Sync {
    fn load_model(&mut self, manifest: &ModelManifest, weight_path: &Path) -> Result<()>;
    fn run_layers(&self, layer_range: Range<u32>, input_tensor: &Tensor,
                  tokenized_prompt: &[u32]) -> Result<Tensor>;
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
}
```

**Implementations**

| Type            | Crate                    | Purpose                              |
|-----------------|--------------------------|--------------------------------------|
| `CandleAdapter` | `noddle-adapter-candle`  | Real inference via candle-core       |
| `StubAdapter`   | `noddle-core`            | Testing — returns deterministic data |

**Model manifests**

`ModelManifest` records live as JSON files in `manifests/` and are bundled into the CLI binary via `include_str!`. Fields include `model_id`, `total_layers`, `weight_format`, `gguf_url`, `tokenizer`, `min_vram_mb`, and `tensor_mb_per_layer_per_512_tokens`.

Adding a new model requires only a new JSON file — no code changes.

---

## Security

**Transport security**

All node-to-node gRPC communication is over TLS. On first run, each node generates a self-signed certificate stored in `~/.config/noddle/`. The certificate's public key is the basis for the node's stable `node_id`.

Currently, certificates are self-signed and not cross-verified. Any node that knows another node's address can connect. Certificate pinning or a PKI layer is planned but not implemented.

**Local IPC**

The CLI communicates with the local node daemon over a Unix domain socket (`$XDG_RUNTIME_DIR/noddle.sock` on Linux, `/tmp/noddle.sock` on macOS). No TLS is used for this path — Unix socket permissions provide isolation. The Admin API (SetActive, GetStatus) is only reachable via this socket and is not exposed over the network.

**Known security gaps**

- No certificate verification between nodes: any node can impersonate another
- No authentication on the gRPC job API: any node that can reach the port can submit jobs
- No rate limiting or job quotas
- Node reputation system not implemented (planned: track node reliability, weight selection accordingly)
- Private routing mode not implemented (planned: route only through nodes above a reputation threshold)

---

## Local IPC and admin API

The `AdminService` proto service is exposed only on the Unix socket:

```
AdminService {
  SetActive(SetActiveRequest) -> SetActiveResponse
  GetStatus(GetStatusRequest) -> GetStatusResponse
}
```

`SetActive` writes the `active` field to `~/.config/noddle/config.toml`. The config file watcher (inotify on Linux, kqueue on macOS) detects the change and updates the node's registry advertisement live, without a restart.

---

## Configuration and live reload

Config file: `~/.config/noddle/config.toml`

```toml
[node]
listen_addr = "0.0.0.0:7900"
role = "auto"          # auto | worker | admin
active = true          # toggle node on/off without restart

[privacy]
advertise_gpu_specs = false
advertise_bandwidth = false

[routing]
fan_out_width = 3
min_success_count = 2
ping_timeout_ms = 100
```

The node watches the config file for changes using OS-native file watch APIs. When `active` changes, the node immediately updates its `NodeCapability` record and gossips the change to peers, so it stops receiving new jobs within one gossip cycle.

Privacy fields control what the node advertises in its `NodeCapability` record. When `advertise_gpu_specs = false`, the node omits GPU model and VRAM from its advertisement.

---

## Node startup sequence

```
  noddle-node starts
        |
        v
  Load config (~/.config/noddle/config.toml)
  (create with defaults if absent)
        |
        v
  Generate or load TLS cert
  (~/.config/noddle/cert.pem, key.pem)
        |
        v
  Detect hardware
  - NVIDIA: nvidia-smi
  - AMD: /sys/class/drm/...
  - Assign role: WORKER (VRAM >= min_vram_mb) or ADMIN
        |
        v
  Scan weight directory
  (~/.local/share/noddle/weights/)
  Match .gguf files against bundled manifests
        |
        v
  Load matched model weights via CandleAdapter
  (tokenizer extracted from GGUF metadata)
        |
        v
  Load bootstrap registry
  (bootstrap-registry.json bundled in binary)
        |
        v
  Start gRPC server (listen_addr, TLS)
        |
        v
  Register self in local registry
  Gossip with bootstrap peers
        |
        v
  Start config file watcher
        |
        v
  Ready to accept jobs
        |
        v
  Background: gossip loop (~60s interval)
```

---

## Single-node fallback

If the registry contains no available peers, the router does not fail the request. Instead, it routes all layers to the local node's `CandleAdapter`.

```
  PromptRequest arrives
        |
        v
  Router queries registry for WORKER peers
        |
        v
  No peers available?
  +----YES--->  Local fallback
  |                    |
  |                    v
  |             CandleAdapter.run_layers(0..total_layers)
  |             (full model runs on this machine)
  |                    |
  NO                   v
  |             Response streamed back
  |
  v
  Normal fan-out routing
```

This means noddle works as a single-machine inference tool out of the box, before any network peers are available. It also handles network partitions gracefully — a node that loses connectivity to all peers continues to serve local requests.

---

## Known limitations

**No KV cache**

This is the most significant performance gap. Without a key-value cache, every token generation step recomputes attention over the full context from scratch. On CPU this is approximately 5 minutes per token for a 2K-token context. On NVIDIA GPU it is much faster but still slower than it should be.

Adding KV cache requires passing the cache state alongside the hidden state tensor between nodes, which adds complexity to the wire protocol and the `InferenceAdapter` trait. This is the highest-priority performance item.

**AMD GPU**

`candle-core` does not support ROCm. AMD nodes are assigned ADMIN role and handle routing and registry duties only. They do not participate in inference chains. This will be resolved when ROCm support lands upstream.

**Greedy decoding only**

Token selection is argmax over logits. Temperature, top-p, top-k, and beam search are not implemented.

**Hardcoded generation limit**

`MAX_NEW_TOKENS` is a compile-time constant. It needs to be configurable per-request.

**Hardcoded chat template**

The Llama 3 Instruct chat template is applied in `CandleAdapter`. Adding support for a new model family currently requires code changes. The intent is to move chat templates into the manifest JSON.

---

## Planned work

Items in rough priority order. None of these are implemented yet.

- **KV cache** — pass KV state in job messages; define cache tensor format in proto
- **Configurable MAX_NEW_TOKENS** — per-request field in `PromptRequest`
- **Chat template in manifest** — remove hardcoded template from adapter code
- **ROCm / AMD GPU acceleration** — blocked on upstream candle support
- **Certificate verification** — either a PKI layer or TOFU (trust-on-first-use) pinning
- **Node reputation system** — track per-node reliability, weight node selection accordingly
- **Private routing mode** — route only through nodes above a reputation threshold
- **Incentive model** — contribute compute, earn access credits; specifics TBD
- **Tauri + Svelte UI** — desktop app for non-CLI users
- **Sampling** — temperature, top-p, top-k
- **Multi-model support** — nodes advertising and running multiple models simultaneously
- **Dynamic layer partitioning** — assign layer ranges based on measured node latency, not uniform split
