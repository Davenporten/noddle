# Distributed AI Inference Network — Project Notes

## The Problem

As dependence on AI coding tools grows, power centralizes into fewer and fewer hands. Companies like Anthropic, OpenAI, and others control the infrastructure that engineers increasingly depend on. The risks:

- A provider could pull the plug entirely
- A provider could raise prices exorbitantly, cornering companies that have already reduced headcount in reliance on AI tools
- Codebases built with AI assistance may become black boxes that engineers no longer fully understand — meaning the dependency isn't just on the tool, but on the ability to maintain their own code
- Barriers to entry (compute costs, energy, proprietary training data) make meaningful competition unlikely. Many "competitors" simply wrap the original companies' APIs, creating an illusion of choice

The deeper concern: **we are betting our future on the continued altruism of a very small number of companies.**

---

## The Vision

A **compute cooperative** — a distributed AI inference network owned and operated by its users. No single company controls it. No one can pull the plug or raise prices arbitrarily. The people running the infrastructure *are* the users.

Think of it as **internet infrastructure for AI** — the way TCP/IP and DNS are nobody's property and everybody's resource.

---

## Key Concepts

### Weights vs. Training
- **Training** is the process of adjusting a model's parameters by exposing it to massive amounts of data
- **Weights** are the result — the frozen numerical snapshot of the trained model
- Open weights (e.g. Meta's LLaMA) give you the finished product but not the process. You can use and fine-tune them but can't reproduce how they were created without enormous compute
- This means *creation* of frontier models stays centralized even if *distribution* opens up

### Inference
- Inference is using a trained model to generate a response — any time a prompt is sent and a response comes back
- Computationally much cheaper than training, but expensive at scale
- The favorable decentralization target: good enough open models can already run inference on a decent laptop or MacBook today

---

## Architecture

### User Experience
The user experience mirrors Claude Code:
- User opens a terminal and starts the client
- They issue a natural language prompt (e.g. "build a web app that allows users to upload and parse files")
- The client handles the rest transparently
- The user gets a response as if talking to a centralized service

### Distributed Inference Model
Inference is split across the network by **model layers**:
- A transformer model has many layers; each node runs a subset of them
- Node A runs layers 1–10, passes the tensor output to the next nodes, which run layers 11–20, and so on
- This chain continues until the final layers are complete and the result is returned to the original prompter
- This is how projects like **Petals** approach distributed inference

### Redundancy and Resilience
Rather than a single chain (which is brittle), multiple parallel chains run simultaneously:
- When a node completes its layers, it fans out to N next nodes
- All N nodes run the same layers independently and fan out again
- Multiple redundant paths race through the network
- The **first complete response** to return wins; a cancellation signal is broadcast to kill or ignore all other in-flight paths
- This tolerates slow nodes, dropped connections, and unreliable volunteer hardware gracefully

### Node Selection
When a node fans out to the next layer, it selects from the registry using three conditions:
1. **Pingable** — the node is reachable
2. **Doesn't already have a job** — the node is idle (backpressure mechanism)
3. **Hasn't seen this job ID** — deduplication guard, prevents cycles and redundant re-entry

The routing requires that N-M nodes accept successfully (where M < N), providing a reliability floor without requiring all dispatches to succeed.

---

## The Registry

### Structure
- Every node maintains a full copy of the registry
- The registry is a list of all known nodes and their advertised capabilities

### Bootstrap
- The client repo ships with a registry snapshot current as of release
- On first install, the new node adds itself to the registry and broadcasts a propagation message to all known nodes

### Maintenance
Two complementary sync mechanisms:
- **Piggyback diffs on work requests** — registry diffs travel alongside job messages, keeping sync efficient during active periods
- **Cron job propagation** — periodic background sync to prevent registry drift during quiet periods

### Stale Entries
Nodes will go offline without warning constantly (volunteer hardware). Stale registry entries are tolerated — the ping check at job dispatch time handles dead nodes gracefully without requiring the registry to be perfectly current.

### Conflict Resolution
When two nodes join simultaneously and propagate concurrently, different parts of the network will briefly have inconsistent views. Resolution strategy: **union merge with timestamp, last write wins on conflicts.**

### Open Source
The client repo is fully open source. The registry is transparent and auditable. No one owns the bootstrap list. Once a node is on the network it has no dependency on the repo.

---

## Job Message Format
Each job message traveling through the network contains:

| Field | Purpose |
|---|---|
| Job ID | Unique identifier for deduplication |
| Layer range | Which layers this node is responsible for |
| Tensor | Numerical output from the previous layer |
| Tokenized prompt | The original prompt in model-ready form |
| Return address | Where the final result is sent |
| Hop metadata | Where in the chain this node sits |
| Timestamp | Allows slow/stale jobs to be abandoned |

The **tensor** is the bulk of the data per hop and can be large (potentially megabytes). Practical network performance is likely more bandwidth-constrained than compute-constrained.

---

## Node Capabilities & Advertising

Nodes advertise their capabilities on registration:

- Connection info (IP, port)
- Available VRAM
- GPU model or compute score
- Bandwidth estimate
- Current load / availability status
- Client version
- **Which model weights are downloaded**

### Privacy Controls
- Nodes can **opt out of advertising GPU specs and bandwidth** — privacy and spam prevention
- The registry advertises availability, not a hardware leaderboard. Routing flows through the network organically; powerful nodes should not be directly targetable
- Nodes can be **switched off at any time** — the user is a participant, not a resource being consumed
- Future enhancement: **scheduled availability** (e.g. on overnight, off during work hours)

### Model Weights
- Nodes only advertise models whose weights they have downloaded
- Job routing must respect this — a job running LLaMA cannot route through a node that only has Mistral weights
- Users choose which weights to download; the client UI encourages downloading more models to increase usefulness to the network
- The client UI shows **network-wide model support levels** so users can see which models have enough nodes for a reliable experience before choosing one for a session

---

## Routing Optimization (open question)

With capability advertising, smarter layer assignment becomes possible:

- **High VRAM / fast GPU** → computationally heavy middle layers (where attention computation is most expensive in transformer models)
- **High bandwidth / modest GPU** → early or late layers, or pass-through roles
- **Low bandwidth / low VRAM** → edge nodes, first/last layers, or registry propagation only

This mirrors how internet topology works — backbone nodes and edge nodes playing different roles based on capacity.

---

## Incentive Model

### Core incentive
**Contribute to the network, get access.** You run the client, your machine participates in the network, you get free access to the AI coding assistant. The incentive is directly tied to what the network produces — no abstract tokens, currency, or fee tiers needed. Everyone who runs the client contributes something.

### Working nodes vs. administrative nodes
Not everyone's hardware is capable of running inference layers. The client assesses the machine on install and assigns a role automatically:

- **Working nodes** — sufficient VRAM and compute to run model layers. These nodes participate directly in inference chains
- **Administrative nodes** — modest or low-spec hardware that cannot run inference meaningfully. These nodes contribute by maintaining and propagating the registry, relaying job messages, and supporting network coordination

This means **no one is excluded**. An old laptop that can't run a single model layer still has a useful role and still earns access. Contribution looks different but the principle is the same — you give what your machine can give, you get what the network produces.

### Why this works
- The incentive is intrinsic and direct — no middleman, no pricing mechanism to game or exploit
- It mirrors how the internet itself works: not every router is a backbone node, but every router contributes to the whole
- It keeps the cooperative spirit intact — there is no class of user who is purely a consumer

### Precedents
- **SETI@home / Folding@home** — millions of people voluntarily donated idle compute for causes they believed in
- **Linux / Wikipedia / Firefox** — massive cooperative efforts that "shouldn't" work by pure economic logic and are some of the most robust infrastructure we have
- **Credit unions** — member-owned, not customer-served
- **Tor network** — relay nodes contribute bandwidth at different capacities; all users benefit regardless of relay status

---

## Security

This section is early thinking, not a complete threat model. Security expertise needed before any production implementation.

### The core threat surface
Nodes are processing data that passes through them. Unlike a centralized service where you trust one operator, here you are routing sensitive data (your prompts, your code) through potentially many strangers' machines. The key threats are:

- **Tensor tampering** — a malicious node modifies the tensor it receives before passing it on, corrupting the inference result
- **Prompt interception** — a node reads the prompt or code passing through it
- **Sybil attack** — a bad actor spins up many fake nodes to gain disproportionate influence over routing or inference
- **Registry poisoning** — a malicious node propagates false registry entries to manipulate routing

### Encryption between nodes
The most straightforward mitigation is **TLS encryption on all node-to-node communication** — the same encryption used between your browser and a website. This addresses prompt interception: data in transit is unreadable to anyone observing the connection. This is well understood, widely implemented, and should be considered a baseline requirement, not an advanced feature.

Tensor tampering is harder because the node has to *decrypt* the tensor to compute on it — you can't run inference on encrypted data in any practical sense today (fully homomorphic encryption exists in theory but is nowhere near usable at this scale). So encryption protects data in transit but not data in use.

### Mitigations for data in use
- **Redundant paths** — already part of the architecture. If one malicious node tampers with a tensor, the other parallel chains should produce consistent results. A result that diverges significantly from the others could be flagged or discarded
- **Result validation heuristics** — the final result can be sanity-checked before returning to the user. A garbled or nonsensical response from one chain is a signal something went wrong
- **Node reputation** — over time, nodes that consistently produce valid results build a trust score. New or low-reputation nodes get less routing traffic until they establish a track record

### Prompt privacy
Users should be aware that their prompts pass through other people's machines. For most coding assistant use cases this is acceptable, but for sensitive codebases it may not be. Options worth exploring:
- Prompt chunking / obfuscation so no single node sees the full prompt
- An opt-in "private mode" that only routes through nodes with established high reputation scores
- Clear documentation so users understand the tradeoff they are making

### Open security questions
- Formal threat model needed from someone with distributed systems security expertise
- Sybil resistance mechanism not yet designed
- Registry integrity — how do we prevent poisoned registry entries from propagating?
- Audit logging — should nodes log what passes through them? Privacy vs. accountability tradeoff

---



- **Node spam / overload** — powerful nodes could still be disproportionately selected even without a hardware leaderboard. Need routing logic that distributes load across the available pool rather than concentrating on the fastest nodes
- **Minimum viable network size** — how many nodes / how much aggregate compute is needed for the user experience to be good enough to retain early contributors? This is the chicken-and-egg bootstrapping problem
- **Institutional anchor** — the network likely needs a university, nonprofit, or aligned organization to seed initial compute and provide credibility. Candidates: MIT, CMU, Berkeley AI labs, Hugging Face partnerships
- **Bandwidth cost on volunteers** — tensor passing between nodes consumes real bandwidth. Heavy users of the network could impose real costs on node operators. May need fairness accounting
- **Job cancellation propagation** — when the winning chain returns a result, the cancel signal needs to reach all in-flight nodes quickly. In a large network this propagation has latency. Need a strategy for nodes that finish work after cancellation (just discard? brief grace period?)
- **Model version consistency** — all nodes in a chain must be running the same version of the same model weights. What happens during a model update rollout when the network is partially upgraded?
- **Security** — see Security section below for current thinking; threat model needs formal definition
- **Client UI** — the dashboard, model availability indicator, opt-in controls, and on/off switch are essential to the feel of the product. This is not an afterthought — it's what makes someone feel like a participant rather than a resource

---

## Relevant Prior Art & Ecosystem

- **Petals** — distributed inference by layer slicing, closest existing implementation to this architecture
- **Hugging Face** — hosts open model weights, datasets, and the Transformers library. Closest institutional analog to the community hub this network needs. Venture-backed, so same tensions exist at a different layer
- **Hugging Face Transformers library** — would allow the client to load and swap models without being tied to any specific one
- **prima.cpp** — research project running 70B parameter models on everyday devices over Wi-Fi
- **Folding@home / SETI@home** — volunteer compute precedents
- **BitTorrent** — gossip protocol and deduplication patterns
- **Bitcoin** — transaction propagation with deduplication is architecturally similar to job propagation in this network


