use crate::config::{config_path, Config};
use crate::session::SessionStore;
use anyhow::{bail, Result};
use dashmap::DashSet;
use noddle_core::adapter::InferenceAdapter;
use noddle_proto::{JobMessage, NodeRole, PromptRequest};
use noddle_registry::registry::SharedRegistry;
use noddle_router::router::{Router, RouterConfig};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Shared state threaded through all gRPC service handlers.
pub struct NodeState {
    pub node_id:           String,
    pub role:              NodeRole,
    pub registry:          SharedRegistry,
    pub router:            Router,
    cancelled:             Arc<DashSet<String>>,
    backend:               Arc<RwLock<Box<dyn InferenceAdapter>>>,
    sessions:              SessionStore,
    load:                  Arc<std::sync::atomic::AtomicU32>, // stored as u32 * 1000
    config:                Arc<RwLock<Config>>,
    /// Model IDs discovered on disk at startup — kept so we can restore them
    /// when transitioning from inactive → active.
    pub discovered_model_ids: Vec<String>,
}

impl NodeState {
    pub fn new(
        node_id: String,
        role: NodeRole,
        registry: SharedRegistry,
        config: Config,
        adapter: Box<dyn InferenceAdapter>,
        discovered_model_ids: Vec<String>,
    ) -> Self {
        let router_cfg = RouterConfig {
            fan_out_width:    config.routing.fan_out_width,
            min_accept_count: config.routing.min_success_count,
        };
        Self {
            router:   Router::new(registry.clone(), router_cfg),
            node_id,
            role,
            registry,
            cancelled:            Arc::new(DashSet::new()),
            backend:              Arc::new(RwLock::new(adapter)),
            sessions:             SessionStore::new(),
            load:                 Arc::new(std::sync::atomic::AtomicU32::new(0)),
            config:               Arc::new(RwLock::new(config)),
            discovered_model_ids,
        }
    }

    /// Toggle whether this node advertises itself to the network.
    /// Writes the new value back to the config file so it survives restarts.
    pub async fn set_active(&self, active: bool) -> Result<()> {
        {
            let mut cfg = self.config.write().await;
            cfg.node.active = active;
            cfg.save(&config_path())?;
        }

        // Update our NodeCapability in the registry so peers see the change
        // immediately via the next gossip round.
        let model_ids = if active { self.discovered_model_ids.clone() } else { vec![] };
        {
            let mut reg = self.registry.write().await;
            if let Some(mut cap) = reg.get(&self.node_id).cloned() {
                cap.model_ids = model_ids;
                cap.sequence += 1;
                cap.last_seen_ms = chrono::Utc::now().timestamp_millis();
                reg.upsert(cap);
            }
        }

        info!(active, "node active state updated");
        Ok(())
    }

    pub async fn is_active(&self) -> bool {
        self.config.read().await.node.active
    }

    pub fn vram_mb(&self) -> u64 {
        crate::role::detect_vram_mb()
    }

    /// Mark a job as cancelled. In-flight handlers check this before
    /// forwarding or returning results.
    pub fn cancel_job(&self, job_id: &str) {
        self.cancelled.insert(job_id.to_string());
    }

    pub fn is_cancelled(&self, job_id: &str) -> bool {
        self.cancelled.contains(job_id)
    }

    /// Current load as a float 0.0–1.0 (used in ping responses and NodeCapability).
    pub fn current_load(&self) -> f32 {
        let raw = self.load.load(std::sync::atomic::Ordering::Relaxed);
        raw as f32 / 1000.0
    }

    /// Execute a job's layer range and either return the result or forward
    /// to the next hop.
    pub async fn run_job(&self, job: JobMessage) -> Result<JobMessage> {
        if self.is_cancelled(&job.job_id) {
            bail!("job {} was cancelled before execution", job.job_id);
        }

        let layer_range = job.layer_range.as_ref()
            .ok_or_else(|| anyhow::anyhow!("job missing layer_range"))?;

        // Run this node's assigned layers
        let input_tensor = noddle_core::tensor::Tensor::from_bytes(job.tensor_data.clone());
        let output_tensor = {
            let backend = self.backend.read().await;
            backend.run_layers(
                layer_range.start..layer_range.end,
                &input_tensor,
                // tokenized_prompt is the raw bytes of the token IDs (little-endian u32 array)
                &bytes_to_token_ids(&job.tokenized_prompt),
            )?
        };

        let total_layers = {
            let backend = self.backend.read().await;
            backend.total_layers()
        };

        // Check if this was the last layer range
        if layer_range.end >= total_layers {
            info!(job_id = %job.job_id, "final layer complete, returning result");
            let mut result = job.clone();
            result.tensor_data = output_tensor.into_bytes();
            return Ok(result);
        }

        // More layers remain — forward to next hop
        let next_range = noddle_proto::LayerRange {
            start: layer_range.end,
            end:   (layer_range.end + (layer_range.end - layer_range.start)).min(total_layers),
        };

        let already_visited: std::collections::HashSet<String> = job
            .hop_metadata
            .as_ref()
            .map(|m| m.path.iter().cloned().collect())
            .unwrap_or_default();

        let mut next_job = job.clone();
        next_job.tensor_data = output_tensor.into_bytes();

        self.router
            .dispatch_next_hop(next_job, next_range, &already_visited)
            .await
    }

    /// Entry point for a prompt submitted by the CLI.
    /// Tokenizes the prompt (with session history prepended if a session_id is given)
    /// and kicks off the first job hop.
    pub async fn handle_prompt(&self, req: PromptRequest) -> Result<String> {
        let backend = self.backend.read().await;

        if backend.loaded_model_id().is_none() {
            bail!("no model loaded — download weights and restart the node");
        }

        // Prepend conversation history when the caller provides a session ID.
        let full_prompt = if req.session_id.is_empty() {
            req.prompt_text.clone()
        } else {
            self.sessions
                .get_or_create(&req.session_id)
                .context_for_next_prompt(&req.prompt_text)
        };

        let tokens = backend.tokenize(&full_prompt)?;
        let total_layers = backend.total_layers();
        let slice_size = (total_layers / 3).max(1);

        drop(backend); // release read lock before async dispatch

        let job = JobMessage {
            job_id:           uuid::Uuid::new_v4().to_string(),
            model_id:         req.model_id.clone(),
            model_version:    String::new(),
            layer_range:      Some(noddle_proto::LayerRange { start: 0, end: slice_size }),
            tensor_data:      vec![],
            tokenized_prompt: token_ids_to_bytes(&tokens),
            return_address:   None,
            hop_metadata:     Some(noddle_proto::HopMetadata {
                depth: 0,
                path:  vec![self.node_id.clone()],
            }),
            timestamp_ms:     chrono::Utc::now().timestamp_millis(),
            cancel_token:     String::new(),
            registry_diff:    None,
        };

        let result = self.run_job(job).await?;
        let text = self.backend.read().await.detokenize(&bytes_to_token_ids(&result.tensor_data))?;

        if !req.session_id.is_empty() {
            self.sessions.record_turn(&req.session_id, req.prompt_text.clone(), text.clone());
        }

        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use noddle_core::adapter::StubAdapter;
    use noddle_proto::{NodeAddress, NodeCapability, NodeRole};
    use noddle_registry::registry::Registry;

    async fn make_state(active: bool) -> NodeState {
        let mut cfg = Config::default();
        cfg.node.active = active;

        let registry = Registry::shared();

        // Pre-populate our own capability so set_active can update it
        registry.write().await.upsert(NodeCapability {
            node_id:        "test-node".to_string(),
            address:        Some(NodeAddress { host: "127.0.0.1".to_string(), port: 7900 }),
            model_ids:      vec!["m/model".to_string()],
            role:           NodeRole::Worker as i32,
            current_load:   0.0,
            client_version: "0.1.0".to_string(),
            last_seen_ms:   1,
            sequence:       1,
            vram_mb:        None,
            gpu_model:      None,
            bandwidth_mbps: None,
        });

        NodeState::new(
            "test-node".to_string(),
            NodeRole::Worker,
            registry,
            cfg,
            Box::new(StubAdapter::new()),
            vec!["m/model".to_string()],
        )
    }

    #[tokio::test]
    async fn is_active_reflects_config() {
        let state = make_state(true).await;
        assert!(state.is_active().await);

        let state = make_state(false).await;
        assert!(!state.is_active().await);
    }

    #[tokio::test]
    async fn set_active_false_clears_registry_model_ids() {
        let state = make_state(true).await;
        state.set_active(false).await.unwrap();
        assert!(!state.is_active().await);

        let reg = state.registry.read().await;
        let cap = reg.get("test-node").unwrap();
        assert!(cap.model_ids.is_empty(), "inactive node should advertise no models");
    }

    #[tokio::test]
    async fn set_active_true_restores_registry_model_ids() {
        let state = make_state(false).await;
        state.set_active(true).await.unwrap();
        assert!(state.is_active().await);

        let reg = state.registry.read().await;
        let cap = reg.get("test-node").unwrap();
        assert_eq!(cap.model_ids, vec!["m/model"]);
    }

    #[tokio::test]
    async fn set_active_increments_sequence() {
        let state = make_state(true).await;
        let seq_before = state.registry.read().await.get("test-node").unwrap().sequence;
        state.set_active(false).await.unwrap();
        let seq_after = state.registry.read().await.get("test-node").unwrap().sequence;
        assert!(seq_after > seq_before, "sequence must increment for LWW propagation");
    }
}

fn bytes_to_token_ids(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn token_ids_to_bytes(tokens: &[u32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}
