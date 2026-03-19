use crate::config::Config;
use anyhow::{bail, Result};
use dashmap::DashSet;
use noddle_core::backend::{InferenceBackend, StubBackend};
use noddle_proto::{JobMessage, NodeRole, PromptRequest};
use noddle_registry::registry::SharedRegistry;
use noddle_router::router::{Router, RouterConfig};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Shared state threaded through all gRPC service handlers.
pub struct NodeState {
    pub node_id:  String,
    pub role:     NodeRole,
    pub registry: SharedRegistry,
    pub router:   Router,
    cancelled:    Arc<DashSet<String>>,
    backend:      Arc<RwLock<Box<dyn InferenceBackend>>>,
    load:         Arc<std::sync::atomic::AtomicU32>, // stored as u32 * 1000
}

impl NodeState {
    pub fn new(
        node_id: String,
        role: NodeRole,
        registry: SharedRegistry,
        config: &Config,
    ) -> Self {
        let router_cfg = RouterConfig {
            fan_out_width:     config.routing.fan_out_width,
            min_accept_count:  config.routing.min_success_count,
        };
        Self {
            router:    Router::new(registry.clone(), router_cfg),
            node_id,
            role,
            registry,
            cancelled: Arc::new(DashSet::new()),
            backend:   Arc::new(RwLock::new(Box::new(StubBackend::new()))),
            load:      Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
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
    /// Tokenizes the prompt and kicks off the first job hop.
    pub async fn handle_prompt(&self, req: PromptRequest) -> Result<String> {
        let backend = self.backend.read().await;

        if backend.loaded_model_id().is_none() {
            bail!("no model loaded — download weights and restart the node");
        }

        let tokens = backend.tokenize(&req.prompt_text)?;
        let total_layers = backend.total_layers();
        let slice_size = (total_layers / 3).max(1); // rough equal thirds for now

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
        Ok(text)
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
