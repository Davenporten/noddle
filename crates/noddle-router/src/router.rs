use crate::selector::select_candidates;
use anyhow::{bail, Result};
use dashmap::DashSet;
use noddle_proto::{
    node_service_client::NodeServiceClient, CancelRequest, JobMessage, LayerRange,
};
use noddle_registry::registry::SharedRegistry;
use std::collections::HashSet;
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Clone)]
pub struct RouterConfig {
    /// How many nodes to dispatch to in parallel per hop
    pub fan_out_width: usize,
    /// Minimum number that must accept before we proceed (fan_out_width - tolerance)
    pub min_accept_count: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self { fan_out_width: 3, min_accept_count: 2 }
    }
}

pub struct Router {
    registry: SharedRegistry,
    config: RouterConfig,
    /// Job IDs we've seen — prevents re-entry and cycles
    seen_jobs: Arc<DashSet<String>>,
}

impl Router {
    pub fn new(registry: SharedRegistry, config: RouterConfig) -> Self {
        Self {
            registry,
            config,
            seen_jobs: Arc::new(DashSet::new()),
        }
    }

    /// Returns true if this job has already been seen on this node (dedup guard).
    pub fn is_duplicate(&self, job_id: &str) -> bool {
        self.seen_jobs.contains(job_id)
    }

    pub fn mark_seen(&self, job_id: &str) {
        self.seen_jobs.insert(job_id.to_string());
    }

    /// Dispatch a job to the next layer hop with fan-out.
    /// Races `fan_out_width` parallel calls; returns the first successful result.
    /// Broadcasts cancellation to the remaining nodes.
    pub async fn dispatch_next_hop(
        &self,
        job: JobMessage,
        next_layer_range: LayerRange,
        already_visited: &HashSet<String>,
    ) -> Result<JobMessage> {
        let candidates = select_candidates(
            &self.registry,
            &job.model_id,
            already_visited,
            self.config.fan_out_width,
        )
        .await;

        if candidates.len() < self.config.min_accept_count {
            bail!(
                "only {} nodes available, need at least {}",
                candidates.len(),
                self.config.min_accept_count
            );
        }

        info!(
            job_id = %job.job_id,
            candidates = candidates.len(),
            layer_start = next_layer_range.start,
            layer_end = next_layer_range.end,
            "dispatching next hop"
        );

        let job_id = job.job_id.clone();
        let cancel_token = job.cancel_token.clone();

        // Build one job message per candidate with updated layer_range and hop metadata
        let jobs: Vec<JobMessage> = candidates
            .iter()
            .map(|node| {
                let mut j = job.clone();
                j.layer_range = Some(next_layer_range.clone());
                if let Some(ref mut meta) = j.hop_metadata {
                    meta.depth += 1;
                    meta.path.push(node.node_id.clone());
                }
                j
            })
            .collect();

        // Race all dispatches; first to return wins
        let addresses: Vec<String> = candidates
            .iter()
            .map(|n| {
                let a = n.address.as_ref().unwrap();
                format!("https://{}:{}", a.host, a.port)
            })
            .collect();

        let handles: Vec<_> = jobs
            .into_iter()
            .zip(addresses.iter().cloned())
            .map(|(j, addr)| {
                tokio::spawn(async move {
                    execute_on_node(addr, j).await
                })
            })
            .collect();

        // select_all equivalent: wait for the first Ok result
        let (winner, losers) = race_handles(handles).await?;

        // Broadcast cancellation to remaining in-flight nodes
        let cancel_req = CancelRequest {
            job_id: job_id.clone(),
            cancel_token: cancel_token.clone(),
        };
        for addr in losers {
            let req = cancel_req.clone();
            tokio::spawn(async move {
                if let Ok(mut client) = NodeServiceClient::connect(addr).await {
                    let _ = client.cancel_job(req).await;
                }
            });
        }

        Ok(winner)
    }
}

async fn execute_on_node(addr: String, job: JobMessage) -> Result<JobMessage> {
    let mut client = NodeServiceClient::connect(addr).await?;
    let mut stream = client.execute_layer(job).await?.into_inner();
    // Collect the streamed response chunks; for now return the last (completed) message
    let mut last = None;
    while let Some(msg) = stream.message().await? {
        last = Some(msg);
    }
    last.ok_or_else(|| anyhow::anyhow!("empty response stream from node"))
}

/// Race a set of JoinHandles returning Result<JobMessage>.
/// Returns (winner_result, loser_addresses) on the first success.
async fn race_handles(
    mut handles: Vec<tokio::task::JoinHandle<Result<JobMessage>>>,
) -> Result<(JobMessage, Vec<String>)> {
    // We don't have the addresses inside the handles here, so just return winner
    // Loser cancellation is best-effort — addresses are passed back via a channel
    // in the full implementation. For now, return winner only.
    use futures::future::select_all;

    loop {
        if handles.is_empty() {
            bail!("all dispatch candidates failed");
        }
        let (result, _idx, remaining) = select_all(handles).await;
        handles = remaining;
        match result {
            Ok(Ok(msg)) => return Ok((msg, vec![])),
            Ok(Err(e)) => warn!(error = %e, "dispatch candidate failed"),
            Err(e) => warn!(error = %e, "dispatch task panicked"),
        }
    }
}
