use anyhow::Result;
use noddle_proto::{
    node_service_client::NodeServiceClient, NodeCapability, PingRequest,
};
use noddle_registry::registry::SharedRegistry;
use std::collections::HashSet;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, warn};

const PING_TIMEOUT: Duration = Duration::from_millis(100);
const MAX_LOAD: f32 = 0.8;

/// Select up to `n` candidate nodes for the next layer hop.
///
/// Criteria (in order):
/// 1. Advertises the required model
/// 2. Not in the excluded set (already-seen job IDs per node, dedup)
/// 3. Load below threshold
/// 4. Responds to a ping within PING_TIMEOUT
pub async fn select_candidates(
    registry: &SharedRegistry,
    model_id: &str,
    excluded_node_ids: &HashSet<String>,
    n: usize,
) -> Vec<NodeCapability> {
    let candidates: Vec<_> = {
        let reg = registry.read().await;
        reg.nodes_for_model(model_id)
            .into_iter()
            .filter(|node| {
                !excluded_node_ids.contains(&node.node_id)
                    && node.current_load < MAX_LOAD
                    && node.address.is_some()
            })
            .collect()
    };

    // Shuffle so we don't always hammer the same nodes
    let mut candidates = candidates;
    {
        use rand::seq::SliceRandom;
        candidates.shuffle(&mut rand::thread_rng());
    }

    let mut accepted = Vec::with_capacity(n);
    for node in candidates {
        if accepted.len() >= n {
            break;
        }
        if ping_node(&node).await.is_ok() {
            debug!(node_id = %node.node_id, "node accepted for routing");
            accepted.push(node);
        } else {
            warn!(node_id = %node.node_id, "node failed ping, skipping");
        }
    }
    accepted
}

async fn ping_node(node: &NodeCapability) -> Result<()> {
    let addr_info = node.address.as_ref().unwrap();
    let addr = format!("https://{}:{}", addr_info.host, addr_info.port);
    let mut client = timeout(PING_TIMEOUT, NodeServiceClient::connect(addr)).await??;
    timeout(
        PING_TIMEOUT,
        client.ping(PingRequest { node_id: node.node_id.clone() }),
    )
    .await??;
    Ok(())
}
