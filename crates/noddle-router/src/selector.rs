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

#[cfg(test)]
mod tests {
    use super::*;
    use noddle_proto::{NodeAddress, NodeCapability};
    use noddle_registry::registry::Registry;

    fn make_node(node_id: &str, model_id: &str, load: f32, with_address: bool) -> NodeCapability {
        NodeCapability {
            node_id:        node_id.to_string(),
            address:        if with_address {
                Some(NodeAddress { host: "127.0.0.1".to_string(), port: 9999 })
            } else {
                None
            },
            model_ids:      vec![model_id.to_string()],
            role:           1,
            current_load:   load,
            client_version: "0.1.0".to_string(),
            last_seen_ms:   1,
            sequence:       1,
            vram_mb:        None,
            gpu_model:      None,
            bandwidth_mbps: None,
        }
    }

    #[tokio::test]
    async fn overloaded_node_not_selected() {
        let registry = Registry::shared();
        registry.write().await.upsert(make_node("n1", "m/model", 0.9, true));
        let result = select_candidates(&registry, "m/model", &HashSet::new(), 3).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn wrong_model_not_selected() {
        let registry = Registry::shared();
        registry.write().await.upsert(make_node("n1", "other/model", 0.1, true));
        let result = select_candidates(&registry, "m/model", &HashSet::new(), 3).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn excluded_node_not_selected() {
        let registry = Registry::shared();
        registry.write().await.upsert(make_node("n1", "m/model", 0.1, true));
        let excluded: HashSet<String> = ["n1".to_string()].into();
        let result = select_candidates(&registry, "m/model", &excluded, 3).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn node_without_address_not_selected() {
        let registry = Registry::shared();
        registry.write().await.upsert(make_node("n1", "m/model", 0.1, false));
        let result = select_candidates(&registry, "m/model", &HashSet::new(), 3).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn at_load_threshold_not_selected() {
        let registry = Registry::shared();
        // exactly 0.8 — the filter is strictly less-than
        registry.write().await.upsert(make_node("n1", "m/model", 0.8, true));
        let result = select_candidates(&registry, "m/model", &HashSet::new(), 3).await;
        assert!(result.is_empty());
    }
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
