use crate::registry::SharedRegistry;
use anyhow::Result;
use noddle_proto::{
    node_service_client::NodeServiceClient, RegistrySyncRequest,
};
use std::time::Duration;
use tracing::{info, warn};

const GOSSIP_INTERVAL: Duration = Duration::from_secs(60);
const GOSSIP_PEER_COUNT: usize = 3;

/// Runs in the background, periodically pushing our registry diff to a random
/// sample of peers. This is the anti-entropy mechanism for quiet periods.
/// During active job routing, diffs also travel piggybacked on job messages.
pub async fn run_gossip_loop(registry: SharedRegistry, node_id: String) {
    let mut interval = tokio::time::interval(GOSSIP_INTERVAL);
    interval.tick().await; // skip the first immediate tick
    loop {
        interval.tick().await;
        if let Err(e) = gossip_round(&registry, &node_id).await {
            warn!(error = %e, "gossip round failed");
        }
    }
}

async fn gossip_round(registry: &SharedRegistry, node_id: &str) -> Result<()> {
    // Sample GOSSIP_PEER_COUNT random peers from the registry
    let peers: Vec<_> = {
        let reg = registry.read().await;
        let all: Vec<_> = reg
            .all_nodes()
            .filter(|n| n.node_id != node_id && n.last_seen_ms > 0)
            .cloned()
            .collect();
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        all.choose_multiple(&mut rng, GOSSIP_PEER_COUNT)
            .cloned()
            .collect()
    };

    if peers.is_empty() {
        return Ok(());
    }

    let diff = {
        let reg = registry.read().await;
        reg.diff_since(0, node_id)
    };

    for peer in peers {
        let addr = format!("https://{}:{}", peer.address.as_ref().map(|a| a.host.as_str()).unwrap_or(""), peer.address.as_ref().map(|a| a.port).unwrap_or(0));
        match NodeServiceClient::connect(addr.clone()).await {
            Ok(mut client) => {
                let req = RegistrySyncRequest {
                    requester_node_id: node_id.to_string(),
                    since_ms: 0,
                    diff: Some(diff.clone()),
                };
                match client.sync_registry(req).await {
                    Ok(resp) => {
                        if let Some(their_diff) = resp.into_inner().diff {
                            let mut reg = registry.write().await;
                            let n = reg.merge(&their_diff);
                            info!(peer = %peer.node_id, merged = n, "gossip sync complete");
                        }
                    }
                    Err(e) => warn!(peer = %peer.node_id, error = %e, "gossip sync rpc failed"),
                }
            }
            Err(e) => warn!(peer = %peer.node_id, addr = %addr, error = %e, "gossip connect failed"),
        }
    }
    Ok(())
}
