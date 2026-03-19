use anyhow::{Context, Result};
use noddle_proto::{NodeCapability, RegistryDiff};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub type SharedRegistry = Arc<RwLock<Registry>>;

/// In-memory store of all known nodes in the network.
///
/// Every node maintains a full copy of the registry. Entries are merged using
/// last-write-wins on the monotonic `sequence` field — a higher sequence number
/// always wins, regardless of wall-clock timestamps. Removals are soft: entries
/// are marked stale (`last_seen_ms = 0`) rather than deleted, so the removal
/// can propagate to peers.
#[derive(Debug, Default)]
pub struct Registry {
    nodes: HashMap<String, NodeCapability>,
    /// Highest sequence seen per node — prevents stale propagation from
    /// overwriting newer state.
    sequences: HashMap<String, u64>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn shared() -> SharedRegistry {
        Arc::new(RwLock::new(Self::new()))
    }

    // ── Upsert / merge ────────────────────────────────────────────────────────

    /// Apply an incoming diff using LWW on sequence number.
    /// Returns the number of entries actually changed.
    pub fn merge(&mut self, diff: &RegistryDiff) -> usize {
        let mut updated = 0;
        for node in &diff.upserted {
            let existing_seq = self.sequences.get(&node.node_id).copied().unwrap_or(0);
            if node.sequence > existing_seq {
                debug!(node_id = %node.node_id, seq = node.sequence, "registry upsert");
                self.sequences.insert(node.node_id.clone(), node.sequence);
                self.nodes.insert(node.node_id.clone(), node.clone());
                updated += 1;
            }
        }
        // Soft-remove: mark last_seen_ms = 0 so routing skips the node,
        // but keep the entry so we can propagate the removal to peers.
        for node_id in &diff.removed_ids {
            if let Some(entry) = self.nodes.get_mut(node_id) {
                warn!(node_id = %node_id, "soft-removing node from registry");
                entry.last_seen_ms = 0;
                updated += 1;
            }
        }
        updated
    }

    /// Insert or update a single node entry (used for self-registration and
    /// direct announces from peers).
    pub fn upsert(&mut self, node: NodeCapability) {
        self.sequences.insert(node.node_id.clone(), node.sequence);
        self.nodes.insert(node.node_id.clone(), node);
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    pub fn get(&self, node_id: &str) -> Option<&NodeCapability> {
        self.nodes.get(node_id)
    }

    pub fn all_nodes(&self) -> impl Iterator<Item = &NodeCapability> {
        self.nodes.values()
    }

    /// Live nodes (not soft-removed) that advertise the given model ID.
    pub fn nodes_for_model(&self, model_id: &str) -> Vec<NodeCapability> {
        self.nodes
            .values()
            .filter(|n| n.last_seen_ms > 0 && n.model_ids.iter().any(|m| m == model_id))
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // ── Diff / sync ───────────────────────────────────────────────────────────

    /// Produce a diff of all entries with `last_seen_ms > since_ms`.
    /// Used both for gossip pushes and for piggybacking on job messages.
    pub fn diff_since(&self, since_ms: i64, origin_node_id: &str) -> RegistryDiff {
        let upserted = self
            .nodes
            .values()
            .filter(|n| n.last_seen_ms > since_ms)
            .cloned()
            .collect();
        RegistryDiff {
            upserted,
            removed_ids: vec![],
            origin_ts_ms: chrono::Utc::now().timestamp_millis(),
            origin_node_id: origin_node_id.to_string(),
        }
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let nodes: Vec<_> = self.nodes.values().collect();
        let json = serde_json::to_string_pretty(&nodes)
            .context("serializing registry")?;
        std::fs::write(path, json).context("writing registry file")?;
        info!(path = %path.display(), nodes = nodes.len(), "registry saved");
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading registry file {:?}", path))?;
        let nodes: Vec<NodeCapability> = serde_json::from_str(&text)
            .context("parsing registry file")?;
        let mut registry = Self::new();
        for node in nodes {
            registry.upsert(node);
        }
        info!(path = %path.display(), nodes = registry.len(), "registry loaded");
        Ok(registry)
    }

    /// Load the bootstrap registry snapshot embedded at compile time.
    /// The snapshot is stored in `bootstrap-registry.json` at the repo root.
    pub fn load_bootstrap() -> Result<Self> {
        const BOOTSTRAP: &str = include_str!("../../../bootstrap-registry.json");
        let nodes: Vec<NodeCapability> = serde_json::from_str(BOOTSTRAP)
            .context("parsing bootstrap registry")?;
        let mut registry = Self::new();
        for node in nodes {
            registry.upsert(node);
        }
        info!(nodes = registry.len(), "bootstrap registry loaded");
        Ok(registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noddle_proto::{NodeAddress, NodeRole};

    fn make_node(id: &str, model_ids: Vec<String>, seq: u64, last_seen_ms: i64) -> NodeCapability {
        NodeCapability {
            node_id: id.to_string(),
            address: Some(NodeAddress { host: "127.0.0.1".to_string(), port: 7900 }),
            model_ids,
            role: NodeRole::Worker as i32,
            current_load: 0.0,
            client_version: "0.1.0".to_string(),
            last_seen_ms,
            sequence: seq,
            vram_mb: None,
            gpu_model: None,
            bandwidth_mbps: None,
        }
    }

    #[test]
    fn upsert_and_retrieve() {
        let mut reg = Registry::new();
        let node = make_node("node-1", vec!["model/a".into()], 1, 1000);
        reg.upsert(node.clone());
        let retrieved = reg.get("node-1").unwrap();
        assert_eq!(retrieved.node_id, "node-1");
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn merge_higher_sequence_wins() {
        let mut reg = Registry::new();
        reg.upsert(make_node("node-1", vec![], 1, 1000));

        // Higher sequence should update
        let diff = RegistryDiff {
            upserted: vec![make_node("node-1", vec!["model/a".into()], 5, 2000)],
            removed_ids: vec![],
            origin_ts_ms: 0,
            origin_node_id: "other".into(),
        };
        let changed = reg.merge(&diff);
        assert_eq!(changed, 1);
        assert_eq!(reg.get("node-1").unwrap().model_ids, vec!["model/a"]);
    }

    #[test]
    fn merge_lower_sequence_loses() {
        let mut reg = Registry::new();
        reg.upsert(make_node("node-1", vec!["model/a".into()], 10, 2000));

        // Lower sequence should NOT overwrite
        let diff = RegistryDiff {
            upserted: vec![make_node("node-1", vec![], 3, 1000)],
            removed_ids: vec![],
            origin_ts_ms: 0,
            origin_node_id: "other".into(),
        };
        let changed = reg.merge(&diff);
        assert_eq!(changed, 0);
        // model_ids should be unchanged
        assert_eq!(reg.get("node-1").unwrap().model_ids, vec!["model/a"]);
    }

    #[test]
    fn merge_same_sequence_is_no_op() {
        let mut reg = Registry::new();
        reg.upsert(make_node("node-1", vec!["model/a".into()], 5, 1000));

        let diff = RegistryDiff {
            upserted: vec![make_node("node-1", vec![], 5, 9999)],
            removed_ids: vec![],
            origin_ts_ms: 0,
            origin_node_id: "other".into(),
        };
        let changed = reg.merge(&diff);
        assert_eq!(changed, 0);
        assert_eq!(reg.get("node-1").unwrap().model_ids, vec!["model/a"]);
    }

    #[test]
    fn soft_remove_zeroes_last_seen() {
        let mut reg = Registry::new();
        reg.upsert(make_node("node-1", vec!["model/a".into()], 1, 1000));

        let diff = RegistryDiff {
            upserted: vec![],
            removed_ids: vec!["node-1".into()],
            origin_ts_ms: 0,
            origin_node_id: "other".into(),
        };
        reg.merge(&diff);

        // Entry still exists but is marked stale
        assert!(reg.get("node-1").is_some());
        assert_eq!(reg.get("node-1").unwrap().last_seen_ms, 0);
    }

    #[test]
    fn nodes_for_model_excludes_stale() {
        let mut reg = Registry::new();
        reg.upsert(make_node("live", vec!["model/a".into()], 1, 1000));
        reg.upsert(make_node("stale", vec!["model/a".into()], 2, 0)); // stale

        let results = reg.nodes_for_model("model/a");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "live");
    }

    #[test]
    fn nodes_for_model_filters_by_model_id() {
        let mut reg = Registry::new();
        reg.upsert(make_node("a", vec!["model/llama".into()], 1, 1000));
        reg.upsert(make_node("b", vec!["model/mistral".into()], 1, 1000));

        assert_eq!(reg.nodes_for_model("model/llama").len(), 1);
        assert_eq!(reg.nodes_for_model("model/mistral").len(), 1);
        assert_eq!(reg.nodes_for_model("model/other").len(), 0);
    }

    #[test]
    fn diff_since_returns_newer_entries() {
        let mut reg = Registry::new();
        reg.upsert(make_node("old", vec![], 1, 500));
        reg.upsert(make_node("new", vec![], 1, 2000));

        let diff = reg.diff_since(1000, "origin");
        assert_eq!(diff.upserted.len(), 1);
        assert_eq!(diff.upserted[0].node_id, "new");
    }

    #[test]
    fn bootstrap_loads_empty_json() {
        // bootstrap-registry.json ships as `[]` initially
        let reg = Registry::load_bootstrap().unwrap();
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn roundtrip_file_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        let mut reg = Registry::new();
        reg.upsert(make_node("node-1", vec!["model/a".into()], 1, 1000));
        reg.save_to_file(&path).unwrap();

        let loaded = Registry::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.get("node-1").is_some());
    }
}
