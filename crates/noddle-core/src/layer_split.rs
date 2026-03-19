use crate::manifest::ModelManifest;
use std::ops::Range;

/// How a model's layers are divided across nodes in a chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerAssignment {
    /// The range of layers this node is responsible for.
    pub range: Range<u32>,
    /// Total number of layers in the model (for context).
    pub total_layers: u32,
}

impl LayerAssignment {
    /// Whether this assignment covers the final layers of the model.
    pub fn is_last(&self) -> bool {
        self.range.end >= self.total_layers
    }

    pub fn layer_count(&self) -> u32 {
        self.range.end - self.range.start
    }
}

/// Compute a layer assignment for a node given its available VRAM and the
/// model manifest. The assignment is the contiguous slice of layers this
/// node can hold resident given its memory constraints.
///
/// The `position` parameter (0-indexed) determines where in the layer stack
/// this node sits, allowing the router to assign non-overlapping slices to
/// a chain of nodes.
///
/// # Layer sizing strategy
/// - Compute how many layers fit in available VRAM using the manifest's
///   per-layer memory estimate.
/// - Cap at `total_layers / min_chain_length` so that a single high-VRAM node
///   doesn't claim all layers, leaving nothing for other nodes in the chain.
/// - Always assign at least 1 layer.
pub fn compute_assignment(
    manifest: &ModelManifest,
    available_vram_mb: u64,
    position: u32,
    min_chain_length: u32,
) -> LayerAssignment {
    let total = manifest.total_layers;

    // How many layers fit in available VRAM (using 512-token sequence as baseline)
    let mb_per_layer = manifest.tensor_mb_per_layer_per_512_tokens.max(0.1) as u64;
    let layers_by_vram = if available_vram_mb > 0 {
        (available_vram_mb / mb_per_layer).max(1) as u32
    } else {
        1
    };

    // Cap so no single node monopolises the model
    let max_per_node = (total / min_chain_length.max(1)).max(1);
    let slice_size = layers_by_vram.min(max_per_node).min(total);

    let start = (position * slice_size).min(total);
    let end = (start + slice_size).min(total);

    LayerAssignment { range: start..end, total_layers: total }
}

/// Divide a model evenly across `n` nodes, returning all assignments.
/// Used by the entry-point node to plan the full chain before dispatch.
pub fn split_evenly(manifest: &ModelManifest, n: u32) -> Vec<LayerAssignment> {
    let total = manifest.total_layers;
    let n = n.max(1);
    let base = total / n;
    let remainder = total % n;

    let mut assignments = Vec::with_capacity(n as usize);
    let mut start = 0u32;

    for i in 0..n {
        // Distribute remainder layers across the first `remainder` nodes
        let size = base + if i < remainder { 1 } else { 0 };
        let end = (start + size).min(total);
        assignments.push(LayerAssignment { range: start..end, total_layers: total });
        start = end;
    }

    assignments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelManifest, TokenizerKind, WeightFormat};

    fn manifest(total_layers: u32, mb_per_layer: f32) -> ModelManifest {
        ModelManifest {
            model_id: "test/model".to_string(),
            model_version: "1.0.0".to_string(),
            total_layers,
            weight_format: WeightFormat::Gguf,
            tokenizer: TokenizerKind::Llama3,
            min_vram_mb: 4096,
            tensor_mb_per_layer_per_512_tokens: mb_per_layer,
            description: String::new(),
        }
    }

    #[test]
    fn split_evenly_divides_exactly() {
        let m = manifest(32, 4.0);
        let splits = split_evenly(&m, 4);
        assert_eq!(splits.len(), 4);
        assert_eq!(splits[0].range, 0..8);
        assert_eq!(splits[1].range, 8..16);
        assert_eq!(splits[2].range, 16..24);
        assert_eq!(splits[3].range, 24..32);
    }

    #[test]
    fn split_evenly_handles_remainder() {
        let m = manifest(10, 4.0);
        let splits = split_evenly(&m, 3);
        // 10 / 3 = 3 remainder 1 → first node gets 4, others get 3
        assert_eq!(splits[0].range, 0..4);
        assert_eq!(splits[1].range, 4..7);
        assert_eq!(splits[2].range, 7..10);
        // All layers covered
        assert_eq!(splits.last().unwrap().range.end, 10);
    }

    #[test]
    fn split_evenly_single_node() {
        let m = manifest(32, 4.0);
        let splits = split_evenly(&m, 1);
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].range, 0..32);
        assert!(splits[0].is_last());
    }

    #[test]
    fn compute_assignment_respects_vram() {
        // 32 MB VRAM, 4 MB per layer → can fit 8 layers
        let m = manifest(32, 4.0);
        let assignment = compute_assignment(&m, 32, 0, 4);
        assert_eq!(assignment.layer_count(), 8);
    }

    #[test]
    fn compute_assignment_caps_per_node() {
        // 10000 MB VRAM, but min_chain_length=4 means max 8 layers per node (32/4)
        let m = manifest(32, 4.0);
        let assignment = compute_assignment(&m, 10_000, 0, 4);
        assert_eq!(assignment.layer_count(), 8);
    }

    #[test]
    fn compute_assignment_last_node_detection() {
        let m = manifest(32, 4.0);
        let last = compute_assignment(&m, 10_000, 3, 4);
        assert!(last.is_last());
    }

    #[test]
    fn compute_assignment_zero_vram_gets_one_layer() {
        let m = manifest(32, 4.0);
        let assignment = compute_assignment(&m, 0, 0, 4);
        assert_eq!(assignment.layer_count(), 1);
    }
}
