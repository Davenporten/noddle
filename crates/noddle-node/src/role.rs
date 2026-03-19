use crate::config::RoleOverride;
use noddle_proto::NodeRole;
use tracing::info;

/// Minimum VRAM in MB to qualify as a WORKER node.
/// Nodes below this threshold become ADMIN nodes (registry relay only).
const WORKER_MIN_VRAM_MB: u64 = 3_000;

/// Assess the local hardware and determine this node's role.
///
/// - `WORKER`  — enough VRAM to run inference layers
/// - `ADMIN`   — registry relay and job coordination only
pub fn assess_role(override_: &RoleOverride) -> NodeRole {
    assess_role_with_vram(override_, detect_vram_mb())
}

fn assess_role_with_vram(override_: &RoleOverride, vram_mb: u64) -> NodeRole {
    match override_ {
        RoleOverride::Worker => {
            info!("role forced to WORKER by config");
            NodeRole::Worker
        }
        RoleOverride::Admin => {
            info!("role forced to ADMIN by config");
            NodeRole::Admin
        }
        RoleOverride::Auto => {
            let role = if vram_mb >= WORKER_MIN_VRAM_MB {
                NodeRole::Worker
            } else {
                NodeRole::Admin
            };
            info!(vram_mb = vram_mb, role = ?role, "auto-detected node role");
            role
        }
    }
}

/// Detect available VRAM in MB.
/// Returns 0 until GPU detection is implemented, causing the node to self-assign as ADMIN.
pub fn detect_vram_mb() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_worker_role() {
        let role = assess_role(&RoleOverride::Worker);
        assert_eq!(role, NodeRole::Worker);
    }

    #[test]
    fn forced_admin_role() {
        let role = assess_role(&RoleOverride::Admin);
        assert_eq!(role, NodeRole::Admin);
    }

    #[test]
    fn auto_role_with_sufficient_vram() {
        let role = assess_role_with_vram(&RoleOverride::Auto, 8000);
        assert_eq!(role, NodeRole::Worker);
    }

    #[test]
    fn auto_role_with_insufficient_vram() {
        let role = assess_role_with_vram(&RoleOverride::Auto, 0);
        assert_eq!(role, NodeRole::Admin);
    }
}
