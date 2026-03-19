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

/// Detect the largest single GPU's VRAM in MB.
///
/// Current coverage:
///   - NVIDIA  — `nvidia-smi` (Linux / Windows with driver installed)
///   - AMD     — sysfs `/sys/class/drm/*/device/mem_info_vram_total` (Linux)
///
/// Not yet implemented:
///   - Apple Silicon — unified memory; Metal API or `system_profiler SPDisplaysDataType`
///     can report the GPU-accessible portion of RAM (macOS only, no sysfs).
///   - Intel Arc — sysfs at `device/tile0/gt0/addr_space/local_mem_size`; path varies
///     across driver generations, needs testing.
///   - Windows (non-NVIDIA) — DXGI adapter enumeration or WMI; no sysfs available.
///   - Vulkan (cross-platform) — `vkGetPhysicalDeviceMemoryProperties` covers all of
///     the above in one path but requires an `ash`/`vulkano` dependency.
///
/// Returns 0 if no GPU is detected, causing the node to self-assign as ADMIN.
pub fn detect_vram_mb() -> u64 {
    nvidia_vram_mb()
        .or_else(amd_vram_mb)
        .unwrap_or(0)
}

/// Query NVIDIA VRAM via `nvidia-smi`. Returns the largest single-GPU value in MB.
fn nvidia_vram_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    std::str::from_utf8(&output.stdout)
        .ok()?
        .lines()
        .filter_map(|line| line.trim().parse::<u64>().ok())
        .max()
}

/// Read AMD VRAM from sysfs. Returns the largest single-GPU value in MB.
fn amd_vram_mb() -> Option<u64> {
    let max_bytes = std::fs::read_dir("/sys/class/drm")
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path().join("device/mem_info_vram_total");
            let content = std::fs::read_to_string(path).ok()?;
            content.trim().parse::<u64>().ok()
        })
        .max()?;

    Some(max_bytes / (1024 * 1024))
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
