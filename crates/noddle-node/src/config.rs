use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level node configuration, loaded from `~/.config/noddle/config.toml`.
/// Missing fields fall back to defaults so the node works out-of-the-box
/// with no config file present.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub node: NodeConfig,
    pub privacy: PrivacyConfig,
    pub routing: RoutingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeConfig {
    /// Directory where model weight files are stored.
    pub weights_dir: PathBuf,
    /// Directory where manifest JSON files are stored.
    pub manifests_dir: PathBuf,
    /// Address the gRPC server binds to.
    pub listen_addr: String,
    /// `auto` detects role from hardware; `worker` or `admin` force it.
    pub role: RoleOverride,
    /// Whether this node is actively contributing compute to the network.
    /// When false, the node stays up for local CLI use and registry gossip
    /// but does not advertise models — other nodes will not route jobs here.
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PrivacyConfig {
    /// If false, GPU model and VRAM are omitted from NodeCapability advertisements.
    pub advertise_gpu_specs: bool,
    /// If false, bandwidth estimate is omitted from NodeCapability advertisements.
    pub advertise_bandwidth: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    /// How many nodes to dispatch to in parallel per hop.
    pub fan_out_width: usize,
    /// Minimum number that must succeed (fan_out_width - tolerance).
    pub min_success_count: usize,
    /// Milliseconds before a ping is considered failed.
    pub ping_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RoleOverride {
    #[default]
    Auto,
    Worker,
    Admin,
}

// ── Defaults ──────────────────────────────────────────────────────────────────

impl Default for Config {
    fn default() -> Self {
        Self {
            node: NodeConfig::default(),
            privacy: PrivacyConfig::default(),
            routing: RoutingConfig::default(),
        }
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        let home = dirs_home();
        Self {
            weights_dir:   home.join(".local/share/noddle/weights"),
            manifests_dir: home.join(".local/share/noddle/manifests"),
            listen_addr:   "0.0.0.0:7900".to_string(),
            role:          RoleOverride::Auto,
            active:        true,
        }
    }
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            advertise_gpu_specs: false,
            advertise_bandwidth: false,
        }
    }
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            fan_out_width:     3,
            min_success_count: 2,
            ping_timeout_ms:   100,
        }
    }
}

// ── Load / save ───────────────────────────────────────────────────────────────

impl Config {
    /// Load from the default path (`~/.config/noddle/config.toml`).
    /// If the file doesn't exist, returns and writes the default config.
    pub fn load() -> Result<Self> {
        let path = config_path();
        if !path.exists() {
            let default = Config::default();
            default.save(&path)?;
            return Ok(default);
        }
        Self::load_from(&path)
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {:?}", path))?;
        toml::from_str(&text).context("parsing config TOML")
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating config dir {:?}", parent))?;
        }
        let text = toml::to_string_pretty(self).context("serializing config")?;
        std::fs::write(path, text).context("writing config file")?;
        Ok(())
    }
}

pub fn config_path() -> PathBuf {
    dirs_home().join(".config/noddle/config.toml")
}

pub fn tls_dir() -> PathBuf {
    dirs_home().join(".config/noddle/tls")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = Config::default();
        assert_eq!(cfg.node.listen_addr, "0.0.0.0:7900");
        assert_eq!(cfg.routing.fan_out_width, 3);
        assert!(!cfg.privacy.advertise_gpu_specs);
        assert!(cfg.node.active, "nodes should be active by default");
    }

    #[test]
    fn active_false_roundtrips_toml() {
        let mut cfg = Config::default();
        cfg.node.active = false;
        let text = toml::to_string_pretty(&cfg).unwrap();
        let parsed: Config = toml::from_str(&text).unwrap();
        assert!(!parsed.node.active);
    }

    #[test]
    fn roundtrip_toml() {
        let cfg = Config::default();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let parsed: Config = toml::from_str(&text).unwrap();
        assert_eq!(parsed.node.listen_addr, cfg.node.listen_addr);
        assert_eq!(parsed.routing.fan_out_width, cfg.routing.fan_out_width);
    }
}
