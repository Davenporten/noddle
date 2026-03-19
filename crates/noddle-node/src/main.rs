mod config;
mod role;
mod services;
mod session;
mod state;
mod tls;
mod weight_discovery;

use anyhow::Result;
use noddle_proto::{
    client_service_server::ClientServiceServer,
    node_service_server::NodeServiceServer,
    NodeCapability, NodeAddress,
};
use noddle_registry::registry::Registry;
use services::{ClientServiceImpl, NodeServiceImpl};
use state::NodeState;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::info;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("noddle=info".parse()?),
        )
        .init();

    info!("noddle-node starting");

    // ── Config ────────────────────────────────────────────────────────────────
    let config = config::Config::load()?;
    info!(listen_addr = %config.node.listen_addr, "config loaded");

    // ── TLS ───────────────────────────────────────────────────────────────────
    let tls = tls::NodeTls::load_or_generate(&config::tls_dir())?;
    let tls_config = tls.server_tls_config()?;

    // ── Registry ──────────────────────────────────────────────────────────────
    let registry = {
        let bootstrap = Registry::load_bootstrap()?;
        noddle_registry::registry::SharedRegistry::new(tokio::sync::RwLock::new(bootstrap))
    };

    // ── Role assessment ───────────────────────────────────────────────────────
    let role = role::assess_role(&config.node.role);

    // ── Node identity ─────────────────────────────────────────────────────────
    // Stable node ID: persisted to disk after first run so it survives restarts.
    let node_id = load_or_create_node_id()?;
    info!(node_id = %node_id, role = ?role, "node identity established");

    // ── Register ourselves in the local registry ──────────────────────────────
    let listen_addr = config.node.listen_addr.clone();
    let (host, port) = parse_addr(&listen_addr)?;
    {
        let capability = NodeCapability {
            node_id:        node_id.clone(),
            address:        Some(NodeAddress { host, port }),
            model_ids:      vec![],  // populated once weights are loaded
            role:           role as i32,
            current_load:   0.0,
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            last_seen_ms:   chrono::Utc::now().timestamp_millis(),
            sequence:       1,
            vram_mb:        if config.privacy.advertise_gpu_specs {
                Some(role::detect_vram_mb())
            } else {
                None
            },
            gpu_model:      None,
            bandwidth_mbps: None,
        };
        let mut reg = registry.write().await;
        reg.upsert(capability);
    }

    // ── Shared state ──────────────────────────────────────────────────────────
    let state = Arc::new(NodeState::new(
        node_id.clone(),
        role,
        registry.clone(),
        &config,
    ));

    // ── Background gossip ─────────────────────────────────────────────────────
    tokio::spawn(noddle_registry::gossip::run_gossip_loop(
        registry.clone(),
        node_id.clone(),
    ));

    // ── gRPC server ───────────────────────────────────────────────────────────
    let addr = listen_addr.parse()?;
    info!(addr = %addr, "starting gRPC server");

    Server::builder()
        .tls_config(tls_config)?
        .add_service(NodeServiceServer::new(NodeServiceImpl { state: state.clone() }))
        .add_service(ClientServiceServer::new(ClientServiceImpl { state: state.clone() }))
        .serve(addr)
        .await?;

    Ok(())
}

fn load_or_create_node_id() -> Result<String> {
    let path = dirs_home().join(".config/noddle/node_id");
    if path.exists() {
        let id = std::fs::read_to_string(&path)?.trim().to_string();
        if !id.is_empty() {
            return Ok(id);
        }
    }
    let id = Uuid::new_v4().to_string();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, &id)?;
    Ok(id)
}

fn parse_addr(addr: &str) -> Result<(String, u32)> {
    let parts: Vec<&str> = addr.rsplitn(2, ':').collect();
    if parts.len() != 2 {
        anyhow::bail!("invalid listen_addr: {}", addr);
    }
    let port = parts[0].parse::<u32>()?;
    let host = parts[1].to_string();
    Ok((host, port))
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
}
