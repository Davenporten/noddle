/// Watches the config file for changes and applies them to the running node.
///
/// Uses OS-native filesystem events (`inotify` on Linux, `kqueue` on macOS,
/// `ReadDirectoryChangesW` on Windows) via the `notify` crate — no polling.
///
/// Currently reacts to changes in `node.active`. Other fields require a
/// restart to take effect (weights_dir, listen_addr, etc. are not hot-reloadable).
use crate::config::{config_path, Config};
use crate::state::NodeState;
use anyhow::Result;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::Arc;
use tracing::{info, warn};

pub fn spawn_config_watcher(state: Arc<NodeState>) {
    tokio::spawn(async move {
        if let Err(e) = watch(state).await {
            warn!(err = %e, "config watcher stopped");
        }
    });
}

async fn watch(state: Arc<NodeState>) -> Result<()> {
    let path = config_path();

    let (tx, mut rx) = tokio::sync::mpsc::channel(8);

    let mut watcher = RecommendedWatcher::new(
        move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                let _ = tx.blocking_send(event);
            }
        },
        notify::Config::default(),
    )?;

    // Watch the directory rather than the file — editors often replace files
    // atomically (write to temp, rename), which would lose a file-level watch.
    if let Some(dir) = path.parent() {
        watcher.watch(dir, RecursiveMode::NonRecursive)?;
    }

    info!(path = %path.display(), "watching config file for changes");

    while let Some(event) = rx.recv().await {
        let is_our_file = event.paths.iter().any(|p| p == &path);
        let is_write = matches!(
            event.kind,
            EventKind::Modify(_) | EventKind::Create(_)
        );

        if !is_our_file || !is_write {
            continue;
        }

        match Config::load_from(&path) {
            Err(e) => warn!(err = %e, "config file changed but could not be parsed, ignoring"),
            Ok(new_cfg) => {
                let current_active = state.is_active().await;
                if new_cfg.node.active != current_active {
                    info!(active = new_cfg.node.active, "config change detected: updating active state");
                    if let Err(e) = state.set_active(new_cfg.node.active).await {
                        warn!(err = %e, "failed to apply active state change");
                    }
                }
            }
        }
    }

    Ok(())
}
