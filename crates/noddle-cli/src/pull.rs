/// `noddle pull <model_id>` — download model weights for a known model.
///
/// Reads the manifest bundled with the software (or from the local manifests
/// directory) to find the download URL, then streams the file to disk with a
/// progress bar. No manual setup required.
use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use noddle_core::manifest::{ManifestRegistry, ModelManifest, WeightFormat};
use std::path::{Path, PathBuf};

/// Manifests bundled with the binary — users get these automatically.
const BUNDLED_MANIFESTS: &[&str] = &[
    include_str!("../../../manifests/Qwen--Qwen2.5-7B-Instruct.json"),
    include_str!("../../../manifests/Qwen--Qwen2.5-Coder-7B-Instruct.json"),
    include_str!("../../../manifests/microsoft--Phi-3.5-mini-instruct.json"),
    include_str!("../../../manifests/HuggingFaceTB--SmolLM2-1.7B-Instruct.json"),
    include_str!("../../../manifests/microsoft--phi-4.json"),
    include_str!("../../../manifests/meta-llama--Llama-3.2-3B-Instruct.json"),
];

pub async fn pull(model_id: &str) -> Result<()> {
    let manifest = find_manifest(model_id)?;

    let weights_dir = weights_dir();
    std::fs::create_dir_all(&weights_dir)
        .context("creating weights directory")?;

    let filename = weight_filename(&manifest);
    let dest = weights_dir.join(&filename);

    if dest.exists() {
        println!("{} already downloaded at {}", model_id, dest.display());
        return Ok(());
    }

    println!("Pulling {} ({})...", manifest.model_id, manifest.description);
    download_file(&manifest.gguf_url, &dest).await
        .with_context(|| format!("downloading {}", manifest.gguf_url))?;

    println!("\nReady. Restart noddle-node to serve this model.");
    Ok(())
}

/// List all models available to pull (bundled + locally installed).
pub fn list_available() -> Result<()> {
    println!("Available models:\n");

    let weights_dir = weights_dir();
    let local = load_local_manifests();

    for raw in BUNDLED_MANIFESTS {
        if let Ok(m) = serde_json::from_str::<ModelManifest>(raw) {
            let filename = weight_filename(&m);
            let downloaded = weights_dir.join(&filename).exists();
            let status = if downloaded { "✓ downloaded" } else { "  not downloaded" };
            println!("  {} [{}]  {}", m.model_id, status, m.description);
        }
    }

    // Show any locally-installed manifests not in the bundled set
    let bundled_ids: Vec<String> = BUNDLED_MANIFESTS
        .iter()
        .filter_map(|raw| serde_json::from_str::<ModelManifest>(raw).ok())
        .map(|m| m.model_id)
        .collect();

    for m in local.all() {
        if !bundled_ids.contains(&m.model_id) {
            let filename = weight_filename(m);
            let downloaded = weights_dir.join(&filename).exists();
            let status = if downloaded { "✓ downloaded" } else { "  not downloaded" };
            println!("  {} [{}]  {} (local manifest)", m.model_id, status, m.description);
        }
    }

    Ok(())
}

fn find_manifest(model_id: &str) -> Result<ModelManifest> {
    // Check bundled manifests first
    for raw in BUNDLED_MANIFESTS {
        if let Ok(m) = serde_json::from_str::<ModelManifest>(raw) {
            if m.model_id == model_id {
                return Ok(m);
            }
        }
    }

    // Fall back to locally installed manifests
    let local = load_local_manifests();
    local.get(model_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!(
            "unknown model '{}'\nRun `noddle models` to see available models.",
            model_id
        ))
}

fn load_local_manifests() -> ManifestRegistry {
    let dir = manifests_dir();
    ManifestRegistry::load_dir(&dir).unwrap_or_default()
}

async fn download_file(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::new();
    let response = client.get(url).send().await?.error_for_status()?;

    let total = response.content_length();
    let bar = ProgressBar::new(total.unwrap_or(0));
    bar.set_style(
        ProgressStyle::with_template(
            "{bar:40} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
        )
        .unwrap(),
    );

    // Stream to a temp file then rename — avoids leaving a partial file on failure
    let tmp = dest.with_extension("tmp");
    let mut file = tokio::fs::File::create(&tmp)
        .await
        .with_context(|| format!("creating {:?}", tmp))?;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading download stream")?;
        bar.inc(chunk.len() as u64);
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
    }

    bar.finish();
    tokio::fs::rename(&tmp, dest).await
        .context("renaming temp file")?;

    Ok(())
}

fn weight_filename(manifest: &ModelManifest) -> String {
    let stem = manifest.model_id.replace('/', "--");
    let ext = match manifest.weight_format {
        WeightFormat::Gguf        => "gguf",
        WeightFormat::Safetensors => "safetensors",
    };
    format!("{}.{}", stem, ext)
}

fn weights_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".local/share/noddle/weights")
}

fn manifests_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".local/share/noddle/manifests")
}
