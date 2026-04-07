use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Which inference backend handles this model's weight format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum WeightFormat {
    /// GGUF weight files
    Gguf,
    /// SafeTensors weight files
    Safetensors,
}

/// Identifies which tokenizer implementation to use.
/// New tokenizer families are added here; nothing else needs to change.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    Llama3,
    Mistral,
    Gemma,
    Phi3,
    /// Generic BPE via a bundled tokenizer.json
    BpeJson,
}

/// Per-model static metadata shipped with the client.
/// Adding support for a new model = write a JSON file matching this struct.
/// No code changes required.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Canonical ID used throughout the system (e.g. "meta-llama/Llama-3-8B")
    pub model_id: String,
    /// Semver-style version string for the specific weights release
    pub model_version: String,
    pub total_layers: u32,
    pub weight_format: WeightFormat,
    pub tokenizer: TokenizerKind,
    /// Minimum VRAM in MB to run at least one layer slice
    pub min_vram_mb: u64,
    /// Approximate MB of tensor data produced per layer per 512-token sequence
    pub tensor_mb_per_layer_per_512_tokens: f32,
    /// Human-readable description shown in the UI
    pub description: String,
    /// Direct download URL for the weight file (GGUF, safetensors, etc.)
    pub gguf_url: String,
    // TODO: add `chat_template: Option<ChatTemplate>` here so each manifest
    // carries its own prompt-formatting rules rather than hard-coding them in
    // the adapter.  ChatTemplate should be an enum (Llama3Instruct, Mistral,
    // ChatML, etc.) or a raw Jinja2 string read from the GGUF metadata.
}

/// Loaded collection of all known manifests.
#[derive(Debug, Default)]
pub struct ManifestRegistry {
    manifests: HashMap<String, ModelManifest>,
}

impl ManifestRegistry {
    /// Seed the registry from a slice of raw JSON strings (bundled via `include_str!`).
    /// On parse failure the bad entry is skipped with a warning rather than aborting.
    pub fn from_bundled(raws: &[&str]) -> Self {
        let mut registry = Self::default();
        for raw in raws {
            match serde_json::from_str::<ModelManifest>(raw) {
                Ok(m) => { registry.manifests.insert(m.model_id.clone(), m); }
                Err(e) => eprintln!("warn: failed to parse bundled manifest: {}", e),
            }
        }
        registry
    }

    /// Merge manifests from a directory into an existing registry.
    /// Disk entries override bundled ones so users can customise locally.
    pub fn merge_dir(&mut self, dir: &Path) {
        if !dir.exists() { return; }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    match Self::load_file(&path) {
                        Ok(m) => { self.manifests.insert(m.model_id.clone(), m); }
                        Err(e) => eprintln!("warn: failed to load manifest {:?}: {}", path, e),
                    }
                }
            }
        }
    }

    /// Load all `*.json` manifest files from a directory.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let mut registry = Self::default();
        if !dir.exists() {
            return Ok(registry);
        }
        for entry in std::fs::read_dir(dir).context("reading manifests dir")? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let manifest = Self::load_file(&path)?;
                registry.manifests.insert(manifest.model_id.clone(), manifest);
            }
        }
        Ok(registry)
    }

    pub fn load_file(path: &Path) -> Result<ModelManifest> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading manifest {:?}", path))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing manifest {:?}", path))
    }

    pub fn get(&self, model_id: &str) -> Option<&ModelManifest> {
        self.manifests.get(model_id)
    }

    pub fn all(&self) -> impl Iterator<Item = &ModelManifest> {
        self.manifests.values()
    }

    pub fn model_ids(&self) -> Vec<String> {
        self.manifests.keys().cloned().collect()
    }
}
