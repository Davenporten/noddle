use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Which inference backend handles this model's weight format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WeightFormat {
    /// GGUF files — loaded via llama.cpp
    Gguf,
    /// SafeTensors / HuggingFace format — loaded via Transformers backend
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
}

/// Loaded collection of all known manifests.
#[derive(Debug, Default)]
pub struct ManifestRegistry {
    manifests: HashMap<String, ModelManifest>,
}

impl ManifestRegistry {
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
