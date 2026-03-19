use anyhow::Result;
use noddle_core::manifest::{ManifestRegistry, WeightFormat};
use std::path::Path;
use tracing::{info, warn};

/// A model whose weights are present on disk and ready to serve.
#[derive(Debug, Clone)]
pub struct AvailableModel {
    pub model_id: String,
    pub weight_path: std::path::PathBuf,
}

/// Scan `weights_dir` for weight files and match them against known manifests.
/// Only models with both a manifest and a weight file are considered available.
///
/// Expected layout:
/// ```
/// weights_dir/
///   meta-llama--Llama-3-8B.gguf
///   mistralai--Mistral-7B-v0.1.safetensors
/// ```
/// The filename stem is the model ID with `/` replaced by `--`.
pub fn discover(weights_dir: &Path, manifests: &ManifestRegistry) -> Result<Vec<AvailableModel>> {
    if !weights_dir.exists() {
        return Ok(vec![]);
    }

    let mut available = Vec::new();

    for entry in std::fs::read_dir(weights_dir)? {
        let entry = entry?;
        let path = entry.path();

        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };

        let expected_format = match ext {
            "gguf"         => WeightFormat::Gguf,
            "safetensors"  => WeightFormat::Safetensors,
            _ => {
                warn!(path = %path.display(), "unrecognised weight file extension, skipping");
                continue;
            }
        };

        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };

        // Reverse the `--` → `/` substitution to recover the model ID
        let model_id = stem.replace("--", "/");

        match manifests.get(&model_id) {
            None => {
                warn!(model_id = %model_id, "weight file found but no manifest, skipping");
            }
            Some(manifest) if manifest.weight_format != expected_format => {
                warn!(
                    model_id = %model_id,
                    expected = ?manifest.weight_format,
                    found = ?expected_format,
                    "weight file format mismatch, skipping"
                );
            }
            Some(_) => {
                info!(model_id = %model_id, path = %path.display(), "weight file discovered");
                available.push(AvailableModel {
                    model_id,
                    weight_path: path,
                });
            }
        }
    }

    Ok(available)
}

/// Convert a model ID to the expected weight filename stem.
/// `meta-llama/Llama-3-8B` → `meta-llama--Llama-3-8B`
pub fn model_id_to_filename(model_id: &str, format: &WeightFormat) -> String {
    let stem = model_id.replace("/", "--");
    let ext = match format {
        WeightFormat::Gguf        => "gguf",
        WeightFormat::Safetensors => "safetensors",
    };
    format!("{}.{}", stem, ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_to_filename_replaces_slash() {
        assert_eq!(
            model_id_to_filename("meta-llama/Llama-3-8B", &WeightFormat::Gguf),
            "meta-llama--Llama-3-8B.gguf"
        );
        assert_eq!(
            model_id_to_filename("mistralai/Mistral-7B", &WeightFormat::Safetensors),
            "mistralai--Mistral-7B.safetensors"
        );
    }

    #[test]
    fn discover_empty_dir_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let manifests = ManifestRegistry::default();
        let result = discover(dir.path(), &manifests).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn discover_nonexistent_dir_returns_empty() {
        let manifests = ManifestRegistry::default();
        let result = discover(Path::new("/nonexistent/path"), &manifests).unwrap();
        assert!(result.is_empty());
    }
}
