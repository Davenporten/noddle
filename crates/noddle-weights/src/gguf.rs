use anyhow::{anyhow, Result};
use gguf_rs::{get_gguf_container, GGUFModel};
use noddle_core::manifest::ModelManifest;
use std::path::Path;
use tracing::info;

pub fn load_model_from_gguf(manifest: &ModelManifest, weight_path: &Path) -> Result<GGUFModel> {
    info!(model_id = %manifest.model_id, path = %weight_path.display(), "loading model");

    let path = weight_path
        .to_str()
        .ok_or_else(|| anyhow!("Weight path is invalid: {}", weight_path.display()))?;

    let mut container = get_gguf_container(path)?;

    let model = container.decode()?;

    info!("GGUF version: {}", model.get_version());
    info!("Architecture: {}", model.model_family());
    info!("Parameters: {}", model.model_parameters());
    info!("File type: {}", model.file_type());
    info!("Number of tensors: {}", model.num_tensor());

    info!("model loaded");

    Ok(model)
}
