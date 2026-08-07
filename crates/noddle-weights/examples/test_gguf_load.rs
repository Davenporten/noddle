use noddle_core::manifest::ModelManifest;
use noddle_weights::gguf::load_model_from_gguf;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let manifest = ModelManifest {};

    let weight_path = Path::new("");

    println!("Attempting to execute load_gguf_model...");
    let _model = load_model_from_gguf(&manifest, &weight_path)?;
    println!("Success!");

    Ok(())
}
