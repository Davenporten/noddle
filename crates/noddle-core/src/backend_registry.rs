use crate::backend::{InferenceBackend, StubBackend};
use crate::manifest::{ModelManifest, WeightFormat};
use std::collections::HashMap;

/// Maps weight formats to the backend implementation that handles them.
///
/// # Adding a new backend
/// 1. Implement `InferenceBackend`.
/// 2. Call `registry.register(WeightFormat::YourFormat, Box::new(YourBackend::new()))`.
/// 3. Add manifest JSON files with `weight_format: "your_format"`.
///    Nothing else needs to change.
pub struct BackendRegistry {
    backends: HashMap<WeightFormat, Box<dyn InferenceBackend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self { backends: HashMap::new() }
    }

    /// Register a backend for a specific weight format.
    /// If a backend for that format already exists it is replaced.
    pub fn register(&mut self, format: WeightFormat, backend: Box<dyn InferenceBackend>) {
        self.backends.insert(format, backend);
    }

    /// Build a registry with only the `StubBackend`, used in tests and
    /// environments where no real weights are available.
    pub fn stub() -> Self {
        let mut reg = Self::new();
        reg.register(WeightFormat::Gguf, Box::new(StubBackend::new()));
        reg.register(WeightFormat::Safetensors, Box::new(StubBackend::new()));
        reg
    }

    /// Return the backend capable of serving the given manifest, if any.
    pub fn backend_for(&self, manifest: &ModelManifest) -> Option<&dyn InferenceBackend> {
        self.backends
            .get(&manifest.weight_format)
            .map(|b: &Box<dyn InferenceBackend>| b.as_ref())
    }

    /// Return a mutable reference to the backend for the given manifest.
    pub fn backend_for_mut(&mut self, manifest: &ModelManifest) -> Option<&mut dyn InferenceBackend> {
        match self.backends.get_mut(&manifest.weight_format) {
            Some(b) => Some(b.as_mut()),
            None => None,
        }
    }

    /// Whether any registered backend supports the given manifest.
    pub fn can_serve(&self, manifest: &ModelManifest) -> bool {
        self.backends
            .get(&manifest.weight_format)
            .is_some_and(|b: &Box<dyn InferenceBackend>| b.supports_model(manifest))
    }

    /// All weight formats this registry can handle.
    pub fn supported_formats(&self) -> Vec<&WeightFormat> {
        self.backends.keys().collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelManifest, TokenizerKind};

    fn gguf_manifest() -> ModelManifest {
        ModelManifest {
            model_id: "test/gguf-model".to_string(),
            model_version: "1.0.0".to_string(),
            total_layers: 32,
            weight_format: WeightFormat::Gguf,
            tokenizer: TokenizerKind::Llama3,
            min_vram_mb: 4096,
            tensor_mb_per_layer_per_512_tokens: 4.0,
            description: "test".to_string(),
        }
    }

    fn safetensors_manifest() -> ModelManifest {
        ModelManifest {
            model_id: "test/st-model".to_string(),
            weight_format: WeightFormat::Safetensors,
            ..gguf_manifest()
        }
    }

    #[test]
    fn stub_registry_can_serve_all_formats() {
        let reg = BackendRegistry::stub();
        assert!(reg.can_serve(&gguf_manifest()));
        assert!(reg.can_serve(&safetensors_manifest()));
    }

    #[test]
    fn empty_registry_cannot_serve() {
        let reg = BackendRegistry::new();
        assert!(!reg.can_serve(&gguf_manifest()));
    }

    #[test]
    fn backend_for_returns_correct_backend() {
        let reg = BackendRegistry::stub();
        let backend = reg.backend_for(&gguf_manifest());
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().backend_name(), "stub");
    }

    #[test]
    fn register_replaces_existing() {
        let mut reg = BackendRegistry::new();
        reg.register(WeightFormat::Gguf, Box::new(StubBackend::new()));
        reg.register(WeightFormat::Gguf, Box::new(StubBackend::new()));
        assert_eq!(reg.supported_formats().len(), 1);
    }
}
