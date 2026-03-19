use crate::adapter::{InferenceAdapter, StubAdapter};
use crate::manifest::{ModelManifest, WeightFormat};
use std::collections::HashMap;

/// Maps weight formats to the adapter implementation that handles them.
///
/// # Adding a new adapter
/// 1. Implement `InferenceAdapter`.
/// 2. Call `registry.register(WeightFormat::YourFormat, Box::new(YourAdapter::new()))`.
/// 3. Add manifest JSON files with `weight_format: "your_format"`.
///    Nothing else needs to change.
pub struct AdapterRegistry {
    adapters: HashMap<WeightFormat, Box<dyn InferenceAdapter>>,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self { adapters: HashMap::new() }
    }

    /// Register an adapter for a specific weight format.
    /// If an adapter for that format already exists it is replaced.
    pub fn register(&mut self, format: WeightFormat, adapter: Box<dyn InferenceAdapter>) {
        self.adapters.insert(format, adapter);
    }

    /// Build a registry with only the `StubAdapter`, used in tests and
    /// environments where no real weights are available.
    pub fn stub() -> Self {
        let mut reg = Self::new();
        reg.register(WeightFormat::Gguf, Box::new(StubAdapter::new()));
        reg.register(WeightFormat::Safetensors, Box::new(StubAdapter::new()));
        reg
    }

    /// Return the adapter capable of serving the given manifest, if any.
    pub fn adapter_for(&self, manifest: &ModelManifest) -> Option<&dyn InferenceAdapter> {
        self.adapters
            .get(&manifest.weight_format)
            .map(|a: &Box<dyn InferenceAdapter>| a.as_ref())
    }

    /// Return a mutable reference to the adapter for the given manifest.
    pub fn adapter_for_mut(&mut self, manifest: &ModelManifest) -> Option<&mut dyn InferenceAdapter> {
        match self.adapters.get_mut(&manifest.weight_format) {
            Some(a) => Some(a.as_mut()),
            None => None,
        }
    }

    /// Whether any registered adapter supports the given manifest.
    pub fn can_serve(&self, manifest: &ModelManifest) -> bool {
        self.adapters
            .get(&manifest.weight_format)
            .is_some_and(|a: &Box<dyn InferenceAdapter>| a.supports_model(manifest))
    }

    /// All weight formats this registry can handle.
    pub fn supported_formats(&self) -> Vec<&WeightFormat> {
        self.adapters.keys().collect()
    }
}

impl Default for AdapterRegistry {
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
        let reg = AdapterRegistry::stub();
        assert!(reg.can_serve(&gguf_manifest()));
        assert!(reg.can_serve(&safetensors_manifest()));
    }

    #[test]
    fn empty_registry_cannot_serve() {
        let reg = AdapterRegistry::new();
        assert!(!reg.can_serve(&gguf_manifest()));
    }

    #[test]
    fn adapter_for_returns_correct_adapter() {
        let reg = AdapterRegistry::stub();
        let adapter = reg.adapter_for(&gguf_manifest());
        assert!(adapter.is_some());
        assert_eq!(adapter.unwrap().adapter_name(), "stub");
    }

    #[test]
    fn register_replaces_existing() {
        let mut reg = AdapterRegistry::new();
        reg.register(WeightFormat::Gguf, Box::new(StubAdapter::new()));
        reg.register(WeightFormat::Gguf, Box::new(StubAdapter::new()));
        assert_eq!(reg.supported_formats().len(), 1);
    }
}
