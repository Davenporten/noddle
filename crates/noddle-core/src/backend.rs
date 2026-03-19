use crate::manifest::ModelManifest;
use crate::tensor::Tensor;
use anyhow::Result;
use std::ops::Range;
use std::path::Path;

/// The single abstraction boundary between the distributed routing system
/// and any specific inference implementation.
///
/// # Adding a new model backend
/// 1. Implement this trait.
/// 2. Register the implementation in the node daemon's backend selector.
/// 3. Add a model manifest JSON file — no other code changes required.
pub trait InferenceBackend: Send + Sync {
    /// Human-readable name for logging/debug (e.g. "llama-cpp", "transformers")
    fn backend_name(&self) -> &str;

    /// Load model weights from disk for the given manifest.
    /// Called once at startup or on model switch.
    fn load_model(&mut self, manifest: &ModelManifest, weight_path: &Path) -> Result<()>;

    /// Unload current weights, freeing VRAM / RAM.
    fn unload_model(&mut self);

    /// Which model is currently loaded, if any.
    fn loaded_model_id(&self) -> Option<&str>;

    /// Total number of transformer layers in the currently loaded model.
    fn total_layers(&self) -> u32;

    /// Convert a text prompt into token IDs using this model's tokenizer.
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>>;

    /// Convert token IDs back to a text string.
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;

    /// Run inference for the specified layer range.
    ///
    /// - `layer_range`: which layers to execute (start inclusive, end exclusive)
    /// - `input_tensor`: output of the previous layer range (empty on first hop —
    ///   the backend derives embeddings from `tokenized_prompt` in that case)
    /// - `tokenized_prompt`: raw token IDs needed for embedding lookup on hop 0
    ///
    /// Returns the output tensor in safetensors wire format.
    fn run_layers(
        &self,
        layer_range: Range<u32>,
        input_tensor: &Tensor,
        tokenized_prompt: &[u32],
    ) -> Result<Tensor>;

    /// Approximate output tensor size in bytes for capacity planning.
    /// Used by the router to decide layer splits without running inference.
    fn estimated_output_bytes(&self, layer_range: &Range<u32>, sequence_len: usize) -> usize;

    /// Whether this backend can serve the given model ID at all
    /// (i.e. its weight format is supported).
    fn supports_model(&self, manifest: &ModelManifest) -> bool;
}

/// Stub backend used in tests and when no weights are available.
/// Returns minimal valid tensors. Lets the router and registry be exercised
/// without a real inference engine.
pub struct StubBackend {
    model_id: Option<String>,
    total_layers: u32,
}

impl StubBackend {
    pub fn new() -> Self {
        Self { model_id: None, total_layers: 0 }
    }

    /// Build a minimal valid safetensors blob for use in tests.
    pub fn stub_tensor() -> Tensor {
        let header = b"{}";
        let header_len = (header.len() as u64).to_le_bytes();
        let mut bytes = Vec::with_capacity(8 + header.len() + 1);
        bytes.extend_from_slice(&header_len);
        bytes.extend_from_slice(header);
        bytes.push(0u8);
        Tensor::from_bytes(bytes)
    }
}

impl Default for StubBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for StubBackend {
    fn backend_name(&self) -> &str { "stub" }

    fn load_model(&mut self, manifest: &ModelManifest, _weight_path: &Path) -> Result<()> {
        self.model_id = Some(manifest.model_id.clone());
        self.total_layers = manifest.total_layers;
        Ok(())
    }

    fn unload_model(&mut self) {
        self.model_id = None;
        self.total_layers = 0;
    }

    fn loaded_model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    fn total_layers(&self) -> u32 { self.total_layers }

    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        // Stub: one token per whitespace-separated word
        Ok(prompt.split_whitespace().enumerate().map(|(i, _)| i as u32).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        Ok(tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "))
    }

    fn run_layers(
        &self,
        _layer_range: Range<u32>,
        _input_tensor: &Tensor,
        _tokenized_prompt: &[u32],
    ) -> Result<Tensor> {
        Ok(Self::stub_tensor())
    }

    fn estimated_output_bytes(&self, _layer_range: &Range<u32>, _sequence_len: usize) -> usize {
        1024
    }

    fn supports_model(&self, _manifest: &ModelManifest) -> bool { true }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelManifest, TokenizerKind, WeightFormat};

    fn test_manifest() -> ModelManifest {
        ModelManifest {
            model_id: "test/model".to_string(),
            model_version: "1.0.0".to_string(),
            total_layers: 32,
            weight_format: WeightFormat::Gguf,
            tokenizer: TokenizerKind::Llama3,
            min_vram_mb: 4096,
            tensor_mb_per_layer_per_512_tokens: 4.0,
            description: "test model".to_string(),
        }
    }

    #[test]
    fn stub_starts_with_no_model() {
        let backend = StubBackend::new();
        assert!(backend.loaded_model_id().is_none());
        assert_eq!(backend.total_layers(), 0);
    }

    #[test]
    fn stub_load_model_sets_state() {
        let mut backend = StubBackend::new();
        backend.load_model(&test_manifest(), Path::new("/fake/path")).unwrap();
        assert_eq!(backend.loaded_model_id(), Some("test/model"));
        assert_eq!(backend.total_layers(), 32);
    }

    #[test]
    fn stub_unload_clears_state() {
        let mut backend = StubBackend::new();
        backend.load_model(&test_manifest(), Path::new("/fake/path")).unwrap();
        backend.unload_model();
        assert!(backend.loaded_model_id().is_none());
        assert_eq!(backend.total_layers(), 0);
    }

    #[test]
    fn stub_run_layers_returns_valid_tensor() {
        let mut backend = StubBackend::new();
        backend.load_model(&test_manifest(), Path::new("/fake/path")).unwrap();
        let input = Tensor::default();
        let output = backend.run_layers(0..16, &input, &[1, 2, 3]).unwrap();
        assert!(output.validate().is_ok(), "stub output tensor should be valid safetensors");
    }

    #[test]
    fn stub_tokenize_returns_one_token_per_word() {
        let backend = StubBackend::new();
        let tokens = backend.tokenize("hello world foo").unwrap();
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn stub_tokenize_empty_string() {
        let backend = StubBackend::new();
        let tokens = backend.tokenize("").unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn stub_supports_any_model() {
        let backend = StubBackend::new();
        assert!(backend.supports_model(&test_manifest()));
    }
}
