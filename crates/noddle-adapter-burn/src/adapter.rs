// TODO: implement these
// use crate::tensor_io;
// use crate::tokenizer::ModelTokenizer;
// use crate::transformer::{self, Transformer};
use anyhow::Result;
use burn_rocm::RocmDevice;
use gguf_rs::GGUFModel;
use noddle_core::adapter::InferenceAdapter;
use noddle_core::manifest::ModelManifest;
use noddle_core::tensor::Tensor as WireTensor;
use noddle_weights::gguf::load_model_from_gguf;
use std::ops::Range;
use std::path::Path;
use tracing::info;

pub struct BurnAdapter {
    model: Option<GGUFModel>,
}

impl BurnAdapter {
    pub fn new() -> Self {
        Self { model: None }
    }
}

impl Default for BurnAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceAdapter for BurnAdapter {
    fn adapter_name(&self) -> &str {
        "burn"
    }

    fn load_model(&mut self, manifest: &ModelManifest, weight_path: &Path) -> Result<()> {
        let model = load_model_from_gguf(manifest, weight_path)?;

        self.model = Some(model);

        Ok(())
    }

    fn unload_model(&mut self) {
        self.model = None;
    }

    fn loaded_model_id(&self) -> Option<&str> {}

    fn total_layers(&self) -> u32 {}

    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {}

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {}

    fn run_layers(
        &self,
        layer_range: Range<u32>,
        input_tensor: &WireTensor,
        tokenized_prompt: &[u32],
    ) -> Result<Tensor> {
    }

    fn estimated_output_bytes(&self, layer_range: &Range<u32>, sequence_len: usize) -> usize {}

    fn eos_token_id(&self) -> Option<u32> {
        None
    }

    fn apply_chat_template(&self, user_prompt: &str) -> String {
        user_prompt.to_string()
    }

    fn supports_model(&self, manifest: &ModelManifest) -> bool {}
}
