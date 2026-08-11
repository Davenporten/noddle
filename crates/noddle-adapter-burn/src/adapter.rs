// TODO: implement these
// use crate::tensor_io;
// use crate::tokenizer::ModelTokenizer;
// use crate::transformer::{self, Transformer};
use anyhow::Result;
use burn_rocm::RocmDevice;
use gguf_rs::GGUFModel;
use noddle_core::adapter::InferenceAdapter;
use noddle_core::manifest::{ModelManifest, WeightFormat};
use noddle_core::tensor::Tensor as WireTensor;
use noddle_weights::gguf::load_model_from_gguf;
use std::ops::Range;
use std::path::Path;

struct BurnModel {
    model_id: String,
    // TODO: create issue about supporting more files types beyond .gguf
    model: GGUFModel,
}

impl BurnModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model(&self) -> &GGUFModel {
        &self.model
    }
}

pub struct BurnAdapter {
    model: Option<BurnModel>,
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

        self.model = Some(BurnModel {
            model_id: manifest.model_id.clone(),
            model: model,
        });

        Ok(())
    }

    fn unload_model(&mut self) {
        self.model = None;
    }

    fn loaded_model_id(&self) -> Option<&str> {
        self.model.as_ref().map(|m| m.model_id())
    }

    fn total_layers(&self) -> u32 {
        let metadata = match &self.model {
            Some(m) => m.model().metadata(),
            None => return 0,
        };

        let to_u32 = |v: &serde_json::Value| v.as_u64().and_then(|n| u32::try_from(n).ok());

        if let Some(count) = metadata.get("general.block_count").and_then(to_u32) {
            return count;
        }

        let arch_prefix = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llm");

        let fallback_key = format!("{}.block_count", arch_prefix);

        metadata.get(&fallback_key).and_then(to_u32).unwrap_or(0)
    }

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

    fn supports_model(&self, manifest: &ModelManifest) -> bool {
        manifest.weight_format == WeightFormat::Gguf
    }
}
