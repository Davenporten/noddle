use crate::tensor_io;
use crate::tokenizer::ModelTokenizer;
use crate::transformer::{self, Transformer};
use anyhow::{Context, Result};
use candle_core::Device;
use noddle_core::adapter::InferenceAdapter;
use noddle_core::manifest::{ModelManifest, WeightFormat};
use noddle_core::tensor::Tensor as WireTensor;
use std::ops::Range;
use std::path::Path;
use tracing::info;

/// Candle-based inference adapter. Loads quantized models from GGUF files
/// and executes arbitrary layer ranges — the core requirement for distributed
/// inference. Adding support for a new model architecture means implementing
/// its forward pass in `transformer.rs`; no changes needed here.
pub struct CandleAdapter {
    model:     Option<LoadedModel>,
    device:    Device,
}

struct LoadedModel {
    model_id:     String,
    transformer:  Transformer,
    tokenizer:    ModelTokenizer,
    eos_token_id: Option<u32>,
}

impl CandleAdapter {
    pub fn new() -> Self {
        // Use Metal on macOS, CUDA if available, otherwise CPU
        let device = Self::best_device();
        info!(device = ?device, "candle adapter initialised");
        Self { model: None, device }
    }

    fn best_device() -> Device {
        #[cfg(feature = "cuda")]
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
        #[cfg(feature = "metal")]
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
        Device::Cpu
    }
}

impl Default for CandleAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceAdapter for CandleAdapter {
    fn adapter_name(&self) -> &str {
        "candle"
    }

    fn load_model(&mut self, manifest: &ModelManifest, weight_path: &Path) -> Result<()> {
        info!(model_id = %manifest.model_id, path = %weight_path.display(), "loading model");

        let transformer = transformer::load_from_gguf(weight_path, &self.device)
            .context("loading transformer from GGUF")?;

        let tokenizer = ModelTokenizer::load(weight_path)
            .context("loading tokenizer")?;

        // Read EOS token ID from the GGUF metadata if present.
        let eos_token_id = {
            use candle_core::quantized::gguf_file;
            let mut f = std::fs::File::open(weight_path)?;
            let content = gguf_file::Content::read(&mut f)?;
            match content.metadata.get("tokenizer.ggml.eos_token_id") {
                Some(gguf_file::Value::U32(v)) => Some(*v),
                _ => None,
            }
        };

        self.model = Some(LoadedModel {
            model_id: manifest.model_id.clone(),
            transformer,
            tokenizer,
            eos_token_id,
        });

        info!(model_id = %manifest.model_id, "model loaded");
        Ok(())
    }

    fn unload_model(&mut self) {
        self.model = None;
    }

    fn loaded_model_id(&self) -> Option<&str> {
        self.model.as_ref().map(|m| m.model_id.as_str())
    }

    fn total_layers(&self) -> u32 {
        self.model.as_ref().map(|m| m.transformer.config.total_layers).unwrap_or(0)
    }

    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        self.model
            .as_ref()
            .context("no model loaded")?
            .tokenizer
            .encode(prompt)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.model
            .as_ref()
            .context("no model loaded")?
            .tokenizer
            .decode(tokens)
    }

    fn run_layers(
        &self,
        layer_range: Range<u32>,
        input_tensor: &WireTensor,
        tokenized_prompt: &[u32],
    ) -> Result<WireTensor> {
        let loaded = self.model.as_ref().context("no model loaded")?;

        let input = if layer_range.start == 0 {
            // First hop — input_tensor is empty; embeddings come from token IDs
            candle_core::Tensor::zeros(
                (1, 1, loaded.transformer.config.hidden_dim),
                candle_core::DType::F32,
                &self.device,
            )?
        } else {
            tensor_io::from_wire(input_tensor, &self.device)
                .context("deserialising input tensor")?
        };

        let output = loaded
            .transformer
            .run_range(layer_range, &input, tokenized_prompt)
            .context("running transformer layers")?;

        tensor_io::to_wire(&output).context("serialising output tensor")
    }

    fn estimated_output_bytes(&self, layer_range: &Range<u32>, sequence_len: usize) -> usize {
        let hidden_dim = self
            .model
            .as_ref()
            .map(|m| m.transformer.config.hidden_dim)
            .unwrap_or(4096);
        let layer_count = (layer_range.end - layer_range.start) as usize;
        // f32 = 4 bytes; shape is [seq_len, hidden_dim] per layer output
        sequence_len * hidden_dim * 4 * layer_count
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.model.as_ref().and_then(|m| m.eos_token_id)
    }

    fn apply_chat_template(&self, user_prompt: &str) -> String {
        match self.model.as_ref().map(|m| m.model_id.as_str()) {
            Some(id) if id.contains("Llama-3") && id.contains("Instruct") => {
                format!(
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    user_prompt
                )
            }
            _ => user_prompt.to_string(),
        }
    }

    fn supports_model(&self, manifest: &ModelManifest) -> bool {
        manifest.weight_format == WeightFormat::Gguf
    }
}
