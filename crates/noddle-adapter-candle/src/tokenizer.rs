/// Tokenizer wrapper. Loads from a `tokenizer.json` file that lives alongside
/// the model weight file. Most published open model distributions include one.
use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct ModelTokenizer {
    inner: Tokenizer,
}

impl ModelTokenizer {
    /// Load a tokenizer from the `tokenizer.json` file next to the weight file.
    pub fn load(weight_path: &Path) -> Result<Self> {
        let tokenizer_path = weight_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer.json");

        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .with_context(|| format!("loading tokenizer from {:?}", tokenizer_path))?;

        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("tokenizing input")?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.inner
            .decode(token_ids, true)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("decoding token ids")
    }
}
