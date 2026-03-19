/// Tokenizer loading.
///
/// Load order:
///   1. `tokenizer.json` alongside the weight file (HuggingFace format)
///   2. Tokenizer vocabulary embedded in the GGUF file itself
///
/// Most GGUF files embed the full tokenizer, so no separate download is needed.
/// `tokenizer.json` takes priority when present so users can override.
use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct ModelTokenizer {
    inner: Tokenizer,
}

impl ModelTokenizer {
    /// Load a tokenizer for the model at `weight_path`.
    /// Tries `tokenizer.json` first, then falls back to the vocabulary
    /// embedded inside the GGUF.
    pub fn load(weight_path: &Path) -> Result<Self> {
        let tokenizer_path = weight_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer.json");

        if tokenizer_path.exists() {
            let inner = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("{}", e))
                .with_context(|| format!("loading tokenizer from {:?}", tokenizer_path))?;
            return Ok(Self { inner });
        }

        // Fall back to the tokenizer embedded in the GGUF
        let mut file = std::fs::File::open(weight_path)
            .with_context(|| format!("opening GGUF for tokenizer: {:?}", weight_path))?;
        let content = gguf_file::Content::read(&mut file)
            .context("reading GGUF for tokenizer")?;

        let inner = tokenizer_from_gguf(&content)
            .context("extracting tokenizer from GGUF")?;

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

/// Build a tokenizer from vocabulary data embedded in a GGUF file.
///
/// Supported GGUF tokenizer types:
///   - `gpt2`  — byte-level BPE (Llama 3, Mistral, etc.)
///   - `llama` — SentencePiece BPE (Llama 2, older models)
///
/// Not yet supported:
///   - `replit` — custom SentencePiece variant
///   - `bert`   — WordPiece
fn tokenizer_from_gguf(content: &gguf_file::Content) -> Result<Tokenizer> {
    use gguf_file::Value;

    let model_type = match content.metadata.get("tokenizer.ggml.model") {
        Some(Value::String(s)) => s.as_str().to_string(),
        _ => "gpt2".to_string(),
    };

    match model_type.as_str() {
        "gpt2" | "llama" => build_bpe_tokenizer(content),
        other => anyhow::bail!("unsupported GGUF tokenizer type: {}", other),
    }
}

fn build_bpe_tokenizer(content: &gguf_file::Content) -> Result<Tokenizer> {
    use gguf_file::Value;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    // ── Vocab ─────────────────────────────────────────────────────────────────
    let tokens = match content.metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
            .collect::<Vec<_>>(),
        _ => anyhow::bail!("GGUF missing tokenizer.ggml.tokens"),
    };

    let vocab: HashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    // ── Merges ────────────────────────────────────────────────────────────────
    let merges: Vec<(String, String)> = match content.metadata.get("tokenizer.ggml.merges") {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| {
                if let Value::String(s) = v {
                    let mut parts = s.splitn(2, ' ');
                    let a = parts.next()?.to_string();
                    let b = parts.next()?.to_string();
                    Some((a, b))
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    };

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow::anyhow!("building BPE tokenizer: {}", e))?;

    let mut tokenizer = Tokenizer::new(bpe);

    // Byte-level pre-tokenizer and matching decoder handle the Ġ-prefixed tokens
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(ByteLevel::default()));

    // ── Special tokens ────────────────────────────────────────────────────────
    let bos_id = match content.metadata.get("tokenizer.ggml.bos_token_id") {
        Some(Value::U32(v)) => Some(*v),
        _ => None,
    };
    let eos_id = match content.metadata.get("tokenizer.ggml.eos_token_id") {
        Some(Value::U32(v)) => Some(*v),
        _ => None,
    };

    let mut special: Vec<tokenizers::AddedToken> = Vec::new();
    if let Some(id) = bos_id {
        if let Some(tok) = tokens.get(id as usize) {
            special.push(tokenizers::AddedToken::from(tok.as_str(), true));
        }
    }
    if let Some(id) = eos_id {
        if let Some(tok) = tokens.get(id as usize) {
            special.push(tokenizers::AddedToken::from(tok.as_str(), true));
        }
    }
    if !special.is_empty() {
        tokenizer.add_special_tokens(&special);
    }

    Ok(tokenizer)
}
