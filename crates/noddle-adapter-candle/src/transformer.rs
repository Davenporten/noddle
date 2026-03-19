/// Transformer model with layer-range execution.
///
/// Implements a Llama-compatible decoder architecture using candle primitives.
/// Since we control the forward pass directly, we can stop and resume at any
/// layer boundary — which is the core requirement for distributed inference.
///
/// All open model families that use this architecture (Llama, Mistral, Phi-3,
/// Gemma, etc.) can be loaded here; they differ only in hyperparameters and
/// weight names, both of which come from the GGUF file.
use anyhow::{Context, Result};
use candle_core::{quantized::QMatMul, DType, Device, Module, Tensor};
use candle_nn::RmsNorm;
use std::ops::Range;

// ── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub hidden_dim:    usize,
    pub num_heads:     usize,
    pub num_kv_heads:  usize,
    pub ffn_dim:       usize,
    pub total_layers:  u32,
    pub rope_theta:    f32,
    pub vocab_size:    usize,
}

impl TransformerConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }
}

// ── Building blocks ────────────────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, theta: f32, device: &Device) -> Result<Self> {
        let theta_vec: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let theta_t = Tensor::new(theta_vec.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let pos_t = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;

        let freqs = pos_t.broadcast_mul(&theta_t.unsqueeze(0)?)?;
        // Duplicate to full head_dim so rotate_half can broadcast against
        // the full [batch, heads, seq, head_dim] query/key tensors.
        let cos_half = freqs.cos()?;
        let sin_half = freqs.sin()?;
        let cos = Tensor::cat(&[&cos_half, &cos_half], 1)?;
        let sin = Tensor::cat(&[&sin_half, &sin_half], 1)?;

        Ok(Self { sin, cos })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        Ok((rotate_half(q, &cos, &sin)?, rotate_half(k, &cos, &sin)?))
    }
}

fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, _, head_dim) = x.dims4()?;
    let half = head_dim / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
}

// ── Attention ─────────────────────────────────────────────────────────────

struct Attention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o: QMatMul,
    num_heads:    usize,
    num_kv_heads: usize,
    head_dim:     usize,
    rotary:       RotaryEmbedding,
}

impl Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self.q.forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.k.forward(x)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.v.forward(x)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary.apply(&q, &k, seq_len)?;

        // Expand KV heads to match Q heads if using grouped-query attention
        let k = if self.num_kv_heads != self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            k.repeat((1, repeat, 1, 1))?
        } else {
            k
        };
        let v = if self.num_kv_heads != self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            v.repeat((1, repeat, 1, 1))?
        } else {
            v
        };

        let scale = (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

        // Causal mask: upper triangle = -inf
        let mask = causal_mask(seq_len, x.device())?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;

        let out = attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        Ok(self.o.forward(&out)?)
    }
}

fn causal_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..size)
        .flat_map(|i| (0..size).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Ok(Tensor::from_vec(mask, (size, size), device)?)
}

// ── MLP ───────────────────────────────────────────────────────────────────

struct Mlp {
    gate: QMatMul,
    up:   QMatMul,
    down: QMatMul,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        let up   = self.up.forward(x)?;
        Ok(self.down.forward(&(gate * up)?)?)
    }
}

// ── Transformer layer ─────────────────────────────────────────────────────

pub struct TransformerLayer {
    attn_norm: RmsNorm,
    attention: Attention,
    ffn_norm:  RmsNorm,
    mlp:       Mlp,
}

impl TransformerLayer {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.attn_norm.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }
}

// ── Full model ────────────────────────────────────────────────────────────

pub struct Transformer {
    pub config:    TransformerConfig,
    embedding:     candle_nn::Embedding,
    pub layers:    Vec<TransformerLayer>,
    final_norm:    RmsNorm,
    lm_head:       QMatMul,
    device:        Device,
}

impl Transformer {
    /// Run layers in the given range over the input.
    ///
    /// - First hop (`layer_range.start == 0`, `input` is empty): embeds tokens first.
    /// - Middle hops: runs the specified layers on the input hidden state.
    /// - Last hop (`layer_range.end == total_layers`): applies final norm and
    ///   language model head to produce logits, then greedily selects the next token.
    pub fn run_range(
        &self,
        layer_range: Range<u32>,
        input: &Tensor,
        token_ids: &[u32],
    ) -> Result<Tensor> {
        let is_first = layer_range.start == 0;
        let is_last  = layer_range.end >= self.config.total_layers;

        let mut hidden = if is_first {
            let ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
            self.embedding.forward(&ids)?
        } else {
            input.clone()
        };

        for idx in layer_range.start..layer_range.end {
            hidden = self.layers[idx as usize].forward(&hidden)?;
        }

        if is_last {
            let hidden = self.final_norm.forward(&hidden)?;
            // Take the last token position and project to vocabulary
            let seq_len = hidden.dim(1)?;
            let last = hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
            Ok(self.lm_head.forward(&last)?)
        } else {
            Ok(hidden)
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ── GGUF loader ───────────────────────────────────────────────────────────

/// Load a quantized transformer from a GGUF file.
/// Reads hyperparameters from the file's metadata and loads each layer's
/// weights by name. Works for any model that follows the standard GGUF
/// tensor naming convention (blk.N.attn_q.weight, etc.).
pub fn load_from_gguf(path: &std::path::Path, device: &Device) -> Result<Transformer> {
    use candle_core::quantized::gguf_file;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening GGUF file {:?}", path))?;
    let content = gguf_file::Content::read(&mut file)
        .context("reading GGUF content")?;

    let config = config_from_gguf(&content)?;
    let max_seq_len = 4096usize;

    // ── Embedding ──────────────────────────────────────────────────────────
    let embed_weight = content
        .tensor(&mut file, "token_embd.weight", device)
        .context("loading token_embd.weight")?
        .dequantize(device)?;
    let embedding = candle_nn::Embedding::new(embed_weight, config.hidden_dim);

    // ── Transformer layers ─────────────────────────────────────────────────
    let mut layers = Vec::with_capacity(config.total_layers as usize);
    for i in 0..config.total_layers as usize {
        let rotary = RotaryEmbedding::new(config.head_dim(), max_seq_len, config.rope_theta, device)?;

        let attn_norm = rms_norm_from_gguf(&content, &mut file, &format!("blk.{}.attn_norm.weight", i), device)?;
        let ffn_norm  = rms_norm_from_gguf(&content, &mut file, &format!("blk.{}.ffn_norm.weight", i),  device)?;

        let attention = Attention {
            q: qmatmul_from_gguf(&content, &mut file, &format!("blk.{}.attn_q.weight", i),      device)?,
            k: qmatmul_from_gguf(&content, &mut file, &format!("blk.{}.attn_k.weight", i),      device)?,
            v: qmatmul_from_gguf(&content, &mut file, &format!("blk.{}.attn_v.weight", i),      device)?,
            o: qmatmul_from_gguf(&content, &mut file, &format!("blk.{}.attn_output.weight", i), device)?,
            num_heads:    config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim:     config.head_dim(),
            rotary,
        };

        let mlp = Mlp {
            gate: qmatmul_from_gguf(&content, &mut file, &format!("blk.{}.ffn_gate.weight", i), device)?,
            up:   qmatmul_from_gguf(&content, &mut file, &format!("blk.{}.ffn_up.weight", i),   device)?,
            down: qmatmul_from_gguf(&content, &mut file, &format!("blk.{}.ffn_down.weight", i), device)?,
        };

        layers.push(TransformerLayer { attn_norm, attention, ffn_norm, mlp });
    }

    // ── Final norm + LM head ───────────────────────────────────────────────
    let final_norm = rms_norm_from_gguf(&content, &mut file, "output_norm.weight", device)?;
    // Some models (e.g. Llama 3.2) use weight tying: the LM head is the same
    // tensor as the token embedding.  Fall back to token_embd.weight when a
    // dedicated output.weight is absent.
    let lm_head = if content.tensor_infos.contains_key("output.weight") {
        qmatmul_from_gguf(&content, &mut file, "output.weight", device)?
    } else {
        qmatmul_from_gguf(&content, &mut file, "token_embd.weight", device)?
    };

    Ok(Transformer { config, embedding, layers, final_norm, lm_head, device: device.clone() })
}

fn config_from_gguf(content: &candle_core::quantized::gguf_file::Content) -> Result<TransformerConfig> {
    use candle_core::quantized::gguf_file::Value;

    // The metadata key prefix matches `general.architecture` (e.g. "llama", "mistral").
    // Fall back to the legacy "llm" prefix used by older GGUF files.
    let arch = match content.metadata.get("general.architecture") {
        Some(Value::String(s)) => s.as_str().to_string(),
        _ => "llm".to_string(),
    };

    let get_u32 = |suffix: &str| -> Result<usize> {
        let key = format!("{}.{}", arch, suffix);
        match content.metadata.get(key.as_str()) {
            Some(Value::U32(v)) => Ok(*v as usize),
            Some(Value::I32(v)) => Ok(*v as usize),
            Some(Value::U64(v)) => Ok(*v as usize),
            _ => anyhow::bail!("missing or wrong type for GGUF key: {}", key),
        }
    };

    let get_f32 = |suffix: &str| -> Result<f32> {
        let key = format!("{}.{}", arch, suffix);
        match content.metadata.get(key.as_str()) {
            Some(Value::F32(v)) => Ok(*v),
            _ => Ok(10_000.0), // default RoPE theta
        }
    };

    Ok(TransformerConfig {
        hidden_dim:   get_u32("embedding_length")?,
        num_heads:    get_u32("attention.head_count")?,
        num_kv_heads: get_u32("attention.head_count_kv").unwrap_or_else(|_| get_u32("attention.head_count").unwrap()),
        ffn_dim:      get_u32("feed_forward_length")?,
        total_layers: get_u32("block_count")? as u32,
        rope_theta:   get_f32("rope.freq_base")?,
        vocab_size:   get_u32("vocab_size").unwrap_or_else(|_| {
            match content.metadata.get("tokenizer.ggml.tokens") {
                Some(Value::Array(arr)) => arr.len(),
                _ => 32_000,
            }
        }),
    })
}

fn qmatmul_from_gguf(
    content: &candle_core::quantized::gguf_file::Content,
    file:    &mut std::fs::File,
    name:    &str,
    device:  &Device,
) -> Result<QMatMul> {
    let qtensor = content
        .tensor(file, name, device)
        .with_context(|| format!("loading GGUF tensor: {}", name))?;
    Ok(QMatMul::from_qtensor(qtensor)?)
}

fn rms_norm_from_gguf(
    content: &candle_core::quantized::gguf_file::Content,
    file:    &mut std::fs::File,
    name:    &str,
    device:  &Device,
) -> Result<RmsNorm> {
    let weight = content
        .tensor(file, name, device)
        .with_context(|| format!("loading GGUF tensor: {}", name))?
        .dequantize(device)?
        .to_dtype(DType::F32)?;
    Ok(RmsNorm::new(weight, 1e-5))
}
