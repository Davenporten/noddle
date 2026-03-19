/// Converts between our wire-format `Tensor` (safetensors bytes) and
/// candle's native `Tensor` type.
///
/// The hidden state passed between nodes is a single 2-D tensor of shape
/// `[seq_len, hidden_dim]` stored as f32. We serialize it as a safetensors
/// file with a single entry named `"hidden_state"`.
use anyhow::{Context, Result};
use candle_core::{Device, Tensor as CandleTensor};
use noddle_core::tensor::Tensor as WireTensor;
const HIDDEN_STATE_KEY: &str = "hidden_state";

/// Serialize a candle tensor to our wire format.
pub fn to_wire(tensor: &CandleTensor) -> Result<WireTensor> {
    let shape = tensor.shape().dims().to_vec();
    let flat: Vec<f32> = tensor
        .flatten_all()
        .context("flattening tensor")?
        .to_vec1()
        .context("converting tensor to vec")?;
    let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Build a minimal safetensors file manually
    // Header: JSON with tensor metadata
    let header = serde_json::json!({
        HIDDEN_STATE_KEY: {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [0, bytes.len()]
        }
    });
    let header_bytes = header.to_string().into_bytes();
    let header_len = (header_bytes.len() as u64).to_le_bytes();

    let mut out = Vec::with_capacity(8 + header_bytes.len() + bytes.len());
    out.extend_from_slice(&header_len);
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&bytes);

    Ok(WireTensor::from_bytes(out))
}

/// Argmax over a wire-format logits tensor — returns the token ID with the highest score.
pub fn argmax_from_wire(wire: &WireTensor) -> Result<u32> {
    let bytes = wire.as_bytes();
    anyhow::ensure!(bytes.len() >= 8, "wire tensor too short");

    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;
    anyhow::ensure!(header_end <= bytes.len(), "safetensors header out of bounds");

    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..header_end]).context("parsing safetensors header")?;
    let entry = header.get(HIDDEN_STATE_KEY).context("missing hidden_state key")?;
    let offsets = entry["data_offsets"].as_array().context("data_offsets not array")?;
    let start = offsets[0].as_u64().unwrap_or(0) as usize;
    let end   = offsets[1].as_u64().unwrap_or(0) as usize;
    let data  = &bytes[header_end + start..header_end + end];

    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if v > best_val { best_val = v; best_idx = i as u32; }
    }
    Ok(best_idx)
}

/// Deserialize a wire-format tensor back to a candle tensor.
pub fn from_wire(wire: &WireTensor, device: &Device) -> Result<CandleTensor> {
    let bytes = wire.as_bytes();
    if bytes.len() < 8 {
        anyhow::bail!("wire tensor too short");
    }

    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;
    anyhow::ensure!(header_end <= bytes.len(), "safetensors header out of bounds");

    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..header_end]).context("parsing safetensors header")?;

    let entry = header
        .get(HIDDEN_STATE_KEY)
        .context("missing hidden_state in safetensors")?;

    let shape: Vec<usize> = entry["shape"]
        .as_array()
        .context("shape not an array")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();

    let offsets = entry["data_offsets"]
        .as_array()
        .context("data_offsets not an array")?;
    let start = offsets[0].as_u64().unwrap_or(0) as usize;
    let end   = offsets[1].as_u64().unwrap_or(0) as usize;

    let data_bytes = &bytes[header_end + start..header_end + end];
    let floats: Vec<f32> = data_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    CandleTensor::from_vec(floats, &shape[..], device).context("building candle tensor")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor as CandleTensor};

    #[test]
    fn round_trip_preserves_values() {
        let device = Device::Cpu;
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CandleTensor::from_vec(data.clone(), (2, 3), &device).unwrap();
        let wire = to_wire(&tensor).unwrap();
        let recovered = from_wire(&wire, &device).unwrap();
        let flat: Vec<f32> = recovered.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
        assert_eq!(flat, data);
    }

    #[test]
    fn round_trip_preserves_shape() {
        let device = Device::Cpu;
        let tensor = CandleTensor::from_vec(vec![0.0_f32; 12], (3, 4), &device).unwrap();
        let wire = to_wire(&tensor).unwrap();
        let recovered = from_wire(&wire, &device).unwrap();
        assert_eq!(recovered.dims(), &[3, 4]);
    }

    #[test]
    fn too_short_bytes_errors() {
        let wire = WireTensor::from_bytes(vec![0u8; 4]);
        assert!(from_wire(&wire, &Device::Cpu).is_err());
    }

    #[test]
    fn missing_hidden_state_key_errors() {
        let header = serde_json::json!({
            "wrong_key": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]}
        });
        let header_bytes = header.to_string().into_bytes();
        let header_len = (header_bytes.len() as u64).to_le_bytes();
        let mut blob: Vec<u8> = Vec::new();
        blob.extend_from_slice(&header_len);
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&[0u8; 8]);
        let wire = WireTensor::from_bytes(blob);
        assert!(from_wire(&wire, &Device::Cpu).is_err());
    }
}
