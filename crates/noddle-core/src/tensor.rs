use anyhow::{bail, Result};

/// A lightweight wrapper around raw tensor bytes in safetensors wire format.
/// The rest of the system treats tensors as opaque blobs; only InferenceAdapter
/// implementations know their internal layout.
#[derive(Debug, Clone, Default)]
pub struct Tensor(pub Vec<u8>);

impl Tensor {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn size_bytes(&self) -> usize {
        self.0.len()
    }

    /// Basic sanity check: non-empty and starts with a valid safetensors header.
    /// Does not perform a full parse — just enough to detect obvious corruption
    /// or tampering before forwarding.
    pub fn validate(&self) -> Result<()> {
        if self.0.is_empty() {
            bail!("tensor is empty");
        }
        // safetensors format: first 8 bytes are a little-endian u64 header length
        if self.0.len() < 8 {
            bail!("tensor too short to contain safetensors header");
        }
        let header_len =
            u64::from_le_bytes(self.0[..8].try_into().unwrap()) as usize;
        if header_len == 0 || 8 + header_len > self.0.len() {
            bail!("safetensors header length field is invalid");
        }
        Ok(())
    }
}

impl From<Vec<u8>> for Tensor {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_tensor() -> Tensor {
        // Minimal valid safetensors: 8-byte LE header_len + header JSON + payload
        let header = b"{}";
        let header_len = (header.len() as u64).to_le_bytes();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&header_len);
        bytes.extend_from_slice(header);
        bytes.push(0u8); // 1-byte payload
        Tensor::from_bytes(bytes)
    }

    #[test]
    fn valid_tensor_passes_validation() {
        assert!(valid_tensor().validate().is_ok());
    }

    #[test]
    fn empty_tensor_fails_validation() {
        let t = Tensor::default();
        assert!(t.validate().is_err());
    }

    #[test]
    fn too_short_tensor_fails_validation() {
        let t = Tensor::from_bytes(vec![0u8; 4]); // less than 8 bytes
        assert!(t.validate().is_err());
    }

    #[test]
    fn header_len_overflow_fails_validation() {
        // header_len claims 9999 bytes but total buffer is only 9 bytes
        let header_len = 9999u64.to_le_bytes();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&header_len);
        bytes.push(0u8);
        let t = Tensor::from_bytes(bytes);
        assert!(t.validate().is_err());
    }

    #[test]
    fn zero_header_len_fails_validation() {
        let header_len = 0u64.to_le_bytes();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&header_len);
        bytes.extend_from_slice(b"{}");
        let t = Tensor::from_bytes(bytes);
        assert!(t.validate().is_err());
    }

    #[test]
    fn roundtrip_bytes() {
        let original = vec![1u8, 2, 3, 4];
        let t = Tensor::from_bytes(original.clone());
        assert_eq!(t.into_bytes(), original);
    }

    #[test]
    fn size_bytes_matches() {
        let t = valid_tensor();
        assert_eq!(t.size_bytes(), t.as_bytes().len());
    }
}
