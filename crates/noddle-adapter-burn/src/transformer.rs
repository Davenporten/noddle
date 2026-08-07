use anyhow::{Context, Result};
use burn::backend::Rocm;
use burn::nn::attention::MultiHeadAttention;
use burn::nn::{Embedding, RmsNorm, RotaryEncoding};
use burn::prelude::DeviceOps;
use burn::tensor::{Tensor, TensorKind};
use burn_rocm::RocmDevice;

pub struct Transformer {}
