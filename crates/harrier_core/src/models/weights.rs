//! Where a backbone's parameters come from: a safetensors `VarBuilder` (dense
//! weights cast to the compute dtype) or a GGUF file (weights stay
//! quantised on the device). The backbones only ever ask for `transformers`
//! parameter paths, so they are written once and run on either.

use super::gguf::GgufFile;
use super::qlinear::QLinear;
use crate::error::{HarrierError, Result};
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use std::sync::Arc;

/// A prefix-scoped view onto a checkpoint.
#[derive(Clone)]
pub enum Weights<'a> {
    Dense(VarBuilder<'a>),
    Gguf {
        file: Arc<GgufFile>,
        prefix: String,
        dtype: DType,
        /// Dequantise linear weights to `dtype` at load instead of keeping
        /// them quantised (dense-speed matmuls at dense-size memory).
        dequantize: bool,
    },
}

impl<'a> Weights<'a> {
    pub fn dense(vb: VarBuilder<'a>) -> Self {
        Self::Dense(vb)
    }

    /// GGUF weights; `dtype` is the activation dtype norms are cast to (and
    /// the weight dtype when `dequantize` is set).
    pub fn gguf(file: Arc<GgufFile>, dtype: DType, dequantize: bool) -> Self {
        Self::Gguf {
            file,
            prefix: String::new(),
            dtype,
            dequantize,
        }
    }

    /// Push a path segment (`layers.0`, `self_attn`).
    pub fn pp(&self, segment: impl AsRef<str>) -> Self {
        match self {
            Self::Dense(vb) => Self::Dense(vb.pp(segment.as_ref())),
            Self::Gguf {
                file,
                prefix,
                dtype,
                dequantize,
            } => Self::Gguf {
                file: file.clone(),
                prefix: if prefix.is_empty() {
                    segment.as_ref().to_string()
                } else {
                    format!("{prefix}.{}", segment.as_ref())
                },
                dtype: *dtype,
                dequantize: *dequantize,
            },
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Dense(vb) => vb.device().clone(),
            Self::Gguf { file, .. } => file.device().clone(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Dense(vb) => vb.dtype(),
            Self::Gguf { dtype, .. } => *dtype,
        }
    }

    /// Whether `name` (relative to this prefix) exists.
    pub fn contains(&self, name: &str) -> bool {
        match self {
            Self::Dense(vb) => vb.contains_tensor(name),
            Self::Gguf { file, prefix, .. } => file
                .tensor_name(&join(prefix, name))
                .is_some_and(|n| file.has_tensor(&n)),
        }
    }

    fn qtensor(&self, name: &str) -> Result<QTensor> {
        match self {
            Self::Dense(_) => unreachable!("qtensor() is only called for GGUF weights"),
            Self::Gguf { file, prefix, .. } => {
                let path = join(prefix, name);
                let gguf_name = file.tensor_name(&path).ok_or_else(|| {
                    HarrierError::Config(format!(
                        "no GGUF tensor name is known for `{path}` ({:?})",
                        file.architecture()
                    ))
                })?;
                file.tensor(&gguf_name)
            }
        }
    }

    /// A bias-free linear layer `[out_dim, in_dim]`.
    pub fn linear(&self, in_dim: usize, out_dim: usize, name: &str) -> Result<Linear> {
        match self {
            Self::Dense(vb) => Ok(Linear::Dense(candle_nn::linear_no_bias(
                in_dim,
                out_dim,
                vb.pp(name),
            )?)),
            Self::Gguf {
                file,
                dtype,
                dequantize,
                ..
            } => {
                let q = self.qtensor(&format!("{name}.weight"))?;
                let dims = q.shape().dims().to_vec();
                if dims != [out_dim, in_dim] {
                    return Err(HarrierError::Config(format!(
                        "{}: `{}` is {dims:?}, expected [{out_dim}, {in_dim}]",
                        file.path().display(),
                        join(self.prefix(), name)
                    )));
                }
                if *dequantize {
                    let w = q.dequantize(file.device())?.to_dtype(*dtype)?;
                    Ok(Linear::Dense(candle_nn::Linear::new(w, None)))
                } else {
                    Ok(Linear::Quantized(QLinear::new(q, file.device())?))
                }
            }
        }
    }

    /// A norm weight of `size` elements in `dtype`.
    pub fn norm_weight(&self, size: usize, name: &str, dtype: DType) -> Result<Tensor> {
        match self {
            Self::Dense(vb) => Ok(vb.get_with_hints_dtype(
                size,
                &format!("{name}.weight"),
                Default::default(),
                dtype,
            )?),
            Self::Gguf { file, .. } => {
                let q = self.qtensor(&format!("{name}.weight"))?;
                let t = q.dequantize(file.device())?;
                if t.dims() != [size] {
                    return Err(HarrierError::Config(format!(
                        "{}: `{}` is {:?}, expected [{size}]",
                        file.path().display(),
                        join(self.prefix(), name),
                        t.dims()
                    )));
                }
                Ok(t.to_dtype(dtype)?)
            }
        }
    }

    /// Whether Gemma norm weights already include the `+1` (llama.cpp's
    /// converter folds it in; `transformers` checkpoints store `w`).
    pub fn gemma_norms_include_one(&self) -> bool {
        matches!(self, Self::Gguf { .. })
    }

    /// The token-embedding table `[vocab, hidden]`.
    pub fn embedding(&self, vocab: usize, hidden: usize, name: &str) -> Result<Embedding> {
        match self {
            Self::Dense(vb) => Ok(Embedding::Dense(candle_nn::embedding(
                vocab,
                hidden,
                vb.pp(name),
            )?)),
            Self::Gguf { file, dtype, .. } => {
                let q = self.qtensor(&format!("{name}.weight"))?;
                let dims = q.shape().dims().to_vec();
                if dims != [vocab, hidden] {
                    return Err(HarrierError::Config(format!(
                        "{}: `{}` is {dims:?}, expected [{vocab}, {hidden}]",
                        file.path().display(),
                        join(self.prefix(), name)
                    )));
                }
                Ok(Embedding::Quantized {
                    table: Arc::new(q),
                    dtype: *dtype,
                })
            }
        }
    }

    fn prefix(&self) -> &str {
        match self {
            Self::Dense(_) => "",
            Self::Gguf { prefix, .. } => prefix,
        }
    }
}

fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Bias-free linear layer over dense or quantised weights.
#[derive(Debug)]
pub enum Linear {
    Dense(candle_nn::Linear),
    Quantized(QLinear),
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized(l) => l.forward(x),
        }
    }
}

/// Token-embedding lookup; quantised tables are gathered on the device
/// without being dequantised in full.
#[derive(Debug)]
pub enum Embedding {
    Dense(candle_nn::Embedding),
    Quantized { table: Arc<QTensor>, dtype: DType },
}

impl Module for Embedding {
    fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(ids),
            Self::Quantized { table, dtype } => table.embedding(ids)?.to_dtype(*dtype),
        }
    }
}
