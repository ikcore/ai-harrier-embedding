//! Transformer backbones and the dispatch between them.

pub mod common;
pub mod gemma3;
pub mod gguf;
pub mod qlinear;
pub mod qwen3;
pub mod weights;

use crate::error::{HarrierError, Result};
use crate::registry::Architecture;
use candle_core::{DType, Device, Tensor};
use common::BatchLayout;
use serde::Deserialize;
use weights::Weights;

/// The fields of `config.json` every backbone shares.
#[derive(Debug, Clone, Deserialize)]
pub struct CommonConfig {
    pub model_type: String,
    pub hidden_size: usize,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub architectures: Vec<String>,
}

/// A loaded decoder stack producing final-normed hidden states.
#[derive(Debug)]
pub enum Backbone {
    Qwen3(qwen3::Qwen3),
    Gemma3(gemma3::Gemma3),
}

impl Backbone {
    /// Parse `config.json` and load the matching architecture from `w`
    /// (safetensors or GGUF). Weight names are looked up both bare
    /// (`layers.0…`, as the Harrier checkpoints are saved) and under a
    /// `model.` prefix (a `*ForCausalLM` export).
    pub fn load(
        config_json: &str,
        w: Weights,
        dtype: DType,
        device: &Device,
        max_positions: usize,
    ) -> Result<Self> {
        let common: CommonConfig = serde_json::from_str(config_json)?;
        let arch = Architecture::from_model_type(&common.model_type).ok_or_else(|| {
            HarrierError::Config(format!(
                "unsupported model_type `{}` (architectures: {:?}); expected qwen3 or gemma3_text",
                common.model_type, common.architectures
            ))
        })?;
        let w = if w.contains("embed_tokens.weight") {
            w
        } else if w.contains("model.embed_tokens.weight") {
            w.pp("model")
        } else {
            return Err(HarrierError::Config(
                "checkpoint has no `embed_tokens.weight` (is this a text model?)".into(),
            ));
        };
        Ok(match arch {
            Architecture::Qwen3 => {
                let cfg: qwen3::Qwen3Config = serde_json::from_str(config_json)?;
                Self::Qwen3(qwen3::Qwen3::load(cfg, &w, dtype, device, max_positions)?)
            }
            Architecture::Gemma3 => {
                let cfg: gemma3::Gemma3Config = serde_json::from_str(config_json)?;
                Self::Gemma3(gemma3::Gemma3::load(cfg, &w, dtype, device, max_positions)?)
            }
        })
    }

    pub fn architecture(&self) -> Architecture {
        match self {
            Self::Qwen3(_) => Architecture::Qwen3,
            Self::Gemma3(_) => Architecture::Gemma3,
        }
    }

    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Qwen3(m) => m.config().hidden_size,
            Self::Gemma3(m) => m.config().hidden_size,
        }
    }

    /// Longest sequence the rotary tables cover.
    pub fn max_positions(&self) -> usize {
        match self {
            Self::Qwen3(m) => m.max_positions(),
            Self::Gemma3(m) => m.max_positions(),
        }
    }

    /// `input_ids [b, s]` (u32, right-padded) → hidden states `[b, s, hidden]`.
    pub fn forward(&self, input_ids: &Tensor, layout: &BatchLayout) -> Result<Tensor> {
        match self {
            Self::Qwen3(m) => m.forward(input_ids, layout),
            Self::Gemma3(m) => m.forward(input_ids, layout),
        }
    }
}
