//! Qwen3 decoder (`Qwen3Model`) as used by `harrier-oss-v1-0.6b`: GQA with
//! per-head q/k RMSNorm, RoPE, SwiGLU MLP, full causal attention.

use super::common::{BatchLayout, RmsNorm, RotaryTable, attention, linear};
use super::weights::{Embedding, Linear, Weights};
use crate::error::{HarrierError, Result};
use candle_core::{DType, Device, Module, Tensor};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// `transformers` 5 layout (`{"rope_theta": 1000000.0, "rope_type": "default"}`).
    #[serde(default)]
    pub rope_parameters: Option<super::gemma3::RopeParameters>,
    /// Legacy `rope_scaling` (must be absent or linear).
    #[serde(default)]
    pub rope_scaling: Option<super::gemma3::RopeScaling>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub hidden_act: Option<String>,
}

fn default_rms_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    10_000.0
}

impl Qwen3Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// `(rope_theta, linear_factor)` from either config layout.
    pub fn rope(&self) -> Result<(f64, f64)> {
        if let Some(p) = &self.rope_parameters {
            return Ok((p.rope_theta.unwrap_or(self.rope_theta), p.linear_factor()?));
        }
        let factor = match &self.rope_scaling {
            None => 1.0,
            Some(s) => match s.rope_type.as_deref() {
                None | Some("linear") | Some("default") => s.factor.unwrap_or(1.0),
                Some(other) => {
                    return Err(HarrierError::Config(format!(
                        "unsupported qwen3 rope_scaling type `{other}`"
                    )));
                }
            },
        };
        Ok((self.rope_theta, factor))
    }
}

#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(cfg: &Qwen3Config, w: &Weights) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        Ok(Self {
            q_proj: linear(cfg.hidden_size, q_dim, w, "q_proj")?,
            k_proj: linear(cfg.hidden_size, kv_dim, w, "k_proj")?,
            v_proj: linear(cfg.hidden_size, kv_dim, w, "v_proj")?,
            o_proj: linear(q_dim, cfg.hidden_size, w, "o_proj")?,
            q_norm: RmsNorm::load(head_dim, cfg.rms_norm_eps, w, "q_norm")?,
            k_norm: RmsNorm::load(head_dim, cfg.rms_norm_eps, w, "k_norm")?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor, layout: &BatchLayout, rope: &RotaryTable) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, self.heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, s, self.kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, s, self.kv_heads, self.head_dim))?;
        // q/k norm is applied per head before the rotation.
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let (q, k) = rope.apply(&q, &k)?;
        let out = attention(&q, &k, &v, layout, None, self.scale, true)?;
        Ok(self.o_proj.forward(&out)?)
    }
}

#[derive(Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &Qwen3Config, w: &Weights) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size, w, "gate_proj")?,
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size, w, "up_proj")?,
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size, w, "down_proj")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

#[derive(Debug)]
struct Layer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &Qwen3Config, w: &Weights) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                w,
                "input_layernorm",
            )?,
            self_attn: Attention::load(cfg, &w.pp("self_attn"))?,
            post_attention_layernorm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                w,
                "post_attention_layernorm",
            )?,
            mlp: Mlp::load(cfg, &w.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, layout: &BatchLayout, rope: &RotaryTable) -> Result<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let x = (x + self.self_attn.forward(&h, layout, rope)?)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        Ok((&x + self.mlp.forward(&h)?)?)
    }
}

/// The full Qwen3 stack, returning final-normed hidden states.
#[derive(Debug)]
pub struct Qwen3 {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    rope: RotaryTable,
    config: Qwen3Config,
}

impl Qwen3 {
    /// `max_positions` bounds the rotary table (and so the longest input).
    pub fn load(
        config: Qwen3Config,
        w: &Weights,
        dtype: DType,
        device: &Device,
        max_positions: usize,
    ) -> Result<Self> {
        if let Some(act) = &config.hidden_act
            && act != "silu"
        {
            return Err(HarrierError::Config(format!(
                "unsupported qwen3 hidden_act `{act}`"
            )));
        }
        let embed_tokens = w.embedding(config.vocab_size, config.hidden_size, "embed_tokens")?;
        let layers = (0..config.num_hidden_layers)
            .map(|i| Layer::load(&config, &w.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, w, "norm")?;
        let (theta, factor) = config.rope()?;
        let rope = RotaryTable::new(
            config.head_dim(),
            theta,
            factor,
            max_positions.min(config.max_position_embeddings).max(1),
            dtype,
            device,
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope,
            config,
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn max_positions(&self) -> usize {
        self.rope.max_positions()
    }

    /// `input_ids [b, s]` (u32, right-padded) → hidden states `[b, s, hidden]`.
    pub fn forward(&self, input_ids: &Tensor, layout: &BatchLayout) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, layout, &self.rope)?;
        }
        Ok(self.norm.forward(&x)?)
    }
}
