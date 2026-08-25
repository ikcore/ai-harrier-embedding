//! Gemma 3 text decoder (`Gemma3TextModel`) as used by `harrier-oss-v1-270m`
//! and `harrier-oss-v1-27b`: scaled embeddings, `(1 + w)` RMSNorms around
//! both attention and MLP, per-head q/k norms, interleaved sliding-window
//! (local RoPE base) and global (scaled RoPE) layers, GELU-tanh MLP.

use super::common::{BatchLayout, GemmaRmsNorm, RotaryTable, attention, linear};
use super::weights::{Embedding, Linear, Weights};
use crate::error::{HarrierError, Result};
use candle_core::{DType, Device, Module, Tensor};
use serde::Deserialize;

/// Legacy `rope_scaling` object (`transformers` 4.x).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default, alias = "type")]
    pub rope_type: Option<String>,
}

/// One entry of `rope_parameters` (`transformers` 5 writes
/// `{"full_attention": {...}, "sliding_attention": {...}}` for Gemma 3 and a
/// flat object for Qwen3).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f64>,
}

impl RopeParameters {
    /// Linear scaling factor implied by `rope_type` / `factor`.
    pub fn linear_factor(&self) -> Result<f64> {
        match self.rope_type.as_deref() {
            None | Some("default") => Ok(1.0),
            Some("linear") => Ok(self.factor.unwrap_or(1.0)),
            Some(other) => Err(HarrierError::Config(format!(
                "unsupported rope_type `{other}` (only default / linear)"
            ))),
        }
    }
}

/// Resolved RoPE settings for the two layer kinds.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeSettings {
    pub global_theta: f64,
    pub global_factor: f64,
    pub local_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_local_base")]
    pub rope_local_base_freq: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    /// `transformers` 5 layout; takes precedence over the legacy fields.
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: f64,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default)]
    pub sliding_window_pattern: Option<usize>,
    /// Older checkpoints spell the pattern with a leading underscore.
    #[serde(default, rename = "_sliding_window_pattern")]
    pub legacy_sliding_window_pattern: Option<usize>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub use_bidirectional_attention: bool,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

fn default_head_dim() -> usize {
    256
}
fn default_max_positions() -> usize {
    131_072
}
fn default_rms_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_local_base() -> f64 {
    10_000.0
}
fn default_query_pre_attn_scalar() -> f64 {
    256.0
}
fn default_sliding_window() -> usize {
    4096
}

impl Gemma3Config {
    /// Per-layer `true` = sliding-window attention. Uses `layer_types` when
    /// present, else the `sliding_window_pattern` rule (every n-th layer is
    /// global), as `Gemma3TextConfig` does.
    pub fn sliding_layers(&self) -> Vec<bool> {
        if let Some(types) = &self.layer_types {
            return types.iter().map(|t| t == "sliding_attention").collect();
        }
        let pattern = self
            .sliding_window_pattern
            .or(self.legacy_sliding_window_pattern)
            .unwrap_or(6)
            .max(1);
        (0..self.num_hidden_layers)
            .map(|i| (i + 1) % pattern != 0)
            .collect()
    }

    /// RoPE base / scaling per layer kind, from `rope_parameters` when
    /// present, else from `rope_theta` + `rope_scaling` (global) and
    /// `rope_local_base_freq` (sliding).
    pub fn rope_settings(&self) -> Result<RopeSettings> {
        if let Some(value) = &self.rope_parameters {
            let nested =
                value.get("full_attention").is_some() || value.get("sliding_attention").is_some();
            let parse = |key: &str| -> Result<RopeParameters> {
                Ok(match value.get(key) {
                    Some(v) => serde_json::from_value(v.clone())?,
                    None => RopeParameters::default(),
                })
            };
            let (global, local) = if nested {
                (parse("full_attention")?, parse("sliding_attention")?)
            } else {
                (
                    serde_json::from_value::<RopeParameters>(value.clone())?,
                    RopeParameters::default(),
                )
            };
            return Ok(RopeSettings {
                global_theta: global.rope_theta.unwrap_or(self.rope_theta),
                global_factor: global.linear_factor()?,
                local_theta: local.rope_theta.unwrap_or(self.rope_local_base_freq),
            });
        }
        let global_factor = match &self.rope_scaling {
            None => 1.0,
            Some(s) => match s.rope_type.as_deref() {
                None | Some("linear") | Some("default") => s.factor.unwrap_or(1.0),
                Some(other) => {
                    return Err(HarrierError::Config(format!(
                        "unsupported gemma3 rope_scaling type `{other}`"
                    )));
                }
            },
        };
        Ok(RopeSettings {
            global_theta: self.rope_theta,
            global_factor,
            local_theta: self.rope_local_base_freq,
        })
    }
}

#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f64,
    window: Option<usize>,
}

impl Attention {
    fn load(cfg: &Gemma3Config, sliding: bool, w: &Weights) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        Ok(Self {
            q_proj: linear(cfg.hidden_size, q_dim, w, "q_proj")?,
            k_proj: linear(cfg.hidden_size, kv_dim, w, "k_proj")?,
            v_proj: linear(cfg.hidden_size, kv_dim, w, "v_proj")?,
            o_proj: linear(q_dim, cfg.hidden_size, w, "o_proj")?,
            q_norm: GemmaRmsNorm::load(head_dim, cfg.rms_norm_eps, w, "q_norm")?,
            k_norm: GemmaRmsNorm::load(head_dim, cfg.rms_norm_eps, w, "k_norm")?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim,
            scale: cfg.query_pre_attn_scalar.powf(-0.5),
            window: sliding.then_some(cfg.sliding_window),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        layout: &BatchLayout,
        rope: &RotaryTable,
        causal: bool,
    ) -> Result<Tensor> {
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
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let (q, k) = rope.apply(&q, &k)?;
        let out = attention(&q, &k, &v, layout, self.window, self.scale, causal)?;
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
    fn load(cfg: &Gemma3Config, w: &Weights) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size, w, "gate_proj")?,
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size, w, "up_proj")?,
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size, w, "down_proj")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // gelu_pytorch_tanh == candle's tanh-approximation `gelu`.
        let gate = self.gate_proj.forward(x)?.gelu()?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

#[derive(Debug)]
struct Layer {
    input_layernorm: GemmaRmsNorm,
    self_attn: Attention,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    mlp: Mlp,
    post_feedforward_layernorm: GemmaRmsNorm,
    sliding: bool,
}

impl Layer {
    fn load(cfg: &Gemma3Config, sliding: bool, w: &Weights) -> Result<Self> {
        let norm = |name: &str| GemmaRmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, w, name);
        Ok(Self {
            input_layernorm: norm("input_layernorm")?,
            self_attn: Attention::load(cfg, sliding, &w.pp("self_attn"))?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm")?,
            mlp: Mlp::load(cfg, &w.pp("mlp"))?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm")?,
            sliding,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        layout: &BatchLayout,
        rope: &RotaryTable,
        causal: bool,
    ) -> Result<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, layout, rope, causal)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let x = (x + h)?;
        let h = self.pre_feedforward_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        Ok((x + h)?)
    }
}

/// The full Gemma 3 text stack, returning final-normed hidden states.
#[derive(Debug)]
pub struct Gemma3 {
    embed_tokens: Embedding,
    embed_scale: Tensor,
    layers: Vec<Layer>,
    norm: GemmaRmsNorm,
    rope_global: RotaryTable,
    rope_local: RotaryTable,
    causal: bool,
    config: Gemma3Config,
}

impl Gemma3 {
    pub fn load(
        config: Gemma3Config,
        w: &Weights,
        dtype: DType,
        device: &Device,
        max_positions: usize,
    ) -> Result<Self> {
        if let Some(act) = &config.hidden_activation
            && act != "gelu_pytorch_tanh"
        {
            return Err(HarrierError::Config(format!(
                "unsupported gemma3 hidden_activation `{act}`"
            )));
        }
        if config.attn_logit_softcapping.is_some_and(|c| c > 0.0) {
            return Err(HarrierError::Config(
                "gemma3 attn_logit_softcapping is not supported".into(),
            ));
        }
        let sliding = config.sliding_layers();
        if sliding.len() != config.num_hidden_layers {
            return Err(HarrierError::Config(format!(
                "layer_types has {} entries for {} layers",
                sliding.len(),
                config.num_hidden_layers
            )));
        }
        let embed_tokens = w.embedding(config.vocab_size, config.hidden_size, "embed_tokens")?;
        // `Gemma3TextScaledWordEmbedding`: sqrt(hidden) stored in f32, cast to
        // the weight dtype before the multiply.
        let embed_scale =
            Tensor::new((config.hidden_size as f64).sqrt() as f32, device)?.to_dtype(dtype)?;
        let layers = sliding
            .iter()
            .enumerate()
            .map(|(i, &is_sliding)| Layer::load(&config, is_sliding, &w.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = GemmaRmsNorm::load(config.hidden_size, config.rms_norm_eps, w, "norm")?;
        let positions = max_positions.min(config.max_position_embeddings).max(1);
        let rope = config.rope_settings()?;
        let rope_global = RotaryTable::new(
            config.head_dim,
            rope.global_theta,
            rope.global_factor,
            positions,
            dtype,
            device,
        )?;
        let rope_local = RotaryTable::new(
            config.head_dim,
            rope.local_theta,
            1.0,
            positions,
            dtype,
            device,
        )?;
        Ok(Self {
            embed_tokens,
            embed_scale,
            layers,
            norm,
            rope_global,
            rope_local,
            causal: !config.use_bidirectional_attention,
            config,
        })
    }

    pub fn config(&self) -> &Gemma3Config {
        &self.config
    }

    pub fn max_positions(&self) -> usize {
        self.rope_global.max_positions()
    }

    /// `input_ids [b, s]` (u32, right-padded) → hidden states `[b, s, hidden]`.
    pub fn forward(&self, input_ids: &Tensor, layout: &BatchLayout) -> Result<Tensor> {
        let mut x = self
            .embed_tokens
            .forward(input_ids)?
            .broadcast_mul(&self.embed_scale)?;
        for layer in &self.layers {
            let rope = if layer.sliding {
                &self.rope_local
            } else {
                &self.rope_global
            };
            x = layer.forward(&x, layout, rope, self.causal)?;
        }
        Ok(self.norm.forward(&x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_pattern_matches_transformers() {
        let json = r#"{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":12,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":10,"sliding_window_pattern":6,"_sliding_window_pattern":6}"#;
        let cfg: Gemma3Config = serde_json::from_str(json).unwrap();
        let s = cfg.sliding_layers();
        assert_eq!(s.iter().filter(|x| !**x).count(), 2);
        assert!(!s[5] && !s[11] && s[0] && s[4] && s[6]);

        let json = r#"{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":10,"layer_types":["sliding_attention","full_attention"],"rope_scaling":{"factor":8.0,"rope_type":"linear"}}"#;
        let cfg: Gemma3Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.sliding_layers(), vec![true, false]);
        assert_eq!(
            cfg.rope_settings().unwrap(),
            RopeSettings {
                global_theta: 1_000_000.0,
                global_factor: 8.0,
                local_theta: 10_000.0
            }
        );
        assert_eq!(cfg.head_dim, 256);
    }

    #[test]
    fn reads_transformers_5_rope_parameters() {
        let json = r#"{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":10,
            "rope_parameters":{"full_attention":{"factor":8.0,"rope_theta":1000000.0,"rope_type":"linear"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}"#;
        let cfg: Gemma3Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.rope_settings().unwrap(),
            RopeSettings {
                global_theta: 1_000_000.0,
                global_factor: 8.0,
                local_theta: 10_000.0
            }
        );
        let json = r#"{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":10,
            "rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"yarn","factor":4.0}}}"#;
        let cfg: Gemma3Config = serde_json::from_str(json).unwrap();
        assert!(cfg.rope_settings().is_err());
    }
}
