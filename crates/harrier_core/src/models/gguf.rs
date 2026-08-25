//! GGUF checkpoints (llama.cpp's container): metadata → `config.json`-shaped
//! hyper-parameters, and the llama.cpp tensor names behind the
//! `transformers` names the backbones ask for.
//!
//! Tensors are read straight onto the compute device as [`QTensor`]s and
//! stay quantised there; the backbones see them through
//! [`super::weights::Weights`].

use crate::error::{HarrierError, Result};
use crate::registry::Architecture;
use candle_core::Device;
use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// An opened GGUF file.
pub struct GgufFile {
    path: PathBuf,
    content: Content,
    reader: Mutex<File>,
    device: Device,
    /// `general.architecture`, the prefix of every hyper-parameter key.
    arch_name: String,
    arch: Architecture,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("path", &self.path)
            .field("arch", &self.arch)
            .field("tensors", &self.content.tensor_infos.len())
            .finish()
    }
}

impl GgufFile {
    pub fn open(path: &Path, device: &Device) -> Result<Self> {
        let mut file = File::open(path)?;
        let content = Content::read(&mut file).map_err(|e| {
            HarrierError::Config(format!("{}: not a readable GGUF file: {e}", path.display()))
        })?;
        let arch_name = match content.metadata.get("general.architecture") {
            Some(v) => v
                .to_string()
                .map_err(|e| HarrierError::Config(format!("general.architecture: {e}")))?
                .clone(),
            None => {
                return Err(HarrierError::Config(format!(
                    "{}: GGUF has no general.architecture",
                    path.display()
                )));
            }
        };
        let arch = Architecture::from_model_type(&arch_name).ok_or_else(|| {
            HarrierError::Config(format!(
                "{}: GGUF architecture `{arch_name}` is not supported (qwen3, gemma3)",
                path.display()
            ))
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            content,
            reader: Mutex::new(file),
            device: device.clone(),
            arch_name,
            arch,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn architecture(&self) -> Architecture {
        self.arch
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The quantisation of the bulk of the weights (`general.file_type` when
    /// present, else the dtype of the first projection), as a lowercase tag
    /// such as `q8_0`.
    pub fn quantization(&self) -> String {
        let probe = self
            .content
            .tensor_infos
            .get("blk.0.attn_q.weight")
            .or_else(|| self.content.tensor_infos.get("blk.0.ffn_down.weight"))
            .map(|t| t.ggml_dtype);
        match probe {
            Some(dtype) => ggml_dtype_name(dtype).to_string(),
            None => "gguf".to_string(),
        }
    }

    /// Total bytes of tensor data in the file.
    pub fn weight_bytes(&self) -> u64 {
        self.content
            .tensor_infos
            .values()
            .map(|t| {
                let n = t.shape.elem_count();
                (n / t.ggml_dtype.block_size() * t.ggml_dtype.type_size()) as u64
            })
            .sum()
    }

    fn value(&self, key: &str) -> Option<&Value> {
        self.content.metadata.get(key)
    }

    /// `<arch>.<key>` as a number, if present and numeric.
    pub fn hp(&self, key: &str) -> Option<f64> {
        let v = self.value(&format!("{}.{key}", self.arch_name))?;
        numeric(v)
    }

    pub fn hp_usize(&self, key: &str) -> Result<usize> {
        self.hp(key).map(|v| v as usize).ok_or_else(|| {
            HarrierError::Config(format!(
                "{}: GGUF metadata is missing {}.{key}",
                self.path.display(),
                self.arch_name
            ))
        })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    pub fn tensor_dims(&self, name: &str) -> Option<Vec<usize>> {
        self.content
            .tensor_infos
            .get(name)
            .map(|t| t.shape.dims().to_vec())
    }

    /// Read one tensor onto the device.
    pub fn tensor(&self, name: &str) -> Result<QTensor> {
        let mut reader = self
            .reader
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.content
            .tensor(&mut *reader, name, &self.device)
            .map_err(|e| HarrierError::Config(format!("{}: {e}", self.path.display())))
    }

    /// A `config.json`-shaped document synthesised from the GGUF metadata,
    /// for checkpoints that ship without one. Covers what
    /// [`super::qwen3::Qwen3Config`] / [`super::gemma3::Gemma3Config`] read.
    pub fn config_json(&self) -> Result<String> {
        let vocab = self
            .tensor_dims("token_embd.weight")
            .and_then(|d| d.first().copied())
            .ok_or_else(|| {
                HarrierError::Config(format!(
                    "{}: GGUF has no token_embd.weight",
                    self.path.display()
                ))
            })?;
        let layers = self.hp_usize("block_count")?;
        let head_count = self.hp_usize("attention.head_count")?;
        let hidden = self.hp_usize("embedding_length")?;
        let head_dim = self
            .hp("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden / head_count.max(1));
        let mut cfg = serde_json::json!({
            "model_type": match self.arch {
                Architecture::Qwen3 => "qwen3",
                Architecture::Gemma3 => "gemma3_text",
            },
            "architectures": [match self.arch {
                Architecture::Qwen3 => "Qwen3Model",
                Architecture::Gemma3 => "Gemma3TextModel",
            }],
            "hidden_size": hidden,
            "intermediate_size": self.hp_usize("feed_forward_length")?,
            "num_hidden_layers": layers,
            "num_attention_heads": head_count,
            "num_key_value_heads": self.hp("attention.head_count_kv").map(|v| v as usize).unwrap_or(head_count),
            "head_dim": head_dim,
            "vocab_size": vocab,
            "max_position_embeddings": self.hp("context_length").map(|v| v as usize).unwrap_or(32_768),
            "rms_norm_eps": self.hp("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            "rope_theta": self.hp("rope.freq_base").unwrap_or(match self.arch {
                Architecture::Qwen3 => 1_000_000.0,
                Architecture::Gemma3 => 1_000_000.0,
            }),
        });
        if let Some(factor) = self.hp("rope.scaling.factor")
            && factor > 0.0
            && factor != 1.0
        {
            cfg["rope_scaling"] = serde_json::json!({"rope_type": "linear", "factor": factor});
        }
        if let Some(pad) = self
            .value("tokenizer.ggml.padding_token_id")
            .and_then(numeric)
        {
            cfg["pad_token_id"] = serde_json::json!(pad as u32);
        }
        if let Some(eos) = self.value("tokenizer.ggml.eos_token_id").and_then(numeric) {
            cfg["eos_token_id"] = serde_json::json!(eos as u32);
        }
        if self.arch == Architecture::Gemma3 {
            // Gemma 3 scales queries by query_pre_attn_scalar^-0.5; llama.cpp
            // derives it from the head size, some converters store it.
            cfg["query_pre_attn_scalar"] = serde_json::json!(
                self.hp("attention.key_length_scale")
                    .unwrap_or(head_dim as f64)
            );
            cfg["rope_local_base_freq"] =
                serde_json::json!(self.hp("rope.local_freq_base").unwrap_or(10_000.0));
            // Sliding window: absent / 0 means every layer attends globally
            // (Harrier 270M). Otherwise llama.cpp's Gemma 3 rule makes every
            // sixth layer global.
            let window = self.hp("attention.sliding_window").unwrap_or(0.0) as usize;
            let period = self
                .hp("attention.sliding_window_pattern")
                .filter(|p| *p > 0.0)
                .map(|p| p as usize)
                .unwrap_or(6);
            let layer_types: Vec<&str> = (0..layers)
                .map(|i| {
                    if window > 0 && (i + 1) % period != 0 {
                        "sliding_attention"
                    } else {
                        "full_attention"
                    }
                })
                .collect();
            cfg["sliding_window"] = serde_json::json!(window.max(1));
            cfg["layer_types"] = serde_json::json!(layer_types);
        }
        Ok(cfg.to_string())
    }

    /// The llama.cpp tensor name for a `transformers` parameter path
    /// (`layers.3.self_attn.q_proj.weight` → `blk.3.attn_q.weight`).
    pub fn tensor_name(&self, hf_path: &str) -> Option<String> {
        tensor_name(self.arch, hf_path)
    }
}

/// Any numeric GGUF value as f64.
fn numeric(v: &Value) -> Option<f64> {
    Some(match v {
        Value::U8(x) => *x as f64,
        Value::I8(x) => *x as f64,
        Value::U16(x) => *x as f64,
        Value::I16(x) => *x as f64,
        Value::U32(x) => *x as f64,
        Value::I32(x) => *x as f64,
        Value::U64(x) => *x as f64,
        Value::I64(x) => *x as f64,
        Value::F32(x) => *x as f64,
        Value::F64(x) => *x,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::String(_) | Value::Array(_) => return None,
    })
}

/// Lowercase ggml dtype tag (`q8_0`, `q4_k`, `f16`, …).
pub fn ggml_dtype_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::BF16 => "bf16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        GgmlDType::Q8K => "q8_k",
    }
}

/// `transformers` parameter path → llama.cpp tensor name. The path is the
/// one the backbones build (relative to the text model, `.weight` suffix).
pub fn tensor_name(arch: Architecture, hf_path: &str) -> Option<String> {
    let path = hf_path.strip_suffix(".weight").unwrap_or(hf_path);
    let path = path.strip_prefix("model.").unwrap_or(path);
    let mapped = match path {
        "embed_tokens" => "token_embd".to_string(),
        "norm" => "output_norm".to_string(),
        _ => {
            let rest = path.strip_prefix("layers.")?;
            let (index, leaf) = rest.split_once('.')?;
            let index: usize = index.parse().ok()?;
            let leaf = match (arch, leaf) {
                (_, "input_layernorm") => "attn_norm",
                (_, "self_attn.q_proj") => "attn_q",
                (_, "self_attn.k_proj") => "attn_k",
                (_, "self_attn.v_proj") => "attn_v",
                (_, "self_attn.o_proj") => "attn_output",
                (_, "self_attn.q_norm") => "attn_q_norm",
                (_, "self_attn.k_norm") => "attn_k_norm",
                (_, "mlp.gate_proj") => "ffn_gate",
                (_, "mlp.up_proj") => "ffn_up",
                (_, "mlp.down_proj") => "ffn_down",
                // Qwen3 (llama layout): the norm before the MLP.
                (Architecture::Qwen3, "post_attention_layernorm") => "ffn_norm",
                // Gemma 3: norms after attention, before and after the MLP.
                (Architecture::Gemma3, "post_attention_layernorm") => "post_attention_norm",
                (Architecture::Gemma3, "pre_feedforward_layernorm") => "ffn_norm",
                (Architecture::Gemma3, "post_feedforward_layernorm") => "post_ffw_norm",
                _ => return None,
            };
            format!("blk.{index}.{leaf}")
        }
    };
    Some(format!("{mapped}.weight"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_transformers_paths_to_llama_cpp_names() {
        let q = |p: &str| tensor_name(Architecture::Qwen3, p);
        let g = |p: &str| tensor_name(Architecture::Gemma3, p);
        assert_eq!(
            q("embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            q("model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            q("layers.3.self_attn.q_proj.weight").as_deref(),
            Some("blk.3.attn_q.weight")
        );
        assert_eq!(
            q("layers.27.mlp.down_proj.weight").as_deref(),
            Some("blk.27.ffn_down.weight")
        );
        assert_eq!(
            q("layers.0.post_attention_layernorm.weight").as_deref(),
            Some("blk.0.ffn_norm.weight")
        );
        assert_eq!(
            g("layers.0.post_attention_layernorm.weight").as_deref(),
            Some("blk.0.post_attention_norm.weight")
        );
        assert_eq!(
            g("layers.0.pre_feedforward_layernorm.weight").as_deref(),
            Some("blk.0.ffn_norm.weight")
        );
        assert_eq!(
            g("layers.0.post_feedforward_layernorm.weight").as_deref(),
            Some("blk.0.post_ffw_norm.weight")
        );
        assert_eq!(q("layers.0.pre_feedforward_layernorm.weight"), None);
        assert_eq!(q("lm_head.weight"), None);
        assert_eq!(q("layers.x.mlp.up_proj.weight"), None);
    }
}
