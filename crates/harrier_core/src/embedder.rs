//! `Embedder`: tokenizer + backbone + last-token pooling + L2 normalization,
//! with length-sorted micro-batching so one request with many inputs runs in
//! a handful of padded forward passes.

use crate::device::{DeviceRequest, default_dtype, device_name, dtype_name, select_device};
use crate::error::{HarrierError, Result};
use crate::loader::{ModelFiles, ModelSource, WeightFiles, model_dir, resolve_source, source_id};
use crate::models::common::{BatchLayout, l2_normalize, last_token_pool};
use crate::models::gguf::GgufFile;
use crate::models::weights::Weights;
use crate::models::{Backbone, CommonConfig};
use crate::registry::{self, Architecture, PROVIDER};
use crate::task::{EmbeddingsRequest, EmbeddingsResponse, Usage, resolve_inputs};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Serialize;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// How to load and drive a model.
#[derive(Debug, Clone)]
pub struct EmbedderOptions {
    pub device: DeviceRequest,
    /// Activation / dense-weight dtype. `None` → bf16 on accelerators and
    /// f32 on CPU for safetensors; f32 everywhere for GGUF (the quantised
    /// kernels take f32 activations, and on the CPU they require it).
    pub dtype: Option<DType>,
    /// Inputs longer than this are truncated (right side). `None` → the
    /// model's documented maximum (32 768 for Harrier).
    pub max_input_tokens: Option<usize>,
    /// For GGUF weights: dequantise the linear layers to `dtype` at load.
    /// Trades the int8 memory saving for dense-weight matmul speed — on a
    /// CPU without int8 dot-product instructions the f32 GEMM is 2–3×
    /// faster than the int8 kernel. No effect on safetensors checkpoints.
    pub dequantize: bool,
    /// Most rows in one forward pass.
    pub max_batch_size: usize,
    /// Most padded tokens (`rows × longest row`) in one forward pass. A
    /// single input longer than this still runs, alone.
    pub max_batch_tokens: usize,
}

impl Default for EmbedderOptions {
    fn default() -> Self {
        Self {
            device: DeviceRequest::Auto,
            dtype: None,
            max_input_tokens: None,
            dequantize: false,
            max_batch_size: 32,
            max_batch_tokens: 16_384,
        }
    }
}

/// What is loaded, for logs and `GET /v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// Bare id (`harrier-oss-v1-0.6b`).
    pub id: String,
    /// `harrier::harrier-oss-v1-0.6b`.
    pub qualified_id: String,
    pub display_name: String,
    /// Hub repo or local path the weights came from.
    pub source: String,
    pub architecture: Architecture,
    pub parameters: Option<String>,
    pub embedding_dimensions: u32,
    pub max_input_tokens: u32,
    pub device: String,
    /// Activation dtype (`f32`, `bf16`); also the weight dtype for
    /// safetensors checkpoints.
    pub dtype: String,
    /// Weight quantisation of a GGUF checkpoint (`q8_0`, `q4_k`), `None`
    /// for dense weights.
    pub quantization: Option<String>,
    /// The GGUF weights were dequantised to `dtype` at load
    /// ([`EmbedderOptions::dequantize`]), so matmuls run dense.
    pub dequantized: bool,
    pub load_seconds: f64,
}

impl ModelInfo {
    /// `q8_0`, `bf16`, `q8_0/bf16`, `q8_0→f32`: what the model runs in, for
    /// logs.
    pub fn precision(&self) -> String {
        match &self.quantization {
            None => self.dtype.clone(),
            Some(q) if self.dequantized => format!("{q}→{}", self.dtype),
            Some(q) if self.dtype == "f32" => q.clone(),
            Some(q) => format!("{q}/{}", self.dtype),
        }
    }
}

/// Vectors plus the token count that produced them.
#[derive(Debug, Clone, PartialEq)]
pub struct Embeddings {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: usize,
}

/// A loaded Harrier model. `Send + Sync`; forward passes are serialized
/// through an internal lock so one instance can be shared behind an `Arc`.
pub struct Embedder {
    tokenizer: Tokenizer,
    backbone: Backbone,
    device: Device,
    pad_token_id: u32,
    info: ModelInfo,
    options: EmbedderOptions,
    compute: Mutex<()>,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder")
            .field("info", &self.info)
            .finish()
    }
}

impl Embedder {
    /// Load by registry name (`0.6b`, `harrier::harrier-oss-v1-270m`), Hub
    /// repo (`microsoft/harrier-oss-v1-0.6b[@rev]`), local directory, GGUF
    /// quantisation (`0.6b:q8_0`) or `.gguf` path. A copy under the model
    /// directory (`$HARRIER_MODEL_DIR`, default `./models`) is used before
    /// anything is downloaded — see [`crate::loader::resolve_source`].
    pub fn load(name: &str, options: EmbedderOptions) -> Result<Self> {
        Self::load_from(name, &model_dir(), options)
    }

    /// [`Embedder::load`] with an explicit model directory.
    pub fn load_from(name: &str, model_dir: &Path, options: EmbedderOptions) -> Result<Self> {
        let spec = registry::resolve(name);
        let source = resolve_source(name, model_dir);
        let files = ModelFiles::locate(&source)?;
        let id = spec
            .map(|s| s.id.to_string())
            .unwrap_or_else(|| source_id(&source));
        Self::load_files(&files, &source, id, options)
    }

    /// Load from already-located files.
    pub fn load_files(
        files: &ModelFiles,
        source: &ModelSource,
        id: String,
        options: EmbedderOptions,
    ) -> Result<Self> {
        let started = Instant::now();
        let device = select_device(options.device)?;
        let gguf = match &files.weights {
            WeightFiles::Gguf(path) => Some(Arc::new(GgufFile::open(path, &device)?)),
            WeightFiles::Safetensors(_) => None,
        };
        let dtype = match options.dtype {
            Some(dtype) => dtype,
            None if gguf.is_some() => DType::F32,
            None => default_dtype(&device),
        };
        if gguf.is_some() && !options.dequantize && device.is_cpu() && dtype != DType::F32 {
            return Err(HarrierError::Device(format!(
                "quantised (GGUF) weights on the CPU need f32 activations, not {}",
                dtype_name(dtype)
            )));
        }

        let config_json = match (&files.config, &gguf) {
            (Some(path), _) => std::fs::read_to_string(path)?,
            (None, Some(gguf)) => gguf.config_json()?,
            (None, None) => {
                return Err(HarrierError::Config("checkpoint has no config.json".into()));
            }
        };
        let common: CommonConfig = serde_json::from_str(&config_json)?;
        if let Some(gguf) = &gguf
            && let Some(arch) = Architecture::from_model_type(&common.model_type)
            && arch != gguf.architecture()
        {
            return Err(HarrierError::Config(format!(
                "{} holds a {:?} model but config.json describes {:?}",
                gguf.path().display(),
                gguf.architecture(),
                arch
            )));
        }
        let spec = registry::resolve(&id);
        let documented_max = spec
            .map(|s| s.max_input_tokens as usize)
            .or(common.max_position_embeddings)
            .unwrap_or(32_768);
        let max_input_tokens = options
            .max_input_tokens
            .unwrap_or(documented_max)
            .min(common.max_position_embeddings.unwrap_or(usize::MAX))
            .max(1);

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)?;
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: max_input_tokens,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        }))?;
        tokenizer.with_padding(None);
        let pad_token_id = common
            .pad_token_id
            .or_else(|| tokenizer.get_padding().map(|p| p.pad_id))
            .or(common.eos_token_id)
            .unwrap_or(0);

        let backbone = match (&files.weights, &gguf) {
            (WeightFiles::Safetensors(paths), _) => {
                // SAFETY: the safetensors files are memory-mapped read-only
                // and are not modified while the model lives.
                let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, &device)? };
                Backbone::load(
                    &config_json,
                    Weights::dense(vb),
                    dtype,
                    &device,
                    max_input_tokens,
                )?
            }
            (WeightFiles::Gguf(_), Some(gguf)) => Backbone::load(
                &config_json,
                Weights::gguf(gguf.clone(), dtype, options.dequantize),
                dtype,
                &device,
                max_input_tokens,
            )?,
            (WeightFiles::Gguf(_), None) => unreachable!("GGUF files are opened above"),
        };

        let info = ModelInfo {
            qualified_id: format!("{PROVIDER}::{id}"),
            display_name: spec
                .map(|s| s.display_name.to_string())
                .unwrap_or_else(|| id.clone()),
            id,
            source: source.describe(),
            architecture: backbone.architecture(),
            parameters: spec.map(|s| s.parameters.to_string()),
            embedding_dimensions: backbone.hidden_size() as u32,
            max_input_tokens: max_input_tokens as u32,
            device: device_name(&device),
            dtype: dtype_name(dtype).to_string(),
            quantization: gguf.as_ref().map(|g| g.quantization()),
            dequantized: gguf.is_some() && options.dequantize,
            load_seconds: started.elapsed().as_secs_f64(),
        };

        Ok(Self {
            tokenizer,
            backbone,
            device,
            pad_token_id,
            info,
            options,
            compute: Mutex::new(()),
        })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn options(&self) -> &EmbedderOptions {
        &self.options
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run one tiny forward pass so kernels / allocators are initialised
    /// before the first real request.
    pub fn warmup(&self) -> Result<()> {
        self.embed(&["warmup".to_string()], true).map(|_| ())
    }

    /// Token ids for one text (after truncation, with special tokens).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, true)?.get_ids().to_vec())
    }

    /// Apply a full request: task instruction, batching, pooling, normalization.
    pub fn embed_request(&self, request: &EmbeddingsRequest) -> Result<EmbeddingsResponse> {
        if request.input.is_empty() {
            return Err(HarrierError::InvalidRequest(
                "`input` must contain at least one text".into(),
            ));
        }
        let resolved = resolve_inputs(request);
        let embeddings = self.embed(&resolved.texts, resolved.normalize)?;
        Ok(EmbeddingsResponse {
            external_id: None,
            output: embeddings.vectors,
            usage: Some(Usage::from_prompt_tokens(embeddings.prompt_tokens)),
        })
    }

    /// Embed texts exactly as given (no instruction handling). Vectors come
    /// back in input order.
    pub fn embed(&self, texts: &[String], normalize: bool) -> Result<Embeddings> {
        if texts.is_empty() {
            return Ok(Embeddings {
                vectors: Vec::new(),
                prompt_tokens: 0,
            });
        }
        let encodings = self.tokenizer.encode_batch(texts.to_vec(), true)?;
        let ids: Vec<&[u32]> = encodings.iter().map(|e| e.get_ids()).collect();
        let prompt_tokens: usize = ids.iter().map(|e| e.len()).sum();

        let batches = plan_batches(
            &ids.iter().map(|e| e.len()).collect::<Vec<_>>(),
            self.options.max_batch_size,
            self.options.max_batch_tokens,
        );

        let mut vectors: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        let _guard = self
            .compute
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for batch in batches {
            let rows: Vec<&[u32]> = batch.iter().map(|&i| ids[i]).collect();
            let pooled = self.forward_batch(&rows, normalize)?;
            for (row, vector) in batch.iter().zip(pooled) {
                vectors[*row] = vector;
            }
        }
        Ok(Embeddings {
            vectors,
            prompt_tokens,
        })
    }

    fn forward_batch(&self, rows: &[&[u32]], normalize: bool) -> Result<Vec<Vec<f32>>> {
        let lengths: Vec<usize> = rows.iter().map(|r| r.len().max(1)).collect();
        let seq_len = *lengths.iter().max().unwrap_or(&1);
        let mut flat = Vec::with_capacity(rows.len() * seq_len);
        for row in rows {
            flat.extend_from_slice(row);
            flat.extend(std::iter::repeat_n(self.pad_token_id, seq_len - row.len()));
        }
        let input_ids = Tensor::from_vec(flat, (rows.len(), seq_len), &self.device)?;
        let layout = BatchLayout { lengths, seq_len };
        let hidden = self.backbone.forward(&input_ids, &layout)?;
        let pooled = last_token_pool(&hidden, &layout)?.to_dtype(DType::F32)?;
        let pooled = if normalize {
            l2_normalize(&pooled)?
        } else {
            pooled
        };
        Ok(pooled.to_vec2::<f32>()?)
    }
}

/// Group row indices into forward passes: longest first, so padding waste is
/// minimal and the first batch is the worst case for memory. A row joins a
/// batch only while the batch stays under `max_batch_size` rows and
/// `max_batch_tokens` padded tokens **and** the row is at least half as long
/// as the batch's longest row — padding a 12-token text to 3 000 tokens
/// costs as much compute as a real 3 000-token text, so mixed lengths are
/// better off in separate passes.
pub fn plan_batches(
    lengths: &[usize],
    max_batch_size: usize,
    max_batch_tokens: usize,
) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by(|&a, &b| lengths[b].cmp(&lengths[a]).then(a.cmp(&b)));
    let max_batch_size = max_batch_size.max(1);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_longest = 0usize;
    for idx in order {
        let len = lengths[idx].max(1);
        let fits = current.is_empty()
            || (current.len() < max_batch_size
                && (current.len() + 1) * current_longest <= max_batch_tokens
                && len * 2 >= current_longest);
        if !fits {
            batches.push(std::mem::take(&mut current));
            current_longest = 0;
        }
        current_longest = current_longest.max(len);
        current.push(idx);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_by_size_and_tokens() {
        // lengths: 10, 6, 8, 5, 9 → sorted desc: 0(10), 4(9), 2(8), 1(6), 3(5)
        let b = plan_batches(&[10, 6, 8, 5, 9], 2, 1000);
        assert_eq!(b, vec![vec![0, 4], vec![2, 1], vec![3]]);

        // token budget 20: 2*10 = 20 fits → [0,4]; then 3*10 = 30 > 20
        let b = plan_batches(&[10, 6, 8, 5, 9], 32, 20);
        assert_eq!(b, vec![vec![0, 4], vec![2, 1], vec![3]]);

        // one input longer than the budget still runs
        let b = plan_batches(&[500, 400], 32, 100);
        assert_eq!(b, vec![vec![0], vec![1]]);

        assert!(plan_batches(&[], 32, 100).is_empty());
        assert_eq!(plan_batches(&[0], 32, 100), vec![vec![0]]);
    }

    #[test]
    fn batches_do_not_pad_short_rows_to_long_ones() {
        // A 3000-token document and three ~12-token texts: the short ones
        // must not be padded to 3000 tokens each.
        let b = plan_batches(&[12, 3000, 14, 11], 32, 16_384);
        assert_eq!(b, vec![vec![1], vec![2, 0, 3]]);
        // Rows at least half the longest still share a pass.
        let b = plan_batches(&[100, 50, 49], 32, 16_384);
        assert_eq!(b, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn derives_ids() {
        assert_eq!(
            source_id(&ModelSource::Hub {
                repo: "microsoft/harrier-oss-v1-0.6b".into(),
                revision: None
            }),
            "harrier-oss-v1-0.6b"
        );
        assert_eq!(
            source_id(&ModelSource::Local("D:/models/my-embedder".into())),
            "my-embedder"
        );
    }
}
