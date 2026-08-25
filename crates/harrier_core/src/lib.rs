//! # harrier_core
//!
//! Microsoft's [Harrier OSS v1](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)
//! text-embedding models on [candle](https://github.com/huggingface/candle),
//! running on CPU, CUDA (`--features cuda`) and Metal (`--features metal`),
//! from the original safetensors weights or an 8-bit GGUF (`0.6b:q8_0`).
//!
//! - `harrier-oss-v1-270m` — Gemma 3 backbone, 640-d vectors
//! - `harrier-oss-v1-0.6b` — Qwen3 backbone, 1 024-d vectors
//! - `harrier-oss-v1-27b` — Gemma 3 backbone, 5 376-d vectors (same code
//!   path as the 270M; needs ~54 GB of bf16 memory)
//!
//! Both backbones are implemented here directly on `candle-nn` primitives
//! (no `candle-transformers`), with real attention masks so one or many
//! inputs can be embedded per forward pass: right padding, causal masking,
//! Gemma's sliding-window layers, last-token pooling and L2 normalization
//! follow the `transformers` reference implementation.
//!
//! The task vocabulary ([`EmbeddingTask`]) and the request/response types are
//! wire-compatible with GAISe's embedding contract; every task becomes the
//! `Instruct: …\nQuery: ` prefix Harrier was trained with.
//!
//! ```no_run
//! use harrier_core::{Embedder, EmbedderOptions, EmbeddingsRequest, EmbeddingTask};
//!
//! let embedder = Embedder::load("harrier-oss-v1-0.6b", EmbedderOptions::default())?;
//! let response = embedder.embed_request(&EmbeddingsRequest {
//!     input: vec!["how much protein should a female eat".to_string()].into(),
//!     task: Some(EmbeddingTask::Query),
//!     ..Default::default()
//! })?;
//! assert_eq!(response.output[0].len(), 1024);
//! # Ok::<(), harrier_core::HarrierError>(())
//! ```

pub mod device;
pub mod embedder;
pub mod error;
pub mod loader;
pub mod models;
pub mod registry;
pub mod task;

pub use candle_core::{DType, Device};
pub use device::{DeviceRequest, compiled_backends, parse_dtype, select_device};
pub use embedder::{Embedder, EmbedderOptions, Embeddings, ModelInfo};
pub use error::{HarrierError, Result};
pub use loader::{
    GgufWeights, MODEL_DIR_ENV, ModelFiles, ModelSource, WeightFiles, export_to_dir, model_dir,
    resolve_source,
};
pub use registry::{
    Architecture, DEFAULT_MODEL_ID, GgufSpec, MODELS, ModelSpec, PROVIDER, split_quant,
};
pub use task::{
    EmbeddingTask, EmbeddingsRequest, EmbeddingsResponse, OneOrMany, ResolvedInputs, Usage,
    prompts, resolve_inputs,
};
