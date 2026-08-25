//! Axum HTTP service for Harrier embeddings, wire-compatible with GAISe's
//! `POST /v1/embeddings` and `GET /v1/models`.
//!
//! | Method | Path | Body | Returns |
//! |---|---|---|---|
//! | `POST` | `/v1/embeddings` | [`EmbeddingsRequest`] | [`EmbeddingsResponse`] |
//! | `GET` | `/v1/models` | — | `{ "models": [ModelRecord] }` |
//! | `GET` | `/v1/models/{id}` | — | `ModelRecord` |
//! | `GET` | `/health` | — | `{ "status": "ok", … }` |

use axum::extract::{Path, Query, State};
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use harrier_core::{
    Embedder, EmbeddingsRequest, EmbeddingsResponse, HarrierError, ModelInfo, PROVIDER, prompts,
    registry,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Loaded models plus the default one.
pub struct AppState {
    /// Keyed by bare id (`harrier-oss-v1-0.6b`).
    pub models: BTreeMap<String, Arc<Embedder>>,
    pub default_model: String,
    pub started: Instant,
    /// Log one line per request to stderr.
    pub log_requests: bool,
}

impl AppState {
    /// Build state from loaded embedders; the first one is the default.
    pub fn new(embedders: Vec<Arc<Embedder>>) -> Self {
        let default_model = embedders
            .first()
            .map(|e| e.info().id.clone())
            .unwrap_or_default();
        let models = embedders
            .into_iter()
            .map(|e| (e.info().id.clone(), e))
            .collect();
        Self {
            models,
            default_model,
            started: Instant::now(),
            log_requests: true,
        }
    }

    /// Find a loaded model by any spelling GAISe or a user might send.
    pub fn model(&self, name: &str) -> Option<&Arc<Embedder>> {
        let bare = registry::bare_id(name);
        if bare.is_empty() {
            return self.models.get(&self.default_model);
        }
        if let Some(m) = self.models.get(bare) {
            return Some(m);
        }
        let lower = bare.to_ascii_lowercase();
        if let Some(spec) = registry::resolve(bare)
            && let Some(m) = self.models.get(spec.id)
        {
            return Some(m);
        }
        self.models.values().find(|m| {
            let info = m.info();
            info.id.eq_ignore_ascii_case(&lower)
                || info.qualified_id.eq_ignore_ascii_case(&lower)
                || info.source.eq_ignore_ascii_case(&lower)
        })
    }

    pub fn loaded_ids(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

/// The router, with state attached.
pub fn create_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/embeddings", post(handle_embeddings))
        .route("/v1/models", get(handle_list_models))
        .route("/v1/models/{model}", get(handle_get_model))
        .route("/health", get(handle_health))
        .route("/", get(handle_health))
        .layer(middleware::from_fn_with_state(state.clone(), log_request))
        .with_state(state)
}

async fn log_request(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let started = Instant::now();
    let response = next.run(request).await;
    if state.log_requests {
        eprintln!(
            "{} {} {} {:.1}ms",
            method,
            path,
            response.status().as_u16(),
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    response
}

/// Error response: status + plain-text message, like gaise-api.
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

impl From<HarrierError> for ApiError {
    fn from(e: HarrierError) -> Self {
        let status = match &e {
            HarrierError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self(status, e.to_string())
    }
}

async fn handle_embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    let embedder = state.model(&request.model).cloned().ok_or_else(|| {
        ApiError(
            StatusCode::NOT_FOUND,
            format!(
                "model `{}` is not loaded (loaded: {})",
                request.model,
                state.loaded_ids().join(", ")
            ),
        )
    })?;
    if request.input.is_empty() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "`input` must contain at least one text".into(),
        ));
    }
    let response = tokio::task::spawn_blocking(move || embedder.embed_request(&request))
        .await
        .map_err(|e| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("embedding task failed: {e}"),
            )
        })??;
    Ok(Json(response))
}

#[derive(Debug, Deserialize, Default)]
pub struct ListModelsQuery {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub operation: Option<String>,
}

async fn handle_list_models(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListModelsQuery>,
) -> Result<Json<ListModelsResponse>, ApiError> {
    if let Some(op) = query.operation.as_deref() {
        let op = op.trim().to_ascii_lowercase();
        if !matches!(
            op.as_str(),
            "embeddings"
                | "embedding"
                | "embed"
                | "instruct"
                | "instruct_stream"
                | "speech"
                | "live"
        ) {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                format!("unknown operation `{op}`"),
            ));
        }
        if !matches!(op.as_str(), "embeddings" | "embedding" | "embed") {
            return Ok(Json(ListModelsResponse::default()));
        }
    }
    if let Some(p) = query.provider.as_deref()
        && !p.trim().eq_ignore_ascii_case(PROVIDER)
    {
        return Ok(Json(ListModelsResponse::default()));
    }
    let models = state
        .models
        .values()
        .map(|m| ModelRecord::from_info(m.info(), m.info().id == state.default_model))
        .collect();
    Ok(Json(ListModelsResponse {
        models,
        errors: Vec::new(),
    }))
}

async fn handle_get_model(
    State(state): State<Arc<AppState>>,
    Path(model): Path<String>,
) -> Result<Json<ModelRecord>, ApiError> {
    let embedder = state.model(&model).ok_or_else(|| {
        ApiError(
            StatusCode::NOT_FOUND,
            format!("model `{model}` is not loaded"),
        )
    })?;
    Ok(Json(ModelRecord::from_info(
        embedder.info(),
        embedder.info().id == state.default_model,
    )))
}

#[derive(Debug, Serialize)]
pub struct Health {
    pub status: &'static str,
    pub service: &'static str,
    pub version: &'static str,
    pub uptime_seconds: u64,
    pub default_model: String,
    pub models: Vec<String>,
    pub backends: &'static [&'static str],
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<Health> {
    Json(Health {
        status: "ok",
        service: "harrier_api",
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: state.started.elapsed().as_secs(),
        default_model: state.default_model.clone(),
        models: state.loaded_ids(),
        backends: harrier_core::compiled_backends(),
    })
}

/// `GET /v1/models` body — the shape of `GaiseListModelsResponse`.
#[derive(Debug, Serialize, Default)]
pub struct ListModelsResponse {
    pub models: Vec<ModelRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<serde_json::Value>,
}

/// One model, in the shape of `GaiseModel` plus an `embedding` profile
/// (GAISe's `EmbeddingProfile`) and the local `runtime` facts.
#[derive(Debug, Serialize)]
pub struct ModelRecord {
    pub id: String,
    pub provider: &'static str,
    pub display_name: String,
    pub status: &'static str,
    pub capabilities: Capabilities,
    pub limits: Limits,
    pub embedding: EmbeddingProfile,
    pub runtime: Runtime,
}

#[derive(Debug, Serialize)]
pub struct Capabilities {
    pub input: &'static [&'static str],
    pub output: &'static [&'static str],
    pub operations: &'static [&'static str],
    pub tools: &'static str,
    pub reasoning: &'static str,
    pub structured_output: &'static str,
    pub sources: &'static [&'static str],
}

#[derive(Debug, Serialize)]
pub struct Limits {
    pub max_input_tokens: u32,
    pub embedding_dimensions: u32,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingProfile {
    pub default_dimensions: u32,
    pub dimension_rule: &'static str,
    pub max_input_tokens: u32,
    pub normalized_output: bool,
    pub normalizes_truncation: bool,
    pub task_control: TaskControl,
    /// The canonical GAISe tasks and the instruction each one becomes.
    pub task_instructions: BTreeMap<&'static str, Option<String>>,
}

#[derive(Debug, Serialize)]
pub struct TaskControl {
    pub query_instruction: String,
}

#[derive(Debug, Serialize)]
pub struct Runtime {
    pub architecture: harrier_core::Architecture,
    pub source: String,
    pub device: String,
    /// Activation dtype (and weight dtype for safetensors checkpoints).
    pub dtype: String,
    /// GGUF weight quantisation (`q8_0`), `null` for dense weights.
    pub quantization: Option<String>,
    /// GGUF weights were dequantised to `dtype` at load (`HARRIER_DEQUANTIZE`).
    pub dequantized: bool,
    pub parameters: Option<String>,
    pub load_seconds: f64,
    pub default: bool,
}

impl ModelRecord {
    pub fn from_info(info: &ModelInfo, default: bool) -> Self {
        use harrier_core::EmbeddingTask as T;
        let tasks = [
            ("document", T::Document),
            ("query", T::Query),
            ("classification", T::Classification),
            ("clustering", T::Clustering),
            ("similarity", T::Similarity),
            ("code_query", T::CodeQuery),
            ("fact_verification", T::FactVerification),
            ("question_answering", T::QuestionAnswering),
        ];
        Self {
            id: info.qualified_id.clone(),
            provider: PROVIDER,
            display_name: info.display_name.clone(),
            status: "active",
            capabilities: Capabilities {
                input: &["text"],
                output: &["embedding"],
                operations: &["embeddings"],
                tools: "unsupported",
                reasoning: "unsupported",
                structured_output: "unsupported",
                sources: &["registry", "provider"],
            },
            limits: Limits {
                max_input_tokens: info.max_input_tokens,
                embedding_dimensions: info.embedding_dimensions,
            },
            embedding: EmbeddingProfile {
                default_dimensions: info.embedding_dimensions,
                dimension_rule: "fixed",
                max_input_tokens: info.max_input_tokens,
                normalized_output: true,
                normalizes_truncation: true,
                task_control: TaskControl {
                    query_instruction: prompts::format(prompts::WEB_SEARCH_QUERY, ""),
                },
                task_instructions: tasks
                    .into_iter()
                    .map(|(name, task)| (name, task.instruction()))
                    .collect(),
            },
            runtime: Runtime {
                architecture: info.architecture,
                source: info.source.clone(),
                device: info.device.clone(),
                dtype: info.dtype.clone(),
                quantization: info.quantization.clone(),
                dequantized: info.dequantized,
                parameters: info.parameters.clone(),
                load_seconds: info.load_seconds,
                default,
            },
        }
    }
}

/// Operator configuration, read from the environment by the binary.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Models to load; the first is the default.
    pub models: Vec<String>,
    /// Where exported models live (`HARRIER_MODEL_DIR`, default `./models`).
    pub model_dir: std::path::PathBuf,
    pub options: harrier_core::EmbedderOptions,
    pub warmup: bool,
    pub log_requests: bool,
}

impl ServerConfig {
    pub const ENV_HELP: &'static str = "\
HARRIER_MODELS            comma-separated models to load; first is the default
                          (registry id, alias like 270m / 0.6b, Hub repo[@rev], local dir,
                          `<model>:<quant>` for a GGUF quantisation such as 0.6b:q8_0,
                          or a path to a .gguf file)
                          default: harrier-oss-v1-0.6b
HARRIER_MODEL             single-model alias for HARRIER_MODELS
HARRIER_MODEL_DIR         directory of exported models, checked before the Hub (default ./models)
HARRIER_DEVICE            auto | cpu | cuda[:n] | metal[:n] | gpu      (default auto)
HARRIER_DTYPE             f32 | bf16 | f16      (default: bf16 on GPU, f32 on CPU;
                          f32 activations for GGUF models)
HARRIER_DEQUANTIZE        dequantise GGUF weights to HARRIER_DTYPE at load: dense-speed
                          matmuls, dense-size memory (default false)
HARRIER_MAX_INPUT_TOKENS  truncate inputs to this many tokens (default: model max, 32768)
HARRIER_MAX_BATCH_SIZE    rows per forward pass (default 32)
HARRIER_MAX_BATCH_TOKENS  padded tokens per forward pass (default 16384)
HARRIER_WARMUP            run one forward pass at startup (default true)
HARRIER_LOG_REQUESTS      log one line per request (default true)
HARRIER_HOST              bind address (default 0.0.0.0)
HARRIER_PORT / PORT       listen port (default 3000)
HF_HOME / HF_TOKEN        Hugging Face cache dir / token for downloads";

    pub fn from_env() -> Result<Self, String> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let models: Vec<String> = var("HARRIER_MODELS")
            .or_else(|| var("HARRIER_MODEL"))
            .unwrap_or_else(|| harrier_core::DEFAULT_MODEL_ID.to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if models.is_empty() {
            return Err("HARRIER_MODELS is empty".into());
        }
        let parse_usize = |k: &str| -> Result<Option<usize>, String> {
            var(k)
                .map(|v| {
                    v.parse::<usize>()
                        .map_err(|_| format!("{k} must be an integer, got `{v}`"))
                })
                .transpose()
        };
        let parse_bool = |k: &str, default: bool| -> Result<bool, String> {
            match var(k).map(|v| v.to_ascii_lowercase()) {
                None => Ok(default),
                Some(v) if ["1", "true", "yes", "on"].contains(&v.as_str()) => Ok(true),
                Some(v) if ["0", "false", "no", "off"].contains(&v.as_str()) => Ok(false),
                Some(v) => Err(format!("{k} must be a boolean, got `{v}`")),
            }
        };
        let defaults = harrier_core::EmbedderOptions::default();
        let options = harrier_core::EmbedderOptions {
            device: var("HARRIER_DEVICE")
                .unwrap_or_default()
                .parse::<harrier_core::DeviceRequest>()
                .map_err(|e| e.to_string())?,
            dtype: var("HARRIER_DTYPE")
                .map(|v| harrier_core::parse_dtype(&v))
                .transpose()
                .map_err(|e| e.to_string())?,
            max_input_tokens: parse_usize("HARRIER_MAX_INPUT_TOKENS")?,
            dequantize: parse_bool("HARRIER_DEQUANTIZE", false)?,
            max_batch_size: parse_usize("HARRIER_MAX_BATCH_SIZE")?
                .unwrap_or(defaults.max_batch_size),
            max_batch_tokens: parse_usize("HARRIER_MAX_BATCH_TOKENS")?
                .unwrap_or(defaults.max_batch_tokens),
        };
        let port = var("HARRIER_PORT")
            .or_else(|| var("PORT"))
            .map(|v| {
                v.parse::<u16>()
                    .map_err(|_| format!("HARRIER_PORT must be a port number, got `{v}`"))
            })
            .transpose()?
            .unwrap_or(3000);
        Ok(Self {
            host: var("HARRIER_HOST").unwrap_or_else(|| "0.0.0.0".into()),
            port,
            models,
            model_dir: harrier_core::model_dir(),
            options,
            warmup: parse_bool("HARRIER_WARMUP", true)?,
            log_requests: parse_bool("HARRIER_LOG_REQUESTS", true)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_record_has_gaise_shape() {
        let info = ModelInfo {
            id: "harrier-oss-v1-0.6b".into(),
            qualified_id: "harrier::harrier-oss-v1-0.6b".into(),
            display_name: "Harrier OSS v1 0.6B".into(),
            source: "microsoft/harrier-oss-v1-0.6b".into(),
            architecture: harrier_core::Architecture::Qwen3,
            parameters: Some("0.6B".into()),
            embedding_dimensions: 1024,
            max_input_tokens: 32768,
            device: "cpu".into(),
            dtype: "f32".into(),
            quantization: None,
            dequantized: false,
            load_seconds: 1.0,
        };
        let record = serde_json::to_value(ModelRecord::from_info(&info, true)).unwrap();
        assert_eq!(record["id"], "harrier::harrier-oss-v1-0.6b");
        assert_eq!(record["provider"], "harrier");
        assert_eq!(record["capabilities"]["operations"][0], "embeddings");
        assert_eq!(record["limits"]["embedding_dimensions"], 1024);
        assert_eq!(record["embedding"]["dimension_rule"], "fixed");
        assert_eq!(
            record["embedding"]["task_instructions"]["document"],
            serde_json::Value::Null
        );
        assert_eq!(record["runtime"]["architecture"], "qwen3");
        assert_eq!(record["runtime"]["default"], true);
        assert_eq!(record["runtime"]["quantization"], serde_json::Value::Null);
    }

    #[test]
    fn empty_state_resolves_nothing() {
        let state = AppState::new(Vec::new());
        assert!(state.model("").is_none());
        assert!(state.model("harrier::harrier-oss-v1-0.6b").is_none());
        assert!(state.loaded_ids().is_empty());
    }
}
