//! The embedding contract: the GAISe task vocabulary, the request/response
//! wire types, and how a task becomes a Harrier instruction.
//!
//! The vocabulary, aliases and wire shapes mirror
//! `gaise_core::contracts::{GaiseEmbeddingTask, GaiseEmbeddingsRequest,
//! GaiseEmbeddingsResponse}` so a GAISe client can talk to this service
//! unchanged. Harrier's own task control is a natural-language instruction
//! (`Instruct: {task}\nQuery: {text}`), so every canonical task maps to one
//! sentence and a custom task string is used as the instruction verbatim.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One string or an array of strings (`"input": "x"` or `"input": ["x", "y"]`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::One(x) => vec![x],
            Self::Many(v) => v,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Many(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Default> Default for OneOrMany<T> {
    fn default() -> Self {
        Self::One(T::default())
    }
}

impl<T> From<T> for OneOrMany<T> {
    fn from(value: T) -> Self {
        Self::One(value)
    }
}

impl<T> From<Vec<T>> for OneOrMany<T> {
    fn from(value: Vec<T>) -> Self {
        Self::Many(value)
    }
}

/// What the vectors are for — the GAISe canonical task vocabulary.
///
/// Harrier is an instruction-following embedding model: every task except
/// `document` becomes an `Instruct: …\nQuery: ` prefix on the text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum EmbeddingTask {
    /// Text being indexed for later retrieval (the default). No instruction.
    #[default]
    Document,
    /// A search query to match against indexed documents.
    Query,
    Classification,
    Clustering,
    /// Symmetric similarity between two texts.
    Similarity,
    /// A natural-language query for code retrieval.
    CodeQuery,
    FactVerification,
    QuestionAnswering,
    /// Anything GAISe has no name for. For Harrier this is the escape hatch:
    /// the raw string is used as the instruction sentence (after checking the
    /// model's named prompts such as `bitext_query`).
    Custom(String),
}

impl EmbeddingTask {
    /// Canonical names and aliases (case-insensitive, `-`/` ` treated as `_`).
    /// Unknown text becomes [`EmbeddingTask::Custom`].
    pub fn parse(value: &str) -> Self {
        let v = normalize_key(value);
        match v.as_str() {
            "document" | "retrieval_document" | "search_document" | "index" | "passage"
            | "generic_index" => Self::Document,
            "query" | "retrieval_query" | "search_query" | "search" | "generic_retrieval"
            | "text_retrieval" | "web_search_query" => Self::Query,
            "classification" | "classify" => Self::Classification,
            "clustering" | "cluster" => Self::Clustering,
            "similarity" | "semantic_similarity" | "sts" | "sentence_similarity" | "sts_query" => {
                Self::Similarity
            }
            "code_query" | "code_retrieval_query" | "code" | "code_retrieval" => Self::CodeQuery,
            "fact_verification" | "fact" | "fact_checking" => Self::FactVerification,
            "question_answering" | "qa" => Self::QuestionAnswering,
            _ => Self::Custom(value.trim().to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Document => "document",
            Self::Query => "query",
            Self::Classification => "classification",
            Self::Clustering => "clustering",
            Self::Similarity => "similarity",
            Self::CodeQuery => "code_query",
            Self::FactVerification => "fact_verification",
            Self::QuestionAnswering => "question_answering",
            Self::Custom(raw) => raw,
        }
    }

    /// True for the query side of asymmetric retrieval.
    pub fn is_query_like(&self) -> bool {
        matches!(
            self,
            Self::Query | Self::CodeQuery | Self::QuestionAnswering | Self::FactVerification
        )
    }

    /// The one-sentence instruction Harrier expects for this task, or `None`
    /// for documents (the model card: "no need to add instructions to the
    /// document side").
    ///
    /// `query` and `similarity` use the prompts shipped in the model's
    /// `config_sentence_transformers.json` (`web_search_query`, `sts_query`);
    /// the others follow the MTEB v2 prompt conventions the model was
    /// evaluated with.
    pub fn instruction(&self) -> Option<String> {
        let s = match self {
            Self::Document => return None,
            Self::Query => prompts::WEB_SEARCH_QUERY,
            Self::Similarity => prompts::STS,
            Self::Classification => prompts::CLASSIFICATION,
            Self::Clustering => prompts::CLUSTERING,
            Self::CodeQuery => prompts::CODE_QUERY,
            Self::FactVerification => prompts::FACT_VERIFICATION,
            Self::QuestionAnswering => prompts::QUESTION_ANSWERING,
            Self::Custom(raw) => {
                return Some(
                    prompts::named(raw)
                        .map(str::to_string)
                        .unwrap_or_else(|| raw.trim().to_string()),
                );
            }
        };
        Some(s.to_string())
    }
}

impl Serialize for EmbeddingTask {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for EmbeddingTask {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::parse(&String::deserialize(deserializer)?))
    }
}

fn normalize_key(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

/// Harrier's instruction sentences.
pub mod prompts {
    /// `config_sentence_transformers.json` → `web_search_query`.
    pub const WEB_SEARCH_QUERY: &str =
        "Given a web search query, retrieve relevant passages that answer the query";
    /// `config_sentence_transformers.json` → `sts_query`.
    pub const STS: &str = "Retrieve semantically similar text";
    /// `config_sentence_transformers.json` → `bitext_query`.
    pub const BITEXT: &str = "Retrieve parallel sentences";
    pub const CLASSIFICATION: &str =
        "Classify the given text into one of the predefined categories";
    pub const CLUSTERING: &str = "Identify the topic or theme of the given text";
    pub const CODE_QUERY: &str = "Given a code search query, retrieve relevant code snippets or documentation that answer the query";
    /// MTEB FEVER / Climate-FEVER prompt.
    pub const FACT_VERIFICATION: &str =
        "Given a claim, retrieve documents that support or refute the claim";
    /// MTEB `AlloprofReranking-query` prompt in the model's own eval file.
    pub const QUESTION_ANSWERING: &str =
        "Given a question, retrieve passages that answer the question";

    /// The prompt names the model card documents
    /// (`model.encode(queries, prompt_name=…)`), resolved for custom tasks.
    pub fn named(name: &str) -> Option<&'static str> {
        match super::normalize_key(name).as_str() {
            "web_search_query" => Some(WEB_SEARCH_QUERY),
            "sts_query" => Some(STS),
            "bitext_query" | "bitext" | "bitext_mining" | "parallel_sentences" => Some(BITEXT),
            _ => None,
        }
    }

    /// Harrier's prompt template.
    pub fn format(instruction: &str, text: &str) -> String {
        format!("Instruct: {instruction}\nQuery: {text}")
    }
}

/// `POST /v1/embeddings` body — wire-compatible with `GaiseEmbeddingsRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingsRequest {
    /// `harrier::harrier-oss-v1-0.6b`, `harrier-oss-v1-0.6b`,
    /// `microsoft/harrier-oss-v1-0.6b`, or empty for the service default.
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    /// One string or an array.
    pub input: OneOrMany<String>,
    /// Intended use of the vectors: `document` (default), `query`,
    /// `classification`, `clustering`, `similarity`, `code_query`,
    /// `fact_verification`, `question_answering`, their aliases, a Harrier
    /// prompt name (`web_search_query`, `sts_query`, `bitext_query`), or any
    /// free-text instruction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<EmbeddingTask>,
    /// Harrier extension: an explicit instruction sentence. Overrides the
    /// task-derived one and is applied to every input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    /// Accepted for compatibility. Harrier vectors are fixed-size, so this is
    /// ignored (GAISe's `fixed` dimension rule).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Unit-length vectors. Unset or `true` → L2-normalized (the model's
    /// documented output); `false` → the raw pooled hidden state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
}

/// Token counters — wire-compatible with `GaiseUsage`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<BTreeMap<String, usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<BTreeMap<String, usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<BTreeMap<String, usize>>,
}

impl Usage {
    pub fn from_prompt_tokens(prompt_tokens: usize) -> Self {
        Self {
            input: Some(BTreeMap::from([(
                "prompt_tokens".to_string(),
                prompt_tokens,
            )])),
            output: None,
            total: Some(BTreeMap::from([(
                "total_tokens".to_string(),
                prompt_tokens,
            )])),
        }
    }
}

/// `POST /v1/embeddings` response — wire-compatible with `GaiseEmbeddingsResponse`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingsResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
    /// One vector per input, in input order.
    pub output: Vec<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// The texts to run through the model after the task has been applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedInputs {
    /// Inputs with the instruction prefix applied where one applies.
    pub texts: Vec<String>,
    /// The instruction that was used, if any.
    pub instruction: Option<String>,
    /// Whether to L2-normalize the pooled vectors.
    pub normalize: bool,
}

/// Apply the task / instruction / normalize rules of a request.
pub fn resolve_inputs(request: &EmbeddingsRequest) -> ResolvedInputs {
    let inputs: Vec<String> = match &request.input {
        OneOrMany::One(s) => vec![s.clone()],
        OneOrMany::Many(v) => v.clone(),
    };
    let instruction = match &request.instruction {
        Some(explicit) if !explicit.trim().is_empty() => Some(explicit.trim().to_string()),
        _ => request.task.as_ref().and_then(EmbeddingTask::instruction),
    };
    let texts = match &instruction {
        Some(instr) => inputs.iter().map(|t| prompts::format(instr, t)).collect(),
        None => inputs,
    };
    ResolvedInputs {
        texts,
        instruction,
        normalize: request.normalize.unwrap_or(true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gaise_aliases() {
        assert_eq!(EmbeddingTask::parse("search_query"), EmbeddingTask::Query);
        assert_eq!(
            EmbeddingTask::parse("RETRIEVAL_DOCUMENT"),
            EmbeddingTask::Document
        );
        assert_eq!(
            EmbeddingTask::parse("semantic-similarity"),
            EmbeddingTask::Similarity
        );
        assert_eq!(
            EmbeddingTask::parse("Code Retrieval"),
            EmbeddingTask::CodeQuery
        );
        assert_eq!(EmbeddingTask::parse("qa"), EmbeddingTask::QuestionAnswering);
        assert_eq!(
            EmbeddingTask::parse("fact"),
            EmbeddingTask::FactVerification
        );
        assert_eq!(EmbeddingTask::parse("cluster"), EmbeddingTask::Clustering);
        assert_eq!(
            EmbeddingTask::parse("classify"),
            EmbeddingTask::Classification
        );
        assert_eq!(
            EmbeddingTask::parse("IMAGE_RETRIEVAL"),
            EmbeddingTask::Custom("IMAGE_RETRIEVAL".into())
        );
        let parsed: EmbeddingTask = serde_json::from_str("\"search-query\"").unwrap();
        assert_eq!(parsed, EmbeddingTask::Query);
        assert_eq!(
            serde_json::to_string(&EmbeddingTask::CodeQuery).unwrap(),
            "\"code_query\""
        );
    }

    #[test]
    fn harrier_prompt_names_resolve() {
        assert_eq!(
            EmbeddingTask::parse("web_search_query"),
            EmbeddingTask::Query
        );
        assert_eq!(EmbeddingTask::parse("sts_query"), EmbeddingTask::Similarity);
        let bitext = EmbeddingTask::parse("bitext_query");
        assert_eq!(bitext, EmbeddingTask::Custom("bitext_query".into()));
        assert_eq!(bitext.instruction().as_deref(), Some(prompts::BITEXT));
        assert_eq!(bitext.as_str(), "bitext_query");
    }

    #[test]
    fn instructions_follow_the_model_card() {
        assert_eq!(EmbeddingTask::Document.instruction(), None);
        assert_eq!(
            EmbeddingTask::Query.instruction().as_deref(),
            Some(prompts::WEB_SEARCH_QUERY)
        );
        assert_eq!(
            EmbeddingTask::Similarity.instruction().as_deref(),
            Some(prompts::STS)
        );
        assert_eq!(
            EmbeddingTask::Custom("Given a legal question, retrieve the relevant statute".into())
                .instruction()
                .as_deref(),
            Some("Given a legal question, retrieve the relevant statute")
        );
        assert_eq!(
            prompts::format(prompts::WEB_SEARCH_QUERY, "summit define"),
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: summit define"
        );
    }

    #[test]
    fn resolves_requests() {
        let r = EmbeddingsRequest {
            input: vec!["a".to_string(), "b".to_string()].into(),
            task: Some(EmbeddingTask::Query),
            ..Default::default()
        };
        let resolved = resolve_inputs(&r);
        assert_eq!(
            resolved.texts,
            vec![
                format!("Instruct: {}\nQuery: a", prompts::WEB_SEARCH_QUERY),
                format!("Instruct: {}\nQuery: b", prompts::WEB_SEARCH_QUERY),
            ]
        );
        assert!(resolved.normalize);

        let r = EmbeddingsRequest {
            input: "doc".to_string().into(),
            normalize: Some(false),
            ..Default::default()
        };
        let resolved = resolve_inputs(&r);
        assert_eq!(resolved.texts, vec!["doc"]);
        assert_eq!(resolved.instruction, None);
        assert!(!resolved.normalize);

        let r = EmbeddingsRequest {
            input: "x".to_string().into(),
            task: Some(EmbeddingTask::Query),
            instruction: Some("Find the matching FAQ entry".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_inputs(&r).texts,
            vec!["Instruct: Find the matching FAQ entry\nQuery: x"]
        );
    }

    #[test]
    fn wire_shapes_match_gaise() {
        let json = r#"{"model":"harrier::harrier-oss-v1-0.6b","input":["a","b"],"task":"search_query","dimensions":256}"#;
        let r: EmbeddingsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(r.input.len(), 2);
        assert_eq!(r.task, Some(EmbeddingTask::Query));
        assert_eq!(r.dimensions, Some(256));

        let json = r#"{"input":"single"}"#;
        let r: EmbeddingsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(r.input, OneOrMany::One("single".into()));
        assert!(r.model.is_empty());

        let resp = EmbeddingsResponse {
            external_id: None,
            output: vec![vec![0.5, 0.5]],
            usage: Some(Usage::from_prompt_tokens(3)),
        };
        assert_eq!(
            serde_json::to_string(&resp).unwrap(),
            r#"{"output":[[0.5,0.5]],"usage":{"input":{"prompt_tokens":3},"total":{"total_tokens":3}}}"#
        );
    }
}
