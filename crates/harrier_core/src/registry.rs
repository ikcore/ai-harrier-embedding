//! The Harrier model family: ids, Hub repos, backbones and documented limits.
//!
//! Anything not in this table can still be loaded by repo id, local
//! directory or `.gguf` path — the backbone is detected from `config.json` /
//! the GGUF metadata — but these entries give the short names (`0.6b`,
//! `harrier::harrier-oss-v1-27b`), the advertised limits for
//! `GET /v1/models`, and where the GGUF quantisations (`0.6b:q8_0`) live.

use serde::{Deserialize, Serialize};

/// The provider key used in `provider::model` ids.
pub const PROVIDER: &str = "harrier";

/// Transformer backbone a checkpoint uses (read from `config.json`
/// `model_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    /// `Qwen3Model` — harrier-oss-v1-0.6b.
    Qwen3,
    /// `Gemma3TextModel` — harrier-oss-v1-270m (and the 27b).
    Gemma3,
}

impl Architecture {
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen3" => Some(Self::Qwen3),
            "gemma3_text" | "gemma3" => Some(Self::Gemma3),
            _ => None,
        }
    }
}

/// A GGUF conversion of a checkpoint: one Hub repo holding several
/// quantisations of the same weights.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GgufSpec {
    /// Hugging Face repo, e.g. `cstr/harrier-0.6b-GGUF`.
    pub repo: &'static str,
    /// `(quantisation tag, file name)` pairs; tags are what `<id>:<tag>`
    /// accepts (`q8_0`, `q4_k`, `f32`).
    pub files: &'static [(&'static str, &'static str)],
}

impl GgufSpec {
    pub fn file(&self, tag: &str) -> Option<&'static str> {
        self.files
            .iter()
            .find(|(t, _)| t.eq_ignore_ascii_case(tag))
            .map(|(_, f)| *f)
    }

    pub fn tags(&self) -> Vec<&'static str> {
        self.files.iter().map(|(t, _)| *t).collect()
    }
}

/// One known checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelSpec {
    /// Bare id, e.g. `harrier-oss-v1-0.6b`.
    pub id: &'static str,
    /// Hugging Face repo, e.g. `microsoft/harrier-oss-v1-0.6b`.
    pub repo: &'static str,
    pub display_name: &'static str,
    pub architecture: Architecture,
    pub parameters: &'static str,
    pub embedding_dimensions: u32,
    /// Documented maximum input length in tokens.
    pub max_input_tokens: u32,
    /// Approximate weight size in bf16, for the operator's benefit.
    pub weights_gb: f32,
    /// Short aliases accepted by [`resolve`].
    pub aliases: &'static [&'static str],
    /// GGUF quantisations, when a conversion is published.
    pub gguf: Option<GgufSpec>,
}

impl ModelSpec {
    /// `harrier::harrier-oss-v1-0.6b`.
    pub fn qualified_id(&self) -> String {
        format!("{PROVIDER}::{}", self.id)
    }
}

/// The supported checkpoints. The 270M and 0.6B are the primary targets
/// (validated against `transformers`, see `tests/reference.rs`); the 27B
/// loads through the same Gemma 3 code path but needs ~54 GB of bf16 memory.
/// Other Qwen3 / Gemma 3 embedding checkpoints load by full repo id or
/// local path.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "harrier-oss-v1-270m",
        repo: "microsoft/harrier-oss-v1-270m",
        display_name: "Harrier OSS v1 270M",
        architecture: Architecture::Gemma3,
        parameters: "270M",
        embedding_dimensions: 640,
        max_input_tokens: 32768,
        weights_gb: 0.54,
        aliases: &["270m"],
        gguf: Some(GgufSpec {
            repo: "cstr/harrier-270m-GGUF",
            files: &[
                ("q8_0", "harrier-270m-q8_0.gguf"),
                ("q4_k", "harrier-270m-q4_k.gguf"),
                ("f32", "harrier-270m.gguf"),
            ],
        }),
    },
    ModelSpec {
        id: "harrier-oss-v1-0.6b",
        repo: "microsoft/harrier-oss-v1-0.6b",
        display_name: "Harrier OSS v1 0.6B",
        architecture: Architecture::Qwen3,
        parameters: "0.6B",
        embedding_dimensions: 1024,
        max_input_tokens: 32768,
        weights_gb: 1.2,
        aliases: &["0.6b", "600m"],
        gguf: Some(GgufSpec {
            repo: "cstr/harrier-0.6b-GGUF",
            files: &[
                ("q8_0", "harrier-0.6b-q8_0.gguf"),
                ("q4_k", "harrier-0.6b-q4_k.gguf"),
                ("f32", "harrier-0.6b.gguf"),
            ],
        }),
    },
    ModelSpec {
        id: "harrier-oss-v1-27b",
        repo: "microsoft/harrier-oss-v1-27b",
        display_name: "Harrier OSS v1 27B",
        architecture: Architecture::Gemma3,
        parameters: "27B",
        embedding_dimensions: 5376,
        max_input_tokens: 32768,
        weights_gb: 54.0,
        aliases: &["27b"],
        gguf: None,
    },
];

/// The default model when the operator configures none.
pub const DEFAULT_MODEL_ID: &str = "harrier-oss-v1-0.6b";

/// Strip a `provider::` prefix (any provider — a GAISe router may forward
/// `harrier::x`, a user may type `x`).
pub fn bare_id(name: &str) -> &str {
    let name = name.trim();
    match name.split_once("::") {
        Some((_, rest)) => rest.trim(),
        None => name,
    }
}

/// Split a `<name>[:<quant>]` model name into the checkpoint name and the
/// GGUF quantisation tag (`0.6b:q8_0` → `("0.6b", Some("q8_0"))`). A tag is
/// a short identifier after the last `:`; Windows drive letters
/// (`D:/models`) and `.gguf` paths are left alone.
pub fn split_quant(name: &str) -> (&str, Option<&str>) {
    let name = name.trim();
    if name.to_ascii_lowercase().ends_with(".gguf") {
        return (name, None);
    }
    match name.rsplit_once(':') {
        Some((base, tag))
            if !base.is_empty()
                && !tag.is_empty()
                && !base.ends_with(':')
                && tag.len() <= 12
                && tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') =>
        {
            (base, Some(tag))
        }
        _ => (name, None),
    }
}

/// Find a known checkpoint by qualified id, bare id, Hub repo, or alias
/// (case-insensitive). A `:<quant>` suffix is ignored.
pub fn resolve(name: &str) -> Option<&'static ModelSpec> {
    let (name, _) = split_quant(bare_id(name));
    let bare = name.to_ascii_lowercase();
    if bare.is_empty() {
        return None;
    }
    MODELS.iter().find(|m| {
        m.id == bare
            || m.repo.to_ascii_lowercase() == bare
            || m.aliases.iter().any(|a| *a == bare)
            || bare == format!("harrier-{}", m.aliases[0])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_every_spelling() {
        for name in [
            "harrier-oss-v1-0.6b",
            "harrier::harrier-oss-v1-0.6b",
            "microsoft/harrier-oss-v1-0.6b",
            "Microsoft/Harrier-OSS-v1-0.6B",
            "0.6b",
            "harrier-0.6b",
            "ollama::harrier-oss-v1-0.6b",
        ] {
            assert_eq!(
                resolve(name).map(|m| m.id),
                Some("harrier-oss-v1-0.6b"),
                "{name}"
            );
        }
        assert_eq!(
            resolve("27b").map(|m| m.architecture),
            Some(Architecture::Gemma3)
        );
        assert_eq!(resolve("270m").map(|m| m.embedding_dimensions), Some(640));
        assert_eq!(
            resolve("harrier-270m").map(|m| m.id),
            Some("harrier-oss-v1-270m")
        );
        assert_eq!(MODELS.len(), 3);
        assert_eq!(resolve("nope"), None);
        assert_eq!(resolve(""), None);
        assert_eq!(bare_id("  harrier::x  "), "x");
        assert_eq!(
            resolve("harrier::0.6b:q8_0").map(|m| m.id),
            Some("harrier-oss-v1-0.6b")
        );
    }

    #[test]
    fn splits_quant_suffixes() {
        assert_eq!(split_quant("0.6b:q8_0"), ("0.6b", Some("q8_0")));
        assert_eq!(
            split_quant("harrier-oss-v1-270m:Q4_K"),
            ("harrier-oss-v1-270m", Some("Q4_K"))
        );
        assert_eq!(split_quant("0.6b"), ("0.6b", None));
        assert_eq!(split_quant("D:/models/x.gguf"), ("D:/models/x.gguf", None));
        assert_eq!(split_quant("D:/models/dir"), ("D:/models/dir", None));
        assert_eq!(split_quant("D:\\models"), ("D:\\models", None));
        assert_eq!(split_quant("owner/repo@rev"), ("owner/repo@rev", None));
        assert_eq!(
            resolve("0.6b").unwrap().gguf.as_ref().unwrap().file("Q8_0"),
            Some("harrier-0.6b-q8_0.gguf")
        );
        assert!(resolve("27b").unwrap().gguf.is_none());
    }
}
