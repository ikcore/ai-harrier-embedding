//! Locating model files: a local directory, a Hugging Face Hub repo
//! (downloaded once into the Hub cache, `HF_HOME` / `HF_TOKEN` honoured), or
//! a GGUF file — either an explicit `.gguf` path or a registry model's
//! published quantisation (`0.6b:q8_0`).
//!
//! The recommended layout for deployment is a flat **model directory** next
//! to the project (`./models/<id>/config.json`, `tokenizer.json`,
//! `*.safetensors` and/or `*.gguf`), created with [`export_to_dir`] /
//! `harrier_api download`. [`resolve_source`] checks it before going to the
//! Hub, so a populated directory means no network at startup — which is
//! what a Docker image or an air-gapped box wants.

use crate::error::{HarrierError, Result};
use crate::registry::{self, split_quant};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Environment variable naming the model directory; default `./models`.
pub const MODEL_DIR_ENV: &str = "HARRIER_MODEL_DIR";

/// The model directory: `$HARRIER_MODEL_DIR`, else `./models`.
pub fn model_dir() -> PathBuf {
    std::env::var_os(MODEL_DIR_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// Turn a model name into a source, preferring local copies:
///
/// 1. a `.gguf` file path as given (tokenizer / config beside it);
/// 2. a directory path as given;
/// 3. `<model_dir>/<bare id>` (for registry ids and aliases, the registry id)
///    when it holds a `config.json`;
/// 4. the registry's Hub repo for known ids / aliases;
/// 5. otherwise the name is a Hub repo id (`owner/name[@rev]`).
///
/// A `:<quant>` suffix (`0.6b:q8_0`) selects that GGUF quantisation: the
/// file from `<model_dir>/<id>/` when present, else the registry's GGUF
/// repo; tokenizer and config come from the base checkpoint as above.
pub fn resolve_source(name: &str, model_dir: &Path) -> ModelSource {
    let (name, quant) = split_quant(registry::bare_id(name));
    if is_gguf_path(name) {
        return ModelSource::Gguf {
            weights: GgufWeights::Local(PathBuf::from(name)),
            base: None,
            quant: None,
        };
    }
    let base = resolve_base(name, model_dir);
    let Some(quant) = quant else {
        return base;
    };
    let quant = quant.to_ascii_lowercase();
    let spec = registry::resolve(name);
    // A GGUF already in the base directory (or the model directory) wins.
    let local_dir = match &base {
        ModelSource::Local(dir) => Some(dir.clone()),
        _ => spec.map(|s| model_dir.join(s.id)),
    };
    let published = spec
        .and_then(|s| s.gguf.as_ref())
        .and_then(|g| g.file(&quant).map(|f| (g.repo, f)));
    if let Some(dir) = &local_dir {
        let candidate = published
            .map(|(_, f)| dir.join(f))
            .filter(|p| p.is_file())
            .or_else(|| find_gguf(dir, &quant));
        if let Some(path) = candidate {
            return ModelSource::Gguf {
                weights: GgufWeights::Local(path),
                base: Some(Box::new(base)),
                quant: Some(quant),
            };
        }
    }
    match published {
        Some((repo, filename)) => ModelSource::Gguf {
            weights: GgufWeights::Hub {
                repo: repo.to_string(),
                filename: filename.to_string(),
            },
            base: Some(Box::new(base)),
            quant: Some(quant),
        },
        // No published file: keep the request so loading reports it clearly.
        None => ModelSource::Gguf {
            weights: GgufWeights::Missing {
                model: name.to_string(),
            },
            base: Some(Box::new(base)),
            quant: Some(quant),
        },
    }
}

fn resolve_base(bare: &str, model_dir: &Path) -> ModelSource {
    if Path::new(bare).is_dir() {
        return ModelSource::Local(PathBuf::from(bare));
    }
    let spec = registry::resolve(bare);
    let local_id = spec.map(|s| s.id).unwrap_or(bare);
    let candidate = model_dir.join(local_id);
    if candidate.join("config.json").is_file() {
        return ModelSource::Local(candidate);
    }
    if let Some(repo_dir_name) = bare.rsplit('/').next().filter(|_| bare.contains('/')) {
        let candidate = model_dir.join(repo_dir_name.split('@').next().unwrap_or(repo_dir_name));
        if candidate.join("config.json").is_file() {
            return ModelSource::Local(candidate);
        }
    }
    match spec {
        Some(spec) => ModelSource::Hub {
            repo: spec.repo.to_string(),
            revision: None,
        },
        None => ModelSource::parse(bare),
    }
}

fn is_gguf_path(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".gguf")
}

/// A `.gguf` in `dir` whose name contains `quant` (`harrier-0.6b-q8_0.gguf`).
fn find_gguf(dir: &Path, quant: &str) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.to_ascii_lowercase().contains(quant))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// Fetch a model (from the Hub or another directory) and copy its files into
/// `<model_dir>/<id>/` as plain files — no symlinks, no cache layout — so the
/// directory can be mounted into a container or committed to an artifact
/// store. Returns the directory. Existing files of the same size are kept.
///
/// `0.6b:q8_0` exports that GGUF plus the tokenizer and config (not the
/// safetensors); a bare name exports the safetensors checkpoint.
pub fn export_to_dir(name: &str, model_dir: &Path) -> Result<PathBuf> {
    let (bare, quant) = split_quant(registry::bare_id(name));
    let spec = registry::resolve(bare);
    let base = match spec {
        Some(spec) => ModelSource::Hub {
            repo: spec.repo.to_string(),
            revision: None,
        },
        None if is_gguf_path(bare) => ModelSource::Gguf {
            weights: GgufWeights::Local(PathBuf::from(bare)),
            base: None,
            quant: None,
        },
        None => ModelSource::parse(bare),
    };
    let source = match quant {
        None => base,
        Some(quant) => {
            let quant = quant.to_ascii_lowercase();
            let spec = spec.ok_or_else(|| {
                HarrierError::InvalidRequest(format!(
                    "`{bare}:{quant}`: quantisations are only published for registry models; \
                     pass a .gguf path instead"
                ))
            })?;
            let gguf = spec.gguf.as_ref().ok_or_else(|| {
                HarrierError::InvalidRequest(format!(
                    "no GGUF quantisation is published for {}; pass a .gguf path instead",
                    spec.id
                ))
            })?;
            let filename = gguf.file(&quant).ok_or_else(|| {
                HarrierError::InvalidRequest(format!(
                    "unknown quantisation `{quant}` for {} (available: {})",
                    spec.id,
                    gguf.tags().join(", ")
                ))
            })?;
            ModelSource::Gguf {
                weights: GgufWeights::Hub {
                    repo: gguf.repo.to_string(),
                    filename: filename.to_string(),
                },
                base: Some(Box::new(base)),
                quant: Some(quant),
            }
        }
    };
    let id = spec
        .map(|s| s.id.to_string())
        .unwrap_or_else(|| source_id(&source));
    let files = ModelFiles::locate(&source)?;
    let target = model_dir.join(&id);
    std::fs::create_dir_all(&target)?;

    let mut copied = Vec::new();
    copied.extend(files.config.iter().cloned());
    copied.push(files.tokenizer.clone());
    copied.extend(files.weights.paths().iter().cloned());
    copied.extend(files.sentence_transformers_config.iter().cloned());
    copied.extend(files.extras.iter().cloned());
    for src in copied {
        let name = src.file_name().ok_or_else(|| {
            HarrierError::Io(std::io::Error::other("weight path has no file name"))
        })?;
        let dst = target.join(name);
        let same = dst
            .metadata()
            .ok()
            .zip(src.metadata().ok())
            .is_some_and(|(d, s)| d.len() == s.len());
        if !same {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(target)
}

/// Bare id for an unregistered source: last path segment of the repo, the
/// directory name, or the GGUF file stem.
pub fn source_id(source: &ModelSource) -> String {
    match source {
        ModelSource::Hub { repo, .. } => repo.rsplit('/').next().unwrap_or(repo).to_string(),
        ModelSource::Local(path) => path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string()),
        ModelSource::Gguf { weights, base, .. } => match (base, weights) {
            (Some(base), _) => source_id(base),
            (None, GgufWeights::Local(path)) => path
                .file_stem()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string()),
            (None, GgufWeights::Hub { filename, .. }) => filename
                .strip_suffix(".gguf")
                .unwrap_or(filename)
                .to_string(),
            (None, GgufWeights::Missing { model }) => model.clone(),
        },
    }
}

/// Where a GGUF file comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufWeights {
    Local(PathBuf),
    Hub {
        repo: String,
        filename: String,
    },
    /// The requested quantisation is not published for this model.
    Missing {
        model: String,
    },
}

/// Where a checkpoint comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// A Hugging Face repo such as `microsoft/harrier-oss-v1-0.6b`.
    Hub {
        repo: String,
        revision: Option<String>,
    },
    /// A directory holding `config.json`, `tokenizer.json` and safetensors.
    Local(PathBuf),
    /// A GGUF file. `base` is where `tokenizer.json` / `config.json` come
    /// from (the original checkpoint); `None` means beside the file.
    Gguf {
        weights: GgufWeights,
        base: Option<Box<ModelSource>>,
        /// The requested quantisation tag (`q8_0`), when known.
        quant: Option<String>,
    },
}

impl ModelSource {
    /// A path that exists on disk is local (a `.gguf` file or a directory);
    /// anything else is a Hub repo id (`owner/name` or `owner/name@revision`).
    pub fn parse(value: &str) -> Self {
        let value = value.trim();
        let path = Path::new(value);
        if path.is_dir() {
            return Self::Local(path.to_path_buf());
        }
        if is_gguf_path(value) {
            return Self::Gguf {
                weights: GgufWeights::Local(path.to_path_buf()),
                base: None,
                quant: None,
            };
        }
        match value.rsplit_once('@') {
            Some((repo, rev)) if !repo.contains('@') && !rev.is_empty() => Self::Hub {
                repo: repo.to_string(),
                revision: Some(rev.to_string()),
            },
            _ => Self::Hub {
                repo: value.to_string(),
                revision: None,
            },
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Hub {
                repo,
                revision: None,
            } => repo.clone(),
            Self::Hub {
                repo,
                revision: Some(rev),
            } => format!("{repo}@{rev}"),
            Self::Local(p) => p.display().to_string(),
            Self::Gguf { weights, .. } => match weights {
                GgufWeights::Local(p) => p.display().to_string(),
                GgufWeights::Hub { repo, filename } => format!("{repo}/{filename}"),
                GgufWeights::Missing { model } => format!("{model} (no GGUF)"),
            },
        }
    }

    pub fn is_gguf(&self) -> bool {
        matches!(self, Self::Gguf { .. })
    }
}

/// The weight files of one model.
#[derive(Debug, Clone)]
pub enum WeightFiles {
    /// Safetensors shards in index order.
    Safetensors(Vec<PathBuf>),
    /// One GGUF file.
    Gguf(PathBuf),
}

impl WeightFiles {
    pub fn paths(&self) -> Vec<PathBuf> {
        match self {
            Self::Safetensors(v) => v.clone(),
            Self::Gguf(p) => vec![p.clone()],
        }
    }
}

/// The files needed to load one model.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// `config.json`; absent only for a GGUF without one (its metadata is
    /// used instead).
    pub config: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub weights: WeightFiles,
    /// Optional `config_sentence_transformers.json` (prompt names).
    pub sentence_transformers_config: Option<PathBuf>,
    /// Other optional files worth carrying along when exporting
    /// (`tokenizer_config.json`, `special_tokens_map.json`, the safetensors
    /// index).
    pub extras: Vec<PathBuf>,
}

/// Optional companions copied by [`export_to_dir`].
const EXTRA_FILES: [&str; 4] = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
    "README.md",
];

impl ModelFiles {
    /// Locate (and for Hub sources, download) the files.
    pub fn locate(source: &ModelSource) -> Result<Self> {
        match source {
            ModelSource::Local(dir) => Self::from_dir(dir),
            ModelSource::Hub { repo, revision } => Self::from_hub(repo, revision.as_deref()),
            ModelSource::Gguf {
                weights,
                base,
                quant,
            } => Self::from_gguf(weights, base.as_deref(), quant.as_deref()),
        }
    }

    fn from_dir(dir: &Path) -> Result<Self> {
        let config = existing(dir.join("config.json"))?;
        let tokenizer = existing(dir.join("tokenizer.json"))?;
        let index = dir.join("model.safetensors.index.json");
        let weights = if index.is_file() {
            shard_names(&std::fs::read_to_string(&index)?)?
                .into_iter()
                .map(|name| existing(dir.join(name)))
                .collect::<Result<Vec<_>>>()?
        } else if dir.join("model.safetensors").is_file() {
            vec![dir.join("model.safetensors")]
        } else {
            let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            shards.sort();
            if shards.is_empty() {
                return Err(HarrierError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "no .safetensors files in {} (for a GGUF, name the file or use `<model>:<quant>`)",
                        dir.display()
                    ),
                )));
            }
            shards
        };
        let st = dir.join("config_sentence_transformers.json");
        Ok(Self {
            config: Some(config),
            tokenizer,
            weights: WeightFiles::Safetensors(weights),
            sentence_transformers_config: st.is_file().then_some(st),
            extras: EXTRA_FILES
                .iter()
                .map(|f| dir.join(f))
                .filter(|p| p.is_file())
                .collect(),
        })
    }

    /// Tokenizer and config (optional) from `base`, weights from the GGUF.
    fn from_gguf(
        weights: &GgufWeights,
        base: Option<&ModelSource>,
        quant: Option<&str>,
    ) -> Result<Self> {
        let gguf = match weights {
            GgufWeights::Local(path) => existing(path.clone())?,
            GgufWeights::Hub { repo, filename } => hub_file(repo, None, filename)?,
            GgufWeights::Missing { model } => {
                let known = registry::resolve(model)
                    .and_then(|s| s.gguf.as_ref())
                    .map(|g| g.tags().join(", "))
                    .unwrap_or_else(|| "none".into());
                return Err(HarrierError::InvalidRequest(format!(
                    "no GGUF quantisation `{}` is published for {model} (available: {known}); \
                     put a .gguf in the model directory or pass its path",
                    quant.unwrap_or("?")
                )));
            }
        };
        // Companions: the base checkpoint, else the GGUF's own directory.
        let companions = match base {
            Some(ModelSource::Local(dir)) => Companions::Dir(dir.clone()),
            Some(ModelSource::Hub { repo, revision }) => Companions::Hub {
                repo: repo.clone(),
                revision: revision.clone(),
            },
            Some(ModelSource::Gguf { .. }) | None => {
                Companions::Dir(gguf.parent().map(Path::to_path_buf).unwrap_or_default())
            }
        };
        let tokenizer = companions.get("tokenizer.json").ok_or_else(|| {
            HarrierError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "no tokenizer.json for {}: put the model's tokenizer.json beside the GGUF \
                     (or use a registry name such as `0.6b:q8_0`, which fetches it)",
                    gguf.display()
                ),
            ))
        })?;
        Ok(Self {
            config: companions.get("config.json"),
            tokenizer,
            weights: WeightFiles::Gguf(gguf),
            sentence_transformers_config: companions.get("config_sentence_transformers.json"),
            extras: EXTRA_FILES
                .iter()
                .filter(|f| **f != "model.safetensors.index.json")
                .filter_map(|f| companions.get(f))
                .collect(),
        })
    }

    fn from_hub(repo: &str, revision: Option<&str>) -> Result<Self> {
        let config = hub_file(repo, revision, "config.json")?;
        let tokenizer = hub_file(repo, revision, "tokenizer.json")?;
        let weights = match hub_file(repo, revision, "model.safetensors.index.json") {
            Ok(index) => shard_names(&std::fs::read_to_string(&index)?)?
                .into_iter()
                .map(|name| hub_file(repo, revision, &name))
                .collect::<Result<Vec<_>>>()?,
            Err(_) => vec![hub_file(repo, revision, "model.safetensors")?],
        };
        let sentence_transformers_config =
            hub_file(repo, revision, "config_sentence_transformers.json").ok();
        let extras = EXTRA_FILES
            .iter()
            .filter_map(|f| hub_file(repo, revision, f).ok())
            .collect();
        Ok(Self {
            config: Some(config),
            tokenizer,
            weights: WeightFiles::Safetensors(weights),
            sentence_transformers_config,
            extras,
        })
    }
}

/// Where a GGUF's tokenizer / config live.
enum Companions {
    Dir(PathBuf),
    Hub {
        repo: String,
        revision: Option<String>,
    },
}

impl Companions {
    fn get(&self, name: &str) -> Option<PathBuf> {
        match self {
            Self::Dir(dir) => Some(dir.join(name)).filter(|p| p.is_file()),
            Self::Hub { repo, revision } => hub_file(repo, revision.as_deref(), name).ok(),
        }
    }
}

/// Download (or find in the cache) one file of a Hub repo.
#[cfg(feature = "hf-hub")]
fn hub_file(repo: &str, revision: Option<&str>, filename: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let hub = |e: hf_hub::api::sync::ApiError| HarrierError::Hub(format!("{repo}/{filename}: {e}"));
    let api = ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(hub)?;
    let repo_handle = api.repo(match revision {
        Some(rev) => Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::new(repo.to_string(), RepoType::Model),
    });
    repo_handle.get(filename).map_err(hub)
}

#[cfg(not(feature = "hf-hub"))]
fn hub_file(repo: &str, _revision: Option<&str>, filename: &str) -> Result<PathBuf> {
    Err(HarrierError::Hub(format!(
        "`{repo}/{filename}` is not available locally and this build has no `hf-hub` feature; \
         download the files and pass their directory"
    )))
}

fn existing(path: PathBuf) -> Result<PathBuf> {
    if path.is_file() {
        Ok(path)
    } else {
        Err(HarrierError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("missing model file {}", path.display()),
        )))
    }
}

/// Unique shard file names from a `model.safetensors.index.json`, in order.
fn shard_names(index_json: &str) -> Result<Vec<String>> {
    #[derive(serde::Deserialize)]
    struct Index {
        weight_map: std::collections::BTreeMap<String, String>,
    }
    let index: Index = serde_json::from_str(index_json)?;
    let names: BTreeSet<String> = index.weight_map.into_values().collect();
    if names.is_empty() {
        return Err(HarrierError::Config(
            "safetensors index has no weights".into(),
        ));
    }
    Ok(names.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sources() {
        assert_eq!(
            ModelSource::parse("microsoft/harrier-oss-v1-0.6b"),
            ModelSource::Hub {
                repo: "microsoft/harrier-oss-v1-0.6b".into(),
                revision: None
            }
        );
        assert_eq!(
            ModelSource::parse("microsoft/harrier-oss-v1-0.6b@abc123"),
            ModelSource::Hub {
                repo: "microsoft/harrier-oss-v1-0.6b".into(),
                revision: Some("abc123".into())
            }
        );
        let here = std::env::current_dir().unwrap();
        assert_eq!(
            ModelSource::parse(here.to_str().unwrap()),
            ModelSource::Local(here)
        );
        assert!(ModelSource::parse("D:/models/x.gguf").is_gguf());
    }

    #[test]
    fn prefers_the_model_dir() {
        let tmp = std::env::temp_dir().join(format!("harrier-modeldir-{}", std::process::id()));
        let local = tmp.join("harrier-oss-v1-0.6b");
        std::fs::create_dir_all(&local).unwrap();
        // Without config.json the directory is ignored → Hub.
        assert!(matches!(
            resolve_source("0.6b", &tmp),
            ModelSource::Hub { .. }
        ));
        std::fs::write(local.join("config.json"), "{}").unwrap();
        for name in [
            "0.6b",
            "harrier::harrier-oss-v1-0.6b",
            "microsoft/harrier-oss-v1-0.6b",
        ] {
            assert_eq!(
                resolve_source(name, &tmp),
                ModelSource::Local(local.clone()),
                "{name}"
            );
        }
        // Unknown repo ids fall through to the Hub; a repo whose last segment
        // matches a directory is found too.
        assert!(matches!(
            resolve_source("someone/other-model", &tmp),
            ModelSource::Hub { .. }
        ));
        let other = tmp.join("other-model");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("config.json"), "{}").unwrap();
        assert_eq!(
            resolve_source("someone/other-model", &tmp),
            ModelSource::Local(other)
        );
        // An explicit directory path wins outright.
        assert_eq!(
            resolve_source(local.to_str().unwrap(), &tmp),
            ModelSource::Local(local.clone())
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn resolves_gguf_quantisations() {
        let tmp = std::env::temp_dir().join(format!("harrier-gguf-{}", std::process::id()));
        let local = tmp.join("harrier-oss-v1-0.6b");
        std::fs::create_dir_all(&local).unwrap();

        // Nothing local: the published file on the Hub, base from the Hub.
        match resolve_source("0.6b:q8_0", &tmp) {
            ModelSource::Gguf {
                weights: GgufWeights::Hub { repo, filename },
                base: Some(base),
                quant: Some(q),
            } => {
                assert_eq!(repo, "cstr/harrier-0.6b-GGUF");
                assert_eq!(filename, "harrier-0.6b-q8_0.gguf");
                assert_eq!(q, "q8_0");
                assert!(matches!(*base, ModelSource::Hub { .. }));
            }
            other => panic!("{other:?}"),
        }
        // Unknown tag → Missing, reported at load time.
        assert!(matches!(
            resolve_source("0.6b:q2_k", &tmp),
            ModelSource::Gguf {
                weights: GgufWeights::Missing { .. },
                ..
            }
        ));
        // A file in the model directory wins, and the base becomes local
        // once config.json is there.
        let gguf = local.join("harrier-0.6b-q8_0.gguf");
        std::fs::write(&gguf, "x").unwrap();
        std::fs::write(local.join("config.json"), "{}").unwrap();
        match resolve_source("harrier::0.6b:Q8_0", &tmp) {
            ModelSource::Gguf {
                weights: GgufWeights::Local(p),
                base: Some(base),
                ..
            } => {
                assert_eq!(p, gguf);
                assert_eq!(*base, ModelSource::Local(local.clone()));
            }
            other => panic!("{other:?}"),
        }
        // Any .gguf containing the tag is accepted for a local directory.
        let custom = local.join("my-q4_k-build.gguf");
        std::fs::write(&custom, "x").unwrap();
        match resolve_source(&format!("{}:q4_k", local.display()), &tmp) {
            ModelSource::Gguf {
                weights: GgufWeights::Local(p),
                ..
            } => assert_eq!(p, custom),
            other => panic!("{other:?}"),
        }
        // An explicit .gguf path stands alone.
        match resolve_source(gguf.to_str().unwrap(), &tmp) {
            ModelSource::Gguf {
                weights: GgufWeights::Local(p),
                base: None,
                quant: None,
            } => assert_eq!(p, gguf),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            source_id(&ModelSource::parse(gguf.to_str().unwrap())),
            "harrier-0.6b-q8_0"
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn reads_shard_index() {
        let json = r#"{"metadata":{"total_size":1},"weight_map":{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors","c":"model-00001-of-00002.safetensors"}}"#;
        assert_eq!(
            shard_names(json).unwrap(),
            vec![
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
    }
}
