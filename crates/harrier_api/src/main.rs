use harrier_api::{AppState, ServerConfig, create_app};
use harrier_core::Embedder;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "\
usage:
  harrier_api                      serve (configured by environment, see below)
  harrier_api download MODEL...    fetch models into the model directory and exit
                                   (MODEL = 270m, 0.6b, 27b, a Hub repo[@rev], a directory,
                                   or a GGUF quantisation such as 0.6b:q8_0; comma-separated
                                   lists are accepted too)
  harrier_api --help";

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "harrier_api {}\n\n{USAGE}\n\nConfiguration (environment):\n{}",
            env!("CARGO_PKG_VERSION"),
            ServerConfig::ENV_HELP
        );
        return;
    }
    let result = match args.first().map(String::as_str) {
        Some("download") => download(&args[1..]),
        Some(other) => Err(format!("unknown argument `{other}`\n{USAGE}").into()),
        None => run().await,
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `harrier_api download 270m 0.6b:q8_0` → `./models/harrier-oss-v1-270m/…`
/// Names may also be comma-separated (the `HARRIER_MODELS` spelling), so one
/// value can drive both a download and the server.
fn download(models: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let models: Vec<&str> = models
        .iter()
        .flat_map(|m| m.split(','))
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .collect();
    if models.is_empty() {
        return Err("download needs at least one model name".into());
    }
    let dir = harrier_core::model_dir();
    for name in models {
        eprintln!("fetching {name} into {}…", dir.display());
        let target = harrier_core::export_to_dir(name, &dir)?;
        let bytes: u64 = std::fs::read_dir(&target)?
            .filter_map(|e| e.ok()?.metadata().ok())
            .map(|m| m.len())
            .sum();
        eprintln!("  {} ({:.2} GB)", target.display(), bytes as f64 / 1e9);
    }
    Ok(())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = ServerConfig::from_env()?;
    eprintln!(
        "harrier_api {} · backends: {} · device: {} · models: {} · model dir: {}",
        env!("CARGO_PKG_VERSION"),
        harrier_core::compiled_backends().join(", "),
        config.options.device,
        config.models.join(", "),
        config.model_dir.display()
    );

    let mut embedders = Vec::with_capacity(config.models.len());
    for name in &config.models {
        let started = Instant::now();
        eprintln!("loading {name}…");
        let options = config.options.clone();
        let name_owned = name.clone();
        let model_dir = config.model_dir.clone();
        let embedder = tokio::task::spawn_blocking(move || {
            Embedder::load_from(&name_owned, &model_dir, options)
        })
        .await??;
        let info = embedder.info();
        eprintln!(
            "loaded {} ({:?}, {}-d, max {} tokens, {} / {}) in {:.1}s",
            info.qualified_id,
            info.architecture,
            info.embedding_dimensions,
            info.max_input_tokens,
            info.device,
            info.precision(),
            started.elapsed().as_secs_f64()
        );
        if config.warmup {
            let started = Instant::now();
            let embedder = Arc::new(embedder);
            let warm = embedder.clone();
            tokio::task::spawn_blocking(move || warm.warmup()).await??;
            eprintln!(
                "warmed up in {:.0} ms",
                started.elapsed().as_secs_f64() * 1000.0
            );
            embedders.push(embedder);
        } else {
            embedders.push(Arc::new(embedder));
        }
    }

    // Two weight variants of one model would share an id and silently
    // replace each other; refuse that configuration up front.
    let mut seen = std::collections::BTreeSet::new();
    for e in &embedders {
        if !seen.insert(e.info().id.clone()) {
            return Err(format!(
                "model `{}` is listed more than once in HARRIER_MODELS (one weight variant per model)",
                e.info().id
            )
            .into());
        }
    }
    let mut state = AppState::new(embedders);
    state.log_requests = config.log_requests;
    let app = create_app(Arc::new(state));

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    eprintln!("shut down");
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
