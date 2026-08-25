//! Embed texts from the command line and print the vectors (or pairwise
//! cosine scores).
//!
//! ```text
//! cargo run --release -p harrier_core --example embed -- \
//!     --model 0.6b --task query "how much protein should a female eat" "summit define"
//! cargo run --release --features cuda -p harrier_core --example embed -- --device cuda --json "text"
//! ```
//!
//! Environment: `HARRIER_DEVICE` (auto|cpu|cuda[:n]|metal[:n]), `HARRIER_DTYPE`.

use harrier_core::{DeviceRequest, Embedder, EmbedderOptions, EmbeddingTask, EmbeddingsRequest};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = std::env::var("HARRIER_MODEL").unwrap_or_else(|_| "harrier-oss-v1-0.6b".into());
    let mut device: DeviceRequest = std::env::var("HARRIER_DEVICE")
        .unwrap_or_default()
        .parse()?;
    let mut dtype = std::env::var("HARRIER_DTYPE").ok();
    let mut dequantize = std::env::var("HARRIER_DEQUANTIZE").is_ok_and(|v| v == "1" || v == "true");
    let mut task: Option<String> = None;
    let mut instruction: Option<String> = None;
    let mut json = false;
    let mut normalize = true;
    let mut texts = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => model = args.next().ok_or("--model needs a value")?,
            "--device" | "-d" => device = args.next().ok_or("--device needs a value")?.parse()?,
            "--dtype" => dtype = args.next(),
            "--dequantize" => dequantize = true,
            "--task" | "-t" => task = args.next(),
            "--instruction" | "-i" => instruction = args.next(),
            "--json" => json = true,
            "--raw" => normalize = false,
            "--help" | "-h" => {
                println!(
                    "usage: embed [--model ID] [--device auto|cpu|cuda|metal] [--dtype f32|bf16|f16] \
                     [--dequantize] [--task TASK] [--instruction TEXT] [--json] [--raw] TEXT..."
                );
                return Ok(());
            }
            _ => texts.push(arg),
        }
    }
    if texts.is_empty() {
        return Err("no texts given (try --help)".into());
    }

    let options = EmbedderOptions {
        device,
        dtype: dtype
            .as_deref()
            .map(harrier_core::parse_dtype)
            .transpose()?,
        dequantize,
        ..Default::default()
    };
    eprintln!("loading {model} on {device}…");
    let embedder = Embedder::load(&model, options)?;
    let info = embedder.info();
    eprintln!(
        "loaded {} ({:?}, {}-d, {} / {}) in {:.1}s",
        info.qualified_id,
        info.architecture,
        info.embedding_dimensions,
        info.device,
        info.precision(),
        info.load_seconds
    );

    let request = EmbeddingsRequest {
        model: model.clone(),
        input: texts.clone().into(),
        task: task.as_deref().map(EmbeddingTask::parse),
        instruction,
        normalize: Some(normalize),
        ..Default::default()
    };
    let started = Instant::now();
    let response = embedder.embed_request(&request)?;
    let elapsed = started.elapsed();
    let tokens = response
        .usage
        .as_ref()
        .and_then(|u| u.input.as_ref())
        .and_then(|i| i.get("prompt_tokens").copied())
        .unwrap_or(0);
    eprintln!(
        "{} vectors, {tokens} tokens in {:.0} ms ({:.0} tok/s)",
        response.output.len(),
        elapsed.as_secs_f64() * 1000.0,
        tokens as f64 / elapsed.as_secs_f64().max(1e-9)
    );

    if json {
        println!("{}", serde_json::to_string(&response)?);
        return Ok(());
    }
    for (text, vector) in texts.iter().zip(&response.output) {
        let head: Vec<String> = vector.iter().take(6).map(|v| format!("{v:+.4}")).collect();
        println!("[{}] {:?}", head.join(" "), truncate(text, 60));
    }
    if texts.len() > 1 {
        println!("\ncosine similarity:");
        for i in 0..texts.len() {
            let row: Vec<String> = (0..texts.len())
                .map(|j| format!("{:.4}", cosine(&response.output[i], &response.output[j])))
                .collect();
            println!("  {}", row.join("  "));
        }
    }
    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}
