//! End-to-end checks against the `transformers` reference implementation for
//! the two primary models (270M, 0.6B), plus a sanity check for the 27B.
//!
//! They need weights (from `./models` if `harrier_api download` put them
//! there, else the Hub), so they are `#[ignore]`d by default:
//!
//! ```text
//! cargo test --release -p harrier_core --test reference -- --ignored
//! HARRIER_TEST_DEVICE=cuda cargo test --release --features cuda -p harrier_core --test reference -- --ignored
//! cargo test --release -p harrier_core --test reference -- --ignored --include-ignored harrier_27b   # 54 GB
//! ```
//!
//! The `q8_0` tests compare the GGUF quantisation against the dense
//! checkpoint loaded in the same run (and the pinned scores, loosely).
//!
//! Reference scores were produced with `transformers` 5.15 / torch 2.x in
//! float32, eager attention, on the model-card example (see the README).

use harrier_core::{DeviceRequest, Embedder, EmbedderOptions, EmbeddingTask, EmbeddingsRequest};

const QUERIES: [&str; 2] = ["how much protein should a female eat", "summit define"];
const DOCUMENTS: [&str; 2] = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
];

fn options() -> EmbedderOptions {
    let device: DeviceRequest = std::env::var("HARRIER_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap();
    EmbedderOptions {
        device,
        dtype: std::env::var("HARRIER_TEST_DTYPE")
            .ok()
            .map(|d| harrier_core::parse_dtype(&d).unwrap()),
        ..Default::default()
    }
}

/// bf16 on an accelerator and int8 weights drift a little from the f32
/// reference.
fn tolerance(embedder: &Embedder) -> f32 {
    let info = embedder.info();
    if info.dtype == "f32" && info.quantization.is_none() {
        0.05
    } else {
        1.0
    }
}

/// Per-row cosine and max |Δ| between two sets of vectors.
fn compare(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<(f32, f32)> {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let dot: f32 = x.iter().zip(y).map(|(p, q)| p * q).sum();
            let nx: f32 = x.iter().map(|p| p * p).sum::<f32>().sqrt();
            let ny: f32 = y.iter().map(|p| p * p).sum::<f32>().sqrt();
            let max_diff = x
                .iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            (dot / (nx * ny), max_diff)
        })
        .collect()
}

/// The GGUF quantisation of `id` against the dense checkpoint: the pinned
/// scores within the loose tolerance, and every vector of a mixed-length
/// padded batch (including a ~3k-token document) within cosine 0.999 of the
/// dense one (0.99 for the empty string) — plus the batch/single agreement
/// check on the quantised path.
fn check_quantized(id: &str, quant: &str, documents: &[&str], expected: &[[f32; 2]; 2]) {
    let dense = Embedder::load(id, options()).unwrap();
    let quantized = Embedder::load(&format!("{id}:{quant}"), options()).unwrap();
    assert_eq!(quantized.info().quantization.as_deref(), Some(quant));
    assert_eq!(quantized.info().id, dense.info().id);

    let scores = score_matrix(&quantized, documents);
    assert_scores(&scores, expected, 1.0);

    let long = std::iter::repeat_n(DOCUMENTS[0], 40)
        .collect::<Vec<_>>()
        .join(" ");
    let texts: Vec<String> = [documents[0], "short", QUERIES[1], "", long.as_str()]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let a = dense.embed(&texts, true).unwrap();
    let b = quantized.embed(&texts, true).unwrap();
    assert!(a.prompt_tokens > 1000, "long document did not tokenise");
    for (i, (cos, max_diff)) in compare(&a.vectors, &b.vectors).iter().enumerate() {
        eprintln!("{id}:{quant} row {i}: cosine {cos:.6}, max |Δ| {max_diff:.2e}");
        // An empty input is a single EOS token whose hidden state is mostly
        // quantisation noise; real text stays much closer.
        let floor = if texts[i].is_empty() { 0.99 } else { 0.999 };
        assert!(
            *cos > floor,
            "row {i}: cosine {cos} vs the dense checkpoint"
        );
    }

    // Batched vs single: exact on the CPU kernel. candle's CUDA / Metal
    // quantised matmuls pick a different kernel for <= 8 rows (MMVQ) than
    // for larger batches (MMQ) and the two quantise activations slightly
    // differently, so there the same row differs at the int8-noise level
    // depending on its batch.
    let single = quantized.embed(&[texts[1].clone()], true).unwrap();
    let (cos, max_diff) = compare(&[b.vectors[1].clone()], &single.vectors)[0];
    let (max_cos, max_abs) = if quantized.device().is_cpu() {
        (0.99999, 1e-3)
    } else {
        (0.9999, 5e-3)
    };
    assert!(
        max_diff < max_abs && cos > max_cos,
        "batched vs single differ: cosine {cos}, max |Δ| {max_diff}"
    );
}

fn score_matrix(embedder: &Embedder, documents: &[&str]) -> Vec<Vec<f32>> {
    let q = embedder
        .embed_request(&EmbeddingsRequest {
            input: QUERIES
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .into(),
            task: Some(EmbeddingTask::Query),
            ..Default::default()
        })
        .unwrap();
    let d = embedder
        .embed_request(&EmbeddingsRequest {
            input: documents
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .into(),
            ..Default::default()
        })
        .unwrap();
    q.output
        .iter()
        .map(|qv| {
            d.output
                .iter()
                .map(|dv| qv.iter().zip(dv).map(|(a, b)| a * b).sum::<f32>() * 100.0)
                .collect()
        })
        .collect()
}

fn assert_scores(got: &[Vec<f32>], expected: &[[f32; 2]; 2], tol: f32) {
    for (i, row) in expected.iter().enumerate() {
        for (j, want) in row.iter().enumerate() {
            assert!(
                (got[i][j] - want).abs() < tol,
                "score[{i}][{j}] = {} but transformers gives {want} (tolerance {tol})\nall: {got:?}",
                got[i][j]
            );
        }
    }
}

#[test]
#[ignore = "downloads microsoft/harrier-oss-v1-0.6b (1.2 GB)"]
fn qwen3_0_6b_matches_transformers() {
    let embedder = Embedder::load("harrier-oss-v1-0.6b", options()).unwrap();
    assert_eq!(embedder.info().embedding_dimensions, 1024);
    let scores = score_matrix(&embedder, &DOCUMENTS);
    assert_scores(
        &scores,
        &[[66.273, 28.462], [29.774, 69.937]],
        tolerance(&embedder),
    );

    // Batched and single-input runs must agree (padding mask correctness).
    let batched = embedder
        .embed(&[DOCUMENTS[0].to_string(), "short".to_string()], true)
        .unwrap();
    let single = embedder.embed(&["short".to_string()], true).unwrap();
    let diff = batched.vectors[1]
        .iter()
        .zip(&single.vectors[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(diff < 1e-4, "batched vs single differ by {diff}");
}

#[test]
#[ignore = "downloads microsoft/harrier-oss-v1-270m (0.5 GB)"]
fn gemma3_270m_matches_transformers() {
    let embedder = Embedder::load("harrier-oss-v1-270m", options()).unwrap();
    assert_eq!(embedder.info().embedding_dimensions, 640);
    // The transformers reference for this model used the first sentence of
    // each document only.
    let short_docs = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain.",
    ];
    let got = score_matrix(&embedder, &short_docs);
    assert_scores(
        &got,
        &[[65.769, 24.787], [30.471, 66.154]],
        tolerance(&embedder),
    );
}

#[test]
#[ignore = "downloads cstr/harrier-0.6b-GGUF q8_0 (0.64 GB) and the dense checkpoint"]
fn qwen3_0_6b_q8_0_tracks_dense() {
    check_quantized(
        "harrier-oss-v1-0.6b",
        "q8_0",
        &DOCUMENTS,
        &[[66.273, 28.462], [29.774, 69.937]],
    );
}

#[test]
#[ignore = "downloads cstr/harrier-270m-GGUF q8_0 (0.3 GB) and the dense checkpoint"]
fn gemma3_270m_q8_0_tracks_dense() {
    let short_docs = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain.",
    ];
    check_quantized(
        "harrier-oss-v1-270m",
        "q8_0",
        &short_docs,
        &[[65.769, 24.787], [30.471, 66.154]],
    );
}

/// The 27B shares the Gemma 3 code path with the 270M (plus sliding-window
/// layers, which are covered by a synthetic check in development). There is
/// no pinned reference here — running the 54 GB checkpoint in `transformers`
/// needs an 80 GB GPU — so this only checks shape, normalization, and that
/// matching query/document pairs outscore the mismatched ones.
#[test]
#[ignore = "needs microsoft/harrier-oss-v1-27b (54 GB) and ~60 GB of device memory"]
fn harrier_27b_sanity() {
    let embedder = Embedder::load("harrier-oss-v1-27b", options()).unwrap();
    assert_eq!(embedder.info().embedding_dimensions, 5376);
    let scores = score_matrix(&embedder, &DOCUMENTS);
    assert!(scores[0][0] > scores[0][1] + 10.0, "{scores:?}");
    assert!(scores[1][1] > scores[1][0] + 10.0, "{scores:?}");
    let v = embedder.embed(&["unit".to_string()], true).unwrap();
    let norm: f32 = v.vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3);
}
