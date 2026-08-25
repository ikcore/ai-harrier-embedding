# harrier_embeddings

A small Rust service for Microsoft's [Harrier OSS v1](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) text-embedding models, built on [candle](https://github.com/huggingface/candle) and [axum](https://github.com/tokio-rs/axum). Runs on **CPU, CUDA and Metal**, from the original **safetensors** (f32 / bf16) or **8-bit GGUF** weights, embeds **one or many inputs per request**, and speaks the same embedding contract as [GAISe](https://github.com/ikcore/gaise) (`task` vocabulary, request/response shapes, `GET /v1/models`).

| Model | Backbone | Dimensions | Max tokens | Weights bf16 / q8_0 | MTEB v2 | |
|---|---|---|---|---|---|---|
| [`harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | Qwen3 | 1 024 | 32 768 | 1.2 GB / 0.64 GB | 69.0 | **primary** — validated against `transformers`; GGUF from [`cstr/harrier-0.6b-GGUF`](https://huggingface.co/cstr/harrier-0.6b-GGUF) |
| [`harrier-oss-v1-270m`](https://huggingface.co/microsoft/harrier-oss-v1-270m) | Gemma 3 | 640 | 32 768 | 0.54 GB / 0.30 GB | 66.5 | validated against `transformers`; GGUF from [`cstr/harrier-270m-GGUF`](https://huggingface.co/cstr/harrier-270m-GGUF) |
| [`harrier-oss-v1-27b`](https://huggingface.co/microsoft/harrier-oss-v1-27b) | Gemma 3 | 5 376 | 32 768 | 54 GB / — | 74.3 | loads through the same Gemma 3 path; needs an 80 GB GPU, not run here |

Two crates:

- **`crates/harrier_core`** — the models (Qwen3 and Gemma 3 decoders written directly on `candle-nn`, no `candle-transformers`), tokenization, length-sorted batching, last-token pooling, L2 normalization, the task vocabulary, and model loading (safetensors or GGUF; local directory first, Hub second).
- **`crates/harrier_api`** — the axum HTTP server (`POST /v1/embeddings`, `GET /v1/models`, `GET /v1/models/{id}`, `GET /health`) and the `download` command.

Dependencies are kept to the candle / axum ecosystems: `candle-core`, `candle-nn`, `tokenizers`, `hf-hub` (optional), `axum`, `tokio`, `serde`, `serde_json`. No tracing, clap, anyhow, tower-http or reqwest.

## Quick start

Three steps: build for your hardware, put the weights next to the project, serve. Rust 1.88+; for NVIDIA the CUDA toolkit (`nvcc`) — on Windows see [CUDA on Windows](#cuda-on-windows) for the three environment variables it needs.

### 1. Build

```bash
cargo build --release -p harrier_api                      # CPU
cargo build --release --features cuda  -p harrier_api     # NVIDIA   (alias: cargo cuda)
cargo build --release --features metal -p harrier_api     # Apple Silicon
```

The binary is `target/release/harrier_api` (or wherever your cargo `target-dir` points).

### 2. Get the weights (once)

```bash
target/release/harrier_api download 0.6b                  # safetensors, 1.2 GB   → bf16 on GPU, f32 on CPU
target/release/harrier_api download 0.6b:q8_0             # 8-bit GGUF, 0.64 GB   → int8 weights on GPU or CPU
target/release/harrier_api download 270m 270m:q8_0        # the small model, both forms
```

Files land in `./models/<id>/` as plain files (see [below](#where-the-models-live)). Skipping this step also works: a model that is not there is fetched from the Hugging Face Hub on first start.

### 3. Serve

```bash
HARRIER_MODELS=0.6b target/release/harrier_api                          # auto-picks CUDA/Metal when built in, else CPU
HARRIER_DEVICE=cuda HARRIER_MODELS=0.6b:q8_0 target/release/harrier_api # 8-bit on the GPU (refuses to fall back to CPU)
HARRIER_MODELS=0.6b:q8_0,270m target/release/harrier_api                # several models; the first is the default
```

PowerShell:

```powershell
$env:HARRIER_DEVICE = "cuda"; $env:HARRIER_MODELS = "0.6b:q8_0"
.\target\release\harrier_api.exe
```

Startup prints what it loaded and where it runs, then listens on `0.0.0.0:3000`:

```text
harrier_api 0.1.0 · backends: cpu, cuda · device: cuda:0 · models: 0.6b:q8_0 · model dir: models
loaded harrier::harrier-oss-v1-0.6b (Qwen3, 1024-d, max 32768 tokens, cuda / q8_0) in 1.3s
warmed up in 88 ms
listening on http://0.0.0.0:3000
```

### 4. Embed

```bash
curl -s localhost:3000/v1/embeddings -H 'content-type: application/json' -d '{
  "model": "harrier::harrier-oss-v1-0.6b",
  "task": "query",
  "input": ["how much protein should a female eat", "summit define"]
}'
```

```json
{
  "output": [[0.0255, -0.0463, 0.0, 0.0294, ...], [0.0025, -0.0588, ...]],
  "usage": { "input": { "prompt_tokens": 49 }, "total": { "total_tokens": 49 } }
}
```

The request is the same whichever weights the model runs from — `model` is the model id (`harrier::harrier-oss-v1-0.6b`, `harrier-oss-v1-0.6b` or just `0.6b`), never the quantisation. `GET /v1/models` shows the `runtime` (`device`, `dtype`, `quantization`) of each loaded model.

Ready-made requests for every endpoint, task, model-name form and error case are in [`examples/http/`](examples/http/) (`embeddings.http`, `models.http`, `health.http`) for the VS Code REST Client and the JetBrains HTTP Client, and as a Postman collection in [`examples/postman/`](examples/postman/): import `harrier_api.postman_collection.json` plus the `local` (port 3000) or `docker-gguf` (port 13265) environment and run it top to bottom — every request carries tests (status, vector count / size / normalisation, the model-card scores), and the collection's `model` variable picks which loaded model the task examples use.

## Where the models live

```
harrier_embeddings/
├── crates/
├── docker/
└── models/                               ← HARRIER_MODEL_DIR (default ./models), git-ignored
    ├── harrier-oss-v1-0.6b/
    │   ├── config.json  tokenizer.json  config_sentence_transformers.json …
    │   ├── model.safetensors                 ← `download 0.6b`
    │   └── harrier-0.6b-q8_0.gguf            ← `download 0.6b:q8_0`
    └── harrier-oss-v1-270m/
        └── …
```

- `harrier_api download 270m 0.6b` fetches a model (Hub or another directory) and copies it into `models/<id>/` as **plain files** — no symlinks, no cache hashes — so the folder can be mounted into a container, baked into an image, or copied to an air-gapped box. `download 0.6b:q8_0` fetches that GGUF plus the tokenizer and config (not the safetensors) into the same folder.
- Resolution order for a model name: a `.gguf` file path → an explicit directory path → `models/<id>` (for ids, aliases and Hub repos alike) → the Hub (`HF_HOME` / `HF_TOKEN` honoured). A `:<quant>` suffix (`0.6b:q8_0`, `0.6b:q4_k`) picks that GGUF from `models/<id>/` or the model's GGUF repo; tokenizer and config come from the base checkpoint. With the folder populated, startup makes no network calls.
- Put the folder somewhere else with `HARRIER_MODEL_DIR=/srv/harrier/models`. Keep it out of version control (it is in `.gitignore`); it is data, not code.

## Configuration

`harrier_api --help` prints the same table. Which weights to run from: **bf16 safetensors** (`0.6b`) on a GPU is the default and the fastest; **q8_0 GGUF** (`0.6b:q8_0`) runs at the same speed on the GPU with half the memory, and on the CPU saves memory at about half the speed of f32 — see [Throughput](#throughput).

| Variable | Meaning | Default |
|---|---|---|
| `HARRIER_MODELS` | Comma-separated models to load; the first is the default. Registry id, alias (`270m`, `0.6b`, `27b`), Hub repo (`owner/name[@rev]`), a directory, `<model>:<quant>` for a GGUF quantisation (`0.6b:q8_0`) or a `.gguf` path (with `tokenizer.json` beside it) | `harrier-oss-v1-0.6b` |
| `HARRIER_MODEL_DIR` | Directory of exported models, checked before the Hub | `./models` |
| `HARRIER_DEVICE` | `auto` · `cpu` · `cuda[:n]` · `metal[:n]` · `gpu` (any accelerator, never CPU) | `auto` |
| `HARRIER_DTYPE` | `f32` · `bf16` · `f16` — the activation dtype (and the weight dtype for safetensors; GGUF weights stay int8) | `bf16` on GPU, `f32` on CPU; `f32` for GGUF |
| `HARRIER_DEQUANTIZE` | Dequantise GGUF weights to `HARRIER_DTYPE` at load: dense-speed matmuls at dense-size memory (see [Throughput](#throughput)) | `false` |
| `HARRIER_MAX_INPUT_TOKENS` | Truncate inputs to this many tokens | model max (32 768) |
| `HARRIER_MAX_BATCH_SIZE` | Rows per forward pass | `32` |
| `HARRIER_MAX_BATCH_TOKENS` | Padded tokens (`rows × longest row`) per forward pass | `16384` |
| `HARRIER_WARMUP` | One forward pass at startup | `true` |
| `HARRIER_LOG_REQUESTS` | One stderr line per request | `true` |
| `HARRIER_HOST`, `HARRIER_PORT` / `PORT` | Bind address | `0.0.0.0`, `3000` |

## The contract

`POST /v1/embeddings` takes the GAISe `GaiseEmbeddingsRequest` and returns a `GaiseEmbeddingsResponse`:

| Field | Meaning |
|---|---|
| `model` | `harrier::harrier-oss-v1-270m`, the bare id, the Hub repo, or an alias. Empty → the service default. Any `provider::` prefix is stripped, so a GAISe router can forward requests unchanged. |
| `input` | one string or an array; vectors come back in input order |
| `task` | what the vectors are for — see below. Default `document`. |
| `instruction` | *(Harrier extension)* an explicit instruction sentence; overrides `task` and applies to every input |
| `dimensions` | accepted, ignored — Harrier vectors are fixed-size (GAISe's `fixed` rule) |
| `normalize` | unset / `true` → unit-length (the model's documented output); `false` → the raw pooled hidden state |
| `correlation_id` | accepted, not used |

### Tasks → instructions

Harrier is an instruction-following embedder: queries are wrapped as `Instruct: {task}\nQuery: {text}`, documents are embedded bare. Every canonical GAISe task (and its aliases, case-insensitive, `-`/space = `_`) maps to one instruction:

| Task | Aliases | Instruction |
|---|---|---|
| `document` (default) | `retrieval_document`, `search_document`, `index`, `passage`, `generic_index` | *none* |
| `query` | `retrieval_query`, `search_query`, `search`, `generic_retrieval`, `text_retrieval`, `web_search_query` | `Given a web search query, retrieve relevant passages that answer the query` ¹ |
| `similarity` | `semantic_similarity`, `sts`, `sentence_similarity`, `sts_query` | `Retrieve semantically similar text` ¹ |
| `classification` | `classify` | `Classify the given text into one of the predefined categories` |
| `clustering` | `cluster` | `Identify the topic or theme of the given text` |
| `code_query` | `code_retrieval_query`, `code`, `code_retrieval` | `Given a code search query, retrieve relevant code snippets or documentation that answer the query` |
| `fact_verification` | `fact`, `fact_checking` | `Given a claim, retrieve documents that support or refute the claim` |
| `question_answering` | `qa` | `Given a question, retrieve passages that answer the question` ² |
| `bitext_query` | `bitext`, `bitext_mining`, `parallel_sentences` | `Retrieve parallel sentences` ¹ |
| anything else | — | used verbatim as the instruction (the escape hatch: `"task": "Given a legal question, retrieve the relevant statute"`) |

¹ The prompts shipped in the model's `config_sentence_transformers.json`. ² From the model's own `mteb_v2_eval_prompts.json`. Symmetric tasks (`similarity`, `classification`, `clustering`) put the instruction on every input, as MTEB does.

`GET /v1/models` returns GAISe-shaped records (`id`, `provider`, `capabilities`, `limits`) plus an `embedding` profile (`dimension_rule: fixed`, `task_control`, the resolved `task_instructions`) and `runtime` facts (device, dtype, `quantization` — `q8_0` for GGUF, `null` for dense — `dequantized`, architecture, where the weights came from). A model keeps its id whichever weights it runs from, so clients never see the quantisation; load one variant of each model. `?operation=` and `?provider=` filter like GAISe; `GET /v1/models/harrier::harrier-oss-v1-0.6b` returns one record.

Errors are plain-text bodies like gaise-api: `400` for bad JSON / empty input / unknown `operation`, `404` for a model that is not loaded, `500` for inference failures.

## Library use

```rust
use harrier_core::{Embedder, EmbedderOptions, EmbeddingTask, EmbeddingsRequest, DeviceRequest};

let embedder = Embedder::load("270m", EmbedderOptions { device: DeviceRequest::Auto, ..Default::default() })?;
let response = embedder.embed_request(&EmbeddingsRequest {
    input: vec!["how much protein should a female eat".into()].into(),
    task: Some(EmbeddingTask::Query),
    ..Default::default()
})?;
assert_eq!(response.output[0].len(), 640);

// or, without the task layer:
let raw = embedder.embed(&["a document".to_string()], true)?;
```

`Embedder` is `Send + Sync`; forward passes are serialized through an internal lock so one `Arc<Embedder>` serves a whole server. A CLI example prints vectors and pairwise cosines:

```bash
cargo run --release -p harrier_core --example embed -- --model 0.6b --task query "how much protein should a female eat" "summit define"
```

## How it works

- **Backbones.** `Qwen3Model` (0.6B) and `Gemma3TextModel` (270M, 27B) are implemented in `harrier_core::models` on `candle-nn` primitives. Gemma's `(1 + w)` RMSNorm is kept in f32 like `transformers`, the √hidden embedding scale is cast to the weight dtype, sliding-window layers use the local RoPE base and global layers the linearly-scaled one, and both `rope_theta`/`rope_scaling` (transformers 4.x) and `rope_parameters` (transformers 5) config layouts are read.
- **Batching with real masks.** Inputs are tokenized, sorted by length and grouped under `HARRIER_MAX_BATCH_SIZE` / `HARRIER_MAX_BATCH_TOKENS`; each group is right-padded and run with a causal + padding (+ sliding-window) mask, then the last real token of every row is pooled. Attention scores are scaled, masked and soft-maxed in f32 and computed in query chunks so memory stays flat for long inputs instead of growing with `seq_len²`.
- **No KV cache, no `candle-transformers`.** An embedding model runs one forward pass per batch, so every module takes `&self`.
- **GGUF weights are a weight source, not a second model.** The same batched backbones load through `models::weights::Weights`, which is either a safetensors `VarBuilder` or a GGUF file whose tensors are read straight onto the device as `QTensor`s and stay quantised: linear layers run candle's fused quantised matmuls (on CUDA the llama.cpp-style **MMQ int8 tensor-core kernel** for Q8_0 / Q4_K; activations go in as f32), the token table is gathered row-by-row from the int8 data, and norms are dequantised once (llama.cpp folds Gemma's `+1` into the stored norms, so those are used as-is). On the CPU candle's generic quantised matmul walks one output row at a time, so Q8_0 uses a **tiled int8 kernel** (`models/qlinear.rs`: activations quantised to Q8_0 on the fly, `maddubs`/`madd` int32 accumulation, runtime AVX2/FMA dispatch, rayon over column strips) instead. `transformers` parameter names are mapped to llama.cpp's (`layers.3.self_attn.q_proj` → `blk.3.attn_q`) in `models/gguf.rs`; a `config.json` beside the GGUF is preferred for hyper-parameters and the GGUF metadata is used when there is none.

### Validation

Outputs were checked against `transformers` 5.15 (float32, eager attention) on the model-card example and mixed-length padded batches, including an empty string and a 3 134-token document:

| Model | Device / dtype | max \|Δ\| vs transformers | cosine |
|---|---|---|---|
| harrier-oss-v1-270m (Gemma 3) | CPU f32 · native and `scratch` container | 3.1e-7 | 1.000000 |
| harrier-oss-v1-270m | CUDA f32 (RTX 4080, container) | 3.4e-7 | 1.000000 |
| harrier-oss-v1-270m | CUDA bf16 | 2.0e-3 | 0.99994 |
| harrier-oss-v1-0.6b (Qwen3) | CPU f32 · native and `scratch` container | 4.3e-6 | 1.000000 |
| harrier-oss-v1-0.6b | CUDA f32 | 2.2e-7 | 1.000000 |
| harrier-oss-v1-0.6b | CUDA bf16 | 9.5e-3 | 0.99963 |
| synthetic Gemma 3 with sliding + global layers, window 16, linear RoPE ×8, GQA (the 27B layer mix) | CPU f32 | 4.8e-7 | 1.000000 |
| harrier-oss-v1-0.6b **q8_0 GGUF** vs the f32 checkpoint (queries, documents, a 3.2k-token document, padded batch) | CPU and CUDA | 4.7e-3 | ≥ 0.99954 |
| harrier-oss-v1-270m **q8_0 GGUF** vs the f32 checkpoint | CPU and CUDA | 7.3e-3 | ≥ 0.99937 |

bf16 rows show the normal half-precision drift (the checkpoints ship in bf16 and that is how `transformers` runs them on GPUs too); f32 rows show the kernels are exact on both backends. The q8_0 rows are int8 weight noise — about five times the bf16 drift — with the model-card scores within 0.3 of the reference; an empty input (a lone EOS token) is the worst case at cosine 0.997. On CUDA the same row can differ at that noise level between a batch of ≤ 8 rows and a larger one, because candle dispatches them to different quantised kernels (MMVQ vs MMQ); the CPU kernel is deterministic.

The model-card scores reproduce exactly (0.6B: `[[66.27, 28.46], [29.77, 69.94]]`). `cargo test --release -p harrier_core --test reference -- --ignored` re-runs the 270M and 0.6B checks from `./models` (or the Hub), including the q8_0 comparisons; `HARRIER_TEST_DEVICE=cuda` / `metal` runs them on an accelerator (bf16 tolerance applies).

### Throughput

RTX 4080 (native Windows CUDA build) and i9-13900K; best of a few requests each, server-side, after warm-up. The 0.6B is the model these numbers are for.

| 0.6B | 32 short docs (~19 tokens) | 32 × 256-token docs | one 3.2k-token document |
|---|---|---|---|
| CUDA · bf16 safetensors (default) | 23 ms · 27 700 tok/s | 208 ms · 34 100 tok/s | 474 ms · 6 750 tok/s |
| CUDA · **q8_0 GGUF** (f32 activations, default) | 25 ms · 25 600 tok/s | 219 ms · 32 400 tok/s | 440 ms · 7 270 tok/s |
| CUDA · q8_0 GGUF, bf16 activations | 25 ms | 235 ms | 478 ms |
| CUDA · f32 safetensors | 40 ms · 15 700 tok/s | 373 ms · 19 000 tok/s | 515 ms · 6 200 tok/s |
| CPU · f32 safetensors | 1.35 s · 465 tok/s | 14.1 s · 503 tok/s | 20.4 s · 157 tok/s |
| CPU · q8_0 GGUF (int8 kernel) | 2.39 s · 264 tok/s | 28.5 s · 249 tok/s | 26.8 s · 120 tok/s |
| CPU · q8_0 GGUF, `HARRIER_DEQUANTIZE=true` | 1.38 s · 457 tok/s | 14.1 s · 504 tok/s | (= f32) |

What that says:

- **GPU:** the int8 MMQ kernels run at **parity with bf16** tensor cores for a model this small (and 1.7× faster than f32), so q8_0 is the pick when the 0.64 GB weight footprint matters — it is not a speed-up over bf16. The 3.2k-token column is bound by the exact, unfused attention (`seq_len²` scores) on every variant; that, not the projections, is the next thing to optimise.
- **CPU:** the int8 kernel is about **half the speed of the f32 GEMM** for batched work. On an AVX2 CPU without VNNI (this i9-13900K), `maddubs`/`madd`/`cvt`/`fmadd` per 32 products has no MAC-rate advantage over f32 FMA, and the `gemm` crate reaches ~1.1 TFLOP/s here against ~0.35 TFLOP/s for the int8 kernel (`cargo test --release -p harrier_core bench_q8 -- --ignored --nocapture`). So on CPU q8_0 buys memory (0.7 GB resident vs 2.4 GB for f32 weights), not speed; `HARRIER_DEQUANTIZE=true` turns a GGUF into f32 weights at load and gets the f32 speed back when RAM allows. `--features mkl` (Intel) / `accelerate` (macOS) speed up the f32 GEMMs further.

Mixed-length requests are split so that short texts are never padded to the length of a long one (a row joins a batch only if it is at least half as long as the batch's longest row).

## Building for each backend

| Backend | Build | Notes |
|---|---|---|
| CPU | `cargo build --release -p harrier_api` | f32 by default; add `--features mkl` or `--features accelerate` for faster GEMMs |
| CUDA | `cargo build --release --features cuda -p harrier_api` or `cargo cuda` (`cudnn` also available) | needs the CUDA toolkit (`nvcc`) and a host compiler it supports; see [Windows](#cuda-on-windows) below. `CUDA_COMPUTE_CAP` selects the GPU target. |
| Metal | `cargo build --release --features metal -p harrier_api` | Apple Silicon; bf16 by default |

Feature flags are plain pass-throughs to candle (`cuda`, `cudnn`, `metal`, `accelerate`, `mkl`); `hf-hub` (default) enables Hub downloads — disable it to load only local directories with fewer dependencies.

#### CUDA on Windows

candle compiles its kernels with `nvcc` at build time, so the CUDA toolkit and the MSVC host compiler have to agree:

- **CUDA 12.x accepts MSVC 2017–2022 only** (`cl.exe` 19.1x–19.4x). With Visual Studio 2026 (MSVC 14.5x) the build stops at `candle-kernels` with `STL1002: Unexpected compiler version, expected CUDA 13.2 or newer` — `-allow-unsupported-compiler` does not help, the STL hard-blocks it. Either install **CUDA 13.2+** or add the "MSVC v143 – VS 2022 C++ x64/x86 build tools" component.
- `cl.exe` must be reachable by nvcc: build from an "x64 Native Tools" prompt, or set `NVCC_CCBIN` to the `cl.exe` path.
- With **CUDA 13.x** also set `NVCC_APPEND_FLAGS="-Xcompiler /Zc:preprocessor"` — its CCCL headers need the conforming preprocessor.
- `CUDA_COMPUTE_CAP=89` (RTX 40xx; 86 = RTX 30xx, 80 = A100, 90 = H100) skips the device probe.

The three variables live best in `~/.cargo/config.toml` under `[env]` (a commented template is in this repo's `.cargo/config.toml`). That is how the build above was made: CUDA 13.3, VS 2026 / MSVC 14.51, RTX 4080, driver 610; `cargo cuda` then takes about two minutes from clean. At run time the binary loads CUDA's DLLs from `PATH` — the toolkit installer adds `<toolkit>\bin` (and, for CUDA 13, `<toolkit>\bin\x64`). The Docker image (below) is the alternative when the toolchain cannot be changed.

### Hardware

| Model | bf16 weights | q8_0 GGUF | Comfortable on |
|---|---|---|---|
| 0.6B | 1.2 GB | 0.64 GB | any modern CPU, any GPU ≥ 4 GB |
| 270M | 0.54 GB | 0.30 GB | any CPU, any GPU |
| 27B | 54 GB | — (a Q8_0 conversion would be ~29 GB; pass it as a `.gguf` path) | 80 GB GPU (A100/H100); does not fit a 16–24 GB card in bf16 |

Activation memory is dominated by attention scores, bounded to ~256 MiB per chunk regardless of length.

## Docker

The quickest route is Compose (`docker-compose.yml` at the repo root):

```bash
docker compose up -d                        # 8-bit GGUF on the GPU, weights baked in, http://localhost:13265
docker compose --profile cuda up -d         # bf16 safetensors on the GPU, port 3000, weights from the volume
docker compose --profile cpu up -d          # CPU image, port 3000, weights from the volume
docker compose run --rm download 0.6b 270m  # fill the shared weights volume for the cuda / cpu profiles
```

`CUDA_COMPUTE_CAP`, `HARRIER_GGUF_MODELS`, the host ports and the `HARRIER_*` server variables can be set in a `.env` file; the GPU services carry the `deploy.resources.reservations.devices` block that attaches the GPU (see the note on `libcuda.so.1` below).

Three images, all built from the repo root and all reading models from `/models` (`HARRIER_MODEL_DIR`) unless baked:

| Image | Base | Contents | Size |
|---|---|---|---|
| `docker/Dockerfile.cpu` | `scratch` | one static musl binary | **11.5 MB** |
| `docker/Dockerfile.cuda` | `gcr.io/distroless/cc-debian12` | glibc binary + `libcublas`, `libcublasLt`, `libcurand`, `libnvrtc` (the only CUDA libraries candle links; `libcuda` is injected from the host driver) | **750 MB** |
| `docker/Dockerfile.cuda.gguf` | same as above | same binary, serves the **8-bit GGUF** (`0.6b:q8_0` by default) on port **13265**; `--target bake` downloads the GGUF + tokenizer + config at build time for a self-contained image | 750 MB (+ 0.66 GB baked) |

```bash
# build
docker build -f docker/Dockerfile.cpu  -t harrier-api:cpu  .
docker build -f docker/Dockerfile.cuda -t harrier-api:cuda . --build-arg CUDA_COMPUTE_CAP=89   # 80 A100 · 86 RTX 30xx · 89 RTX 40xx · 90 H100

# mount the model folder (download once, on the host or through the image)
docker run --rm -v ./models:/models harrier-api:cpu download 270m 0.6b
docker run --rm -p 3000:3000 -v ./models:/models -e HARRIER_MODELS=270m,0.6b harrier-api:cpu
docker run --rm --gpus all -p 3000:3000 -v ./models:/models -e HARRIER_MODELS=0.6b harrier-api:cuda

# or bake the models in for a self-contained image (no volume, no network)
docker build -f docker/Dockerfile.cpu --target bake -t harrier-api:cpu-models .

# 8-bit GGUF on the GPU: the bake target fetches the weights itself (no host download needed)
docker build -f docker/Dockerfile.cuda.gguf --target bake -t harrier-api:cuda-gguf-models . --build-arg CUDA_COMPUTE_CAP=89
docker run --rm --gpus all -p 13265:13265 harrier-api:cuda-gguf-models
curl -s localhost:13265/v1/models            # runtime.quantization: "q8_0"
# HARRIER_MODELS=0.6b:q8_0,270m:q8_0 as a --build-arg bakes several; without --target bake it reads /models like the others
```

Both images were built and run here: the CPU one on the host (exact match to the reference), the CUDA one on an RTX 4080 through Docker Desktop's WSL2 GPU support (f32 exact, bf16 within rounding).

Docker Desktop on Windows: a bind mount of a Windows path (`-v ./models:/models`) goes through a slow 9p bridge, so model loading takes ~20 s instead of ~2 s; a named volume (`docker volume create harrier-models`, then `-v harrier-models:/models` and `… download 270m 0.6b` once) or a baked image avoids that.

The CUDA images need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host (Docker Desktop on Windows/WSL2 ships it), a driver that supports the toolkit version the image was built with (`CUDA_VERSION` build arg, default 12.6), **and the GPU attached at start** — `docker run --gpus all …` or `gpus: all` in compose. `libcuda.so.1` is the host driver and is only mounted into the container when a GPU is attached; without it the binary fails before `main` with `error while loading shared libraries: libcuda.so.1: cannot open shared object file` (Docker Desktop's "Run" button does not attach GPUs). candle links the driver at load time (`cudarc/dynamic-linking`), so there is no CPU fallback in these images — use `Dockerfile.cpu` for that. The Dockerfile's build stage is also the easiest way to produce a Linux CUDA binary from a Windows machine whose MSVC is newer than nvcc supports.

**Metal cannot run in a container** — Docker on macOS runs a Linux VM with no GPU access. For Apple Silicon build natively (`cargo build --release --features metal -p harrier_api`) and run the binary directly, or as a `launchd` service; the same `./models` folder layout applies.

## What it does not do (yet)

- Flash / paged attention — attention is exact and chunked, not fused; it dominates long inputs on every weight format.
- Lower GGUF quantisations are loadable (`0.6b:q4_k` is published and runs through the same MMQ / generic kernels) but only q8_0 has been validated against the reference; the CPU fast path is Q8_0-only, other quantisations fall back to candle's generic CPU matmul.
- Matryoshka truncation — `dimensions` is accepted for contract compatibility and ignored, as Harrier documents fixed-size vectors.
- Cross-request batching — each request is batched internally; concurrent requests queue on the model lock.

## License

[GNU AGPL v3 or later](LICENSE). The models themselves are released by Microsoft under MIT.
