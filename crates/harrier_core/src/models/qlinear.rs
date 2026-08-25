//! Quantised linear layers.
//!
//! On CUDA and Metal a [`QMatMul`] runs candle's fused quantised kernels
//! (the llama.cpp-style MMQ int8 tensor-core matmul for Q8_0 / Q4_K on
//! CUDA). On the CPU candle's generic quantised matmul walks one output row
//! at a time and re-reads the whole weight matrix per row, which at
//! embedding shapes (hundreds of tokens × 1–3k features) leaves most cores
//! idle; [`forward_q8`] instead tiles over strips of output columns across
//! *all* rows, so each weight block is loaded once per strip and every core
//! gets a fat task. The inner product is the ggml Q8_0×Q8_0 recipe:
//! activations are quantised to Q8_0 on the fly, int8 products accumulate in
//! int32, one FMA per block applies the two scales. AVX2/FMA is selected at
//! runtime; other CPUs take the scalar path.
//!
//! Know its limits: without VNNI (`vpdpbusd`) an int8 dot product has no
//! MAC-rate advantage over f32 FMA, so on such CPUs this kernel runs at
//! roughly a third of the `gemm` crate's f32 throughput for batched work
//! (`bench_q8_vs_f32` below). It buys memory, not speed; for speed on the
//! CPU dequantise at load (`EmbedderOptions::dequantize`).

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use rayon::prelude::*;

/// Elements per Q8_0 block.
const BLOCK: usize = 32;
/// Bytes per Q8_0 block on disk: f16 scale + 32 int8.
const BLOCK_BYTES: usize = 2 + BLOCK;
/// Output columns per rayon task.
const COL_TILE: usize = 16;
/// Weight rows handled together per activation row (independent accumulators).
const COL_BLOCK: usize = 4;

/// Q8_0 data split into planes: one `[i8; 32]` per block plus its scale as
/// f32, so the inner loop never converts f16 and both planes stream linearly.
struct Q8Planes {
    qs: Vec<[i8; BLOCK]>,
    scales: Vec<f32>,
    rows: usize,
    blocks_per_row: usize,
}

impl Q8Planes {
    #[inline]
    fn row_qs(&self, i: usize) -> &[[i8; BLOCK]] {
        &self.qs[i * self.blocks_per_row..(i + 1) * self.blocks_per_row]
    }

    #[inline]
    fn row_scales(&self, i: usize) -> &[f32] {
        &self.scales[i * self.blocks_per_row..(i + 1) * self.blocks_per_row]
    }
}

/// Q8_0 weights: `out` rows of `k / 32` blocks.
struct Q8Matrix {
    out: usize,
    k: usize,
    planes: Q8Planes,
}

impl Q8Matrix {
    fn from_qtensor(q: &QTensor) -> Result<Self> {
        let dims = q.shape().dims();
        if dims.len() != 2 {
            candle_core::bail!("Q8 linear expects a 2-D weight, got {dims:?}");
        }
        let (out, k) = (dims[0], dims[1]);
        if !k.is_multiple_of(BLOCK) {
            candle_core::bail!("Q8_0 weight width {k} is not a multiple of {BLOCK}");
        }
        let blocks_per_row = k / BLOCK;
        let bytes = q.data()?;
        let expected = out * blocks_per_row * BLOCK_BYTES;
        if bytes.len() != expected {
            candle_core::bail!(
                "Q8_0 weight has {} bytes, expected {expected} for {out}x{k}",
                bytes.len()
            );
        }
        let mut qs = Vec::with_capacity(out * blocks_per_row);
        let mut scales = Vec::with_capacity(out * blocks_per_row);
        for raw in bytes.as_chunks::<BLOCK_BYTES>().0 {
            let mut block = [0i8; BLOCK];
            for (dst, src) in block.iter_mut().zip(&raw[2..]) {
                *dst = *src as i8;
            }
            qs.push(block);
            scales.push(half::f16::from_le_bytes([raw[0], raw[1]]).to_f32());
        }
        Ok(Self {
            out,
            k,
            planes: Q8Planes {
                qs,
                scales,
                rows: out,
                blocks_per_row,
            },
        })
    }
}

/// Quantise `m` rows of `k` f32 activations to Q8_0 planes.
fn quantize_rows(x: &[f32], m: usize, k: usize) -> Q8Planes {
    let blocks_per_row = k / BLOCK;
    let mut qs = Vec::with_capacity(m * blocks_per_row);
    let mut scales = Vec::with_capacity(m * blocks_per_row);
    for chunk in x[..m * k].as_chunks::<BLOCK>().0 {
        let amax = chunk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut block = [0i8; BLOCK];
        for (q, &v) in block.iter_mut().zip(chunk) {
            *q = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        qs.push(block);
        // Round-trip through f16 so activations carry exactly the precision
        // the ggml reference kernels assume.
        scales.push(half::f16::from_f32(d).to_f32());
    }
    Q8Planes {
        qs,
        scales,
        rows: m,
        blocks_per_row,
    }
}

/// One activation row against `W` weight rows: `out[j] = x · w_j`.
#[inline]
fn dot_scalar<const W: usize>(
    xq: &[[i8; BLOCK]],
    xs: &[f32],
    wq: [&[[i8; BLOCK]]; W],
    ws: [&[f32]; W],
) -> [f32; W] {
    let mut acc = [0f32; W];
    for b in 0..xq.len() {
        for j in 0..W {
            let mut sum = 0i32;
            for i in 0..BLOCK {
                sum += xq[b][i] as i32 * wq[j][b][i] as i32;
            }
            acc[j] += sum as f32 * xs[b] * ws[j][b];
        }
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const W: usize>(
    xq: &[[i8; BLOCK]],
    xs: &[f32],
    wq: [&[[i8; BLOCK]]; W],
    ws: [&[f32]; W],
) -> [f32; W] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); W];
    for b in 0..xq.len() {
        // SAFETY: every block is exactly 32 bytes — one unaligned 256-bit load.
        let qx = unsafe { _mm256_loadu_si256(xq[b].as_ptr() as *const __m256i) };
        // maddubs needs one unsigned operand: |qx| against sign-adjusted weights.
        let ax = _mm256_sign_epi8(qx, qx);
        let dx = xs[b];
        for j in 0..W {
            let qw = unsafe { _mm256_loadu_si256(wq[j][b].as_ptr() as *const __m256i) };
            let sw = _mm256_sign_epi8(qw, qx);
            let p16 = _mm256_maddubs_epi16(ax, sw); // 16 × i16, no saturation (|q| ≤ 127)
            let p32 = _mm256_madd_epi16(p16, ones); // 8 × i32
            let d = _mm256_set1_ps(dx * ws[j][b]);
            acc[j] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p32), acc[j]);
        }
    }
    let mut out = [0f32; W];
    for j in 0..W {
        let hi = _mm256_extractf128_ps(acc[j], 1);
        let lo = _mm256_castps256_ps128(acc[j]);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        out[j] = _mm_cvtss_f32(s);
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
}

fn detect_kernel() -> Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Kernel::Avx2;
        }
    }
    Kernel::Scalar
}

#[inline]
fn dot<const W: usize>(
    kernel: Kernel,
    xq: &[[i8; BLOCK]],
    xs: &[f32],
    wq: [&[[i8; BLOCK]]; W],
    ws: [&[f32]; W],
) -> [f32; W] {
    match kernel {
        Kernel::Scalar => dot_scalar::<W>(xq, xs, wq, ws),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        // SAFETY: `Avx2` is only produced after a successful runtime check.
        Kernel::Avx2 => unsafe { dot_avx2::<W>(xq, xs, wq, ws) },
    }
}

/// `xs`: `m` rows of `w.k` f32 activations → `m × w.out` f32, row-major.
fn forward_q8(w: &Q8Matrix, xs: &[f32], m: usize, kernel: Kernel) -> Vec<f32> {
    let xq = quantize_rows(xs, m, w.k);
    let n = w.out;
    let wp = &w.planes;

    // Each task: one strip of COL_TILE output columns across every row, four
    // columns at a time so the activation block is loaded once per four dots.
    let strips: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .step_by(COL_TILE)
        .map(|c0| {
            let c1 = (c0 + COL_TILE).min(n);
            let width = c1 - c0;
            let mut tile = vec![0f32; m * width];
            for r in 0..xq.rows {
                let xr = xq.row_qs(r);
                let xsr = xq.row_scales(r);
                let mut c = c0;
                while c + COL_BLOCK <= c1 {
                    let out = dot::<COL_BLOCK>(
                        kernel,
                        xr,
                        xsr,
                        [
                            wp.row_qs(c),
                            wp.row_qs(c + 1),
                            wp.row_qs(c + 2),
                            wp.row_qs(c + 3),
                        ],
                        [
                            wp.row_scales(c),
                            wp.row_scales(c + 1),
                            wp.row_scales(c + 2),
                            wp.row_scales(c + 3),
                        ],
                    );
                    tile[r * width + (c - c0)..r * width + (c - c0) + COL_BLOCK]
                        .copy_from_slice(&out);
                    c += COL_BLOCK;
                }
                while c < c1 {
                    let out = dot::<1>(kernel, xr, xsr, [wp.row_qs(c)], [wp.row_scales(c)]);
                    tile[r * width + (c - c0)] = out[0];
                    c += 1;
                }
            }
            tile
        })
        .collect();

    let mut out = vec![0f32; m * n];
    for (i, tile) in strips.iter().enumerate() {
        let c0 = i * COL_TILE;
        let width = (c0 + COL_TILE).min(n) - c0;
        for r in 0..m {
            out[r * n + c0..r * n + c0 + width].copy_from_slice(&tile[r * width..(r + 1) * width]);
        }
    }
    out
}

enum Inner {
    /// The tiled CPU kernel (Q8_0 weights on the CPU).
    Q8 { w: Q8Matrix, kernel: Kernel },
    /// candle's quantised matmul (accelerators, other quantisations).
    Generic(QMatMul),
}

/// A linear layer over quantised weights (`[out, in]`, no bias).
pub struct QLinear {
    inner: Inner,
    in_dim: usize,
    out_dim: usize,
}

impl std::fmt::Debug for QLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.inner {
            Inner::Q8 { kernel, .. } => format!("q8_0 cpu {kernel:?}"),
            Inner::Generic(_) => "qmatmul".to_string(),
        };
        f.debug_struct("QLinear")
            .field("in", &self.in_dim)
            .field("out", &self.out_dim)
            .field("kind", &kind)
            .finish()
    }
}

impl QLinear {
    pub fn new(q: QTensor, device: &Device) -> Result<Self> {
        let dims = q.shape().dims();
        if dims.len() != 2 {
            candle_core::bail!("quantised linear expects a 2-D weight, got {dims:?}");
        }
        let (out_dim, in_dim) = (dims[0], dims[1]);
        let inner = if device.is_cpu() && q.dtype() == GgmlDType::Q8_0 {
            Inner::Q8 {
                w: Q8Matrix::from_qtensor(&q)?,
                kernel: detect_kernel(),
            }
        } else {
            Inner::Generic(QMatMul::from_qtensor(q)?)
        };
        Ok(Self {
            inner,
            in_dim,
            out_dim,
        })
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}

impl Module for QLinear {
    /// `x: (…, in)` → `(…, out)`, any number of leading dimensions. The
    /// output has the input's dtype; the quantised kernels themselves run on
    /// f32 activations.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.inner {
            Inner::Q8 { w, kernel } => {
                let dims = x.dims().to_vec();
                let k = *dims.last().unwrap_or(&0);
                if k != w.k {
                    candle_core::bail!(
                        "Q8 linear: input {:?} does not match weight {}x{}",
                        dims,
                        w.out,
                        w.k
                    );
                }
                let m = x.elem_count() / k.max(1);
                let dtype = x.dtype();
                let xs = x.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let out = forward_q8(w, &xs, m, *kernel);
                let mut out_dims = dims;
                *out_dims.last_mut().unwrap() = w.out;
                Tensor::from_vec(out, out_dims, x.device())?.to_dtype(dtype)
            }
            Inner::Generic(m) => {
                // candle's quantised matmuls want contiguous f32 (CPU) or
                // f32 / f16 / bf16 (CUDA, Metal) activations.
                let x = x.contiguous()?;
                if x.device().is_cpu() && x.dtype() != DType::F32 {
                    let dtype = x.dtype();
                    m.forward(&x.to_dtype(DType::F32)?)?.to_dtype(dtype)
                } else {
                    m.forward(&x)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cargo test --release -p harrier_core bench_q8 -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark"]
    fn bench_q8_vs_f32() {
        use std::time::Instant;
        let device = Device::Cpu;
        for (m, k, n) in [
            (256usize, 1024usize, 3072usize),
            (2048, 1024, 3072),
            (2048, 3072, 1024),
        ] {
            let w = Tensor::randn(0f32, 1.0, (n, k), &device).unwrap();
            let x = Tensor::randn(0f32, 1.0, (1, m, k), &device).unwrap();
            let q = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
            let lin = QLinear::new(q, &device).unwrap();
            let dense = candle_nn::Linear::new(w.clone(), None);
            let _ = lin.forward(&x).unwrap();
            let _ = dense.forward(&x).unwrap();
            let reps = 5;
            let t = Instant::now();
            for _ in 0..reps {
                let _ = lin.forward(&x).unwrap();
            }
            let q8 = t.elapsed() / reps;
            let t = Instant::now();
            for _ in 0..reps {
                let _ = dense.forward(&x).unwrap();
            }
            let f32t = t.elapsed() / reps;
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            eprintln!(
                "m={m} k={k} n={n}: q8 {:?} ({:.0} GFLOP/s) · f32 gemm {:?} ({:.0} GFLOP/s)",
                q8,
                flops / q8.as_secs_f64() / 1e9,
                f32t,
                flops / f32t.as_secs_f64() / 1e9
            );
        }
    }

    #[test]
    fn avx2_matches_scalar_dot() {
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed % 255) as i32 - 127
        };
        let blocks = 8;
        let mut mk = |scale: f32| {
            let mut qs = Vec::new();
            let mut scales = Vec::new();
            for b in 0..blocks {
                let mut q = [0i8; BLOCK];
                for v in q.iter_mut() {
                    *v = next() as i8;
                }
                qs.push(q);
                scales.push(scale * (b + 1) as f32);
            }
            (qs, scales)
        };
        let (xq, xs) = mk(0.01);
        let (w0q, w0s) = mk(0.02);
        let (w1q, w1s) = mk(0.03);
        let (w2q, w2s) = mk(0.05);
        let (w3q, w3s) = mk(0.07);
        let wq = [&w0q[..], &w1q[..], &w2q[..], &w3q[..]];
        let ws = [&w0s[..], &w1s[..], &w2s[..], &w3s[..]];
        let s = dot_scalar::<4>(&xq, &xs, wq, ws);
        let k = dot::<4>(detect_kernel(), &xq, &xs, wq, ws);
        for j in 0..4 {
            assert!(
                (s[j] - k[j]).abs() <= 1e-3 * s[j].abs().max(1.0),
                "col {j}: scalar {} vs kernel {}",
                s[j],
                k[j]
            );
        }
        let one = dot::<1>(detect_kernel(), &xq, &xs, [&w2q[..]], [&w2s[..]]);
        assert!((one[0] - s[2]).abs() <= 1e-3 * s[2].abs().max(1.0));
    }

    #[test]
    fn quantised_forward_tracks_f32_matmul_for_batched_input() {
        let device = Device::Cpu;
        let (b, s, k, n) = (2usize, 5usize, 64usize, 37usize);
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7919) % 201) as f32 / 100.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..b * s * k)
            .map(|i| ((i * 104_729) % 113) as f32 / 56.0 - 1.0)
            .collect();
        let wt = Tensor::from_vec(w, (n, k), &device).unwrap();
        let xt = Tensor::from_vec(x, (b, s, k), &device).unwrap();
        let q = QTensor::quantize(&wt, GgmlDType::Q8_0).unwrap();
        let lin = QLinear::new(q, &device).unwrap();
        let got = lin.forward(&xt).unwrap();
        assert_eq!(got.dims(), &[b, s, n]);
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = xt
            .broadcast_matmul(&wt.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 0.05 * w.abs().max(1.0), "{g} vs {w}");
        }
        // 2-D input works the same way.
        let got2 = lin.forward(&xt.reshape((b * s, k)).unwrap()).unwrap();
        assert_eq!(got2.dims(), &[b * s, n]);
    }
}
