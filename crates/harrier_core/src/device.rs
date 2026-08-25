//! Device and dtype selection across CPU, CUDA and Metal builds.

use crate::error::{HarrierError, Result};
use candle_core::{DType, Device};
use std::fmt;
use std::str::FromStr;

/// Where to run. Parsed from `auto`, `cpu`, `cuda`, `cuda:1`, `metal`,
/// `metal:0`, or `gpu` (= auto but refuse CPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceRequest {
    /// CUDA if built with `cuda` and a GPU is present, then Metal if built
    /// with `metal`, otherwise CPU.
    #[default]
    Auto,
    Cpu,
    Cuda(usize),
    Metal(usize),
    /// Any accelerator; error rather than silently fall back to CPU.
    Gpu,
}

impl FromStr for DeviceRequest {
    type Err = HarrierError;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        let (kind, ordinal) = match s.split_once(':') {
            Some((k, o)) => (
                k.to_string(),
                Some(o.parse::<usize>().map_err(|_| {
                    HarrierError::Device(format!("invalid device ordinal in `{s}`"))
                })?),
            ),
            None => (s.clone(), None),
        };
        Ok(match kind.as_str() {
            "" | "auto" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" | "nvidia" => Self::Cuda(ordinal.unwrap_or(0)),
            "metal" | "mps" | "apple" => Self::Metal(ordinal.unwrap_or(0)),
            "gpu" => Self::Gpu,
            other => {
                return Err(HarrierError::Device(format!(
                    "unknown device `{other}` (expected auto, cpu, cuda[:n], metal[:n] or gpu)"
                )));
            }
        })
    }
}

impl fmt::Display for DeviceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(n) => write!(f, "cuda:{n}"),
            Self::Metal(n) => write!(f, "metal:{n}"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

/// Backends compiled into this build.
pub fn compiled_backends() -> &'static [&'static str] {
    match (cfg!(feature = "cuda"), cfg!(feature = "metal")) {
        (true, true) => &["cpu", "cuda", "metal"],
        (true, false) => &["cpu", "cuda"],
        (false, true) => &["cpu", "metal"],
        (false, false) => &["cpu"],
    }
}

/// Resolve a request to a concrete candle device.
pub fn select_device(request: DeviceRequest) -> Result<Device> {
    match request {
        DeviceRequest::Cpu => Ok(Device::Cpu),
        DeviceRequest::Cuda(ordinal) => cuda(ordinal),
        DeviceRequest::Metal(ordinal) => metal(ordinal),
        DeviceRequest::Auto => Ok(first_accelerator().unwrap_or(Device::Cpu)),
        DeviceRequest::Gpu => first_accelerator().ok_or_else(|| {
            HarrierError::Device(format!(
                "no accelerator available (compiled backends: {})",
                compiled_backends().join(", ")
            ))
        }),
    }
}

fn first_accelerator() -> Option<Device> {
    if cfg!(feature = "cuda")
        && let Ok(d) = cuda(0)
    {
        return Some(d);
    }
    if cfg!(feature = "metal")
        && let Ok(d) = metal(0)
    {
        return Some(d);
    }
    None
}

#[cfg(feature = "cuda")]
fn cuda(ordinal: usize) -> Result<Device> {
    Device::new_cuda(ordinal).map_err(|e| HarrierError::Device(format!("cuda:{ordinal}: {e}")))
}

#[cfg(not(feature = "cuda"))]
fn cuda(ordinal: usize) -> Result<Device> {
    Err(HarrierError::Device(format!(
        "cuda:{ordinal} requested but this binary was built without the `cuda` feature"
    )))
}

#[cfg(feature = "metal")]
fn metal(ordinal: usize) -> Result<Device> {
    Device::new_metal(ordinal).map_err(|e| HarrierError::Device(format!("metal:{ordinal}: {e}")))
}

#[cfg(not(feature = "metal"))]
fn metal(ordinal: usize) -> Result<Device> {
    Err(HarrierError::Device(format!(
        "metal:{ordinal} requested but this binary was built without the `metal` feature"
    )))
}

/// Human-readable device name for logs and `/v1/models`.
pub fn device_name(device: &Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(_) => "cuda".to_string(),
        Device::Metal(_) => "metal".to_string(),
    }
}

/// `bf16` on accelerators (the checkpoints ship in bf16), `f32` on CPU where
/// bf16 matmuls are slow and f32 is the accurate reference.
pub fn default_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    }
}

/// Parse `f32` / `bf16` / `f16` (aliases: `float32`, `bfloat16`, `float16`, `half`).
pub fn parse_dtype(s: &str) -> Result<DType> {
    match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => Ok(DType::F32),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "float16" | "half" => Ok(DType::F16),
        other => Err(HarrierError::Device(format!(
            "unknown dtype `{other}` (expected f32, bf16 or f16)"
        ))),
    }
}

pub fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F64 => "f64",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_requests() {
        assert_eq!(
            "auto".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Auto
        );
        assert_eq!("CPU".parse::<DeviceRequest>().unwrap(), DeviceRequest::Cpu);
        assert_eq!(
            "cuda".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Cuda(0)
        );
        assert_eq!(
            "cuda:2".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Cuda(2)
        );
        assert_eq!(
            "metal:1".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Metal(1)
        );
        assert_eq!("gpu".parse::<DeviceRequest>().unwrap(), DeviceRequest::Gpu);
        assert!("tpu".parse::<DeviceRequest>().is_err());
        assert!("cuda:x".parse::<DeviceRequest>().is_err());
    }

    #[test]
    fn cpu_always_resolves() {
        assert!(select_device(DeviceRequest::Cpu).unwrap().is_cpu());
        assert_eq!(default_dtype(&Device::Cpu), DType::F32);
        assert_eq!(parse_dtype("bfloat16").unwrap(), DType::BF16);
        assert!(parse_dtype("int8").is_err());
    }
}
