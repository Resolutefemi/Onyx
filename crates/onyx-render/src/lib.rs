//! # onyx-render
//!
//! Rendering subsystem for the Onyx Engine — `docs/PRD.md` §2 (Rendering &
//! Graphics Pipeline), §3 (Lighting, Shadows & Atmosphere) and §4 (Materials,
//! Shaders & Post-Processing).
//!
//! M2 (first increment) lands the hardware foundation:
//!
//! - [`renderer::Renderer`] — `wgpu` instance/adapter/device/queue setup with
//!   graceful degradation: GPU-less (headless) machines get
//!   [`renderer::RenderError::NoAdapter`], never a panic,
//! - [`graph::RenderGraph`] — named nodes with explicit dependencies,
//!   Kahn topological execution order, cycle detection (resource-based
//!   automatic barrier insertion lands with the real passes),
//! - [`offscreen::render_offscreen`] — clear-pass rendering into an offscreen
//!   RGBA8 sRGB texture with GPU readback, validating the whole device →
//!   encoder → submit → map pipeline without needing a window.
//!
//! Still ahead in M2: surfaces and present, the DAG-driven parallel graph
//! execution with automatic barriers, forward PBR, the 2D suite and the
//! post-processing stack.

pub mod graph;
pub mod offscreen;
pub mod renderer;

/// Physical graphics backend targeted by the hardware abstraction layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Vulkan (primary desktop backend; raw `ash` fast-path in M2).
    Vulkan,
    /// DirectX 12 (Windows).
    Dx12,
    /// Metal (macOS, iOS).
    Metal,
    /// WebGPU (browser target).
    WebGPU,
}

impl Backend {
    /// Stable machine-readable backend name.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::Dx12 => "dx12",
            Self::Metal => "metal",
            Self::WebGPU => "webgpu",
        }
    }

    /// Every backend the HAL targets, in preference order.
    #[must_use]
    pub fn all() -> &'static [Backend] {
        &[Self::Vulkan, Self::Dx12, Self::Metal, Self::WebGPU]
    }
}

/// Renderer configuration resolved at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RendererConfig {
    /// Requested backend; `None` lets the HAL pick the best available.
    pub backend: Option<Backend>,
    /// Whether presentation syncs to the display's refresh rate.
    pub vsync: bool,
    /// MSAA sample count for the forward path.
    pub msaa_samples: u32,
    /// Initial backbuffer width in pixels.
    pub backbuffer_width: u32,
    /// Initial backbuffer height in pixels.
    pub backbuffer_height: u32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            backend: None,
            vsync: true,
            msaa_samples: 4,
            backbuffer_width: 1280,
            backbuffer_height: 720,
        }
    }
}

/// Handle to a pipeline registered with the render graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(pub u32);

pub use graph::{GraphError, RenderGraph, RenderGraphNode};
pub use offscreen::{render_offscreen, OffscreenFrame};
pub use renderer::{RenderError, Renderer};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_are_unique() {
        let names: Vec<&str> = Backend::all().iter().map(Backend::as_str).collect();
        assert_eq!(names.len(), 4);
        for (i, a) in names.iter().enumerate() {
            for b in names.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn default_config_uses_auto_backend_and_vsync() {
        let config = RendererConfig::default();
        assert_eq!(config.backend, None);
        assert!(config.vsync);
        assert_eq!(config.msaa_samples, 4);
    }
}
