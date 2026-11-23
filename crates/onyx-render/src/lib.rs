//! # onyx-render
//!
//! Rendering subsystem for the Onyx Engine — `docs/PRD.md` §2 (Rendering &
//! Graphics Pipeline), §3 (Lighting, Shadows & Atmosphere) and §4 (Materials,
//! Shaders & Post-Processing).
//!
//! M2 (first increment) landed the hardware foundation:
//!
//! - [`renderer::Renderer`] — `wgpu` instance/adapter/device/queue setup with
//!   graceful degradation: GPU-less (headless) machines get
//!   [`renderer::RenderError::NoAdapter`], never a panic,
//! - [`offscreen::render_offscreen`] — clear-pass rendering into an offscreen
//!   RGBA8 sRGB texture with GPU readback.
//!
//! M2 (second increment) adds the first real GPU work:
//!
//! - [`graph::RenderGraph`] — nodes declare resource reads/writes; edges are
//!   inferred (producer before consumer, conflicting writers in declaration
//!   order) and cycles through resources are rejected,
//! - [`pipeline`] — the first WGSL pipeline: a vertex-colored triangle with a
//!   deterministic pixel contract (center = triangle color, corners =
//!   background),
//! - [`executor::execute_offscreen`] — graph execution: one command encoder,
//!   passes in topological order, one submission, per-target readback. wgpu
//!   inserts GPU barriers automatically from real resource usage within a
//!   submission.
//!
//! M2 (third increment) adds bind groups and the resource layer:
//!
//! - [`resources`] — bind group layout + bind group (uniform, texture,
//!   sampler), `QuadUniformBuffer` with queue writes, `Rgba8UnormSrgb`
//!   texture upload, nearest/linear samplers, a vertexless textured-quad
//!   pipeline with a pixel-exact round-trip contract (tint in linear space,
//!   uv sub-window sampling) and 4× MSAA with resolve,
//! - [`resources::srgb8_to_linear`] — the exact sRGB inverse for test
//!   contracts.
//!
//! M2 (fourth increment) scales the graph executor:
//!
//! - [`graph::RenderGraph::execution_waves`] — wavefront decomposition: wave
//!   0 is the producer-free set, a consumer lands after its latest producer;
//!   conflicting writers always cross waves,
//! - [`executor::execute_offscreen_parallel`] — each wave's nodes encode
//!   concurrently on their own threads and command encoders, each wave
//!   submits in one call, and submission order makes cross-wave dependencies
//!   safe; results are bit-identical to the sequential executor (tested),
//! - [`executor::ResourceSpec::samples`] — per-target MSAA: multisampled
//!   attachments resolve into single-sample readback textures, pipelines are
//!   built per sample count and validated against the adapter up front.
//!
//! Still ahead in M2: sampling passes inside the graph (reads), surfaces and
//! present, forward PBR, the 2D suite and the post-processing stack.

pub mod executor;
pub mod graph;
pub mod offscreen;
pub mod pipeline;
pub mod renderer;
pub mod resources;

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

pub use executor::{execute_offscreen, execute_offscreen_parallel, PassKind, ResourceSpec};
pub use graph::{GraphError, RenderGraph, RenderGraphNode};
pub use offscreen::{linear_to_srgb8, render_offscreen, OffscreenFrame};
pub use pipeline::{
    create_triangle_pipeline, render_triangle_offscreen, triangle_vertex_buffer, vertex_bytes,
    Vertex, TRIANGLE_WGSL,
};
pub use renderer::{RenderError, Renderer};
pub use resources::{
    create_quad_bind_group, create_quad_bind_layout, create_quad_pipeline, create_rgba_texture,
    create_sampler, create_triangle_pipeline_msaa, render_quad_offscreen,
    render_triangle_msaa_offscreen, srgb8_to_linear, texture_view, QuadUniform, QuadUniformBuffer,
    QUAD_WGSL,
};

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
