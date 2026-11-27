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
//! M2 (fifth increment) makes sampling passes first-class graph citizens:
//!
//! - [`executor::PassKind::TexturedQuad`] — a node whose declared read
//!   resource is bound as a texture through the quad bind group: it clears
//!   its write target and draws the source through tint and uv window;
//!   nearest + full-window + white tint is a pixel-exact copy (tested),
//! - one shared validation path (`plan_node`) for both executors: read
//!   declarations must match the pass kind, sampling requires declared
//!   single-sample resources on both ends,
//! - target textures gain `TEXTURE_BINDING` so they can be sampled by later
//!   passes; wgpu inserts the render-attachment → texture-binding barrier
//!   automatically.
//!
//! M2 (sixth increment) wires the executor to a real window:
//!
//! - [`executor::ResourceSpec::format`] — the executor is format-agnostic:
//!   pipelines are built per (sample count, format), BGRA readbacks are
//!   swizzled back to RGBA and linear (`*Unorm`) targets have their own
//!   byte-exact contract,
//! - [`executor::ExternalTarget`] +
//!   [`executor::execute_with_external_target`] — one declared target can be
//!   caller-provided (the swapchain frame): validated against its
//!   declaration, drawn into like any other target, never allocated, sampled
//!   or read back,
//! - [`surface`] — the deterministic swapchain policy (sRGB format, Fifo
//!   vsync) and [`surface::SurfaceRenderer`]: acquire → render the
//!   `backbuffer` node onto the swapchain → present.
//!
//! M2 (seventh increment) adds forward PBR:
//!
//! - [`pbr`] — the Cook-Torrance material pass: GGX distribution, Schlick
//!   visibility and fresnel, metallic/roughness parameters. The WGSL shader
//!   and the CPU mirror [`pbr::pbr_shaded`] share the exact operation
//!   sequence (only exactly-rounded IEEE f32 operations), so the GPU output
//!   is predictable byte for byte on every backend,
//! - [`executor::PassKind::Pbr`] — the material pass in the graph: it shades
//!   a camera-facing plane under one directional light (parameters validated
//!   up front), renders into any target format and supports per-target
//!   MSAA, and its output feeds sampling passes like any other resource.
//!
//! M2 (eighth increment) adds the 2D sprite suite:
//!
//! - [`sprite`] — batched 2D sprites, mirroring `bevy_sprite`'s batching
//!   and Godot's `CanvasItem.draw_texture_rect`: `Sprite2d` (destination
//!   rect, uv window, linear tint), the 1:1 viewport camera
//!   [`sprite::Camera2d`] (top-left origin, y down) whose projection the
//!   CPU mirror [`sprite::Camera2d::project`] reproduces bit for bit, and
//!   one batch → one vertex buffer → one draw call,
//! - [`executor::PassKind::Sprites2d`] — the sprite pass in the graph: it
//!   reads one source texture, draws the batch in submission order (later
//!   sprites overwrite earlier ones; no blending yet), renders any target
//!   format including per-target MSAA, and a full-viewport white-tint
//!   sprite is a pixel-exact copy of its source.
//!
//! M2 (ninth increment) opens the post-processing stack:
//!
//! - [`post`] — screen adjustments, mirroring Godot's `Environment`
//!   adjustments: brightness, contrast and saturation applied in a fixed
//!   order in linear space (Rec. 709 luma for the saturation mix), with the
//!   CPU mirror [`post::adjust`] sharing the exact operation sequence of
//!   the WGSL shader,
//! - [`executor::PassKind::Adjust`] — the adjustment pass in the graph: it
//!   reads one source texture, applies the chain over the fullscreen quad
//!   (parameters validated to [0, 8]), renders any target format including
//!   per-target MSAA, leaves alpha untouched, and identity parameters are a
//!   pixel-exact copy of the source.
//!
//! M2 (tenth increment) completes the post-processing stack — and M2 —
//! with the separable tent blur:
//!
//! - [`post`] gains the 3-tap tent blur (`BLUR_WGSL`, weights
//!   `[0.25, 0.5, 0.25]` on all four channels), with [`post::tent3`] as
//!   the bit-for-bit CPU mirror of the shader's blend,
//! - [`executor::PassKind::Blur`] — the blur pass in the graph: it reads
//!   one source texture, tents along a direction in texel units
//!   (validated non-zero, components in [-8, 8]; the executor scales it
//!   to the source's per-tap UV step), renders any target format
//!   including per-target MSAA, and two chained nodes — horizontal then
//!   vertical — form the separable blur that glows and blooms build on.
//!
//! That closes M2's milestone scope — the wgpu foundation, the render
//! graph with both executors, triangle/quad/sampling passes, MSAA,
//! windowed present, forward PBR, the 2D sprite suite and the
//! post-processing stack. M3 (the MCP control plane, PRD §16) is next.

pub mod executor;
pub mod graph;
pub mod offscreen;
pub mod pbr;
pub mod pipeline;
pub mod post;
pub mod renderer;
pub mod resources;
pub mod sprite;
pub mod surface;

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
