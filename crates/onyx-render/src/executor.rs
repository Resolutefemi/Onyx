//! Offscreen graph execution: passes in the graph's topological order,
//! single- or multi-encoder submissions, then per-target readback
//! (`docs/PRD.md` §2).
//!
//! Two executors share one planning phase ([`plan_wave`], which validates
//! every node, resolves its targets and its sampled source before any GPU
//! work) and one allocation phase:
//!
//! - [`execute_offscreen`] — the sequential path: one command encoder, all
//!   passes in topological order, a single submission,
//! - [`execute_offscreen_parallel`] — the parallel path: the graph decomposes
//!   into *waves* (see [`crate::graph::RenderGraph::execution_waves`]); every
//!   node in a wave encodes on its own thread into its own command encoder,
//!   each wave submits its command buffers in one call, and wave N+1's
//!   submission follows wave N's on the queue. Write-write conflicts and
//!   write-read pairs always cross waves (the graph edges force it), so
//!   parallel results are bit-identical to sequential ones — proven by a
//!   test.
//!
//! Within one submission wgpu inserts GPU barriers automatically from the
//! real resource usage of each pass — including the render-attachment →
//! texture-binding transition a sampling pass needs — and submissions to one
//! queue execute in order: the graph fixes ORDER, wgpu guarantees
//! synchronization.
//!
//! Targets may declare MSAA via [`ResourceSpec::samples`]: the executor then
//! creates a multisampled attachment and resolves it into the single-sample
//! readback texture on every pass. Pipelines are built per sample count, and
//! sample counts are validated against the adapter before any encoding.
//!
//! Sampling passes ([`PassKind::TexturedQuad`]) bind the node's declared
//! read resource as a texture through the quad bind group (uniform +
//! texture + sampler, `resources`): a node that writes "post" and reads
//! "scene" draws "scene" onto "post". This increment requires single-sample
//! resources on both ends of a sampling pass; an MSAA quad pipeline and
//! multi-target passes land later.
//!
//! PBR passes ([`PassKind::Pbr`], [`crate::pbr`]) shade a camera-facing
//! plane with the Cook-Torrance GGX BRDF: the shader and the CPU mirror
//! [`crate::pbr::pbr_shaded`] share the exact operation sequence (only
//! exactly-rounded IEEE f32 operations), so tests compare the GPU output
//! against the mirror byte for byte.
//!
//! The executor is format-agnostic ([`ResourceSpec::format`]): pipelines are
//! built per (sample count, format) and BGRA readbacks are swizzled back to
//! RGBA, so [`OffscreenFrame`] stays RGBA-ordered for every target. One
//! target may also be caller-provided ([`ExternalTarget`],
//! [`execute_with_external_target`]) — that is the swapchain frame for
//! windowed rendering: validated against its declaration, drawn into like
//! any other target, never allocated, sampled or read back; the caller
//! presents it.

use crate::graph::{GraphError, RenderGraph};
use crate::offscreen::{readback_texture, OffscreenFrame};
use crate::pbr;
use crate::pipeline;
use crate::renderer::{RenderError, Renderer};
use crate::resources::{self, QuadUniform, QuadUniformBuffer};

/// What a node renders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PassKind {
    /// Clear the target to a color.
    Clear([f64; 4]),
    /// Clear the target to `background`, then draw the cached triangle in
    /// `triangle` (vertex-colored, no blending).
    Triangle {
        /// Triangle RGB(A) color.
        triangle: [f64; 4],
        /// Load-op clear color behind the triangle.
        background: [f64; 4],
    },
    /// Clear the target to `background`, then draw a fullscreen quad sampling
    /// the node's declared read resource: `tint` multiplies the sampled texel
    /// in linear space, `uv_window` (offset xy, scale zw) selects the source
    /// sub-rect. Nearest sampling, full-window white-tint draws are
    /// pixel-exact copies of the source readback.
    TexturedQuad {
        /// RGBA tint multiplied over the sampled texel (linear space).
        tint: [f64; 4],
        /// Load-op clear color behind the quad.
        background: [f64; 4],
        /// UV window: offset xy, scale zw over the source texture.
        uv_window: [f64; 4],
    },
    /// A forward PBR material pass ([`crate::pbr`]): shade a fullscreen
    /// camera-facing plane (N = V = +Z) under one directional light with the
    /// Cook-Torrance GGX BRDF. The flat, camera-facing geometry makes the
    /// shaded color exactly predictable per pass — the "material chip" view —
    /// and [`crate::pbr::pbr_shaded`] mirrors the shader bit for bit.
    Pbr {
        /// Surface albedo (base color), linear, components in [0, 1].
        albedo: [f64; 3],
        /// Roughness in [0, 1] (GGX α = roughness²).
        roughness: f64,
        /// Metallic in [0, 1]: 0 = dielectric, 1 = metal (no diffuse,
        /// albedo-tinted F0).
        metallic: f64,
        /// Directional light direction, pointing at the surface (normalized
        /// by the executor; must be non-zero).
        light_dir: [f64; 3],
        /// Light color, linear, components in [0, 1].
        light_color: [f64; 3],
        /// Ambient term, linear, components in [0, 1].
        ambient: [f64; 3],
    },
}

/// A named render target the executor allocates and reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceSpec {
    /// Resource name, matching node `writes`/`reads` declarations.
    pub name: &'static str,
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
    /// MSAA sample count for this target: `1` renders directly, `> 1` draws
    /// into a multisampled attachment that resolves into the readback texture.
    pub samples: u32,
    /// Pixel format of the target. The executor is format-agnostic: pipelines
    /// are built per format, and BGRA readbacks are swizzled back to RGBA.
    pub format: wgpu::TextureFormat,
}

impl ResourceSpec {
    /// A single-sample sRGB target — the common case.
    #[must_use]
    pub const fn new(name: &'static str, width: u32, height: u32) -> Self {
        Self {
            name,
            width,
            height,
            samples: 1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        }
    }

    /// A target drawn with `samples`× MSAA before resolve.
    #[must_use]
    pub const fn with_samples(name: &'static str, width: u32, height: u32, samples: u32) -> Self {
        Self {
            name,
            width,
            height,
            samples,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        }
    }

    /// A target with an explicit format (single-sample). Use this for
    /// non-sRGB or BGRA targets — e.g. matching a swapchain format.
    #[must_use]
    pub const fn with_format(
        name: &'static str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            name,
            width,
            height,
            samples: 1,
            format,
        }
    }
}

/// A target the caller provides instead of the executor allocating it —
/// currently the swapchain frame for windowed rendering
/// ([`crate::surface::SurfaceRenderer::render_to_surface`]). The executor
/// draws into its view like any other declared target, but never allocates,
/// samples or reads it back; the caller owns acquisition and present.
pub struct ExternalTarget {
    /// The declared resource name this target stands in for.
    pub name: &'static str,
    /// The caller-provided texture (e.g. the current swapchain frame).
    pub texture: wgpu::Texture,
    /// The texture's format — must match the declared [`ResourceSpec::format`].
    pub format: wgpu::TextureFormat,
    /// Texture width — must match the declared [`ResourceSpec::width`].
    pub width: u32,
    /// Texture height — must match the declared [`ResourceSpec::height`].
    pub height: u32,
}

/// One validated node: its pass kind, the target it writes and — for
/// sampling passes — the target it reads.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PlannedPass {
    node: &'static str,
    kind: PassKind,
    target_index: usize,
    read_index: Option<usize>,
}

/// A declared target and everything the executors need to draw into it.
struct Target {
    name: &'static str,
    width: u32,
    height: u32,
    /// Single-sample texture: the resolve target when `msaa_view` is set,
    /// the render target otherwise; always the readback source and the
    /// sampling source for read declarations.
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// `Some` when `samples > 1`: the multisampled attachment views.
    msaa_view: Option<wgpu::TextureView>,
    samples: u32,
    format: wgpu::TextureFormat,
}

/// Resources shared by every pass encoding of one execution — built once on
/// the calling thread, shared across worker threads (`RenderPipeline`,
/// `BindGroupLayout` and `Sampler` are all `Send + Sync`). The quad and PBR
/// pipelines are built per needed format/(samples) up front so encoding
/// threads only look up.
struct PassEncoderContext {
    quad_pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    pbr_pipelines: Vec<((u32, wgpu::TextureFormat), wgpu::RenderPipeline)>,
    pbr_layout: wgpu::BindGroupLayout,
    quad_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl PassEncoderContext {
    /// Build the quad pipeline for every distinct `quad_formats` entry (the
    /// target formats of this execution's sampling passes) and the PBR
    /// pipeline for every distinct `pbr_keys` entry — `(samples, format)` of
    /// this execution's PBR passes.
    fn new(
        renderer: &Renderer,
        quad_formats: impl Iterator<Item = wgpu::TextureFormat>,
        pbr_keys: impl Iterator<Item = (u32, wgpu::TextureFormat)>,
    ) -> Self {
        let mut quad_pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)> = Vec::new();
        for format in quad_formats {
            if !quad_pipelines
                .iter()
                .any(|(existing, _)| *existing == format)
            {
                let built = resources::create_quad_pipeline(renderer, format);
                quad_pipelines.push((format, built));
            }
        }
        let mut pbr_pipelines: Vec<((u32, wgpu::TextureFormat), wgpu::RenderPipeline)> = Vec::new();
        for key in pbr_keys {
            if !pbr_pipelines.iter().any(|(existing, _)| *existing == key) {
                let built = pbr::create_pbr_pipeline(renderer, key.1, key.0);
                pbr_pipelines.push((key, built));
            }
        }
        Self {
            quad_pipelines,
            pbr_pipelines,
            pbr_layout: pbr::create_pbr_bind_layout(renderer),
            quad_layout: resources::create_quad_bind_layout(renderer),
            sampler: resources::create_sampler(renderer, true),
        }
    }

    /// The quad pipeline for `format` (built during [`Self::new`]).
    fn quad_pipeline(&self, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        &self
            .quad_pipelines
            .iter()
            .find(|(existing, _)| *existing == format)
            .expect("quad pipeline prebuilt for every sampling format")
            .1
    }

    /// The PBR pipeline for `(samples, format)` (built during [`Self::new`]).
    fn pbr_pipeline(&self, samples: u32, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        &self
            .pbr_pipelines
            .iter()
            .find(|(existing, _)| *existing == (samples, format))
            .expect("pbr pipeline prebuilt for every pbr (samples, format)")
            .1
    }
}

/// Validate sample counts against the adapter before any GPU work.
fn validate_sample_count(renderer: &Renderer, name: &str, samples: u32) -> Result<(), RenderError> {
    if samples <= 1 {
        return Ok(());
    }
    let supported = renderer
        .adapter()
        .get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
        .flags
        .supported_sample_counts();
    if supported.contains(&samples) {
        return Ok(());
    }
    Err(RenderError::Unsupported {
        message: format!(
            "target '{name}' requests {samples}x MSAA; adapter supports {supported:?}"
        ),
    })
}

/// Validate one node and resolve its declarations into a [`PlannedPass`]:
/// pass kind required, exactly one declared write target, and read
/// declarations that match the pass kind (`Clear`/`Triangle` read nothing,
/// `TexturedQuad` reads exactly one declared single-sample source). Runs
/// before any GPU work so both executors reject bad graphs up front.
fn plan_node(
    renderer: &Renderer,
    graph: &RenderGraph,
    resources: &[ResourceSpec],
    kind_of: &dyn Fn(&str) -> Option<PassKind>,
    node_name: &'static str,
) -> Result<PlannedPass, RenderError> {
    let Some(kind) = kind_of(node_name) else {
        return Err(RenderError::Unsupported {
            message: format!("node '{node_name}' has no pass kind"),
        });
    };
    let node = graph
        .nodes()
        .iter()
        .find(|node| node.name == node_name)
        .expect("order names come from the graph");
    let reads_expected = match kind {
        PassKind::Clear(_) | PassKind::Triangle { .. } | PassKind::Pbr { .. } => 0,
        PassKind::TexturedQuad { .. } => 1,
    };
    if node.reads.len() != reads_expected {
        return Err(RenderError::Unsupported {
            message: format!(
                "node '{}' ({}) must read exactly {} resource(s), declares {}",
                node.name,
                match kind {
                    PassKind::Clear(_) => "Clear",
                    PassKind::Triangle { .. } => "Triangle",
                    PassKind::TexturedQuad { .. } => "TexturedQuad",
                    PassKind::Pbr { .. } => "Pbr",
                },
                reads_expected,
                node.reads.len()
            ),
        });
    }
    if node.writes.len() != 1 {
        return Err(RenderError::Unsupported {
            message: format!(
                "node '{}' writes {} targets; multi-target (MRT) passes land later",
                node.name,
                node.writes.len()
            ),
        });
    }
    let written = node.writes[0];
    let target_index = resources
        .iter()
        .position(|spec| spec.name == written)
        .ok_or_else(|| RenderError::Unsupported {
            message: format!(
                "node '{}' writes undeclared resource '{written}'",
                node.name
            ),
        })?;
    validate_sample_count(renderer, written, resources[target_index].samples)?;
    if let PassKind::Pbr {
        albedo,
        roughness,
        metallic,
        light_dir,
        light_color,
        ambient,
    } = kind
    {
        pbr::validate_pbr_params(
            node_name,
            albedo,
            roughness,
            metallic,
            light_dir,
            light_color,
            ambient,
        )?;
    }

    // A sampling pass reads a texture: the source must be a declared
    // single-sample target (multisampled textures are not sampleable by this
    // pipeline), and this increment draws the quad only into single-sample
    // targets.
    let read_index = match kind {
        PassKind::TexturedQuad { .. } => {
            let source = node.reads[0];
            let index = resources
                .iter()
                .position(|spec| spec.name == source)
                .ok_or(RenderError::Unsupported {
                    message: format!("node '{}' reads undeclared resource '{source}'", node.name),
                })?;
            let source_samples = resources[index].samples;
            if source_samples != 1 || resources[target_index].samples != 1 {
                return Err(RenderError::Unsupported {
                    message: format!(
                        "sampling pass '{}' requires single-sample resources: \
                         source '{source}' is {source_samples}x, \
                         target '{written}' is {}x",
                        node.name, resources[target_index].samples
                    ),
                });
            }
            Some(index)
        }
        _ => None,
    };
    Ok(PlannedPass {
        node: node_name,
        kind,
        target_index,
        read_index,
    })
}

/// Validate one wave of nodes (in wave order) into planned passes.
fn plan_wave(
    renderer: &Renderer,
    graph: &RenderGraph,
    resources: &[ResourceSpec],
    kind_of: &dyn Fn(&str) -> Option<PassKind>,
    wave: &[&'static str],
) -> Result<Vec<PlannedPass>, RenderError> {
    wave.iter()
        .map(|&node_name| plan_node(renderer, graph, resources, kind_of, node_name))
        .collect()
}

/// Allocate every declared resource (or splice in the caller-provided
/// [`ExternalTarget`] in its declared position). Targets with `samples > 1`
/// get a multisampled attachment alongside the single-sample resolve texture;
/// the single-sample texture carries `TEXTURE_BINDING` so sampling passes can
/// bind it.
fn allocate_targets(
    renderer: &Renderer,
    resources: &[ResourceSpec],
    external: Option<&ExternalTarget>,
) -> Vec<Target> {
    let device = &renderer.device;
    resources
        .iter()
        .map(|spec| {
            if let Some(external) = external.filter(|external| external.name == spec.name) {
                let view = external.texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("onyx-graph-external-view"),
                    ..wgpu::TextureViewDescriptor::default()
                });
                return Target {
                    name: spec.name,
                    width: external.width,
                    height: external.height,
                    texture: external.texture.clone(),
                    view,
                    msaa_view: None,
                    samples: 1,
                    format: external.format,
                };
            }
            let extent = wgpu::Extent3d {
                width: spec.width,
                height: spec.height,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("onyx-graph-target"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: spec.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let msaa_view = if spec.samples > 1 {
                let msaa = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("onyx-graph-msaa"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: spec.samples,
                    dimension: wgpu::TextureDimension::D2,
                    format: spec.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                Some(msaa.create_view(&wgpu::TextureViewDescriptor::default()))
            } else {
                None
            };
            Target {
                name: spec.name,
                width: spec.width,
                height: spec.height,
                texture,
                view,
                msaa_view,
                samples: spec.samples,
                format: spec.format,
            }
        })
        .collect()
}

/// Validate the external target against its declared [`ResourceSpec`] before
/// any GPU work: it must be declared, single-sample, format- and
/// size-matched, and must not be sampled (swapchain textures carry only the
/// render-attachment usage).
fn validate_external(
    resources: &[ResourceSpec],
    planned: &[PlannedPass],
    external: &ExternalTarget,
) -> Result<(), RenderError> {
    let Some(spec) = resources.iter().find(|spec| spec.name == external.name) else {
        return Err(RenderError::Unsupported {
            message: format!(
                "external target '{}' is not a declared resource",
                external.name
            ),
        });
    };
    if spec.samples != 1 {
        return Err(RenderError::Unsupported {
            message: format!(
                "external target '{}' must be single-sample, declared {}x MSAA",
                external.name, spec.samples
            ),
        });
    }
    if spec.format != external.format {
        return Err(RenderError::Unsupported {
            message: format!(
                "external target '{}' format mismatch: declared {:?}, provided {:?}",
                external.name, spec.format, external.format
            ),
        });
    }
    if spec.width != external.width || spec.height != external.height {
        return Err(RenderError::Unsupported {
            message: format!(
                "external target '{}' size mismatch: declared {}x{}, provided {}x{}",
                external.name, spec.width, spec.height, external.width, external.height
            ),
        });
    }
    if planned
        .iter()
        .any(|pass| pass.read_index == Some(spec_index(resources, external.name)))
    {
        return Err(RenderError::Unsupported {
            message: format!(
                "external target '{}' cannot be sampled; blit through an \
                 intermediate target instead",
                external.name
            ),
        });
    }
    Ok(())
}

/// The declared index of resource `name` (validated callers only).
fn spec_index(resources: &[ResourceSpec], name: &str) -> usize {
    resources
        .iter()
        .position(|spec| spec.name == name)
        .expect("validated callers ensure the resource is declared")
}

/// The triangle pipeline for `(samples, format)` — the plain pipeline at 1×,
/// the MSAA variant otherwise. Cached per sample count and format.
fn triangle_pipeline_for(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
    samples: u32,
    cache: &mut Vec<((u32, wgpu::TextureFormat), wgpu::RenderPipeline)>,
) -> wgpu::RenderPipeline {
    if let Some((_, existing)) = cache
        .iter()
        .find(|((count, cached_format), _)| *count == samples && *cached_format == format)
    {
        return existing.clone();
    }
    let built = if samples <= 1 {
        pipeline::create_triangle_pipeline(renderer, format)
    } else {
        resources::create_triangle_pipeline_msaa(renderer, format, samples)
    };
    cache.push(((samples, format), built.clone()));
    built
}

/// Encode one planned pass into `encoder`, writing `targets[pass.
/// target_index]` and — for sampling passes — binding
/// `targets[pass.read_index]` as the quad texture.
fn encode_pass(
    renderer: &Renderer,
    encoder: &mut wgpu::CommandEncoder,
    pass: &PlannedPass,
    targets: &[Target],
    context: &PassEncoderContext,
    triangle_pipelines: &mut Vec<((u32, wgpu::TextureFormat), wgpu::RenderPipeline)>,
) {
    let target = &targets[pass.target_index];
    let attachment_view = target.msaa_view.as_ref().unwrap_or(&target.view);
    let resolve_target = target.msaa_view.as_ref().map(|_| &target.view);
    match pass.kind {
        PassKind::Clear(color) => {
            let attachment = wgpu::RenderPassColorAttachment {
                view: attachment_view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: color[0],
                        g: color[1],
                        b: color[2],
                        a: color[3],
                    }),
                    store: wgpu::StoreOp::Store,
                },
            };
            let pass_handle = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("onyx-graph-clear"),
                color_attachments: &[Some(attachment)],
                ..wgpu::RenderPassDescriptor::default()
            });
            // Drop the pass handle: the clear happens on store.
            drop(pass_handle);
        }
        PassKind::Triangle {
            triangle,
            background,
        } => {
            let vertex_buffer = pipeline::triangle_vertex_buffer(renderer, triangle);
            let attachment = wgpu::RenderPassColorAttachment {
                view: attachment_view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: background[0],
                        g: background[1],
                        b: background[2],
                        a: background[3],
                    }),
                    store: wgpu::StoreOp::Store,
                },
            };
            let mut pass_handle = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("onyx-graph-triangle"),
                color_attachments: &[Some(attachment)],
                ..wgpu::RenderPassDescriptor::default()
            });
            let pipeline =
                triangle_pipeline_for(renderer, target.format, target.samples, triangle_pipelines);
            pass_handle.set_pipeline(&pipeline);
            pass_handle.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass_handle.draw(0..3, 0..1);
        }
        PassKind::TexturedQuad {
            tint,
            background,
            uv_window,
        } => {
            // Each sampling pass owns its uniform buffer and bind group —
            // safe on worker threads: Device is Send + Sync and the queue
            // write behind create_buffer_init is ordered before the next
            // submission.
            let uniform = QuadUniform {
                tint: [
                    tint[0] as f32,
                    tint[1] as f32,
                    tint[2] as f32,
                    tint[3] as f32,
                ],
                uv_window: [
                    uv_window[0] as f32,
                    uv_window[1] as f32,
                    uv_window[2] as f32,
                    uv_window[3] as f32,
                ],
            };
            let uniform_buffer = QuadUniformBuffer::new(renderer, &uniform);
            let read_index = pass.read_index.expect("TexturedQuad plans a read");
            let source_view = &targets[read_index].view;
            let bind_group = resources::create_quad_bind_group(
                renderer,
                &context.quad_layout,
                &uniform_buffer,
                source_view,
                &context.sampler,
            );
            let attachment = wgpu::RenderPassColorAttachment {
                view: attachment_view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: background[0],
                        g: background[1],
                        b: background[2],
                        a: background[3],
                    }),
                    store: wgpu::StoreOp::Store,
                },
            };
            let mut pass_handle = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("onyx-graph-textured-quad"),
                color_attachments: &[Some(attachment)],
                ..wgpu::RenderPassDescriptor::default()
            });
            pass_handle.set_pipeline(context.quad_pipeline(target.format));
            pass_handle.set_bind_group(0, &bind_group, &[]);
            pass_handle.draw(0..6, 0..1);
        }
        PassKind::Pbr {
            albedo,
            roughness,
            metallic,
            light_dir,
            light_color,
            ambient,
        } => {
            // Fixed camera-facing geometry — every dot product is uniform
            // (see `PbrParams::new`); each pass owns its uniform buffer and
            // bind group, like the sampling passes.
            let params =
                pbr::PbrParams::new(albedo, roughness, metallic, light_dir, light_color, ambient);
            let uniform_buffer = pbr::PbrUniformBuffer::new(renderer, &params);
            let bind_group =
                pbr::create_pbr_bind_group(renderer, &context.pbr_layout, &uniform_buffer);
            let attachment = wgpu::RenderPassColorAttachment {
                view: attachment_view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            };
            let mut pass_handle = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("onyx-graph-pbr"),
                color_attachments: &[Some(attachment)],
                ..wgpu::RenderPassDescriptor::default()
            });
            pass_handle.set_pipeline(context.pbr_pipeline(target.samples, target.format));
            pass_handle.set_bind_group(0, &bind_group, &[]);
            pass_handle.draw(0..6, 0..1);
        }
    }
}

/// `true` for BGRA storage — readback bytes need an R/B channel swap so
/// [`OffscreenFrame`] stays RGBA-ordered regardless of target format.
fn is_bgra(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    )
}

/// Read back every target in resource declaration order (except
/// `skip`, the external target the caller presents themselves). BGRA
/// targets are swizzled back to RGBA byte order.
fn readback_targets(
    renderer: &Renderer,
    targets: &[Target],
    skip: Option<&str>,
) -> Result<Vec<(&'static str, OffscreenFrame)>, RenderError> {
    let mut frames = Vec::with_capacity(targets.len());
    for target in targets {
        if skip == Some(target.name) {
            continue;
        }
        let mut pixels = readback_texture(renderer, &target.texture, target.width, target.height)?;
        if is_bgra(target.format) {
            for channel in pixels.chunks_exact_mut(4) {
                channel.swap(0, 2);
            }
        }
        frames.push((
            target.name,
            OffscreenFrame {
                width: target.width,
                height: target.height,
                pixels,
            },
        ));
    }
    Ok(frames)
}

/// Execute `graph` offscreen sequentially: allocate every declared resource,
/// encode all passes into one encoder in topological order, submit once, and
/// read back every target (in resource declaration order).
///
/// # Errors
/// [`RenderError::Graph`] for graph scheduling failures, [`RenderError::
/// Unsupported`] for node declarations or sample counts this executor cannot
/// yet encode, [`RenderError::Poll`] / [`RenderError::Map`] on readback
/// failure.
pub fn execute_offscreen(
    renderer: &Renderer,
    graph: &RenderGraph,
    resources: &[ResourceSpec],
    kind_of: &dyn Fn(&str) -> Option<PassKind>,
) -> Result<Vec<(&'static str, OffscreenFrame)>, RenderError> {
    execute_sequential(renderer, graph, resources, kind_of, None)
}

/// Execute `graph` sequentially with one caller-provided target — the
/// swapchain frame for windowed rendering. The external target is validated
/// against its declaration, drawn into like any other target, but neither
/// allocated, sampled nor read back; the caller presents it.
///
/// # Errors
/// Same as [`execute_offscreen`], plus [`RenderError::Unsupported`] when the
/// external target mismatches its declaration or a sampling pass tries to
/// read it.
pub fn execute_with_external_target(
    renderer: &Renderer,
    graph: &RenderGraph,
    resources: &[ResourceSpec],
    kind_of: &dyn Fn(&str) -> Option<PassKind>,
    external: &ExternalTarget,
) -> Result<Vec<(&'static str, OffscreenFrame)>, RenderError> {
    execute_sequential(renderer, graph, resources, kind_of, Some(external))
}

/// The sequential body shared by [`execute_offscreen`] and
/// [`execute_with_external_target`].
fn execute_sequential(
    renderer: &Renderer,
    graph: &RenderGraph,
    resources: &[ResourceSpec],
    kind_of: &dyn Fn(&str) -> Option<PassKind>,
    external: Option<&ExternalTarget>,
) -> Result<Vec<(&'static str, OffscreenFrame)>, RenderError> {
    let waves = graph
        .execution_waves()
        .map_err(|error: GraphError| RenderError::Graph {
            message: error.to_string(),
        })?;
    let mut planned = Vec::new();
    for wave in &waves {
        planned.extend(plan_wave(renderer, graph, resources, kind_of, wave)?);
    }
    if let Some(external) = external {
        validate_external(resources, &planned, external)?;
    }
    let targets = allocate_targets(renderer, resources, external);
    let quad_formats = planned
        .iter()
        .filter(|pass| matches!(pass.kind, PassKind::TexturedQuad { .. }))
        .map(|pass| targets[pass.target_index].format);
    let pbr_keys = planned
        .iter()
        .filter(|pass| matches!(pass.kind, PassKind::Pbr { .. }))
        .map(|pass| {
            let target = &targets[pass.target_index];
            (target.samples, target.format)
        });
    let context = PassEncoderContext::new(renderer, quad_formats, pbr_keys);
    let mut triangle_pipelines = Vec::new();
    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("onyx-graph-executor"),
        });
    for pass in &planned {
        encode_pass(
            renderer,
            &mut encoder,
            pass,
            &targets,
            &context,
            &mut triangle_pipelines,
        );
    }
    renderer.queue.submit([encoder.finish()]);
    readback_targets(renderer, &targets, external.map(|external| external.name))
}

/// Execute `graph` offscreen in parallel: waves of independent nodes encode
/// concurrently (one thread and one command encoder per node group), each
/// wave submits its command buffers in one call, and later waves follow on
/// the queue — submission order makes cross-wave dependencies safe, so a
/// sampling pass sees its source written by earlier waves.
///
/// `workers` caps the threads per wave (clamped to at least 1). Results are
/// bit-identical to [`execute_offscreen`]: the graph's edges force
/// write-write conflicts into separate waves, so wave boundaries are the only
/// synchronization the executor needs.
///
/// # Errors
/// Same as [`execute_offscreen`].
pub fn execute_offscreen_parallel(
    renderer: &Renderer,
    graph: &RenderGraph,
    resources: &[ResourceSpec],
    kind_of: &dyn Fn(&str) -> Option<PassKind>,
    workers: usize,
) -> Result<Vec<(&'static str, OffscreenFrame)>, RenderError> {
    let waves = graph
        .execution_waves()
        .map_err(|error: GraphError| RenderError::Graph {
            message: error.to_string(),
        })?;
    // Plan every pass up front so the worker threads only encode.
    let planned: Vec<Vec<PlannedPass>> = waves
        .iter()
        .map(|wave| plan_wave(renderer, graph, resources, kind_of, wave))
        .collect::<Result<Vec<Vec<PlannedPass>>, RenderError>>()?;

    let targets = allocate_targets(renderer, resources, None);
    let quad_formats = planned
        .iter()
        .flatten()
        .filter(|pass| matches!(pass.kind, PassKind::TexturedQuad { .. }))
        .map(|pass| targets[pass.target_index].format);
    let pbr_keys = planned
        .iter()
        .flatten()
        .filter(|pass| matches!(pass.kind, PassKind::Pbr { .. }))
        .map(|pass| {
            let target = &targets[pass.target_index];
            (target.samples, target.format)
        });
    let context = PassEncoderContext::new(renderer, quad_formats, pbr_keys);
    let workers = workers.max(1);

    for wave in &planned {
        // Prebuild the triangle pipelines this wave needs —
        // RenderPipeline is Send + Sync, so threads can share them.
        let mut triangle_pipelines: Vec<((u32, wgpu::TextureFormat), wgpu::RenderPipeline)> =
            Vec::new();
        for pass in wave {
            if matches!(pass.kind, PassKind::Triangle { .. }) {
                let target = &targets[pass.target_index];
                let key = (target.samples, target.format);
                if !triangle_pipelines.iter().any(|(cached, _)| *cached == key) {
                    let built = if key.0 <= 1 {
                        pipeline::create_triangle_pipeline(renderer, key.1)
                    } else {
                        resources::create_triangle_pipeline_msaa(renderer, key.1, key.0)
                    };
                    triangle_pipelines.push((key, built));
                }
            }
        }
        let chunk_size = wave.len().div_ceil(workers);
        let targets_ref = &targets;
        let context_ref = &context;
        let triangle_pipelines_ref = &triangle_pipelines;
        let command_buffers: Vec<wgpu::CommandBuffer> = std::thread::scope(|scope| {
            wave.chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let renderer = &*renderer;
                        let mut encoder = renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("onyx-graph-wave-encoder"),
                            },
                        );
                        let mut pipeline_cache = triangle_pipelines_ref.clone();
                        for pass in chunk {
                            encode_pass(
                                renderer,
                                &mut encoder,
                                pass,
                                targets_ref,
                                context_ref,
                                &mut pipeline_cache,
                            );
                        }
                        encoder.finish()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("wave encoder thread"))
                .collect()
        });
        renderer.queue.submit(command_buffers);
    }

    readback_targets(renderer, &targets, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RenderGraphNode;
    use crate::offscreen::linear_to_srgb8;
    use crate::resources::srgb8_to_linear;
    use crate::RendererConfig;

    fn node(
        name: &'static str,
        reads: &'static [&'static str],
        writes: &'static [&'static str],
    ) -> RenderGraphNode {
        RenderGraphNode {
            name,
            reads,
            writes,
        }
    }

    fn graph(nodes: &[RenderGraphNode]) -> RenderGraph {
        let mut graph = RenderGraph::new();
        for node in nodes {
            graph
                .try_add_node(node.clone())
                .expect("demo graph is valid");
        }
        graph
    }

    #[test]
    fn writer_conflicts_run_in_declaration_order_on_one_target() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let graph = &graph(&[node("triangle", &[], &["t"]), node("clear", &[], &["t"])]);
                let resources = [ResourceSpec::new("t", 16, 16)];
                let kinds = |name: &str| match name {
                    "triangle" => Some(PassKind::Triangle {
                        triangle: [1.0, 0.5, 0.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                    }),
                    "clear" => Some(PassKind::Clear([0.2, 0.4, 0.8, 1.0])),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                assert_eq!(frames.len(), 1);
                let frame = &frames[0].1;
                // 'clear' is declared second, so it wins the target.
                assert_eq!(
                    &frame.pixels[0..4],
                    &[
                        linear_to_srgb8(0.2),
                        linear_to_srgb8(0.4),
                        linear_to_srgb8(0.8),
                        255
                    ]
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn reversed_declaration_flips_the_winner() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let graph = &graph(&[node("clear", &[], &["t"]), node("triangle", &[], &["t"])]);
                let resources = [ResourceSpec::new("t", 16, 16)];
                let kinds = |name: &str| match name {
                    "triangle" => Some(PassKind::Triangle {
                        triangle: [1.0, 0.5, 0.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                    }),
                    "clear" => Some(PassKind::Clear([0.2, 0.4, 0.8, 1.0])),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                let frame = &frames[0].1;
                let center = (8 * 16 + 8) * 4;
                assert_eq!(
                    &frame.pixels[center..center + 4],
                    &[
                        linear_to_srgb8(1.0),
                        linear_to_srgb8(0.5),
                        linear_to_srgb8(0.0),
                        255
                    ],
                    "triangle runs last and owns the center pixel"
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn missing_pass_kind_is_reported_before_any_gpu_work() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let graph = &graph(&[node("mystery", &[], &["t"])]);
                let resources = [ResourceSpec::new("t", 4, 4)];
                let kinds = |_: &str| None;
                let error =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect_err("unknown");
                assert!(
                    error.to_string().contains("mystery"),
                    "error must name the node: {error}"
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn msaa_target_resolves_inside_the_graph() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let graph = &graph(&[node("triangle", &[], &["t"])]);
                let resources = [ResourceSpec::with_samples("t", 16, 16, 4)];
                let kinds = |name: &str| match name {
                    "triangle" => Some(PassKind::Triangle {
                        triangle: [1.0, 1.0, 1.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                    }),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                let frame = &frames[0].1;
                assert_eq!(frame.pixels.len(), 16 * 16 * 4);
                let center = (8 * 16 + 8) * 4;
                assert_eq!(
                    &frame.pixels[center..center + 4],
                    &[255, 255, 255, 255],
                    "deep interior resolves to exactly the triangle color"
                );
                assert_eq!(
                    &frame.pixels[0..4],
                    &[0, 0, 0, 255],
                    "far outside stays exactly the background"
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn unsupported_sample_count_is_rejected_before_encoding() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // 3x is not a valid sample count on any adapter.
                let graph = &graph(&[node("clear", &[], &["t"])]);
                let resources = [ResourceSpec::with_samples("t", 4, 4, 3)];
                let kinds = |name: &str| match name {
                    "clear" => Some(PassKind::Clear([0.0, 0.0, 0.0, 1.0])),
                    _ => None,
                };
                let error =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect_err("3x msaa");
                assert!(
                    error.to_string().contains("MSAA"),
                    "error must mention MSAA support: {error}"
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn sampling_pass_round_trips_source_pixels_exactly() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // 'draw' renders the triangle into "scene"; 'blit' reads
                // "scene" and draws it onto "post" (white tint, full window).
                let graph = &graph(&[
                    node("draw", &[], &["scene"]),
                    node("blit", &["scene"], &["post"]),
                ]);
                let resources = [
                    ResourceSpec::new("scene", 16, 16),
                    ResourceSpec::new("post", 16, 16),
                ];
                let kinds = |name: &str| match name {
                    "draw" => Some(PassKind::Triangle {
                        triangle: [1.0, 0.5, 0.0, 1.0],
                        background: [0.05, 0.1, 0.15, 1.0],
                    }),
                    "blit" => Some(PassKind::TexturedQuad {
                        tint: [1.0, 1.0, 1.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                        uv_window: [0.0, 0.0, 1.0, 1.0],
                    }),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                assert_eq!(frames.len(), 2);
                assert_eq!(frames[0].0, "scene");
                assert_eq!(frames[1].0, "post");
                assert_eq!(
                    frames[1].1.pixels, frames[0].1.pixels,
                    "nearest + full window + white tint is a pixel-exact copy"
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn sampling_quad_uv_window_selects_source_region() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // 'paint' fills "scene" with four quadrant colors as 8×8
                // blocks (via clears won by declaration order), then 'blit'
                // draws the bottom-right quadrant stretched over "post".
                let graph = &graph(&[
                    node("paint", &[], &["scene"]),
                    node("blit", &["scene"], &["post"]),
                ]);
                let resources = [
                    ResourceSpec::new("scene", 16, 16),
                    ResourceSpec::new("post", 16, 16),
                ];
                let kinds = |name: &str| match name {
                    // The source content doesn't matter: the assertion below
                    // checks the uv mapping against whatever "scene" holds
                    // (a full-screen triangle over black here).
                    "paint" => Some(PassKind::Triangle {
                        triangle: [1.0, 1.0, 1.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                    }),
                    "blit" => Some(PassKind::TexturedQuad {
                        tint: [1.0, 1.0, 1.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                        uv_window: [0.5, 0.5, 0.5, 0.5],
                    }),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                let scene = &frames[0].1;
                let post = &frames[1].1;
                // Pixel (x, y) of "post" samples texel (8 + x/2, 8 + y/2) of
                // "scene" — the bottom-right quadrant stretched to full size.
                for y in 0..16u32 {
                    for x in 0..16u32 {
                        let src = ((8 + y / 2) * 16 + 8 + x / 2) as usize * 4;
                        let dst = ((y * 16 + x) * 4) as usize;
                        assert_eq!(
                            &post.pixels[dst..dst + 4],
                            &scene.pixels[src..src + 4],
                            "pixel ({x}, {y}) must sample the quadrant texel"
                        );
                    }
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn sampling_quad_tint_multiplies_in_linear_space() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // A solid "scene" (full-screen triangle over a clear of the
                // same color), tinted non-trivially by 'blit'.
                let graph = &graph(&[
                    node("paint", &[], &["scene"]),
                    node("blit", &["scene"], &["post"]),
                ]);
                let resources = [
                    ResourceSpec::new("scene", 8, 8),
                    ResourceSpec::new("post", 8, 8),
                ];
                let solid = [0.9, 0.6, 0.3, 1.0];
                let kinds = |name: &str| match name {
                    "paint" => Some(PassKind::Clear(solid)),
                    "blit" => Some(PassKind::TexturedQuad {
                        tint: [1.0, 0.5, 0.25, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                        uv_window: [0.0, 0.0, 1.0, 1.0],
                    }),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                let source_bytes: Vec<u8> = frames[0]
                    .1
                    .pixels
                    .chunks_exact(4)
                    .flat_map(|p| [p[0], p[1], p[2], p[3]])
                    .collect();
                let expected = [
                    linear_to_srgb8(srgb8_to_linear(source_bytes[0])),
                    linear_to_srgb8(srgb8_to_linear(source_bytes[1]) * 0.5),
                    linear_to_srgb8(srgb8_to_linear(source_bytes[2]) * 0.25),
                    source_bytes[3],
                ];
                for chunk in frames[1].1.pixels.chunks_exact(4) {
                    // ×1.0 and alpha are bit-exact; tinted channels allow
                    // ±1 LSB (GPU encodes in f32, this model in f64).
                    for k in 0..4 {
                        let tolerance = if k == 1 || k == 2 { 1 } else { 0 };
                        assert!(
                            (i32::from(chunk[k]) - i32::from(expected[k])).abs() <= tolerance,
                            "tint applies in linear space: pixel {chunk:?} vs {expected:?}"
                        );
                    }
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn sampling_rejects_multisampled_source_or_target() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let kinds = |name: &str| match name {
                    "paint" => Some(PassKind::Clear([0.0, 0.0, 0.0, 1.0])),
                    "blit" => Some(PassKind::TexturedQuad {
                        tint: [1.0, 1.0, 1.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                        uv_window: [0.0, 0.0, 1.0, 1.0],
                    }),
                    _ => None,
                };
                // MSAA source.
                let graph = &graph(&[
                    node("paint", &[], &["scene"]),
                    node("blit", &["scene"], &["post"]),
                ]);
                let resources = [
                    ResourceSpec::with_samples("scene", 16, 16, 4),
                    ResourceSpec::new("post", 16, 16),
                ];
                let error =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect_err("source");
                assert!(
                    error.to_string().contains("single-sample"),
                    "source error must explain the constraint: {error}"
                );
                // MSAA target.
                let resources = [
                    ResourceSpec::new("scene", 16, 16),
                    ResourceSpec::with_samples("post", 16, 16, 4),
                ];
                let error =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect_err("target");
                assert!(
                    error.to_string().contains("single-sample"),
                    "target error must explain the constraint: {error}"
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn sampling_with_wrong_read_count_is_rejected() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // A Clear that reads (should read nothing).
                let graph = &graph(&[node("bad", &["scene"], &["post"])]);
                let resources = [
                    ResourceSpec::new("scene", 8, 8),
                    ResourceSpec::new("post", 8, 8),
                ];
                let kinds = |name: &str| match name {
                    "bad" => Some(PassKind::Clear([0.0, 0.0, 0.0, 1.0])),
                    _ => None,
                };
                let error =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect_err("reads");
                assert!(
                    error.to_string().contains("must read"),
                    "error must explain the read contract: {error}"
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    /// A mixed graph: three independent targets, a conflicting-writer pair
    /// (`scene_clear` then `clear_over` on "scene" — declaration order must
    /// survive parallel execution), a 4× MSAA target, and a sampling node
    /// two waves after its producer.
    fn mixed_graph() -> (
        RenderGraph,
        [ResourceSpec; 5],
        impl Fn(&str) -> Option<PassKind>,
    ) {
        let graph = graph(&[
            node("scene_clear", &[], &["scene"]),
            node("tri_a", &[], &["a"]),
            node("tri_b", &[], &["b"]),
            node("clear_over", &[], &["scene"]),
            node("clear_c", &[], &["c"]),
            node("blit", &["scene"], &["d"]),
        ]);
        let resources = [
            ResourceSpec::new("scene", 16, 16),
            ResourceSpec::with_samples("a", 16, 16, 4),
            ResourceSpec::new("b", 16, 16),
            ResourceSpec::new("c", 16, 16),
            ResourceSpec::new("d", 16, 16),
        ];
        let kinds = move |name: &str| match name {
            "scene_clear" => Some(PassKind::Clear([0.1, 0.2, 0.3, 1.0])),
            "clear_over" => Some(PassKind::Clear([0.9, 0.8, 0.7, 1.0])),
            "clear_c" => Some(PassKind::Clear([0.0, 1.0, 0.0, 1.0])),
            "tri_a" => Some(PassKind::Triangle {
                triangle: [1.0, 0.5, 0.0, 1.0],
                background: [0.0, 0.0, 0.0, 1.0],
            }),
            "tri_b" => Some(PassKind::Triangle {
                triangle: [0.2, 0.4, 1.0, 1.0],
                background: [0.5, 0.0, 0.5, 1.0],
            }),
            "blit" => Some(PassKind::TexturedQuad {
                tint: [1.0, 1.0, 1.0, 1.0],
                background: [0.0, 0.0, 0.0, 1.0],
                uv_window: [0.0, 0.0, 1.0, 1.0],
            }),
            _ => None,
        };
        (graph, resources, kinds)
    }

    #[test]
    fn parallel_matches_sequential_byte_for_byte() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let (graph, resources, kinds) = mixed_graph();
                let sequential =
                    execute_offscreen(&renderer, &graph, &resources, &kinds).expect("sequential");
                let parallel = execute_offscreen_parallel(&renderer, &graph, &resources, &kinds, 4)
                    .expect("parallel");
                assert_eq!(sequential.len(), parallel.len());
                for ((s_name, s_frame), (p_name, p_frame)) in sequential.iter().zip(&parallel) {
                    assert_eq!(s_name, p_name);
                    assert_eq!(
                        s_frame.pixels, p_frame.pixels,
                        "target '{s_name}' must render identically in both executors"
                    );
                }
                // The conflicting-writer pair ordered 'clear_over' last on
                // "scene"; parallel execution must preserve that.
                let scene = &parallel[0].1;
                assert_eq!(
                    &scene.pixels[0..4],
                    &[
                        linear_to_srgb8(0.9),
                        linear_to_srgb8(0.8),
                        linear_to_srgb8(0.7),
                        255
                    ]
                );
                // Target "a" went through the 4× MSAA path: center exact.
                let center = (8 * 16 + 8) * 4;
                assert_eq!(
                    &parallel[1].1.pixels[center..center + 4],
                    &[
                        linear_to_srgb8(1.0),
                        linear_to_srgb8(0.5),
                        linear_to_srgb8(0.0),
                        255
                    ]
                );
                // "d" is the sampling copy of "scene" — same bytes in the
                // parallel run, proving cross-wave read-after-write.
                assert_eq!(parallel[4].1.pixels, parallel[0].1.pixels);
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn parallel_stress_many_independent_targets() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                const STRESS_NODES: [&str; 8] = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"];
                const STRESS_TARGETS: [&str; 8] = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"];
                const STRESS_WRITES: [&[&str]; 8] = [
                    &["t0"],
                    &["t1"],
                    &["t2"],
                    &["t3"],
                    &["t4"],
                    &["t5"],
                    &["t6"],
                    &["t7"],
                ];
                let mut graph = RenderGraph::new();
                for index in 0..8 {
                    graph
                        .try_add_node(node(STRESS_NODES[index], &[], STRESS_WRITES[index]))
                        .expect("stress graph is valid");
                }
                let resources: Vec<ResourceSpec> = STRESS_TARGETS
                    .iter()
                    .map(|&name| ResourceSpec::new(name, 8, 8))
                    .collect();
                let kinds = |name: &str| {
                    // kind_of receives NODE names; node n_i writes target t_i.
                    STRESS_NODES
                        .iter()
                        .position(|node_name| *node_name == name)
                        .map(|index| {
                            let channel = 0.125 * index as f64 + 0.05;
                            PassKind::Clear([channel, 1.0 - channel, channel * 0.5, 1.0])
                        })
                };
                let frames = execute_offscreen_parallel(&renderer, &graph, &resources, &kinds, 4)
                    .expect("parallel stress");
                assert_eq!(frames.len(), 8);
                for (index, (_, frame)) in frames.iter().enumerate() {
                    let channel = 0.125 * index as f64 + 0.05;
                    let expected = [
                        linear_to_srgb8(channel),
                        linear_to_srgb8(1.0 - channel),
                        linear_to_srgb8(channel * 0.5),
                        255,
                    ];
                    for chunk in frame.pixels.chunks_exact(4) {
                        assert_eq!(chunk, &expected, "target {index} must match its color");
                    }
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    /// Byte value of linear `value` under the `*Unorm` (non-sRGB) encoding:
    /// the GPU maps [0, 1] linearly onto [0, 255].
    fn linear8(value: f64) -> u8 {
        (value * 255.0).round().clamp(0.0, 255.0) as u8
    }

    #[test]
    fn linear_target_uses_unorm_encoding() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // A clear on a non-sRGB target must store the linear value
                // directly (round(f * 255)), NOT through the sRGB curve.
                let graph = &graph(&[node("clear", &[], &["t"])]);
                let resources = [ResourceSpec::with_format(
                    "t",
                    8,
                    8,
                    wgpu::TextureFormat::Rgba8Unorm,
                )];
                let kinds = |name: &str| match name {
                    "clear" => Some(PassKind::Clear([0.5, 0.25, 1.0, 1.0])),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                let expected = [linear8(0.5), linear8(0.25), linear8(1.0), 255];
                for chunk in frames[0].1.pixels.chunks_exact(4) {
                    assert_eq!(chunk, &expected, "Unorm encoding must be linear");
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn bgra_readback_swizzles_to_rgba() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // BGRA stores (b, g, r, a); the readback must restore RGBA
                // order so OffscreenFrame stays RGBA-ordered.
                let graph = &graph(&[node("clear", &[], &["t"])]);
                let resources = [ResourceSpec::with_format(
                    "t",
                    8,
                    8,
                    wgpu::TextureFormat::Bgra8UnormSrgb,
                )];
                let kinds = |name: &str| match name {
                    "clear" => Some(PassKind::Clear([1.0, 0.0, 0.5, 1.0])),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                let expected = [
                    linear_to_srgb8(1.0),
                    linear_to_srgb8(0.0),
                    linear_to_srgb8(0.5),
                    255,
                ];
                for chunk in frames[0].1.pixels.chunks_exact(4) {
                    assert_eq!(chunk, &expected, "readback must be RGBA-ordered");
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn mixed_format_graph_matches_parallel() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let graph = &graph(&[
                    node("srgb_clear", &[], &["srgb"]),
                    node("linear_tri", &[], &["linear"]),
                    node("bgra_clear", &[], &["bgra"]),
                ]);
                let resources = [
                    ResourceSpec::new("srgb", 8, 8),
                    ResourceSpec::with_format("linear", 8, 8, wgpu::TextureFormat::Rgba8Unorm),
                    ResourceSpec::with_format("bgra", 8, 8, wgpu::TextureFormat::Bgra8UnormSrgb),
                ];
                let kinds = |name: &str| match name {
                    "srgb_clear" => Some(PassKind::Clear([0.1, 0.2, 0.3, 1.0])),
                    "linear_tri" => Some(PassKind::Triangle {
                        triangle: [1.0, 0.5, 0.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                    }),
                    "bgra_clear" => Some(PassKind::Clear([0.4, 0.6, 0.8, 1.0])),
                    _ => None,
                };
                let sequential =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("sequential");
                let parallel = execute_offscreen_parallel(&renderer, graph, &resources, &kinds, 3)
                    .expect("parallel");
                for ((s_name, s_frame), (p_name, p_frame)) in sequential.iter().zip(&parallel) {
                    assert_eq!(s_name, p_name);
                    assert_eq!(s_frame.pixels, p_frame.pixels, "target '{s_name}'");
                }
                // The linear triangle target uses the Unorm contract.
                let center = (4 * 8 + 4) * 4;
                assert_eq!(
                    &parallel[1].1.pixels[center..center + 4],
                    &[linear8(1.0), linear8(0.5), linear8(0.0), 255]
                );
                // The BGRA target is swizzled back to RGBA in both runs.
                assert_eq!(
                    &parallel[2].1.pixels[0..4],
                    &[
                        linear_to_srgb8(0.4),
                        linear_to_srgb8(0.6),
                        linear_to_srgb8(0.8),
                        255
                    ]
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn external_backbuffer_renders_identically_and_skips_readback() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let graph = &graph(&[node("tri", &[], &["swap"])]);
                let resources = [ResourceSpec::new("swap", 16, 16)];
                let kinds = |name: &str| match name {
                    "tri" => Some(PassKind::Triangle {
                        triangle: [1.0, 0.5, 0.0, 1.0],
                        background: [0.1, 0.1, 0.1, 1.0],
                    }),
                    _ => None,
                };
                // The same graph, rendered into an allocated target:
                let internal =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("internal");
                // ... and into a caller-provided texture (swapchain-shaped):
                let texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("onyx-test-swapchain"),
                    size: wgpu::Extent3d {
                        width: 16,
                        height: 16,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    // COPY_SRC so the test can verify the pixels — real
                    // swapchain frames carry only RENDER_ATTACHMENT.
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let external = ExternalTarget {
                    name: "swap",
                    texture: texture.clone(),
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    width: 16,
                    height: 16,
                };
                let frames =
                    execute_with_external_target(&renderer, graph, &resources, &kinds, &external)
                        .expect("external");
                // The external target is never read back...
                assert!(frames.iter().all(|(name, _)| *name != "swap"));
                // ...but its pixels match the internal render byte for byte:
                let external_pixels =
                    crate::offscreen::readback_texture(&renderer, &texture, 16, 16)
                        .expect("external readback");
                assert_eq!(external_pixels, internal[0].1.pixels);
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn external_backbuffer_validates_against_its_declaration() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("onyx-test-swapchain"),
                    size: wgpu::Extent3d {
                        width: 16,
                        height: 16,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let external = ExternalTarget {
                    name: "swap",
                    texture: texture.clone(),
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    width: 16,
                    height: 16,
                };
                let kinds = |name: &str| match name {
                    "clear" => Some(PassKind::Clear([0.0, 0.0, 0.0, 1.0])),
                    _ => None,
                };
                // Undeclared resource.
                let graph = &graph(&[node("clear", &[], &["swap"])]);
                let resources: [ResourceSpec; 1] = [ResourceSpec::new("other", 16, 16)];
                let error =
                    execute_with_external_target(&renderer, graph, &resources, &kinds, &external)
                        .expect_err("undeclared");
                assert!(error.to_string().contains("declared"), "{error}");
                // Format mismatch.
                let resources = [ResourceSpec::new("swap", 16, 16)];
                let error =
                    execute_with_external_target(&renderer, graph, &resources, &kinds, &external)
                        .expect_err("format mismatch");
                assert!(error.to_string().contains("format mismatch"), "{error}");
                // Size mismatch.
                let external = ExternalTarget {
                    name: "swap",
                    texture,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    width: 32,
                    height: 32,
                };
                let error =
                    execute_with_external_target(&renderer, graph, &resources, &kinds, &external)
                        .expect_err("size mismatch");
                assert!(error.to_string().contains("size mismatch"), "{error}");
                // MSAA declaration.
                let resources = [ResourceSpec::with_samples("swap", 16, 16, 4)];
                let error =
                    execute_with_external_target(&renderer, graph, &resources, &kinds, &external)
                        .expect_err("msaa");
                assert!(error.to_string().contains("single-sample"), "{error}");
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn sampling_the_external_backbuffer_is_rejected() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // A TexturedQuad that reads the swapchain target has no
                // sampleable source — rejected before encoding.
                let graph = &graph(&[node("blit", &["swap"], &["post"])]);
                let resources = [
                    ResourceSpec::new("swap", 16, 16),
                    ResourceSpec::new("post", 16, 16),
                ];
                let kinds = |name: &str| match name {
                    "blit" => Some(PassKind::TexturedQuad {
                        tint: [1.0, 1.0, 1.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                        uv_window: [0.0, 0.0, 1.0, 1.0],
                    }),
                    _ => None,
                };
                let texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("onyx-test-swapchain"),
                    size: wgpu::Extent3d {
                        width: 16,
                        height: 16,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let external = ExternalTarget {
                    name: "swap",
                    texture,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    width: 16,
                    height: 16,
                };
                let error =
                    execute_with_external_target(&renderer, graph, &resources, &kinds, &external)
                        .expect_err("sampling");
                assert!(error.to_string().contains("cannot be sampled"), "{error}");
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    /// The shader mirror: build [`crate::pbr::PbrParams`] from a `PassKind::Pbr`
    /// and shade it with [`crate::pbr::pbr_shaded`] — the exact bytes the GPU
    /// must produce (up to the sRGB encode).
    fn pbr_mirror(
        albedo: [f64; 3],
        roughness: f64,
        metallic: f64,
        light_dir: [f64; 3],
        light_color: [f64; 3],
        ambient: [f64; 3],
    ) -> [f32; 3] {
        let params =
            pbr::PbrParams::new(albedo, roughness, metallic, light_dir, light_color, ambient);
        pbr::pbr_shaded(&params)
    }

    /// Compare GPU pixels against the CPU mirror: each channel within 1/255
    /// (the hardware sRGB encode runs in f32; the reference in f64 — both
    /// round to the same byte except across a knife-edge boundary).
    fn assert_matches_mirror(pixels: &[u8], expected: [f32; 3]) {
        let expected = [
            linear_to_srgb8(f64::from(expected[0])),
            linear_to_srgb8(f64::from(expected[1])),
            linear_to_srgb8(f64::from(expected[2])),
        ];
        for chunk in pixels.chunks_exact(4) {
            for (channel, (got, want)) in chunk[..3].iter().zip(&expected).enumerate() {
                assert!(
                    got.abs_diff(*want) <= 1,
                    "channel {channel}: got {got}, expected within 1 of {want}"
                );
            }
            assert_eq!(chunk[3], 255);
        }
    }

    #[test]
    fn pbr_pass_matches_the_cpu_brdf_mirror() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let graph = &graph(&[node("chip", &[], &["mat"])]);
                let resources = [ResourceSpec::new("mat", 16, 16)];
                let albedo = [0.7, 0.5, 0.3];
                let roughness = 0.35;
                let metallic = 0.25;
                let light_dir = [1.0, 1.0, 1.0];
                let light_color = [1.0, 0.9, 0.8];
                let ambient = [0.1, 0.1, 0.15];
                let kinds = |name: &str| match name {
                    "chip" => Some(PassKind::Pbr {
                        albedo,
                        roughness,
                        metallic,
                        light_dir,
                        light_color,
                        ambient,
                    }),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].1.pixels.len(), 16 * 16 * 4);
                let expected =
                    pbr_mirror(albedo, roughness, metallic, light_dir, light_color, ambient);
                assert_matches_mirror(&frames[0].1.pixels, expected);
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn pbr_grazing_light_matches_ambient_albedo_exactly() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // Light perpendicular to the surface (ndotl = 0): diffuse
                // and specular vanish, so the pass renders ambient * albedo
                // per channel — exact bytes, no tolerance.
                let graph = &graph(&[node("chip", &[], &["mat"])]);
                let resources = [ResourceSpec::new("mat", 8, 8)];
                let kinds = |name: &str| match name {
                    "chip" => Some(PassKind::Pbr {
                        albedo: [1.0, 0.5, 0.25],
                        roughness: 0.4,
                        metallic: 0.0,
                        light_dir: [1.0, 0.0, 0.0],
                        light_color: [1.0; 3],
                        ambient: [0.5, 0.25, 0.5],
                    }),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                let expected = [
                    linear_to_srgb8(0.5),
                    linear_to_srgb8(0.125),
                    linear_to_srgb8(0.125),
                ];
                for chunk in frames[0].1.pixels.chunks_exact(4) {
                    assert_eq!(chunk, &[expected[0], expected[1], expected[2], 255]);
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn pbr_metal_in_shadow_renders_black() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // Metallic kills diffuse and ambient; ndotl = 0 kills
                // specular: a metal facing away from the light is black.
                let graph = &graph(&[node("chip", &[], &["mat"])]);
                let resources = [ResourceSpec::new("mat", 8, 8)];
                let kinds = |name: &str| match name {
                    "chip" => Some(PassKind::Pbr {
                        albedo: [0.9, 0.6, 0.2],
                        roughness: 0.2,
                        metallic: 1.0,
                        light_dir: [0.0, 1.0, 0.0],
                        light_color: [1.0; 3],
                        ambient: [0.5; 3],
                    }),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                for chunk in frames[0].1.pixels.chunks_exact(4) {
                    assert_eq!(chunk, &[0, 0, 0, 255]);
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn pbr_parallel_matches_sequential() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let graph = &graph(&[
                    node("dielectric", &[], &["a"]),
                    node("rough_metal", &[], &["b"]),
                    node("glossy", &[], &["c"]),
                ]);
                let resources = [
                    ResourceSpec::new("a", 8, 8),
                    ResourceSpec::new("b", 8, 8),
                    ResourceSpec::new("c", 8, 8),
                ];
                let kinds = |name: &str| match name {
                    "dielectric" => Some(PassKind::Pbr {
                        albedo: [0.7, 0.5, 0.3],
                        roughness: 0.35,
                        metallic: 0.0,
                        light_dir: [1.0, 1.0, 1.0],
                        light_color: [1.0, 0.9, 0.8],
                        ambient: [0.1; 3],
                    }),
                    "rough_metal" => Some(PassKind::Pbr {
                        albedo: [0.8, 0.7, 0.6],
                        roughness: 0.9,
                        metallic: 1.0,
                        light_dir: [0.5, 0.5, 1.0],
                        light_color: [1.0; 3],
                        ambient: [0.05; 3],
                    }),
                    "glossy" => Some(PassKind::Pbr {
                        albedo: [0.2, 0.4, 0.9],
                        roughness: 0.1,
                        metallic: 0.2,
                        light_dir: [0.0, 0.3, 1.0],
                        light_color: [1.0; 3],
                        ambient: [0.0, 0.1, 0.2],
                    }),
                    _ => None,
                };
                let sequential =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("sequential");
                let parallel = execute_offscreen_parallel(&renderer, graph, &resources, &kinds, 2)
                    .expect("parallel");
                for ((s_name, s_frame), (p_name, p_frame)) in sequential.iter().zip(&parallel) {
                    assert_eq!(s_name, p_name);
                    assert_eq!(s_frame.pixels, p_frame.pixels, "target '{s_name}'");
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn pbr_target_can_be_sampled_by_a_textured_quad() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // PBR renders the chip; a sampling pass blits it through the
                // quad pipeline — the pixel-exact copy contract of increment
                // 5, now over PBR-generated content.
                let graph = &graph(&[
                    node("chip", &[], &["mat"]),
                    node("blit", &["mat"], &["post"]),
                ]);
                let resources = [
                    ResourceSpec::new("mat", 8, 8),
                    ResourceSpec::new("post", 8, 8),
                ];
                let kinds = |name: &str| match name {
                    "chip" => Some(PassKind::Pbr {
                        albedo: [0.7, 0.5, 0.3],
                        roughness: 0.35,
                        metallic: 0.25,
                        light_dir: [1.0, 1.0, 1.0],
                        light_color: [1.0, 0.9, 0.8],
                        ambient: [0.1, 0.1, 0.15],
                    }),
                    "blit" => Some(PassKind::TexturedQuad {
                        tint: [1.0, 1.0, 1.0, 1.0],
                        background: [0.0, 0.0, 0.0, 1.0],
                        uv_window: [0.0, 0.0, 1.0, 1.0],
                    }),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                assert_eq!(frames[0].1.pixels, frames[1].1.pixels);
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn pbr_pass_supports_msaa_targets() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // The chip pass renders into a 4× MSAA target; the resolve
                // machinery lands the shaded color in the readback texture.
                let graph = &graph(&[node("chip", &[], &["mat"])]);
                let resources = [ResourceSpec::with_samples("mat", 8, 8, 4)];
                let albedo = [0.6, 0.4, 0.2];
                let light_dir = [0.5, 0.5, 1.0];
                let kinds = |name: &str| match name {
                    "chip" => Some(PassKind::Pbr {
                        albedo,
                        roughness: 0.3,
                        metallic: 0.1,
                        light_dir,
                        light_color: [1.0; 3],
                        ambient: [0.1; 3],
                    }),
                    _ => None,
                };
                let frames =
                    execute_offscreen(&renderer, graph, &resources, &kinds).expect("execute");
                let expected = pbr_mirror(albedo, 0.3, 0.1, light_dir, [1.0; 3], [0.1; 3]);
                assert_matches_mirror(&frames[0].1.pixels, expected);
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn pbr_passes_reject_invalid_declarations() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let base_graph = &graph(&[node("chip", &[], &["mat"])]);
                let reads_graph = &graph(&[node("chip", &["mat"], &["mat"])]);
                let resources = [ResourceSpec::new("mat", 8, 8)];
                // Out-of-range parameters are rejected before any GPU work.
                let base = PassKind::Pbr {
                    albedo: [0.7, 0.5, 0.3],
                    roughness: 0.35,
                    metallic: 0.25,
                    light_dir: [1.0, 1.0, 1.0],
                    light_color: [1.0, 0.9, 0.8],
                    ambient: [0.1, 0.1, 0.15],
                };
                let PassKind::Pbr {
                    albedo,
                    roughness,
                    metallic,
                    light_dir,
                    light_color,
                    ambient,
                } = base
                else {
                    unreachable!("base is a Pbr pass");
                };
                for (kind, needle) in [
                    (
                        PassKind::Pbr {
                            roughness: 1.5,
                            albedo,
                            metallic,
                            light_dir,
                            light_color,
                            ambient,
                        },
                        "roughness",
                    ),
                    (
                        PassKind::Pbr {
                            metallic: -0.1,
                            albedo,
                            roughness,
                            light_dir,
                            light_color,
                            ambient,
                        },
                        "metallic",
                    ),
                    (
                        PassKind::Pbr {
                            albedo: [0.7, 2.5, 0.3],
                            roughness,
                            metallic,
                            light_dir,
                            light_color,
                            ambient,
                        },
                        "albedo[1]",
                    ),
                    (
                        PassKind::Pbr {
                            light_dir: [0.0, 0.0, 0.0],
                            albedo,
                            roughness,
                            metallic,
                            light_color,
                            ambient,
                        },
                        "zero vector",
                    ),
                ] {
                    let kinds = move |name: &str| match name {
                        "chip" => Some(kind),
                        _ => None,
                    };
                    let error = execute_offscreen(&renderer, base_graph, &resources, &kinds)
                        .expect_err("invalid pbr parameters");
                    assert!(error.to_string().contains(needle), "{error}");
                }
                // Pbr reads nothing: a read declaration is a mismatch.
                let kinds = |name: &str| match name {
                    "chip" => Some(base),
                    _ => None,
                };
                let error = execute_offscreen(&renderer, reads_graph, &resources, &kinds)
                    .expect_err("reads");
                assert!(error.to_string().contains("must read exactly 0"), "{error}");
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }
}
