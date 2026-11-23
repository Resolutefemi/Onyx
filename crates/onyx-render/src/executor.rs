//! Offscreen graph execution: passes in the graph's topological order,
//! single- or multi-encoder submissions, then per-target readback
//! (`docs/PRD.md` §2).
//!
//! Two executors share one planning phase ([`plan_passes`], which validates
//! every node and resolves targets before any GPU work) and one allocation
//! phase:
//!
//! - [`execute_offscreen`] — the sequential path: one command encoder, all
//!   passes in topological order, a single submission,
//! - [`execute_offscreen_parallel`] — the parallel path: the graph decomposes
//!   into *waves* (see [`crate::graph::RenderGraph::execution_waves`]); every
//!   node in a wave encodes on its own thread into its own command encoder,
//!   each wave submits its command buffers in one call, and wave N+1's
//!   submission follows wave N's on the queue. Write-write conflicts always
//!   cross waves (the graph edges force it), so parallel results are
//!   bit-identical to sequential ones — proven by a test.
//!
//! Within one submission wgpu inserts GPU barriers automatically from the
//! real resource usage of each pass, and submissions to one queue execute in
//! order — the graph fixes ORDER, wgpu guarantees synchronization.
//!
//! Targets may declare MSAA via [`ResourceSpec::samples`]: the executor then
//! creates a multisampled attachment and resolves it into the single-sample
//! readback texture on every pass. Pipelines are built per sample count, and
//! sample counts are validated against the adapter before any encoding.

use crate::graph::{GraphError, RenderGraph};
use crate::offscreen::{readback_texture, OffscreenFrame};
use crate::pipeline;
use crate::renderer::{RenderError, Renderer};
use crate::resources;
/// What a node renders this increment.
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
}

impl ResourceSpec {
    /// A single-sample target — the common case.
    #[must_use]
    pub const fn new(name: &'static str, width: u32, height: u32) -> Self {
        Self {
            name,
            width,
            height,
            samples: 1,
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
        }
    }
}

/// One validated node: its pass kind, the target it writes and that target's
/// sample count.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PlannedPass {
    node: &'static str,
    kind: PassKind,
    target_index: usize,
}

/// A declared target and everything the executors need to draw into it.
struct Target {
    name: &'static str,
    width: u32,
    height: u32,
    /// Single-sample texture: the resolve target when `msaa_view` is set,
    /// the render target otherwise; always the readback source.
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// `Some` when `samples > 1`: the multisampled attachment views.
    msaa_view: Option<wgpu::TextureView>,
    samples: u32,
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

/// Walk the graph in topological order and validate every node before any
/// GPU work: each node must have a pass kind, read nothing, write exactly one
/// declared target, and that target's sample count must be supported.
fn plan_passes(
    renderer: &Renderer,
    graph: &RenderGraph,
    resources: &[ResourceSpec],
    kind_of: &dyn Fn(&str) -> Option<PassKind>,
) -> Result<Vec<PlannedPass>, RenderError> {
    let order = graph
        .execution_order()
        .map_err(|error: GraphError| RenderError::Graph {
            message: error.to_string(),
        })?;
    let mut passes = Vec::with_capacity(order.len());
    for node_name in order {
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
        if !node.reads.is_empty() {
            return Err(RenderError::Unsupported {
                message: format!(
                    "node '{}' reads resources; sampling passes land with the 2D suite",
                    node.name
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
        passes.push(PlannedPass {
            node: node_name,
            kind,
            target_index,
        });
    }
    Ok(passes)
}

/// Allocate every declared resource. Targets with `samples > 1` get a
/// multisampled attachment alongside the single-sample resolve texture.
fn allocate_targets(renderer: &Renderer, resources: &[ResourceSpec]) -> Vec<Target> {
    let device = &renderer.device;
    resources
        .iter()
        .map(|spec| {
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
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
            }
        })
        .collect()
}

/// The triangle pipeline for `samples` — the plain pipeline at 1×, the MSAA
/// variant otherwise. Cached per sample count.
fn triangle_pipeline_for(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
    samples: u32,
    cache: &mut Vec<(u32, wgpu::RenderPipeline)>,
) -> wgpu::RenderPipeline {
    if let Some((_, existing)) = cache.iter().find(|(count, _)| *count == samples) {
        return existing.clone();
    }
    let built = if samples <= 1 {
        pipeline::create_triangle_pipeline(renderer, format)
    } else {
        resources::create_triangle_pipeline_msaa(renderer, format, samples)
    };
    cache.push((samples, built.clone()));
    built
}

/// Encode one planned pass into `encoder`, targeting `target`.
fn encode_pass(
    renderer: &Renderer,
    encoder: &mut wgpu::CommandEncoder,
    pass: &PlannedPass,
    target: &Target,
    pipelines: &mut Vec<(u32, wgpu::RenderPipeline)>,
) {
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
            let pipeline = triangle_pipeline_for(
                renderer,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                target.samples,
                pipelines,
            );
            pass_handle.set_pipeline(&pipeline);
            pass_handle.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass_handle.draw(0..3, 0..1);
        }
    }
}

/// Read back every target in resource declaration order.
fn readback_targets(
    renderer: &Renderer,
    targets: &[Target],
) -> Result<Vec<(&'static str, OffscreenFrame)>, RenderError> {
    let mut frames = Vec::with_capacity(targets.len());
    for target in targets {
        let pixels = readback_texture(renderer, &target.texture, target.width, target.height)?;
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
    let passes = plan_passes(renderer, graph, resources, kind_of)?;
    let targets = allocate_targets(renderer, resources);
    let mut pipelines = Vec::new();
    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("onyx-graph-executor"),
        });
    for pass in &passes {
        encode_pass(
            renderer,
            &mut encoder,
            pass,
            &targets[pass.target_index],
            &mut pipelines,
        );
    }
    renderer.queue.submit([encoder.finish()]);
    readback_targets(renderer, &targets)
}

/// Execute `graph` offscreen in parallel: waves of independent nodes encode
/// concurrently (one thread and one command encoder per node group), each
/// wave submits its command buffers in one call, and later waves follow on
/// the queue — submission order makes cross-wave dependencies safe.
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
        .map(|wave| {
            wave.iter()
                .map(|&node_name| {
                    let Some(kind) = kind_of(node_name) else {
                        return Err(RenderError::Unsupported {
                            message: format!("node '{node_name}' has no pass kind"),
                        });
                    };
                    let node = graph
                        .nodes()
                        .iter()
                        .find(|node| node.name == node_name)
                        .expect("wave names come from the graph");
                    if !node.reads.is_empty() {
                        return Err(RenderError::Unsupported {
                            message: format!(
                                "node '{}' reads resources; sampling passes land with the 2D suite",
                                node.name
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
                    Ok(PlannedPass {
                        node: node_name,
                        kind,
                        target_index,
                    })
                })
                .collect::<Result<Vec<PlannedPass>, RenderError>>()
        })
        .collect::<Result<Vec<Vec<PlannedPass>>, RenderError>>()?;

    let targets = allocate_targets(renderer, resources);
    let workers = workers.max(1);

    for wave in &planned {
        // Prebuild the pipelines this wave needs — RenderPipeline is
        // Send + Sync, so threads can share them.
        let mut pipelines: Vec<(u32, wgpu::RenderPipeline)> = Vec::new();
        for pass in wave {
            if matches!(pass.kind, PassKind::Triangle { .. }) {
                let target_samples = targets[pass.target_index].samples;
                if !pipelines.iter().any(|(count, _)| *count == target_samples) {
                    let built = if target_samples <= 1 {
                        pipeline::create_triangle_pipeline(
                            renderer,
                            wgpu::TextureFormat::Rgba8UnormSrgb,
                        )
                    } else {
                        resources::create_triangle_pipeline_msaa(
                            renderer,
                            wgpu::TextureFormat::Rgba8UnormSrgb,
                            target_samples,
                        )
                    };
                    pipelines.push((target_samples, built));
                }
            }
        }
        let chunk_size = wave.len().div_ceil(workers);
        let targets_ref = &targets;
        let pipelines_ref = &pipelines;
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
                        let mut pipeline_cache = pipelines_ref.clone();
                        for pass in chunk {
                            encode_pass(
                                renderer,
                                &mut encoder,
                                pass,
                                &targets_ref[pass.target_index],
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

    readback_targets(renderer, &targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RenderGraphNode;
    use crate::offscreen::linear_to_srgb8;
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

    /// A mixed graph: three independent targets plus a conflicting-writer
    /// pair (`scene_clear` then `clear_over` on "scene" — declaration order
    /// must survive parallel execution). Target "a" is 4× MSAA.
    fn mixed_graph() -> (
        RenderGraph,
        [ResourceSpec; 4],
        impl Fn(&str) -> Option<PassKind>,
    ) {
        let graph = graph(&[
            node("scene_clear", &[], &["scene"]),
            node("tri_a", &[], &["a"]),
            node("tri_b", &[], &["b"]),
            node("clear_over", &[], &["scene"]),
            node("clear_c", &[], &["c"]),
        ]);
        let resources = [
            ResourceSpec::new("scene", 16, 16),
            ResourceSpec::with_samples("a", 16, 16, 4),
            ResourceSpec::new("b", 16, 16),
            ResourceSpec::new("c", 16, 16),
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
                    STRESS_TARGETS
                        .iter()
                        .position(|target| *target == name)
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
}
