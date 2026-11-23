//! Offscreen graph execution: one command encoder, passes in the graph's
//! topological order, a single submission, then per-target readback
//! (`docs/PRD.md` §2).
//!
//! Within one submission, wgpu inserts GPU barriers automatically from the
//! real resource usage of each pass — the graph fixes pass ORDER, wgpu
//! guarantees synchronization. This is the automatic-barrier story behind the
//! PRD's DAG render graph: nodes that read a resource run after its writers,
//! conflicting writers run in declaration order, and the executor never
//! needs manual barriers.
//!
//! Increment scope: nodes render clears or the cached triangle, each writing
//! exactly one target and reading none. Bind groups / resource sampling and
//! callback-driven passes land with the 2D suite and PBR.

use crate::graph::{GraphError, RenderGraph};
use crate::offscreen::{readback_texture, OffscreenFrame};
use crate::pipeline;
use crate::renderer::{RenderError, Renderer};

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
}

/// Execute `graph` offscreen: allocate every declared resource, encode all
/// passes into one encoder in topological order, submit once, and read back
/// every target (in resource declaration order).
///
/// # Errors
/// [`RenderError::Graph`] for graph scheduling failures, [`RenderError::
/// Unsupported`] for node declarations this executor cannot yet encode,
/// [`RenderError::Poll`] / [`RenderError::Map`] on readback failure.
pub fn execute_offscreen(
    renderer: &Renderer,
    graph: &RenderGraph,
    resources: &[ResourceSpec],
    kind_of: &dyn Fn(&str) -> Option<PassKind>,
) -> Result<Vec<(&'static str, OffscreenFrame)>, RenderError> {
    let order = graph
        .execution_order()
        .map_err(|error: GraphError| RenderError::Graph {
            message: error.to_string(),
        })?;

    let device = &renderer.device;
    let mut targets: Vec<(&'static str, wgpu::Texture, wgpu::TextureView)> =
        Vec::with_capacity(resources.len());
    for spec in resources {
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
        targets.push((spec.name, texture, view));
    }

    let triangle_pipeline =
        pipeline::create_triangle_pipeline(renderer, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("onyx-graph-executor"),
    });

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
                    "node '{}' reads resources; sampling lands with bind groups",
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
        let Some((_, _, view)) = targets.iter().find(|(name, _, _)| *name == written) else {
            return Err(RenderError::Unsupported {
                message: format!(
                    "node '{}' writes undeclared resource '{written}'",
                    node.name
                ),
            });
        };

        match kind {
            PassKind::Clear(color) => {
                let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("onyx-graph-clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: color[0],
                                g: color[1],
                                b: color[2],
                                a: color[3],
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..wgpu::RenderPassDescriptor::default()
                });
                // Drop the pass handle: the clear happens on store.
                drop(pass);
            }
            PassKind::Triangle {
                triangle,
                background,
            } => {
                let vertex_buffer = pipeline::triangle_vertex_buffer(renderer, triangle);
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("onyx-graph-triangle"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: background[0],
                                g: background[1],
                                b: background[2],
                                a: background[3],
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..wgpu::RenderPassDescriptor::default()
                });
                pass.set_pipeline(&triangle_pipeline);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.draw(0..3, 0..1);
            }
        }
    }

    renderer.queue.submit([encoder.finish()]);

    let mut frames = Vec::with_capacity(targets.len());
    for (index, spec) in resources.iter().enumerate() {
        let (_, texture, _) = &targets[index];
        let pixels = readback_texture(renderer, texture, spec.width, spec.height)?;
        frames.push((
            spec.name,
            OffscreenFrame {
                width: spec.width,
                height: spec.height,
                pixels,
            },
        ));
    }
    Ok(frames)
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
                let resources = [ResourceSpec {
                    name: "t",
                    width: 16,
                    height: 16,
                }];
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
                let resources = [ResourceSpec {
                    name: "t",
                    width: 16,
                    height: 16,
                }];
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
                let resources = [ResourceSpec {
                    name: "t",
                    width: 4,
                    height: 4,
                }];
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
}
