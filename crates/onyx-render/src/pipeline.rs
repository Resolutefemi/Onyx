//! The first real shader pipeline: a vertex-colored WGSL triangle
//! (`docs/PRD.md` §2).
//!
//! Exercises shader compilation, pipeline creation, vertex buffers and draw
//! calls end to end — with no bind groups yet (uniforms, textures and
//! samplers land with the 2D suite). The WGSL is inlined so the pipeline is
//! testable headless; assets-on-disk shader loading arrives with the asset
//! system.

use crate::offscreen::{readback_texture, OffscreenFrame};
use crate::renderer::{RenderError, Renderer};

/// The triangle shader: position and color come from the vertex buffer.
pub const TRIANGLE_WGSL: &str = r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs(
    @builtin(vertex_index) index: u32,
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
) -> VsOut {
    var out: VsOut;
    let ndc = vec3<f32>(position, 0.0);
    out.position = vec4<f32>(ndc, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

/// A position/color vertex. 20 bytes, packed little-endian — matches
/// [`Vertex::LAYOUT`]; bytes are built explicitly (no unsafe casts).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    /// Clip-space position (pre-projection; this pipeline applies none).
    pub position: [f32; 2],
    /// Per-vertex RGB color.
    pub color: [f32; 3],
}

impl Vertex {
    /// wgpu vertex layout for [`Vertex`]: two attributes, 20-byte stride.
    #[must_use]
    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: 20,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset: 8,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Pack vertices into little-endian f32 bytes matching [`Vertex::layout`].
#[must_use]
pub fn vertex_bytes(vertices: &[Vertex]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vertices.len() * 20);
    for vertex in vertices {
        for value in vertex.position.iter().chain(vertex.color.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// Compile the triangle shader and create the render pipeline for `format`.
#[must_use]
pub fn create_triangle_pipeline(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let device = &renderer.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("onyx-triangle-shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TRIANGLE_WGSL)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("onyx-triangle-layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("onyx-triangle-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(Vertex::layout())],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Build the triangle's vertex buffer (three same-colored vertices).
#[must_use]
pub fn triangle_vertex_buffer(renderer: &Renderer, color: [f64; 4]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let color32 = [color[0] as f32, color[1] as f32, color[2] as f32];
    let vertices = [
        Vertex {
            position: [0.0, 0.6],
            color: color32,
        },
        Vertex {
            position: [-0.6, -0.6],
            color: color32,
        },
        Vertex {
            position: [0.6, -0.6],
            color: color32,
        },
    ];
    renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("onyx-triangle-vertices"),
            contents: &vertex_bytes(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
}

/// Render the triangle into an offscreen RGBA8 sRGB texture and read it back.
///
/// The pixel contract is deterministic: the frame center is the triangle
/// color and the corners stay the background color (no MSAA, no blending).
///
/// # Errors
/// [`RenderError::Poll`] / [`RenderError::Map`] on readback failure.
pub fn render_triangle_offscreen(
    renderer: &Renderer,
    width: u32,
    height: u32,
    background: [f64; 4],
    triangle: [f64; 4],
) -> Result<OffscreenFrame, RenderError> {
    let device = &renderer.device;
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("onyx-triangle-target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let pipeline = create_triangle_pipeline(renderer, wgpu::TextureFormat::Rgba8UnormSrgb);
    let vertex_buffer = triangle_vertex_buffer(renderer, triangle);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("onyx-triangle-encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("onyx-triangle-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..3, 0..1);
    }
    renderer.queue.submit([encoder.finish()]);

    let pixels = readback_texture(renderer, &texture, width, height)?;
    Ok(OffscreenFrame {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offscreen::linear_to_srgb8;
    use crate::RendererConfig;

    #[test]
    fn vertex_bytes_match_declared_layout() {
        let vertices = [Vertex {
            position: [1.5, -2.0],
            color: [0.25, 0.5, 1.0],
        }];
        let bytes = vertex_bytes(&vertices);
        assert_eq!(bytes.len(), 20);
        assert_eq!(&bytes[0..4], &1.5f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &(-2.0f32).to_le_bytes());
        assert_eq!(&bytes[8..12], &0.25f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0.5f32.to_le_bytes());
        assert_eq!(&bytes[16..20], &1.0f32.to_le_bytes());
    }

    #[test]
    fn triangle_renders_its_color_at_center_and_keeps_background_at_corners() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let background = [0.05, 0.05, 0.1, 1.0];
                let triangle = [1.0, 0.5, 0.0, 1.0];
                let frame = render_triangle_offscreen(&renderer, 64, 64, background, triangle)
                    .expect("triangle frame");
                assert_eq!(frame.pixels.len(), 64 * 64 * 4);

                let center = (32 * 64 + 32) * 4;
                assert_eq!(
                    &frame.pixels[center..center + 4],
                    &[
                        linear_to_srgb8(1.0),
                        linear_to_srgb8(0.5),
                        linear_to_srgb8(0.0),
                        255
                    ],
                    "frame center must be the triangle color"
                );
                let corner = 0;
                assert_eq!(
                    &frame.pixels[corner..corner + 4],
                    &[
                        linear_to_srgb8(0.05),
                        linear_to_srgb8(0.05),
                        linear_to_srgb8(0.1),
                        255
                    ],
                    "frame corner must stay the background color"
                );
            }
            Err(RenderError::NoAdapter { .. }) => {
                // Headless CI: renderer tests cover the graceful path.
            }
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }
}
