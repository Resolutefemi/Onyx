//! Batched 2D sprites — the CanvasItem layer (`docs/PRD.md` §4, §10).
//!
//! Mirrors `bevy_sprite`'s batched rendering and Godot's
//! `CanvasItem.draw_texture_rect`: one pass submits any number of sprites
//! as one vertex buffer and one draw call. Each sprite is a top-left
//! anchored destination rect, a uv window into the bound source texture
//! and a linear-space tint; sprites draw in submission order, later
//! sprites overwriting earlier ones (no blending in this increment).
//!
//! The camera is the 1:1 viewport camera — Godot's default canvas
//! transform: top-left origin, y down, one canvas pixel = one target
//! pixel. [`Camera2d::project`] is the CPU mirror of the shader's
//! `view_projection * vec4(p, 0.0, 1.0)` with the identical operation
//! sequence, so sprite placement is predictable bit for bit on every
//! backend.

use crate::renderer::Renderer;

/// The sprite shader: one `mat4x4` camera uniform, the source texture and
/// sampler, per-vertex canvas positions projected in the vertex stage, and
/// the quad pass's tint multiply in the fragment stage.
pub const SPRITE_WGSL: &str = r#"
struct Camera {
    view_projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsIn {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tint: vec4<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = camera.view_projection * vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    out.tint = in.tint;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv) * in.tint;
}
"#;

/// One 2D sprite: the Godot `draw_texture_rect(rect, src_rect, modulate)`
/// shape — a destination rect, a source uv window and a tint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sprite2d {
    /// Destination rect in canvas pixels, top-left anchored: `[x, y, w, h]`.
    pub rect: [f64; 4],
    /// Source window into the bound texture: offset xy, scale zw (the same
    /// convention as the textured-quad pass's `uv_window`).
    pub uv: [f64; 4],
    /// RGBA tint multiplied over the sampled texel in linear space.
    pub tint: [f64; 4],
}

impl Sprite2d {
    /// A sprite covering the whole viewport, sampling the full texture.
    #[must_use]
    pub const fn full_viewport(width: f64, height: f64) -> Self {
        Self {
            rect: [0.0, 0.0, width, height],
            uv: [0.0, 0.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// One sprite-batch vertex: a canvas-space corner with its uv and tint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteVertex {
    /// Canvas position in pixels (projected by [`Camera2d`] in the shader).
    pub position: [f32; 2],
    /// Texture uv of this corner.
    pub uv: [f32; 2],
    /// RGBA tint in linear space, flat per sprite.
    pub tint: [f32; 4],
}

impl SpriteVertex {
    /// wgpu vertex layout: position, uv, tint — 32-byte stride.
    #[must_use]
    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: 32,
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
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    shader_location: 2,
                    offset: 16,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// The 1:1 viewport camera: top-left origin, y down, one canvas pixel =
/// one target pixel. Free cameras, canvas transforms and layers land with
/// the scene-graph milestone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2d {
    /// Viewport width in pixels.
    pub width: u32,
    /// Viewport height in pixels.
    pub height: u32,
}

impl Camera2d {
    /// The camera for a `width`×`height` viewport.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Column-major view-projection: `ndc = vp * (px, py, 0, 1)` maps
    /// canvas pixel (0, 0) to the top-left corner (-1, 1) and
    /// `(width, height)` to the bottom-right corner (1, -1).
    #[must_use]
    pub fn view_projection(&self) -> [[f32; 4]; 4] {
        let sx = 2.0 / self.width as f32;
        let sy = -2.0 / self.height as f32;
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
        ]
    }

    /// The CPU mirror of the shader's `view_projection *
    /// vec4(position, 0.0, 1.0)`: the same column-scalar accumulation, so
    /// the projected corners are identical bit for bit. For this camera the
    /// x row carries exactly one non-zero scale (columns 1 and 2 contribute
    /// exact zeros, column 3 an exact ±1), and so does the y row — the
    /// addition order cannot change the bits.
    #[must_use]
    pub fn project(&self, position: [f32; 2]) -> [f32; 2] {
        let vp = self.view_projection();
        let mut ndc = [0.0f32; 2];
        for row in 0..2 {
            let mut acc = vp[0][row] * position[0];
            acc += vp[1][row] * position[1];
            acc += vp[2][row] * 0.0;
            acc += vp[3][row] * 1.0;
            ndc[row] = acc;
        }
        ndc
    }

    /// The uniform bytes: the view-projection matrix as four `vec4<f32>`
    /// columns, little-endian (64 bytes).
    #[must_use]
    pub fn uniform_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        for column in &self.view_projection() {
            for value in column {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }
}

/// Build the batch's vertex buffer contents: six vertices per sprite (two
/// triangles — the same TL, TR, BR / TL, BR, BL split as the vertexless
/// quad), with corner uvs from the sprite's window and a flat tint.
#[must_use]
pub fn sprite_vertices(sprites: &[Sprite2d]) -> Vec<SpriteVertex> {
    let mut vertices = Vec::with_capacity(sprites.len() * 6);
    for sprite in sprites {
        let [x, y, w, h] = sprite.rect;
        let [u0, v0, uw, vh] = sprite.uv;
        let tint = [
            sprite.tint[0] as f32,
            sprite.tint[1] as f32,
            sprite.tint[2] as f32,
            sprite.tint[3] as f32,
        ];
        let corners = [
            ([x, y], [u0, v0]),
            ([x + w, y], [u0 + uw, v0]),
            ([x + w, y + h], [u0 + uw, v0 + vh]),
            ([x, y + h], [u0, v0 + vh]),
        ];
        for &(px_py, uv) in &[
            corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
        ] {
            vertices.push(SpriteVertex {
                position: [px_py[0] as f32, px_py[1] as f32],
                uv: [uv[0] as f32, uv[1] as f32],
                tint,
            });
        }
    }
    vertices
}

/// Pack sprite vertices into little-endian f32 bytes matching
/// [`SpriteVertex::layout`].
#[must_use]
pub fn sprite_vertex_bytes(vertices: &[SpriteVertex]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vertices.len() * 32);
    for vertex in vertices {
        for value in vertex.position.iter().chain(vertex.uv.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.tint {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// A `VERTEX | COPY_DST` buffer holding one sprite batch's vertices.
pub struct SpriteVertexBuffer {
    buffer: wgpu::Buffer,
}

impl SpriteVertexBuffer {
    /// Create the buffer and upload `vertices` immediately.
    #[must_use]
    pub fn new(renderer: &Renderer, vertices: &[SpriteVertex]) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("onyx-sprite-vertices"),
                contents: &sprite_vertex_bytes(vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        Self { buffer }
    }

    /// The underlying wgpu buffer (for draw calls).
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// A `UNIFORM | COPY_DST` buffer holding one [`Camera2d`]'s matrix.
pub struct CameraUniformBuffer {
    buffer: wgpu::Buffer,
}

impl CameraUniformBuffer {
    /// Create the buffer and upload `camera` immediately.
    #[must_use]
    pub fn new(renderer: &Renderer, camera: &Camera2d) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("onyx-sprite-camera"),
                contents: &camera.uniform_bytes(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        Self { buffer }
    }

    /// The underlying wgpu buffer (for bind group construction).
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Bind group layout for the sprite pipeline: camera uniform (0), texture
/// (1), sampler (2) — the quad layout's shape.
#[must_use]
pub fn create_sprite_bind_layout(renderer: &Renderer) -> wgpu::BindGroupLayout {
    renderer
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("onyx-sprite-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
}

/// The sprite bind group: camera uniform (0), texture (1), sampler (2).
#[must_use]
pub fn create_sprite_bind_group(
    renderer: &Renderer,
    layout: &wgpu::BindGroupLayout,
    camera: &CameraUniformBuffer,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("onyx-sprite-bind-group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
}

/// Compile the sprite shader and create its pipeline (bind group 0) for
/// `(samples, format)`. The sprite layout (position, uv, tint) provides
/// the vertex buffers; the pass provides one draw call per batch.
#[must_use]
pub fn create_sprite_pipeline(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::RenderPipeline {
    let device = &renderer.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("onyx-sprite-shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SPRITE_WGSL)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("onyx-sprite-pipeline-layout"),
        bind_group_layouts: &[Some(&create_sprite_bind_layout(renderer))],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("onyx-sprite-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(SpriteVertex::layout())],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: samples,
            ..wgpu::MultisampleState::default()
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_projects_corners_to_ndc_exactly() {
        let camera = Camera2d::new(16, 16);
        assert_eq!(camera.project([0.0, 0.0]), [-1.0, 1.0]);
        assert_eq!(camera.project([16.0, 16.0]), [1.0, -1.0]);
        assert_eq!(camera.project([8.0, 8.0]), [0.0, 0.0]);
        // Non-square viewports keep the y-down top-left convention.
        let wide = Camera2d::new(32, 16);
        assert_eq!(wide.project([0.0, 0.0]), [-1.0, 1.0]);
        assert_eq!(wide.project([32.0, 16.0]), [1.0, -1.0]);
    }

    #[test]
    fn camera_projects_pixel_centers() {
        let camera = Camera2d::new(16, 16);
        // sx = 0.125, sy = -0.125: centers land exactly between corners.
        assert_eq!(camera.project([0.5, 0.5]), [-0.9375, 0.9375]);
        assert_eq!(camera.project([15.5, 15.5]), [0.9375, -0.9375]);
    }

    #[test]
    fn camera_uniform_bytes_are_four_column_vec4s() {
        let camera = Camera2d::new(16, 16);
        let bytes = camera.uniform_bytes();
        assert_eq!(bytes.len(), 64);
        assert_eq!(&bytes[0..4], &0.125f32.to_le_bytes()); // sx
        assert_eq!(&bytes[4..8], &0.0f32.to_le_bytes());
        assert_eq!(&bytes[20..24], &(-0.125f32).to_le_bytes()); // sy (col 1, row 1)
        assert_eq!(&bytes[48..52], &(-1.0f32).to_le_bytes()); // tx (col 3, row 0)
        assert_eq!(&bytes[52..56], &1.0f32.to_le_bytes()); // ty (col 3, row 1)
    }

    #[test]
    fn vertex_layout_is_position_uv_tint() {
        let layout = SpriteVertex::layout();
        assert_eq!(layout.array_stride, 32);
        assert_eq!(layout.attributes.len(), 3);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].offset, 8);
        assert_eq!(layout.attributes[2].offset, 16);
    }

    #[test]
    fn sprite_vertices_build_two_triangles_per_sprite() {
        let sprite = Sprite2d {
            rect: [2.0, 3.0, 8.0, 4.0],
            uv: [0.25, 0.25, 0.5, 0.5],
            tint: [0.5, 0.25, 0.75, 1.0],
        };
        let vertices = sprite_vertices(&[sprite]);
        assert_eq!(vertices.len(), 6);
        // Corner order: TL, TR, BR / TL, BR, BL — the quad pass's split.
        let expected = [
            ([2.0, 3.0], [0.25, 0.25]),
            ([10.0, 3.0], [0.75, 0.25]),
            ([10.0, 7.0], [0.75, 0.75]),
            ([2.0, 3.0], [0.25, 0.25]),
            ([10.0, 7.0], [0.75, 0.75]),
            ([2.0, 7.0], [0.25, 0.75]),
        ];
        for (vertex, (position, uv)) in vertices.iter().zip(expected) {
            assert_eq!(vertex.position, position);
            assert_eq!(vertex.uv, uv);
            assert_eq!(vertex.tint, [0.5, 0.25, 0.75, 1.0]);
        }
    }

    #[test]
    fn sprite_vertices_batch_all_sprites_in_order() {
        let sprites = [
            Sprite2d {
                rect: [0.0, 0.0, 8.0, 8.0],
                uv: [0.0, 0.0, 0.5, 0.5],
                tint: [1.0, 1.0, 1.0, 1.0],
            },
            Sprite2d {
                rect: [8.0, 8.0, 8.0, 8.0],
                uv: [0.5, 0.5, 0.5, 0.5],
                tint: [0.5, 0.5, 0.5, 1.0],
            },
        ];
        let vertices = sprite_vertices(&sprites);
        assert_eq!(vertices.len(), 12);
        // First vertex: the first sprite's top-left.
        assert_eq!(vertices[0].position, [0.0, 0.0]);
        assert_eq!(vertices[0].uv, [0.0, 0.0]);
        assert_eq!(vertices[0].tint, [1.0, 1.0, 1.0, 1.0]);
        // Last vertex: the second sprite's bottom-left.
        assert_eq!(vertices[11].position, [8.0, 16.0]);
        assert_eq!(vertices[11].uv, [0.5, 1.0]);
        assert_eq!(vertices[11].tint, [0.5, 0.5, 0.5, 1.0]);
    }

    #[test]
    fn sprite_vertex_bytes_pack_the_layout() {
        let vertices = sprite_vertices(&[Sprite2d {
            rect: [1.0, 2.0, 4.0, 4.0],
            uv: [0.0, 0.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0, 1.0],
        }]);
        let bytes = sprite_vertex_bytes(&vertices);
        assert_eq!(bytes.len(), 6 * 32);
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &2.0f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &0.0f32.to_le_bytes());
        assert_eq!(&bytes[16..20], &1.0f32.to_le_bytes()); // tint r
    }
}
