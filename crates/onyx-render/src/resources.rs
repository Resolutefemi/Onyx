//! Bind groups, uniform buffers, textures and samplers — the resource layer
//! (`docs/PRD.md` §2).
//!
//! The triangle pipeline drew with vertex data only; this module adds the
//! first real resource bindings:
//!
//! - [`QuadUniformBuffer`] — a `UNIFORM | COPY_DST` buffer with queue writes
//!   and an explicitly packed 32-byte layout (two `vec4`s),
//! - [`create_rgba_texture`] / [`create_sampler`] — sRGB textures uploaded
//!   through `write_texture` and nearest/linear samplers,
//! - [`create_quad_bind_layout`] / [`create_quad_bind_group`] — one bind
//!   group: uniform (binding 0), texture (binding 1), sampler (binding 2),
//! - [`create_quad_pipeline`] + [`render_quad_offscreen`] — a vertexless
//!   fullscreen quad (vertices generated from `vertex_index`) sampling the
//!   texture through the uniform's uv window and tint, with a deterministic
//!   pixel contract: with tint white and the full uv window, the readback is
//!   the texture bytes exactly,
//! - [`render_triangle_msaa_offscreen`] — 4× MSAA: draw into a multisampled
//!   attachment and resolve into a single-sample readback target,
//! - [`srgb8_to_linear`] — exact inverse of [`linear_to_srgb8`] so tests can
//!   compute expected tint results per channel.
//!
//! MSAA inside the render graph (per-target sample counts + resolve nodes)
//! lands with the next increment; this module proves the pass structure
//! end to end.

use crate::offscreen::{readback_texture, OffscreenFrame};
use crate::renderer::{RenderError, Renderer};

/// The quad shader: a fullscreen quad (vertices from `vertex_index`) with a
/// uniform-driven uv window and tint over a bound texture.
pub const QUAD_WGSL: &str = r#"
struct Uniforms {
    tint: vec4<f32>,
    // xy = uv offset, zw = uv scale
    uv_window: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    var pos = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    // Corner uv (0,0) at the top-left so texture rows map 1:1 onto
    // readback rows: pixel (x, y) samples texel (x, y).
    var uv = array<vec2<f32>, 6>(
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0),
    );
    var out: VsOut;
    out.position = vec4<f32>(pos[index], 0.0, 1.0);
    out.uv = uv[index] * uniforms.uv_window.zw + uniforms.uv_window.xy;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv) * uniforms.tint;
}
"#;

/// Uniform data for the quad shader, packed little-endian into 32 bytes
/// (two `vec4`s): `tint` then `uv_window` (xy = offset, zw = scale).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadUniform {
    /// RGBA tint multiplied over the sampled texel (in linear space).
    pub tint: [f32; 4],
    /// UV window: offset xy, scale zw — the sub-rect of the texture drawn.
    pub uv_window: [f32; 4],
}

impl QuadUniform {
    /// Tint white, full texture window.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            tint: [1.0, 1.0, 1.0, 1.0],
            uv_window: [0.0, 0.0, 1.0, 1.0],
        }
    }

    /// Pack into the WGSL `Uniforms` layout: two `vec4<f32>`s, 32 bytes.
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        for value in self.tint.iter().chain(self.uv_window.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// A `UNIFORM | COPY_DST` buffer holding one [`QuadUniform`]; `write` pushes
/// updates through the queue without recreating the buffer.
pub struct QuadUniformBuffer {
    buffer: wgpu::Buffer,
}

impl QuadUniformBuffer {
    /// Create the buffer and upload `uniform` immediately.
    #[must_use]
    pub fn new(renderer: &Renderer, uniform: &QuadUniform) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("onyx-quad-uniform"),
                contents: &uniform.bytes(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        Self { buffer }
    }

    /// Overwrite the uniform contents through the queue.
    pub fn write(&self, renderer: &Renderer, uniform: &QuadUniform) {
        renderer
            .queue
            .write_buffer(&self.buffer, 0, &uniform.bytes());
    }

    /// The underlying wgpu buffer (for bind group construction).
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Bind group layout for the quad pipeline: uniform (0), texture (1),
/// sampler (2).
#[must_use]
pub fn create_quad_bind_layout(renderer: &Renderer) -> wgpu::BindGroupLayout {
    renderer
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("onyx-quad-bind-layout"),
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

/// Bind the quad resources: uniform buffer, texture view and sampler into
/// `layout`.
#[must_use]
pub fn create_quad_bind_group(
    renderer: &Renderer,
    layout: &wgpu::BindGroupLayout,
    uniform: &QuadUniformBuffer,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("onyx-quad-bind-group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.buffer().as_entire_binding(),
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

/// Upload `srgb8_pixels` (`width * height * 4` RGBA bytes, row-major) as an
/// `Rgba8UnormSrgb` texture. Rows are uploaded top-first; sampling uv (0,0)
/// hits the first row.
///
/// # Panics
/// If `pixels.len() != (width * height * 4) as usize`.
#[must_use]
pub fn create_rgba_texture(
    renderer: &Renderer,
    width: u32,
    height: u32,
    srgb8_pixels: &[u8],
) -> wgpu::Texture {
    assert_eq!(
        srgb8_pixels.len(),
        (width * height * 4) as usize,
        "texture upload needs width * height * 4 bytes"
    );
    let texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("onyx-rgba-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    renderer.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        srgb8_pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture
}

/// A sampling-only view of `texture`.
#[must_use]
pub fn texture_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// A clamp-to-edge sampler; `nearest` gives texel-exact lookups (the pixel
/// contracts rely on it), otherwise bilinear.
#[must_use]
pub fn create_sampler(renderer: &Renderer, nearest: bool) -> wgpu::Sampler {
    let filter = if nearest {
        wgpu::FilterMode::Nearest
    } else {
        wgpu::FilterMode::Linear
    };
    renderer.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("onyx-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..wgpu::SamplerDescriptor::default()
    })
}

/// Compile the quad shader and create its pipeline (bind group 0) for
/// `format`. No vertex buffers: the quad is generated in the shader.
#[must_use]
pub fn create_quad_pipeline(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let device = &renderer.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("onyx-quad-shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(QUAD_WGSL)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("onyx-quad-pipeline-layout"),
        bind_group_layouts: &[Some(&create_quad_bind_layout(renderer))],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("onyx-quad-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
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

/// Draw the textured quad into an offscreen RGBA8 sRGB texture and read it
/// back.
///
/// Deterministic contract: `width == texture_width && height ==
/// texture_height`, nearest sampling, full uv window, white tint → the
/// readback equals the uploaded texels exactly (the sRGB decode/encode
/// round-trips 8-bit values losslessly).
///
/// # Errors
/// [`RenderError::Poll`] / [`RenderError::Map`] on readback failure.
pub fn render_quad_offscreen(
    renderer: &Renderer,
    width: u32,
    height: u32,
    srgb8_texels: &[u8],
    uniform: &QuadUniform,
) -> Result<OffscreenFrame, RenderError> {
    let device = &renderer.device;
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("onyx-quad-target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let texture = create_rgba_texture(renderer, width, height, srgb8_texels);
    let view = texture_view(&texture);
    let sampler = create_sampler(renderer, true);
    let uniform_buffer = QuadUniformBuffer::new(renderer, uniform);
    let layout = create_quad_bind_layout(renderer);
    let bind_group = create_quad_bind_group(renderer, &layout, &uniform_buffer, &view, &sampler);
    let pipeline = create_quad_pipeline(renderer, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("onyx-quad-encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("onyx-quad-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..wgpu::RenderPassDescriptor::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
    renderer.queue.submit([encoder.finish()]);

    let pixels = readback_texture(renderer, &target, width, height)?;
    Ok(OffscreenFrame {
        width,
        height,
        pixels,
    })
}

/// The triangle pipeline with `samples` per pixel — matches the multisampled
/// attachment in [`render_triangle_msaa_offscreen`].
#[must_use]
pub fn create_triangle_pipeline_msaa(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::RenderPipeline {
    let device = &renderer.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("onyx-triangle-shader-msaa"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            crate::pipeline::TRIANGLE_WGSL,
        )),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("onyx-triangle-layout-msaa"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("onyx-triangle-pipeline-msaa"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(crate::pipeline::Vertex::layout())],
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

/// Render the triangle with `samples`× MSAA: draw into a multisampled
/// attachment, resolve into a single-sample target and read the resolve back.
///
/// Deterministic contract: deep-interior pixels are exactly the triangle
/// color and far-outside pixels are exactly the background — MSAA only
/// changes edge pixels, whose exact values are implementation-defined
/// (sample positions).
///
/// # Errors
/// [`RenderError::Poll`] / [`RenderError::Map`] on readback failure.
pub fn render_triangle_msaa_offscreen(
    renderer: &Renderer,
    width: u32,
    height: u32,
    samples: u32,
    background: [f64; 4],
    triangle: [f64; 4],
) -> Result<OffscreenFrame, RenderError> {
    let device = &renderer.device;
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let resolve = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("onyx-msaa-resolve"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let resolve_view = resolve.create_view(&wgpu::TextureViewDescriptor::default());
    let msaa = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("onyx-msaa-attachment"),
        size: extent,
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let msaa_view = msaa.create_view(&wgpu::TextureViewDescriptor::default());

    let pipeline =
        create_triangle_pipeline_msaa(renderer, wgpu::TextureFormat::Rgba8UnormSrgb, samples);
    let vertex_buffer = crate::pipeline::triangle_vertex_buffer(renderer, triangle);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("onyx-msaa-encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("onyx-msaa-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &msaa_view,
                depth_slice: None,
                resolve_target: Some(&resolve_view),
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

    let pixels = readback_texture(renderer, &resolve, width, height)?;
    Ok(OffscreenFrame {
        width,
        height,
        pixels,
    })
}

/// 8-bit sRGB to linear [0, 1] — the exact inverse of
/// [`crate::offscreen::linear_to_srgb8`] (before its final rounding), so
/// tests can compute expected tinted pixels channel by channel.
#[must_use]
pub fn srgb8_to_linear(value: u8) -> f64 {
    let srgb = f64::from(value) / 255.0;
    if srgb <= 0.040_45 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offscreen::linear_to_srgb8;
    use crate::RendererConfig;

    #[test]
    fn quad_uniform_packs_two_vec4s_little_endian() {
        let uniform = QuadUniform {
            tint: [0.25, 0.5, 1.0, 1.0],
            uv_window: [0.5, 0.25, 2.0, -1.0],
        };
        let bytes = uniform.bytes();
        assert_eq!(bytes.len(), 32);
        for (index, value) in uniform
            .tint
            .iter()
            .chain(uniform.uv_window.iter())
            .enumerate()
        {
            let start = index * 4;
            assert_eq!(&bytes[start..start + 4], &value.to_le_bytes());
        }
    }

    #[test]
    fn srgb8_round_trips_every_value() {
        for value in 0..=255u8 {
            assert_eq!(linear_to_srgb8(srgb8_to_linear(value)), value);
        }
    }

    #[test]
    fn textured_quad_round_trips_texels_exactly() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // Distinct texel values; 8×8 target over an 8×8 texture.
                let texels: Vec<u8> = (0u32..8 * 8 * 4)
                    .map(|i| (((i * 7 + 13) % 251) + 2) as u8)
                    .collect();
                let frame = render_quad_offscreen(&renderer, 8, 8, &texels, &QuadUniform::full())
                    .expect("quad frame");
                assert_eq!(frame.pixels, texels, "tint 1 and full window: pixel-exact");
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn quad_tint_multiplies_in_linear_space() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // Alpha pinned to 255: with tint alpha 1.0 the sampled alpha
                // passes through unchanged on every backend (varying alpha
                // values would make the assertion driver-dependent).
                let channel = |i: u32| (((i * 11 + 29) % 243) + 6) as u8;
                let texels: Vec<u8> = (0u32..8 * 8)
                    .flat_map(|i| {
                        [
                            channel(i * 3),
                            channel(i * 3 + 1),
                            channel(i * 3 + 2),
                            255u8,
                        ]
                    })
                    .collect();
                let uniform = QuadUniform {
                    tint: [1.0, 0.5, 0.25, 1.0],
                    uv_window: [0.0, 0.0, 1.0, 1.0],
                };
                let frame =
                    render_quad_offscreen(&renderer, 8, 8, &texels, &uniform).expect("quad frame");
                for (pixel, texel) in frame.pixels.chunks_exact(4).zip(texels.chunks_exact(4)) {
                    let expected = [
                        linear_to_srgb8(srgb8_to_linear(texel[0])),
                        linear_to_srgb8(srgb8_to_linear(texel[1]) * 0.5),
                        linear_to_srgb8(srgb8_to_linear(texel[2]) * 0.25),
                        texel[3],
                    ];
                    assert_eq!(pixel, &expected, "tint applies in linear space");
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn quad_uv_window_selects_the_bottom_right_quadrant() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                // Four quadrant colors (top-left, top-right, bottom-left,
                // bottom-right) as 2×2 blocks in an 8×8 texture.
                let quadrant = |q: usize| {
                    [
                        (q * 40 + 20) as u8,
                        (q * 30 + 50) as u8,
                        (q * 20 + 80) as u8,
                        255u8,
                    ]
                };
                let mut texels = vec![0u8; 8 * 8 * 4];
                for y in 0..8u32 {
                    for x in 0..8u32 {
                        let q = usize::from(y >= 4) * 2 + usize::from(x >= 4);
                        texels[((y * 8 + x) * 4) as usize..((y * 8 + x) * 4 + 4) as usize]
                            .copy_from_slice(&quadrant(q));
                    }
                }
                // Window = bottom-right quadrant: uv offset (0.5, 0.5),
                // scale (0.5, 0.5).
                let uniform = QuadUniform {
                    tint: [1.0, 1.0, 1.0, 1.0],
                    uv_window: [0.5, 0.5, 0.5, 0.5],
                };
                let frame =
                    render_quad_offscreen(&renderer, 8, 8, &texels, &uniform).expect("quad frame");
                // Pixel (x, y) samples texel (4 + x/2, 4 + y/2) — the
                // bottom-right quadrant stretched over the whole target.
                for y in 0..8u32 {
                    for x in 0..8u32 {
                        let src = ((4 + y / 2) * 8 + 4 + x / 2) as usize * 4;
                        let dst = ((y * 8 + x) * 4) as usize;
                        assert_eq!(
                            &frame.pixels[dst..dst + 4],
                            &texels[src..src + 4],
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
    fn quad_uniform_writes_take_effect_between_draws() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let texels = vec![200u8; 4 * 4 * 4];
                let buffer = QuadUniformBuffer::new(&renderer, &QuadUniform::full());
                buffer.write(
                    &renderer,
                    &QuadUniform {
                        tint: [0.5, 0.5, 0.5, 1.0],
                        uv_window: [0.0, 0.0, 1.0, 1.0],
                    },
                );
                // The write went through the queue; render with the same
                // buffer bound to confirm the new contents are live.
                let target = renderer.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("onyx-quad-uniform-test-target"),
                    size: wgpu::Extent3d {
                        width: 4,
                        height: 4,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let target_view = texture_view(&target);
                let texture = create_rgba_texture(&renderer, 4, 4, &texels);
                let view = texture_view(&texture);
                let sampler = create_sampler(&renderer, true);
                let layout = create_quad_bind_layout(&renderer);
                let bind_group =
                    create_quad_bind_group(&renderer, &layout, &buffer, &view, &sampler);
                let pipeline = create_quad_pipeline(&renderer, wgpu::TextureFormat::Rgba8UnormSrgb);
                let mut encoder = renderer
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..wgpu::RenderPassDescriptor::default()
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.draw(0..6, 0..1);
                }
                renderer.queue.submit([encoder.finish()]);
                let frame = OffscreenFrame {
                    width: 4,
                    height: 4,
                    pixels: readback_texture(&renderer, &target, 4, 4)
                        .expect("uniform-test readback"),
                };
                let expected = linear_to_srgb8(srgb8_to_linear(200) * 0.5);
                for chunk in frame.pixels.chunks_exact(4) {
                    assert_eq!(&chunk[..3], &[expected, expected, expected]);
                }
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn msaa_triangle_keeps_interior_and_corners_exact() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let background = [0.1, 0.1, 0.2, 1.0];
                let triangle = [1.0, 1.0, 1.0, 1.0];
                let frame =
                    render_triangle_msaa_offscreen(&renderer, 64, 64, 4, background, triangle)
                        .expect("msaa frame");
                assert_eq!(frame.pixels.len(), 64 * 64 * 4);

                let center = (32 * 64 + 32) * 4;
                assert_eq!(
                    &frame.pixels[center..center + 4],
                    &[
                        linear_to_srgb8(1.0),
                        linear_to_srgb8(1.0),
                        linear_to_srgb8(1.0),
                        255
                    ],
                    "deep interior must be exactly the triangle color"
                );
                let corner = 0;
                assert_eq!(
                    &frame.pixels[corner..corner + 4],
                    &[
                        linear_to_srgb8(0.1),
                        linear_to_srgb8(0.1),
                        linear_to_srgb8(0.2),
                        255
                    ],
                    "far outside must stay exactly the background"
                );
                // A second deep-interior point (NDC ≈ (0.016, −0.266), well
                // clear of every edge): MSAA only affects edge pixels.
                let interior2 = (40 * 64 + 32) * 4;
                assert_eq!(
                    &frame.pixels[interior2..interior2 + 4],
                    &[255, 255, 255, 255],
                    "deep interior point must resolve to the triangle color"
                );
            }
            Err(RenderError::NoAdapter { .. }) => {}
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }
}
