//! Post-processing screen adjustments and separable blur (`docs/PRD.md` §4,
//! §10).
//!
//! Mirrors Godot's `Environment` adjustments (brightness, contrast,
//! saturation) and Bevy's post-processing node placement: one pass samples
//! the node's declared read resource fullscreen — the same vertexless quad
//! as the textured-quad pass — and applies the three adjustments in a
//! fixed order, in linear space: brightness multiply, contrast around
//! mid-gray 0.5, then saturation toward the Rec. 709 luma. Alpha passes
//! through untouched; identity parameters (1, 1, 1) are a pixel-exact
//! copy of the source.
//!
//! [`adjust`] is the CPU mirror of the shader's `adjust` function with the
//! identical operation sequence, so the adjusted color is predictable bit
//! for bit on every backend.
//!
//! The separable tent blur runs the same fullscreen quad: one pass blends
//! a 3-tap tent — weights `[0.25, 0.5, 0.25]` — along a direction in texel
//! units (the executor scales it to the per-tap UV step of the source).
//! Two chained passes, horizontal then vertical, form the separable blur
//! that glows and blooms build on; [`tent3`] mirrors the shader's blend
//! with the identical operation sequence.

use crate::renderer::{RenderError, Renderer};

/// The adjust shader: one `vec4` uniform (brightness, contrast, saturation,
/// unused), the source texture and sampler, the fullscreen quad of the
/// textured-quad pass, and the fixed-order adjustment chain.
pub const ADJUST_WGSL: &str = r#"
struct Params {
    // x = brightness, y = contrast, z = saturation (identity 1.0)
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

// Rec. 709 luma weights (linear space) — the same literals as [`LUMA_709`].
const LUMA_709 = vec3<f32>(0.2126, 0.7152, 0.0722);

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
    out.uv = uv[index];
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    return vec4<f32>(adjust(texel.rgb), texel.a);
}

fn adjust(rgb: vec3<f32>) -> vec3<f32> {
    // brightness
    var v = rgb * params.params.x;
    // contrast around mid-gray 0.5
    v = (v - vec3<f32>(0.5)) * params.params.y + vec3<f32>(0.5);
    // saturation: mix toward the Rec. 709 luma (the WGSL `mix`
    // expression `x + (y - x) * a`, written out)
    let l = v.x * LUMA_709.x + v.y * LUMA_709.y + v.z * LUMA_709.z;
    v = vec3<f32>(l) + (v - vec3<f32>(l)) * params.params.z;
    return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
}
"#;

/// Rec. 709 luma weights for linear RGB — the same f32 literals as the
/// shader's `LUMA_709` constant.
pub const LUMA_709: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Uniform data for the adjust shader: `(brightness, contrast, saturation,
/// 0)` — one `vec4<f32>`, 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdjustParams {
    /// Brightness multiplier, identity 1.0.
    pub brightness: f32,
    /// Contrast around mid-gray 0.5, identity 1.0.
    pub contrast: f32,
    /// Saturation toward the Rec. 709 luma: identity 1.0, 0 = grayscale.
    pub saturation: f32,
}

impl AdjustParams {
    /// `(1, 1, 1)` — the pixel-exact copy parameters.
    pub const IDENTITY: Self = Self {
        brightness: 1.0,
        contrast: 1.0,
        saturation: 1.0,
    };

    /// Pack into the WGSL `Params` layout: one `vec4<f32>`, 16 bytes.
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        for value in [self.brightness, self.contrast, self.saturation, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// Validate adjust parameters for one node: all three in `[0, 8]` —
/// negative values and NaN are unphysical for screen adjustments, and 8.0
/// leaves generous headroom over the useful range (identity is 1.0).
///
/// # Errors
/// [`RenderError::Unsupported`] naming the first out-of-range parameter.
pub fn validate_adjust_params(
    node: &str,
    brightness: f64,
    contrast: f64,
    saturation: f64,
) -> Result<(), RenderError> {
    for (name, value) in [
        ("brightness", brightness),
        ("contrast", contrast),
        ("saturation", saturation),
    ] {
        if !(0.0..=8.0).contains(&value) {
            return Err(RenderError::Unsupported {
                message: format!(
                    "node '{node}' ({name} = {value}) out of range: \
                     adjust parameters must be within [0, 8]"
                ),
            });
        }
    }
    Ok(())
}

/// The CPU mirror of the shader's `adjust`: brightness multiply, contrast
/// around mid-gray, Rec. 709 saturation mix (the `x + (y - x) * a` form),
/// clamp — with the identical operation sequence, so the GPU output is
/// predictable bit for bit.
#[must_use]
pub fn adjust(rgb: [f32; 3], brightness: f32, contrast: f32, saturation: f32) -> [f32; 3] {
    let mut v = [
        rgb[0] * brightness,
        rgb[1] * brightness,
        rgb[2] * brightness,
    ];
    for channel in &mut v {
        *channel = (*channel - 0.5) * contrast + 0.5;
    }
    let luma = v[0] * LUMA_709[0] + v[1] * LUMA_709[1] + v[2] * LUMA_709[2];
    for channel in &mut v {
        *channel = luma + (*channel - luma) * saturation;
    }
    [
        v[0].clamp(0.0, 1.0),
        v[1].clamp(0.0, 1.0),
        v[2].clamp(0.0, 1.0),
    ]
}

/// A `UNIFORM | COPY_DST` buffer holding one [`AdjustParams`].
pub struct AdjustUniformBuffer {
    buffer: wgpu::Buffer,
}

impl AdjustUniformBuffer {
    /// Create the buffer and upload `params` immediately.
    #[must_use]
    pub fn new(renderer: &Renderer, params: &AdjustParams) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("onyx-adjust-params"),
                contents: &params.bytes(),
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

/// Bind group layout for the adjust pipeline: params uniform (0), texture
/// (1), sampler (2) — the quad layout's shape.
#[must_use]
pub fn create_adjust_bind_layout(renderer: &Renderer) -> wgpu::BindGroupLayout {
    renderer
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("onyx-adjust-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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

/// The adjust bind group: params uniform (0), texture (1), sampler (2).
#[must_use]
pub fn create_adjust_bind_group(
    renderer: &Renderer,
    layout: &wgpu::BindGroupLayout,
    params: &AdjustUniformBuffer,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("onyx-adjust-bind-group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.buffer().as_entire_binding(),
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

/// Compile the adjust shader and create its pipeline (bind group 0) for
/// `(samples, format)`: the vertexless fullscreen quad provides the
/// geometry, one draw call covers the target.
#[must_use]
pub fn create_adjust_pipeline(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::RenderPipeline {
    let device = &renderer.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("onyx-adjust-shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ADJUST_WGSL)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("onyx-adjust-pipeline-layout"),
        bind_group_layouts: &[Some(&create_adjust_bind_layout(renderer))],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("onyx-adjust-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
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

/// The blur shader: one `vec4` uniform carrying the per-tap UV step, the
/// source texture and sampler, the fullscreen quad of the textured-quad
/// pass, and the 3-tap tent blend over the three samples (all four
/// channels, so opaque alpha stays exactly opaque).
pub const BLUR_WGSL: &str = r#"
struct Params {
    // xy = UV offset per tap (one texel for the standard separable passes)
    step: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: Params;
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
    out.uv = uv[index];
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let s0 = textureSample(tex, samp, in.uv - params.step.xy);
    let s1 = textureSample(tex, samp, in.uv);
    let s2 = textureSample(tex, samp, in.uv + params.step.xy);
    // 3-tap tent: [0.25, 0.5, 0.25]
    return s0 * 0.25 + s1 * 0.5 + s2 * 0.25;
}
"#;

/// Uniform data for the blur shader: `(step.x, step.y, 0, 0)` — one
/// `vec4<f32>`, 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlurParams {
    /// UV offset per tap (one texel for the standard separable passes).
    pub step: [f32; 2],
}

impl BlurParams {
    /// Pack into the WGSL `Params` layout: one `vec4<f32>`, 16 bytes.
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        for value in [self.step[0], self.step[1], 0.0, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// Validate blur parameters for one node: the direction must be non-zero
/// (a blur that does not move is a copy, not a blur) with both components
/// finite and within `[-8, 8]` texels per tap.
///
/// # Errors
/// [`RenderError::Unsupported`] naming the first invalid component.
pub fn validate_blur_params(node: &str, direction: [f64; 2]) -> Result<(), RenderError> {
    if direction == [0.0, 0.0] {
        return Err(RenderError::Unsupported {
            message: format!("node '{node}' blurs along the zero direction: a blur pass must move"),
        });
    }
    for (name, value) in [("x", direction[0]), ("y", direction[1])] {
        if !(-8.0..=8.0).contains(&value) {
            return Err(RenderError::Unsupported {
                message: format!(
                    "node '{node}' (direction {name} = {value}) out of range: \
                     blur direction components must be within [-8, 8]"
                ),
            });
        }
    }
    Ok(())
}

/// The CPU mirror of the shader's 3-tap tent blend: per channel,
/// `a * 0.25 + b * 0.5 + c * 0.25` — the identical operation sequence, so
/// a blur of known texels is predictable bit for bit.
#[must_use]
pub fn tent3(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for k in 0..4 {
        out[k] = a[k] * 0.25 + b[k] * 0.5 + c[k] * 0.25;
    }
    out
}

/// A `UNIFORM | COPY_DST` buffer holding one [`BlurParams`].
pub struct BlurUniformBuffer {
    buffer: wgpu::Buffer,
}

impl BlurUniformBuffer {
    /// Create the buffer and upload `params` immediately.
    #[must_use]
    pub fn new(renderer: &Renderer, params: &BlurParams) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("onyx-blur-params"),
                contents: &params.bytes(),
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

/// Bind group layout for the blur pipeline: params uniform (0), texture
/// (1), sampler (2) — the quad layout's shape.
#[must_use]
pub fn create_blur_bind_layout(renderer: &Renderer) -> wgpu::BindGroupLayout {
    renderer
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("onyx-blur-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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

/// The blur bind group: params uniform (0), texture (1), sampler (2).
#[must_use]
pub fn create_blur_bind_group(
    renderer: &Renderer,
    layout: &wgpu::BindGroupLayout,
    params: &BlurUniformBuffer,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("onyx-blur-bind-group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.buffer().as_entire_binding(),
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

/// Compile the blur shader and create its pipeline (bind group 0) for
/// `(samples, format)`: the vertexless fullscreen quad provides the
/// geometry, one draw call covers the target.
#[must_use]
pub fn create_blur_pipeline(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::RenderPipeline {
    let device = &renderer.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("onyx-blur-shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BLUR_WGSL)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("onyx-blur-pipeline-layout"),
        bind_group_layouts: &[Some(&create_blur_bind_layout(renderer))],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("onyx-blur-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
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
    fn luma_weights_sum_to_one() {
        let sum = f64::from(LUMA_709[0]) + f64::from(LUMA_709[1]) + f64::from(LUMA_709[2]);
        assert!((sum - 1.0).abs() < 1e-6, "luma weights sum to {sum}");
    }

    #[test]
    fn params_bytes_pack_one_vec4() {
        let params = AdjustParams {
            brightness: 1.25,
            contrast: 0.75,
            saturation: 0.5,
        };
        let bytes = params.bytes();
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[0..4], &1.25f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &0.75f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &0.5f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0.0f32.to_le_bytes());
    }

    #[test]
    fn validation_accepts_the_range_and_rejects_the_rest() {
        assert!(validate_adjust_params("chip", 0.0, 1.0, 8.0).is_ok());
        for (brightness, contrast, saturation) in
            [(-0.1, 1.0, 1.0), (1.0, 8.1, 1.0), (1.0, 1.0, f64::NAN)]
        {
            let error =
                validate_adjust_params("chip", brightness, contrast, saturation).expect_err("out");
            assert!(error.to_string().contains("[0, 8]"), "{error}");
        }
    }

    #[test]
    fn adjust_halves_brightness_exactly() {
        // ×0.5 with identity contrast and grayscale output (saturation 0
        // is exact: `luma + (v - luma) * 0.0` collapses to `luma`).
        let out = adjust([0.5, 0.25, 0.75], 0.5, 1.0, 0.0);
        for channel in out {
            let luma = 0.25 * 0.2126 + 0.125 * 0.7152 + 0.375 * 0.0722;
            assert!(
                (f64::from(channel) - luma).abs() < 1e-6,
                "channel {channel} vs luma {luma}"
            );
        }
    }

    #[test]
    fn adjust_doubles_contrast_exactly() {
        // (0.75 - 0.5) * 2.0 + 0.5 = 1.0 exactly; (0.25 - 0.5) * 2.0 +
        // 0.5 = 0.0 exactly; mid-gray stays. Grayscale output again.
        let out = adjust([0.75, 0.5, 0.25], 1.0, 2.0, 0.0);
        let luma = 1.0 * 0.2126 + 0.5 * 0.7152;
        for channel in out {
            assert!(
                (f64::from(channel) - luma).abs() < 1e-6,
                "channel {channel} vs luma {luma}"
            );
        }
    }

    #[test]
    fn adjust_clamps_to_the_target_range() {
        // brightness 2 doubles everything first: 0.75 → 1.5 (clamps to
        // exactly 1.0), 0.5 → 1.0 (stays at the boundary), 0.25 → 0.5.
        let out = adjust([0.75, 0.5, 0.25], 2.0, 1.0, 1.0);
        assert_eq!(out[0], 1.0);
        assert!((f64::from(out[1]) - 1.0).abs() < 1e-6, "out: {out:?}");
        assert!((f64::from(out[2]) - 0.5).abs() < 1e-6, "out: {out:?}");
    }

    #[test]
    fn adjust_keeps_grays_unchanged() {
        // A gray input has one value per channel, so every stage of the
        // chain (including the luma mix) returns that value.
        for gray in [0.1f32, 0.25, 0.5, 0.9] {
            let out = adjust([gray, gray, gray], 0.8, 1.5, 0.5);
            for channel in out {
                assert!(
                    (channel - out[0]).abs() < 1e-7,
                    "gray {gray} stays gray: {out:?}"
                );
            }
        }
    }

    #[test]
    fn tent3_blends_dyadic_values_exactly() {
        // 0.25 * 0.5 + 0.5 * 1.0 + 0.25 * 0.25 = 0.6875 exactly in f32.
        let out = tent3([0.5; 4], [1.0; 4], [0.25; 4]);
        assert_eq!(out, [0.6875; 4]);
    }

    #[test]
    fn tent3_of_equal_texels_returns_them() {
        // The weights sum to 1.0 — 0.25 + 0.5 + 0.25 is exact in f32 — and
        // dyadic texel values keep every step of the sum exact.
        for value in [0.25f32, 0.5, 1.0] {
            assert_eq!(tent3([value; 4], [value; 4], [value; 4]), [value; 4]);
        }
    }

    #[test]
    fn blur_params_bytes_pack_step_and_zeros() {
        let params = BlurParams {
            step: [0.125, 0.0625],
        };
        let bytes = params.bytes();
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[0..4], &0.125f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &0.0625f32.to_le_bytes());
        assert_eq!(&bytes[8..16], &[0u8; 8]);
    }

    #[test]
    fn blur_validation_accepts_directions_and_rejects_the_rest() {
        for direction in [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-8.0, 0.0]] {
            assert!(
                validate_blur_params("hpass", direction).is_ok(),
                "direction {direction:?}"
            );
        }
        let error = validate_blur_params("hpass", [0.0, 0.0]).expect_err("zero");
        assert!(error.to_string().contains("must move"), "{error}");
        for direction in [[f64::NAN, 0.0], [8.5, 0.0], [0.0, -9.0]] {
            let error = validate_blur_params("hpass", direction).expect_err("range");
            assert!(error.to_string().contains("[-8, 8]"), "{error}");
        }
    }
}
