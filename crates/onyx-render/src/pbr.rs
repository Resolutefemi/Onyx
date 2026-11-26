//! Forward PBR: the Cook-Torrance material pass (`docs/PRD.md` §3).
//!
//! The pass shades a fullscreen plane facing the camera (`N = V = +Z`) under
//! one directional light — the "material chip" view. Because the geometry is
//! fixed and camera-facing, every dot product is uniform per pass, so the
//! shaded color is exactly predictable: [`pbr_shaded`] is the CPU mirror of
//! the WGSL fragment shader, written with the *identical operation sequence*.
//!
//! Both use only exactly-rounded IEEE f32 operations (`+`, `-`, `*`, `/`,
//! `max`, `min`/`clamp`, and fifth powers as five multiplications — no
//! `pow`, no `normalize`), so every backend (Vulkan, Metal, DX12, GL)
//! computes the same bits, and tests can compare the GPU output against the
//! mirror byte for byte.
//!
//! The BRDF is Cook-Torrance with the GGX distribution (D), the Schlick
//! visibility approximation (V, the `k` formulation), Schlick fresnel (F,
//! metallic-tinted F0), Lambertian diffuse (metals have none) and a
//! metallic-scaled ambient term.

use crate::renderer::{RenderError, Renderer};

/// π as f32, the exact literal mirrored in [`PBR_WGSL`].
const PI_F32: f32 = std::f32::consts::PI;

/// The PBR shader: a vertexless fullscreen quad (generated from
/// `vertex_index`) shaded by the Cook-Torrance BRDF. The fragment math
/// mirrors [`pbr_shaded`] operation for operation.
pub const PBR_WGSL: &str = r#"
struct PbrParams {
    albedo: vec4<f32>,
    light: vec4<f32>,
    ambient: vec4<f32>,
    scalars: vec4<f32>,
    dots: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: PbrParams;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    var out: VertexOut;
    let xy = quad[vertex_index];
    out.position = vec4<f32>(xy, 0.0, 1.0);
    return out;
}

fn shade_channel(albedo: f32, light_color: f32, ambient: f32,
                 roughness: f32, metallic: f32,
                 ndotl: f32, ndotv: f32, ndoth: f32, vdoth: f32) -> f32 {
    // The exact operation sequence of `onyx_render::pbr::pbr_shaded` —
    // only exactly-rounded IEEE f32 operations.
    var a = roughness * roughness;
    a = max(a, 0.0001);
    let a2 = a * a;
    // D — the GGX distribution.
    let d_denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    let d = a2 / (3.14159274101257324 * d_denom * d_denom);
    // V — Schlick visibility (the k formulation).
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let v = 0.25 / ((ndotv * (1.0 - k) + k) * (ndotl * (1.0 - k) + k));
    // F0 — dielectric 0.04 base, albedo-tinted for metals.
    let f0 = 0.04 * (1.0 - metallic) + albedo * metallic;
    // F — Schlick fresnel, the fifth power as five exact multiplications.
    let f_base = 1.0 - vdoth;
    let f_pow5 = f_base * f_base * f_base * f_base * f_base;
    let f = f0 + (1.0 - f0) * f_pow5;
    // Lambertian diffuse (metals have no diffuse) + Cook-Torrance specular.
    let diffuse = albedo * (1.0 - metallic) / 3.14159274101257324;
    let specular = d * v * f;
    var lo = (diffuse * ndotl + specular * ndotl) * light_color
        + ambient * albedo * (1.0 - metallic);
    lo = clamp(lo, 0.0, 1.0);
    return lo;
}

@fragment
fn fs(_in: VertexOut) -> @location(0) vec4<f32> {
    let r = shade_channel(params.albedo.r, params.light.r, params.ambient.r,
        params.scalars.x, params.scalars.y, params.scalars.z, params.scalars.w,
        params.dots.x, params.dots.y);
    let g = shade_channel(params.albedo.g, params.light.g, params.ambient.g,
        params.scalars.x, params.scalars.y, params.scalars.z, params.scalars.w,
        params.dots.x, params.dots.y);
    let b = shade_channel(params.albedo.b, params.light.b, params.ambient.b,
        params.scalars.x, params.scalars.y, params.scalars.z, params.scalars.w,
        params.dots.x, params.dots.y);
    return vec4<f32>(r, g, b, 1.0);
}
"#;

/// The GPU-ready PBR parameters: material, light and the precomputed dot
/// products of the fixed camera-facing geometry (`N = V = +Z`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PbrParams {
    /// Surface albedo (base color), linear, components in [0, 1].
    pub albedo: [f32; 3],
    /// Roughness in [0, 1]; the GGX α is `roughness * roughness`.
    pub roughness: f32,
    /// Metallic in [0, 1]: 0 = dielectric (diffuse albedo tint, F0 = 0.04),
    /// 1 = metal (no diffuse, F0 tinted by albedo).
    pub metallic: f32,
    /// Light color, linear, components in [0, 1].
    pub light_color: [f32; 3],
    /// Ambient term, linear, components in [0, 1].
    pub ambient: [f32; 3],
    /// `N · L` with the normalized light direction.
    pub ndotl: f32,
    /// `N · V` (always 1 for the camera-facing plane).
    pub ndotv: f32,
    /// `N · H` with `H = normalize(L + V)`.
    pub ndoth: f32,
    /// `V · H`.
    pub vdoth: f32,
}

impl PbrParams {
    /// Build the parameters from pass inputs: the light direction is
    /// normalized in f32 (the exact ops the shader would do), and the dot
    /// products of the fixed geometry (`N = V = +Z`) are precomputed here so
    /// the shader does only exactly-rounded scalar arithmetic.
    #[must_use]
    pub fn new(
        albedo: [f64; 3],
        roughness: f64,
        metallic: f64,
        light_dir: [f64; 3],
        light_color: [f64; 3],
        ambient: [f64; 3],
    ) -> Self {
        let l = [
            light_dir[0] as f32,
            light_dir[1] as f32,
            light_dir[2] as f32,
        ];
        let l_len = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        let l_inv = 1.0 / l_len;
        let l = [l[0] * l_inv, l[1] * l_inv, l[2] * l_inv];
        // H = normalize(L + V) with V = +Z.
        let h = [l[0], l[1], l[2] + 1.0];
        let h_len = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
        let h_inv = 1.0 / h_len;
        let h = [h[0] * h_inv, h[1] * h_inv, h[2] * h_inv];
        Self {
            albedo: [albedo[0] as f32, albedo[1] as f32, albedo[2] as f32],
            roughness: roughness as f32,
            metallic: metallic as f32,
            light_color: [
                light_color[0] as f32,
                light_color[1] as f32,
                light_color[2] as f32,
            ],
            ambient: [ambient[0] as f32, ambient[1] as f32, ambient[2] as f32],
            ndotl: l[2],
            ndotv: 1.0,
            ndoth: h[2],
            vdoth: h[2],
        }
    }

    /// The uniform bytes: five `vec4<f32>` rows (80 bytes) matching the WGSL
    /// `PbrParams` layout, little-endian.
    #[must_use]
    pub fn uniform_bytes(&self) -> Vec<u8> {
        let rows = self.as_uniform();
        let mut bytes = Vec::with_capacity(80);
        for row in &rows {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    /// The uniform as five `vec4<f32>` rows:
    /// `[albedo, light, ambient, [roughness, metallic, ndotl, ndotv],
    /// [ndoth, vdoth, 0, 0]]`.
    #[must_use]
    pub fn as_uniform(&self) -> [[f32; 4]; 5] {
        [
            [self.albedo[0], self.albedo[1], self.albedo[2], 0.0],
            [
                self.light_color[0],
                self.light_color[1],
                self.light_color[2],
                0.0,
            ],
            [self.ambient[0], self.ambient[1], self.ambient[2], 0.0],
            [self.roughness, self.metallic, self.ndotl, self.ndotv],
            [self.ndoth, self.vdoth, 0.0, 0.0],
        ]
    }
}

/// The CPU mirror of the PBR fragment shader: the shaded linear color of the
/// camera-facing plane under the directional light. Operation-for-operation
/// identical to [`PBR_WGSL`]'s `shade_channel` — only exactly-rounded IEEE
/// f32 operations, so the GPU and this function produce the same bits on
/// every backend.
#[must_use]
pub fn pbr_shaded(params: &PbrParams) -> [f32; 3] {
    let mut a = params.roughness * params.roughness;
    a = a.max(0.0001);
    let a2 = a * a;
    // D — the GGX distribution.
    let d_denom = params.ndoth * params.ndoth * (a2 - 1.0) + 1.0;
    let d = a2 / (PI_F32 * d_denom * d_denom);
    // V — Schlick visibility (the k formulation).
    let k = (params.roughness + 1.0) * (params.roughness + 1.0) / 8.0;
    let v = 0.25 / ((params.ndotv * (1.0 - k) + k) * (params.ndotl * (1.0 - k) + k));
    let mut out = [0.0f32; 3];
    for channel in 0..3 {
        let albedo = params.albedo[channel];
        // F0 — dielectric 0.04 base, albedo-tinted for metals.
        let f0 = 0.04 * (1.0 - params.metallic) + albedo * params.metallic;
        // F — Schlick fresnel, the fifth power as five exact multiplications.
        let f_base = 1.0 - params.vdoth;
        let f_pow5 = f_base * f_base * f_base * f_base * f_base;
        let f = f0 + (1.0 - f0) * f_pow5;
        // Lambertian diffuse (metals have no diffuse) + Cook-Torrance specular.
        let diffuse = albedo * (1.0 - params.metallic) / PI_F32;
        let specular = d * v * f;
        let mut lo = (diffuse * params.ndotl + specular * params.ndotl)
            * params.light_color[channel]
            + params.ambient[channel] * albedo * (1.0 - params.metallic);
        lo = lo.clamp(0.0, 1.0);
        out[channel] = lo;
    }
    out
}

/// Validate PBR pass parameters before any GPU work: color components and
/// material parameters stay within [0, 1], and the light direction is not
/// the zero vector. Deterministic — both executors reject bad passes up
/// front.
///
/// # Errors
/// [`RenderError::Unsupported`] naming the offending parameter.
pub fn validate_pbr_params(
    node: &str,
    albedo: [f64; 3],
    roughness: f64,
    metallic: f64,
    light_dir: [f64; 3],
    light_color: [f64; 3],
    ambient: [f64; 3],
) -> Result<(), RenderError> {
    fn unit_interval(node: &str, param: &str, value: f64) -> Result<(), RenderError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(RenderError::Unsupported {
                message: format!("Pbr node '{node}': {param} = {value} is outside [0, 1]"),
            });
        }
        Ok(())
    }
    for (index, component) in albedo.iter().enumerate() {
        unit_interval(node, &format!("albedo[{index}]"), *component)?;
    }
    for (index, component) in light_color.iter().enumerate() {
        unit_interval(node, &format!("light_color[{index}]"), *component)?;
    }
    for (index, component) in ambient.iter().enumerate() {
        unit_interval(node, &format!("ambient[{index}]"), *component)?;
    }
    unit_interval(node, "roughness", roughness)?;
    unit_interval(node, "metallic", metallic)?;
    let length_sq =
        light_dir[0] * light_dir[0] + light_dir[1] * light_dir[1] + light_dir[2] * light_dir[2];
    if length_sq == 0.0 {
        return Err(RenderError::Unsupported {
            message: format!("Pbr node '{node}': light_dir is the zero vector"),
        });
    }
    Ok(())
}

/// A `UNIFORM | COPY_DST` buffer holding one [`PbrParams`]' uniform bytes.
pub struct PbrUniformBuffer {
    buffer: wgpu::Buffer,
}

impl PbrUniformBuffer {
    /// Create the buffer and upload `params` immediately.
    #[must_use]
    pub fn new(renderer: &Renderer, params: &PbrParams) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("onyx-pbr-uniform"),
                contents: &params.uniform_bytes(),
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

/// Bind group layout for the PBR pipeline: one uniform at binding 0.
#[must_use]
pub fn create_pbr_bind_layout(renderer: &Renderer) -> wgpu::BindGroupLayout {
    renderer
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("onyx-pbr-bind-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
}

/// The PBR bind group: the uniform buffer at binding 0.
#[must_use]
pub fn create_pbr_bind_group(
    renderer: &Renderer,
    layout: &wgpu::BindGroupLayout,
    uniform: &PbrUniformBuffer,
) -> wgpu::BindGroup {
    renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("onyx-pbr-bind-group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.buffer().as_entire_binding(),
            }],
        })
}

/// Compile the PBR shader and create its pipeline (bind group 0) for
/// `(samples, format)`. No vertex buffers: the quad is generated in the
/// shader, like the textured-quad pipeline.
#[must_use]
pub fn create_pbr_pipeline(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::RenderPipeline {
    let device = &renderer.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("onyx-pbr-shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(PBR_WGSL)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("onyx-pbr-pipeline-layout"),
        bind_group_layouts: &[Some(&create_pbr_bind_layout(renderer))],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("onyx-pbr-pipeline"),
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
    fn shaded_light_with_no_ndotl_is_ambient_times_albedo() {
        // Light perpendicular to the surface: ndotl = 0 kills diffuse and
        // specular; only ambient * albedo * (1 - metallic) remains.
        let params = PbrParams {
            albedo: [0.5, 0.25, 1.0],
            roughness: 0.4,
            metallic: 0.0,
            light_color: [1.0; 3],
            ambient: [0.25, 0.5, 0.75],
            ndotl: 0.0,
            ndotv: 1.0,
            ndoth: 1.0,
            vdoth: 1.0,
        };
        let shaded = pbr_shaded(&params);
        let expected = [0.25f32 * 0.5f32, 0.5f32 * 0.25f32, 0.75f32 * 1.0f32];
        assert_eq!(shaded, expected);
    }

    #[test]
    fn shaded_metal_with_no_ndotl_is_black() {
        // Metals have no diffuse and no ambient; ndotl = 0 kills specular.
        let params = PbrParams {
            albedo: [0.9, 0.5, 0.2],
            roughness: 0.2,
            metallic: 1.0,
            light_color: [1.0; 3],
            ambient: [0.5; 3],
            ndotl: 0.0,
            ndotv: 1.0,
            ndoth: 1.0,
            vdoth: 1.0,
        };
        assert_eq!(pbr_shaded(&params), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn shaded_clamps_to_unit_range() {
        // The mirror is pure math: out-of-range light colors clamp to 1
        // (pass parameters are validated, the mirror still guards itself).
        let params = PbrParams {
            albedo: [1.0; 3],
            roughness: 0.5,
            metallic: 0.0,
            light_color: [5.0; 3],
            ambient: [0.0; 3],
            ndotl: 1.0,
            ndotv: 1.0,
            ndoth: 1.0,
            vdoth: 1.0,
        };
        assert_eq!(pbr_shaded(&params), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn params_normalize_light_and_precompute_dots_in_f32() {
        let params = PbrParams::new(
            [0.7, 0.5, 0.3],
            0.35,
            0.25,
            [1.0, 1.0, 1.0],
            [1.0, 0.9, 0.8],
            [0.1, 0.1, 0.15],
        );
        // The exact normalization the executor performs — f32 ops.
        let l = [1.0f32, 1.0, 1.0];
        let l_len = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        let l_inv = 1.0 / l_len;
        assert_eq!(params.ndotl, l[2] * l_inv);
        assert_eq!(params.ndotv, 1.0);
        let h = [l[0] * l_inv, l[1] * l_inv, l[2] * l_inv + 1.0];
        let h_len = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
        let h_inv = 1.0 / h_len;
        assert_eq!(params.ndoth, h[2] * h_inv);
        assert_eq!(params.vdoth, params.ndoth);
    }

    #[test]
    fn uniform_bytes_are_five_vec4_rows_little_endian() {
        let params = PbrParams {
            albedo: [1.0, 0.5, 0.25],
            roughness: 0.4,
            metallic: 0.9,
            light_color: [0.8, 0.6, 0.4],
            ambient: [0.1, 0.2, 0.3],
            ndotl: 0.5,
            ndotv: 1.0,
            ndoth: 0.75,
            vdoth: 0.75,
        };
        let bytes = params.uniform_bytes();
        assert_eq!(bytes.len(), 80);
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &0.5f32.to_le_bytes());
        assert_eq!(&bytes[48..52], &0.4f32.to_le_bytes()); // roughness
        assert_eq!(&bytes[64..68], &0.75f32.to_le_bytes()); // ndoth
        assert_eq!(&bytes[76..80], &0.0f32.to_le_bytes()); // pad
    }

    #[test]
    fn validate_rejects_out_of_range_parameters() {
        let good = |albedo, roughness, metallic, light_dir| {
            validate_pbr_params(
                "node", albedo, roughness, metallic, light_dir, [1.0; 3], [0.1; 3],
            )
        };
        assert!(
            good([0.7, 0.5, 0.3], 0.35, 0.25, [1.0, 1.0, 1.0]).is_ok(),
            "in-range parameters pass"
        );
        for (error, param) in [
            (
                good([2.0, 0.5, 0.3], 0.35, 0.25, [1.0, 1.0, 1.0]),
                "albedo[0]",
            ),
            (
                good([0.7, 0.5, 0.3], 1.5, 0.25, [1.0, 1.0, 1.0]),
                "roughness",
            ),
            (
                good([0.7, 0.5, 0.3], 0.35, -0.1, [1.0, 1.0, 1.0]),
                "metallic",
            ),
            (
                good([0.7, 0.5, 0.3], 0.35, 0.25, [0.0, 0.0, 0.0]),
                "zero vector",
            ),
        ] {
            let error = error.expect_err("out-of-range must fail");
            assert!(error.to_string().contains(param), "{error}");
        }
    }
}
