//! Offscreen clear-pass rendering with GPU readback (`docs/PRD.md` §2).
//!
//! The headless validation path for the M2 device layer: render a clear pass
//! into an offscreen RGBA8 sRGB texture, copy it into a `MAP_READ` buffer,
//! map it and repack the rows (256-byte alignment padding is stripped). This
//! exercises device → encoder → submit → poll → map end to end without a
//! window or presentable surface, and gives tests a deterministic pixel
//! contract: the readback bytes must equal the requested clear color under
//! the sRGB transfer function.

use crate::renderer::{RenderError, Renderer};

/// A finished offscreen frame: RGBA8 sRGB bytes, row-major, no padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffscreenFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// `width * height * 4` bytes, RGBA order.
    pub pixels: Vec<u8>,
}

impl OffscreenFrame {
    /// Deterministic FNV-1a checksum of the pixel bytes — a stable signature
    /// for regression tests and the `onyx render` CLI probe.
    #[must_use]
    pub fn checksum(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &byte in &self.pixels {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        }
        hash
    }
}

/// Linear [0, 1] to 8-bit sRGB, matching the GPU's `Rgba8UnormSrgb` encoding.
#[must_use]
pub fn linear_to_srgb8(value: f64) -> u8 {
    let srgb = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Clear an offscreen texture to `color` and read the pixels back.
///
/// # Errors
/// [`RenderError::Poll`] if awaiting the submission fails,
/// [`RenderError::Map`] if the readback buffer cannot be mapped.
pub fn render_offscreen(
    renderer: &Renderer,
    width: u32,
    height: u32,
    color: [f64; 4],
) -> Result<OffscreenFrame, RenderError> {
    let device = &renderer.device;
    let queue = &renderer.queue;
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("onyx-offscreen"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("onyx-encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("onyx-clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
        // A clear-only pass: no draw calls yet. M2's next increment attaches
        // real pipelines here.
        let _ = &mut pass;
    }

    // Readback rows are aligned to 256 bytes; repack strips the padding.
    let unpadded_row = u64::from(width) * 4;
    let bytes_per_row = (width * 4).div_ceil(256) * 256;
    let buffer_size = u64::from(bytes_per_row) * u64::from(height);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("onyx-readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        extent,
    );

    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .map_err(|error| RenderError::Poll {
            message: error.to_string(),
        })?;
    receiver
        .recv()
        .map_err(|error| RenderError::Map {
            message: error.to_string(),
        })?
        .map_err(|error| RenderError::Map {
            message: error.to_string(),
        })?;

    let mapping = readback
        .get_mapped_range(..)
        .map_err(|error| RenderError::Map {
            message: error.to_string(),
        })?;
    let mut pixels = Vec::with_capacity(buffer_size as usize);
    for row in 0..u64::from(height) {
        let start = (row * u64::from(bytes_per_row)) as usize;
        pixels.extend_from_slice(&mapping[start..start + unpadded_row as usize]);
    }
    drop(mapping);
    readback.unmap();

    Ok(OffscreenFrame {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RendererConfig;

    #[test]
    fn srgb_conversion_matches_the_gpu_transfer_function() {
        assert_eq!(linear_to_srgb8(0.0), 0);
        assert_eq!(linear_to_srgb8(1.0), 255);
        // 0.5 linear ≈ 188 sRGB (188.07 rounds to 188).
        assert_eq!(linear_to_srgb8(0.5), 188);
        assert_eq!(linear_to_srgb8(0.25), 137);
    }

    #[test]
    fn checksum_is_deterministic_and_content_sensitive() {
        let frame = OffscreenFrame {
            width: 1,
            height: 1,
            pixels: vec![1, 2, 3, 4],
        };
        let again = frame.clone();
        assert_eq!(frame.checksum(), again.checksum());
        let changed = OffscreenFrame {
            pixels: vec![1, 2, 3, 5],
            ..frame.clone()
        };
        assert_ne!(frame.checksum(), changed.checksum());
    }

    #[test]
    fn offscreen_clear_matches_requested_color_when_gpu_is_available() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let frame = render_offscreen(&renderer, 5, 2, [0.25, 0.5, 0.75, 1.0])
                    .expect("offscreen frame");
                assert_eq!(frame.width, 5);
                assert_eq!(frame.height, 2);
                assert_eq!(frame.pixels.len(), 5 * 2 * 4);
                let expected = [
                    linear_to_srgb8(0.25),
                    linear_to_srgb8(0.5),
                    linear_to_srgb8(0.75),
                    255,
                ];
                for chunk in frame.pixels.chunks_exact(4) {
                    assert_eq!(chunk, &expected, "every pixel must equal the clear color");
                }
            }
            Err(RenderError::NoAdapter { .. }) => {
                // Headless CI: nothing to assert — the renderer test covers
                // the graceful path.
            }
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }
}
