//! Windowed present: swapchain policy and the surface renderer
//! (`docs/PRD.md` §2).
//!
//! The executor already renders into any declared target; this module wires
//! that machinery to the swapchain:
//!
//! - [`choose_swapchain_format`] / [`choose_present_mode`] — the
//!   deterministic swapchain policy: prefer an sRGB format (so the pixel
//!   contracts of `offscreen` and `resources` carry over to the window) and
//!   Fifo present mode (vsync — the one mode every adapter must support),
//! - [`SurfaceRenderer`] — owns a `wgpu::Surface` created from a
//!   caller-provided window handle ([`wgpu::SurfaceTargetUnsafe`], e.g. from
//!   `winit` via `SurfaceTargetUnsafe::from_window`), configures it with the
//!   policy, and per frame: acquires the swapchain texture, renders the
//!   graph with the `backbuffer` node's target bound to that texture
//!   ([`crate::executor::execute_with_external_target`]) and presents.
//!
//! The swapchain frame is never read back — present is the readback's
//! replacement. On GPU-less machines no surface can be created; the engine
//! stays headless and offscreen rendering keeps working.

use crate::executor::{execute_with_external_target, ExternalTarget, PassKind, ResourceSpec};
use crate::graph::RenderGraph;
use crate::renderer::{RenderError, Renderer};

/// Pick the swapchain format from the surface's supported list: the first
/// sRGB-capable format, falling back to the surface's first preference.
/// `None` only when the surface offers no formats (never in practice).
#[must_use]
pub fn choose_swapchain_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    formats
        .iter()
        .find(|format| format.is_srgb())
        .or_else(|| formats.first())
        .copied()
}

/// Pick the present mode from the surface's supported list: Fifo (vsync —
/// the one mode every adapter must support) when offered, otherwise the
/// surface's first preference. Deterministic over any capability list.
#[must_use]
pub fn choose_present_mode(modes: &[wgpu::PresentMode]) -> Option<wgpu::PresentMode> {
    if modes.contains(&wgpu::PresentMode::Fifo) {
        Some(wgpu::PresentMode::Fifo)
    } else {
        modes.first().copied()
    }
}

/// Windowed rendering on top of the graph executor: one swapchain, one
/// designated `backbuffer` node, present per frame.
pub struct SurfaceRenderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

impl SurfaceRenderer {
    /// Create a surface from a raw window handle, negotiate format/present
    /// mode with [`choose_swapchain_format`] / [`choose_present_mode`] and
    /// configure it.
    ///
    /// # Errors
    /// [`RenderError::Surface`] when the handle is invalid, the surface
    /// offers no formats/modes, or configuration fails.
    pub fn try_new(
        renderer: &Renderer,
        target: wgpu::SurfaceTargetUnsafe,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        // SAFETY: the caller asserts the window handle is valid by
        // constructing the `SurfaceTargetUnsafe`; wgpu validates it lazily on
        // the capability query and configure below, returning an error
        // instead of misbehaving for a merely-wrong handle.
        let surface =
            unsafe { renderer.instance().create_surface_unsafe(target) }.map_err(|error| {
                RenderError::Surface {
                    message: error.to_string(),
                }
            })?;
        let capabilities = surface.get_capabilities(renderer.adapter());
        let format =
            choose_swapchain_format(&capabilities.formats).ok_or(RenderError::Surface {
                message: "surface offers no formats".to_string(),
            })?;
        let present_mode =
            choose_present_mode(&capabilities.present_modes).ok_or(RenderError::Surface {
                message: "surface offers no present modes".to_string(),
            })?;
        let alpha_mode = *capabilities
            .alpha_modes
            .first()
            .ok_or(RenderError::Surface {
                message: "surface offers no alpha modes".to_string(),
            })?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode,
            desired_maximum_frame_latency: 2,
            view_formats: Vec::new(),
        };
        surface.configure(&renderer.device, &config);
        Ok(Self { surface, config })
    }

    /// Reconfigure for a resized window (clamped to at least 1×1).
    pub fn resize(&mut self, renderer: &Renderer, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&renderer.device, &self.config);
    }

    /// The configured swapchain format — declare the backbuffer resource
    /// with it ([`ResourceSpec::with_format`]) so the executor's validation
    /// matches the swapchain.
    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// The configured present mode.
    #[must_use]
    pub fn present_mode(&self) -> wgpu::PresentMode {
        self.config.present_mode
    }

    /// The configured swapchain size (width, height).
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Acquire the next swapchain frame, render `graph` into it — the node
    /// writing the `backbuffer` resource draws onto the swapchain — and
    /// present. The backbuffer frame is never read back.
    ///
    /// # Errors
    /// [`RenderError::Surface`] on frame acquisition failure,
    /// [`RenderError::Graph`] / [`RenderError::Unsupported`] from planning
    /// and validation, [`RenderError::Poll`] / [`RenderError::Map`] if any
    /// other target is read back.
    pub fn render_to_surface(
        &self,
        renderer: &Renderer,
        graph: &RenderGraph,
        resources: &[ResourceSpec],
        kind_of: &dyn Fn(&str) -> Option<PassKind>,
        backbuffer: &'static str,
    ) -> Result<(), RenderError> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            // Still a usable frame; the next `configure` (resize) settles it.
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            other => {
                return Err(RenderError::Surface {
                    message: format!("frame acquisition failed: {other:?}"),
                })
            }
        };
        let external = ExternalTarget {
            name: backbuffer,
            texture: frame.texture.clone(),
            format: self.config.format,
            width: self.config.width,
            height: self.config.height,
        };
        execute_with_external_target(renderer, graph, resources, kind_of, &external)?;
        renderer.queue.present(frame);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swapchain_format_policy_prefers_srgb() {
        // sRGB entries win over linear ones, in surface-offer order.
        let formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Rgba16Float,
        ];
        assert_eq!(
            choose_swapchain_format(&formats),
            Some(wgpu::TextureFormat::Rgba8UnormSrgb)
        );
        // Without any sRGB format, the first offer is the fallback.
        let formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba16Float,
        ];
        assert_eq!(
            choose_swapchain_format(&formats),
            Some(wgpu::TextureFormat::Bgra8Unorm)
        );
        // No formats — the only case that yields None.
        assert_eq!(choose_swapchain_format(&[]), None);
    }

    #[test]
    fn present_mode_policy_always_picks_fifo_when_offered() {
        let modes = [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Fifo];
        assert_eq!(choose_present_mode(&modes), Some(wgpu::PresentMode::Fifo));
        let modes = [wgpu::PresentMode::Fifo];
        assert_eq!(choose_present_mode(&modes), Some(wgpu::PresentMode::Fifo));
        // Fifo is guaranteed by the spec; the fallback only exists for
        // degenerate capability lists.
        let modes = [wgpu::PresentMode::Immediate];
        assert_eq!(
            choose_present_mode(&modes),
            Some(wgpu::PresentMode::Immediate)
        );
        assert_eq!(choose_present_mode(&[]), None);
    }
}
