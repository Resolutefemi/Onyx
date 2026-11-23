//! wgpu device/adapter initialization for the Onyx renderer (`docs/PRD.md` §2).
//!
//! [`Renderer::try_new`] creates a real GPU context or fails cleanly:
//! headless machines (CI containers without a GPU/driver) receive
//! [`RenderError::NoAdapter`] instead of a panic, so engine code can probe
//! hardware availability and degrade gracefully.

use std::fmt;

use crate::{Backend, RendererConfig};

/// Renderer failures, all carrying a driver-reported message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderError {
    /// No compatible GPU adapter was found (headless machine, missing drivers,
    /// or a backend the platform cannot serve).
    NoAdapter {
        /// Adapter-request error text from wgpu.
        message: String,
    },
    /// An adapter was found but the device/queue request failed.
    DeviceRequest {
        /// Device-request error text from wgpu.
        message: String,
    },
    /// A GPU submission could not be awaited.
    Poll {
        /// Poll error text from wgpu.
        message: String,
    },
    /// A readback buffer could not be mapped.
    Map {
        /// Map error text from wgpu.
        message: String,
    },
    /// A render graph could not be scheduled.
    Graph {
        /// Scheduling error text from the graph.
        message: String,
    },
    /// The configuration or node declaration is not supported (yet).
    Unsupported {
        /// What is unsupported and what replaces it.
        message: String,
    },
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter { message } => write!(f, "no GPU adapter available: {message}"),
            Self::DeviceRequest { message } => write!(f, "device request failed: {message}"),
            Self::Poll { message } => write!(f, "GPU poll failed: {message}"),
            Self::Map { message } => write!(f, "readback map failed: {message}"),
            Self::Graph { message } => write!(f, "render graph error: {message}"),
            Self::Unsupported { message } => write!(f, "unsupported: {message}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// A live wgpu context: instance, adapter, device and queue, ready to render.
pub struct Renderer {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
}

impl Renderer {
    /// Create a renderer from `config`. `config.backend` filters the instance
    /// backends; `None` probes every backend wgpu supports. Fails with
    /// [`RenderError::NoAdapter`] on GPU-less machines instead of panicking.
    ///
    /// # Errors
    /// [`RenderError::NoAdapter`] when no adapter matches,
    /// [`RenderError::DeviceRequest`] when the adapter cannot provide a device.
    pub fn try_new(config: &RendererConfig) -> Result<Self, RenderError> {
        let backends = config
            .backend
            .map_or_else(wgpu::Backends::all, |backend| match backend {
                Backend::Vulkan => wgpu::Backends::VULKAN,
                Backend::Dx12 => wgpu::Backends::DX12,
                Backend::Metal => wgpu::Backends::METAL,
                Backend::WebGPU => wgpu::Backends::BROWSER_WEBGPU,
            });
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        };
        let adapter = pollster::block_on(instance.request_adapter(&options)).map_err(|error| {
            RenderError::NoAdapter {
                message: error.to_string(),
            }
        })?;
        let info = adapter.get_info();
        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("onyx-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&device_descriptor))
            .map_err(|error| RenderError::DeviceRequest {
                message: error.to_string(),
            })?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            info,
        })
    }

    /// Adapter metadata as reported by the driver (name, vendor, backend).
    #[must_use]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// The wgpu instance this renderer was created from.
    #[must_use]
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// The resolved wgpu adapter.
    #[must_use]
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_is_graceful_with_and_without_a_gpu() {
        match Renderer::try_new(&RendererConfig::default()) {
            Ok(renderer) => {
                let info = renderer.adapter_info();
                assert!(!info.name.is_empty(), "adapter must report a name");
            }
            Err(RenderError::NoAdapter { message }) => {
                assert!(!message.is_empty(), "headless machines get a real message");
            }
            Err(other) => panic!("unexpected renderer error: {other}"),
        }
    }

    #[test]
    fn backend_filters_never_panic() {
        for backend in Backend::all() {
            let config = RendererConfig {
                backend: Some(*backend),
                ..RendererConfig::default()
            };
            // Either resolves an adapter or fails cleanly — never panics.
            let _ = Renderer::try_new(&config);
        }
    }
}
