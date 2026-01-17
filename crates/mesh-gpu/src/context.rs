//! GPU context management for mesh processing.
//!
//! Provides lazy initialization of the GPU device and queue, with automatic
//! fallback detection when GPU is unavailable.

use std::sync::OnceLock;
use tracing::{debug, info, warn};
use wgpu::{Device, DeviceDescriptor, Instance, Queue, RequestAdapterOptions};

use crate::error::{GpuError, GpuResult};

/// Global GPU context, lazily initialized on first access.
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

/// GPU device preference for adapter selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuDevicePreference {
    /// Automatically select the best available device.
    #[default]
    Auto,
    /// Prefer high-performance discrete GPU.
    HighPerformance,
    /// Prefer low-power integrated GPU.
    LowPower,
}

/// Information about the GPU adapter.
#[derive(Debug, Clone)]
pub struct GpuAdapterInfo {
    /// Device name.
    pub name: String,
    /// Vendor name.
    pub vendor: String,
    /// Device type (discrete, integrated, etc.).
    pub device_type: String,
    /// Backend API (Vulkan, Metal, DX12, etc.).
    pub backend: String,
}

impl From<wgpu::AdapterInfo> for GpuAdapterInfo {
    fn from(info: wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name,
            vendor: format!("{}", info.vendor),
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
        }
    }
}

/// GPU context containing device, queue, and adapter information.
///
/// This is a lazy-initialized singleton that provides access to GPU resources.
/// Use `GpuContext::get()` to access the global context, or `GpuContext::try_get()`
/// for error handling.
pub struct GpuContext {
    /// The WGPU device for creating resources and pipelines.
    pub device: Device,
    /// The command queue for submitting work.
    pub queue: Queue,
    /// Information about the GPU adapter.
    pub adapter_info: GpuAdapterInfo,
    /// Device limits for resource allocation.
    pub limits: wgpu::Limits,
}

impl GpuContext {
    /// Get or initialize the global GPU context.
    ///
    /// Returns `Some(&GpuContext)` if a GPU is available, `None` otherwise.
    /// The context is lazily initialized on first call.
    ///
    /// # Example
    /// ```no_run
    /// use mesh_gpu::context::GpuContext;
    ///
    /// if let Some(ctx) = GpuContext::get() {
    ///     println!("GPU: {}", ctx.adapter_info.name);
    /// } else {
    ///     println!("No GPU available");
    /// }
    /// ```
    pub fn get() -> Option<&'static GpuContext> {
        GPU_CONTEXT
            .get_or_init(
                || match pollster::block_on(Self::try_init(GpuDevicePreference::Auto)) {
                    Ok(ctx) => {
                        info!(
                            adapter = %ctx.adapter_info.name,
                            backend = %ctx.adapter_info.backend,
                            "GPU context initialized"
                        );
                        Some(ctx)
                    }
                    Err(e) => {
                        warn!("GPU initialization failed: {}", e);
                        None
                    }
                },
            )
            .as_ref()
    }

    /// Try to get the global GPU context, returning an error if unavailable.
    ///
    /// # Errors
    /// Returns `GpuError::NotAvailable` if no GPU is available.
    pub fn try_get() -> GpuResult<&'static GpuContext> {
        Self::get().ok_or(GpuError::NotAvailable)
    }

    /// Check if a GPU is available without initializing the context.
    ///
    /// Note: This will initialize the context on first call to determine availability.
    pub fn is_available() -> bool {
        Self::get().is_some()
    }

    /// Try to initialize a new GPU context with the specified device preference.
    async fn try_init(preference: GpuDevicePreference) -> GpuResult<GpuContext> {
        debug!("Initializing GPU context with preference: {:?}", preference);

        // Create WGPU instance with all available backends
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter based on preference
        let power_preference = match preference {
            GpuDevicePreference::Auto | GpuDevicePreference::HighPerformance => {
                wgpu::PowerPreference::HighPerformance
            }
            GpuDevicePreference::LowPower => wgpu::PowerPreference::LowPower,
        };

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(GpuError::NotAvailable)?;

        let adapter_info = adapter.get_info();
        debug!(
            name = %adapter_info.name,
            vendor = adapter_info.vendor,
            device_type = ?adapter_info.device_type,
            backend = ?adapter_info.backend,
            "GPU adapter found"
        );

        // Request device with default limits
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("mesh-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| GpuError::Execution(format!("device request failed: {}", e)))?;

        let limits = device.limits();

        Ok(GpuContext {
            device,
            queue,
            adapter_info: adapter_info.into(),
            limits,
        })
    }

    /// Get the maximum buffer size supported by this device.
    pub fn max_buffer_size(&self) -> u64 {
        self.limits.max_buffer_size
    }

    /// Get the maximum storage buffer binding size.
    pub fn max_storage_buffer_size(&self) -> u32 {
        self.limits.max_storage_buffer_binding_size
    }

    /// Get the maximum compute workgroup size in each dimension.
    pub fn max_workgroup_size(&self) -> [u32; 3] {
        [
            self.limits.max_compute_workgroup_size_x,
            self.limits.max_compute_workgroup_size_y,
            self.limits.max_compute_workgroup_size_z,
        ]
    }

    /// Get the maximum number of compute invocations per workgroup.
    pub fn max_invocations_per_workgroup(&self) -> u32 {
        self.limits.max_compute_invocations_per_workgroup
    }

    /// Estimate available GPU memory (conservative estimate).
    ///
    /// Note: WGPU doesn't expose exact memory info, so this returns a
    /// conservative estimate based on buffer limits.
    pub fn estimate_available_memory(&self) -> u64 {
        // Use max buffer size as a proxy for available memory
        // In practice, we can often allocate more via multiple buffers
        self.limits.max_buffer_size
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter_info", &self.adapter_info)
            .field("max_buffer_size", &self.limits.max_buffer_size)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability_check() {
        // This test just checks that is_available() doesn't panic
        let _available = GpuContext::is_available();
    }

    #[test]
    fn test_gpu_context_get() {
        // This test checks that get() returns consistent results
        let first = GpuContext::get();
        let second = GpuContext::get();

        // Both calls should return the same result
        assert_eq!(first.is_some(), second.is_some());

        if let Some(ctx) = first {
            assert!(!ctx.adapter_info.name.is_empty());
        }
    }

    #[test]
    fn test_gpu_device_preference_default() {
        assert_eq!(GpuDevicePreference::default(), GpuDevicePreference::Auto);
    }
}
