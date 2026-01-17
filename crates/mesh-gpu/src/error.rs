//! GPU error types for mesh processing.

use thiserror::Error;

/// Errors that can occur during GPU operations.
#[derive(Debug, Error)]
pub enum GpuError {
    /// GPU device is not available on this system.
    #[error("GPU not available: no compatible device found")]
    NotAvailable,

    /// GPU ran out of memory during computation.
    #[error("GPU out of memory: required {required} bytes, available {available} bytes")]
    OutOfMemory { required: u64, available: u64 },

    /// Shader compilation failed.
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// GPU device was lost during computation.
    #[error("GPU device lost")]
    DeviceLost,

    /// GPU execution failed.
    #[error("GPU execution failed: {0}")]
    Execution(String),

    /// Buffer mapping failed.
    #[error("buffer mapping failed: {0}")]
    BufferMapping(String),

    /// Grid dimensions exceed GPU limits.
    #[error("grid too large for GPU: {dims:?} voxels ({total} total), max supported: {max}")]
    GridTooLarge {
        dims: [usize; 3],
        total: usize,
        max: usize,
    },

    /// Mesh too large for GPU.
    #[error("mesh too large for GPU: {triangles} triangles, max supported: {max}")]
    MeshTooLarge { triangles: usize, max: usize },
}

/// Result type for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;
