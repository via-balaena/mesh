//! GPU-accelerated mesh processing using WGPU compute shaders.
//!
//! This crate provides GPU-accelerated implementations of computationally
//! intensive mesh processing operations:
//!
//! - **SDF Computation**: Parallel signed distance field calculation
//! - **Surface Nets**: GPU-accelerated isosurface extraction
//! - **Collision Detection**: Parallel self-intersection detection
//!
//! # Performance Summary
//!
//! Based on benchmarks (see `BENCHMARKS.md` for full details):
//!
//! | Operation | GPU vs CPU | Recommendation |
//! |-----------|------------|----------------|
//! | SDF Computation | **3-68x faster** for meshes <5k tri | Use GPU for shell generation |
//! | Surface Nets | 0.03-0.37x (slower) | Always use CPU |
//! | Collision Detection | 0.03-1.0x (slower) | Always use CPU with BVH |
//!
//! ## When to Use GPU
//!
//! **SDF Computation** is the primary use case for GPU acceleration:
//! - Small meshes (<100 tri) with 128³ grid: **48-68x speedup**
//! - Medium meshes (320 tri) with 128³ grid: **14x speedup**
//! - Large meshes (>5k tri): GPU overhead dominates, CPU is faster
//!
//! **Best use case**: Shell generation for 3D printing, where typical scan
//! meshes (1k-10k triangles) are processed with high-resolution grids (128³+).
//!
//! ## When to Use CPU
//!
//! - **Surface Nets**: The `fast-surface-nets` crate is highly optimized
//! - **Collision Detection**: BVH-accelerated CPU is O(n log n) vs GPU's O(n²)
//! - **Large meshes**: GPU transfer overhead exceeds computation savings
//!
//! # Feature Flags
//!
//! This crate is designed to be an optional dependency. Enable it via
//! the `gpu` feature flag in dependent crates:
//!
//! ```toml
//! [dependencies]
//! mesh-shell = { version = "0.1", features = ["gpu"] }
//! ```
//!
//! # GPU Availability
//!
//! GPU support is automatically detected at runtime. Use [`context::GpuContext::is_available()`]
//! to check if a GPU is available, or use the `try_*` variants of computation
//! functions which return `None` instead of erroring when GPU is unavailable.
//!
//! # Example
//!
//! ```no_run
//! use mesh_gpu::context::GpuContext;
//! use mesh_gpu::sdf::{compute_sdf_gpu, GpuSdfParams};
//! use mesh_repair::Mesh;
//!
//! // Check GPU availability
//! if GpuContext::is_available() {
//!     println!("GPU available: {}", GpuContext::get().unwrap().adapter_info.name);
//! }
//!
//! // Compute SDF (will error if no GPU)
//! let mesh = Mesh::new();
//! let params = GpuSdfParams {
//!     dims: [100, 100, 100],
//!     origin: [0.0, 0.0, 0.0],
//!     voxel_size: 0.1,
//! };
//!
//! match compute_sdf_gpu(&mesh, &params) {
//!     Ok(result) => println!("Computed {} voxels in {:.2}ms",
//!         result.values.len(), result.compute_time_ms),
//!     Err(e) => eprintln!("GPU computation failed: {}", e),
//! }
//! ```
//!
//! # Automatic Fallback
//!
//! For production use, consider implementing automatic CPU fallback:
//!
//! ```no_run
//! use mesh_gpu::sdf::{try_compute_sdf_gpu, GpuSdfParams};
//! use mesh_repair::Mesh;
//!
//! fn compute_sdf_with_fallback(mesh: &Mesh, params: &GpuSdfParams) -> Vec<f32> {
//!     // Try GPU first
//!     if let Some(result) = try_compute_sdf_gpu(mesh, params) {
//!         return result.values;
//!     }
//!
//!     // Fall back to CPU implementation
//!     // ... CPU implementation here ...
//!     vec![]
//! }
//! ```
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Run all GPU benchmarks
//! cargo bench -p mesh-gpu
//!
//! # Run specific benchmark group
//! cargo bench -p mesh-gpu -- "SDF Computation"
//!
//! # View HTML reports
//! open target/criterion/report/index.html
//! ```

pub mod buffers;
pub mod collision;
pub mod context;
pub mod error;
pub mod sdf;
pub mod surface_nets;

// Re-export commonly used types
pub use collision::{
    GpuCollisionParams, GpuCollisionResult, detect_self_intersections_gpu,
    try_detect_self_intersections_gpu,
};
pub use context::{GpuContext, GpuDevicePreference};
pub use error::{GpuError, GpuResult};
pub use sdf::{GpuSdfParams, GpuSdfResult, compute_sdf_gpu, try_compute_sdf_gpu};
pub use surface_nets::{
    GpuSurfaceNetsParams, GpuSurfaceNetsResult, extract_isosurface_gpu, try_extract_isosurface_gpu,
};
