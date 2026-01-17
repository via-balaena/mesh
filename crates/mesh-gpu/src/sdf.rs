//! GPU-accelerated SDF (Signed Distance Field) computation.
//!
//! This module provides GPU-accelerated computation of signed distance fields
//! from triangle meshes. It uses WGPU compute shaders for parallel processing.

use tracing::{debug, info, warn};
use wgpu::{BindGroupLayout, ComputePipeline, ShaderModule};

use mesh_repair::Mesh;

use crate::buffers::{MeshBuffers, SdfGridBuffers, TileConfig};
use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};

/// Shader source for SDF computation.
const SDF_SHADER: &str = include_str!("shaders/sdf_compute.wgsl");

/// Parameters for GPU SDF computation.
#[derive(Debug, Clone)]
pub struct GpuSdfParams {
    /// Grid dimensions [x, y, z].
    pub dims: [usize; 3],
    /// Grid origin in world coordinates.
    pub origin: [f32; 3],
    /// Voxel size in world units.
    pub voxel_size: f32,
}

/// Result of GPU SDF computation.
#[derive(Debug)]
pub struct GpuSdfResult {
    /// Computed SDF values.
    pub values: Vec<f32>,
    /// Grid dimensions.
    pub dims: [usize; 3],
    /// Computation time in milliseconds.
    pub compute_time_ms: f64,
}

/// Pipeline for GPU SDF computation.
///
/// This struct caches the compiled shader and pipeline, allowing efficient
/// reuse across multiple SDF computations.
pub struct SdfPipeline {
    #[allow(dead_code)] // Kept for potential future use (shader introspection)
    shader: ShaderModule,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl SdfPipeline {
    /// Create a new SDF computation pipeline.
    pub fn new(ctx: &GpuContext) -> GpuResult<Self> {
        debug!("Creating SDF compute pipeline");

        // Compile shader
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sdf_compute"),
                source: wgpu::ShaderSource::Wgsl(SDF_SHADER.into()),
            });

        // Create bind group layout
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("sdf_bind_group_layout"),
                    entries: &[
                        // Triangles storage buffer (read-only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Grid params uniform buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // SDF values storage buffer (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create pipeline layout
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sdf_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create compute pipeline
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sdf_compute_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("compute_sdf"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            shader,
            pipeline,
            bind_group_layout,
        })
    }

    /// Compute SDF for a mesh.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `mesh_buffers` - Mesh data already uploaded to GPU
    /// * `params` - SDF computation parameters
    ///
    /// # Returns
    /// The computed SDF values.
    pub fn compute(
        &self,
        ctx: &GpuContext,
        mesh_buffers: &MeshBuffers,
        params: &GpuSdfParams,
    ) -> GpuResult<GpuSdfResult> {
        let start = std::time::Instant::now();
        let total_voxels = params.dims[0] * params.dims[1] * params.dims[2];

        info!(
            dims = ?params.dims,
            total_voxels = total_voxels,
            triangles = mesh_buffers.triangle_count,
            "Computing SDF on GPU"
        );

        // Allocate grid buffers
        let grid_buffers = SdfGridBuffers::allocate(
            ctx,
            params.dims,
            params.origin,
            params.voxel_size,
            mesh_buffers.triangle_count,
        )?;

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdf_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh_buffers.triangles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_buffers.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_buffers.values.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sdf_compute_encoder"),
            });

        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdf_compute_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Workgroup size is 256, so dispatch enough workgroups
            let workgroups = (total_voxels as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Submit commands
        ctx.queue.submit([encoder.finish()]);

        // Download results
        let values = grid_buffers.download_values(ctx)?;

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        info!(
            voxels = total_voxels,
            time_ms = compute_time_ms,
            "SDF computation complete"
        );

        Ok(GpuSdfResult {
            values,
            dims: params.dims,
            compute_time_ms,
        })
    }
}

/// Compute SDF on GPU with automatic fallback.
///
/// This is the main entry point for GPU SDF computation. It handles:
/// - GPU availability detection
/// - Pipeline creation and caching
/// - Automatic tiling for large grids
/// - Error handling with graceful fallback
///
/// # Arguments
/// * `mesh` - Source mesh
/// * `params` - SDF computation parameters
///
/// # Returns
/// The computed SDF values, or an error if GPU computation fails.
pub fn compute_sdf_gpu(mesh: &Mesh, params: &GpuSdfParams) -> GpuResult<GpuSdfResult> {
    let ctx = GpuContext::try_get()?;

    // Upload mesh to GPU
    let mesh_buffers = MeshBuffers::from_mesh(ctx, mesh)?;

    // Check if we need tiling
    let total_voxels = params.dims[0] * params.dims[1] * params.dims[2];
    let max_voxels = ctx.max_storage_buffer_size() as usize / std::mem::size_of::<f32>();

    if total_voxels > max_voxels {
        // Use tiled computation
        compute_sdf_tiled(ctx, &mesh_buffers, params)
    } else {
        // Direct computation
        let pipeline = SdfPipeline::new(ctx)?;
        pipeline.compute(ctx, &mesh_buffers, params)
    }
}

/// Compute SDF using tiled processing for large grids.
fn compute_sdf_tiled(
    ctx: &GpuContext,
    mesh_buffers: &MeshBuffers,
    params: &GpuSdfParams,
) -> GpuResult<GpuSdfResult> {
    let start = std::time::Instant::now();
    let total_voxels = params.dims[0] * params.dims[1] * params.dims[2];

    // Determine tile configuration based on available memory
    let available_memory = ctx.estimate_available_memory();
    // Reserve memory for mesh and overhead
    let grid_memory =
        available_memory.saturating_sub(mesh_buffers.triangles_size() + 256 * 1024 * 1024);
    let tile_config = TileConfig::for_memory_budget(grid_memory);

    let tile_counts = tile_config.tile_count(params.dims);
    let total_tiles = tile_config.total_tiles(params.dims);

    info!(
        grid_dims = ?params.dims,
        tile_size = ?tile_config.tile_size,
        tiles = total_tiles,
        "Using tiled SDF computation"
    );

    // Allocate result buffer
    let mut result = vec![0.0f32; total_voxels];

    // Create pipeline once
    let pipeline = SdfPipeline::new(ctx)?;

    // Process each tile
    for tz in 0..tile_counts[2] {
        for ty in 0..tile_counts[1] {
            for tx in 0..tile_counts[0] {
                let tile_origin_voxels = [
                    tx * tile_config.tile_size[0],
                    ty * tile_config.tile_size[1],
                    tz * tile_config.tile_size[2],
                ];

                // Calculate tile dimensions (may be smaller at edges)
                let tile_dims = [
                    (params.dims[0] - tile_origin_voxels[0]).min(tile_config.tile_size[0]),
                    (params.dims[1] - tile_origin_voxels[1]).min(tile_config.tile_size[1]),
                    (params.dims[2] - tile_origin_voxels[2]).min(tile_config.tile_size[2]),
                ];

                // Calculate tile origin in world coordinates
                let tile_origin_world = [
                    params.origin[0] + (tile_origin_voxels[0] as f32) * params.voxel_size,
                    params.origin[1] + (tile_origin_voxels[1] as f32) * params.voxel_size,
                    params.origin[2] + (tile_origin_voxels[2] as f32) * params.voxel_size,
                ];

                let tile_params = GpuSdfParams {
                    dims: tile_dims,
                    origin: tile_origin_world,
                    voxel_size: params.voxel_size,
                };

                // Compute tile
                let tile_result = pipeline.compute(ctx, mesh_buffers, &tile_params)?;

                // Copy tile results to main grid
                copy_tile_to_grid(
                    &tile_result.values,
                    &mut result,
                    params.dims,
                    tile_origin_voxels,
                    tile_dims,
                );

                debug!(tile_x = tx, tile_y = ty, tile_z = tz, "Tile processed");
            }
        }
    }

    let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    info!(
        tiles = total_tiles,
        time_ms = compute_time_ms,
        "Tiled SDF computation complete"
    );

    Ok(GpuSdfResult {
        values: result,
        dims: params.dims,
        compute_time_ms,
    })
}

/// Copy tile results to the main grid.
fn copy_tile_to_grid(
    tile_values: &[f32],
    grid_values: &mut [f32],
    grid_dims: [usize; 3],
    tile_origin: [usize; 3],
    tile_dims: [usize; 3],
) {
    // Use ZYX ordering to match mesh_to_sdf's layout
    for z in 0..tile_dims[2] {
        for y in 0..tile_dims[1] {
            for x in 0..tile_dims[0] {
                let tile_idx = z + y * tile_dims[2] + x * tile_dims[1] * tile_dims[2];
                let grid_x = tile_origin[0] + x;
                let grid_y = tile_origin[1] + y;
                let grid_z = tile_origin[2] + z;
                let grid_idx =
                    grid_z + grid_y * grid_dims[2] + grid_x * grid_dims[1] * grid_dims[2];

                if grid_idx < grid_values.len() && tile_idx < tile_values.len() {
                    grid_values[grid_idx] = tile_values[tile_idx];
                }
            }
        }
    }
}

/// Try to compute SDF on GPU, returning None if GPU is unavailable.
///
/// This is a convenience function that doesn't return an error for GPU
/// unavailability, making it easy to implement fallback logic.
pub fn try_compute_sdf_gpu(mesh: &Mesh, params: &GpuSdfParams) -> Option<GpuSdfResult> {
    match compute_sdf_gpu(mesh, params) {
        Ok(result) => Some(result),
        Err(GpuError::NotAvailable) => {
            debug!("GPU not available for SDF computation");
            None
        }
        Err(e) => {
            warn!("GPU SDF computation failed: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();

        // Unit cube centered at origin
        let coords = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];

        for c in &coords {
            mesh.vertices.push(Vertex::from_coords(c[0], c[1], c[2]));
        }

        // Cube faces (2 triangles per face)
        let faces = [
            [0, 1, 2],
            [0, 2, 3], // Front
            [4, 6, 5],
            [4, 7, 6], // Back
            [0, 5, 1],
            [0, 4, 5], // Bottom
            [2, 7, 3],
            [2, 6, 7], // Top
            [0, 3, 7],
            [0, 7, 4], // Left
            [1, 5, 6],
            [1, 6, 2], // Right
        ];

        for f in &faces {
            mesh.faces.push(*f);
        }

        mesh
    }

    #[test]
    fn test_gpu_sdf_params() {
        let params = GpuSdfParams {
            dims: [10, 10, 10],
            origin: [-2.0, -2.0, -2.0],
            voxel_size: 0.4,
        };

        assert_eq!(params.dims[0] * params.dims[1] * params.dims[2], 1000);
    }

    #[test]
    fn test_try_compute_sdf_gpu() {
        let mesh = create_test_cube();
        let params = GpuSdfParams {
            dims: [5, 5, 5],
            origin: [-2.0, -2.0, -2.0],
            voxel_size: 0.8,
        };

        // This test will pass whether or not GPU is available
        let _result = try_compute_sdf_gpu(&mesh, &params);
    }
}
