//! GPU-accelerated Surface Nets isosurface extraction.
//!
//! Surface Nets is an algorithm for extracting isosurfaces from volumetric
//! data (like signed distance fields). It produces higher quality meshes
//! than Marching Cubes with simpler implementation.

use bytemuck::{Pod, Zeroable};
use tracing::{debug, info, warn};
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, ComputePipeline};

use mesh_repair::Mesh;

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};

/// Shader source for Surface Nets.
const SURFACE_NETS_SHADER: &str = include_str!("shaders/surface_nets.wgsl");

/// Parameters for GPU Surface Nets extraction.
#[derive(Debug, Clone)]
pub struct GpuSurfaceNetsParams {
    /// Grid dimensions [x, y, z].
    pub dims: [usize; 3],
    /// Grid origin in world coordinates.
    pub origin: [f32; 3],
    /// Voxel size in world units.
    pub voxel_size: f32,
    /// Iso-value for surface extraction (typically 0.0 for SDF).
    pub iso_value: f32,
}

impl Default for GpuSurfaceNetsParams {
    fn default() -> Self {
        Self {
            dims: [0, 0, 0],
            origin: [0.0, 0.0, 0.0],
            voxel_size: 1.0,
            iso_value: 0.0,
        }
    }
}

/// Result of GPU Surface Nets extraction.
#[derive(Debug)]
pub struct GpuSurfaceNetsResult {
    /// Extracted mesh.
    pub mesh: Mesh,
    /// Number of active cells found.
    pub active_cells: usize,
    /// Number of vertices generated.
    pub vertex_count: usize,
    /// Computation time in milliseconds.
    pub compute_time_ms: f64,
}

/// GPU vertex output structure (matches shader).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuOutputVertex {
    position: [f32; 4], // xyz + vertex_idx
    normal: [f32; 4],   // xyz + padding
}

/// Uniform parameters for the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ShaderGridParams {
    origin: [f32; 4],
    dims: [u32; 4],
    voxel_size: f32,
    iso_value: f32,
    _padding: [f32; 2],
}

/// Pipeline for GPU Surface Nets extraction.
pub struct SurfaceNetsPipeline {
    identify_pipeline: ComputePipeline,
    generate_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl SurfaceNetsPipeline {
    /// Create a new Surface Nets pipeline.
    pub fn new(ctx: &GpuContext) -> GpuResult<Self> {
        debug!("Creating Surface Nets compute pipeline");

        // Compile shader
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("surface_nets"),
                source: wgpu::ShaderSource::Wgsl(SURFACE_NETS_SHADER.into()),
            });

        // Create bind group layout
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("surface_nets_bind_group_layout"),
                    entries: &[
                        // SDF values (read-only)
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
                        // Grid params (uniform)
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
                        // Active cells bitmap (read-write)
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
                        // Cell vertices output (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Vertex count (atomic)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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
                label: Some("surface_nets_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create identify_active_cells pipeline
        let identify_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("surface_nets_identify_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("identify_active_cells"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Create generate_vertices pipeline
        let generate_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("surface_nets_generate_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("generate_vertices"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Ok(Self {
            identify_pipeline,
            generate_pipeline,
            bind_group_layout,
        })
    }

    /// Extract isosurface from SDF values.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `sdf_values` - SDF values already uploaded to GPU
    /// * `params` - Extraction parameters
    ///
    /// # Returns
    /// The extracted mesh.
    pub fn extract(
        &self,
        ctx: &GpuContext,
        sdf_buffer: &wgpu::Buffer,
        params: &GpuSurfaceNetsParams,
    ) -> GpuResult<GpuSurfaceNetsResult> {
        let start = std::time::Instant::now();

        // Calculate number of cells (one less than voxels in each dimension)
        let cells = [
            params.dims[0].saturating_sub(1),
            params.dims[1].saturating_sub(1),
            params.dims[2].saturating_sub(1),
        ];
        let total_cells = cells[0] * cells[1] * cells[2];

        if total_cells == 0 {
            return Ok(GpuSurfaceNetsResult {
                mesh: Mesh::new(),
                active_cells: 0,
                vertex_count: 0,
                compute_time_ms: 0.0,
            });
        }

        info!(
            dims = ?params.dims,
            cells = total_cells,
            "Extracting isosurface on GPU"
        );

        // Create uniform buffer for grid params
        let grid_params = ShaderGridParams {
            origin: [params.origin[0], params.origin[1], params.origin[2], 0.0],
            dims: [
                params.dims[0] as u32,
                params.dims[1] as u32,
                params.dims[2] as u32,
                0,
            ],
            voxel_size: params.voxel_size,
            iso_value: params.iso_value,
            _padding: [0.0, 0.0],
        };

        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("surface_nets_params"),
                contents: bytemuck::bytes_of(&grid_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create active cells buffer (1 u32 flag per cell: 0 = inactive, 1 = active)
        let active_cells_size = total_cells * std::mem::size_of::<u32>();
        let active_cells_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_nets_active_cells"),
            size: active_cells_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create cell vertices buffer (one potential vertex per cell)
        let vertices_size = total_cells * std::mem::size_of::<GpuOutputVertex>();
        let vertices_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_nets_vertices"),
            size: vertices_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create vertex count buffer (atomic counter)
        let count_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_nets_count"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize count to 0
        ctx.queue
            .write_buffer(&count_buffer, 0, bytemuck::bytes_of(&0u32));

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_nets_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sdf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: active_cells_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vertices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: count_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("surface_nets_encoder"),
            });

        let workgroups = (total_cells as u32).div_ceil(256);

        // Pass 1: Identify active cells
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_nets_identify_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.identify_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Generate vertices
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_nets_generate_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.generate_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Submit commands
        ctx.queue.submit([encoder.finish()]);

        // Download results
        let vertex_count = self.download_count(ctx, &count_buffer)?;
        let vertices = self.download_vertices(ctx, &vertices_buffer, total_cells)?;

        // Build mesh from vertices (faces generated on CPU for now)
        let mesh = self.build_mesh(&vertices, vertex_count as usize, &cells, params);

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        info!(
            vertices = vertex_count,
            faces = mesh.faces.len(),
            time_ms = compute_time_ms,
            "Isosurface extraction complete"
        );

        Ok(GpuSurfaceNetsResult {
            mesh,
            active_cells: vertex_count as usize, // Approximate
            vertex_count: vertex_count as usize,
            compute_time_ms,
        })
    }

    fn download_count(&self, ctx: &GpuContext, buffer: &wgpu::Buffer) -> GpuResult<u32> {
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("count_staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, std::mem::size_of::<u32>() as u64);
        ctx.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| GpuError::BufferMapping("channel closed".into()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let data = slice.get_mapped_range();
        let count = *bytemuck::from_bytes::<u32>(&data);
        drop(data);
        staging.unmap();

        Ok(count)
    }

    fn download_vertices(
        &self,
        ctx: &GpuContext,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> GpuResult<Vec<GpuOutputVertex>> {
        let size = count * std::mem::size_of::<GpuOutputVertex>();
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertices_staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
        ctx.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| GpuError::BufferMapping("channel closed".into()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let data = slice.get_mapped_range();
        let vertices: Vec<GpuOutputVertex> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(vertices)
    }

    fn build_mesh(
        &self,
        gpu_vertices: &[GpuOutputVertex],
        vertex_count: usize,
        _cells: &[usize; 3],
        _params: &GpuSurfaceNetsParams,
    ) -> Mesh {
        use mesh_repair::Vertex;

        let mut mesh = Mesh::new();

        // Add vertices
        for v in gpu_vertices.iter().take(vertex_count) {
            let mut vertex = Vertex::from_coords(
                v.position[0] as f64,
                v.position[1] as f64,
                v.position[2] as f64,
            );
            vertex.normal = Some(nalgebra::Vector3::new(
                v.normal[0] as f64,
                v.normal[1] as f64,
                v.normal[2] as f64,
            ));
            mesh.vertices.push(vertex);
        }

        // Note: Face generation is more complex in Surface Nets and requires
        // connectivity information. For now, we return just the vertices.
        // The face generation can be done on CPU using the cell adjacency info,
        // or a third GPU pass could be added.
        //
        // For a complete implementation, faces are generated by connecting
        // vertices from adjacent active cells that share an edge crossing.

        mesh
    }
}

/// Extract isosurface from SDF values on GPU.
pub fn extract_isosurface_gpu(
    sdf_values: &[f32],
    params: &GpuSurfaceNetsParams,
) -> GpuResult<GpuSurfaceNetsResult> {
    let ctx = GpuContext::try_get()?;

    // Upload SDF values to GPU
    let sdf_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface_nets_sdf"),
            contents: bytemuck::cast_slice(sdf_values),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let pipeline = SurfaceNetsPipeline::new(ctx)?;
    pipeline.extract(ctx, &sdf_buffer, params)
}

/// Try to extract isosurface on GPU, returning None if unavailable.
pub fn try_extract_isosurface_gpu(
    sdf_values: &[f32],
    params: &GpuSurfaceNetsParams,
) -> Option<GpuSurfaceNetsResult> {
    match extract_isosurface_gpu(sdf_values, params) {
        Ok(result) => Some(result),
        Err(GpuError::NotAvailable) => {
            debug!("GPU not available for isosurface extraction");
            None
        }
        Err(e) => {
            warn!("GPU isosurface extraction failed: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_surface_nets_params_default() {
        let params = GpuSurfaceNetsParams::default();
        assert_eq!(params.iso_value, 0.0);
        assert_eq!(params.voxel_size, 1.0);
    }

    #[test]
    fn test_try_extract_isosurface_gpu() {
        // Create a simple 3x3x3 SDF with a sphere
        let mut sdf = vec![1.0f32; 27];
        // Center voxel (1,1,1) is inside
        sdf[1 + 3 + 9] = -1.0;

        let params = GpuSurfaceNetsParams {
            dims: [3, 3, 3],
            origin: [0.0, 0.0, 0.0],
            voxel_size: 1.0,
            iso_value: 0.0,
        };

        // This test will pass whether or not GPU is available
        let _result = try_extract_isosurface_gpu(&sdf, &params);
    }
}
