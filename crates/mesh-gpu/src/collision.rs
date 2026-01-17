//! GPU-accelerated collision detection for self-intersection testing.
//!
//! This module provides GPU-accelerated self-intersection detection using
//! WGPU compute shaders. It uses AABB-based broad phase culling followed
//! by exact triangle-triangle intersection tests using the Separating Axis
//! Theorem (SAT).

use bytemuck::{Pod, Zeroable};
use tracing::{debug, info, warn};
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, ComputePipeline};

use mesh_repair::Mesh;

use crate::buffers::MeshBuffers;
use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};

/// Shader source for collision detection.
const COLLISION_SHADER: &str = include_str!("shaders/collision.wgsl");

/// Parameters for GPU collision detection.
#[derive(Debug, Clone)]
pub struct GpuCollisionParams {
    /// Maximum number of intersection pairs to report.
    /// Set to 0 for unlimited (up to buffer size).
    pub max_pairs: usize,
    /// Epsilon for geometric comparisons.
    pub epsilon: f32,
    /// Whether to skip adjacent triangles (sharing vertices).
    pub skip_adjacent: bool,
}

impl Default for GpuCollisionParams {
    fn default() -> Self {
        Self {
            max_pairs: 1000,
            epsilon: 1e-7,
            skip_adjacent: true,
        }
    }
}

/// Result of GPU collision detection.
#[derive(Debug)]
pub struct GpuCollisionResult {
    /// Whether any self-intersections were found.
    pub has_intersections: bool,
    /// Number of intersecting triangle pairs found.
    pub intersection_count: usize,
    /// List of intersecting triangle pairs (face_idx_a, face_idx_b).
    pub intersecting_pairs: Vec<(u32, u32)>,
    /// Whether the search was truncated due to max_pairs limit.
    pub truncated: bool,
    /// Computation time in milliseconds.
    pub compute_time_ms: f64,
}

/// Uniform parameters for the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ShaderCollisionParams {
    triangle_count: u32,
    max_pairs: u32,
    epsilon: f32,
    skip_adjacent: u32,
}

/// AABB structure (matches shader).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuAABB {
    min: [f32; 3],
    _padding1: f32,
    max: [f32; 3],
    _padding2: f32,
}

/// Intersection pair (matches shader).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuIntersectionPair {
    tri_a: u32,
    tri_b: u32,
}

/// Pipeline for GPU collision detection.
pub struct CollisionPipeline {
    aabb_pipeline: ComputePipeline,
    test_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl CollisionPipeline {
    /// Create a new collision detection pipeline.
    pub fn new(ctx: &GpuContext) -> GpuResult<Self> {
        debug!("Creating collision detection compute pipeline");

        // Compile shader
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("collision"),
                source: wgpu::ShaderSource::Wgsl(COLLISION_SHADER.into()),
            });

        // Create bind group layout
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("collision_bind_group_layout"),
                    entries: &[
                        // Triangles (read-only)
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
                        // Params (uniform)
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
                        // AABBs (read-write)
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
                        // Intersection pairs (read-write)
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
                        // Pair count (atomic)
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
                label: Some("collision_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create AABB computation pipeline
        let aabb_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("collision_aabb_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("compute_aabbs"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create intersection test pipeline
        let test_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("collision_test_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("test_intersections"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            aabb_pipeline,
            test_pipeline,
            bind_group_layout,
        })
    }

    /// Detect self-intersections in a mesh.
    pub fn detect(
        &self,
        ctx: &GpuContext,
        mesh_buffers: &MeshBuffers,
        params: &GpuCollisionParams,
    ) -> GpuResult<GpuCollisionResult> {
        let start = std::time::Instant::now();
        let triangle_count = mesh_buffers.triangle_count as usize;

        if triangle_count < 2 {
            return Ok(GpuCollisionResult {
                has_intersections: false,
                intersection_count: 0,
                intersecting_pairs: Vec::new(),
                truncated: false,
                compute_time_ms: 0.0,
            });
        }

        let max_pairs = if params.max_pairs == 0 {
            triangle_count * triangle_count / 2 // Upper bound
        } else {
            params.max_pairs
        };

        info!(
            triangles = triangle_count,
            max_pairs = max_pairs,
            "Detecting self-intersections on GPU"
        );

        // Create uniform buffer
        let shader_params = ShaderCollisionParams {
            triangle_count: triangle_count as u32,
            max_pairs: max_pairs as u32,
            epsilon: params.epsilon,
            skip_adjacent: if params.skip_adjacent { 1 } else { 0 },
        };

        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("collision_params"),
                contents: bytemuck::bytes_of(&shader_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create AABB buffer
        let aabb_size = triangle_count * std::mem::size_of::<GpuAABB>();
        let aabb_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("collision_aabbs"),
            size: aabb_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create intersection pairs buffer
        let pairs_size = max_pairs * std::mem::size_of::<GpuIntersectionPair>();
        let pairs_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("collision_pairs"),
            size: pairs_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create pair count buffer
        let count_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("collision_count"),
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
            label: Some("collision_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh_buffers.triangles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: aabb_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pairs_buffer.as_entire_binding(),
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
                label: Some("collision_encoder"),
            });

        let workgroups = (triangle_count as u32).div_ceil(256);

        // Pass 1: Compute AABBs
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("collision_aabb_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.aabb_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Test intersections
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("collision_test_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.test_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Submit commands
        ctx.queue.submit([encoder.finish()]);

        // Download results
        let pair_count = self.download_count(ctx, &count_buffer)?;
        let pairs = self.download_pairs(ctx, &pairs_buffer, pair_count.min(max_pairs as u32))?;

        let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let intersecting_pairs: Vec<(u32, u32)> =
            pairs.iter().map(|p| (p.tri_a, p.tri_b)).collect();

        info!(
            pairs_found = pair_count,
            time_ms = compute_time_ms,
            "Collision detection complete"
        );

        Ok(GpuCollisionResult {
            has_intersections: pair_count > 0,
            intersection_count: pair_count as usize,
            intersecting_pairs,
            truncated: pair_count as usize >= max_pairs,
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

    fn download_pairs(
        &self,
        ctx: &GpuContext,
        buffer: &wgpu::Buffer,
        count: u32,
    ) -> GpuResult<Vec<GpuIntersectionPair>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let size = (count as usize) * std::mem::size_of::<GpuIntersectionPair>();
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pairs_staging"),
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
        let pairs: Vec<GpuIntersectionPair> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(pairs)
    }
}

/// Detect self-intersections in a mesh on GPU.
pub fn detect_self_intersections_gpu(
    mesh: &Mesh,
    params: &GpuCollisionParams,
) -> GpuResult<GpuCollisionResult> {
    let ctx = GpuContext::try_get()?;

    // Upload mesh to GPU
    let mesh_buffers = MeshBuffers::from_mesh(ctx, mesh)?;

    let pipeline = CollisionPipeline::new(ctx)?;
    pipeline.detect(ctx, &mesh_buffers, params)
}

/// Try to detect self-intersections on GPU, returning None if unavailable.
pub fn try_detect_self_intersections_gpu(
    mesh: &Mesh,
    params: &GpuCollisionParams,
) -> Option<GpuCollisionResult> {
    match detect_self_intersections_gpu(mesh, params) {
        Ok(result) => Some(result),
        Err(GpuError::NotAvailable) => {
            debug!("GPU not available for collision detection");
            None
        }
        Err(e) => {
            warn!("GPU collision detection failed: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_simple_mesh() -> Mesh {
        let mut mesh = Mesh::new();

        // Single triangle
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        mesh
    }

    #[test]
    fn test_gpu_collision_params_default() {
        let params = GpuCollisionParams::default();
        assert!(params.skip_adjacent);
        assert_eq!(params.max_pairs, 1000);
    }

    #[test]
    fn test_try_detect_self_intersections_gpu() {
        let mesh = create_simple_mesh();
        let params = GpuCollisionParams::default();

        // This test will pass whether or not GPU is available
        let _result = try_detect_self_intersections_gpu(&mesh, &params);
    }
}
