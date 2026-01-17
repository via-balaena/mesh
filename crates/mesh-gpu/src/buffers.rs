//! GPU buffer types for mesh and SDF grid data.
//!
//! This module provides structures for efficiently transferring mesh and grid
//! data to and from the GPU.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages};

use mesh_repair::Mesh;

use crate::context::GpuContext;
use crate::error::{GpuError, GpuResult};

/// GPU-friendly triangle representation with aligned fields.
///
/// Uses 4-component vectors for proper GPU alignment (vec4 = 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuTriangle {
    /// First vertex position (xyz) + padding.
    pub v0: [f32; 4],
    /// Second vertex position (xyz) + padding.
    pub v1: [f32; 4],
    /// Third vertex position (xyz) + padding.
    pub v2: [f32; 4],
}

impl GpuTriangle {
    /// Create a GPU triangle from vertex positions.
    pub fn new(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> Self {
        Self {
            v0: [v0[0], v0[1], v0[2], 0.0],
            v1: [v1[0], v1[1], v1[2], 0.0],
            v2: [v2[0], v2[1], v2[2], 0.0],
        }
    }
}

/// GPU-friendly vertex representation with offset data.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVertex {
    /// Vertex position (xyz) + padding.
    pub position: [f32; 4],
    /// Offset value for variable thickness shells.
    pub offset: f32,
    /// Vertex tag/region ID.
    pub tag: u32,
    /// Padding for alignment.
    pub _padding: [f32; 2],
}

/// GPU buffers containing mesh geometry data.
pub struct MeshBuffers {
    /// Buffer containing triangle data.
    pub triangles: Buffer,
    /// Buffer containing vertex data (for offset queries).
    pub vertices: Buffer,
    /// Number of triangles.
    pub triangle_count: u32,
    /// Number of vertices.
    pub vertex_count: u32,
}

impl MeshBuffers {
    /// Create GPU buffers from a mesh.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `mesh` - Source mesh to upload
    ///
    /// # Returns
    /// GPU buffers containing the mesh data.
    pub fn from_mesh(ctx: &GpuContext, mesh: &Mesh) -> GpuResult<Self> {
        let triangle_count = mesh.faces.len();
        let vertex_count = mesh.vertices.len();

        // Check mesh size limits
        let max_triangles =
            ctx.max_storage_buffer_size() as usize / std::mem::size_of::<GpuTriangle>();
        if triangle_count > max_triangles {
            return Err(GpuError::MeshTooLarge {
                triangles: triangle_count,
                max: max_triangles,
            });
        }

        // Convert triangles to GPU format
        let gpu_triangles: Vec<GpuTriangle> = mesh
            .faces
            .iter()
            .map(|face| {
                let v0 = &mesh.vertices[face[0] as usize].position;
                let v1 = &mesh.vertices[face[1] as usize].position;
                let v2 = &mesh.vertices[face[2] as usize].position;
                GpuTriangle::new(
                    [v0.x as f32, v0.y as f32, v0.z as f32],
                    [v1.x as f32, v1.y as f32, v1.z as f32],
                    [v2.x as f32, v2.y as f32, v2.z as f32],
                )
            })
            .collect();

        // Convert vertices to GPU format
        let gpu_vertices: Vec<GpuVertex> = mesh
            .vertices
            .iter()
            .map(|v| GpuVertex {
                position: [
                    v.position.x as f32,
                    v.position.y as f32,
                    v.position.z as f32,
                    0.0,
                ],
                offset: v.offset.unwrap_or(0.0),
                tag: v.tag.unwrap_or(0),
                _padding: [0.0, 0.0],
            })
            .collect();

        // Create GPU buffers
        let triangles = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_triangles"),
                contents: bytemuck::cast_slice(&gpu_triangles),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            });

        let vertices = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_vertices"),
                contents: bytemuck::cast_slice(&gpu_vertices),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            });

        Ok(Self {
            triangles,
            vertices,
            triangle_count: triangle_count as u32,
            vertex_count: vertex_count as u32,
        })
    }

    /// Get the size of triangle buffer in bytes.
    pub fn triangles_size(&self) -> u64 {
        self.triangles.size()
    }

    /// Get the size of vertex buffer in bytes.
    pub fn vertices_size(&self) -> u64 {
        self.vertices.size()
    }
}

/// Uniform parameters for SDF grid computation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuGridParams {
    /// Grid origin (min corner) in world coordinates.
    pub origin: [f32; 4],
    /// Grid dimensions (x, y, z, padding).
    pub dims: [u32; 4],
    /// Voxel size in world units.
    pub voxel_size: f32,
    /// Number of triangles in mesh.
    pub triangle_count: u32,
    /// Padding for alignment.
    pub _padding: [f32; 2],
}

/// GPU buffers for SDF grid computation and storage.
pub struct SdfGridBuffers {
    /// Uniform buffer containing grid parameters.
    pub params: Buffer,
    /// Storage buffer for SDF values (read/write).
    pub values: Buffer,
    /// Storage buffer for offset values (read/write).
    pub offsets: Buffer,
    /// Grid dimensions.
    pub dims: [usize; 3],
    /// Total number of voxels.
    pub total_voxels: usize,
}

impl SdfGridBuffers {
    /// Allocate GPU buffers for an SDF grid.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `dims` - Grid dimensions [x, y, z]
    /// * `origin` - Grid origin in world coordinates
    /// * `voxel_size` - Size of each voxel
    /// * `triangle_count` - Number of triangles in mesh
    ///
    /// # Returns
    /// Allocated GPU buffers for the grid.
    pub fn allocate(
        ctx: &GpuContext,
        dims: [usize; 3],
        origin: [f32; 3],
        voxel_size: f32,
        triangle_count: u32,
    ) -> GpuResult<Self> {
        let total_voxels = dims[0] * dims[1] * dims[2];

        // Check grid size limits
        let max_voxels = ctx.max_storage_buffer_size() as usize / std::mem::size_of::<f32>();
        if total_voxels > max_voxels {
            return Err(GpuError::GridTooLarge {
                dims,
                total: total_voxels,
                max: max_voxels,
            });
        }

        let grid_params = GpuGridParams {
            origin: [origin[0], origin[1], origin[2], 0.0],
            dims: [dims[0] as u32, dims[1] as u32, dims[2] as u32, 0],
            voxel_size,
            triangle_count,
            _padding: [0.0, 0.0],
        };

        // Create uniform buffer for grid parameters
        let params = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sdf_grid_params"),
                contents: bytemuck::bytes_of(&grid_params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        // Create storage buffer for SDF values
        let values_size = (total_voxels * std::mem::size_of::<f32>()) as u64;
        let values = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_values"),
            size: values_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create storage buffer for offset values
        let offsets = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_offsets"),
            size: values_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            params,
            values,
            offsets,
            dims,
            total_voxels,
        })
    }

    /// Download SDF values from GPU to CPU.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    ///
    /// # Returns
    /// Vector of SDF values.
    pub fn download_values(&self, ctx: &GpuContext) -> GpuResult<Vec<f32>> {
        self.download_buffer(ctx, &self.values)
    }

    /// Download offset values from GPU to CPU.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    ///
    /// # Returns
    /// Vector of offset values.
    pub fn download_offsets(&self, ctx: &GpuContext) -> GpuResult<Vec<f32>> {
        self.download_buffer(ctx, &self.offsets)
    }

    /// Download a buffer's contents to CPU memory.
    fn download_buffer(&self, ctx: &GpuContext, buffer: &Buffer) -> GpuResult<Vec<f32>> {
        let buffer_size = buffer.size();

        // Create staging buffer for readback
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_staging"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy GPU buffer to staging buffer
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sdf_download"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer_size);
        ctx.queue.submit([encoder.finish()]);

        // Map staging buffer and read data
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // Wait for GPU to finish
        ctx.device.poll(wgpu::Maintain::Wait);

        // Check mapping result
        rx.recv()
            .map_err(|_| GpuError::BufferMapping("channel closed".into()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        // Read data from mapped buffer
        let data = slice.get_mapped_range();
        let values: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        // Unmap buffer (drop guard first)
        drop(data);
        staging.unmap();

        Ok(values)
    }

    /// Upload SDF values from CPU to GPU.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `values` - SDF values to upload
    pub fn upload_values(&self, ctx: &GpuContext, values: &[f32]) -> GpuResult<()> {
        if values.len() != self.total_voxels {
            return Err(GpuError::Execution(format!(
                "value count mismatch: expected {}, got {}",
                self.total_voxels,
                values.len()
            )));
        }
        ctx.queue
            .write_buffer(&self.values, 0, bytemuck::cast_slice(values));
        Ok(())
    }

    /// Upload offset values from CPU to GPU.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `offsets` - Offset values to upload
    pub fn upload_offsets(&self, ctx: &GpuContext, offsets: &[f32]) -> GpuResult<()> {
        if offsets.len() != self.total_voxels {
            return Err(GpuError::Execution(format!(
                "offset count mismatch: expected {}, got {}",
                self.total_voxels,
                offsets.len()
            )));
        }
        ctx.queue
            .write_buffer(&self.offsets, 0, bytemuck::cast_slice(offsets));
        Ok(())
    }
}

/// Configuration for tiled processing of large grids.
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Size of each tile in voxels [x, y, z].
    pub tile_size: [usize; 3],
    /// Overlap between tiles (for algorithms that need neighbor data).
    pub overlap: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_size: [128, 128, 128],
            overlap: 2,
        }
    }
}

impl TileConfig {
    /// Create a tile configuration optimized for available GPU memory.
    pub fn for_memory_budget(budget_bytes: u64) -> Self {
        // Each voxel needs: f32 for SDF + f32 for offset = 8 bytes
        let bytes_per_voxel = 8u64;
        let max_voxels = budget_bytes / bytes_per_voxel;

        // Find cube root to get equal-sided tiles
        let side = (max_voxels as f64).cbrt() as usize;
        let side = side.clamp(32, 256); // Clamp to reasonable range

        Self {
            tile_size: [side, side, side],
            overlap: 2,
        }
    }

    /// Calculate the number of tiles needed for given grid dimensions.
    pub fn tile_count(&self, grid_dims: [usize; 3]) -> [usize; 3] {
        [
            grid_dims[0].div_ceil(self.tile_size[0]),
            grid_dims[1].div_ceil(self.tile_size[1]),
            grid_dims[2].div_ceil(self.tile_size[2]),
        ]
    }

    /// Calculate the total number of tiles.
    pub fn total_tiles(&self, grid_dims: [usize; 3]) -> usize {
        let count = self.tile_count(grid_dims);
        count[0] * count[1] * count[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_triangle_alignment() {
        // GpuTriangle should be 48 bytes (3 x vec4)
        assert_eq!(std::mem::size_of::<GpuTriangle>(), 48);
    }

    #[test]
    fn test_gpu_vertex_alignment() {
        // GpuVertex should be 32 bytes (vec4 + f32 + u32 + 2xf32)
        assert_eq!(std::mem::size_of::<GpuVertex>(), 32);
    }

    #[test]
    fn test_gpu_grid_params_alignment() {
        // GpuGridParams should be 48 bytes
        assert_eq!(std::mem::size_of::<GpuGridParams>(), 48);
    }

    #[test]
    fn test_tile_config_default() {
        let config = TileConfig::default();
        assert_eq!(config.tile_size, [128, 128, 128]);
        assert_eq!(config.overlap, 2);
    }

    #[test]
    fn test_tile_count_calculation() {
        let config = TileConfig {
            tile_size: [100, 100, 100],
            overlap: 0,
        };

        // 250x250x250 grid should need 3x3x3 = 27 tiles
        let count = config.tile_count([250, 250, 250]);
        assert_eq!(count, [3, 3, 3]);
        assert_eq!(config.total_tiles([250, 250, 250]), 27);
    }
}
